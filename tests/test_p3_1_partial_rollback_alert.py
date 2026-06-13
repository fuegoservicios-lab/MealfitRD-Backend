"""[P3-1 · 2026-05-08] Tests del rollback parcial + alerta SOP en
`replace_shopping_list_only_items`.

Bug original (audit 2026-05-07):
  En `db_inventory.py:1410`, P3-D introdujo `stats["rolled_back"] = restored > 0`
  como bool único. Si la restauración fallaba en mitad (ej. 5 de 10 filas
  restauradas), `rolled_back=True` no distinguía:
    - Rollback completo (todas restauradas) → estado consistente.
    - Rollback parcial (algunas filas perdidas) → inventario inconsistente.
  El SRE no podía detectar el segundo caso sin grep manual al log
  `[P3-D/ROLLBACK] Error restaurando row`.

Fix:
  1. Stats expandidos:
     - `rolled_back_count` (int): filas efectivamente restauradas.
     - `rolled_back_total` (int): filas en snapshot pre-DELETE.
     - `rolled_back_partial` (bool): True si 0 < count < total.
     - `rolled_back` (bool, preserva backwards-compat): count > 0.
  2. Cuando `is_partial=True`, INSERT a `system_alerts` con
     `severity='critical'`, `alert_type='shopping_list_partial_rollback'`,
     `alert_key='shopping_list_replace_partial_rollback:{user_id}'` y
     metadata con `failed_row_ids` (cap 50) + SOP de recovery manual en el
     `message` (paso a paso para el SRE).
  3. Best-effort: si la persistencia del alert falla, el flujo del usuario
     continúa (no abortar por escalación a SRE).

Cobertura:
  - Stats nuevos campos presentes con defaults correctos.
  - Backwards-compat: `rolled_back` sigue siendo bool con misma semántica.
  - Rollback completo (todas restauradas) → no alert, partial=False.
  - Rollback parcial (algunas restauradas) → alert critical + partial=True.
  - Rollback nada (cero restauradas) → no alert, count=0, partial=False.
  - Alert metadata incluye failed_row_ids capped a 50.
  - Alert metadata incluye SOP en message con pasos numerados.
  - Si el INSERT del alert falla, el flow continúa (no excepción).

[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: `replace_shopping_list_only_items`
migró de PostgREST (`supabase.table(...).select/delete/insert`) a SQL
directo via `execute_sql_query`/`execute_sql_write` (db_core). El fake
simula el transporte SQL dispatcheando por el literal SQL recibido:
  - SELECT count(*) ... source <> 'shopping_list'  → preserved count.
  - SELECT ... source = 'shopping_list' (fetch_all) → snapshot pre-DELETE.
  - DELETE ... RETURNING id                         → rows borrados.
  - INSERT INTO user_inventory                      → restore best-effort
                                                      (failures intercalados).
  - INSERT INTO system_alerts                       → alerta P3-1 (capturada).
El guard de disponibilidad es `db_core.connection_pool` (lazy via
`_db_available()`), no `db_inventory.supabase` (símbolo removido).
Tipos de los rows mockeados reflejan el SELECT nuevo: `user_id::text` →
str, `quantity/reserved_quantity::float8` → float; `id` se retorna SIN
cast (uuid.UUID en prod — el código solo lo usa via str() para el log).
"""
import json
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------------

def _snapshot_row(i: int, name: str) -> dict:
    """Row con el shape/tipos del SELECT snapshot post-Neon (set de columnas
    explícito; ver db_inventory.replace_shopping_list_only_items)."""
    return {
        "id": uuid.uuid4(),  # sin ::text en el SELECT — uuid.UUID en prod
        "user_id": "u1",
        "ingredient_name": name,
        "quantity": 1.0,
        "unit": "unidad",
        "master_ingredient_id": None,
        "reserved_quantity": 0.0,
        "reservation_details": None,
        "last_mutation_type": None,
        "source": "shopping_list",
        "category": None,
    }


class _FakeSqlBackend:
    """Fake del transporte SQL de db_core, configurable por escenario.

    Dispatchea por el literal SQL recibido (los literales son los del
    source productivo — si cambian, estos tests fallan loud y guían el
    re-anclaje).
    """

    def __init__(self, *, snapshot_rows: list, restore_failures: int = 0,
                 alert_insert_raises: bool = False):
        self._snapshot_rows = snapshot_rows
        self._restore_failures = restore_failures
        self._restore_call_count = 0
        self._alert_insert_raises = alert_insert_raises
        self.write_calls: list = []  # (sql, params) de cada execute_sql_write

    # --- execute_sql_query ---------------------------------------------
    def query(self, sql, params=None, fetch_one=False, fetch_all=False):
        if "count(*)" in sql and "<> 'shopping_list'" in sql:
            return {"count": 0}  # preserved manual rows
        if "count(*)" in sql:
            return {"count": 0}  # inserted_rows post-restock (no se alcanza aquí)
        if fetch_all and "source = 'shopping_list'" in sql:
            return list(self._snapshot_rows)  # snapshot pre-DELETE
        return [] if fetch_all else None

    # --- execute_sql_write ---------------------------------------------
    def write(self, sql, params=None, returning=False, **kwargs):
        self.write_calls.append((sql, params))
        if "DELETE FROM user_inventory" in sql:
            return [{"id": r["id"]} for r in self._snapshot_rows]
        if "INSERT INTO user_inventory" in sql:
            # Simula failures intercalados en el restore best-effort.
            self._restore_call_count += 1
            if self._restore_call_count <= self._restore_failures:
                raise RuntimeError(
                    f"simulated insert failure #{self._restore_call_count}"
                )
            return None
        if "INSERT INTO system_alerts" in sql:
            if self._alert_insert_raises:
                raise RuntimeError("DB blip")
            return None
        return None

    # --- helpers de aserción --------------------------------------------
    def alert_inserts(self):
        return [
            (sql, params) for sql, params in self.write_calls
            if "INSERT INTO system_alerts" in sql
        ]


def _patch_sql_backend(backend: "_FakeSqlBackend"):
    """Context managers para instalar el fake en db_inventory + db_core.

    `_db_available()` lee `db_core.connection_pool` lazy — lo forzamos a
    truthy para no depender de DB real del entorno.
    """
    import db_core
    import db_inventory
    return (
        patch.object(db_core, "connection_pool", object()),
        patch.object(db_inventory, "execute_sql_query", backend.query),
        patch.object(db_inventory, "execute_sql_write", backend.write),
    )


# ---------------------------------------------------------------------------
# 1. Stats shape
# ---------------------------------------------------------------------------
def test_stats_shape_includes_new_fields_on_noop():
    import db_inventory
    stats = db_inventory.replace_shopping_list_only_items("", [{"name": "x"}])
    expected_keys = {
        "deleted_shopping_rows", "inserted_rows", "preserved_manual_rows",
        "rolled_back", "rolled_back_count", "rolled_back_total",
        "rolled_back_partial",
    }
    assert set(stats.keys()) == expected_keys, (
        f"Stats keys mismatch. Got {set(stats.keys())}, expected {expected_keys}."
    )
    assert stats["rolled_back"] is False
    assert stats["rolled_back_count"] == 0
    assert stats["rolled_back_total"] == 0
    assert stats["rolled_back_partial"] is False


# ---------------------------------------------------------------------------
# 2. Rollback completo (todas restauradas) → no alert
# ---------------------------------------------------------------------------
def test_full_rollback_no_alert():
    import db_inventory
    snapshot = [_snapshot_row(1, "tomate"), _snapshot_row(2, "cebolla")]
    backend = _FakeSqlBackend(snapshot_rows=snapshot, restore_failures=0)
    pool_patch, query_patch, write_patch = _patch_sql_backend(backend)
    with pool_patch, query_patch, write_patch, \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is True
    assert stats["rolled_back_count"] == 2
    assert stats["rolled_back_total"] == 2
    assert stats["rolled_back_partial"] is False
    # No debe haber INSERT a system_alerts cuando rollback es completo.
    assert len(backend.alert_inserts()) == 0


# ---------------------------------------------------------------------------
# 3. Rollback parcial → alert critical
# ---------------------------------------------------------------------------
def test_partial_rollback_emits_critical_alert():
    import db_inventory
    snapshot = [
        _snapshot_row(1, "tomate"),
        _snapshot_row(2, "cebolla"),
        _snapshot_row(3, "ajo"),
    ]
    backend = _FakeSqlBackend(snapshot_rows=snapshot, restore_failures=2)  # 2 fallan, 1 pasa
    pool_patch, query_patch, write_patch = _patch_sql_backend(backend)
    with pool_patch, query_patch, write_patch, \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is True
    assert stats["rolled_back_count"] == 1
    assert stats["rolled_back_total"] == 3
    assert stats["rolled_back_partial"] is True
    # Debe haber UN INSERT a system_alerts.
    inserts = backend.alert_inserts()
    assert len(inserts) == 1, f"Esperado 1 alert, got {len(inserts)}"
    # Validar shape del INSERT (params: alert_key, title, message, metadata, affected).
    _sql, insert_params = inserts[0]
    assert insert_params[0] == "shopping_list_replace_partial_rollback:u1"
    # Title (índice 1 en el INSERT post alert_key).
    assert "u1" in insert_params[1]
    # Message tiene SOP.
    message = insert_params[2]
    assert "SOP recovery manual" in message
    assert "1." in message and "2." in message  # pasos numerados
    # Metadata.
    metadata = json.loads(insert_params[3])
    assert metadata["user_id"] == "u1"
    assert metadata["rows_restored"] == 1
    assert metadata["rows_in_snapshot"] == 3
    assert metadata["rows_lost"] == 2
    assert isinstance(metadata["failed_row_ids"], list)
    assert len(metadata["failed_row_ids"]) == 2


# ---------------------------------------------------------------------------
# 4. Rollback nada (cero restauradas) → no alert (no hay limbo, NO partial)
# ---------------------------------------------------------------------------
def test_zero_rollback_no_alert_no_partial():
    import db_inventory
    snapshot = [_snapshot_row(1, "tomate"), _snapshot_row(2, "cebolla")]
    # Todas las restore fallan.
    backend = _FakeSqlBackend(snapshot_rows=snapshot, restore_failures=999)
    pool_patch, query_patch, write_patch = _patch_sql_backend(backend)
    with pool_patch, query_patch, write_patch, \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is False
    assert stats["rolled_back_count"] == 0
    assert stats["rolled_back_total"] == 2
    # `partial` solo cuando 0 < count < total. count=0 NO es parcial (es total-fail).
    assert stats["rolled_back_partial"] is False
    # No alert porque la condición es `if is_partial:`.
    assert len(backend.alert_inserts()) == 0


# ---------------------------------------------------------------------------
# 5. failed_row_ids capped a 50
# ---------------------------------------------------------------------------
def test_failed_row_ids_capped_at_50():
    import db_inventory
    # 60 filas, todas fallan menos la última.
    snapshot = [_snapshot_row(i, f"item-{i}") for i in range(60)]
    backend = _FakeSqlBackend(snapshot_rows=snapshot, restore_failures=59)
    pool_patch, query_patch, write_patch = _patch_sql_backend(backend)
    with pool_patch, query_patch, write_patch, \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "x"}])
    assert stats["rolled_back_partial"] is True
    inserts = backend.alert_inserts()
    assert len(inserts) == 1
    metadata = json.loads(inserts[0][1][3])
    assert len(metadata["failed_row_ids"]) <= 50, (
        f"failed_row_ids debe estar capped a 50; got {len(metadata['failed_row_ids'])}"
    )


# ---------------------------------------------------------------------------
# 6. Alert raise → flujo continúa (best-effort)
# ---------------------------------------------------------------------------
def test_alert_persistence_failure_does_not_abort_flow():
    import db_inventory
    snapshot = [_snapshot_row(1, "tomate"), _snapshot_row(2, "cebolla")]
    # [P1-NEON-DB-MIGRATION] execute_sql_write ahora también ejecuta el
    # DELETE y los restore-INSERTs — el fail simulado es SOLO el INSERT
    # del alert (misma propiedad que el test original: la persistencia
    # del alert es best-effort y NO aborta el flujo del usuario).
    backend = _FakeSqlBackend(
        snapshot_rows=snapshot, restore_failures=1, alert_insert_raises=True,
    )
    pool_patch, query_patch, write_patch = _patch_sql_backend(backend)
    with pool_patch, query_patch, write_patch, \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        # NO debe levantar — el alert es best-effort.
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back_partial"] is True
    assert stats["rolled_back_count"] == 1
    assert stats["rolled_back_total"] == 2
    # El INSERT del alert SÍ se intentó (y explotó) — fue swallowed.
    assert len(backend.alert_inserts()) == 1
