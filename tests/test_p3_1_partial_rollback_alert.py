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
  - Si execute_sql_write falla, el flow continúa (no excepción).
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------------

class _FakeSupabase:
    """Mock minimal de supabase chainable. Configurable por escenario."""

    def __init__(self, *, snapshot_rows: list, restore_failures: int = 0,
                 restock_returns_falsy: bool = True):
        self._snapshot_rows = snapshot_rows
        self._restore_failures = restore_failures
        self._restore_call_count = 0
        self._mode = None

    def table(self, name):
        return self

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def neq(self, *args, **kwargs):
        return self

    def execute(self):
        if self._mode == "snapshot":
            res = MagicMock()
            res.data = self._snapshot_rows
            return res
        if self._mode == "preserved":
            res = MagicMock()
            res.count = 0
            res.data = []
            return res
        if self._mode == "delete":
            res = MagicMock()
            res.count = len(self._snapshot_rows)
            return res
        # default
        res = MagicMock()
        res.data = []
        res.count = 0
        return res

    def delete(self, count=None):
        self._mode = "delete"
        return self

    def insert(self, row):
        # Simula failures intercalados.
        self._restore_call_count += 1
        if self._restore_call_count <= self._restore_failures:
            raise RuntimeError(f"simulated insert failure #{self._restore_call_count}")
        res = MagicMock()
        res.data = [row]
        return self.__class__._WrappedExecute(res)

    class _WrappedExecute:
        def __init__(self, res):
            self._res = res

        def execute(self):
            return self._res


def _setup_fake_supabase(snapshot_rows, restore_failures=0):
    """Configura módulo db_inventory con un supabase fake."""
    import db_inventory
    fake = _FakeSupabase(
        snapshot_rows=snapshot_rows,
        restore_failures=restore_failures,
    )

    # Hack para hacer que el .select(...).eq(...).execute() del snapshot
    # devuelva el snapshot data y luego el delete devuelva count.
    original_select = fake.select

    def select_with_mode(*args, **kwargs):
        # Si el caller pidió "*", es snapshot. Si pidió "id", es preserved/count.
        if args and args[0] == "*":
            fake._mode = "snapshot"
        else:
            fake._mode = "preserved"
        return fake

    fake.select = select_with_mode
    return fake


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
    snapshot = [
        {"id": "r1", "user_id": "u1", "source": "shopping_list", "name": "tomate"},
        {"id": "r2", "user_id": "u1", "source": "shopping_list", "name": "cebolla"},
    ]
    fake = _setup_fake_supabase(snapshot, restore_failures=0)
    with patch.object(db_inventory, "supabase", fake), \
         patch.object(db_inventory, "restock_inventory", return_value=False), \
         patch.object(db_inventory, "execute_sql_write") as mock_write:
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is True
    assert stats["rolled_back_count"] == 2
    assert stats["rolled_back_total"] == 2
    assert stats["rolled_back_partial"] is False
    # No debe haber INSERT a system_alerts cuando rollback es completo.
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    assert len(inserts) == 0


# ---------------------------------------------------------------------------
# 3. Rollback parcial → alert critical
# ---------------------------------------------------------------------------
def test_partial_rollback_emits_critical_alert():
    import db_inventory
    snapshot = [
        {"id": "r1", "user_id": "u1", "source": "shopping_list", "name": "tomate"},
        {"id": "r2", "user_id": "u1", "source": "shopping_list", "name": "cebolla"},
        {"id": "r3", "user_id": "u1", "source": "shopping_list", "name": "ajo"},
    ]
    fake = _setup_fake_supabase(snapshot, restore_failures=2)  # 2 fallan, 1 pasa
    with patch.object(db_inventory, "supabase", fake), \
         patch.object(db_inventory, "restock_inventory", return_value=False), \
         patch.object(db_inventory, "execute_sql_write") as mock_write:
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is True
    assert stats["rolled_back_count"] == 1
    assert stats["rolled_back_total"] == 3
    assert stats["rolled_back_partial"] is True
    # Debe haber UN INSERT a system_alerts.
    inserts = [
        c for c in mock_write.call_args_list
        if "INSERT INTO system_alerts" in str(c.args[0])
    ]
    assert len(inserts) == 1, f"Esperado 1 alert, got {len(inserts)}"
    # Validar shape del INSERT.
    insert_args = inserts[0].args[1]
    assert insert_args[0] == "shopping_list_replace_partial_rollback:u1"
    # Title (índice 1 en el INSERT post alert_key).
    assert "u1" in insert_args[1]
    # Message tiene SOP.
    message = insert_args[2]
    assert "SOP recovery manual" in message
    assert "1." in message and "2." in message  # pasos numerados
    # Metadata.
    metadata = json.loads(insert_args[3])
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
    snapshot = [
        {"id": "r1", "user_id": "u1", "source": "shopping_list", "name": "tomate"},
        {"id": "r2", "user_id": "u1", "source": "shopping_list", "name": "cebolla"},
    ]
    # Todas las restore fallan.
    fake = _setup_fake_supabase(snapshot, restore_failures=999)
    with patch.object(db_inventory, "supabase", fake), \
         patch.object(db_inventory, "restock_inventory", return_value=False), \
         patch.object(db_inventory, "execute_sql_write") as mock_write:
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back"] is False
    assert stats["rolled_back_count"] == 0
    assert stats["rolled_back_total"] == 2
    # `partial` solo cuando 0 < count < total. count=0 NO es parcial (es total-fail).
    assert stats["rolled_back_partial"] is False
    # No alert porque la condición es `if is_partial:`.
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    assert len(inserts) == 0


# ---------------------------------------------------------------------------
# 5. failed_row_ids capped a 50
# ---------------------------------------------------------------------------
def test_failed_row_ids_capped_at_50():
    import db_inventory
    # 60 filas, todas fallan menos la última.
    snapshot = [
        {"id": f"r{i}", "user_id": "u1", "source": "shopping_list", "name": f"item-{i}"}
        for i in range(60)
    ]
    fake = _setup_fake_supabase(snapshot, restore_failures=59)
    with patch.object(db_inventory, "supabase", fake), \
         patch.object(db_inventory, "restock_inventory", return_value=False), \
         patch.object(db_inventory, "execute_sql_write") as mock_write:
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "x"}])
    assert stats["rolled_back_partial"] is True
    inserts = [
        c for c in mock_write.call_args_list
        if "INSERT INTO system_alerts" in str(c.args[0])
    ]
    assert len(inserts) == 1
    metadata = json.loads(inserts[0].args[1][3])
    assert len(metadata["failed_row_ids"]) <= 50, (
        f"failed_row_ids debe estar capped a 50; got {len(metadata['failed_row_ids'])}"
    )


# ---------------------------------------------------------------------------
# 6. Alert raise → flujo continúa (best-effort)
# ---------------------------------------------------------------------------
def test_alert_persistence_failure_does_not_abort_flow():
    import db_inventory
    snapshot = [
        {"id": "r1", "user_id": "u1", "source": "shopping_list", "name": "tomate"},
        {"id": "r2", "user_id": "u1", "source": "shopping_list", "name": "cebolla"},
    ]
    fake = _setup_fake_supabase(snapshot, restore_failures=1)
    with patch.object(db_inventory, "supabase", fake), \
         patch.object(db_inventory, "restock_inventory", return_value=False), \
         patch.object(db_inventory, "execute_sql_write",
                      side_effect=RuntimeError("DB blip")):
        # NO debe levantar — el alert es best-effort.
        stats = db_inventory.replace_shopping_list_only_items("u1", [{"name": "lechuga"}])
    assert stats["rolled_back_partial"] is True
    assert stats["rolled_back_count"] == 1
    assert stats["rolled_back_total"] == 2
