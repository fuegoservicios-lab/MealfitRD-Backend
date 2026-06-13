"""[P0.2] Tests para la columna `source` en user_inventory.

Cubre el "MERGE inteligente" del flujo de restock que distingue items que el
usuario añadió a mano (`source='manual'`) de los que vinieron de una lista de
compras (`source='shopping_list'`):

  1. add_or_update_inventory_item INSERT respeta el parámetro `source`.
  2. add_or_update_inventory_item sobre fila existente NO modifica `source` —
     first-writer-wins (un restock que sume sobre un item manual lo deja como
     manual). Post P0-4 el path de suma delega a la RPC `apply_inventory_delta`,
     que no toca `source`; el INSERT `ON CONFLICT DO UPDATE` (P3-PROD-AUDIT-2)
     tampoco lo incluye en su SET.
  3. restock_inventory pasa source='shopping_list' a las nuevas filas.
  4. replace_shopping_list_only_items borra sólo source='shopping_list' y
     preserva el resto.

[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado al transporte SQL directo:
los mocks simulan `execute_sql_query`/`execute_sql_write` del módulo
db_inventory en lugar del builder PostgREST legacy. Las propiedades
verificadas (source first-writer-wins, DELETE filtrado por user_id+source,
snapshot/rollback P3-D) son LAS MISMAS.

Ejecutar:
    cd backend && python -m pytest tests/test_inventory_source_column.py -v
"""
import re
import sys
import os
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)

import db_inventory  # noqa: E402


# ---------------------------------------------------------------------------
# Mock del transporte SQL directo (execute_sql_query / execute_sql_write).
# Captura cada statement para que los tests aserten lo enviado.
# ---------------------------------------------------------------------------
class _Captured:
    """Registra cada SELECT/INSERT/DELETE/RPC con su (sql, params) final."""
    def __init__(self):
        self.inserts = []    # list[(sql, params)] — INSERT INTO user_inventory
        self.deletes = []    # list[(sql, params)] — DELETE FROM user_inventory
        self.rpc_calls = []  # list[(sql, params)] — SELECT public.apply_inventory_delta(...)
        self.selects = []    # list[(sql, params)]


def _make_fake_sql(captured: _Captured, existing_rows_by_name=None,
                   snapshot_rows=None, snapshot_raises=False):
    """Construye el par (fake_query, fake_write). `existing_rows_by_name`
    simula filas en DB para el SELECT de add_or_update_inventory_item;
    `snapshot_rows` alimenta el snapshot pre-DELETE de
    replace_shopping_list_only_items (rollback P3-D).
    """
    existing_rows_by_name = existing_rows_by_name or {}

    def fake_query(sql, params=None, fetch_one=False, fetch_all=False):
        if "apply_inventory_delta" in sql:
            captured.rpc_calls.append((sql, params))
            return {"result": {"status": "ok"}}
        captured.selects.append((sql, params))
        if "count(*)" in sql:
            return {"count": 0}
        if "source = 'shopping_list'" in sql:
            # Snapshot pre-DELETE del rollback P3-D.
            if snapshot_raises:
                raise RuntimeError("simulated snapshot fetch error")
            return list(snapshot_rows or [])
        if "FROM user_inventory" in sql and params and len(params) >= 2:
            # SELECT existing de add_or_update_inventory_item: (user_id, name).
            return [dict(r) for r in existing_rows_by_name.get(params[1], [])]
        return []

    def fake_write(sql, params=None, returning=False):
        stripped = sql.strip()
        if stripped.upper().startswith("INSERT INTO USER_INVENTORY"):
            captured.inserts.append((sql, params))
            return [{"id": 1}] if returning else None
        if stripped.upper().startswith("DELETE FROM USER_INVENTORY"):
            captured.deletes.append((sql, params))
            # RETURNING id → lista de ids borrados (simula 1 fila).
            return [{"id": 1}] if returning else None
        # Otros writes (system_alerts, UPDATE legacy) — no relevantes acá.
        return [] if returning else None

    return fake_query, fake_write


def _patch_sql(captured: _Captured, **kwargs):
    fake_query, fake_write = _make_fake_sql(captured, **kwargs)
    return (
        patch.object(db_inventory, "execute_sql_query", side_effect=fake_query),
        patch.object(db_inventory, "execute_sql_write", side_effect=fake_write),
        patch("db_core.connection_pool", MagicMock()),  # _db_available() → True
    )


def _insert_row_dict(sql: str, params: tuple) -> dict:
    """Reconstruye {col: val} de un INSERT dinámico (rollback P3-D) parseando
    la lista de columnas del propio SQL."""
    m = re.search(r"INSERT INTO user_inventory\s*\((.*?)\)\s*VALUES", sql, re.DOTALL)
    assert m, f"INSERT sin lista de columnas parseable: {sql!r}"
    cols = [c.strip() for c in m.group(1).split(",")]
    return dict(zip(cols, params))


# Orden canónico de params del INSERT estático de add_or_update_inventory_item:
# (user_id, ingredient_name, quantity, unit, master_ingredient_id,
#  last_mutation_type, source) — source en índice 6.
_INSERT_SOURCE_IDX = 6


# ---------------------------------------------------------------------------
# 1. INSERT respeta el parámetro `source`
# ---------------------------------------------------------------------------
def test_insert_with_source_manual():
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        ok = db_inventory.add_or_update_inventory_item(
            "u1", "Manzana", 3.0, "ud", source="manual",
        )

    assert ok is True
    assert len(captured.inserts) == 1
    sql, params = captured.inserts[0]
    assert params[_INSERT_SOURCE_IDX] == "manual"
    # [P3-PROD-AUDIT-2] El INSERT cierra la race del doble-INSERT con upsert
    # increment; el DO UPDATE NO debe tocar `source` (first-writer-wins P0.2).
    assert "ON CONFLICT (user_id, ingredient_name, unit)" in sql
    do_update_clause = sql.split("DO UPDATE", 1)[1]
    assert "source" not in do_update_clause, (
        "El DO UPDATE del upsert NO debe sobrescribir `source` — "
        "first-writer-wins (P0.2)."
    )


def test_insert_with_source_shopping_list():
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        ok = db_inventory.add_or_update_inventory_item(
            "u1", "Pollo", 500.0, "g", source="shopping_list",
        )

    assert ok is True
    assert captured.inserts[0][1][_INSERT_SOURCE_IDX] == "shopping_list"


def test_insert_default_source_is_manual():
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        db_inventory.add_or_update_inventory_item("u1", "Arroz", 100.0, "g")

    assert captured.inserts[0][1][_INSERT_SOURCE_IDX] == "manual"


# ---------------------------------------------------------------------------
# 2. Suma sobre fila existente preserva `source` (first-writer-wins)
# ---------------------------------------------------------------------------
def test_update_does_not_overwrite_source():
    """Si una fila existe como manual y un restock suma sobre ella, source
    debe permanecer 'manual'. Post P0-4 la suma viaja por la RPC
    `apply_inventory_delta` (delta atómico FOR UPDATE) — la llamada NO debe
    referenciar `source` y NO debe emitirse INSERT alguno.
    """
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={
        "Manzana": [
            {"id": 42, "quantity": 3.0, "unit": "ud"},
        ],
    })

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=2.0):
        # Restock que pretende sumar 2 manzanas más sobre la fila manual.
        ok = db_inventory.add_or_update_inventory_item(
            "u1", "Manzana", 2.0, "ud", source="shopping_list",
        )

    assert ok is True
    # No debe haber INSERT (la fila ya existía y se sumó vía RPC).
    assert captured.inserts == []
    # Debe haber exactamente una llamada a la RPC atómica...
    assert len(captured.rpc_calls) == 1
    rpc_sql, rpc_params = captured.rpc_calls[0]
    assert "public.apply_inventory_delta" in rpc_sql
    # ...que NO toca `source` (first-writer-wins: la fila conserva 'manual').
    assert "source" not in rpc_sql
    # Params canónicos: (user_id, row_id, delta, mutation_type, master_id).
    assert rpc_params[0] == "u1"
    assert rpc_params[1] == 42
    assert rpc_params[2] == 2.0  # delta convertido — la RPC suma 3 + 2 en DB
    assert rpc_params[3] == "manual"  # mutation_type default
    assert rpc_params[4] is None


# ---------------------------------------------------------------------------
# 3. restock_inventory tagea source='shopping_list'
# ---------------------------------------------------------------------------
def test_restock_inventory_tags_new_rows_as_shopping_list():
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] restock_inventory retorna
        # tupla (success, persisted_names). Validamos ambos: la flag de éxito
        # + la lista de names que efectivamente se persistieron (sin
        # normalize_name aplicado en la ruta estructurada).
        ok, persisted = db_inventory.restock_inventory("u1", [
            {"name": "Tomate", "quantity": 4, "unit": "ud"},
            {"name": "Cebolla", "quantity": 1, "unit": "kg"},
        ])

    assert ok is True
    assert len(captured.inserts) == 2
    sources = [params[_INSERT_SOURCE_IDX] for _sql, params in captured.inserts]
    assert sources == ["shopping_list", "shopping_list"]
    # [P0-RESTOCK-DEDUP-NAME] persisted_names contiene los names en orden.
    assert persisted == ["Tomate", "Cebolla"]


# ---------------------------------------------------------------------------
# 4. replace_shopping_list_only_items preserva manual
# ---------------------------------------------------------------------------
def test_replace_shopping_list_only_deletes_only_shopping_rows():
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", return_value=(True, ["Tomate"])) as rmock:
        # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] Mock retorna tupla matching
        # la signature nueva.
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Debe haber emitido un DELETE filtrado por user_id Y source='shopping_list'.
    assert len(captured.deletes) == 1
    delete_sql, delete_params = captured.deletes[0]
    assert "user_id = %s" in delete_sql
    assert "source = 'shopping_list'" in delete_sql
    assert delete_params == ("u1",)

    # Y debe haber delegado el INSERT al restock_inventory existente
    # (que ya tagea las filas nuevas como source='shopping_list').
    rmock.assert_called_once()
    called_args = rmock.call_args[0]
    assert called_args[0] == "u1"
    assert called_args[1] == [{"name": "Tomate", "quantity": 2, "unit": "ud"}]

    assert "deleted_shopping_rows" in stats
    assert "preserved_manual_rows" in stats


def test_replace_shopping_list_only_handles_empty_list_gracefully():
    """Lista vacía: borra los shopping_list previos pero no llama a restock."""
    captured = _Captured()
    p_query, p_write, p_pool = _patch_sql(captured, existing_rows_by_name={})

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory") as rmock:
        stats = db_inventory.replace_shopping_list_only_items("u1", [])

    rmock.assert_not_called()
    # El DELETE igualmente se ejecutó (limpió shopping_list previos).
    assert len(captured.deletes) == 1


def test_replace_shopping_list_only_noop_on_empty_user():
    stats = db_inventory.replace_shopping_list_only_items("", [{"name": "x"}])
    # [P3-D · 2026-05-07] Stats shape ahora incluye `rolled_back` (default False).
    # [P3-1 · 2026-05-08] Añadidos `rolled_back_count`, `rolled_back_total`,
    # `rolled_back_partial` para distinguir rollback completo vs parcial.
    assert stats == {
        "deleted_shopping_rows": 0,
        "inserted_rows": 0,
        "preserved_manual_rows": 0,
        "rolled_back": False,
        "rolled_back_count": 0,
        "rolled_back_total": 0,
        "rolled_back_partial": False,
    }


# ---------------------------------------------------------------------------
# 5. [P3-D · 2026-05-07] Rollback de seguridad cuando restock_inventory falla.
# ---------------------------------------------------------------------------
# Antes del fix: si DELETE tenía éxito pero `restock_inventory` retornaba False
# o lanzaba excepción, el usuario quedaba sin lista de compras sin posibilidad
# de recovery. Después: snapshot pre-DELETE → restore-on-failure.
#
# Estos tests verifican el comportamiento bajo los modos de fallo que el fix
# pretende cubrir y el knob de kill-switch operacional.
# ---------------------------------------------------------------------------
def test_rollback_restores_snapshot_when_restock_returns_false():
    """`restock_inventory` retorna False → rollback restaura las filas
    snapshotted vía INSERT."""
    captured = _Captured()
    snapshot = [
        {"id": 1, "user_id": "u1", "ingredient_name": "Pollo", "quantity": 500.0,
         "unit": "g", "source": "shopping_list", "created_at": "2026-05-01T00:00:00Z"},
        {"id": 2, "user_id": "u1", "ingredient_name": "Arroz", "quantity": 1000.0,
         "unit": "g", "source": "shopping_list", "created_at": "2026-05-01T00:00:00Z"},
    ]
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_rows=snapshot)

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", return_value=(False, [])):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Rollback se activó
    assert stats["rolled_back"] is True
    # Se intentó restaurar las 2 filas snapshotted (vía INSERT)
    assert len(captured.inserts) == 2
    # Y los inserts NO contienen las columnas managed-by-DB
    for sql, params in captured.inserts:
        ins = _insert_row_dict(sql, params)
        assert "id" not in ins
        assert "created_at" not in ins
        assert ins["source"] == "shopping_list"
        assert ins["user_id"] == "u1"


def test_rollback_restores_snapshot_when_restock_raises():
    """`restock_inventory` lanza excepción → rollback igual que cuando retorna False.
    El error es capturado, no se propaga al caller."""
    captured = _Captured()
    snapshot = [
        {"id": 99, "user_id": "u1", "ingredient_name": "Sal", "quantity": 1.0,
         "unit": "kg", "source": "shopping_list"},
    ]
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_rows=snapshot)

    def _raise(*_a, **_kw):
        raise RuntimeError("simulated DB blip")

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", side_effect=_raise):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    assert stats["rolled_back"] is True
    assert len(captured.inserts) == 1


def test_no_rollback_when_restock_succeeds():
    """`restock_inventory` exitoso → NO se intenta restore (filas snapshot no
    se re-insertan, sólo el restock_inventory mockeado)."""
    captured = _Captured()
    snapshot = [
        {"id": 1, "user_id": "u1", "ingredient_name": "Pollo", "quantity": 500.0,
         "unit": "g", "source": "shopping_list"},
    ]
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_rows=snapshot)

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", return_value=(True, ["Tomate"])):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    assert stats["rolled_back"] is False
    # Las únicas inserciones serían las del restock mockeado (que no usa el
    # transporte fake, devolvió la tupla directamente). Por lo tanto
    # captured.inserts debe estar vacío.
    assert len(captured.inserts) == 0


def test_knob_off_disables_snapshot_and_rollback(monkeypatch):
    """`MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK=off` → no snapshot, no rollback
    incluso si restock falla. Modo legacy preservado para kill switch operacional."""
    monkeypatch.setenv("MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK", "off")
    captured = _Captured()
    snapshot = [
        {"id": 1, "user_id": "u1", "ingredient_name": "Pollo", "quantity": 500.0,
         "unit": "g", "source": "shopping_list"},
    ]
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_rows=snapshot)

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", return_value=(False, [])):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Sin rollback (knob off)
    assert stats["rolled_back"] is False
    # Sin re-inserción de snapshot
    assert len(captured.inserts) == 0


def test_knob_default_is_on():
    """Sin env var seteada → comportamiento por default es rollback ON.
    Garantiza que un deploy fresco hereda la safety automáticamente."""
    captured = _Captured()
    snapshot = [
        {"id": 1, "user_id": "u1", "ingredient_name": "Pollo", "quantity": 500.0,
         "unit": "g", "source": "shopping_list"},
    ]
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_rows=snapshot)

    # Asegurar que la env var NO está seteada (defensive — no monkeypatch.setenv)
    if "MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK" in os.environ:
        old = os.environ.pop("MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK")
    else:
        old = None
    try:
        with p_query, p_write, p_pool, \
             patch.object(db_inventory, "restock_inventory", return_value=(False, [])):
            stats = db_inventory.replace_shopping_list_only_items(
                "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
            )
        assert stats["rolled_back"] is True
    finally:
        if old is not None:
            os.environ["MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK"] = old


def test_snapshot_select_failure_degrades_to_legacy():
    """Si el SELECT del snapshot falla (DB blip durante la lectura previa al
    DELETE), el código degrada a modo legacy (sin rollback) en lugar de abortar.
    Garantiza que un blip transient no bloquea al usuario."""
    captured = _Captured()
    # snapshot_raises=True → el SELECT del snapshot (source='shopping_list')
    # lanza; el count(*) y el DELETE siguen funcionando.
    p_query, p_write, p_pool = _patch_sql(captured, snapshot_raises=True)

    with p_query, p_write, p_pool, \
         patch.object(db_inventory, "restock_inventory", return_value=(False, [])):
        # No debe levantar — snapshot falla, degrada legacy, restock falla
        # también pero sin snapshot rollback.
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Sin rollback (snapshot fue vacío por el fallo)
    assert stats["rolled_back"] is False
    # El DELETE sí corrió (modo legacy preservado).
    assert len(captured.deletes) == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
