"""[P0.1] Tests para sync_inventory_after_chunk_completion.

Cubre el reconciliador que se ejecuta al cerrar cada chunk:

  1. La función deduce ingredientes de filas consumed_meals NO sincronizadas
     y las marca con inventory_synced_at = NOW().
  2. Filas ya sincronizadas (inventory_synced_at != NULL) se omiten — sin
     doble-deducción incluso si el chunk se reprocesa.
  3. Filas con ingredientes vacíos se marcan como sincronizadas igual (para
     no reintentarlas en cada cierre de chunk).
  4. Filas fuera de la ventana del chunk no se tocan.
  5. Caller args inválidos (user_id vacío, ventanas faltantes) → no-op seguro.
  6. log_consumed_meal con mark_inventory_synced=True persiste
     inventory_synced_at desde el INSERT (path del agente que ya deduce).

Ejecutar:
    cd backend && python -m pytest tests/test_pantry_auto_sync_between_chunks.py -v
"""
import sys
import os
import types
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos para importar db_inventory aislado del resto del backend.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)

import db_inventory  # noqa: E402
import db_facts  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeDB:
    """Simula execute_sql_query / execute_sql_write contra una lista de rows."""

    def __init__(self, rows):
        # Cada row es un dict con keys: id, ingredients, consumed_at, inventory_synced_at
        self.rows = list(rows)
        self.updates = []  # (id,) tuplas que recibieron UPDATE

    def query(self, query, params=None, fetch_one=False, fetch_all=False):
        # La función bajo test sólo emite el SELECT con filtros user_id +
        # ventana + inventory_synced_at IS NULL. Filtramos en memoria.
        if "FROM consumed_meals" not in query:
            return [] if fetch_all else None
        user_id, win_start, win_end = params
        out = []
        for r in self.rows:
            if r.get("user_id") != user_id:
                continue
            if r.get("inventory_synced_at") is not None:
                continue
            ca = r.get("consumed_at")
            if ca is None:
                continue
            if not (win_start <= ca < win_end):
                continue
            out.append({
                "id": r.get("id"),
                "ingredients": r.get("ingredients"),
                "consumed_at": ca,
            })
        return out if fetch_all else (out[0] if out else None)

    def write(self, query, params=None):
        if "UPDATE consumed_meals" in query and "inventory_synced_at" in query:
            (row_id,) = params
            self.updates.append(row_id)
            for r in self.rows:
                if r.get("id") == row_id:
                    r["inventory_synced_at"] = "2026-05-02T12:00:00+00:00"


# ---------------------------------------------------------------------------
# 1. Filas no sincronizadas se deducen y marcan
# ---------------------------------------------------------------------------
def test_reconciles_unsynced_rows_in_window():
    fake = _FakeDB([
        {
            "id": 1, "user_id": "u1",
            "ingredients": ["200g pollo", "100g arroz"],
            "consumed_at": "2026-05-01T10:00:00+00:00",
            "inventory_synced_at": None,
        },
        {
            "id": 2, "user_id": "u1",
            "ingredients": ["1 huevo"],
            "consumed_at": "2026-05-02T08:00:00+00:00",
            "inventory_synced_at": None,
        },
    ])
    deducted_calls = []

    def _fake_deduct(user_id, ingredients):
        deducted_calls.append((user_id, list(ingredients)))

    # connection_pool truthy → entra por la rama execute_sql_query.
    with patch("db_core.connection_pool", object()), \
         patch("db_core.execute_sql_query", side_effect=fake.query), \
         patch.object(db_inventory, "execute_sql_write", side_effect=fake.write), \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory", side_effect=_fake_deduct):
        stats = db_inventory.sync_inventory_after_chunk_completion(
            "u1",
            "2026-04-30T00:00:00+00:00",
            "2026-05-03T00:00:00+00:00",
        )

    assert stats["reconciled_count"] == 2
    assert stats["items_deducted"] == 3  # 2 + 1 ingredientes
    assert len(deducted_calls) == 2
    assert sorted(fake.updates) == [1, 2]


# ---------------------------------------------------------------------------
# 2. Filas ya sincronizadas se omiten (idempotencia)
# ---------------------------------------------------------------------------
def test_skips_already_synced_rows():
    fake = _FakeDB([
        {
            "id": 10, "user_id": "u1",
            "ingredients": ["200g pollo"],
            "consumed_at": "2026-05-01T10:00:00+00:00",
            "inventory_synced_at": "2026-05-01T10:00:01+00:00",  # ya sync
        },
    ])
    deducted_calls = []

    with patch("db_core.connection_pool", object()), \
         patch("db_core.execute_sql_query", side_effect=fake.query), \
         patch.object(db_inventory, "execute_sql_write", side_effect=fake.write), \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      side_effect=lambda *a, **kw: deducted_calls.append(a)):
        stats = db_inventory.sync_inventory_after_chunk_completion(
            "u1",
            "2026-04-30T00:00:00+00:00",
            "2026-05-03T00:00:00+00:00",
        )

    assert stats["reconciled_count"] == 0
    assert stats["items_deducted"] == 0
    assert deducted_calls == []
    assert fake.updates == []


# ---------------------------------------------------------------------------
# 3. Filas con ingredients vacíos se marcan igual (no se reintentan)
# ---------------------------------------------------------------------------
def test_marks_empty_ingredient_rows_as_synced_without_deducting():
    fake = _FakeDB([
        {
            "id": 20, "user_id": "u1",
            "ingredients": [],  # log manual del frontend sin ingredientes
            "consumed_at": "2026-05-01T10:00:00+00:00",
            "inventory_synced_at": None,
        },
    ])
    deducted_calls = []

    with patch("db_core.connection_pool", object()), \
         patch("db_core.execute_sql_query", side_effect=fake.query), \
         patch.object(db_inventory, "execute_sql_write", side_effect=fake.write), \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      side_effect=lambda *a, **kw: deducted_calls.append(a)):
        stats = db_inventory.sync_inventory_after_chunk_completion(
            "u1",
            "2026-04-30T00:00:00+00:00",
            "2026-05-03T00:00:00+00:00",
        )

    assert stats["reconciled_count"] == 1
    assert stats["items_deducted"] == 0
    assert deducted_calls == []  # no deduction sin ingredientes
    assert fake.updates == [20]  # pero sí se marca para no reintentarla


# ---------------------------------------------------------------------------
# 4. Filas fuera de la ventana no se tocan
# ---------------------------------------------------------------------------
def test_ignores_rows_outside_chunk_window():
    fake = _FakeDB([
        {
            "id": 30, "user_id": "u1",
            "ingredients": ["200g pollo"],
            "consumed_at": "2026-04-29T10:00:00+00:00",  # antes de la ventana
            "inventory_synced_at": None,
        },
        {
            "id": 31, "user_id": "u1",
            "ingredients": ["200g res"],
            "consumed_at": "2026-05-04T10:00:00+00:00",  # después de la ventana
            "inventory_synced_at": None,
        },
    ])

    with patch("db_core.connection_pool", object()), \
         patch("db_core.execute_sql_query", side_effect=fake.query), \
         patch.object(db_inventory, "execute_sql_write", side_effect=fake.write), \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      side_effect=lambda *a, **kw: None):
        stats = db_inventory.sync_inventory_after_chunk_completion(
            "u1",
            "2026-04-30T00:00:00+00:00",
            "2026-05-03T00:00:00+00:00",
        )

    assert stats["reconciled_count"] == 0
    assert fake.updates == []


# ---------------------------------------------------------------------------
# 5. Args inválidos → no-op
# ---------------------------------------------------------------------------
def test_invalid_args_are_noop():
    stats = db_inventory.sync_inventory_after_chunk_completion("", "x", "y")
    assert stats == {"reconciled_count": 0, "items_deducted": 0}

    stats = db_inventory.sync_inventory_after_chunk_completion("u", "", "y")
    assert stats == {"reconciled_count": 0, "items_deducted": 0}

    stats = db_inventory.sync_inventory_after_chunk_completion("u", "x", "")
    assert stats == {"reconciled_count": 0, "items_deducted": 0}


# ---------------------------------------------------------------------------
# 6. log_consumed_meal con mark_inventory_synced=True
# ---------------------------------------------------------------------------
def test_log_consumed_meal_marks_inventory_synced_when_requested():
    captured = {}

    def fake_write(query, params=None):
        captured["query"] = query
        captured["params"] = params

    with patch.object(db_facts, "connection_pool", object()), \
         patch.object(db_facts, "execute_sql_write", side_effect=fake_write):
        db_facts.log_consumed_meal(
            user_id="u1",
            meal_name="Pollo con arroz",
            calories=500, protein=40,
            ingredients=["200g pollo", "100g arroz"],
            mark_inventory_synced=True,
        )

    # El SQL debe incluir la columna inventory_synced_at y el último parámetro
    # debe ser el timestamp ISO (no None).
    assert "inventory_synced_at" in captured["query"]
    synced_at_param = captured["params"][-1]
    assert synced_at_param is not None
    assert "T" in synced_at_param  # formato ISO


def test_log_consumed_meal_default_does_not_mark_synced():
    captured = {}

    def fake_write(query, params=None):
        captured["params"] = params

    with patch.object(db_facts, "connection_pool", object()), \
         patch.object(db_facts, "execute_sql_write", side_effect=fake_write):
        db_facts.log_consumed_meal(
            user_id="u1",
            meal_name="Pollo manual",
            calories=300, protein=20,
            ingredients=None,  # path del frontend manual
        )

    # Sin mark_inventory_synced=True, el parámetro debe ser None → la fila
    # quedará pendiente para que la reconciliación al cierre del chunk decida.
    synced_at_param = captured["params"][-1]
    assert synced_at_param is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
