"""[P0.2] Tests para la columna `source` en user_inventory.

Cubre el "MERGE inteligente" del flujo de restock que distingue items que el
usuario añadió a mano (`source='manual'`) de los que vinieron de una lista de
compras (`source='shopping_list'`):

  1. add_or_update_inventory_item INSERT respeta el parámetro `source`.
  2. add_or_update_inventory_item UPDATE NO modifica `source` — first-writer-wins
     (un restock que sume sobre un item manual lo deja como manual).
  3. restock_inventory pasa source='shopping_list' a las nuevas filas.
  4. replace_shopping_list_only_items borra sólo source='shopping_list' y
     preserva el resto.

Ejecutar:
    cd backend && python -m pytest tests/test_inventory_source_column.py -v
"""
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
# Mock de un client Supabase con cadena fluent (table().select().eq().execute()).
# Captura cada llamada para que los tests aserten lo enviado.
# ---------------------------------------------------------------------------
class _Captured:
    """Registra cada query/insert/update/delete con su payload final."""
    def __init__(self):
        self.inserts = []   # list[dict] — payload pasado a .insert()
        self.updates = []   # list[(filter_eq, payload)]
        self.deletes = []   # list[filter_eq]
        self.selects = []   # list[(filters, count_mode)]


def _make_fake_supabase(captured: _Captured, existing_rows_by_name=None):
    """Construye un mock fluent. `existing_rows_by_name` simula filas en DB.
    Cada nodo de la cadena devuelve self salvo `execute()` que devuelve
    un MagicMock con .data y .count.
    """
    existing_rows_by_name = existing_rows_by_name or {}

    class _Chain:
        def __init__(self, table):
            self._table = table
            self._op = None
            self._filters = {}
            self._neq_filters = {}
            self._payload = None
            self._count_mode = None

        # ------- ops ----------------------------------------------------
        def select(self, *_cols, count=None):
            self._op = "select"
            self._count_mode = count
            return self

        def insert(self, payload):
            self._op = "insert"
            self._payload = payload
            return self

        def update(self, payload):
            self._op = "update"
            self._payload = payload
            return self

        def delete(self, count=None):
            self._op = "delete"
            self._count_mode = count
            return self

        # ------- filters ------------------------------------------------
        def eq(self, col, val):
            self._filters[col] = val
            return self

        def neq(self, col, val):
            self._neq_filters[col] = val
            return self

        def gt(self, col, val):
            self._filters[f"{col}__gt"] = val
            return self

        def gte(self, col, val):
            self._filters[f"{col}__gte"] = val
            return self

        def lt(self, col, val):
            self._filters[f"{col}__lt"] = val
            return self

        def is_(self, col, val):
            self._filters[f"{col}__is"] = val
            return self

        # ------- terminator ---------------------------------------------
        def execute(self):
            res = MagicMock()
            if self._op == "select":
                if self._table == "user_inventory":
                    name = self._filters.get("ingredient_name")
                    rows = existing_rows_by_name.get(name, [])
                    if "source" in self._neq_filters:
                        rows = [r for r in rows if r.get("source") != self._neq_filters["source"]]
                    if self._filters.get("source"):
                        rows = [r for r in rows if r.get("source") == self._filters["source"]]
                    captured.selects.append(
                        (dict(self._filters), dict(self._neq_filters), self._count_mode)
                    )
                    res.data = rows
                    res.count = len(rows)
                else:
                    res.data = []
                    res.count = 0
            elif self._op == "insert":
                captured.inserts.append(dict(self._payload))
                res.data = [self._payload]
            elif self._op == "update":
                captured.updates.append((dict(self._filters), dict(self._payload)))
                res.data = []
            elif self._op == "delete":
                captured.deletes.append((dict(self._filters), dict(self._neq_filters)))
                res.data = []
                res.count = 1  # arbitrario para test
            return res

    fake = MagicMock()
    fake.table = lambda t: _Chain(t)
    return fake


# ---------------------------------------------------------------------------
# 1. INSERT respeta el parámetro `source`
# ---------------------------------------------------------------------------
def test_insert_with_source_manual():
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        ok = db_inventory.add_or_update_inventory_item(
            "u1", "Manzana", 3.0, "ud", source="manual",
        )

    assert ok is True
    assert len(captured.inserts) == 1
    assert captured.inserts[0]["source"] == "manual"


def test_insert_with_source_shopping_list():
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        ok = db_inventory.add_or_update_inventory_item(
            "u1", "Pollo", 500.0, "g", source="shopping_list",
        )

    assert ok is True
    assert captured.inserts[0]["source"] == "shopping_list"


def test_insert_default_source_is_manual():
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]):
        db_inventory.add_or_update_inventory_item("u1", "Arroz", 100.0, "g")

    assert captured.inserts[0]["source"] == "manual"


# ---------------------------------------------------------------------------
# 2. UPDATE preserva `source` (first-writer-wins)
# ---------------------------------------------------------------------------
def test_update_does_not_overwrite_source():
    """Si una fila existe como manual y un restock suma sobre ella, source
    debe permanecer 'manual'. El UPDATE payload NO debe contener 'source'.
    """
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={
        "Manzana": [
            {"id": 42, "quantity": 3.0, "unit": "ud", "source": "manual"},
        ],
    })

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=2.0):
        # Restock que pretende sumar 2 manzanas más sobre la fila manual.
        db_inventory.add_or_update_inventory_item(
            "u1", "Manzana", 2.0, "ud", source="shopping_list",
        )

    # No debe haber INSERT (la fila ya existía y se sumó).
    assert captured.inserts == []
    # Debe haber un UPDATE...
    assert len(captured.updates) == 1
    update_payload = captured.updates[0][1]
    # ...y el payload del UPDATE NO debe contener 'source' (preserva manual).
    assert "source" not in update_payload
    # Sí debe actualizar quantity y last_mutation_type.
    assert update_payload["quantity"] == 5.0  # 3 + 2
    assert "last_mutation_type" in update_payload


# ---------------------------------------------------------------------------
# 3. restock_inventory tagea source='shopping_list'
# ---------------------------------------------------------------------------
def test_restock_inventory_tags_new_rows_as_shopping_list():
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    def _fake_normalize_name(n):
        return n

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch("shopping_calculator.normalize_name", side_effect=_fake_normalize_name, create=True):
        ok = db_inventory.restock_inventory("u1", [
            {"name": "Tomate", "quantity": 4, "unit": "ud"},
            {"name": "Cebolla", "quantity": 1, "unit": "kg"},
        ])

    assert ok is True
    assert len(captured.inserts) == 2
    sources = [r["source"] for r in captured.inserts]
    assert sources == ["shopping_list", "shopping_list"]


# ---------------------------------------------------------------------------
# 4. replace_shopping_list_only_items preserva manual
# ---------------------------------------------------------------------------
def test_replace_shopping_list_only_deletes_only_shopping_rows():
    captured = _Captured()
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory", return_value=True) as rmock:
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Debe haber emitido un DELETE filtrado por source='shopping_list'.
    assert len(captured.deletes) == 1
    delete_filters = captured.deletes[0][0]
    assert delete_filters.get("user_id") == "u1"
    assert delete_filters.get("source") == "shopping_list"

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
    fake_sb = _make_fake_supabase(captured, existing_rows_by_name={})

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory") as rmock:
        stats = db_inventory.replace_shopping_list_only_items("u1", [])

    rmock.assert_not_called()
    # El DELETE igualmente se ejecutó (limpió shopping_list previos).
    assert len(captured.deletes) == 1


def test_replace_shopping_list_only_noop_on_empty_user():
    stats = db_inventory.replace_shopping_list_only_items("", [{"name": "x"}])
    assert stats == {"deleted_shopping_rows": 0, "inserted_rows": 0, "preserved_manual_rows": 0}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
