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
def _make_fake_supabase_with_snapshot(captured: _Captured, snapshot_rows: list):
    """Variante de _make_fake_supabase que devuelve `snapshot_rows` cuando se
    consulta SELECT * con filtros (user_id, source). Necesario para testar el
    rollback path que requiere snapshot real, no `[]`."""
    class _Chain:
        def __init__(self, table):
            self._table = table
            self._op = None
            self._filters = {}
            self._neq_filters = {}
            self._payload = None
            self._count_mode = None

        def select(self, *_cols, count=None):
            self._op = "select"
            self._count_mode = count
            return self

        def insert(self, payload):
            self._op = "insert"
            self._payload = payload
            return self

        def delete(self, count=None):
            self._op = "delete"
            self._count_mode = count
            return self

        def eq(self, col, val):
            self._filters[col] = val
            return self

        def neq(self, col, val):
            self._neq_filters[col] = val
            return self

        def execute(self):
            res = MagicMock()
            if self._op == "select":
                if (
                    self._table == "user_inventory"
                    and self._filters.get("source") == "shopping_list"
                ):
                    captured.selects.append(
                        (dict(self._filters), dict(self._neq_filters), self._count_mode)
                    )
                    res.data = list(snapshot_rows)
                    res.count = len(snapshot_rows)
                else:
                    captured.selects.append(
                        (dict(self._filters), dict(self._neq_filters), self._count_mode)
                    )
                    res.data = []
                    res.count = 0
            elif self._op == "insert":
                captured.inserts.append(dict(self._payload))
                res.data = [self._payload]
            elif self._op == "delete":
                captured.deletes.append((dict(self._filters), dict(self._neq_filters)))
                res.data = []
                res.count = len(snapshot_rows)
            return res

    fake = MagicMock()
    fake.table = lambda t: _Chain(t)
    return fake


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
    fake_sb = _make_fake_supabase_with_snapshot(captured, snapshot_rows=snapshot)

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Rollback se activó
    assert stats["rolled_back"] is True
    # Se intentó restaurar las 2 filas snapshotted (vía INSERT)
    assert len(captured.inserts) == 2
    # Y los inserts NO contienen las columnas managed-by-DB
    for ins in captured.inserts:
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
    fake_sb = _make_fake_supabase_with_snapshot(captured, snapshot_rows=snapshot)

    def _raise(*_a, **_kw):
        raise RuntimeError("simulated DB blip")

    with patch.object(db_inventory, "supabase", fake_sb), \
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
    fake_sb = _make_fake_supabase_with_snapshot(captured, snapshot_rows=snapshot)

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory", return_value=True):
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    assert stats["rolled_back"] is False
    # Las únicas inserciones son las del restock mockeado (que no usa el fake_sb,
    # devolvió True directamente). Por lo tanto captured.inserts debe estar vacío.
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
    fake_sb = _make_fake_supabase_with_snapshot(captured, snapshot_rows=snapshot)

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory", return_value=False):
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
    fake_sb = _make_fake_supabase_with_snapshot(captured, snapshot_rows=snapshot)

    # Asegurar que la env var NO está seteada (defensive — no monkeypatch.setenv)
    if "MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK" in os.environ:
        old = os.environ.pop("MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK")
    else:
        old = None
    try:
        with patch.object(db_inventory, "supabase", fake_sb), \
             patch.object(db_inventory, "restock_inventory", return_value=False):
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

    def _fake_table(t):
        chain = MagicMock()
        chain._table = t
        # SELECT con count="exact" (preserved manual rows) → ok
        # SELECT con `*` (snapshot) → raise
        # DELETE → ok
        call_state = {"select_count": 0}

        def _select(*_cols, count=None):
            call_state["select_count"] += 1
            chain._is_snapshot = (count is None)  # snapshot usa select("*") sin count
            chain._count_mode = count
            return chain

        def _eq(col, val):
            chain._last_filters = getattr(chain, "_last_filters", {})
            chain._last_filters[col] = val
            return chain

        def _neq(col, val):
            return chain

        def _delete(count=None):
            chain._op = "delete"
            return chain

        def _execute():
            res = MagicMock()
            if getattr(chain, "_op", None) == "delete":
                res.data, res.count = [], 0
                return res
            if getattr(chain, "_is_snapshot", False):
                raise RuntimeError("simulated snapshot fetch error")
            res.data, res.count = [], 0
            return res

        chain.select = _select
        chain.insert = lambda p: chain
        chain.delete = _delete
        chain.eq = _eq
        chain.neq = _neq
        chain.execute = _execute
        return chain

    fake_sb = MagicMock()
    fake_sb.table = _fake_table

    with patch.object(db_inventory, "supabase", fake_sb), \
         patch.object(db_inventory, "restock_inventory", return_value=False):
        # No debe levantar — snapshot falla, degrada legacy, restock falla
        # también pero sin snapshot rollback.
        stats = db_inventory.replace_shopping_list_only_items(
            "u1", [{"name": "Tomate", "quantity": 2, "unit": "ud"}],
        )

    # Sin rollback (snapshot fue None)
    assert stats["rolled_back"] is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
