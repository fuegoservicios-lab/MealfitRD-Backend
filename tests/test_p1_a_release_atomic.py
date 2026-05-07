"""[P1-A] Tests de la liberación transaccional de reservas en `release_chunk_reservations`.

Bug original: si Supabase devolvía error a mitad del loop de UPDATEs por fila, el chunk
quedaba con reservas parcialmente liberadas. La próxima invocación NO podía detectar
las pendientes porque las keys ya removidas no aparecían en `reservation_details`.

Fix: batch atómico vía `execute_sql_transaction`. Si cualquier UPDATE falla, psycopg
hace rollback automático y NINGUNA fila queda modificada.

Cubre:
  1. SELECT vacío → no llama a la transacción ni retorna nada.
  2. Filas sin keys del chunk objetivo → skip transparente.
  3. Filas con keys del chunk → transacción atómica con un query por fila afectada.
  4. Si la transacción atómica falla → retorna 0 sin modificaciones.
  5. Sin connection_pool → fallback no-transaccional preserva compatibilidad.
  6. Conteo de keys liberadas (no de filas) preservado del contrato P0-4.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_a_release_atomic.py -v
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos necesarios para importar db_inventory aislado.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


# [P0-5] Mirror the eviction guard from test_inventory_cas_p0_4.py: if another test
# file already installed a `shopping_calculator` stub that lacks any of the three
# functions `db_inventory` imports (cron_tasks pulls in db_inventory transitively
# via `from db_inventory import ...` at module-load), evict the stub so Python
# can reload the real module. Without this, `import db_inventory` raised
# `ImportError: cannot import name 'get_plural_unit' from 'shopping_calculator'`
# whenever a propagation/pantry-validation test loaded first with a partial stub.
_sc_existing = sys.modules.get("shopping_calculator")
if _sc_existing is not None and not all(
    hasattr(_sc_existing, _attr)
    for _attr in ("_parse_quantity", "get_plural_unit", "get_master_ingredients")
):
    sys.modules.pop("shopping_calculator", None)

try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_plural_unit=lambda *_a, **_kw: "uds",
        get_master_ingredients=lambda *_a, **_kw: [],
        _parse_quantity=_stub_parse_quantity,
    )

if "db_core" not in sys.modules:
    _install_stub(
        "db_core",
        supabase=None,
        execute_sql_query=lambda *_a, **_kw: None,
        execute_sql_write=lambda *_a, **_kw: None,
        execute_sql_transaction=lambda *_a, **_kw: True,
        connection_pool=None,
    )

# Otros tests P0/P1 instalan stubs de `db_inventory` para evitar cargar el módulo real
# (que arrastra dependencias pesadas). Pero ESTE test SÍ necesita el módulo real porque
# está testeando una de sus funciones (release_chunk_reservations). Si encontramos un
# stub previo en sys.modules (detectable porque le falta el atributo `supabase` que el
# real siempre tiene), lo removemos para que el `import db_inventory` cargue el real.
if "db_inventory" in sys.modules and not hasattr(sys.modules["db_inventory"], "supabase"):
    del sys.modules["db_inventory"]


from unittest.mock import patch, MagicMock
import db_inventory


# ---------------------------------------------------------------------------
# Helpers para mockear el SELECT inicial
# ---------------------------------------------------------------------------
class _FakeSupabaseTable:
    """Simula la chain `supabase.table(...).select(...).eq(...).gt(...).execute()`."""

    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_kw):
        return self

    def eq(self, *_a, **_kw):
        return self

    def gt(self, *_a, **_kw):
        return self

    def execute(self):
        return MagicMock(data=self._rows)


def _patch_supabase_with_rows(rows):
    fake = MagicMock()
    fake.table = lambda *_a, **_kw: _FakeSupabaseTable(rows)
    return patch("db_inventory.supabase", fake)


# ---------------------------------------------------------------------------
# 1. SELECT vacío → no transacción
# ---------------------------------------------------------------------------
def test_no_rows_returns_zero_no_transaction():
    tx_calls = []

    with _patch_supabase_with_rows([]), \
         patch("db_core.connection_pool", MagicMock()), \
         patch("db_core.execute_sql_transaction", side_effect=lambda q: tx_calls.append(q)):
        released = db_inventory.release_chunk_reservations("user-1", "chunk-aaa")

    assert released == 0
    assert tx_calls == [], "no debe iniciar transacción si no hay filas"


# ---------------------------------------------------------------------------
# 2. Filas sin keys del chunk objetivo → skip
# ---------------------------------------------------------------------------
def test_rows_without_target_chunk_key_skipped():
    rows = [
        {
            "id": "row-1",
            "reserved_quantity": 5.0,
            "reservation_details": {"chunk:OTHER:meal:pollo": 5.0},
        },
    ]
    tx_calls = []

    with _patch_supabase_with_rows(rows), \
         patch("db_core.connection_pool", MagicMock()), \
         patch("db_core.execute_sql_transaction", side_effect=lambda q: tx_calls.append(q)):
        released = db_inventory.release_chunk_reservations("user-1", "chunk-TARGET")

    assert released == 0
    assert tx_calls == [], "no debe transaccionar si ninguna fila tiene keys del chunk objetivo"


# ---------------------------------------------------------------------------
# 3. Filas con keys → transacción atómica con un query por fila
# ---------------------------------------------------------------------------
def test_rows_with_target_keys_transaction_per_affected_row():
    rows = [
        {
            "id": "row-pollo",
            "reserved_quantity": 5.0,
            "reservation_details": {
                "chunk:TARGET:meal:pollo": 3.0,
                "chunk:OTHER:meal:pollo": 2.0,
            },
        },
        {
            "id": "row-arroz",
            "reserved_quantity": 4.0,
            "reservation_details": {
                "chunk:TARGET:meal:arroz1": 2.0,
                "chunk:TARGET:meal:arroz2": 1.5,
            },
        },
        {
            "id": "row-cebolla",
            "reserved_quantity": 1.0,
            "reservation_details": {"chunk:UNRELATED:meal:cebolla": 1.0},
        },
    ]
    captured_tx = []

    def fake_tx(queries):
        captured_tx.append(queries)
        return True

    with _patch_supabase_with_rows(rows), \
         patch("db_core.connection_pool", MagicMock()), \
         patch("db_core.execute_sql_transaction", side_effect=fake_tx):
        released = db_inventory.release_chunk_reservations("user-1", "TARGET")

    # 3 keys liberadas (1 en row-pollo, 2 en row-arroz). row-cebolla no se toca.
    assert released == 3
    # 1 transacción.
    assert len(captured_tx) == 1
    # Dentro de la transacción: 2 UPDATEs (uno por fila afectada, no por key).
    queries = captured_tx[0]
    assert len(queries) == 2
    affected_row_ids = {q[1][2] for q in queries}
    assert affected_row_ids == {"row-pollo", "row-arroz"}
    # row-cebolla NO está en queries.
    assert "row-cebolla" not in affected_row_ids
    # Verificar la nueva reserved_quantity para row-pollo: 5.0 - 3.0 = 2.0.
    pollo_query = next(q for q in queries if q[1][2] == "row-pollo")
    assert pollo_query[1][0] == 2.0
    # row-arroz: 4.0 - 2.0 - 1.5 = 0.5.
    arroz_query = next(q for q in queries if q[1][2] == "row-arroz")
    assert arroz_query[1][0] == 0.5


# ---------------------------------------------------------------------------
# 4. Transacción falla → retorna 0 sin modificaciones
# ---------------------------------------------------------------------------
def test_transaction_failure_returns_zero():
    rows = [
        {
            "id": "row-pollo",
            "reserved_quantity": 5.0,
            "reservation_details": {"chunk:TARGET:meal:pollo": 3.0},
        },
    ]

    def boom(*_a, **_kw):
        raise RuntimeError("DB connection lost mid-transaction")

    with _patch_supabase_with_rows(rows), \
         patch("db_core.connection_pool", MagicMock()), \
         patch("db_core.execute_sql_transaction", side_effect=boom):
        released = db_inventory.release_chunk_reservations("user-1", "TARGET")

    # All-or-nothing: si la transacción falló, retorna 0. El cron de cleanup recogerá.
    assert released == 0


# ---------------------------------------------------------------------------
# 5. Sin connection_pool → fallback no-transaccional
# ---------------------------------------------------------------------------
def test_no_pool_falls_back_to_supabase_per_row():
    rows = [
        {
            "id": "row-pollo",
            "reserved_quantity": 5.0,
            "reservation_details": {"chunk:TARGET:meal:pollo": 3.0},
        },
    ]
    update_calls = []

    def fake_update(row_id, reserved, details):
        update_calls.append((row_id, reserved, details))

    with _patch_supabase_with_rows(rows), \
         patch("db_core.connection_pool", None), \
         patch("db_inventory._update_row_reservation", side_effect=fake_update):
        released = db_inventory.release_chunk_reservations("user-1", "TARGET")

    # Fallback sigue siendo funcional aunque no atómico.
    assert released == 1
    assert len(update_calls) == 1
    assert update_calls[0][0] == "row-pollo"
    assert update_calls[0][1] == 2.0  # 5.0 - 3.0


# ---------------------------------------------------------------------------
# 6. Inputs inválidos → early return
# ---------------------------------------------------------------------------
def test_empty_user_id_returns_zero():
    assert db_inventory.release_chunk_reservations("", "chunk-x") == 0


def test_empty_chunk_id_returns_zero():
    assert db_inventory.release_chunk_reservations("user-1", "") == 0


# ---------------------------------------------------------------------------
# 7. reservation_details como string JSON (caso real de Supabase)
# ---------------------------------------------------------------------------
def test_reservation_details_as_json_string():
    """Algunas integraciones devuelven `reservation_details` serializado como string."""
    rows = [
        {
            "id": "row-1",
            "reserved_quantity": 4.0,
            "reservation_details": '{"chunk:TARGET:meal:pollo": 2.5}',
        },
    ]
    captured_tx = []

    with _patch_supabase_with_rows(rows), \
         patch("db_core.connection_pool", MagicMock()), \
         patch("db_core.execute_sql_transaction", side_effect=lambda q: captured_tx.append(q)):
        released = db_inventory.release_chunk_reservations("user-1", "TARGET")

    assert released == 1
    assert len(captured_tx) == 1
    assert captured_tx[0][0][1][0] == 1.5  # 4.0 - 2.5
