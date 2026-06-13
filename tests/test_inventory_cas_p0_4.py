"""P0-4: tests para CAS-with-retry en `_apply_reservation_delta`.

Antes el flujo SELECT → modify in Python → UPDATE WHERE id=X tenía una ventana
TOCTOU que permitía lost-updates: dos writers concurrentes podían leer el mismo
`reserved_quantity` y ambos sumar su delta, sobreescribiéndose mutuamente.
La implementación incluye el CAS token en el propio UPDATE SQL
(`WHERE id = %s AND reserved_quantity = %s::numeric RETURNING id`) y reintenta
el ciclo SELECT+modify+UPDATE hasta 4 veces con backoff antes de rendirse.

[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado al transporte SQL directo:
los mocks simulan `execute_sql_query` (SELECT) y `execute_sql_write` (UPDATE
CAS con RETURNING id) en lugar del builder PostgREST legacy. La semántica
verificada es LA MISMA: el UPDATE solo aplica si `reserved_quantity` actual
coincide con el valor leído (expected_old redondeado a 4 decimales).

Ejecutar con:
    cd backend && python -m pytest tests/test_inventory_cas_p0_4.py -v --noconftest
"""
import sys
import os
import types
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos para que db_inventory pueda importar sin conexión real.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_k: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_k: None)

# Si otro test file ya stubeó db_core como módulo opaco sin los helpers SQL,
# lo reemplazamos con uno que sí los expone (db_inventory importa
# execute_sql_query/execute_sql_write de db_core a nivel de módulo).
_db_core_stub = sys.modules.get("db_core")
if _db_core_stub is None or not all(
    hasattr(_db_core_stub, _attr)
    for _attr in ("execute_sql_query", "execute_sql_write", "connection_pool")
):
    sys.modules.pop("db_core", None)
    _install_stub(
        "db_core",
        supabase=MagicMock(),
        execute_sql_write=lambda *_a, **_k: None,
        execute_sql_query=lambda *_a, **_k: None,
        execute_sql_transaction=lambda *_a, **_k: True,
        connection_pool=None,
    )

# Si otro test file (e.g. test_chunked_learning_propagation) stubeó shopping_calculator
# como módulo opaco sin las funciones que db_inventory necesita, evict-eamos para que
# Python lo importe de verdad cuando db_inventory haga `from shopping_calculator import ...`.
# db_inventory.py requires _parse_quantity, get_plural_unit, get_master_ingredients —
# evict if ANY of them is missing on the cached stub. Previously we only checked
# _parse_quantity, so a stub providing only that one slipped past the guard and
# crashed db_inventory's import on get_plural_unit.
_sc_stub = sys.modules.get("shopping_calculator")
if _sc_stub is not None and not all(
    hasattr(_sc_stub, _attr)
    for _attr in ("_parse_quantity", "get_plural_unit", "get_master_ingredients")
):
    sys.modules.pop("shopping_calculator", None)

# Si otro test file (e.g. test_chunked_learning_propagation) stubeó db_inventory
# como módulo opaco con solo unas pocas funciones, evict-eamos y re-importamos
# el módulo real para poder testear su lógica interna.
_db_inv_stub = sys.modules.get("db_inventory")
if _db_inv_stub is not None and not hasattr(_db_inv_stub, "_apply_reservation_delta"):
    sys.modules.pop("db_inventory", None)

import db_inventory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers para construir un mock del transporte SQL que simula CAS
# ---------------------------------------------------------------------------

class _FakeCasStore:
    """Mock del par `execute_sql_query`/`execute_sql_write` con semántica CAS.

    El SELECT (`fake_query`) devuelve siempre copias del estado actual de las
    filas (mutables). El UPDATE (`fake_write`) compara el CAS token
    (`reserved_quantity = %s::numeric`, último param) contra el row actual; si
    matchea, aplica el patch y retorna `[{"id": ...}]` (RETURNING id). Si no,
    retorna `[]` (CAS conflict — 0 filas afectadas).

    `before_write(call_idx)` permite inyectar mutaciones concurrentes JUSTO
    antes de cada UPDATE, simulando otro writer entre SELECT y UPDATE.
    """

    def __init__(self, rows):
        # `rows` es lista de dicts mutables. Cada UPDATE exitoso muta el row.
        self._rows = rows
        self.select_calls = 0
        self.write_calls = []  # [(sql, params)]
        self.before_write = None

    def fake_query(self, sql, params=None, fetch_one=False, fetch_all=False):
        assert "FROM user_inventory" in sql, f"SELECT inesperado: {sql!r}"
        assert "user_id = %s" in sql, "el SELECT debe filtrar por user_id"
        self.select_calls += 1
        user_id, ingredient = params[0], params[1]
        return [
            dict(r) for r in self._rows
            if r.get("user_id") == user_id and r.get("ingredient_name") == ingredient
        ]

    def fake_write(self, sql, params=None, returning=False):
        if self.before_write is not None:
            self.before_write(len(self.write_calls))
        self.write_calls.append((sql, params))
        assert "UPDATE user_inventory" in sql, f"UPDATE inesperado: {sql!r}"
        # Invariante CAS: el WHERE debe comparar reserved_quantity contra el
        # token leído (espacio numeric) y usar RETURNING id como detector.
        assert "reserved_quantity = %s::numeric" in sql
        assert "RETURNING id" in sql
        new_reserved, details_wrapped, row_id, expected_old = params
        new_details = getattr(details_wrapped, "obj", details_wrapped)
        updated = []
        for row in self._rows:
            if row.get("id") != row_id:
                continue
            if round(float(row.get("reserved_quantity") or 0), 4) != float(expected_old):
                continue  # CAS conflict → 0 filas afectadas
            row["reserved_quantity"] = new_reserved
            row["reservation_details"] = dict(new_details)
            updated.append({"id": row_id})
        return updated if returning else None


def _patch_store(store):
    return (
        patch.object(db_inventory, "execute_sql_query", side_effect=store.fake_query),
        patch.object(db_inventory, "execute_sql_write", side_effect=store.fake_write),
    )


def _make_inventory_row(row_id="r1", ingredient="Pollo", quantity=500.0, reserved=0.0, details=None, unit="g"):
    return {
        "id": row_id,
        "user_id": "u1",
        "ingredient_name": ingredient,
        "quantity": quantity,
        "unit": unit,
        "reserved_quantity": reserved,
        "reservation_details": details if details is not None else {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_p0_4_cas_succeeds_on_first_attempt_when_no_conflict():
    """Caso happy path: nadie está modificando la fila concurrentemente, el CAS
    matchea en el primer intento."""
    row = _make_inventory_row(reserved=0.0)
    store = _FakeCasStore([row])
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is True
    assert row["reserved_quantity"] == 100.0
    assert row["reservation_details"] == {"chunk:abc:meal:test": 100.0}


def test_p0_4_cas_retries_and_converges_after_concurrent_modification():
    """Simula que entre nuestro SELECT y nuestro UPDATE, otro writer cambió
    `reserved_quantity` de 0→50. El primer CAS falla (expected=0 pero actual=50),
    reintentamos, leemos 50, sumamos nuestro delta y converge a 150."""
    row = _make_inventory_row(reserved=0.0)
    store = _FakeCasStore([row])

    def inject_concurrent_writer(call_idx):
        # Justo antes del primer UPDATE, mutamos la fila para simular un
        # writer concurrente que reservó 50 con otra key.
        if call_idx == 0:
            row["reserved_quantity"] = 50.0
            row["reservation_details"] = {"chunk:other:meal:x": 50.0}

    store.before_write = inject_concurrent_writer
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch("time.sleep"):  # acelerar el backoff
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is True
    # Tras la convergencia, el reserved final debe sumar la reserva concurrente + la nuestra.
    assert row["reserved_quantity"] == 150.0
    assert row["reservation_details"]["chunk:other:meal:x"] == 50.0
    assert row["reservation_details"]["chunk:abc:meal:test"] == 100.0
    # El retry implica al menos 2 SELECTs (el segundo debe ser fresh).
    assert store.select_calls >= 2


def test_p0_4_cas_returns_false_after_max_retries_on_persistent_conflict():
    """Si un writer concurrente está cambiando reserved_quantity sin parar entre
    nuestros SELECT y UPDATE, agotamos los reintentos y devolvemos False sin
    corromper el estado."""
    row = _make_inventory_row(reserved=0.0)
    store = _FakeCasStore([row])

    def always_conflict(call_idx):
        # Mutar la fila ANTES de cada UPDATE para que el CAS siempre falle.
        row["reserved_quantity"] = float((call_idx + 1) * 10)

    store.before_write = always_conflict
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch("time.sleep"):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
            max_retries=3,
        )

    assert ok is False
    # Debe haberse intentado el UPDATE al menos `max_retries` veces.
    assert len(store.write_calls) >= 3
    # La key NO debe haber quedado registrada (ningún UPDATE matcheó).
    assert "chunk:abc:meal:test" not in row["reservation_details"]


def test_p0_4_cas_release_only_path_uses_same_atomic_protection():
    """release_only=True también debe pasar por CAS para no perder releases en
    presencia de writers concurrentes (e.g., dos rechazos de la misma comida)."""
    initial_details = {"chunk:abc:meal:test": 100.0}
    row = _make_inventory_row(reserved=100.0, details=initial_details)
    store = _FakeCasStore([row])
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=0.0):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=0.0, unit="g", reservation_key="chunk:abc:meal:test",
            release_only=True,
        )

    assert ok is True
    assert row["reserved_quantity"] == 0.0
    assert "chunk:abc:meal:test" not in row["reservation_details"]
    # El release también viajó por el UPDATE CAS (no por un UPDATE ciego).
    assert len(store.write_calls) == 1
    assert store.write_calls[0][1][3] == 100.0  # expected_old = lo leído


def test_p0_4_cas_returns_false_when_no_compatible_unit_row_exists():
    """Si la fila de inventario tiene unidad incompatible (convert_amount → None),
    no es un problema de race — no reintentar."""
    row = _make_inventory_row(unit="cabeza")  # incompatible con "g" si convert_amount falla
    store = _FakeCasStore([row])
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=None) as mock_convert, \
         patch("time.sleep") as mock_sleep:
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
            max_retries=4,
        )

    assert ok is False
    # Si no hay fila compatible, no tiene sentido reintentar: backoff nunca debe correr.
    assert mock_sleep.call_count == 0
    # convert_amount se llama una vez (un row, una vez).
    assert mock_convert.call_count == 1
    # Y jamás se emite un UPDATE.
    assert store.write_calls == []


def test_p0_4_cas_returns_false_when_user_has_no_inventory_row():
    """Si el SELECT no devuelve filas (ingrediente nunca registrado), devolver
    False inmediatamente sin reintentar."""
    store = _FakeCasStore([])  # sin filas
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch("time.sleep") as mock_sleep:
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is False
    assert mock_sleep.call_count == 0
    assert store.write_calls == []


def test_p0_4_cas_idempotent_when_same_reservation_already_applied():
    """Si la reserva con el mismo key ya está aplicada con el mismo target_qty,
    el caller obtiene True sin más writes (evita escrituras innecesarias)."""
    row = _make_inventory_row(
        reserved=100.0,
        details={"chunk:abc:meal:test": 100.0},
    )
    store = _FakeCasStore([row])
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is True
    # No debe haber UPDATE — el state ya era el deseado.
    assert store.write_calls == []


def test_p0_4_cas_uses_rounded_values_in_eq_filter_to_avoid_float_precision_drift():
    """Floats con decimales largos pueden no comparar exactamente entre Python
    y Postgres; la helper redondea a 4 decimales el nuevo valor Y el token CAS
    antes de pasarlos como params del UPDATE."""
    row = _make_inventory_row(reserved=0.0)
    store = _FakeCasStore([row])
    p_query, p_write = _patch_store(store)

    with p_query, p_write, \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=33.333333):
        db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=33.333333, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert len(store.write_calls) >= 1
    sql, params = store.write_calls[0]
    # params = (new_reserved, Jsonb(details), row_id, expected_old)
    # expected_old=0.0 → el CAS token viaja redondeado como 0.0
    assert params[3] == 0.0
    # new_reserved viaja redondeado a 4 decimales (no 33.333333 crudo).
    assert params[0] == round(33.333333, 4)
