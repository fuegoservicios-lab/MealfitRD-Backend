"""P0-4: tests para CAS-with-retry en `_apply_reservation_delta`.

Antes el flujo SELECT → modify in Python → UPDATE WHERE id=X tenía una ventana
TOCTOU que permitía lost-updates: dos writers concurrentes podían leer el mismo
`reserved_quantity` y ambos sumar su delta, sobreescribiéndose mutuamente.
La nueva implementación incluye `eq("reserved_quantity", expected_old)` en el
UPDATE como CAS token y reintenta el ciclo SELECT+modify+UPDATE hasta 4 veces
con backoff antes de rendirse.

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
if "langchain_google_genai" not in sys.modules:
    _install_stub("langchain_google_genai", GoogleGenerativeAIEmbeddings=object)

# Si otro test file ya stubeó db_core como módulo opaco sin `supabase`, lo
# reemplazamos con uno que sí lo expone (el real lo necesita).
_db_core_stub = sys.modules.get("db_core")
if _db_core_stub is None or not hasattr(_db_core_stub, "supabase"):
    sys.modules.pop("db_core", None)
    _install_stub(
        "db_core",
        supabase=MagicMock(),
        execute_sql_write=lambda *_a, **_k: None,
        execute_sql_query=lambda *_a, **_k: None,
        connection_pool=None,
    )

# Si otro test file (e.g. test_chunked_learning_propagation) stubeó shopping_calculator
# como módulo opaco sin las funciones que db_inventory necesita, evict-eamos para que
# Python lo importe de verdad cuando db_inventory haga `from shopping_calculator import ...`.
# db_inventory.py:8 requires _parse_quantity, get_plural_unit, get_master_ingredients —
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
# Helpers para construir un mock-supabase que simula CAS
# ---------------------------------------------------------------------------

class _FakeSupabaseTable:
    """Mock de supabase.table('user_inventory') con SELECT y UPDATE-CAS programables.

    El SELECT devuelve siempre el estado actual del row (mutable).
    El UPDATE comprueba los `eq` filters; si todos matchean el row actual, aplica
    el patch. Si no, devuelve `data=[]` (CAS conflict).
    """

    def __init__(self, rows):
        # `rows` es lista de dicts mutables. Cada UPDATE exitoso muta el row.
        self._rows = rows
        self._select_filters: dict = {}
        self._update_filters: dict = {}
        self._update_payload: dict | None = None
        self._mode: str | None = None  # "select" | "update"

    def select(self, *_cols):
        self._mode = "select"
        self._select_filters = {}
        return self

    def update(self, payload):
        self._mode = "update"
        self._update_filters = {}
        self._update_payload = dict(payload)
        return self

    def eq(self, column, value):
        target = self._select_filters if self._mode == "select" else self._update_filters
        target[column] = value
        return self

    def gt(self, column, value):
        # No usado por _apply_reservation_delta pero por completeness.
        target = self._select_filters if self._mode == "select" else self._update_filters
        target[f"_gt:{column}"] = value
        return self

    def execute(self):
        if self._mode == "select":
            matches = []
            for row in self._rows:
                if all(row.get(k) == v for k, v in self._select_filters.items() if not k.startswith("_gt:")):
                    matches.append(dict(row))  # devolvemos copia para que el caller no mute el row interno
            return MagicMock(data=matches)

        if self._mode == "update":
            # Simular CAS: el UPDATE solo aplica si los filtros eq matchean el row actual.
            updated = []
            for row in self._rows:
                if all(row.get(k) == v for k, v in self._update_filters.items()):
                    row.update(self._update_payload)
                    updated.append(dict(row))
            return MagicMock(data=updated)

        return MagicMock(data=[])


class _FakeSupabaseClient:
    def __init__(self, rows):
        self._rows = rows
        self._table = _FakeSupabaseTable(rows)

    def table(self, name):
        # Reusamos la misma instancia para que SELECT y UPDATE compartan estado.
        return self._table


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
    fake_client = _FakeSupabaseClient([row])

    with patch.object(db_inventory, "supabase", fake_client), \
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
    fake_client = _FakeSupabaseClient([row])

    # Inyectar mutación entre SELECT y UPDATE en el primer intento.
    original_execute = fake_client._table.execute
    call_count = {"n": 0}

    def execute_with_injection():
        call_count["n"] += 1
        # Cuando estamos a punto de ejecutar el primer UPDATE (3rd call: select, intercept, update),
        # mutamos la fila para simular un writer concurrente.
        if call_count["n"] == 2 and fake_client._table._mode == "update":
            row["reserved_quantity"] = 50.0
            row["reservation_details"] = {"chunk:other:meal:x": 50.0}
        return original_execute()

    with patch.object(db_inventory, "supabase", fake_client), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch.object(fake_client._table, "execute", side_effect=execute_with_injection), \
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


def test_p0_4_cas_returns_false_after_max_retries_on_persistent_conflict():
    """Si un writer concurrente está cambiando reserved_quantity sin parar entre
    nuestros SELECT y UPDATE, agotamos los reintentos y devolvemos False sin
    corromper el estado."""
    row = _make_inventory_row(reserved=0.0)
    fake_client = _FakeSupabaseClient([row])

    # Mutar la fila ANTES de cada UPDATE para que el CAS siempre falle.
    original_execute = fake_client._table.execute
    mutation_counter = {"n": 0}

    def always_conflict():
        if fake_client._table._mode == "update":
            mutation_counter["n"] += 1
            row["reserved_quantity"] = float(mutation_counter["n"] * 10)
        return original_execute()

    with patch.object(db_inventory, "supabase", fake_client), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch.object(fake_client._table, "execute", side_effect=always_conflict), \
         patch("time.sleep"):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
            max_retries=3,
        )

    assert ok is False
    # El conflict counter debe reflejar que se hicieron al menos `max_retries` intentos.
    assert mutation_counter["n"] >= 3
    # La key NO debe haber quedado registrada (ningún UPDATE matcheó).
    assert "chunk:abc:meal:test" not in row["reservation_details"]


def test_p0_4_cas_release_only_path_uses_same_atomic_protection():
    """release_only=True también debe pasar por CAS para no perder releases en
    presencia de writers concurrentes (e.g., dos rechazos de la misma comida)."""
    initial_details = {"chunk:abc:meal:test": 100.0}
    row = _make_inventory_row(reserved=100.0, details=initial_details)
    fake_client = _FakeSupabaseClient([row])

    with patch.object(db_inventory, "supabase", fake_client), \
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


def test_p0_4_cas_returns_false_when_no_compatible_unit_row_exists():
    """Si la fila de inventario tiene unidad incompatible (convert_amount → None),
    no es un problema de race — no reintentar."""
    row = _make_inventory_row(unit="cabeza")  # incompatible con "g" si convert_amount falla
    fake_client = _FakeSupabaseClient([row])

    with patch.object(db_inventory, "supabase", fake_client), \
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


def test_p0_4_cas_returns_false_when_user_has_no_inventory_row():
    """Si el SELECT no devuelve filas (ingrediente nunca registrado), devolver
    False inmediatamente sin reintentar."""
    fake_client = _FakeSupabaseClient([])  # sin filas

    with patch.object(db_inventory, "supabase", fake_client), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch("time.sleep") as mock_sleep:
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is False
    assert mock_sleep.call_count == 0


def test_p0_4_cas_idempotent_when_same_reservation_already_applied():
    """Si la reserva con el mismo key ya está aplicada con el mismo target_qty,
    el caller obtiene True sin más writes (evita escrituras innecesarias)."""
    row = _make_inventory_row(
        reserved=100.0,
        details={"chunk:abc:meal:test": 100.0},
    )
    fake_client = _FakeSupabaseClient([row])

    update_count = {"n": 0}
    original_update = fake_client._table.update
    def count_update(payload):
        update_count["n"] += 1
        return original_update(payload)

    with patch.object(db_inventory, "supabase", fake_client), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=100.0), \
         patch.object(fake_client._table, "update", side_effect=count_update):
        ok = db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=100.0, unit="g", reservation_key="chunk:abc:meal:test",
        )

    assert ok is True
    # No debe haber UPDATE — el state ya era el deseado.
    assert update_count["n"] == 0


def test_p0_4_cas_uses_rounded_values_in_eq_filter_to_avoid_float_precision_drift():
    """Floats con decimales largos pueden no comparar exactamente entre Python
    y Postgres; la helper redondea a 4 decimales antes de pasar a `.eq`."""
    captured_filters = []

    class _CapturingTable(_FakeSupabaseTable):
        def execute(self):
            if self._mode == "update":
                captured_filters.append(dict(self._update_filters))
            return super().execute()

    row = _make_inventory_row(reserved=0.0)
    client = _FakeSupabaseClient([row])
    client._table = _CapturingTable([row])

    with patch.object(db_inventory, "supabase", client), \
         patch.object(db_inventory, "get_master_ingredients", return_value=[]), \
         patch.object(db_inventory, "convert_amount", return_value=33.333333):
        db_inventory._apply_reservation_delta(
            user_id="u1", ingredient_name="Pollo",
            quantity=33.333333, unit="g", reservation_key="chunk:abc:meal:test",
        )

    # El UPDATE debe contener `reserved_quantity` redondeado a 4 decimales en el eq filter.
    assert len(captured_filters) >= 1
    # expected_old=0.0 → en el filter aparece como 0.0
    assert captured_filters[0]["reserved_quantity"] == 0.0
