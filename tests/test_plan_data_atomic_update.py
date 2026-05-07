"""
[P0-2] Tests para `db_plans.update_plan_data_atomic`.

Valida que:
  A. La función emite SELECT … FOR UPDATE antes del UPDATE (evita la race
     condition read-modify-write donde dos chunks sobrescribían el contador
     del otro).
  B. La función llama a SET LOCAL lock_timeout antes de tomar el lock para no
     bloquear indefinidamente cuando otro plan tiene el row.
  C. Si la fila no existe (caso `save_new_meal_plan_atomic` la canceló entre
     callers), retorna {} y NO ejecuta UPDATE.
  D. Si el mutator retorna False, NO se ejecuta UPDATE (semántica "no-op").
  E. Si el mutator retorna un dict nuevo, se persiste con UPDATE.
  F. Si el mutator muta in-place y retorna None, también se persiste.
"""
import sys
from unittest.mock import patch, MagicMock, call

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


class _FakeCursor:
    def __init__(self, fetch_value):
        self._fetch_value = fetch_value
        self.executed = []  # list of (sql, params)

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetch_value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def transaction(self):
        return _FakeTransaction()

    def cursor(self, **kwargs):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakePool:
    def __init__(self, cursor):
        self._cursor = cursor

    def connection(self):
        return _FakeConnection(self._cursor)


def _build_pool(plan_data):
    fetch = {"plan_data": plan_data} if plan_data is not None else None
    cursor = _FakeCursor(fetch)
    return _FakePool(cursor), cursor


def test_a_select_for_update_emitted_before_update():
    """[P0-2] La función debe emitir SELECT … FOR UPDATE antes del UPDATE."""
    pool, cursor = _build_pool({"foo": "bar"})
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic

        def mut(pd):
            pd["touched"] = True
            return pd

        result = update_plan_data_atomic("plan-xyz", mut)

    sqls = [s for s, _ in cursor.executed]
    select_idx = next((i for i, s in enumerate(sqls) if "SELECT" in s and "FOR UPDATE" in s), None)
    update_idx = next((i for i, s in enumerate(sqls) if s.startswith("UPDATE meal_plans")), None)
    assert select_idx is not None, "Debe emitirse SELECT … FOR UPDATE"
    assert update_idx is not None, "Debe emitirse UPDATE"
    assert select_idx < update_idx, "SELECT FOR UPDATE debe ir ANTES del UPDATE"
    assert result["touched"] is True


def test_b_lock_timeout_set_before_select():
    """[P0-2] SET LOCAL lock_timeout debe ejecutarse antes del SELECT FOR UPDATE."""
    pool, cursor = _build_pool({"x": 1})
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic
        update_plan_data_atomic("plan-xyz", lambda pd: pd, lock_timeout_ms=5000)

    sqls = [s for s, _ in cursor.executed]
    timeout_idx = next((i for i, s in enumerate(sqls) if "SET LOCAL lock_timeout" in s), None)
    select_idx = next((i for i, s in enumerate(sqls) if "FOR UPDATE" in s), None)
    assert timeout_idx is not None, "Debe setear lock_timeout"
    assert select_idx is not None
    assert timeout_idx < select_idx, "lock_timeout debe ir antes del SELECT FOR UPDATE"
    assert "5000ms" in sqls[timeout_idx]


def test_c_missing_row_returns_empty_no_update():
    """[P0-2] Si meal_plan no existe, retornar {} sin UPDATE."""
    pool, cursor = _build_pool(None)
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic
        mutator_called = []
        result = update_plan_data_atomic(
            "plan-missing",
            lambda pd: mutator_called.append(pd) or pd,
        )
    assert result == {}
    assert not any(s.startswith("UPDATE meal_plans") for s, _ in cursor.executed)
    assert mutator_called == [], "No se debe llamar al mutator si la fila no existe"


def test_d_mutator_returns_false_skips_update():
    """[P0-2] Si el mutator retorna False, NO se persiste."""
    pool, cursor = _build_pool({"x": 1})
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic
        result = update_plan_data_atomic("plan-xyz", lambda pd: False)
    assert result == {"x": 1}
    assert not any(s.startswith("UPDATE meal_plans") for s, _ in cursor.executed)


def test_e_mutator_returns_new_dict_persists():
    """[P0-2] Si el mutator retorna un dict nuevo, se persiste vía UPDATE."""
    pool, cursor = _build_pool({"x": 1})
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic
        result = update_plan_data_atomic("plan-xyz", lambda pd: {"y": 2})
    assert result == {"y": 2}
    update_calls = [(s, p) for s, p in cursor.executed if s.startswith("UPDATE meal_plans")]
    assert len(update_calls) == 1
    # El psycopg Jsonb wrapper hace que p[0] no sea exactamente el dict; solo verifico que
    # el segundo parámetro es el plan_id.
    assert update_calls[0][1][1] == "plan-xyz"


def test_f_mutator_in_place_mutation_persists():
    """[P0-2] Mutación in-place + return None también se persiste."""
    pool, cursor = _build_pool({"counter": 5})
    with patch("db_plans.connection_pool", pool):
        from db_plans import update_plan_data_atomic

        def bump(pd):
            pd["counter"] = pd.get("counter", 0) + 1
            # no return → None

        result = update_plan_data_atomic("plan-xyz", bump)
    assert result["counter"] == 6
    assert any(s.startswith("UPDATE meal_plans") for s, _ in cursor.executed)
