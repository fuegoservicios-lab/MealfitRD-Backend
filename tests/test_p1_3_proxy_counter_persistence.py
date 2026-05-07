"""[P1-3] Tests para la persistencia atómica de contadores de proxy.

Cubre la unificación de la fuente de verdad de `_consecutive_proxy_chunks`,
`_lifetime_proxy_chunks` y `_lifetime_total_chunks`. Antes el contador vivía
solo en `pipeline_snapshot`; si dos chunks paralelos leían snapshots
desfasados o si el snapshot se reescribía completo (no via jsonb_set), el
cap consecutivo se podía burlar.

Después: `meal_plans.plan_data` es la fuente canónica, escrita vía
`update_plan_data_atomic` (SELECT … FOR UPDATE). El `pipeline_snapshot`
se mantiene como mirror para back-compat con callers legacy.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_3_proxy_counter_persistence.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cron_tasks import _read_proxy_counter


# ---------------------------------------------------------------------------
# 1. plan_data toma precedencia sobre snapshot
# ---------------------------------------------------------------------------
def test_plan_data_takes_precedence_over_snapshot():
    """Si ambos tienen el campo, plan_data gana."""
    plan_data = {"_consecutive_proxy_chunks": 5}
    snapshot = {"_consecutive_proxy_chunks": 1}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 5


def test_plan_data_zero_overrides_snapshot_nonzero():
    """plan_data=0 debe ganar sobre snapshot=N (caso reset post-strong-chunk)."""
    plan_data = {"_consecutive_proxy_chunks": 0}
    snapshot = {"_consecutive_proxy_chunks": 7}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 0, "plan_data=0 es valor explícito, no debe caer al snapshot"


# ---------------------------------------------------------------------------
# 2. Fallback a snapshot cuando plan_data no tiene el campo
# ---------------------------------------------------------------------------
def test_falls_back_to_snapshot_when_plan_data_missing_field():
    """plan_data sin el campo → cae al snapshot (back-compat)."""
    plan_data = {"days": [], "other_field": "x"}
    snapshot = {"_consecutive_proxy_chunks": 3}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 3


def test_falls_back_to_snapshot_when_plan_data_is_none():
    """plan_data=None → cae al snapshot."""
    snapshot = {"_consecutive_proxy_chunks": 2}

    result = _read_proxy_counter(None, snapshot, "_consecutive_proxy_chunks")

    assert result == 2


def test_falls_back_to_snapshot_when_plan_data_not_dict():
    """plan_data corrupto (string, lista, etc.) → cae al snapshot."""
    snapshot = {"_consecutive_proxy_chunks": 4}

    for corrupt in ["string", ["list"], 42, True]:
        result = _read_proxy_counter(corrupt, snapshot, "_consecutive_proxy_chunks")
        assert result == 4, f"plan_data={corrupt!r} debe caer al snapshot"


# ---------------------------------------------------------------------------
# 3. Defensa contra valores corruptos
# ---------------------------------------------------------------------------
def test_corrupt_plan_data_value_falls_back_to_snapshot():
    """plan_data tiene el campo pero con valor no-int → cae al snapshot."""
    plan_data = {"_consecutive_proxy_chunks": "bad-value"}
    snapshot = {"_consecutive_proxy_chunks": 3}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 3


def test_corrupt_snapshot_value_returns_zero():
    """Si plan_data y snapshot son ambos corruptos/missing, retorna 0."""
    plan_data = {"_consecutive_proxy_chunks": "bad"}
    snapshot = {"_consecutive_proxy_chunks": ["also-bad"]}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 0


# ---------------------------------------------------------------------------
# 4. Default 0 cuando ningún source existe
# ---------------------------------------------------------------------------
def test_returns_zero_when_neither_source_has_field():
    plan_data = {}
    snapshot = {}

    result = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")

    assert result == 0


def test_returns_zero_when_both_are_none():
    result = _read_proxy_counter(None, None, "_consecutive_proxy_chunks")

    assert result == 0


# ---------------------------------------------------------------------------
# 5. Funciona para los 3 contadores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", [
    "_consecutive_proxy_chunks",
    "_lifetime_proxy_chunks",
    "_lifetime_total_chunks",
])
def test_works_for_all_three_proxy_counter_fields(field):
    plan_data = {field: 9}
    snapshot = {field: 1}

    assert _read_proxy_counter(plan_data, snapshot, field) == 9
    assert _read_proxy_counter({}, {field: 1}, field) == 1
    assert _read_proxy_counter({}, {}, field) == 0


# ---------------------------------------------------------------------------
# 6. Coerción de tipos válidos
# ---------------------------------------------------------------------------
def test_accepts_float_in_plan_data_coerces_to_int():
    """JSONB puede deserializar 5.0 como float; el helper lo coerce."""
    plan_data = {"_consecutive_proxy_chunks": 5.0}

    assert _read_proxy_counter(plan_data, {}, "_consecutive_proxy_chunks") == 5


def test_accepts_string_int_in_snapshot():
    """Algunos paths persisten via str(int) en el snapshot; el helper lo coerce."""
    snapshot = {"_consecutive_proxy_chunks": "3"}

    assert _read_proxy_counter({}, snapshot, "_consecutive_proxy_chunks") == 3


# ---------------------------------------------------------------------------
# 7. Aislamiento por field: un campo corrupto no contamina otros
# ---------------------------------------------------------------------------
def test_corrupt_field_does_not_affect_others():
    plan_data = {
        "_consecutive_proxy_chunks": "broken",
        "_lifetime_proxy_chunks": 7,
        "_lifetime_total_chunks": 10,
    }
    snapshot = {
        "_consecutive_proxy_chunks": 2,
        "_lifetime_proxy_chunks": 0,
        "_lifetime_total_chunks": 0,
    }

    # consec corrupto en plan_data → snapshot=2
    assert _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks") == 2
    # lifetime sano → plan_data=7
    assert _read_proxy_counter(plan_data, snapshot, "_lifetime_proxy_chunks") == 7
    # total sano → plan_data=10
    assert _read_proxy_counter(plan_data, snapshot, "_lifetime_total_chunks") == 10
