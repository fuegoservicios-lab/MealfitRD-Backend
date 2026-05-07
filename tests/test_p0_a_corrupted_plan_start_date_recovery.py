"""[P0-A] Recovery cuando `_plan_start_date` está corrupto pero no vacío.

Antes de este fix, `_check_chunk_learning_ready` solo verificaba truthy:

    plan_start_date_str = form_data.get("_plan_start_date")
    if not plan_start_date_str:        # solo None / "" / 0
        ...cascada de recovery...

Cualquier string corrupto pero truthy (gibberish "abc", fecha inválida
"2025-13-45", año absurdo "2099-12-31") saltaba el cascade y aterrizaba en
`safe_fromisoformat` aguas abajo, causando uno de dos modos de fallo:

  1. ValueError no atrapado → worker muere → retry infinito hasta dead-letter.
  2. Parse exitoso a fecha imposible (futuro lejano) → temporal gate diferia
     hasta agotar `CHUNK_TEMPORAL_GATE_MAX_RETRIES` (~5 min de atasco visible
     al usuario como "tu plan está pendiente de generarse").

Este test suite verifica que ahora:
  - Strings corruptos disparan el mismo cascade que valores ausentes.
  - El gate emite telemetría dedicada (`corrupted_plan_start_date:<reason>`).
  - El valor recuperado se persiste back to meal_plans.plan_data para que la
    corrupción no se reintroduzca en futuros chunks.
  - Si el cascade tampoco resuelve un valor parseable, el chunk pausa con
    reason='unrecoverable_corrupted_date' en lugar de crashear.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest


# ============================================================================
# UNIT TESTS — validate_plan_start_date()
# ============================================================================

def test_validator_accepts_valid_iso_date():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("2026-04-15T00:00:00+00:00")
    assert reason is None
    assert dt is not None
    assert dt.year == 2026 and dt.month == 4 and dt.day == 15


def test_validator_accepts_valid_iso_date_naive_promoted_to_utc():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("2026-04-15")
    assert reason is None
    assert dt.tzinfo is not None  # se promueve a UTC


def test_validator_rejects_none():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date(None)
    assert dt is None
    assert reason == "empty"


def test_validator_rejects_empty_string():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("")
    assert dt is None
    assert reason == "empty"


def test_validator_rejects_whitespace_only():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("   ")
    assert dt is None
    assert reason == "empty"


def test_validator_rejects_non_string():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date(12345)
    assert dt is None
    assert reason.startswith("wrong_type:")


def test_validator_rejects_gibberish():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("not-a-date-at-all")
    assert dt is None
    assert reason.startswith("unparseable:")


def test_validator_rejects_invalid_month_day():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("2025-13-45")
    assert dt is None
    assert reason.startswith("unparseable:")


def test_validator_rejects_serialized_json_garbage():
    """Snapshot rebuild puede meter "{}" o "[]" si el merge de form_data falla."""
    from constants import validate_plan_start_date

    for junk in ("{}", "[]", "null", "{'key': 'value'}"):
        dt, reason = validate_plan_start_date(junk)
        assert dt is None, f"junk={junk!r} no debería parsear"
        assert reason.startswith("unparseable:"), f"junk={junk!r} reason={reason!r}"


def test_validator_rejects_year_too_far_in_past():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("1900-01-01")
    assert dt is None
    assert reason == "out_of_bounds:past"


def test_validator_rejects_year_just_below_minimum():
    from constants import validate_plan_start_date, PLAN_START_DATE_MIN_YEAR

    dt, reason = validate_plan_start_date(f"{PLAN_START_DATE_MIN_YEAR - 1}-12-31")
    assert dt is None
    assert reason == "out_of_bounds:past"


def test_validator_rejects_year_far_in_future():
    from constants import validate_plan_start_date

    dt, reason = validate_plan_start_date("2099-12-31")
    assert dt is None
    assert reason == "out_of_bounds:future"


def test_validator_accepts_date_within_future_window():
    """Un plan recién renovado podría tener _plan_start_date hasta ~30d adelante."""
    from constants import validate_plan_start_date

    soon = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
    dt, reason = validate_plan_start_date(soon)
    assert reason is None
    assert dt is not None


def test_validator_uses_injected_now_for_determinism():
    from constants import validate_plan_start_date

    fixed_now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    # 60 días después de fixed_now → válido (dentro de 90d)
    dt, reason = validate_plan_start_date("2026-06-30T00:00:00+00:00", now=fixed_now)
    assert reason is None
    # 100 días después de fixed_now → out_of_bounds:future
    dt2, reason2 = validate_plan_start_date("2026-08-09T00:00:00+00:00", now=fixed_now)
    assert reason2 == "out_of_bounds:future"


# ============================================================================
# INTEGRATION TESTS — _check_chunk_learning_ready con valores corruptos
# ============================================================================

def _plan_data_with_prior_chunk():
    """Plan_data mínimo con días del chunk previo (week 1) presentes."""
    return {
        "days": [
            {"day": 1, "meals": [{"name": "M1"}]},
            {"day": 2, "meals": [{"name": "M2"}]},
            {"day": 3, "meals": [{"name": "M3"}]},
        ],
    }


# [test fix] Algunos test files instalan un stub `db_inventory.get_inventory_activity_since`
# en `sys.modules` que retorna un list `[]` (ver `tests/test_p0_a_synthesized_telemetry.py`).
# Cuando esos files corren primero en la misma sesión pytest, el stub queda activo y
# `cron_tasks.py:11408` (`activity.get(...)`) cruje con AttributeError. Para que estos
# tests sean order-independent, parcheamos el binding ya importado en `cron_tasks`
# directamente. La firma real (db_inventory.py:810) retorna un Dict.
_INVENTORY_ACTIVITY_STUB = {
    "mutations_count": 0,
    "last_mutation_at": None,
    "low_stock_items": 0,
    "consumption_mutations_count": 0,
    "manual_mutations_count": 0,
}


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_corrupt_gibberish_falls_back_to_grocery_start_date(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """Gibberish en snapshot → cascade lee grocery_start_date desde meal_plans."""
    from cron_tasks import _check_chunk_learning_ready

    user_id = "u-corrupt-1"
    plan_id = "p-corrupt-1"
    today = datetime.now(timezone.utc)
    valid_recovery_date = today.isoformat()

    # Cualquier SELECT a meal_plans devuelve grocery_start_date válido.
    # Otras queries (user_profiles, plan_chunk_queue, etc.) → None.
    def _sql_router(query, params=None, fetch_one=False):
        if "FROM meal_plans" in query and "grocery_start_date" in query:
            return {"gsd": valid_recovery_date, "created_at": None}
        return None

    mock_execute_sql.side_effect = _sql_router

    snapshot = {
        "form_data": {
            "totalDays": 3,
            "_plan_start_date": "this-is-not-a-date",  # corrupto pero truthy
        }
    }

    res = _check_chunk_learning_ready(
        user_id, plan_id, week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    # El gate devuelve "no listo" por temporal (chunk previo termina hoy),
    # NO por corrupción no recuperada.
    assert res.get("ready") is False
    assert res.get("reason") != "unrecoverable_corrupted_date"

    # Debe haber registrado telemetría de corrupción.
    corrupt_calls = [
        c for c in mock_deferral.call_args_list
        if str(c.kwargs.get("reason", "")).startswith("corrupted_plan_start_date:")
    ]
    assert len(corrupt_calls) == 1, (
        f"Esperaba 1 telemetría corrupted_plan_start_date, hubo "
        f"{len(corrupt_calls)}: {mock_deferral.call_args_list}"
    )
    # El sub-reason debe identificar el modo de corrupción (unparseable).
    assert "unparseable" in corrupt_calls[0].kwargs["reason"]


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_corrupt_far_future_year_no_longer_freezes_gate(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """`"2099-12-31"` parsea pero out_of_bounds:future → cascade en lugar de
    diferirse hasta agotar P1-C max retries."""
    from cron_tasks import _check_chunk_learning_ready

    today = datetime.now(timezone.utc)
    # Plan empezó hace una semana → chunk 2 debería estar listo.
    legitimate_start = (today - timedelta(days=7)).isoformat()

    def _sql_router(query, params=None, fetch_one=False):
        if "FROM meal_plans" in query and "grocery_start_date" in query:
            return {"gsd": legitimate_start, "created_at": None}
        return None

    mock_execute_sql.side_effect = _sql_router

    snapshot = {
        "form_data": {
            "totalDays": 7,
            "_plan_start_date": "2099-12-31T00:00:00+00:00",  # parseable pero absurdo
        }
    }

    res = _check_chunk_learning_ready(
        "u-future", "p-future", week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    # Telemetría de out_of_bounds:future
    corrupt_calls = [
        c for c in mock_deferral.call_args_list
        if str(c.kwargs.get("reason", "")).startswith("corrupted_plan_start_date:")
    ]
    assert len(corrupt_calls) == 1
    assert "out_of_bounds:future" in corrupt_calls[0].kwargs["reason"]

    # El chunk previo (días 1-3, plan empezó hace 7 días) ya transcurrió:
    # el gate NO debe diferirse — debe estar ready=True o caer a no-anchor logic.
    # No debe estar pendiente por el motivo "prev_chunk_day_not_yet_elapsed"
    # (que sería el resultado si la fecha 2099 hubiera pasado al cálculo).
    assert res.get("reason") != "prev_chunk_day_not_yet_elapsed"


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_corrupt_no_recovery_anchors_returns_no_anchor(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """Si snapshot corrupto Y meal_plans no tiene gsd ni created_at →
    bloquea con missing_start_date_no_anchor (NO crashea)."""
    from cron_tasks import _check_chunk_learning_ready

    def _sql_router(query, params=None, fetch_one=False):
        if "FROM meal_plans" in query and "grocery_start_date" in query:
            return {"gsd": None, "created_at": None}
        return None

    mock_execute_sql.side_effect = _sql_router

    snapshot = {
        "form_data": {
            "totalDays": 3,
            "_plan_start_date": "2025-13-45",  # mes/día imposibles
        }
    }

    res = _check_chunk_learning_ready(
        "u-no-anchor", "p-no-anchor", week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    assert res.get("ready") is False
    assert res.get("reason") == "missing_start_date_no_anchor"

    # Dos entradas de telemetría: corruption + no_anchor.
    reasons = [c.kwargs.get("reason") for c in mock_deferral.call_args_list]
    assert any(str(r).startswith("corrupted_plan_start_date:") for r in reasons)
    assert "start_date_fallback:no_anchor" in reasons


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_corruption_heals_meal_plans_plan_data(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """Cuando el cascade recupera un valor válido tras detectar corrupción,
    debe persistirse de regreso en meal_plans.plan_data._plan_start_date para
    que rebuilds futuros del snapshot no reintroduzcan la basura."""
    from cron_tasks import _check_chunk_learning_ready

    plan_id = "p-heal-1"
    today = datetime.now(timezone.utc)
    valid_recovery_date = today.isoformat()

    def _sql_router(query, params=None, fetch_one=False):
        if "FROM meal_plans" in query and "grocery_start_date" in query:
            return {"gsd": valid_recovery_date, "created_at": None}
        return None

    mock_execute_sql.side_effect = _sql_router

    snapshot = {
        "form_data": {
            "totalDays": 3,
            "_plan_start_date": "garbage-string",
        }
    }

    _check_chunk_learning_ready(
        "u-heal", plan_id, week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    # Debe haberse llamado UPDATE meal_plans SET plan_data ... _plan_start_date
    heal_calls = [
        c for c in mock_write.call_args_list
        if c.args and "UPDATE meal_plans" in c.args[0]
        and "_plan_start_date" in c.args[0]
    ]
    assert len(heal_calls) >= 1, (
        f"Esperaba al menos 1 UPDATE meal_plans para curar _plan_start_date, "
        f"hubo {len(heal_calls)}. Calls: "
        f"{[c.args[0][:120] for c in mock_write.call_args_list]}"
    )
    # El parámetro de la fecha debe ser el valor recuperado, no la basura.
    args = heal_calls[0].args
    assert valid_recovery_date in args[1] or any(
        valid_recovery_date == p for p in args[1]
    )


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_valid_date_skips_corruption_telemetry_and_heal(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """Cuando _plan_start_date es válido, NO se registra telemetría de
    corrupción y NO se ejecuta el heal a meal_plans (path no-op para el
    99% de casos sanos)."""
    from cron_tasks import _check_chunk_learning_ready

    today = datetime.now(timezone.utc)
    valid_iso = (today - timedelta(days=7)).isoformat()

    mock_execute_sql.return_value = None

    snapshot = {
        "form_data": {
            "totalDays": 7,
            "_plan_start_date": valid_iso,
        }
    }

    _check_chunk_learning_ready(
        "u-clean", "p-clean", week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    # Cero entradas de corruption telemetry
    corrupt_calls = [
        c for c in mock_deferral.call_args_list
        if str(c.kwargs.get("reason", "")).startswith("corrupted_plan_start_date:")
    ]
    assert len(corrupt_calls) == 0

    # Cero UPDATE meal_plans para _plan_start_date heal
    heal_calls = [
        c for c in mock_write.call_args_list
        if c.args and "UPDATE meal_plans" in c.args[0]
        and "_plan_start_date" in c.args[0]
    ]
    assert len(heal_calls) == 0


@patch("cron_tasks.get_inventory_activity_since", return_value=_INVENTORY_ACTIVITY_STUB)
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_unrecoverable_corruption_returns_pause_signal(
    mock_execute_sql, mock_write, mock_deferral, mock_push,
    _mock_consumed, _mock_activity,
):
    """Defensa en profundidad: si el cascade recupera un valor que TAMBIÉN
    falla parsing (e.g., grocery_start_date corrupto en plan_data), el gate
    debe devolver ready=False con reason='unrecoverable_corrupted_date' en
    lugar de propagar ValueError al worker."""
    from cron_tasks import _check_chunk_learning_ready

    def _sql_router(query, params=None, fetch_one=False):
        if "FROM meal_plans" in query and "grocery_start_date" in query:
            # gsd está presente pero ES basura. El cascade lo "recupera" como
            # plan_start_date_str y luego safe_fromisoformat falla aguas abajo.
            return {"gsd": "still-not-a-date", "created_at": None}
        return None

    mock_execute_sql.side_effect = _sql_router

    snapshot = {
        "form_data": {
            "totalDays": 3,
            "_plan_start_date": "first-garbage",
        }
    }

    res = _check_chunk_learning_ready(
        "u-unrecoverable", "p-unrecoverable", week_number=2, days_offset=3,
        plan_data=_plan_data_with_prior_chunk(),
        snapshot=snapshot,
    )

    assert res.get("ready") is False
    assert res.get("reason") == "unrecoverable_corrupted_date"

    # Telemetría: una para detectar la corrupción inicial, otra unrecoverable
    reasons = [c.kwargs.get("reason") for c in mock_deferral.call_args_list]
    assert any(str(r).startswith("corrupted_plan_start_date:") for r in reasons)
    assert "unrecoverable_corrupted_date" in reasons
