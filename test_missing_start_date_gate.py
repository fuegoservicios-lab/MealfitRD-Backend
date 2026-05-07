from unittest.mock import patch

import pytest
from datetime import datetime, timezone, timedelta
from cron_tasks import _check_chunk_learning_ready

@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_missing_start_date_recovers_and_enforces_gate(mock_execute_sql, mock_write, mock_deferral):
    """
    Test P0-3 fix: Ensures that if _plan_start_date is missing from a chunk snapshot,
    the temporal gate does not silently bypass. Instead, it recovers the date from
    meal_plans and correctly defers the chunk if the prior chunk hasn't elapsed.
    """
    user_id = "test_user_p0_3"
    meal_plan_id = "plan-missing-date"
    week_number = 2
    days_offset = 3

    today = datetime.now(timezone.utc)
    # Simulate a plan that started TODAY. So a chunk at days_offset=3 (day 4-6) is NOT ready.
    plan_start_date = today.isoformat()

    plan_data = {
        "days": [
            {"day": 1, "meals": [{"name": "M1"}]},
            {"day": 2, "meals": [{"name": "M2"}]},
            {"day": 3, "meals": [{"name": "M3"}]}
        ],
    }

    # Snapshot is MISSING _plan_start_date
    snapshot = {
        "form_data": {
            "totalDays": 3,
            # "_plan_start_date" is MISSING!
        }
    }

    # Mock execute_sql_query to return the fallback grocery_start_date
    mock_execute_sql.return_value = {"gsd": plan_start_date}

    # Execute
    res = _check_chunk_learning_ready(
        user_id, meal_plan_id, week_number, days_offset, plan_data, snapshot
    )

    # Validate it didn't silently bypass
    assert res.get("ready") is False, "Should enforce temporal gate, not silently bypass"
    assert res.get("reason") == "prev_chunk_day_not_yet_elapsed", (
        f"Should fail because it's too early; got reason={res.get('reason')!r}"
    )

    # Verify the fallback query was made (mock puede recibir más calls para tz_offset
    # vivo del user_profile; lo relevante es que la SELECT a meal_plans haya ocurrido).
    _meal_plans_calls = [
        c for c in mock_execute_sql.call_args_list
        if c.args and "FROM meal_plans" in c.args[0] and "grocery_start_date" in c.args[0]
    ]
    assert len(_meal_plans_calls) == 1, (
        f"Esperaba 1 SELECT a meal_plans para grocery_start_date, hubo {len(_meal_plans_calls)}"
    )

    # [P0-2 v2] Telemetría debe registrar el fallback exitoso desde grocery_start_date
    # para que dashboards puedan alertar si >5% de chunks dependen de este camino.
    _gsd_calls = [
        c for c in mock_deferral.call_args_list
        if c.kwargs.get("reason") == "start_date_fallback:grocery_start_date"
    ]
    assert len(_gsd_calls) == 1, (
        f"Esperaba 1 telemetría fallback:grocery_start_date, hubo "
        f"{len(_gsd_calls)}. Calls: {mock_deferral.call_args_list}"
    )


@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_missing_start_date_no_anchor_blocks_with_pause(mock_execute_sql, mock_write, mock_deferral):
    """
    Test P0-2 v2: Cuando NI _plan_start_date, NI grocery_start_date, NI created_at
    están disponibles, el gate NO debe fabricar una fecha con NOW() (eso rompía el
    temporal gate y dejaba el plan atascado). En su lugar debe retornar
    ready=False con reason='missing_start_date_no_anchor' y emitir telemetría.
    """
    user_id = "test_user_p0_2"
    meal_plan_id = "plan-no-anchor"
    week_number = 2
    days_offset = 3

    plan_data = {
        "days": [
            {"day": 1, "meals": [{"name": "M1"}]},
            {"day": 2, "meals": [{"name": "M2"}]},
            {"day": 3, "meals": [{"name": "M3"}]}
        ],
    }
    snapshot = {"form_data": {"totalDays": 3}}  # falta _plan_start_date

    # Sin ancla en meal_plans: gsd=None y created_at=None
    mock_execute_sql.return_value = {"gsd": None, "created_at": None}

    res = _check_chunk_learning_ready(
        user_id, meal_plan_id, week_number, days_offset, plan_data, snapshot
    )

    assert res.get("ready") is False, "Sin ancla, el gate debe bloquear"
    assert res.get("reason") == "missing_start_date_no_anchor", (
        f"Esperaba reason='missing_start_date_no_anchor', recibí {res.get('reason')!r}"
    )
    assert res.get("_fallback_source") == "no_anchor"

    # Telemetría: una sola entrada con reason 'start_date_fallback:no_anchor'
    _no_anchor_calls = [
        c for c in mock_deferral.call_args_list
        if c.kwargs.get("reason") == "start_date_fallback:no_anchor"
    ]
    assert len(_no_anchor_calls) == 1, (
        f"Esperaba 1 telemetría fallback:no_anchor, hubo {len(_no_anchor_calls)}"
    )

    # Crítico: no debe haberse persistido NOW() como grocery_start_date.
    # Verificamos que ningún UPDATE a meal_plans haya sido emitido.
    _meal_plans_writes = [
        c for c in mock_write.call_args_list
        if "UPDATE meal_plans" in (c.args[0] if c.args else "")
    ]
    assert len(_meal_plans_writes) == 0, (
        f"NO debe persistirse fecha sintética en meal_plans cuando no hay ancla. "
        f"UPDATEs detectados: {_meal_plans_writes}"
    )
