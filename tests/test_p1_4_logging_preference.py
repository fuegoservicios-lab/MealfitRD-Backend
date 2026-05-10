"""
P1-4: Test de la preferencia de logging por usuario.

Cubre:
  1. Default: usuarios sin la columna seteada caen en 'manual' y siguen pausando
     por learning_proxy_exhausted / chronic_zero_logging.
  2. 'auto_proxy': el usuario opta por confiar en el plan. _check_chunk_learning_ready
     NO devuelve learning_proxy_exhausted aunque _consecutive_proxy_chunks supere el cap;
     en su lugar marca inventory_proxy_used=True y signal=weak.
  3. 'auto_proxy': lo mismo pero ante chronic_zero_logging (lifetime ratio alto).
"""
import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import _check_chunk_learning_ready

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, "_opened", False):
    connection_pool.open()


@pytest.fixture
def fresh_user_and_plan():
    user_row = execute_sql_query("SELECT id FROM user_profiles LIMIT 1", fetch_one=True)
    if not user_row:
        pytest.skip("No user_profiles disponible.")
    user_id = str(user_row["id"])

    # Snapshot original preference
    orig = execute_sql_query(
        "SELECT logging_preference FROM user_profiles WHERE id = %s",
        (user_id,), fetch_one=True,
    )
    orig_pref = (orig or {}).get("logging_preference") or "manual"

    plan_id = str(uuid.uuid4())
    past_start = datetime.now(timezone.utc) - timedelta(days=30)
    # Sembramos comidas reales para que planned_total > 0 y se active zero_log_proxy.
    plan_data = {
        "days": [
            {
                "day": d,
                "meals": [
                    {"name": f"Desayuno día {d}", "type": "Desayuno", "ingredients": ["pollo"]},
                    {"name": f"Almuerzo día {d}", "type": "Almuerzo", "ingredients": ["res"]},
                ],
            }
            for d in range(1, 13)
        ],
        "total_days_requested": 30,
        "total_days_generated": 12,
        "generation_status": "partial",
        "_plan_start_date": past_start.isoformat(),
    }
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(plan_data)),
    )
    yield user_id, plan_id, plan_data, past_start
    execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))
    execute_sql_write(
        "UPDATE user_profiles SET logging_preference = %s WHERE id = %s",
        (orig_pref, user_id),
    )


def _make_snapshot(plan_start_dt, consecutive_proxy=0, lifetime_proxy=0, lifetime_total=0):
    return {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 9,
            "_plan_start_date": plan_start_dt.isoformat(),
            "current_pantry_ingredients": [],
            "householdSize": 1,
            "dietType": "Omnívora",
        },
        "_consecutive_proxy_chunks": consecutive_proxy,
        "_lifetime_proxy_chunks": lifetime_proxy,
        "_lifetime_total_chunks": lifetime_total,
    }


def _stub_inventory_activity(*_args, **_kwargs):
    # Simula >=2 mutaciones de inventario (CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS),
    # condición necesaria para que el código entre al branch de cap consecutive.
    return {"consumption_mutations_count": 5, "mutations_count": 5}


def _stub_no_consumed(*_args, **_kwargs):
    # Sin logs explícitos → activa zero_log_proxy.
    return []


@patch("cron_tasks.get_inventory_activity_since", side_effect=_stub_inventory_activity)
@patch("cron_tasks.get_consumed_meals_since", side_effect=_stub_no_consumed)
def test_p1_4_manual_user_pauses_on_consecutive_proxy(_c, _i, fresh_user_and_plan):
    user_id, plan_id, plan_data, past_start = fresh_user_and_plan
    execute_sql_write(
        "UPDATE user_profiles SET logging_preference = 'manual' WHERE id = %s",
        (user_id,),
    )
    from constants import CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING
    snapshot = _make_snapshot(past_start, consecutive_proxy=CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING)

    result = _check_chunk_learning_ready(
        user_id=user_id, meal_plan_id=plan_id, week_number=4,
        days_offset=9, plan_data=plan_data, snapshot=snapshot,
    )
    assert result.get("ready") is False
    assert result.get("reason") == "learning_proxy_exhausted"


@patch("cron_tasks.get_inventory_activity_since", side_effect=_stub_inventory_activity)
@patch("cron_tasks.get_consumed_meals_since", side_effect=_stub_no_consumed)
def test_p1_4_auto_proxy_skips_consecutive_proxy_pause(_c, _i, fresh_user_and_plan):
    user_id, plan_id, plan_data, past_start = fresh_user_and_plan
    execute_sql_write(
        "UPDATE user_profiles SET logging_preference = 'auto_proxy' WHERE id = %s",
        (user_id,),
    )
    from constants import CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING
    snapshot = _make_snapshot(past_start, consecutive_proxy=CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING + 2)

    result = _check_chunk_learning_ready(
        user_id=user_id, meal_plan_id=plan_id, week_number=4,
        days_offset=9, plan_data=plan_data, snapshot=snapshot,
    )
    assert result.get("reason") != "learning_proxy_exhausted", (
        "auto_proxy debe omitir la pausa por consecutive proxy."
    )
    assert result.get("inventory_proxy_used") is True
    assert result.get("learning_signal_strength") == "weak"
    assert result.get("ready") is True


@patch("cron_tasks.get_inventory_activity_since", side_effect=_stub_inventory_activity)
@patch("cron_tasks.get_consumed_meals_since", side_effect=_stub_no_consumed)
def test_p1_4_auto_proxy_skips_chronic_zero_logging_pause(_c, _i, fresh_user_and_plan):
    user_id, plan_id, plan_data, past_start = fresh_user_and_plan
    execute_sql_write(
        "UPDATE user_profiles SET logging_preference = 'auto_proxy' WHERE id = %s",
        (user_id,),
    )
    from constants import CHUNK_LIFETIME_PROXY_MIN_TOTAL, CHUNK_MAX_LIFETIME_PROXY_RATIO
    # Construye condiciones para que NO entre por consecutive (consec=0) pero SÍ por chronic.
    _total = max(CHUNK_LIFETIME_PROXY_MIN_TOTAL, 4)
    _proxy = int(_total * CHUNK_MAX_LIFETIME_PROXY_RATIO) + 1
    snapshot = _make_snapshot(
        past_start, consecutive_proxy=0,
        lifetime_proxy=_proxy, lifetime_total=_total,
    )

    result = _check_chunk_learning_ready(
        user_id=user_id, meal_plan_id=plan_id, week_number=4,
        days_offset=9, plan_data=plan_data, snapshot=snapshot,
    )
    assert result.get("reason") != "chronic_zero_logging", (
        "auto_proxy debe omitir la pausa por chronic_zero_logging."
    )
    assert result.get("inventory_proxy_used") is True
    assert result.get("ready") is True
