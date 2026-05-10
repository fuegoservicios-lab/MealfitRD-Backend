"""
P1-2: Test de telemetría de chunk_deferrals.

Cuando _check_chunk_learning_ready detecta que el chunk previo aún no terminó
(temporal_gate), debe:
  1. Devolver ready=False con reason='prev_chunk_day_not_yet_elapsed'.
  2. Insertar una fila en chunk_deferrals con reason='temporal_gate' y
     days_until_prev_end > 0.
  3. NO bloquear si la tabla falla (escritura best-effort en try/except).

Cubre:
  - Caso normal: temporal gate dispara → fila insertada.
  - Caso pasivo: chunk previo ya terminó → NO se inserta nada.
"""
import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import _check_chunk_learning_ready

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, "_opened", False):
    connection_pool.open()


@pytest.fixture
def fresh_plan():
    user_row = execute_sql_query("SELECT id FROM user_profiles LIMIT 1", fetch_one=True)
    if not user_row:
        pytest.skip("No user_profiles disponible.")
    user_id = str(user_row["id"])
    plan_id = str(uuid.uuid4())
    yield user_id, plan_id
    execute_sql_write(
        "DELETE FROM chunk_deferrals WHERE meal_plan_id = %s", (plan_id,)
    )
    execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))


def _seed_plan(user_id, plan_id, plan_start_dt):
    plan_data = {
        "days": [{"day": d, "meals": []} for d in range(1, 4)],
        "total_days_requested": 30,
        "total_days_generated": 3,
        "generation_status": "partial",
        "_plan_start_date": plan_start_dt.isoformat(),
    }
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(plan_data)),
    )
    return plan_data


def test_p1_2_temporal_gate_writes_chunk_deferrals_row(fresh_plan):
    user_id, plan_id = fresh_plan
    # plan_start_date en el FUTURO → el chunk previo aún no terminó.
    future_start = datetime.now(timezone.utc) + timedelta(days=10)
    plan_data = _seed_plan(user_id, plan_id, future_start)

    snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 3,
            "_plan_start_date": future_start.isoformat(),
            "householdSize": 1,
            "dietType": "Omnívora",
        },
    }

    rows_before = execute_sql_query(
        "SELECT COUNT(*)::int AS n FROM chunk_deferrals WHERE meal_plan_id = %s",
        (plan_id,), fetch_one=True,
    )
    n_before = rows_before["n"] if rows_before else 0

    result = _check_chunk_learning_ready(
        user_id=user_id,
        meal_plan_id=plan_id,
        week_number=2,
        days_offset=3,
        plan_data=plan_data,
        snapshot=snapshot,
    )

    assert result.get("ready") is False
    assert result.get("reason") == "prev_chunk_day_not_yet_elapsed"
    assert result.get("days_until_prev_end") is not None
    assert int(result["days_until_prev_end"]) > 0

    rows_after = execute_sql_query(
        "SELECT week_number, reason, days_until_prev_end "
        "FROM chunk_deferrals WHERE meal_plan_id = %s ORDER BY id DESC LIMIT 1",
        (plan_id,), fetch_one=True,
    )
    assert rows_after is not None, "Se esperaba una fila nueva en chunk_deferrals."
    assert rows_after["reason"] == "temporal_gate"
    assert rows_after["week_number"] == 2
    assert int(rows_after["days_until_prev_end"]) == int(result["days_until_prev_end"])

    count_after = execute_sql_query(
        "SELECT COUNT(*)::int AS n FROM chunk_deferrals WHERE meal_plan_id = %s",
        (plan_id,), fetch_one=True,
    )
    assert count_after["n"] == n_before + 1


def test_p1_2_no_telemetry_when_prev_chunk_already_elapsed(fresh_plan):
    user_id, plan_id = fresh_plan
    # plan_start_date en el PASADO lejano → chunk previo ya concluyó hace mucho.
    past_start = datetime.now(timezone.utc) - timedelta(days=60)
    plan_data = _seed_plan(user_id, plan_id, past_start)

    snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 3,
            "_plan_start_date": past_start.isoformat(),
            "householdSize": 1,
            "dietType": "Omnívora",
        },
    }

    _ = _check_chunk_learning_ready(
        user_id=user_id,
        meal_plan_id=plan_id,
        week_number=2,
        days_offset=3,
        plan_data=plan_data,
        snapshot=snapshot,
    )

    # Aunque el resultado pueda tener otras razones (zero_log, etc.) lo importante:
    # el temporal_gate NO debe haber disparado, así que NO hay fila con reason='temporal_gate'.
    rows = execute_sql_query(
        "SELECT COUNT(*)::int AS n FROM chunk_deferrals "
        "WHERE meal_plan_id = %s AND reason = 'temporal_gate'",
        (plan_id,), fetch_one=True,
    )
    assert rows["n"] == 0, (
        "No debe haber telemetría de temporal_gate cuando el chunk previo ya terminó."
    )
