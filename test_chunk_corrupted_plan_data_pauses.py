"""
P1-1: Test de la guardia anti-corrupción de plan_data.

Si un chunk N>3 dentro de un plan ≥15d encuentra plan_data sin las lecciones
previas esperadas (_recent_chunk_lessons truncado/vacío/corrompido), el worker
debe pausar el chunk con status='pending_user_action' y reason='missing_prior_lessons'
en vez de generar silenciosamente sin aprendizaje.

Cubre:
  1. plan_data sin _recent_chunk_lessons en chunk 5 de plan 30d → pausa.
  2. plan_data con _recent_chunk_lessons más corto del esperado → pausa.
  3. Chunks tempranos (≤3) NO se pausan aunque falten lecciones (seed normal).
  4. Planes <15d (7d) NO se pausan (la guardia solo aplica a planes largos).
  5. plan_data sano con suficientes lecciones → NO se pausa, ejecuta normal.

Ejecutar con:
    cd backend && python -m pytest test_chunk_corrupted_plan_data_pauses.py -v
"""
import json
import uuid
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, "_opened", False):
    connection_pool.open()


def _mock_pipeline_should_not_run(*args, **kwargs):
    """El pipeline NO debe ejecutarse si la guardia P1-1 funciona."""
    raise AssertionError(
        "[P1-1] run_plan_pipeline fue invocado pese a plan_data corrupto. La guardia falló."
    )


def _mock_pipeline_minimal(form_data, *args, **kwargs):
    """Mock benigno cuando se espera que el chunk sí ejecute."""
    offset = form_data.get("_days_offset", 0)
    count = form_data.get("_days_to_generate", 3)
    days = [
        {
            "day": offset + i + 1,
            "daily_summary": f"Day {offset + i + 1}",
            "meals": [{"name": f"Meal {offset + i + 1}", "type": "Almuerzo", "ingredients": ["pollo"]}],
        }
        for i in range(count)
    ]
    return {"days": days, "generation_status": "partial"}


@pytest.fixture
def fresh_plan():
    user_row = execute_sql_query("SELECT id FROM user_profiles LIMIT 1", fetch_one=True)
    if not user_row:
        pytest.skip("No user_profiles disponible para el test E2E.")
    user_id = user_row["id"]
    plan_id = str(uuid.uuid4())
    yield user_id, plan_id
    execute_sql_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,))
    execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))


def _seed_plan(user_id, plan_id, total_days, recent_lessons=None, days_generated=12):
    """Inserta un plan con plan_data específico."""
    plan_data = {
        "days": [{"day": d, "meals": []} for d in range(1, days_generated + 1)],
        "total_days_requested": total_days,
        "total_days_generated": days_generated,
        "generation_status": "partial",
    }
    if recent_lessons is not None:
        plan_data["_recent_chunk_lessons"] = recent_lessons
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(plan_data)),
    )


def _enqueue_and_force(user_id, plan_id, week_number, days_offset, total_days):
    snapshot = {
        "totalDays": total_days,
        "form_data": {
            "_days_offset": days_offset,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        },
    }
    _enqueue_plan_chunk(
        user_id, plan_id,
        week_number=week_number,
        days_offset=days_offset,
        days_count=3,
        pipeline_snapshot=snapshot,
    )
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' "
        "WHERE meal_plan_id = %s",
        (plan_id,),
    )


def _read_chunk_status(plan_id, week_number):
    return execute_sql_query(
        "SELECT status, pipeline_snapshot FROM plan_chunk_queue "
        "WHERE meal_plan_id = %s AND week_number = %s",
        (plan_id, week_number),
        fetch_one=True,
    )


# ---------------------------------------------------------------------------
# Test 1: plan_data SIN _recent_chunk_lessons → pausa
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_should_not_run)
def test_p11_pauses_when_recent_lessons_missing(_mock_pipe, fresh_plan):
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=30, recent_lessons=None, days_generated=12)
    _enqueue_and_force(user_id, plan_id, week_number=5, days_offset=12, total_days=30)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk_status(plan_id, week_number=5)
    assert row is not None, "El chunk 5 debe seguir en la cola tras la pausa."
    assert row["status"] == "pending_user_action", (
        f"Chunk con plan_data corrupto debe estar en pending_user_action, got {row['status']}"
    )
    snap = row["pipeline_snapshot"]
    if isinstance(snap, str):
        snap = json.loads(snap)
    assert snap.get("_pause_reason") == "missing_prior_lessons"
    assert snap.get("_p1_1_actual_lessons") == 0
    assert snap.get("_p1_1_expected_lessons") == 4  # min(5-1, 8) = 4


# ---------------------------------------------------------------------------
# Test 2: _recent_chunk_lessons más corto del esperado → pausa
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_should_not_run)
def test_p11_pauses_when_recent_lessons_truncated(_mock_pipe, fresh_plan):
    user_id, plan_id = fresh_plan
    truncated = [{"chunk": 1, "summary": "stub"}]  # solo 1, se esperan >=4
    _seed_plan(user_id, plan_id, total_days=30, recent_lessons=truncated, days_generated=12)
    _enqueue_and_force(user_id, plan_id, week_number=5, days_offset=12, total_days=30)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk_status(plan_id, week_number=5)
    assert row["status"] == "pending_user_action"
    snap = row["pipeline_snapshot"]
    if isinstance(snap, str):
        snap = json.loads(snap)
    assert snap.get("_pause_reason") == "missing_prior_lessons"
    assert snap.get("_p1_1_actual_lessons") == 1
    assert snap.get("_p1_1_expected_lessons") == 4


# ---------------------------------------------------------------------------
# Test 3: chunk N=3 NO se pausa (la guardia solo aplica a N>3)
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_minimal)
def test_p11_does_not_pause_early_chunks(_mock_pipe, fresh_plan):
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=30, recent_lessons=None, days_generated=6)
    _enqueue_and_force(user_id, plan_id, week_number=3, days_offset=6, total_days=30)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk_status(plan_id, week_number=3)
    snap = row["pipeline_snapshot"]
    if isinstance(snap, str):
        snap = json.loads(snap)
    assert snap.get("_pause_reason") != "missing_prior_lessons", (
        "Chunk N=3 NO debe gatillar la guardia P1-1."
    )


# ---------------------------------------------------------------------------
# Test 4: plan corto (7d) NO gatilla la guardia
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_minimal)
def test_p11_does_not_pause_short_plans(_mock_pipe, fresh_plan):
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=7, recent_lessons=None, days_generated=3)
    _enqueue_and_force(user_id, plan_id, week_number=2, days_offset=3, total_days=7)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk_status(plan_id, week_number=2)
    snap = row["pipeline_snapshot"]
    if isinstance(snap, str):
        snap = json.loads(snap)
    assert snap.get("_pause_reason") != "missing_prior_lessons", (
        "Plan de 7 días NO debe gatillar la guardia P1-1."
    )


# ---------------------------------------------------------------------------
# Test 5: plan_data sano con suficientes lecciones → NO pausa
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_minimal)
def test_p11_passes_when_lessons_sufficient(_mock_pipe, fresh_plan):
    user_id, plan_id = fresh_plan
    healthy_lessons = [{"chunk": i, "summary": f"lesson {i}"} for i in range(1, 5)]
    _seed_plan(
        user_id, plan_id, total_days=30,
        recent_lessons=healthy_lessons, days_generated=12,
    )
    _enqueue_and_force(user_id, plan_id, week_number=5, days_offset=12, total_days=30)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk_status(plan_id, week_number=5)
    snap = row["pipeline_snapshot"]
    if isinstance(snap, str):
        snap = json.loads(snap)
    assert snap.get("_pause_reason") != "missing_prior_lessons", (
        "Plan sano con 4 lecciones NO debe gatillar la guardia P1-1."
    )
