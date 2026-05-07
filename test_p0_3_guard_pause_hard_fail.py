"""
Tests P0-3: Guard pause hard-fail cuando la pausa misma falla.

Escenario: chunk N>3 de plan ≥15d encuentra _recent_chunk_lessons corrupto.
La auto-recovery también falla. La guardia intenta pausar el chunk con
status='pending_user_action', pero el UPDATE falla (e.g., DB blip, snap
no serializable, schema corrupto).

Antes: la excepción se propagaba al outer catch → chunk marcado 'failed'
con backoff retry, reintentando contra plan_data corrupto hasta agotar
attempts en dead_letter — sin señal clara al operador.

Ahora: hard-fail explícito con dead_letter_reason='guard_pause_failed:...'
para inspección manual. NO retry.

Cubre:
  1. Pausa falla con DB exception → chunk se marca 'failed' con dead_letter_reason.
  2. Pausa falla y hard-fail también falla → log error, no crashea.
  3. Pausa exitosa (caso de control) → status='pending_user_action' (P1-1 existing).
"""
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, "_opened", False):
    connection_pool.open()


def _mock_pipeline_should_not_run(*args, **kwargs):
    """El pipeline NO debe ejecutarse: la guardia P0-3 debe hacer hard-fail antes."""
    raise AssertionError(
        "[P0-3] run_plan_pipeline fue invocado pese a fallo de guard_pause. "
        "El hard-fail debió bloquear el pipeline."
    )


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


def _seed_plan(user_id, plan_id, total_days=30, days_generated=12):
    plan_data = {
        "days": [{"day": d, "meals": []} for d in range(1, days_generated + 1)],
        "total_days_requested": total_days,
        "total_days_generated": days_generated,
        "generation_status": "partial",
        # _recent_chunk_lessons OMITIDO a propósito — gatilla guardia P1-1
    }
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(plan_data)),
    )


def _enqueue_chunk(user_id, plan_id, week_number=5, days_offset=12, total_days=30):
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
        week_number=week_number, days_offset=days_offset, days_count=3,
        pipeline_snapshot=snapshot,
    )
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' "
        "WHERE meal_plan_id = %s",
        (plan_id,),
    )


def _read_chunk(plan_id, week_number):
    return execute_sql_query(
        "SELECT status, dead_letter_reason, dead_lettered_at FROM plan_chunk_queue "
        "WHERE meal_plan_id = %s AND week_number = %s",
        (plan_id, week_number),
        fetch_one=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Pausa falla → chunk se marca 'failed' con dead_letter_reason
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_should_not_run)
@patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[])
def test_p0_3_pause_failure_triggers_hard_fail(_mock_rebuild, _mock_pipe, fresh_plan):
    """
    Si el UPDATE de pausa falla (DB blip), el chunk debe marcarse 'failed'
    con dead_letter_reason='guard_pause_failed:missing_prior_lessons:...'
    en lugar de propagar la excepción al outer catch standard.
    """
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=30, days_generated=12)
    _enqueue_chunk(user_id, plan_id, week_number=5, days_offset=12)

    real_execute_sql_write = execute_sql_write
    pause_attempts = []

    def selective_failure(query, params=None, **kwargs):
        """Falla SOLO en el UPDATE de pausa (status='pending_user_action')."""
        if query and "status = 'pending_user_action'" in query:
            pause_attempts.append(query)
            raise Exception("Simulated DB blip during pause UPDATE")
        return real_execute_sql_write(query, params, **kwargs)

    with patch("cron_tasks.execute_sql_write", side_effect=selective_failure):
        process_plan_chunk_queue(target_plan_id=plan_id)

    # La guardia debió haber intentado la pausa al menos una vez
    assert len(pause_attempts) >= 1, (
        f"Esperaba al menos 1 intento de pausa, hubo {len(pause_attempts)}"
    )

    row = _read_chunk(plan_id, week_number=5)
    assert row is not None
    assert row["status"] == "failed", (
        f"Esperaba status='failed', got {row['status']!r}"
    )
    dead_reason = row.get("dead_letter_reason") or ""
    assert "guard_pause_failed" in dead_reason, (
        f"Esperaba dead_letter_reason con 'guard_pause_failed', got {dead_reason!r}"
    )
    assert "missing_prior_lessons" in dead_reason, (
        f"Esperaba dead_letter_reason con 'missing_prior_lessons', got {dead_reason!r}"
    )
    assert row.get("dead_lettered_at") is not None, (
        "dead_lettered_at debe persistirse cuando hard-fail tuvo éxito"
    )


# ---------------------------------------------------------------------------
# Test 2: Si pausa Y hard-fail fallan → log error pero no crashea el worker
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_should_not_run)
@patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[])
def test_p0_3_pause_and_hard_fail_both_failing(_mock_rebuild, _mock_pipe, fresh_plan):
    """
    Si tanto la pausa COMO el hard-fail fallan (DB completamente caída),
    el worker no debe crashear. El chunk caerá al outer catch retry standard
    como último recurso, pero el operador verá log CRITICAL en ambos puntos.
    """
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=30, days_generated=12)
    _enqueue_chunk(user_id, plan_id, week_number=5, days_offset=12)

    real_execute_sql_write = execute_sql_write

    def fail_pause_and_hardfail(query, params=None, **kwargs):
        if query and (
            "status = 'pending_user_action'" in query
            or "dead_letter_reason" in query
        ):
            raise Exception("DB completamente caída")
        return real_execute_sql_write(query, params, **kwargs)

    with patch("cron_tasks.execute_sql_write", side_effect=fail_pause_and_hardfail):
        # No debe crashear pese a doble fallo.
        process_plan_chunk_queue(target_plan_id=plan_id)

    # El chunk no quedó como 'completed'; cualquier estado terminal o 'pending'
    # con backoff retry es aceptable como último recurso.
    row = _read_chunk(plan_id, week_number=5)
    assert row is not None
    assert row["status"] != "completed", (
        f"Chunk con plan_data corrupto NUNCA debe completarse. status={row['status']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 (control): Pausa exitosa → status='pending_user_action' (regresión P1-1)
# ---------------------------------------------------------------------------
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_pipeline_should_not_run)
@patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[])
def test_p0_3_pause_success_keeps_legacy_behavior(_mock_rebuild, _mock_pipe, fresh_plan):
    """
    Caso de control: si la pausa funciona (sin mockear failures), el chunk
    queda en pending_user_action como antes (regresión del flujo P1-1).
    """
    user_id, plan_id = fresh_plan
    _seed_plan(user_id, plan_id, total_days=30, days_generated=12)
    _enqueue_chunk(user_id, plan_id, week_number=5, days_offset=12)

    process_plan_chunk_queue(target_plan_id=plan_id)

    row = _read_chunk(plan_id, week_number=5)
    assert row["status"] == "pending_user_action", (
        f"Esperaba pending_user_action en flujo normal, got {row['status']!r}"
    )
    # No debe haber dead_letter_reason si la pausa fue exitosa
    assert not (row.get("dead_letter_reason") or "").startswith("guard_pause_failed"), (
        "Pausa exitosa NO debe poblar dead_letter_reason de guard_pause_failed."
    )
