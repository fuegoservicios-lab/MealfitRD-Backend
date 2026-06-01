"""[P0-3] Tests para la pausa y recovery de chunks bloqueados por chunk previo no concluido.

Cuando el chunk N+1 quiere generarse pero el calendario del usuario aún muestra que el
chunk N no terminó (TZ skew, _plan_start_date corrupto, CHUNK_PROACTIVE_MARGIN_DAYS):

1. Tras `CHUNK_LEARNING_READY_MAX_DEFERRALS` deferrals, el chunk pasa a
   `pending_user_action` con `_pantry_pause_reason = "prev_chunk_not_concluded"`.
   Antes (P0-1/LEARNING-FORCED) forzaba la generación con _force_variety, generando
   platos sobre días aún en consumo.

2. `_recover_pantry_paused_chunks` debe:
   - Reanudar como pending si el gate temporal ahora pasa (día previo concluyó).
   - Escalar a flexible_mode tras `CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS` sin resolución
     (mejor un plan en flexible que congelado).

Tests basados en mocks (patrón seguido por test_p1_2_anchor_recovery_escalation):
los tests de DB real chocarían con un bug pre-existente en
`_recover_pantry_paused_chunks` que llama `execute_sql_query` sin `fetch_all=True`
y siempre recibe [] en producción. Ese bug es de scope separado.
"""
from unittest.mock import patch

import pytest


def _paused_row(reason: str = "prev_chunk_not_concluded", paused_seconds: int = 600,
                ttl_hours: int = 24, meal_plan_id="plan-A", chunk_id="chunk-1",
                week_number: int = 2):
    return {
        "id": chunk_id,
        "user_id": "user-A",
        "meal_plan_id": meal_plan_id,
        "week_number": week_number,
        # [G-B2 · P2-CRON-OPT-4] days_offset ahora viaja en el batch SELECT (antes se
        # re-query-aba por fila vía "SELECT days_offset, week_number FROM plan_chunk_queue").
        "days_offset": 3,
        "pipeline_snapshot": {
            "form_data": {
                "totalDays": 7,
                "_plan_start_date": "2026-04-28",
                "tz_offset_minutes": 240,
            },
            "_pantry_pause_reason": reason,
            "_pantry_pause_started_at": "2026-04-30T00:00:00+00:00",
            "_pantry_pause_ttl_hours": ttl_hours,
            "_pantry_pause_reminders": 0,
            "_prev_chunk_pause_meta": {
                "previous_chunk_start_day": 1,
                "previous_chunk_end_day": 3,
                "prev_end_date": "2026-04-30",
                "days_until_prev_end": 0,
            },
        },
        "paused_seconds": paused_seconds,
    }


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks._check_chunk_learning_ready")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_resumes_chunk_when_temporal_gate_now_ready(
    mock_query, mock_write, mock_gate, mock_flex, mock_push,
):
    """Re-eval del gate retorna ready=True → chunk se reanuda como pending."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused = _paused_row(reason="prev_chunk_not_concluded", paused_seconds=600)

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused]
        if "SELECT plan_data FROM meal_plans" in sql:
            return {"plan_data": {"days": []}}
        if "SELECT days_offset, week_number FROM plan_chunk_queue" in sql:
            return {"days_offset": 3, "week_number": 2}
        return None

    mock_query.side_effect = query_side_effect
    mock_gate.return_value = {"ready": True, "reason": "passed"}

    _recover_pantry_paused_chunks()

    pending_updates = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0] and "'pending'" in c.args[0]
    ]
    assert len(pending_updates) >= 1, (
        f"Esperaba un UPDATE → status='pending'. Calls: {mock_write.call_args_list!r}"
    )
    # No debe activar flexible_mode si el gate ya pasa.
    mock_flex.assert_not_called()


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks._check_chunk_learning_ready")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_escalates_to_flexible_after_ttl(
    mock_query, mock_write, mock_gate, mock_flex, mock_push,
):
    """Tras TTL sin resolución, el chunk se reanuda en flexible_mode (no se congela)."""
    from cron_tasks import _recover_pantry_paused_chunks

    # paused_seconds (25h) > ttl_hours (24h) y gate sigue bloqueando
    paused = _paused_row(
        reason="prev_chunk_not_concluded",
        paused_seconds=25 * 3600,
        ttl_hours=24,
    )

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused]
        if "SELECT plan_data FROM meal_plans" in sql:
            return {"plan_data": {"days": []}}
        if "SELECT days_offset, week_number FROM plan_chunk_queue" in sql:
            return {"days_offset": 3, "week_number": 2}
        return None

    mock_query.side_effect = query_side_effect
    mock_gate.return_value = {"ready": False, "reason": "prev_chunk_day_not_yet_elapsed"}
    # _activate_flexible_mode devuelve un dict serializable (necesario para json.dumps).
    mock_flex.return_value = {"_pantry_flexible_mode": True}

    _recover_pantry_paused_chunks()

    mock_flex.assert_called_once()
    flex_call = mock_flex.call_args
    assert flex_call.kwargs.get("reason") == "prev_chunk_not_concluded_ttl"
    assert flex_call.kwargs.get("learning_flexible") is True

    # Debe encolar como pending tras flexible_mode.
    pending_updates = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0] and "'pending'" in c.args[0]
    ]
    assert len(pending_updates) >= 1


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks._check_chunk_learning_ready")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_keeps_paused_before_ttl_when_gate_still_blocks(
    mock_query, mock_write, mock_gate, mock_flex, mock_push,
):
    """Antes del TTL y con gate aún bloqueando, el chunk sigue pausado."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused = _paused_row(
        reason="prev_chunk_not_concluded",
        paused_seconds=3 * 3600,  # 3h < 24h TTL
        ttl_hours=24,
    )

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused]
        if "SELECT plan_data FROM meal_plans" in sql:
            return {"plan_data": {"days": []}}
        if "SELECT days_offset, week_number FROM plan_chunk_queue" in sql:
            return {"days_offset": 3, "week_number": 2}
        return None

    mock_query.side_effect = query_side_effect
    mock_gate.return_value = {"ready": False, "reason": "prev_chunk_day_not_yet_elapsed"}

    _recover_pantry_paused_chunks()

    # No debe escalar a flexible_mode.
    mock_flex.assert_not_called()
    # No debe haber UPDATE → status='pending' del row del paused chunk
    pending_updates_for_chunk = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0]
        and "'pending'" in c.args[0]
        and "chunk-1" in str(c.args)
    ]
    assert len(pending_updates_for_chunk) == 0, (
        "Antes del TTL con gate-still-blocking, el chunk no debe reanudarse"
    )


# ----------------------------------------------------------------------------
# [P0-bonus] Sanity check end-to-end con DB real: el recovery ahora SÍ ve filas.
# ----------------------------------------------------------------------------

@pytest.mark.e2e
def test_recovery_sees_paused_rows_with_real_db(seeded_user_profile):
    """[P0-bonus] Antes, _recover_pantry_paused_chunks llamaba a execute_sql_query
    sin fetch_all=True y siempre recibía []. Tras la fix (default seguro en el helper
    y fetch_all=True explícito en el callsite), el recovery debe ver el row real.

    Si este test falla, _recover_pantry_paused_chunks volvió a estar ciego.
    """
    import json
    import uuid as _uuid
    from db_core import execute_sql_query, execute_sql_write
    from cron_tasks import _recover_pantry_paused_chunks

    user_id, _ = seeded_user_profile
    plan_id = str(_uuid.uuid4())
    chunk_id = str(_uuid.uuid4())

    snapshot = {
        "form_data": {"totalDays": 7},
        "_pantry_pause_reason": "prev_chunk_not_concluded",
        "_pantry_pause_ttl_hours": 24,
    }
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, name, plan_data, calories, macros) "
        "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)",
        (plan_id, user_id, "Plan recovery sanity",
         json.dumps({"days": []}, ensure_ascii=False),
         2000, json.dumps({})),
    )
    execute_sql_write(
        "INSERT INTO plan_chunk_queue "
        "(id, user_id, meal_plan_id, week_number, days_offset, days_count, "
        " pipeline_snapshot, status, execute_after, updated_at) "
        "VALUES (%s, %s, %s, 2, 3, 4, %s::jsonb, 'pending_user_action', NOW(), NOW())",
        (chunk_id, user_id, plan_id, json.dumps(snapshot, ensure_ascii=False)),
    )

    try:
        with patch("cron_tasks._check_chunk_learning_ready",
                   return_value={"ready": True, "reason": "passed"}):
            _recover_pantry_paused_chunks()

        # Antes del fix: status seguiría siendo pending_user_action porque el cron
        # nunca veía la fila. Tras el fix: el cron debe haber reanudado el chunk.
        result = execute_sql_query(
            "SELECT status FROM plan_chunk_queue WHERE id = %s",
            (chunk_id,), fetch_one=True,
        )
        assert result and result.get("status") == "pending", (
            f"Recovery debió reanudar el chunk; status actual: {result!r}"
        )
    finally:
        execute_sql_write("DELETE FROM plan_chunk_queue WHERE id = %s", (chunk_id,))
        execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))
