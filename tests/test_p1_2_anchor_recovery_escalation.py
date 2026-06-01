"""[P1-2] Tests para escalación de chunks pausados sin ancla recuperable.

Cubre el escenario "plan congelado":
  - Plan corrupto sin `_plan_start_date`, `grocery_start_date`, ni `created_at`
    útil. El gate temporal devuelve `missing_start_date_no_anchor`.
  - Antes: el chunk se quedaba en `pending_user_action` indefinidamente; el
    cron `_recover_pantry_paused_chunks` no tenía branch para este reason.
  - Ahora: cada tick incrementa `plan_data._anchor_recovery_attempts` y, al
    superar `CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS` (default 3), invoca
    `_escalate_unrecoverable_chunk(reason='unrecoverable_missing_anchor')`,
    que dead-letterea el chunk, marca `meal_plans.plan_data._user_action_required`
    y manda push pidiendo regenerar el plan.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_2_anchor_recovery_escalation.py -v
"""
import os
import sys
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# 1. _escalate_unrecoverable_chunk acepta escalation_reason y emite copy correcto
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_escalate_unrecoverable_chunk_default_reason_is_recovery_exhausted(
    mock_write, mock_push
):
    """Backwards compat: sin pasar escalation_reason, comportamiento previo
    se conserva (reason='recovery_exhausted', push de regeneración estándar)."""
    from cron_tasks import _escalate_unrecoverable_chunk

    _escalate_unrecoverable_chunk(
        task_id="task-1",
        user_id="user-1",
        plan_id="plan-1",
        week_number=3,
        recovery_attempts=2,
    )

    # SQL del UPDATE plan_chunk_queue debe llevar reason=recovery_exhausted en %s.
    queue_update_call = next(
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0]
    )
    assert "recovery_exhausted" in queue_update_call.args[1]

    # Push debe usar el copy de "recovery_exhausted".
    mock_push.assert_called_once()
    push_kwargs = mock_push.call_args.kwargs
    assert "regenera tu plan" in push_kwargs["body"].lower()
    assert "recovery_exhausted=1" in push_kwargs["url"]


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_escalate_unrecoverable_chunk_missing_anchor_uses_specific_copy(
    mock_write, mock_push
):
    """Con escalation_reason='unrecoverable_missing_anchor', push debe llevar
    copy específico ('Tu plan necesita regenerarse') y deeplink distinto."""
    from cron_tasks import _escalate_unrecoverable_chunk

    _escalate_unrecoverable_chunk(
        task_id="task-2",
        user_id="user-2",
        plan_id="plan-2",
        week_number=4,
        recovery_attempts=3,
        escalation_reason="unrecoverable_missing_anchor",
    )

    mock_push.assert_called_once()
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["title"] == "Tu plan necesita regenerarse"
    assert "problema técnico" in push_kwargs["body"]
    assert push_kwargs["url"] == "/dashboard?action_required=missing_anchor"


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_escalate_unrecoverable_chunk_persists_user_action_required(
    mock_write, mock_push
):
    """El UPDATE meal_plans debe agregar `_user_action_required` con reason y
    week_number para que el frontend pueda renderizar banner de acción."""
    from cron_tasks import _escalate_unrecoverable_chunk

    _escalate_unrecoverable_chunk(
        task_id="task-3",
        user_id="user-3",
        plan_id="plan-3",
        week_number=5,
        recovery_attempts=3,
        escalation_reason="unrecoverable_missing_anchor",
    )

    plan_update_call = next(
        c for c in mock_write.call_args_list
        if "UPDATE meal_plans" in c.args[0]
    )
    sql = plan_update_call.args[0]
    params = plan_update_call.args[1]
    assert "_user_action_required" in sql
    assert "_recovery_exhausted_chunks" in sql
    # El reason aparece dos veces en params: una para _recovery_exhausted_chunks
    # entry y otra para _user_action_required.
    assert params.count("unrecoverable_missing_anchor") == 2


# ---------------------------------------------------------------------------
# 2. _recover_pantry_paused_chunks: reanuda si el ancla apareció
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_resumes_chunk_when_anchor_recovered(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Si plan_data ahora tiene `grocery_start_date` u otro ancla, el chunk
    se reanuda como `pending` y NO escala."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-row-1",
        "user_id": "user-A",
        "meal_plan_id": "plan-A",
        "week_number": 2,
        "pipeline_snapshot": {"_pantry_pause_reason": "missing_start_date_no_anchor"},
        "paused_seconds": 600,
    }
    anchor_row = {
        "psd": None,
        "gsd": "2026-04-15",
        "created_at": None,
        "attempts": 1,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    # No debe escalar (ancla recuperada).
    mock_escalate.assert_not_called()

    # Debe haber un UPDATE plan_chunk_queue con status='pending'.
    pending_updates = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0] and "'pending'" in c.args[0]
    ]
    assert len(pending_updates) >= 1, "Chunk debe reanudarse como pending"


# ---------------------------------------------------------------------------
# 3. _recover_pantry_paused_chunks: incrementa contador si ancla sigue ausente
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_increments_attempts_when_anchor_still_missing(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Sin ancla y attempts<MAX, incrementa contador y NO escala."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-row-2",
        "user_id": "user-B",
        "meal_plan_id": "plan-B",
        "week_number": 3,
        "pipeline_snapshot": {"_pantry_pause_reason": "missing_start_date_no_anchor"},
        "paused_seconds": 600,
    }
    anchor_row = {
        "psd": None,
        "gsd": None,
        "created_at": None,
        "attempts": 0,  # primer intento
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    # Debe haber UPDATE incrementando attempts.
    persist_updates = [
        c for c in mock_write.call_args_list
        if "_anchor_recovery_attempts" in c.args[0]
    ]
    assert len(persist_updates) == 1
    assert persist_updates[0].args[1][0] == 1, "attempts debe incrementarse a 1"

    # NO debe escalar todavía.
    mock_escalate.assert_not_called()


# ---------------------------------------------------------------------------
# 4. _recover_pantry_paused_chunks: escala al exceder umbral
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_escalates_when_attempts_exceed_max(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Sin ancla y attempts>=MAX, debe llamar a _escalate_unrecoverable_chunk
    con escalation_reason='unrecoverable_missing_anchor'."""
    from cron_tasks import _recover_pantry_paused_chunks
    from constants import CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS

    paused_row = {
        "id": "chunk-row-3",
        "user_id": "user-C",
        "meal_plan_id": "plan-C",
        "week_number": 4,
        "pipeline_snapshot": {"_pantry_pause_reason": "missing_start_date_no_anchor"},
        "paused_seconds": 9999,
    }
    # attempts ya está en MAX-1; este tick lo lleva a MAX → escala.
    # [P1-CHUNK-4] started_at antiguo → el gate wall-clock (max_attempts ×
    # CHUNK_RECOVERY_MIN_WALL_MINUTES_PER_ATTEMPT) se cumple y permite escalar.
    anchor_row = {
        "psd": None,
        "gsd": None,
        "created_at": None,
        "attempts": CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS - 1,
        "started_at": "2020-01-01T00:00:00+00:00",
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    mock_escalate.assert_called_once()
    call_kwargs = mock_escalate.call_args.kwargs
    assert call_kwargs["escalation_reason"] == "unrecoverable_missing_anchor"
    assert call_kwargs["plan_id"] == "plan-C"
    assert call_kwargs["week_number"] == 4
    assert call_kwargs["recovery_attempts"] == CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# 5. _recover_pantry_paused_chunks: chunk sin meal_plan_id no rompe
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_skips_anchor_chunk_when_meal_plan_id_null(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Defensa: si por alguna razón meal_plan_id es NULL en la fila,
    saltamos el chunk sin crashear ni escalar."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-row-4",
        "user_id": "user-D",
        "meal_plan_id": None,
        "week_number": 2,
        "pipeline_snapshot": {"_pantry_pause_reason": "missing_start_date_no_anchor"},
        "paused_seconds": 600,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_side_effect

    # No debe crashear.
    _recover_pantry_paused_chunks()

    mock_escalate.assert_not_called()
