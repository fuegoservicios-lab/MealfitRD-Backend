"""[P0-2] Tests para el rechazo de enqueue cuando _resolve_chunk_start_anchor cae a forced_8am_utc.

Antes: si snapshot, profile, y último plan no tenían TZ resoluble, el chunk se
encolaba con execute_after = 8am UTC + delay_days. Para usuarios en TZs negativas
(Bogotá UTC-5 → 3am local; PST UTC-8 → midnight local) eso disparaba la
generación antes de que el usuario registrara nada del día previo, rompiendo el
aprendizaje continuo.

Ahora: el chunk se inserta vía UPSERT y luego se flipea a 'pending_user_action'
con reason='tz_unresolved'. Recovery cron retenta `_resolve_chunk_start_anchor`
en cada tick; cuando el usuario abre la app y se persiste tz_offset_minutes,
reanuda con execute_after correcto.

Casos:
  1. Helper _maybe_notify_user_tz_unresolved: push, cooldown 24h, guest skip,
     fallo no propaga.
  2. _enqueue_plan_chunk con forced_8am_utc → flip a pending_user_action y push.
  3. _enqueue_plan_chunk con anchor 'snapshot' (TZ resuelta) → no flip.
  4. CHUNK_REJECT_FORCED_UTC_ENQUEUE=False → comportamiento legacy preservado.
  5. Recovery cron: TZ ahora resoluble → reanuda como pending con execute_after correcto.
  6. Recovery cron: TZ persistente irresoluble → tras MAX_ATTEMPTS escala a dead_letter.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))


def _hp_with_last(iso: str | None) -> dict:
    return {"health_profile": {"_tz_unresolved_notified_at": iso}} if iso else {"health_profile": {}}


# ---------- Helper push: cooldown / guest / fallo ----------

@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_helper_first_call_emits_push_and_persists_cooldown(mock_push, mock_query, mock_write):
    import cron_tasks
    mock_query.return_value = _hp_with_last(None)

    sent = cron_tasks._maybe_notify_user_tz_unresolved("user-p02-1")

    assert sent is True
    assert mock_push.called
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["user_id"] == "user-p02-1"
    assert "zona horaria" in push_kwargs["body"].lower() or "zona horaria" in push_kwargs["title"].lower()
    assert push_kwargs["url"].startswith("/dashboard?action_required=tz_unresolved")
    # Cooldown persistido
    assert mock_write.called
    sql_text = mock_write.call_args.args[0]
    assert "_tz_unresolved_notified_at" in sql_text


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_helper_cooldown_active_blocks_push(mock_push, mock_query, mock_write):
    import cron_tasks
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mock_query.return_value = _hp_with_last(recent)

    sent = cron_tasks._maybe_notify_user_tz_unresolved("user-p02-2")

    assert sent is False
    assert not mock_push.called
    assert not mock_write.called


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_helper_cooldown_expired_emits_push(mock_push, mock_query, mock_write):
    import cron_tasks
    expired = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    mock_query.return_value = _hp_with_last(expired)

    sent = cron_tasks._maybe_notify_user_tz_unresolved("user-p02-3")

    assert sent is True
    assert mock_push.called


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_helper_guest_skipped(mock_push, mock_query, mock_write):
    import cron_tasks
    sent = cron_tasks._maybe_notify_user_tz_unresolved("guest")
    assert sent is False
    assert not mock_push.called


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_helper_push_failure_returns_false(mock_push, mock_query, mock_write):
    import cron_tasks
    mock_query.return_value = _hp_with_last(None)
    mock_push.side_effect = Exception("push down")

    sent = cron_tasks._maybe_notify_user_tz_unresolved("user-p02-4")
    assert sent is False  # not raising, just returning False


# ---------- _enqueue_plan_chunk: flip cuando forced_8am_utc ----------

@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_flips_to_pending_user_action_when_forced_utc(
    mock_write, mock_query, mock_resolve, mock_notify
):
    import cron_tasks

    # _resolve_chunk_start_anchor → forced_8am_utc
    mock_resolve.return_value = (None, 0, "forced_8am_utc")
    # UPSERT exitoso
    mock_query.return_value = {"id": "chunk-p02-flip", "status": "pending", "inserted": True}

    snapshot = {"form_data": {"_plan_start_date": None}}
    cron_tasks._enqueue_plan_chunk(
        user_id="user-p02-flip",
        meal_plan_id="plan-p02-flip",
        week_number=2,
        days_offset=3,
        days_count=4,
        pipeline_snapshot=snapshot,
        chunk_kind="rolling_refill",
    )

    # Debe haber un UPDATE flipping a pending_user_action.
    flip_calls = [
        call for call in mock_write.call_args_list
        if "pending_user_action" in (call.args[0] if call.args else "")
        and "tz_unresolved" in str(call.args[1] if len(call.args) > 1 else "")
    ]
    assert flip_calls, (
        f"esperaba UPDATE flipping a pending_user_action con tz_unresolved; "
        f"recibí: {[c.args for c in mock_write.call_args_list]}"
    )
    flip_args = flip_calls[0].args
    payload = flip_args[1][0]  # first param of UPDATE = jsonb snapshot
    snap_dict = json.loads(payload)
    assert snap_dict.get("_pantry_pause_reason") == "tz_unresolved"
    assert snap_dict.get("_tz_recovery_attempts") == 0
    assert "_pantry_pause_started_at" in snap_dict
    # Push debe haber sido invocado
    assert mock_notify.called


@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_does_not_flip_when_anchor_resolved(
    mock_write, mock_query, mock_resolve, mock_notify
):
    """anchor_source != 'forced_8am_utc' → comportamiento normal, sin flip."""
    import cron_tasks

    # TZ resuelta vía snapshot
    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, -300, "snapshot")  # Bogotá UTC-5
    mock_query.return_value = {"id": "chunk-p02-ok", "status": "pending", "inserted": True}

    snapshot = {"form_data": {"_plan_start_date": start_dt.isoformat()}}
    # [P0-4] Mockear pantry suficiente para que P0-4 no flipee por nevera vacía
    # (lo que falsearía la prueba de "P0-2 no flipea").
    with patch(
        "db_inventory.get_user_inventory_net",
        return_value=["500g pollo", "300g arroz", "200g brocoli", "1 cebolla"],
    ):
        cron_tasks._enqueue_plan_chunk(
            user_id="user-p02-ok",
            meal_plan_id="plan-p02-ok",
            week_number=2,
            days_offset=3,
            days_count=4,
            pipeline_snapshot=snapshot,
            chunk_kind="rolling_refill",
        )

    # NO debe haber flip a pending_user_action
    flip_calls = [
        call for call in mock_write.call_args_list
        if "pending_user_action" in (call.args[0] if call.args else "")
    ]
    assert not flip_calls, (
        f"con anchor resuelto NO debe flipear; recibí: {[c.args[0] for c in flip_calls]}"
    )
    assert not mock_notify.called


@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_legacy_when_flag_disabled(
    mock_write, mock_query, mock_resolve, mock_notify
):
    """CHUNK_REJECT_FORCED_UTC_ENQUEUE=False → comportamiento legacy (sin flip)."""
    import cron_tasks

    mock_resolve.return_value = (None, 0, "forced_8am_utc")
    mock_query.return_value = {"id": "chunk-p02-legacy", "status": "pending", "inserted": True}

    snapshot = {"form_data": {}}
    # [P0-4] El guard proactivo de pantry corre tras el bloque P0-2 cuando éste
    # no aplica. Lo desactivamos aquí para aislar la prueba al comportamiento
    # legacy del flag P0-2.
    with patch("cron_tasks.CHUNK_REJECT_FORCED_UTC_ENQUEUE", False), \
         patch("cron_tasks.CHUNK_PANTRY_PROACTIVE_GUARD", False):
        cron_tasks._enqueue_plan_chunk(
            user_id="user-p02-legacy",
            meal_plan_id="plan-p02-legacy",
            week_number=2,
            days_offset=3,
            days_count=4,
            pipeline_snapshot=snapshot,
            chunk_kind="rolling_refill",
        )

    flip_calls = [
        call for call in mock_write.call_args_list
        if "pending_user_action" in (call.args[0] if call.args else "")
    ]
    assert not flip_calls
    assert not mock_notify.called


# ---------- Recovery cron: tz_unresolved branch ----------

@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_resumes_chunk_when_tz_now_resolved(
    mock_query, mock_write, mock_resolve, mock_escalate, mock_notify
):
    """Cuando el recovery cron encuentra un chunk pausado con tz_unresolved y
    `_resolve_chunk_start_anchor` ahora devuelve un source válido, debe reanudar
    el chunk como 'pending' con execute_after recalculado."""
    import cron_tasks

    paused_row = {
        "id": "chunk-rec-1",
        "user_id": "user-rec-1",
        "meal_plan_id": "plan-rec-1",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "tz_unresolved",
            "_tz_recovery_attempts": 1,
            "form_data": {},
        },
        "paused_seconds": 600,
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        if "days_offset FROM plan_chunk_queue" in sql:
            return {"days_offset": 3}
        return None

    mock_query.side_effect = query_router

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, -300, "profile_today")  # TZ ahora resuelta

    cron_tasks._recover_pantry_paused_chunks()

    # Debe haber un UPDATE reanudando: status='pending' y tzOffset propagado.
    resume_calls = [
        call for call in mock_write.call_args_list
        if call.args
        and "status = 'pending'" in call.args[0]
        and "execute_after" in call.args[0]
    ]
    assert resume_calls, "el recovery debe reanudar el chunk como 'pending'"
    payload = json.loads(resume_calls[0].args[1][1])
    assert payload["form_data"]["tz_offset_minutes"] == -300
    assert payload["form_data"]["_chunk_anchor_source"] == "profile_today"
    assert payload["_pantry_pause_resolution"] == "tz_recovered"

    # No escalación
    assert not mock_escalate.called


@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_escalates_after_max_attempts(
    mock_query, mock_write, mock_resolve, mock_escalate, mock_notify
):
    """Si el recovery cron llega a CHUNK_TZ_RECOVERY_MAX_ATTEMPTS sin resolver
    la TZ, debe escalar a dead_letter con reason='unrecoverable_tz_unresolved'."""
    import cron_tasks
    from constants import CHUNK_TZ_RECOVERY_MAX_ATTEMPTS

    # Simular ya estando un intento por debajo del max.
    paused_row = {
        "id": "chunk-rec-esc",
        "user_id": "user-rec-esc",
        "meal_plan_id": "plan-rec-esc",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "tz_unresolved",
            "_tz_recovery_attempts": CHUNK_TZ_RECOVERY_MAX_ATTEMPTS - 1,
            "form_data": {},
        },
        "paused_seconds": 7200,
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_router
    mock_resolve.return_value = (None, 0, "forced_8am_utc")  # Sigue sin resolver

    cron_tasks._recover_pantry_paused_chunks()

    assert mock_escalate.called, "debe llamar a _escalate_unrecoverable_chunk"
    esc_kwargs = mock_escalate.call_args.kwargs
    assert esc_kwargs["escalation_reason"] == "unrecoverable_tz_unresolved"
    assert esc_kwargs["recovery_attempts"] == CHUNK_TZ_RECOVERY_MAX_ATTEMPTS
    assert esc_kwargs["task_id"] == "chunk-rec-esc"


@patch("cron_tasks._maybe_notify_user_tz_unresolved")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_increments_counter_when_still_unresolved(
    mock_query, mock_write, mock_resolve, mock_escalate, mock_notify
):
    """Si TZ sigue irresoluble pero quedan intentos, incrementar el contador y
    notificar al usuario (con cooldown), sin escalar."""
    import cron_tasks

    paused_row = {
        "id": "chunk-rec-wait",
        "user_id": "user-rec-wait",
        "meal_plan_id": "plan-rec-wait",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "tz_unresolved",
            "_tz_recovery_attempts": 1,  # Bajo el max
            "form_data": {},
        },
        "paused_seconds": 1800,
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_router
    mock_resolve.return_value = (None, 0, "forced_8am_utc")

    cron_tasks._recover_pantry_paused_chunks()

    # No escalación
    assert not mock_escalate.called

    # Persistencia del contador incrementado
    persist_calls = [
        call for call in mock_write.call_args_list
        if call.args
        and "pipeline_snapshot" in call.args[0]
        and "status" not in call.args[0]
    ]
    assert persist_calls, "debe persistir el contador incrementado"
    payload = json.loads(persist_calls[0].args[1][0])
    assert payload["_tz_recovery_attempts"] == 2  # 1 + 1

    # Notificación al usuario (con cooldown — no se valida si efectivamente envía,
    # solo que se invoque el helper)
    assert mock_notify.called
