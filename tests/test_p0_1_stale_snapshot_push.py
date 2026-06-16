"""[P0-1] Tests para notificación de chunk pausado por stale_snapshot.

Cubre el gap donde `_pause_chunk_for_stale_inventory(reason="stale_snapshot")`
dejaba al usuario sin enterarse de la pausa. Ahora se dispara push con cooldown
24h, y la variante `stale_snapshot_live_unreachable` no genera doble push porque
el caller ya notifica antes de pausar.

Casos:
  1. reason="stale_snapshot" + sin cooldown previo → push enviado y cooldown persistido.
  2. reason="stale_snapshot" + cooldown activo (<24h) → no push.
  3. reason="stale_snapshot" + cooldown vencido (>24h) → push enviado.
  4. reason="stale_snapshot_live_unreachable" → helper NO llamado (caller maneja).
  5. user_id="guest" → no push.
  6. Push falla → pausa igual persiste (best-effort).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))


def _hp_with_last(iso: str | None) -> dict:
    return {"health_profile": {"_stale_snapshot_paused_notified_at": iso}} if iso else {"health_profile": {}}


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_first_pause_emits_push_and_persists_cooldown(mock_push, mock_query, mock_write):
    # [test fix · P1-2 CAS · 2026-05-10] El cooldown ya NO es SELECT-then-UPDATE
    # en Python: `_maybe_notify_user_stale_snapshot_paused` delega a
    # `_claim_push_cooldown_slot`, que hace UN `UPDATE user_profiles ... RETURNING id`
    # con la condición de cooldown embebida en el WHERE. Slot ganado = rows
    # devueltas. El slot_key viaja como PARÁMETRO (`ARRAY[%s]::text[]` /
    # `health_profile ->> %s`), no como literal en el SQL string.
    import cron_tasks
    # Slot ganado: el CAS UPDATE devuelve >=1 fila.
    mock_write.return_value = [{"id": "user-p01-1"}]

    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "user-p01-1", snapshot_age_hours=8.4
    )

    assert sent is True, "primer pause sin cooldown previo debe enviar push"
    assert mock_push.called, "_dispatch_push_notification debe ser invocado"
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["user_id"] == "user-p01-1"
    assert "pausa" in push_kwargs["title"].lower() or "pausa" in push_kwargs["body"].lower()
    assert "8h" in push_kwargs["body"], "el body debe incluir la edad del snapshot"

    # Cooldown reclamado vía CAS sobre user_profiles. El slot_key se pasa como
    # parámetro (no como literal en el SQL string).
    assert mock_write.called, "el cooldown CAS debe ejecutarse"
    write_args = mock_write.call_args
    sql_text = write_args.args[0]
    assert "user_profiles" in sql_text
    assert "RETURNING" in sql_text.upper(), "el CAS debe usar UPDATE ... RETURNING"
    sql_params = write_args.args[1]
    assert "_stale_snapshot_paused_notified_at" in sql_params, (
        "el slot_key debe viajar como parámetro del CAS UPDATE"
    )


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_cooldown_active_blocks_push(mock_push, mock_query, mock_write):
    # [test fix · P1-2 CAS] Cooldown activo = el CAS UPDATE no matchea el WHERE
    # (timestamp <24h) → 0 filas → `_claim_push_cooldown_slot` retorna False →
    # no push. Lo modelamos devolviendo lista vacía de `execute_sql_write`.
    import cron_tasks
    mock_write.return_value = []

    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "user-p01-2", snapshot_age_hours=10.0
    )

    assert sent is False, "cooldown <24h (CAS pierde) debe bloquear el push"
    assert not mock_push.called, "no debe disparar _dispatch_push_notification"


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_cooldown_expired_emits_push(mock_push, mock_query, mock_write):
    import cron_tasks
    expired = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    mock_query.return_value = _hp_with_last(expired)

    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "user-p01-3", snapshot_age_hours=12.0
    )

    assert sent is True, "cooldown >24h debe permitir nuevo push"
    assert mock_push.called


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_guest_user_does_not_emit_push(mock_push, mock_query, mock_write):
    import cron_tasks
    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "guest", snapshot_age_hours=10.0
    )
    assert sent is False
    assert not mock_push.called
    assert not mock_query.called, "ni siquiera debe leer health_profile para guest"


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_empty_user_id_does_not_emit_push(mock_push, mock_query, mock_write):
    import cron_tasks
    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "", snapshot_age_hours=10.0
    )
    assert sent is False
    assert not mock_push.called


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._dispatch_push_notification")
def test_push_failure_returns_false_without_raising(mock_push, mock_query, mock_write):
    """Si el push backend falla, la función no debe propagar la excepción."""
    import cron_tasks
    mock_query.return_value = _hp_with_last(None)
    mock_push.side_effect = Exception("push backend down")

    sent = cron_tasks._maybe_notify_user_stale_snapshot_paused(
        "user-p01-4", snapshot_age_hours=10.0
    )

    assert sent is False, "fallo de push debe devolver False, no propagar"


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._maybe_notify_user_stale_snapshot_paused")
def test_pause_for_pure_stale_snapshot_calls_notifier(mock_notify, mock_query, mock_write):
    """`_pause_chunk_for_stale_inventory` con reason por defecto ('stale_snapshot')
    debe llamar al helper de notificación."""
    import cron_tasks
    mock_query.return_value = {"pipeline_snapshot": {}}

    cron_tasks._pause_chunk_for_stale_inventory(
        task_id="task-p01-pure",
        user_id="user-p01-pure",
        week_number=2,
        snapshot_age_hours=9.7,
    )

    assert mock_notify.called, (
        "stale_snapshot puro debe disparar _maybe_notify_user_stale_snapshot_paused"
    )
    args, kwargs = mock_notify.call_args
    assert args[0] == "user-p01-pure"
    assert args[1] == 9.7


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._maybe_notify_user_stale_snapshot_paused")
def test_pause_for_live_unreachable_skips_notifier(mock_notify, mock_query, mock_write):
    """`_pause_chunk_for_stale_inventory` con reason='stale_snapshot_live_unreachable'
    NO debe llamar al helper porque el caller ya emitió un push contextual."""
    import cron_tasks
    mock_query.return_value = {"pipeline_snapshot": {}}

    cron_tasks._pause_chunk_for_stale_inventory(
        task_id="task-p01-live",
        user_id="user-p01-live",
        week_number=2,
        snapshot_age_hours=30.0,
        reason="stale_snapshot_live_unreachable",
    )

    assert not mock_notify.called, (
        "stale_snapshot_live_unreachable NO debe disparar el helper "
        "(el caller ya envió push); evita doble-push"
    )


@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._maybe_notify_user_stale_snapshot_paused")
def test_pause_persists_state_even_if_notifier_raises(mock_notify, mock_query, mock_write):
    """Best-effort: si el helper de notificación lanza, la pausa debe quedar
    persistida igual (no debe romper la transición de estado del chunk)."""
    import cron_tasks
    mock_query.return_value = {"pipeline_snapshot": {}}
    mock_notify.side_effect = Exception("notify backend down")

    # No debe propagar
    cron_tasks._pause_chunk_for_stale_inventory(
        task_id="task-p01-resilient",
        user_id="user-p01-resilient",
        week_number=3,
        snapshot_age_hours=8.0,
    )

    # El UPDATE de pause snapshot SÍ debe haberse ejecutado
    update_calls = [
        call for call in mock_write.call_args_list
        if "plan_chunk_queue" in (call.args[0] if call.args else "")
        and "pending_user_action" in (call.args[0] if call.args else "")
    ]
    assert update_calls, (
        "el UPDATE a status='pending_user_action' debe persistirse aunque el "
        "helper de notificación falle"
    )
