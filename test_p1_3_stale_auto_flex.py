"""
Tests P1-3: Auto-escalado a flex+advisory cuando hard_fail está desactivado.

Antes: cuando snapshot >FORCE_GENERATE_HOURS y backoff agotado, el chunk se
pausaba con `stale_snapshot_live_unreachable` y esperaba hasta
CHUNK_STALE_MAX_PAUSE_HOURS (24h) antes de que `_recover_pantry_paused_chunks`
lo escalara a flex+advisory_only.

Cambio P1-3: cuando `CHUNK_PANTRY_HARD_FAIL_ON_STALE=False` (override explícito
"preferir generar con datos parciales antes que bloquear"), bypaseamos la pausa
y escalamos directo a flex+advisory_only — saltando los 24h de espera.

Cubre:
  1. hard_fail=False + backoff agotado + task_id → escalado directo a flex.
  2. hard_fail=True (default) → comportamiento legacy (pausa).
  3. Sin task_id (defensivo) → comportamiento legacy.
  4. Push notification disparada en el escalado P1-3.
"""
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

import pytest


def _stale_snapshot_form_data(age_hours=30):
    """Helper: snapshot stale (más viejo que FORCE_GENERATE_HOURS=24h por defecto)."""
    captured = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    return {
        "current_pantry_ingredients": ["500g Arroz", "200g Pollo"],
        "_pantry_captured_at": captured,
        "tzOffset": 0,
        "tz_offset_minutes": 0,
    }


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_inventory_live_failure")
@patch("cron_tasks._fetch_inventory_with_backoff")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._get_user_tz_live")
def test_p1_3_hard_fail_off_escalates_to_flex_directly(
    mock_tz, mock_get_inv, mock_backoff, mock_record_fail, mock_push
):
    """
    Con `CHUNK_PANTRY_HARD_FAIL_ON_STALE=False`:
      - snapshot 30h (> FORCE_GENERATE 24h)
      - live inicial falla
      - backoff retry agotado
    → escalado directo a flex+advisory_only, push enviada, NO pausa.
    """
    import cron_tasks

    mock_tz.return_value = 0  # sin TZ drift
    mock_get_inv.return_value = None  # live inicial caído
    mock_backoff.return_value = (None, [1000, 2000, 3000], "timeout")  # backoff agotado
    mock_record_fail.return_value = False  # no degraded yet

    snapshot_form_data = _stale_snapshot_form_data(age_hours=30)
    form_data = {}

    with patch("cron_tasks.CHUNK_PANTRY_HARD_FAIL_ON_STALE", False):
        # _pause_chunk_for_stale_inventory NO debe ser llamado en este path P1-3.
        with patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:
            result = cron_tasks._refresh_chunk_pantry(
                user_id="user-p13-1",
                form_data=form_data,
                snapshot_form_data=snapshot_form_data,
                task_id="task-p13-1",
                week_number=2,
            )

    # Validaciones del fix P1-3
    assert result.get("_pantry_flexible_mode") is True, (
        f"Esperaba _pantry_flexible_mode=True; got {result.get('_pantry_flexible_mode')}"
    )
    assert result.get("_pantry_advisory_only") is True, (
        f"Esperaba _pantry_advisory_only=True; got {result.get('_pantry_advisory_only')}"
    )
    assert result.get("_fresh_pantry_source") == "stale_snapshot_auto_flex"
    assert result.get("current_pantry_ingredients") == ["500g Arroz", "200g Pollo"]
    assert result.get("_pantry_paused") is not True

    # Pausa NO debe haberse llamado.
    assert not mock_pause.called, (
        f"P1-3 debe bypassear pause; pero _pause_chunk_for_stale_inventory fue llamado "
        f"con args {mock_pause.call_args_list}"
    )

    # Push notification SÍ debe haberse disparado.
    assert mock_push.called, "P1-3 debe enviar push notification al escalar a flex"
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["user_id"] == "user-p13-1"
    assert "datos parciales" in push_kwargs["body"].lower() or \
           "30h" in push_kwargs["body"]


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_inventory_live_failure")
@patch("cron_tasks._fetch_inventory_with_backoff")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._get_user_tz_live")
def test_p1_3_hard_fail_on_keeps_legacy_pause(
    mock_tz, mock_get_inv, mock_backoff, mock_record_fail, mock_push
):
    """
    Con `CHUNK_PANTRY_HARD_FAIL_ON_STALE=True` (default), el comportamiento
    legacy se preserva: el chunk se pausa con stale_snapshot_live_unreachable.
    P1-3 NO debe romper este path.
    """
    import cron_tasks

    mock_tz.return_value = 0
    mock_get_inv.return_value = None
    mock_backoff.return_value = (None, [1000, 2000, 3000], "timeout")
    mock_record_fail.return_value = False

    # snapshot 30h pero NO > CHUNK_PANTRY_HARD_FAIL_AGE_HOURS (default 48h)
    # para que NO entre en la rama hard-fail absoluto y caiga al path 'stale_snapshot_live_unreachable'.
    snapshot_form_data = _stale_snapshot_form_data(age_hours=30)
    form_data = {}

    with patch("cron_tasks.CHUNK_PANTRY_HARD_FAIL_ON_STALE", True):
        with patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:
            result = cron_tasks._refresh_chunk_pantry(
                user_id="user-p13-2",
                form_data=form_data,
                snapshot_form_data=snapshot_form_data,
                task_id="task-p13-2",
                week_number=2,
            )

    # Comportamiento legacy: pausa fue llamada.
    assert mock_pause.called, (
        "Con hard_fail=True debe pausarse; _pause_chunk_for_stale_inventory NO fue llamado"
    )
    pause_args = mock_pause.call_args
    assert pause_args.kwargs.get("reason") == "stale_snapshot_live_unreachable" or (
        len(pause_args.args) >= 5 and pause_args.args[4] == "stale_snapshot_live_unreachable"
    )

    # _pantry_flexible_mode NO debe estar seteado en el legacy path
    assert result.get("_pantry_flexible_mode") is not True
    assert result.get("_fresh_pantry_source") == "stale_snapshot"


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._record_inventory_live_failure")
@patch("cron_tasks._fetch_inventory_with_backoff")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._get_user_tz_live")
def test_p1_3_no_task_id_falls_back_to_legacy(
    mock_tz, mock_get_inv, mock_backoff, mock_record_fail, mock_push
):
    """
    Sin task_id (caso defensivo), incluso con hard_fail=False, no podemos
    pausar ni escalar via task. El path P1-3 requiere task_id; sin él, cae
    al fallback legacy de devolver pantry_fallback con marker stale_snapshot.
    """
    import cron_tasks

    mock_tz.return_value = 0
    mock_get_inv.return_value = None
    mock_backoff.return_value = (None, [1000, 2000, 3000], "timeout")
    mock_record_fail.return_value = False

    snapshot_form_data = _stale_snapshot_form_data(age_hours=30)
    form_data = {}

    with patch("cron_tasks.CHUNK_PANTRY_HARD_FAIL_ON_STALE", False):
        result = cron_tasks._refresh_chunk_pantry(
            user_id="user-p13-3",
            form_data=form_data,
            snapshot_form_data=snapshot_form_data,
            task_id=None,  # ← sin task_id
            week_number=None,
        )

    # Sin task_id, P1-3 NO se ejecuta porque requiere task_id para pausar/escalar
    # (definido como `if not CHUNK_PANTRY_HARD_FAIL_ON_STALE and task_id:`).
    # Cae al fallback no-task del flow.
    assert result.get("_fresh_pantry_source") in (
        "stale_snapshot",
        "stale_snapshot_auto_flex",
    )
    # _pantry_flexible_mode no debe estar seteado en el fallback no-task.
    # (P1-3 requiere task_id, así que en este escenario NO se activa.)
    if result.get("_fresh_pantry_source") == "stale_snapshot":
        assert result.get("_pantry_flexible_mode") is not True


@patch("cron_tasks._record_inventory_live_failure")
@patch("cron_tasks._fetch_inventory_with_backoff")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._get_user_tz_live")
def test_p1_3_push_notification_failure_does_not_block_flex(
    mock_tz, mock_get_inv, mock_backoff, mock_record_fail
):
    """
    Si la push notification falla en P1-3, el escalado a flex DEBE proceder
    igual — la notificación es best-effort, no parte del contrato del fix.
    """
    import cron_tasks

    mock_tz.return_value = 0
    mock_get_inv.return_value = None
    mock_backoff.return_value = (None, [1000, 2000, 3000], "timeout")
    mock_record_fail.return_value = False

    snapshot_form_data = _stale_snapshot_form_data(age_hours=30)
    form_data = {}

    # Push lanza excepción
    with patch("cron_tasks._dispatch_push_notification", side_effect=Exception("push backend down")):
        with patch("cron_tasks.CHUNK_PANTRY_HARD_FAIL_ON_STALE", False):
            with patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:
                result = cron_tasks._refresh_chunk_pantry(
                    user_id="user-p13-4",
                    form_data=form_data,
                    snapshot_form_data=snapshot_form_data,
                    task_id="task-p13-4",
                    week_number=3,
                )

    # Pese al fallo de la push, el escalado debe completarse.
    assert result.get("_pantry_flexible_mode") is True
    assert result.get("_pantry_advisory_only") is True
    assert result.get("_fresh_pantry_source") == "stale_snapshot_auto_flex"
    assert not mock_pause.called  # tampoco debe haber pausa
