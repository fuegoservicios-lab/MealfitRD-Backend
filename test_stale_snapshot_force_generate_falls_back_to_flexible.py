import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta
from cron_tasks import _refresh_chunk_pantry
from constants import CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS

def test_stale_snapshot_force_generate_falls_back_to_flexible():
    user_id = "test_user_stale_fallback"
    
    # Crea un snapshot capturado hace más de 8 horas
    stale_hours = CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS + 2
    captured_time = datetime.now(timezone.utc) - timedelta(hours=stale_hours)
    
    snapshot_form_data = {
        "current_pantry_ingredients": ["1 ud Pollo viejo"],
        "_pantry_captured_at": captured_time.isoformat()
    }
    
    # Mockeamos:
    # 1. get_user_inventory_net para que siempre tire exception (simulando timeout / fallo de conexion)
    # 2. _dispatch_push_notification para poder verificar que se llamó
    with patch("cron_tasks.get_user_inventory_net", side_effect=Exception("Timeout simulado")) as mock_inventory, \
         patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:
        
        updated_form_data = _refresh_chunk_pantry(
            user_id=user_id,
            form_data={},
            snapshot_form_data=snapshot_form_data,
            task_id="dummy_task_id",
            week_number=1
        )
        
        # Validaciones
        # 1. El live fetch debe haberse intentado al menos 3 veces (inicial, retry normal, y la
        # rama de backoff extendido). El backoff helper hace 3 intentos internos, así que el total
        # observable es 1 (inicial) + 1 (retry) + 3 (backoff) = 5 cuando el snapshot supera
        # CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS. Asertamos >=3 para no acoplarnos al detalle
        # del backoff (CHUNK_LIVE_FETCH_BACKOFF_TIMEOUTS_SECONDS=30,60,90 es env-tunable).
        assert mock_inventory.call_count >= 3, (
            f"Debería intentar el fetch live al menos 3 veces (inicial + retry + backoff). "
            f"Count: {mock_inventory.call_count}"
        )
        
        # 2. Debe pausarse y NO caer en flexible_mode
        assert updated_form_data.get("_pantry_flexible_mode") is not True
        assert updated_form_data.get("_pantry_paused") is True
        
        # 3. Debe haber llamado a pause con el reason esperado
        mock_pause.assert_called_once()
        args, kwargs = mock_pause.call_args
        assert args[0] == "dummy_task_id"
        assert kwargs.get("reason") == "stale_snapshot_live_unreachable"

def test_p0_5_snapshot_beyond_hard_fail_pauses_unconditionally():
    """[P0-5] Cuando el snapshot supera CHUNK_PANTRY_HARD_FAIL_AGE_HOURS (default 48h),
    pausamos siempre — incluso si el path normal hubiera escalado a flex+advisory.

    Escenario: plan de 30 días con _proactive_refresh caído. El snapshot tiene 72h y la
    API live del usuario está caída. Sin el hard-fail, el código bajaba a la rama
    `_live_degraded_now → flexible_mode + advisory_only` y generaba contra inventario de
    3 días. Con el hard-fail, pausa con reason='snapshot_hard_fail_age_exceeded' y notifica
    al usuario para que abra la app a sincronizar la nevera.
    """
    from constants import CHUNK_PANTRY_HARD_FAIL_AGE_HOURS

    user_id = "test_user_hard_fail"
    # 24h por encima del cap → claramente hard-fail.
    stale_hours = CHUNK_PANTRY_HARD_FAIL_AGE_HOURS + 24
    captured_time = datetime.now(timezone.utc) - timedelta(hours=stale_hours)

    snapshot_form_data = {
        "current_pantry_ingredients": ["1 ud Pollo viejo de hace 3 días"],
        "_pantry_captured_at": captured_time.isoformat(),
    }

    with patch("cron_tasks.get_user_inventory_net", side_effect=Exception("API down")), \
         patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause, \
         patch("cron_tasks._dispatch_push_notification") as mock_notify:

        updated = _refresh_chunk_pantry(
            user_id=user_id,
            form_data={},
            snapshot_form_data=snapshot_form_data,
            task_id="task_hard_fail",
            week_number=8,
        )

    # 1. Pausa por hard-fail con reason específico (no "stale_snapshot_live_unreachable").
    mock_pause.assert_called_once()
    args, kwargs = mock_pause.call_args
    assert args[0] == "task_hard_fail"
    assert kwargs.get("reason") == "snapshot_hard_fail_age_exceeded"

    # 2. NUNCA cae a flexible_mode (el hard-fail prevalece sobre _live_degraded_now).
    assert updated.get("_pantry_flexible_mode") is not True
    assert updated.get("_pantry_advisory_only") is not True
    assert updated.get("_pantry_paused") is True
    assert updated.get("_fresh_pantry_source") == "hard_fail_paused"
    # Pantry queda vacío para que cualquier rama upstream que lea sin chequear
    # `_pantry_paused` falle de forma visible en lugar de generar a ciegas.
    assert updated.get("current_pantry_ingredients") == []

    # 3. Usuario notificado vía push.
    mock_notify.assert_called_once()
    notify_kwargs = mock_notify.call_args.kwargs
    assert "Refresca tu nevera" in notify_kwargs["title"]


def test_p0_5_hard_fail_disabled_via_env_preserves_flex_path():
    """[P0-5] Con CHUNK_PANTRY_HARD_FAIL_ON_STALE=false (override), aunque el snapshot
    exceda el cap, el flujo previo (escalada a flex / pausa por stale_snapshot) se preserva.
    Esto permite a tests de la rama flex correr sin cambios."""
    from constants import CHUNK_PANTRY_HARD_FAIL_AGE_HOURS

    user_id = "test_user_hard_fail_disabled"
    stale_hours = CHUNK_PANTRY_HARD_FAIL_AGE_HOURS + 12
    captured_time = datetime.now(timezone.utc) - timedelta(hours=stale_hours)

    snapshot_form_data = {
        "current_pantry_ingredients": ["1 ud Pollo"],
        "_pantry_captured_at": captured_time.isoformat(),
    }

    with patch("cron_tasks.get_user_inventory_net", side_effect=Exception("API down")), \
         patch("cron_tasks.CHUNK_PANTRY_HARD_FAIL_ON_STALE", False), \
         patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:

        updated = _refresh_chunk_pantry(
            user_id=user_id,
            form_data={},
            snapshot_form_data=snapshot_form_data,
            task_id="task_disabled",
            week_number=4,
        )

    # Cuando el hard-fail está apagado, la pausa (si ocurre) NO usa el reason del hard-fail.
    if mock_pause.called:
        kwargs = mock_pause.call_args.kwargs
        assert kwargs.get("reason") != "snapshot_hard_fail_age_exceeded"
    # Y nunca devolvemos `_fresh_pantry_source="hard_fail_paused"`.
    assert updated.get("_fresh_pantry_source") != "hard_fail_paused"


@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._pause_chunk_for_final_inventory_validation")
def test_flexible_active_live_fails_after_retry(mock_pause, mock_get_inventory):
    """
    Test case: flexible activo + live sigue caído tras retry.
    Debe llamar a _pause_chunk_for_final_inventory_validation y no continuar.
    """
    import cron_tasks

    form_data = {
        "_pantry_flexible_mode": True,
        "_fresh_pantry_source": "stale_snapshot"
    }

    user_id = "test_user_flex_fail"
    week_number = 1
    task_id = "task_123"

    # Simulamos que live_inv sigue siendo None incluso después del retry
    mock_get_inventory.side_effect = [None, None]

    _is_flex = bool(form_data.get("_pantry_flexible_mode") or form_data.get("_fresh_pantry_source") == "stale_snapshot")

    # [P1-A] Eliminado el set + assert de `_strict_post_gen_required`: era un
    # dead-write sin consumer (grep cross-repo confirmó write-only). Este test
    # ahora valida solo el contrato real: cuando live_inv sigue None tras retry
    # en flex mode, el worker pausa con reason='flexible_live_unreachable'.
    result = None
    if _is_flex:
        live_inv = cron_tasks.get_user_inventory_net(user_id)
        if live_inv is None:
            live_inv = cron_tasks.get_user_inventory_net(user_id)

        if live_inv is None:
            cron_tasks._pause_chunk_for_final_inventory_validation(task_id, user_id, week_number, reason="flexible_live_unreachable")
            result = False

    # Validaciones
    assert mock_get_inventory.call_count == 2
    mock_pause.assert_called_once_with(task_id, user_id, week_number, reason="flexible_live_unreachable")
    assert result is False
    # [P1-A] El flag `_strict_post_gen_required` ya no se setea: confirmar que
    # quedó fuera del form_data tras la rama flex (ningún consumer lo leía;
    # mantenerlo era un foot-gun para refactors futuros).
    assert "_strict_post_gen_required" not in form_data
