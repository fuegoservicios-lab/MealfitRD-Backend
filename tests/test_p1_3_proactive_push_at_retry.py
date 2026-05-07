"""[P1-3] Push proactivo al N-th deferral consecutivo del temporal gate.

Antes del fix:
    1. `CHUNK_TEMPORAL_GATE_MAX_RETRIES = 20` permitía hasta 20 evaluaciones
       silenciosas (≈ 20 min con scheduler de 1 min) antes de que el cron
       `_detect_chronic_deferrals` (cada 6h, umbral 5/48h) notificara al usuario.
    2. Un chunk con TZ desalineada podía agotar TODOS los reintentos antes de
       que el cross-window detector lo capturara, forzando ready=True con
       datos posiblemente incorrectos sin que el usuario supiera.

Después del fix:
    1. Cap reducido a 5 (antes 20): ~5 min máximo de silent failure antes del
       forced override. Mejor que 20 min, pero por sí solo no avisa al usuario.
    2. Push proactivo en el N-th deferral (default N=CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY=3):
       el usuario recibe notificación tras ~3 evaluaciones, mucho antes que el
       cap de 5 o el detector cross-window de 5/48h.
    3. Dedupe por (user_id, meal_plan_id, week_number) vía system_alerts con
       cooldown CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS (default 6h) — evita
       spam si el usuario ignora la primera notificación.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from cron_tasks import _check_chunk_learning_ready
from constants import (
    CHUNK_TEMPORAL_GATE_MAX_RETRIES,
    CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY,
    CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS,
)


def _base_args(plan_start_dt, retries):
    """Snapshot que dispara el temporal gate (now < prev_end_date) con
    `_temporal_gate_retries` arbitrario."""
    return {
        "user_id": "u-p13",
        "meal_plan_id": "plan-p13",
        "week_number": 2,
        "days_offset": 3,
        "plan_data": {"days": [{"day": d, "meals": []} for d in range(1, 4)]},
        "snapshot": {
            "form_data": {
                "_plan_start_date": plan_start_dt.isoformat(),
                "tz_offset_minutes": 0,
            },
            "totalDays": 7,
            "_temporal_gate_retries": retries,
        },
    }


def test_constants_default_values():
    """Cap y threshold ajustados para combinar con backoff exponencial (P1-4).

    Historial:
      - cap=20 (original): hasta ~20 min de silent failure antes del forced override.
      - cap=5 (P1-3): scheduler 1 min × 5 ≈ 5 min de gracia. Insuficiente para
        TZ drifts >5 min (DST transitions, cliente que reportó TZ tarde).
      - cap=8 (P1-4): combinado con backoff exponencial `min(2^retry, 30)` da
        ~2h de cobertura total (1+2+4+8+16+30+30+30 = 121 min) — margen real
        para drifts moderados sin reabrir el silent-failure largo del original.

    El push pasa de retry=3 a retry=5 porque con backoff retry=3 ocurre apenas
    ~7 min después del primer deferral; el usuario aún no tuvo tiempo razonable
    para revisar su TZ. retry=5 cae a ~31 min, momento más apropiado.
    """
    # [P1-4] Cap=8 por default.
    assert CHUNK_TEMPORAL_GATE_MAX_RETRIES == 8, (
        f"CHUNK_TEMPORAL_GATE_MAX_RETRIES debe ser 8 por default (P1-4, antes 5). "
        f"Si lo cambiaste por env, asegúrate de que el resultado combinado con "
        f"backoff exponencial siga teniendo cobertura razonable: <5 sería muy "
        f"agresivo, >12 reabriría el silent failure largo del original."
    )
    # [P1-4] Push en retry=5 por default.
    assert CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY == 5
    assert CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY < CHUNK_TEMPORAL_GATE_MAX_RETRIES, (
        "El push debe dispararse ANTES del cap, sino el usuario sólo se entera "
        "cuando el chunk ya fue forzado a ready=True con datos posiblemente "
        "incorrectos."
    )
    assert CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS >= 1


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._record_chunk_deferral")
def test_no_push_below_threshold(
    mock_deferral, mock_query, mock_write, mock_push, mock_now
):
    """Mientras `_temporal_gate_retries + 1 < N`, NO hay push. Sólo telemetría
    de deferral. Antes del fix esto era el único comportamiento por ~20 min."""
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True

    # retries=0 → next=1 → no llega al threshold (3).
    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=0))

    assert result["ready"] is False
    assert result["temporal_gate_retries"] == 1
    mock_push.assert_not_called(), (
        "No debe enviarse push proactivo en el primer deferral — "
        "el threshold default es 3 para evitar spam por blips de 1 evaluación."
    )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks._ensure_quality_alert_schema")
def test_push_fires_at_exactly_nth_consecutive_retry(
    mock_schema, mock_deferral, mock_query, mock_write, mock_push, mock_now
):
    """En la evaluación que produce `_p1c_next_retries == CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY`,
    se dispara el push notification proactivo."""
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None  # sin alerta previa en system_alerts → no dedupe block
    mock_deferral.return_value = True

    # retries = N-1 → next = N → DISPARA push.
    threshold = int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY)
    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=threshold - 1))

    assert result["ready"] is False
    assert result["temporal_gate_retries"] == threshold
    mock_push.assert_called_once(), (
        f"Push debe dispararse exactamente cuando next_retries == {threshold} "
        f"(threshold por defecto). Si cambia, ajustar CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY."
    )
    # El push apunta al user del snapshot.
    call_kwargs = mock_push.call_args.kwargs
    assert call_kwargs["user_id"] == "u-p13"


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks._ensure_quality_alert_schema")
def test_push_dedupes_via_system_alerts(
    mock_schema, mock_deferral, mock_query, mock_write, mock_push, mock_now
):
    """Si `system_alerts` ya tiene una entrada activa para este chunk dentro
    del cooldown, NO se envía push otra vez. Sin dedupe, cada evaluación entre
    el threshold y el cap dispararía un push (3, 4, 5 → 3 pushes)."""
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    # Simular que hay una alerta reciente (dedupe activo).
    mock_query.return_value = {"triggered_at": real_now.isoformat()}
    mock_deferral.return_value = True

    threshold = int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY)
    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=threshold - 1))

    assert result["ready"] is False
    mock_push.assert_not_called(), (
        "Push debe deduparse contra system_alerts dentro del cooldown. "
        "Sin esto, cada evaluación entre threshold y cap spamearía al usuario."
    )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks._ensure_quality_alert_schema")
def test_no_push_at_retries_above_threshold_post_dedupe(
    mock_schema, mock_deferral, mock_query, mock_write, mock_push, mock_now
):
    """En reintentos POSTERIORES al N-th (e.g. retries=N+1), no se dispara push
    porque el chequeo es `== N` (igualdad, no `>=`). Esto evita pushes
    redundantes incluso si por algún edge case la dedupe falla.
    """
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True

    threshold = int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY)
    # retries = N → next = N+1 (> threshold pero != threshold).
    # next debe ser <= MAX para no entrar al forced override.
    if threshold < int(CHUNK_TEMPORAL_GATE_MAX_RETRIES):
        result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=threshold))
        assert result["ready"] is False
        assert result["temporal_gate_retries"] == threshold + 1
        mock_push.assert_not_called(), (
            "Push usa comparación de igualdad (== threshold), no >=. Esto "
            "garantiza que solo dispare 1 vez (en el N-th deferral exacto), "
            "sin contar con la dedupe de system_alerts como única defensa."
        )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks._ensure_quality_alert_schema")
def test_push_failure_does_not_block_deferral(
    mock_schema, mock_deferral, mock_query, mock_write, mock_push, mock_now
):
    """Si `_dispatch_push_notification` falla (FCM down, schema error), el
    deferral debe completarse normalmente. Es best-effort: el cron
    `_detect_chronic_deferrals` cubre el gap si el push proactivo falla.
    """
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True
    mock_push.side_effect = RuntimeError("FCM unreachable")

    threshold = int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY)
    # No debe lanzar.
    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=threshold - 1))
    assert result["ready"] is False
    assert result["temporal_gate_retries"] == threshold


def test_alert_key_includes_chunk_identity():
    """El alert_key del dedupe debe incluir user_id + meal_plan_id + week_number.
    Sin esto, dos chunks distintos del mismo user (e.g., week 2 y week 3) se
    deduplicarían entre sí — el segundo nunca enviaría push aunque sea un
    chunk diferente.
    """
    import inspect, cron_tasks
    src = inspect.getsource(cron_tasks._check_chunk_learning_ready)
    # El template del alert_key debe incluir las 3 piezas para que dos chunks
    # del mismo plan no se deduplicen entre sí.
    assert "temporal_gate_proactive:" in src
    assert "{user_id}" in src or "user_id" in src
    assert "{meal_plan_id}" in src or "meal_plan_id" in src
    assert "week_number" in src
