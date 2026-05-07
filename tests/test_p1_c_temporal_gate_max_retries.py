"""[P1-C] Tests para el cap de reintentos del temporal gate.

Cubre:
  1. Chunk con `_temporal_gate_retries < MAX` → ready=False, counter incrementado en BD.
  2. Chunk con `_temporal_gate_retries == MAX` (next > MAX) → ready=True
     forzado, reason='temporal_gate_max_retries_exceeded', telemetría en
     `chunk_deferrals` con esa razón, snapshot reseteado a 0 + timestamp anotado.
  3. Snapshot sin contador (chunk fresco): primer reintento persiste retries=1.
  4. El path que NO entra al temporal_gate (días previos ya elapsados) no incrementa.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_c_temporal_gate_max_retries.py -v
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from cron_tasks import _check_chunk_learning_ready
from constants import CHUNK_TEMPORAL_GATE_MAX_RETRIES


def _base_args(plan_start_dt, retries=0, week_number=2, days_offset=3):
    """Construye args estándar para _check_chunk_learning_ready con un snapshot
    que dispara el temporal gate (now < prev_end_date)."""
    snapshot = {
        "form_data": {
            "_plan_start_date": plan_start_dt.isoformat(),
            "tz_offset_minutes": 0,
        },
        "totalDays": 7,
        "_temporal_gate_retries": retries,
    }
    plan_data = {"days": [{"day": d, "meals": []} for d in range(1, 4)]}
    return {
        "user_id": "u-p1c",
        "meal_plan_id": "plan-p1c",
        "week_number": week_number,
        "days_offset": days_offset,
        "plan_data": plan_data,
        "snapshot": snapshot,
    }


# ---------------------------------------------------------------------------
# 1. Chunk con retries < MAX → ready=False y counter incrementado en BD
# ---------------------------------------------------------------------------
@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks._record_chunk_deferral")
def test_below_cap_increments_counter_and_defers(
    mock_deferral, mock_write, mock_query, mock_now
):
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Día 3 a las 23:00 — temporal gate NO debe permitir el chunk de week=2 todavía.
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True

    # [P1-3] Parametric en CHUNK_TEMPORAL_GATE_MAX_RETRIES para sobrevivir cambios
    # futuros del cap (antes era 20, ahora 5; mañana podría ser otro valor). Tomamos
    # un retries 2 por debajo del cap para dejar margen de incremento sin escalar.
    _retries_below = max(0, int(CHUNK_TEMPORAL_GATE_MAX_RETRIES) - 2)
    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=_retries_below))

    assert result["ready"] is False
    assert result["reason"] == "prev_chunk_day_not_yet_elapsed"
    assert result["temporal_gate_retries"] == _retries_below + 1, (
        f"retries debe incrementarse: {_retries_below} → {_retries_below + 1}"
    )

    # Telemetría con reason="temporal_gate" (no la de override).
    mock_deferral.assert_called_once()
    assert mock_deferral.call_args.kwargs["reason"] == "temporal_gate"

    # Persistencia del counter incrementado en plan_chunk_queue.
    persistence_calls = [
        c for c in mock_write.call_args_list
        if "_temporal_gate_retries" in (c.args[0] if c.args else "")
    ]
    assert len(persistence_calls) >= 1, (
        f"esperaba al menos 1 UPDATE persistiendo el counter; got {persistence_calls}"
    )
    # El value del counter persistido debe ser _retries_below + 1.
    persisted_value = persistence_calls[0].args[1][0]
    assert persisted_value == _retries_below + 1


# ---------------------------------------------------------------------------
# 2. Cap excedido → forzar ready=True con telemetría especial
# ---------------------------------------------------------------------------
@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks._record_chunk_deferral")
def test_cap_exceeded_forces_ready_true_with_override_telemetry(
    mock_deferral, mock_write, mock_query, mock_now
):
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True

    # retries == MAX → next = MAX+1 > MAX → escalation.
    args = _base_args(plan_start_dt, retries=int(CHUNK_TEMPORAL_GATE_MAX_RETRIES))
    result = _check_chunk_learning_ready(**args)

    assert result["ready"] is True, (
        f"con retries=={CHUNK_TEMPORAL_GATE_MAX_RETRIES}, next supera el cap → debe forzar ready=True"
    )
    assert result["reason"] == "temporal_gate_max_retries_exceeded"
    assert result.get("forced_override") is True

    # Telemetría con reason="temporal_gate_max_retries_exceeded".
    mock_deferral.assert_called_once()
    assert mock_deferral.call_args.kwargs["reason"] == "temporal_gate_max_retries_exceeded"

    # Reset del counter y timestamp anotado en snapshot vía jsonb_set anidado.
    reset_calls = [
        c for c in mock_write.call_args_list
        if "_temporal_gate_max_retries_overridden_at" in (c.args[0] if c.args else "")
    ]
    assert len(reset_calls) == 1, (
        f"esperaba 1 UPDATE de reset+timestamp; got {len(reset_calls)}"
    )


# ---------------------------------------------------------------------------
# 3. Snapshot sin contador previo → primer reintento persiste retries=1
# ---------------------------------------------------------------------------
@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks._record_chunk_deferral")
def test_no_prior_counter_persists_one(
    mock_deferral, mock_write, mock_query, mock_now
):
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    mock_now.return_value = plan_start_dt + timedelta(days=2, hours=23)
    mock_query.return_value = None
    mock_deferral.return_value = True

    args = _base_args(plan_start_dt)
    # Eliminar la key para simular chunk fresco.
    args["snapshot"].pop("_temporal_gate_retries", None)
    result = _check_chunk_learning_ready(**args)

    assert result["ready"] is False
    assert result["temporal_gate_retries"] == 1, "snapshot sin key → retries empieza 0, next=1"


# ---------------------------------------------------------------------------
# 4. Path normal (días elapsed) NO incrementa counter
# ---------------------------------------------------------------------------
@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.get_consumed_meals_since", return_value=[])
@patch("cron_tasks.get_inventory_activity_since", return_value={})
def test_elapsed_path_does_not_increment_counter(
    _ma, _mc, mock_deferral, mock_write, mock_query, mock_now
):
    """Si el chunk previo SÍ concluyó (now >= prev_end), pasamos del temporal_gate
    y no debemos persistir el counter incrementado ni grabar telemetría de gate."""
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=4)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Día 5 — chunk previo (D1-D3) ya concluyó hace tiempo.
    mock_now.return_value = plan_start_dt + timedelta(days=4, hours=12)
    mock_query.return_value = None

    result = _check_chunk_learning_ready(**_base_args(plan_start_dt, retries=10))

    # El gate debe haber pasado (cuando pasa, la función no setea `reason` o setea
    # uno distinto al gate-block).
    assert result.get("reason") != "prev_chunk_day_not_yet_elapsed"
    assert result.get("reason") != "temporal_gate_max_retries_exceeded"
    # No debe haber telemetría de temporal_gate.
    gate_deferrals = [
        c for c in mock_deferral.call_args_list
        if c.kwargs.get("reason", "").startswith("temporal_gate")
    ]
    assert gate_deferrals == [], "no debió grabar telemetría de gate cuando el path no entra al gate"
    # No debe haber UPDATE persistiendo _temporal_gate_retries.
    counter_persists = [
        c for c in mock_write.call_args_list
        if "_temporal_gate_retries" in (c.args[0] if c.args else "")
    ]
    assert counter_persists == [], (
        "el counter no debe persistirse cuando no entramos al temporal_gate"
    )
