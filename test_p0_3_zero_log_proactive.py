"""[P0-3] Tests para detección proactiva de zero-log al borde de chunk.

Antes el sistema solo detectaba zero-log REACTIVAMENTE: el worker levantaba el
chunk N+1, computaba `learning_ready`, confirmaba `zero_log=True`, y difería
el chunk hasta CHUNK_LEARNING_READY_MAX_DEFERRALS×CHUNK_LEARNING_READY_DELAY_HOURS
(2-4h de retraso silencioso) antes de pausar. Durante ese tiempo
`_synthesize_last_chunk_learning_from_plan_days` producía un stub vacío que el
LLM podía interpretar como "no hubo violaciones".

Ahora `_enqueue_plan_chunk` prueba al ENQUEUE: si en los últimos `days_count`
días no hubo logs explícitos NI mutaciones de inventario, marca el chunk con
`_zero_log_proactive_detected=True` + `_learning_ready_deferrals=MAX` para que
el worker, al picking, salte deferrals y pause inmediatamente con
`learning_zero_logs`.

Casos:
  1. Helper _detect_proactive_zero_log_at_boundary: detecta zero-log puro,
     ignora si hay logs / mutaciones, returns None on query error.
  2. _enqueue_plan_chunk con week_number > 1 + zero-log → flag persistido en
     plan_chunk_queue.
  3. _enqueue_plan_chunk con week_number == 1 → no probe (initial chunk).
  4. _enqueue_plan_chunk con chunk_kind == "initial_plan" → no probe.
  5. _enqueue_plan_chunk con CHUNK_ZERO_LOG_PROACTIVE_DETECTION=False →
     comportamiento legacy.
  6. _enqueue_plan_chunk cuando la probe encuentra logs → no flag (graceful).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))


# ---------- Helper: _detect_proactive_zero_log_at_boundary ----------

@patch("cron_tasks.execute_sql_query")
def test_detector_returns_signal_when_logs_and_mutations_zero(mock_query):
    import cron_tasks

    mock_query.return_value = {"log_count": 0}
    with patch("db_inventory.get_inventory_activity_since") as mock_inv:
        mock_inv.return_value = {"consumption_mutations_count": 0}
        result = cron_tasks._detect_proactive_zero_log_at_boundary(
            user_id="u-zl-1",
            meal_plan_id="plan-zl-1",
            lookback_days=3,
        )
    assert result is not None
    assert result["log_count"] == 0
    assert result["consumption_mutations_count"] == 0
    assert result["lookback_days"] == 3
    assert "probed_since" in result


@patch("cron_tasks.execute_sql_query")
def test_detector_returns_none_when_logs_present(mock_query):
    import cron_tasks
    mock_query.return_value = {"log_count": 5}
    with patch("db_inventory.get_inventory_activity_since") as mock_inv:
        mock_inv.return_value = {"consumption_mutations_count": 0}
        result = cron_tasks._detect_proactive_zero_log_at_boundary(
            "u-zl-2", "plan-zl-2", 3
        )
    assert result is None, "logs > 0 debe descartar detección de zero-log"


@patch("cron_tasks.execute_sql_query")
def test_detector_returns_none_when_mutations_present(mock_query):
    import cron_tasks
    mock_query.return_value = {"log_count": 0}
    with patch("db_inventory.get_inventory_activity_since") as mock_inv:
        mock_inv.return_value = {"consumption_mutations_count": 4}
        result = cron_tasks._detect_proactive_zero_log_at_boundary(
            "u-zl-3", "plan-zl-3", 3
        )
    assert result is None, "mutations > 0 debe descartar detección de zero-log"


@patch("cron_tasks.execute_sql_query")
def test_detector_conservative_on_query_error(mock_query):
    """Si la query de logs falla, retornamos None — no falsos positivos."""
    import cron_tasks
    mock_query.side_effect = Exception("DB blip")
    result = cron_tasks._detect_proactive_zero_log_at_boundary(
        "u-zl-4", "plan-zl-4", 3
    )
    assert result is None


@patch("cron_tasks.execute_sql_query")
def test_detector_conservative_on_inventory_error(mock_query):
    """Si get_inventory_activity_since falla, retornamos None."""
    import cron_tasks
    mock_query.return_value = {"log_count": 0}
    with patch("db_inventory.get_inventory_activity_since", side_effect=Exception("inv blip")):
        result = cron_tasks._detect_proactive_zero_log_at_boundary(
            "u-zl-5", "plan-zl-5", 3
        )
    assert result is None


def test_detector_skips_guest_user():
    import cron_tasks
    result = cron_tasks._detect_proactive_zero_log_at_boundary(
        "guest", "plan-zl-6", 3
    )
    assert result is None


def test_detector_skips_zero_or_negative_lookback():
    import cron_tasks
    assert cron_tasks._detect_proactive_zero_log_at_boundary("u", "p", 0) is None
    assert cron_tasks._detect_proactive_zero_log_at_boundary("u", "p", -3) is None


# ---------- _enqueue_plan_chunk: integración con detector ----------

@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_marks_chunk_when_zero_log_detected(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """Chunk no-inicial con week_number=2 + detector positivo → UPDATE inyecta
    `_zero_log_proactive_detected=True` y `_learning_ready_deferrals=MAX`."""
    import cron_tasks
    from constants import CHUNK_LEARNING_READY_MAX_DEFERRALS

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, -300, "snapshot")
    mock_query.return_value = {"id": "chunk-zl-flag", "status": "pending", "inserted": True}
    mock_detect.return_value = {
        "log_count": 0,
        "consumption_mutations_count": 0,
        "probed_since": start_dt.isoformat(),
        "lookback_days": 3,
    }

    snapshot = {"form_data": {"_plan_start_date": start_dt.isoformat()}}
    # [P0-4] El guard proactivo de pantry corre antes que P0-3. Mockeamos
    # `get_user_inventory_net` para devolver pantry suficiente, así P0-4 no
    # short-circuit y el detector P0-3 sí se llama.
    with patch(
        "db_inventory.get_user_inventory_net",
        return_value=["500g pollo", "300g arroz", "200g brocoli", "1 cebolla"],
    ):
        cron_tasks._enqueue_plan_chunk(
            user_id="u-zl-flag",
            meal_plan_id="plan-zl-flag",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot=snapshot,
            chunk_kind="rolling_refill",
        )

    # Confirmar detector llamado con lookback_days=days_count
    assert mock_detect.called
    detect_kwargs = mock_detect.call_args.kwargs
    assert detect_kwargs["lookback_days"] == 3
    assert detect_kwargs["user_id"] == "u-zl-flag"

    # Confirmar UPDATE con flag y deferrals saturados
    flag_writes = [
        c for c in mock_write.call_args_list
        if c.args
        and "pipeline_snapshot" in c.args[0]
        and "_zero_log_proactive_detected" in str(c.args[1] if len(c.args) > 1 else "")
    ]
    assert flag_writes, (
        f"esperaba UPDATE con _zero_log_proactive_detected; "
        f"recibí: {[c.args for c in mock_write.call_args_list]}"
    )
    payload = json.loads(flag_writes[0].args[1][0])
    assert payload["_zero_log_proactive_detected"] is True
    assert payload["_learning_ready_deferrals"] == CHUNK_LEARNING_READY_MAX_DEFERRALS
    assert payload["form_data"]["_force_variety"] is True
    assert payload["_zero_log_proactive_signal"]["log_count"] == 0


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_for_initial_plan(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """chunk_kind == 'initial_plan' → no probe (todos los chunks de un plan
    nuevo se encolan upfront, no hay ventana previa)."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-init", "status": "pending", "inserted": True}

    cron_tasks._enqueue_plan_chunk(
        user_id="u-init",
        meal_plan_id="plan-init",
        week_number=2,
        days_offset=3,
        days_count=3,
        pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
        chunk_kind="initial_plan",
    )

    assert not mock_detect.called, "initial_plan no debe probar zero-log"


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_for_first_week(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """week_number == 1 → no probe (no hay chunk previo aún)."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-w1", "status": "pending", "inserted": True}

    cron_tasks._enqueue_plan_chunk(
        user_id="u-w1",
        meal_plan_id="plan-w1",
        week_number=1,
        days_offset=0,
        days_count=3,
        pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
        chunk_kind="rolling_refill",
    )

    assert not mock_detect.called


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_when_flag_disabled(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """CHUNK_ZERO_LOG_PROACTIVE_DETECTION=False → comportamiento legacy."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-off", "status": "pending", "inserted": True}

    with patch("cron_tasks.CHUNK_ZERO_LOG_PROACTIVE_DETECTION", False):
        cron_tasks._enqueue_plan_chunk(
            user_id="u-off",
            meal_plan_id="plan-off",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )

    assert not mock_detect.called


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_no_flag_when_detector_returns_none(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """Si el detector retorna None (logs/mutaciones presentes o error de query),
    no inyectar el flag."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-none", "status": "pending", "inserted": True}
    mock_detect.return_value = None

    # [P0-4] Mockear pantry suficiente para que P0-4 no haga short-circuit.
    with patch(
        "db_inventory.get_user_inventory_net",
        return_value=["500g pollo", "300g arroz", "200g brocoli", "1 cebolla"],
    ):
        cron_tasks._enqueue_plan_chunk(
            user_id="u-none",
            meal_plan_id="plan-none",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )

    # El detector SÍ debe haberse llamado (chunk no-inicial)
    assert mock_detect.called
    # Pero NO debe haber UPDATE inyectando flag
    flag_writes = [
        c for c in mock_write.call_args_list
        if c.args
        and "_zero_log_proactive_detected" in str(c.args[1] if len(c.args) > 1 else "")
    ]
    assert not flag_writes


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_when_chunk_reactivated(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """UPSERT que retorna `inserted=False` (reactivación de chunk failed) no
    debe probar — ya tuvo una pasada anterior."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-react", "status": "pending", "inserted": False}

    cron_tasks._enqueue_plan_chunk(
        user_id="u-react",
        meal_plan_id="plan-react",
        week_number=2,
        days_offset=3,
        days_count=3,
        pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
        chunk_kind="rolling_refill",
    )

    assert not mock_detect.called, "reactivación de chunk failed no debe probar"


@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_does_not_probe_when_tz_unresolved_path_taken(
    mock_write, mock_query, mock_resolve, mock_detect
):
    """Si el flujo P0-2 (forced_8am_utc) ya pausó el chunk en pending_user_action,
    el P0-3 no aplica — el chunk no llegará al worker mientras esté pausado."""
    import cron_tasks

    mock_resolve.return_value = (None, 0, "forced_8am_utc")
    mock_query.return_value = {"id": "c-tz", "status": "pending", "inserted": True}

    cron_tasks._enqueue_plan_chunk(
        user_id="u-tz",
        meal_plan_id="plan-tz",
        week_number=2,
        days_offset=3,
        days_count=3,
        pipeline_snapshot={"form_data": {}},
        chunk_kind="rolling_refill",
    )

    assert not mock_detect.called, (
        "cuando P0-2 flipea a tz_unresolved, P0-3 debe saltarse"
    )
