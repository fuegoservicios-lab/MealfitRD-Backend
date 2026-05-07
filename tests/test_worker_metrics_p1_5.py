"""
[P1-5] Tests del wrapper `_with_worker_metrics` y del estado in-memory
`_LAST_WORKER_RUN` que el endpoint /chunk-queue-health expone.

Antes: el worker corría cada minuto sin emitir log de duración. Si tardaba más
del intervalo (overlap) el problema se descubría solo cuando un usuario reportaba
plan atrasado. Ahora cada corrida emite:
  - INFO: chunk_queue_run_complete con duration, pending_before/after, delta.
  - WARNING: worker_overlapping si duration > 80% del intervalo.
Y el estado queda en `_LAST_WORKER_RUN` para que el endpoint admin lo lea.

Valida:
  A. Wrapper mide duración y popula _LAST_WORKER_RUN.
  B. WARNING emitido si duration > 0.8 * interval.
  C. INFO en lugar de WARNING si duration < 0.8 * interval.
  D. target_plan_id != None bypassa las métricas (ejecución manual admin).
  E. Excepción del impl no rompe la persistencia de la métrica (try/finally).
"""
import sys
import time
from unittest.mock import patch, MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def test_a_wrapper_populates_last_worker_run_state():
    """[P1-5] Tras una corrida, _LAST_WORKER_RUN refleja duración y backlog."""
    pending_values = iter([5, 3])  # antes, después → delta=-2 (drenó 2)

    def fake_count():
        return next(pending_values)

    impl_called = {"n": 0}
    def fake_impl(target_plan_id=None):
        impl_called["n"] += 1
        # [P0-5] 100ms sleep + 80ms assertion threshold leaves slack for the
        # Windows clock-tick granularity (~15.6ms). The previous 50ms/50ms pair
        # tripped on machines where the wrapper's start_at→end_at delta could be
        # ~47ms even when `time.sleep(0.05)` returned, and the test failed flaky.
        time.sleep(0.1)  # forzar duración mensurable

    # Reset para test aislado
    import cron_tasks
    cron_tasks._LAST_WORKER_RUN.update({k: None for k in cron_tasks._LAST_WORKER_RUN})

    with patch("cron_tasks._count_pending_chunks_due", side_effect=fake_count):
        wrapped = cron_tasks._with_worker_metrics(fake_impl)
        wrapped()

    state = cron_tasks._LAST_WORKER_RUN
    assert impl_called["n"] == 1
    assert state["duration_seconds"] is not None
    assert state["duration_seconds"] >= 0.08
    assert state["pending_before"] == 5
    assert state["pending_after"] == 3
    assert state["interval_seconds"] is not None


def test_b_warning_emitted_when_duration_exceeds_80_percent_interval():
    """[P1-5] WARNING worker_overlapping si duration > 80% del intervalo."""
    import cron_tasks
    from constants import CHUNK_SCHEDULER_INTERVAL_MINUTES
    # Forzar duración alta: 0.85 * intervalo
    threshold_s = CHUNK_SCHEDULER_INTERVAL_MINUTES * 60 * 0.85
    cron_tasks._LAST_WORKER_RUN.update({k: None for k in cron_tasks._LAST_WORKER_RUN})

    with patch("cron_tasks._count_pending_chunks_due", return_value=0), \
         patch.object(cron_tasks.logger, "warning") as mock_warn, \
         patch.object(cron_tasks.logger, "info") as mock_info:
        cron_tasks._emit_worker_run_metric(threshold_s, 0, 0)

    assert cron_tasks._LAST_WORKER_RUN["overlap_warning"] is True
    # Al menos una llamada a warning con "worker_overlapping"
    warn_msgs = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert any("worker_overlapping" in m for m in warn_msgs), \
        f"WARNING worker_overlapping debe emitirse, got: {warn_msgs}"
    # No debe haber sido INFO
    info_msgs = [str(c.args[0]) for c in mock_info.call_args_list]
    assert not any("chunk_queue_run_complete" in m for m in info_msgs), \
        "Cuando hay overlap, no debe emitirse INFO de éxito en paralelo"


def test_c_info_emitted_when_duration_under_threshold():
    """[P1-5] INFO chunk_queue_run_complete si duration < 80% del intervalo."""
    import cron_tasks
    from constants import CHUNK_SCHEDULER_INTERVAL_MINUTES
    safe_duration_s = CHUNK_SCHEDULER_INTERVAL_MINUTES * 60 * 0.5  # 50% del intervalo
    cron_tasks._LAST_WORKER_RUN.update({k: None for k in cron_tasks._LAST_WORKER_RUN})

    with patch.object(cron_tasks.logger, "warning") as mock_warn, \
         patch.object(cron_tasks.logger, "info") as mock_info:
        cron_tasks._emit_worker_run_metric(safe_duration_s, 10, 5)

    assert cron_tasks._LAST_WORKER_RUN["overlap_warning"] is False
    info_msgs = [str(c.args[0]) for c in mock_info.call_args_list]
    assert any("chunk_queue_run_complete" in m for m in info_msgs)
    # No debe disparar WARNING de overlap
    warn_msgs = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert not any("worker_overlapping" in m for m in warn_msgs)


def test_d_target_plan_id_bypasses_metrics():
    """[P1-5] Si se llama process_plan_chunk_queue('plan-x') (admin manual),
    el wrapper NO invoca _count_pending_chunks_due ni emite métricas — esas
    corridas son de debugging y no deben contaminar el estado del cron."""
    import cron_tasks
    cron_tasks._LAST_WORKER_RUN.update({k: None for k in cron_tasks._LAST_WORKER_RUN})

    impl_calls = []
    def fake_impl(target_plan_id=None):
        impl_calls.append(target_plan_id)

    count_calls = {"n": 0}
    def fake_count():
        count_calls["n"] += 1
        return 0

    with patch("cron_tasks._count_pending_chunks_due", side_effect=fake_count):
        wrapped = cron_tasks._with_worker_metrics(fake_impl)
        wrapped(target_plan_id="plan-x")

    assert impl_calls == ["plan-x"], "El impl debe ejecutarse"
    assert count_calls["n"] == 0, "Bypass: no contar pending en runs manuales"
    # Estado in-memory NO debe cambiar
    assert cron_tasks._LAST_WORKER_RUN["duration_seconds"] is None


def test_e_impl_exception_still_emits_metric():
    """[P1-5] try/finally garantiza que la métrica se emita aunque el body explote."""
    import cron_tasks
    cron_tasks._LAST_WORKER_RUN.update({k: None for k in cron_tasks._LAST_WORKER_RUN})

    def fake_impl(target_plan_id=None):
        raise RuntimeError("simulated worker crash")

    with patch("cron_tasks._count_pending_chunks_due", return_value=7):
        wrapped = cron_tasks._with_worker_metrics(fake_impl)
        try:
            wrapped()
        except RuntimeError:
            pass  # se espera que propague

    # El estado debe haberse actualizado a pesar del crash
    assert cron_tasks._LAST_WORKER_RUN["duration_seconds"] is not None
    assert cron_tasks._LAST_WORKER_RUN["pending_before"] == 7
    assert cron_tasks._LAST_WORKER_RUN["pending_after"] == 7
