"""[P0-3-COHERENCE-WATCHDOG · 2026-05-10] Regression guard: liveness check
del watchdog horario `_aggregate_coherence_block_history_metrics`.

Bug observado en el audit 2026-05-10:
    `pipeline_metrics` registró sólo 1 fila de
    `_aggregate_coherence_block_history_metrics` en 7 días. El cron estaba
    registrado y se invocaba, pero su query interna golpeaba
    `meal_plans.updated_at` (columna inexistente) → PostgREST devolvía 400
    silenciosamente y la función retornaba sin INSERT. P0-OBS-1 arregló el
    filtro a `created_at`. Este nuevo cron (P0-3) protege contra el siguiente
    modo de fallo equivalente — drift de schema, error nuevo, blip persistente
    en pipeline_metrics — sin esperar 7 días para que alguien lo note.

Cobertura de este test:
    1. Sin filas en ventana → emite alert `coherence_watchdog_silent`.
    2. Con fila en ventana → resuelve alerta previa (UPDATE resolved_at) y
       NO crea fila nueva.
    3. Knob `MEALFIT_COHERENCE_WATCHDOG_SILENT_THRESHOLD_H` se pasa al
       parámetro SQL.
    4. Threshold <1 cae al default de 2h.
    5. SELECT falla → no crash, no emite.
    6. INSERT falla → no crash (best-effort).
    7. Cron registrado con id correcto y trigger interval.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Sin filas → emite alerta
# ---------------------------------------------------------------------------
class TestSilentDetection:
    def test_no_rows_in_window_emits_alert(self):
        from cron_tasks import _alert_coherence_watchdog_silent

        write_mock = MagicMock()
        # SELECT devuelve None → ventana vacía.
        with patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_coherence_watchdog_silent()

        assert write_mock.called
        sql = write_mock.call_args[0][0]
        args = write_mock.call_args[0][1]
        assert "INSERT INTO system_alerts" in sql
        assert args[0] == "coherence_watchdog_silent"
        assert "MUDO" in args[1] or "watchdog" in args[1].lower()
        assert "ON CONFLICT" in sql

    def test_row_present_resolves_previous_and_no_new_alert(self):
        """Fila presente en ventana → UPDATE resolved_at, sin nuevo INSERT."""
        from cron_tasks import _alert_coherence_watchdog_silent

        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value={"?column?": 1}), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_coherence_watchdog_silent()

        # Llamó a write para UPDATE resolved_at, no para INSERT.
        assert write_mock.called
        sql = write_mock.call_args[0][0]
        assert "UPDATE system_alerts" in sql
        assert "resolved_at" in sql
        assert "INSERT INTO system_alerts" not in sql

    def test_row_present_resolve_failure_swallowed(self):
        """Si UPDATE resolved_at falla, no debe crashear ni intentar INSERT."""
        from cron_tasks import _alert_coherence_watchdog_silent

        with patch("cron_tasks.execute_sql_query", return_value={"v": 1}), \
             patch("cron_tasks.execute_sql_write", side_effect=RuntimeError("update boom")):
            _alert_coherence_watchdog_silent()  # debe no levantar


# ---------------------------------------------------------------------------
# 2. Knobs: threshold y SQL params
# ---------------------------------------------------------------------------
class TestKnobs:
    def test_threshold_passed_to_sql_param(self, monkeypatch):
        """`MEALFIT_COHERENCE_WATCHDOG_SILENT_THRESHOLD_H=4` → param SQL = 4."""
        from cron_tasks import _alert_coherence_watchdog_silent

        monkeypatch.setenv("MEALFIT_COHERENCE_WATCHDOG_SILENT_THRESHOLD_H", "4")

        query_mock = MagicMock(return_value=None)
        with patch("cron_tasks.execute_sql_query", query_mock), \
             patch("cron_tasks.execute_sql_write"):
            _alert_coherence_watchdog_silent()

        params = query_mock.call_args[0][1]
        assert params[0] == 4

    def test_threshold_below_1_clamped_to_default(self, monkeypatch):
        """Threshold ≤0 → cae al default 2."""
        from cron_tasks import _alert_coherence_watchdog_silent

        monkeypatch.setenv("MEALFIT_COHERENCE_WATCHDOG_SILENT_THRESHOLD_H", "0")

        query_mock = MagicMock(return_value=None)
        with patch("cron_tasks.execute_sql_query", query_mock), \
             patch("cron_tasks.execute_sql_write"):
            _alert_coherence_watchdog_silent()

        params = query_mock.call_args[0][1]
        assert params[0] == 2

    def test_query_filters_correct_node(self):
        """SELECT filtra por `node='_aggregate_coherence_block_history_metrics'`."""
        from cron_tasks import _alert_coherence_watchdog_silent

        query_mock = MagicMock(return_value=None)
        with patch("cron_tasks.execute_sql_query", query_mock), \
             patch("cron_tasks.execute_sql_write"):
            _alert_coherence_watchdog_silent()

        sql = query_mock.call_args[0][0]
        assert "_aggregate_coherence_block_history_metrics" in sql
        assert "pipeline_metrics" in sql


# ---------------------------------------------------------------------------
# 3. Robustez
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_select_failure_returns_silently(self):
        """SELECT lanza → no crash, no emite alerta."""
        from cron_tasks import _alert_coherence_watchdog_silent

        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", side_effect=RuntimeError("db down")), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_coherence_watchdog_silent()  # debe no levantar

        assert not write_mock.called

    def test_insert_failure_does_not_crash(self):
        """Si INSERT falla, loguea pero no crashea el cron."""
        from cron_tasks import _alert_coherence_watchdog_silent

        with patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", side_effect=RuntimeError("write boom")):
            _alert_coherence_watchdog_silent()  # debe no levantar


# ---------------------------------------------------------------------------
# 4. Cron registrado
# ---------------------------------------------------------------------------
def test_cron_registered_with_correct_id():
    from cron_tasks import register_plan_chunk_scheduler, _alert_coherence_watchdog_silent

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    job_ids = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
    assert "alert_coherence_watchdog_silent" in job_ids

    liveness_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_coherence_watchdog_silent"
    )
    # [test fix · P2-CRON-CORRELATION] _add_job_jittered envuelve la func del cron
    # en `_corr_wrapped` (scope de correlation_id, default ON) vía functools.wraps,
    # que setea __wrapped__ → la func que llega a add_job es el wrapper, no la bare
    # function. Desenvolver con __wrapped__ recupera la identidad exacta de prod.
    scheduled_fn = getattr(liveness_call.args[0], "__wrapped__", liveness_call.args[0])
    assert scheduled_fn is _alert_coherence_watchdog_silent
    assert liveness_call.args[1] == "interval"
    assert "minutes" in liveness_call.kwargs
    # Default 60 min, clamp <15 → 60.
    assert liveness_call.kwargs["minutes"] >= 15


def test_interval_clamped_when_too_low(monkeypatch):
    """Knob `MEALFIT_COHERENCE_WATCHDOG_LIVENESS_INTERVAL_MIN=5` cae al default 60."""
    from cron_tasks import register_plan_chunk_scheduler

    monkeypatch.setenv("MEALFIT_COHERENCE_WATCHDOG_LIVENESS_INTERVAL_MIN", "5")

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    liveness_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_coherence_watchdog_silent"
    )
    assert liveness_call.kwargs["minutes"] == 60
