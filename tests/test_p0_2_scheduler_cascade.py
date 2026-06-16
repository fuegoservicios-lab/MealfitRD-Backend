"""[P0-2-SCHEDULER-CASCADE · 2026-05-10] Regression guard: watchdog de
cascada de MISSED en APScheduler (`_alert_scheduler_cascade_missed`).

Bug observado en el audit 2026-05-10:
    `system_alerts` registró 25+ filas `scheduler_missed_*` en últimas 24h
    con bursts simultáneos a las 14:11, 15:07 y 16:49 UTC. El listener
    (`app._scheduler_alert_listener`) emite UN alert por job_id missed,
    pero nadie escalaba la cascada — operador tenía que correr SQL ad-hoc
    para descubrir que el scheduler estaba saturado.

Fix:
    1. Bump defaults: 20→32 workers, 60→180s grace_time (en app.py).
    2. Watchdog `_alert_scheduler_cascade_missed`: cron 30 min lee
       `system_alerts` última hora, cuenta job_ids distintos, emite
       crítica si >=N (default 3).
    3. Endpoint `/admin/cron-health` para diagnóstico en 1 seg.

Cobertura de este test:
    1. Watchdog lee `system_alerts` con filtro correcto (alert_type +
       prefix + ventana lookback).
    2. Cascada detectada → emite alerta crítica con summary correcto.
    3. <umbral → no emite (no falsos positivos).
    4. Sin filas en ventana → no-op.
    5. Knobs respetados (lookback, threshold, interval).
    6. Cron registrado en register_plan_chunk_scheduler con id correcto.
    7. job_id extraído de metadata + fallback a alert_key suffix.
    8. SELECT falla → no crash.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: separar calls a `system_alerts` del tick observable
# `_scheduler_cascade_check_tick` (P2-B-OBS · 2026-05-11). Antes de P2-B,
# `_alert_scheduler_cascade_missed` solo escribía cuando detectaba cascada.
# Tras P2-B, escribe SIEMPRE el tick observable (para distinguir "cron
# corrió sin cascada" de "cron MISSED"). Estos helpers preservan los
# asserts originales (cascada emit / no emit) ignorando el tick.
# ---------------------------------------------------------------------------
def _system_alerts_calls(write_mock: MagicMock) -> list:
    """Filtra calls a `execute_sql_write` que INSERTAN al alert de cascada
    en `system_alerts`.

    [P2-B-OBS · 2026-05-11] No basta con buscar "system_alerts" en el SQL:
    tras el autohealer (P0-NEW-2-AUTOHEAL), el cascade detector también
    invoca `_resolve_stale_scheduler_alerts()` que hace `UPDATE
    system_alerts SET resolved_at = NOW() ... RETURNING alert_key`. Ese
    UPDATE no es el alert de cascada — es el sweep posterior. Solo nos
    interesa el INSERT del alert critical."""
    matches = []
    for c in write_mock.call_args_list:
        if not c.args or not isinstance(c.args[0], str):
            continue
        sql = c.args[0]
        # INSERT INTO system_alerts (el alert de cascada).
        # Excluye UPDATE system_alerts del sweep.
        if "INSERT INTO system_alerts" in sql or "INSERT INTO\n                system_alerts" in sql:
            matches.append(c)
            continue
        # Variante con espacios/saltos de línea distintos: buscar la combinación
        # `INSERT` ... `system_alerts` y descartar `UPDATE system_alerts`.
        if "system_alerts" in sql and "UPDATE" not in sql.upper().split("WHERE")[0]:
            if "INSERT" in sql.upper():
                matches.append(c)
    return matches


def _pipeline_metrics_calls(write_mock: MagicMock, node_substring: str = "") -> list:
    """Filtra calls a `execute_sql_write` que escriben a `pipeline_metrics`.
    Opcional: filtra por substring del node (e.g. `_scheduler_cascade_check_tick`)."""
    matches = []
    for c in write_mock.call_args_list:
        if not c.args or not isinstance(c.args[0], str):
            continue
        if "pipeline_metrics" not in c.args[0]:
            continue
        if node_substring:
            params = c.args[1] if len(c.args) > 1 else None
            if not params or node_substring not in str(params):
                continue
        matches.append(c)
    return matches


# ---------------------------------------------------------------------------
# 1. Detección de cascada — happy path
# ---------------------------------------------------------------------------
class TestCascadeDetection:
    def test_three_distinct_jobs_triggers_alert(self, monkeypatch):
        """3 jobs distintos missed → alerta crítica."""
        from cron_tasks import _alert_scheduler_cascade_missed

        # Forzamos defaults explícitos para aislamiento.
        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "3")
        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_LOOKBACK_HOURS", "1")

        rows = [
            {"alert_key": "scheduler_missed_jobA", "metadata": {"job_id": "jobA"}, "triggered_at": "2026-05-10T15:00:00Z"},
            {"alert_key": "scheduler_missed_jobB", "metadata": {"job_id": "jobB"}, "triggered_at": "2026-05-10T15:00:01Z"},
            {"alert_key": "scheduler_missed_jobC", "metadata": {"job_id": "jobC"}, "triggered_at": "2026-05-10T15:00:02Z"},
        ]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        assert write_mock.called
        # [P2-B-OBS · 2026-05-11] Buscar el call específico a `system_alerts`
        # (no `call_args` único, que ahora puede apuntar al tick observable
        # `_scheduler_cascade_check_tick` o al autoheal pipeline_metrics).
        alert_calls = _system_alerts_calls(write_mock)
        assert len(alert_calls) == 1, (
            f"Esperaba 1 call a system_alerts, recibidos {len(alert_calls)}"
        )
        sql = alert_calls[0].args[0]
        args = alert_calls[0].args[1]
        assert "system_alerts" in sql
        assert args[0] == "scheduler_cascade_missed"
        # Severity literal en VALUES; mensaje incluye conteo.
        assert "Cascada" in args[1]
        assert "3 jobs" in args[2]

    def test_below_threshold_no_alert(self, monkeypatch):
        """2 jobs distintos con umbral=3 → no emite alerta (pero sí tick)."""
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "3")

        rows = [
            {"alert_key": "scheduler_missed_a", "metadata": {"job_id": "a"}, "triggered_at": "x"},
            {"alert_key": "scheduler_missed_b", "metadata": {"job_id": "b"}, "triggered_at": "x"},
        ]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        # [P2-B-OBS] No emite `system_alerts` (espíritu original del test),
        # pero SÍ emite el tick observable a pipeline_metrics.
        assert _system_alerts_calls(write_mock) == []
        assert len(_pipeline_metrics_calls(write_mock, "_scheduler_cascade_check_tick")) == 1

    def test_repeated_same_job_does_not_count_extra(self, monkeypatch):
        """Mismo job_id repetido N veces cuenta como 1 distinct → no emite alerta."""
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "3")

        rows = [
            {"alert_key": "scheduler_missed_a", "metadata": {"job_id": "a"}, "triggered_at": "x"},
            {"alert_key": "scheduler_missed_a", "metadata": {"job_id": "a"}, "triggered_at": "y"},
            {"alert_key": "scheduler_missed_a", "metadata": {"job_id": "a"}, "triggered_at": "z"},
        ]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        # [P2-B-OBS] No emite alerta de cascada (1 distinct < threshold=3)
        # pero el tick observable se emite siempre.
        assert _system_alerts_calls(write_mock) == []

    def test_no_rows_no_op(self):
        """Sin MISSED en ventana → early-return ANTES del tick (no escribe nada).

        Caso degenerate: sin rows, ni siquiera el tick observable corre
        — el early return ocurre antes de calcular `cascade_detected`.
        Esto preserva el ahorro del cron en estado healthy total.
        """
        from cron_tasks import _alert_scheduler_cascade_missed

        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=[]), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        assert not write_mock.called


# ---------------------------------------------------------------------------
# 2. Extracción de job_id (metadata preferido, fallback a alert_key)
# ---------------------------------------------------------------------------
class TestJobIdExtraction:
    def test_uses_metadata_job_id_when_present(self, monkeypatch):
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "2")

        rows = [
            {"alert_key": "scheduler_missed_x", "metadata": {"job_id": "real_job_a"}, "triggered_at": "x"},
            {"alert_key": "scheduler_missed_y", "metadata": {"job_id": "real_job_b"}, "triggered_at": "x"},
        ]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        assert write_mock.called
        # [P2-B-OBS · 2026-05-11] Aislar el call a system_alerts (metadata
        # del cascade alert vive en args[1][3], no en el tick observable).
        alert_calls = _system_alerts_calls(write_mock)
        assert len(alert_calls) == 1
        meta_json = alert_calls[0].args[1][3]
        import json as _j
        meta = _j.loads(meta_json)
        assert "real_job_a" in meta["top_jobs_by_count"]
        assert "real_job_b" in meta["top_jobs_by_count"]

    def test_falls_back_to_alert_key_suffix(self, monkeypatch):
        """Si metadata no tiene job_id, deriva del prefix `scheduler_missed_`."""
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "2")

        rows = [
            {"alert_key": "scheduler_missed_jobA", "metadata": None, "triggered_at": "x"},
            {"alert_key": "scheduler_missed_jobB", "metadata": {}, "triggered_at": "x"},
        ]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        assert write_mock.called
        # [P2-B-OBS] Aislar el call a system_alerts (no el tick observable).
        alert_calls = _system_alerts_calls(write_mock)
        assert len(alert_calls) == 1
        import json as _j
        meta = _j.loads(alert_calls[0].args[1][3])
        assert "jobA" in meta["top_jobs_by_count"]
        assert "jobB" in meta["top_jobs_by_count"]


# ---------------------------------------------------------------------------
# 3. Knobs y SQL del watchdog
# ---------------------------------------------------------------------------
class TestKnobsAndSQL:
    def test_lookback_hours_passed_to_sql(self, monkeypatch):
        """El knob `MEALFIT_SCHEDULER_CASCADE_LOOKBACK_HOURS` pasa como param SQL."""
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_LOOKBACK_HOURS", "6")

        query_mock = MagicMock(return_value=[])
        with patch("cron_tasks.execute_sql_query", query_mock), \
             patch("cron_tasks.execute_sql_write"):
            _alert_scheduler_cascade_missed()

        # El segundo arg de execute_sql_query es la tupla de params.
        params = query_mock.call_args[0][1]
        assert params[0] == 6

    def test_query_filters_alert_type_and_prefix(self):
        """SELECT filtra alert_type='scheduler' AND alert_key LIKE 'scheduler_missed_%'."""
        from cron_tasks import _alert_scheduler_cascade_missed

        query_mock = MagicMock(return_value=[])
        with patch("cron_tasks.execute_sql_query", query_mock), \
             patch("cron_tasks.execute_sql_write"):
            _alert_scheduler_cascade_missed()

        sql = query_mock.call_args[0][0]
        assert "alert_type = 'scheduler'" in sql
        assert "scheduler_missed_" in sql

    def test_threshold_below_2_clamped(self, monkeypatch):
        """Threshold negativo / 0 / 1 cae al mínimo de 2 (no emite alerta con 1 job)."""
        from cron_tasks import _alert_scheduler_cascade_missed

        monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_THRESHOLD", "0")

        rows = [{"alert_key": "scheduler_missed_a", "metadata": {"job_id": "a"}, "triggered_at": "x"}]
        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()

        # [P2-B-OBS] 1 job único < clamp(2) → no emite alerta de cascada.
        # Tick observable se emite siempre (path "cron corrió sin cascada").
        assert _system_alerts_calls(write_mock) == []


# ---------------------------------------------------------------------------
# 4. Robustez: SELECT falla, sin crash
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_select_failure_returns_silently(self):
        from cron_tasks import _alert_scheduler_cascade_missed

        write_mock = MagicMock()
        with patch("cron_tasks.execute_sql_query", side_effect=RuntimeError("db down")), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_scheduler_cascade_missed()  # debe no levantar

        assert not write_mock.called

    def test_write_failure_does_not_crash(self):
        """Si el INSERT falla, loguea pero no crashea el cron."""
        from cron_tasks import _alert_scheduler_cascade_missed

        rows = [
            {"alert_key": f"scheduler_missed_j{i}", "metadata": {"job_id": f"j{i}"}, "triggered_at": "x"}
            for i in range(5)
        ]
        with patch("cron_tasks.execute_sql_query", return_value=rows), \
             patch("cron_tasks.execute_sql_write", side_effect=RuntimeError("write boom")):
            _alert_scheduler_cascade_missed()  # debe no levantar


# ---------------------------------------------------------------------------
# 5. Cron registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------
def test_cascade_cron_registered():
    """`alert_scheduler_cascade_missed` registrado con interval+minutes."""
    from cron_tasks import register_plan_chunk_scheduler, _alert_scheduler_cascade_missed

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    job_ids = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
    assert "alert_scheduler_cascade_missed" in job_ids

    cascade_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_scheduler_cascade_missed"
    )
    # [test fix · P2-CRON-CORRELATION] _add_job_jittered envuelve la func del cron
    # en `_corr_wrapped` (scope de correlation_id, default ON) vía functools.wraps,
    # que preserva __name__ y setea __wrapped__ → la func que llega a add_job NO es
    # la bare function sino el wrapper. Desenvolver con __wrapped__ recupera la
    # identidad exacta (assert sigue siendo fuerte: liga al objeto de prod).
    scheduled_fn = getattr(cascade_call.args[0], "__wrapped__", cascade_call.args[0])
    assert scheduled_fn is _alert_scheduler_cascade_missed
    assert cascade_call.args[1] == "interval"
    assert "minutes" in cascade_call.kwargs
    # Default 30 min + clamp >=5.
    assert cascade_call.kwargs["minutes"] >= 5


def test_interval_clamped_when_too_low(monkeypatch):
    """Knob `MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN=1` cae al default (30)."""
    from cron_tasks import register_plan_chunk_scheduler

    monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN", "1")

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    cascade_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_scheduler_cascade_missed"
    )
    # <5 cae al default 30.
    assert cascade_call.kwargs["minutes"] == 30


# ---------------------------------------------------------------------------
# 6. Defaults bumped en app.py (lectura de source — evita importar app)
# ---------------------------------------------------------------------------
def test_app_py_max_workers_default_is_32():
    """`MEALFIT_SCHEDULER_MAX_WORKERS` default debe ser 32 (no el viejo 20)."""
    from pathlib import Path
    import re

    app_py = Path(__file__).resolve().parent.parent / "app.py"
    text = app_py.read_text(encoding="utf-8")
    m = re.search(r'_env_int\(\s*["\']MEALFIT_SCHEDULER_MAX_WORKERS["\']\s*,\s*(\d+)\s*\)', text)
    assert m is not None, "No se encontró el knob MEALFIT_SCHEDULER_MAX_WORKERS en app.py"
    assert int(m.group(1)) == 32, (
        f"MEALFIT_SCHEDULER_MAX_WORKERS default es {m.group(1)}, esperado 32 "
        f"(P0-2 bump del 2026-05-10)."
    )


def test_app_py_misfire_grace_default_is_180():
    """`MEALFIT_SCHEDULER_MISFIRE_GRACE_S` default debe ser 180 (no el viejo 60)."""
    from pathlib import Path
    import re

    app_py = Path(__file__).resolve().parent.parent / "app.py"
    text = app_py.read_text(encoding="utf-8")
    m = re.search(r'_env_int\(\s*["\']MEALFIT_SCHEDULER_MISFIRE_GRACE_S["\']\s*,\s*(\d+)\s*\)', text)
    assert m is not None, "No se encontró el knob MEALFIT_SCHEDULER_MISFIRE_GRACE_S en app.py"
    assert int(m.group(1)) == 180, (
        f"MEALFIT_SCHEDULER_MISFIRE_GRACE_S default es {m.group(1)}, esperado 180 "
        f"(P0-2 bump del 2026-05-10)."
    )


# ---------------------------------------------------------------------------
# 7. Endpoint /admin/cron-health declarado en app.py
# ---------------------------------------------------------------------------
def test_admin_cron_health_endpoint_declared():
    """app.py declara `@app.get("/admin/cron-health")` con el handler
    `admin_cron_health`. Test estático (no importa app)."""
    from pathlib import Path
    import re

    app_py = Path(__file__).resolve().parent.parent / "app.py"
    text = app_py.read_text(encoding="utf-8")
    assert '@app.get("/admin/cron-health")' in text or "@app.get('/admin/cron-health')" in text
    assert re.search(r"def\s+admin_cron_health\s*\(", text), (
        "Función `admin_cron_health` no declarada en app.py."
    )
