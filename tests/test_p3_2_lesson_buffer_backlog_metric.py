"""[P3-2 · 2026-05-07] Tests para la métrica de backlog del buffer lesson_telemetry.

Cierra P3-2 del backlog del audit `project_audit_p0_p1_close_2026_05_07.md`:
`_flush_pending_lesson_telemetry` no emitía métrica observable de
`stats["remaining"]`. Si DB quedaba caída días, el buffer crecía silencioso
hasta el cap FIFO (`CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS`).

Contrato P3-2:
  - Si `remaining >= MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD` → log WARNING
    `[P3-2/BACKLOG-ALERT]`.
  - Independiente del threshold, emitir un row a `pipeline_metrics` con
    `node='_flush_pending_lesson_telemetry'`, `confidence=1.0` si bajo
    threshold y `0.0` si encima, y metadata={flushed, remaining,
    discarded_invalid, threshold}.
  - Si la INSERT a pipeline_metrics falla (DB sigue caída), se silencia
    (debug log) — el path no debe re-buffear ni excepciónar.
"""
import json
import logging
from unittest.mock import patch

import pytest

import cron_tasks


def _seed_buffer(buf_path, n_records: int) -> None:
    """Pre-popula buffer con N records válidos (UUIDs canónicos)."""
    lines = [
        json.dumps({
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "week_number": i, "event": f"evt_{i}",
            "synthesized_count": 0, "queue_count": 0, "metadata": {},
        })
        for i in range(n_records)
    ]
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Threshold breach → log WARNING
# ---------------------------------------------------------------------------
def test_backlog_alert_triggers_when_remaining_above_threshold(tmp_path, monkeypatch, caplog):
    buf_path = tmp_path / "lt.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 1000)
    monkeypatch.setattr("constants.MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 5)
    _seed_buffer(buf_path, 10)

    # DB write para chunk_lesson_telemetry falla → todos pasan a remaining.
    # DB write para pipeline_metrics no debe explotar el flow.
    metric_calls = []

    def _selective(query, params=None, **kwargs):
        if "pipeline_metrics" in query:
            metric_calls.append((query, params))
            return None
        raise ConnectionError("simulated_outage")

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        with patch.object(cron_tasks, "execute_sql_write", side_effect=_selective):
            stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["remaining"] == 10
    assert any("[P3-2/BACKLOG-ALERT]" in rec.message and "remaining=10" in rec.message
               and "threshold=5" in rec.message
               for rec in caplog.records), \
        "Esperado log WARNING [P3-2/BACKLOG-ALERT]"
    # La métrica debe haber sido emitida con confidence=0.0 (encima threshold).
    assert len(metric_calls) == 1
    _, params = metric_calls[0]
    assert params[6] == 0.0, f"confidence esperada 0.0 (above threshold), got {params[6]}"


# ---------------------------------------------------------------------------
# 2. Below threshold → NO log WARNING, métrica con confidence=1.0
# ---------------------------------------------------------------------------
def test_no_backlog_alert_when_remaining_below_threshold(tmp_path, monkeypatch, caplog):
    buf_path = tmp_path / "lt.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 1000)
    monkeypatch.setattr("constants.MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 100)
    _seed_buffer(buf_path, 3)

    metric_calls = []

    def _selective(query, params=None, **kwargs):
        if "pipeline_metrics" in query:
            metric_calls.append((query, params))
            return None
        # chunk_lesson_telemetry: 2 succeed, 1 fail (transient)
        if len(metric_calls) == 0 and "chunk_lesson_telemetry" in query:
            # All succeed — remaining = 0
            return None
        return None

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        with patch.object(cron_tasks, "execute_sql_write", side_effect=_selective):
            stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["remaining"] == 0
    assert not any("[P3-2/BACKLOG-ALERT]" in rec.message for rec in caplog.records), \
        "NO debe loggear backlog alert por debajo del threshold"
    # La métrica se emite igualmente con confidence=1.0
    assert len(metric_calls) == 1
    _, params = metric_calls[0]
    assert params[6] == 1.0, f"confidence esperada 1.0 (below threshold), got {params[6]}"


# ---------------------------------------------------------------------------
# 3. Métrica best-effort: si pipeline_metrics INSERT falla, no se propaga.
# ---------------------------------------------------------------------------
def test_metric_emit_is_best_effort_no_exception_on_db_failure(tmp_path, monkeypatch, caplog):
    buf_path = tmp_path / "lt.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 1000)
    monkeypatch.setattr("constants.MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 500)
    _seed_buffer(buf_path, 2)

    def _all_fail(*_a, **_kw):
        raise ConnectionError("DB completamente caída")

    with caplog.at_level(logging.DEBUG, logger="cron_tasks"):
        with patch.object(cron_tasks, "execute_sql_write", side_effect=_all_fail):
            # No debe levantar excepción, debe devolver stats normalmente.
            stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["remaining"] == 2
    assert any("[P3-2/METRIC]" in rec.message and "DB degradada" in rec.message
               for rec in caplog.records), \
        "Esperado debug log que silencia el fallo de la métrica"


# ---------------------------------------------------------------------------
# 4. Metadata de la métrica incluye los 4 campos contractuales
# ---------------------------------------------------------------------------
def test_metric_metadata_contains_required_fields(tmp_path, monkeypatch):
    buf_path = tmp_path / "lt.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 1000)
    monkeypatch.setattr("constants.MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 7)
    _seed_buffer(buf_path, 4)

    metric_params = []

    def _selective(query, params=None, **kwargs):
        if "pipeline_metrics" in query:
            metric_params.append(params)
            return None
        raise ConnectionError("transient")  # → 4 a remaining

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_selective):
        cron_tasks._flush_pending_lesson_telemetry()

    assert len(metric_params) == 1
    params = metric_params[0]
    # Schema pipeline_metrics: (user_id, session_id, node, duration_ms,
    # retries, tokens_estimated, confidence, metadata)
    assert params[0] is None  # user_id
    assert params[1] == "__cron__"
    assert params[2] == "_flush_pending_lesson_telemetry"
    metadata = json.loads(params[7])
    assert metadata == {
        "flushed": 0,
        "remaining": 4,
        "discarded_invalid": 0,
        "threshold": 7,
    }


# ---------------------------------------------------------------------------
# 5. Si nada cambió (buffer vacío), NO se emite métrica
# ---------------------------------------------------------------------------
def test_no_metric_when_buffer_is_empty_or_absent(tmp_path, monkeypatch):
    buf_path = tmp_path / "lt_does_not_exist.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 500)

    metric_calls = []

    def _selective(query, params=None, **kwargs):
        if "pipeline_metrics" in query:
            metric_calls.append(params)
        return None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_selective):
        stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats == {"flushed": 0, "remaining": 0, "discarded_invalid": 0}
    assert metric_calls == [], "buffer ausente NO debe emitir métrica spam"
