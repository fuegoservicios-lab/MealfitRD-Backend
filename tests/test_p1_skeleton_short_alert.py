"""[P1-SKELETON-SHORT-ALERT · 2026-07-29] Adversarial review of the P1-FALLBACK-CAUSE-SPLIT fix
round found that the `_partial_repair`/`_fallback_source`/`_repair_stats` flags and the
`❌ [SKELETON-SHORT]` log line were never wired into `system_alerts` or `pipeline_metrics` — the
codebase's actual paging/monitoring layer. A recurrence (e.g. a prompt/provider regression that
truncates `plan_skeleton.days` on EVERY generation) would have been invisible to any automated
alerting path; the only way to discover it was a human deciding to grep raw production logs —
exactly how the original incident (corr=23c65543) was found.

Fix: `_persist_skeleton_short_repair_alert` (graph_orchestrator.py), wired into BOTH branches of
`_apply_final_defense_guardrails` that currently set `_fallback_source` after a short-skeleton
repair (`guardrail_partial_repair` and `guardrail_all_synthetic`). Same pattern as the existing
`_persist_pipeline_crash_alert` (P2-PIPELINE-CRASH-NO-ALERT): `INSERT INTO system_alerts` with a
GLOBAL `alert_key` (per `fallback_source`, not per-plan) so N occurrences collapse into ONE row an
SRE actually sees, instead of N spam rows.
"""
from __future__ import annotations

import pathlib

import graph_orchestrator as g


_NUTR = {"target_calories": 2000, "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60}}
_FORM = {"mainGoal": "Salud General", "user_id": "user-abc-123"}


def _valid_day(n=1):
    return {
        "day": n,
        "meals": [{"meal": "Desayuno", "name": "Avena con leche", "ingredients": ["1 taza Avena"], "cals": 300}],
    }


class _RecordingExecuteSqlWrite:
    """Captura cada llamada a `execute_sql_write` sin tocar Neon."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, params=None, *a, **kw):
        self.calls.append((query, params))
        return True


# ---------------------------------------------------------------------------
# `_persist_skeleton_short_repair_alert`: severidad + metadata + idempotencia SQL
# ---------------------------------------------------------------------------

def test_alert_severity_critical_when_zero_real_days(monkeypatch):
    recorder = _RecordingExecuteSqlWrite()
    monkeypatch.setattr("db_core.execute_sql_write", recorder)
    g._persist_skeleton_short_repair_alert(
        "user-1", "plan-1",
        {"real_days": 0, "requested_days": 3, "replaced_count": 3, "filled_count": 0},
        "guardrail_all_synthetic",
    )
    assert len(recorder.calls) == 1
    query, params = recorder.calls[0]
    assert "INSERT INTO system_alerts" in query
    assert "ON CONFLICT (alert_key) DO UPDATE" in query
    alert_key, severity = params[0], params[1]
    assert alert_key == "skeleton_short_repair:guardrail_all_synthetic"
    assert severity == "critical"


def test_alert_severity_warning_when_real_days_survive(monkeypatch):
    recorder = _RecordingExecuteSqlWrite()
    monkeypatch.setattr("db_core.execute_sql_write", recorder)
    g._persist_skeleton_short_repair_alert(
        "user-1", "plan-1",
        {"real_days": 1, "requested_days": 3, "replaced_count": 0, "filled_count": 2},
        "guardrail_partial_repair",
    )
    assert len(recorder.calls) == 1
    _query, params = recorder.calls[0]
    alert_key, severity = params[0], params[1]
    assert alert_key == "skeleton_short_repair:guardrail_partial_repair"
    assert severity == "warning"


def test_alert_metadata_carries_repair_stats_and_plan_id(monkeypatch):
    recorder = _RecordingExecuteSqlWrite()
    monkeypatch.setattr("db_core.execute_sql_write", recorder)
    g._persist_skeleton_short_repair_alert(
        "user-42", "plan-99",
        {"real_days": 1, "requested_days": 3, "replaced_count": 0, "filled_count": 2},
        "guardrail_partial_repair",
    )
    _query, params = recorder.calls[0]
    metadata_json = params[4]
    assert '"real_days": 1' in metadata_json
    assert '"requested_days": 3' in metadata_json
    assert '"plan_id": "plan-99"' in metadata_json
    affected_users_json = params[5]
    assert "user-42" in affected_users_json


def test_alert_key_defaults_to_unknown_when_fallback_source_missing(monkeypatch):
    recorder = _RecordingExecuteSqlWrite()
    monkeypatch.setattr("db_core.execute_sql_write", recorder)
    g._persist_skeleton_short_repair_alert("user-1", "plan-1", {"real_days": 0, "requested_days": 2}, None)
    _query, params = recorder.calls[0]
    assert params[0] == "skeleton_short_repair:unknown"


def test_alert_is_best_effort_never_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("Neon down")
    monkeypatch.setattr("db_core.execute_sql_write", _boom)
    # No debe propagar — el guardrail no debe abortar la entrega del plan por un
    # fallo del emit de la alerta (best-effort, igual que `_persist_pipeline_crash_alert`).
    g._persist_skeleton_short_repair_alert("user-1", "plan-1", {"real_days": 1, "requested_days": 3}, "x")


# ---------------------------------------------------------------------------
# Wiring: `_apply_final_defense_guardrails` invoca el helper en ambas ramas
# ---------------------------------------------------------------------------

def test_guardrail_invokes_alert_on_partial_repair(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)  # aislar de Neon
    calls = []
    monkeypatch.setattr(
        g, "_persist_skeleton_short_repair_alert",
        lambda user_id, plan_id, rstats, fallback_source: calls.append(
            (user_id, plan_id, rstats, fallback_source)
        ),
    )
    plan = {"days": [_valid_day(1)]}
    final_state = {"plan_result": plan, "review_passed": True}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    assert len(calls) == 1
    user_id, _plan_id, rstats, fallback_source = calls[0]
    assert user_id == "user-abc-123"
    assert fallback_source == "guardrail_partial_repair"
    assert rstats["real_days"] == 1


def test_guardrail_invokes_alert_on_all_synthetic(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)
    calls = []
    monkeypatch.setattr(
        g, "_persist_skeleton_short_repair_alert",
        lambda user_id, plan_id, rstats, fallback_source: calls.append(fallback_source),
    )
    plan = {"days": []}
    final_state = {"plan_result": plan, "review_passed": False}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    assert calls == ["guardrail_all_synthetic"]


def test_alert_helper_source_present_and_documented():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-SKELETON-SHORT-ALERT" in src
    assert "def _persist_skeleton_short_repair_alert" in src
    assert src.count("_persist_skeleton_short_repair_alert(") >= 3, (
        "esperado: 1 definición + 2 callsites (guardrail_partial_repair / guardrail_all_synthetic)"
    )


def test_alert_key_documented_in_system_alerts_table():
    """Cross-link con el test drift-detector P2-AUDIT-4 (docs/system_alerts_resolution_table.md)."""
    doc = (pathlib.Path(g.__file__).resolve().parent / "docs" / "system_alerts_resolution_table.md")
    src = doc.read_text(encoding="utf-8")
    assert "skeleton_short_repair" in src
    assert "_persist_skeleton_short_repair_alert" in src
