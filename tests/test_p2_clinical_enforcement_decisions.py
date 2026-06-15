"""[P2-CLINICAL-LAYER-CONSUMER + decisiones G9/G18 · 2026-06-15] Batch B del gap-audit.

G8: `_clinical_layer_incomplete` deja de ser dead-write → cuando la capa clínica corrió incompleta para un
perfil con condición/alergia, marca _quality_degraded (banner) + emite system_alert.

G9/G18: decisiones de producto documentadas (HTA/dislipidemia = sub-based + advisory, NO techo duro;
planes no-críticos fallidos se entregan con banner, NO fallback). Ancladas para que un refactor no asuma
otro comportamiento.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DECISIONS_DOC = _BACKEND / "docs" / "clinical_enforcement_decisions.md"


# ── G8: consumidor de _clinical_layer_incomplete ──
def _plan_incomplete():
    return {"_clinical_layer_incomplete": True, "_clinical_layer_incomplete_reason": "renal_cap_unresolved_protein"}


def test_g8_marks_degraded_for_profile_with_condition(monkeypatch):
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: None)  # sin DB
    plan = _plan_incomplete()
    marked = go._maybe_mark_clinical_layer_incomplete_degraded(plan, {"medicalConditions": ["enfermedad renal"]}, False)
    assert marked is True
    assert plan["_quality_degraded"] is True
    assert plan["_quality_degraded_reason"] == "clinical_layer_incomplete"
    assert plan["_quality_degraded_severity"] == "high"


def test_g8_noop_without_condition_or_allergy(monkeypatch):
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: None)
    plan = _plan_incomplete()
    # Sin condición/alergia real, la capa incompleta es inocua → no marca.
    assert go._maybe_mark_clinical_layer_incomplete_degraded(plan, {"medicalConditions": ["Ninguna"]}, False) is False
    assert "_quality_degraded" not in plan


def test_g8_noop_when_flag_absent(monkeypatch):
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: None)
    plan = {}
    assert go._maybe_mark_clinical_layer_incomplete_degraded(plan, {"medicalConditions": ["diabetes"]}, False) is False


def test_g8_noop_for_fallback(monkeypatch):
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: None)
    plan = _plan_incomplete()
    assert go._maybe_mark_clinical_layer_incomplete_degraded(plan, {"medicalConditions": ["enfermedad renal"]}, True) is False


def test_g8_does_not_override_worse_reason(monkeypatch):
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: None)
    plan = _plan_incomplete()
    plan["_quality_degraded"] = True
    plan["_quality_degraded_reason"] = "critical"
    assert go._maybe_mark_clinical_layer_incomplete_degraded(plan, {"medicalConditions": ["enfermedad renal"]}, False) is False
    assert plan["_quality_degraded_reason"] == "critical"


def test_g8_invoked_in_scoring_path():
    assert "_maybe_mark_clinical_layer_incomplete_degraded(plan, actual_form_data, delivered_was_fallback)" in _SRC, (
        "El consumidor G8 debe invocarse en el bloque de scoring (junto a band-gate/panel-degraded)."
    )


# ── G9/G18: decisiones documentadas + ancla de código ──
def test_decisions_doc_exists_and_covers_both():
    assert _DECISIONS_DOC.exists(), "Falta backend/docs/clinical_enforcement_decisions.md (G9/G18)."
    txt = _DECISIONS_DOC.read_text(encoding="utf-8")
    assert "## G9" in txt and "## G18" in txt, "El doc debe cubrir G9 y G18."
    assert "SUSTITUCIÓN" in txt and "techo duro" in txt, "G9: enforcement por sustitución, no techo duro."
    assert "banner" in txt and "fallback" in txt, "G18: entrega con banner vs fallback."


def test_g9_code_matches_decision():
    """HTA/dislipidemia enforced via sustitución (no techo duro que rechace)."""
    assert "_apply_condition_substitutions" in _SRC, (
        "G9: el enforcement HTA/dislipidemia es por `_apply_condition_substitutions` (sub-based)."
    )


def test_g18_code_matches_decision():
    """Crítico → fallback; no-crítico → entregado con banner."""
    assert "needs_critical_fallback" in _SRC, "G18: los rechazos críticos caen a fallback."
    assert "_review_failed_but_delivered" in _SRC, "G18: los no-críticos se entregan marcados con banner."
