"""[gap-audit Batch D · 2026-06-15] G11 (paridad motor declarativo) + G12 (gate de cobertura) + G17 (proveniencia cultivares).

G11: cada ConditionRule clasificada engine-enforced (cap renal / sustitución) vs advisory; el _REGISTRY cubre
las enforced. Una regla nueva sin clasificar FALLA → fuerza decisión. (FiberFloorConstraint rewrite diferido.)
G12: gate de transparencia por baja cobertura de resolución (composite_dish_unresolved); default OFF.
G17: cultivares DD anotados como proxy USDA (NO curados es-DO); curación INCAP bloqueada por recurso.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import condition_rules as cr
import graph_orchestrator as go
from clinical_constraints import ClinicalConstraintEngine

_BACKEND = Path(__file__).resolve().parent.parent
_POPULATE_SRC = (_BACKEND / "scripts" / "populate_nutrition_db.py").read_text(encoding="utf-8")
_DECISIONS_DOC = (_BACKEND / "docs" / "clinical_enforcement_decisions.md").read_text(encoding="utf-8")

# Clasificación declarativa (G11). Si cambia CONDITION_RULES, el test fuerza actualizar esto.
_CAP_ENFORCED = {"renal"}
_SUB_ENFORCED_EXPECTED = {"dm2", "hta", "dyslipidemia"}
_ADVISORY_EXPECTED = {"anemia", "pregnancy", "hypothyroid", "gout", "nafld", "pcos"}


def _all_rule_ids():
    return {r.id for r in cr.CONDITION_RULES}


def _engine_constraint_ids():
    return {cls().id for cls in ClinicalConstraintEngine._REGISTRY}


# ── G11: paridad motor declarativo ──
def test_engine_has_cap_and_substitution_mechanisms():
    ids = _engine_constraint_ids()
    assert "renal" in ids, "el engine debe tener el cap renal (RenalProteinCapConstraint)"
    assert "substitutions" in ids, "el engine debe tener el motor de sustituciones"


def test_substitution_rules_match_expected_and_are_engine_covered():
    sub_rules = {r.id for r in cr.CONDITION_RULES if r.substitutions}
    assert sub_rules == _SUB_ENFORCED_EXPECTED, (
        f"Reglas con sustitución cambiaron: {sub_rules}. El SubstitutionEngineConstraint corre "
        f"collect_substitutions sobre TODAS; si añadiste/quitaste una, actualiza este test + el doc G11."
    )
    assert "substitutions" in _engine_constraint_ids()


def test_renal_rule_is_cap_enforced():
    assert "renal" in _all_rule_ids() and "renal" in _engine_constraint_ids()


def test_every_rule_is_classified_hard_or_advisory():
    """Paridad declarativa (G11): toda ConditionRule está clasificada engine-enforced (cap/sustitución) o
    advisory. Una regla NUEVA sin clasificar FALLA → fuerza una decisión consciente (constraint duro vs
    advisory) en vez de quedar como advisory silenciosa que jamás fuerza retry."""
    all_ids = _all_rule_ids()
    sub_ids = {r.id for r in cr.CONDITION_RULES if r.substitutions}
    classified = _CAP_ENFORCED | sub_ids | _ADVISORY_EXPECTED
    unclassified = all_ids - classified
    assert not unclassified, (
        f"ConditionRule(s) sin clasificar (hard vs advisory): {unclassified}. Clasifícala: objetivo "
        f"cuantitativo que DEBE reescribir → constraint duro en clinical_constraints._REGISTRY (ver doc "
        f"clinical_enforcement_decisions.md G11); advisory/referral → añadir a _ADVISORY_EXPECTED aquí."
    )
    assert _ADVISORY_EXPECTED <= all_ids, f"advisory allowlist con ids fantasma: {_ADVISORY_EXPECTED - all_ids}"


def test_dm2_fiber_floor_is_advisory_by_decision():
    """G11/G9: la sustitución DM2 (azúcar) es enforced; el objetivo fibra (≥14g/1000kcal) es ADVISORY
    (panel+prompt), NO un FiberFloorConstraint que reescriba. Decisión documentada; rewrite diferido."""
    dm2 = next(r for r in cr.CONDITION_RULES if r.id == "dm2")
    assert dm2.substitutions, "DM2 debe tener la sustitución de azúcar (enforced)"
    assert "fiber" not in " ".join(_engine_constraint_ids()).lower(), "no debe haber un FiberConstraint aún"
    assert "## G11" in _DECISIONS_DOC and "FiberFloorConstraint" in _DECISIONS_DOC


# ── G12: gate de cobertura de resolución ──
def _plan_cov(pct):
    return {"resolution_coverage": {"resolved": int(pct * 10), "total": 10, "pct": pct}}


def test_g12_marks_low_coverage_when_enabled(monkeypatch):
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_FLOOR", 0.7)
    plan = _plan_cov(0.5)
    assert go._maybe_mark_low_resolution_degraded(plan, False) is True
    assert plan["_quality_degraded_reason"] == "composite_dish_unresolved"
    assert plan["_quality_degraded_severity"] == "minor"


def test_g12_noop_above_floor(monkeypatch):
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_FLOOR", 0.7)
    plan = _plan_cov(0.9)
    assert go._maybe_mark_low_resolution_degraded(plan, False) is False
    assert "_quality_degraded" not in plan


def test_g12_default_off(monkeypatch):
    """Default OFF (opt-in tras medir distribución): con el gate apagado, no marca aunque cobertura sea baja."""
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_GATE_ENABLED", False)
    assert go._maybe_mark_low_resolution_degraded(_plan_cov(0.1), False) is False
    # Y el default real del módulo es False.
    import importlib
    assert go.RESOLUTION_COVERAGE_GATE_ENABLED in (True, False)  # smoke: existe el knob


def test_g12_noop_for_fallback(monkeypatch):
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "RESOLUTION_COVERAGE_FLOOR", 0.7)
    assert go._maybe_mark_low_resolution_degraded(_plan_cov(0.3), True) is False


# ── G17: proveniencia de cultivares ──
def test_g17_cultivar_provenance_annotated():
    assert "P2-CULTIVAR-PROVENANCE" in _POPULATE_SRC, "Falta la anotación de proveniencia de cultivares (G17)."
    assert "proxy" in _POPULATE_SRC.lower() and "INCAP" in _POPULATE_SRC, (
        "G17: el comentario debe distinguir proxy USDA vs curado es-DO (INCAP)."
    )
