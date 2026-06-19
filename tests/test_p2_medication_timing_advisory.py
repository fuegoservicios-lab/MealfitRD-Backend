"""[P2-MEDICATION-TIMING-ADVISORY · 2026-06-18] (audit fresco P2-1) Banner de timing dedicado para
fármacos timing-sensitive (levotiroxina ↔ calcio/hierro/soya). Distinto del advisory de interacción
general: el frontend lo renderiza prominente. Construido sobre el motor P1-MEDICATION-RULES.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import medication_rules as mr


def test_levothyroxine_is_timing_sensitive():
    rule = next(r for r in mr.MEDICATION_RULES if r.id == "levothyroxine")
    assert rule.timing_sensitive is True


def test_anticoagulant_not_timing_sensitive():
    rule = next(r for r in mr.MEDICATION_RULES if r.id == "anticoagulant")
    assert rule.timing_sensitive is False


def test_build_timing_advisories_levothyroxine():
    advs = mr.build_timing_advisories({"medications": ["Levotiroxina"]})
    assert len(advs) == 1
    assert advs[0]["medicamento"] == "Levotiroxina (tiroides)"
    assert "ayunas" in advs[0]["recomendacion"].lower()


def test_build_timing_advisories_empty_for_non_timing_meds():
    # Warfarina y metformina NO son timing-sensitive (su issue no es el momento de la comida).
    assert mr.build_timing_advisories({"medications": ["Warfarina", "Metformina"]}) == []


def test_build_timing_advisories_accepts_rule_list():
    rules = mr.detect_active_medications({"medications": ["Levotiroxina", "Metformina"]})
    advs = mr.build_timing_advisories(rules)
    assert [a["medicamento"] for a in advs] == ["Levotiroxina (tiroides)"]


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _simple_plan():
    return {"days": [{"meals": [{"name": "Almuerzo", "ingredients": ["100g de arroz", "150g de pollo"]}]}]}


def test_guard8d_surfaces_timing_advisories_for_levothyroxine(go):
    out = go._apply_deterministic_clinical_layer(
        _simple_plan(), {"medications": ["Levotiroxina"], "gender": "female"}, {})
    mr_block = out.get("medication_review") or {}
    assert "timing_advisories" in mr_block
    assert mr_block["timing_advisories"][0]["medicamento"] == "Levotiroxina (tiroides)"


def test_guard8d_no_timing_advisories_for_warfarin_only(go):
    out = go._apply_deterministic_clinical_layer(
        _simple_plan(), {"medications": ["Warfarina"], "gender": "female"}, {})
    mr_block = out.get("medication_review") or {}
    assert "timing_advisories" not in mr_block  # warfarina no es timing-sensitive


def test_orchestrator_anchor(go):
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-MEDICATION-TIMING-ADVISORY" in src
    assert "build_timing_advisories" in src
