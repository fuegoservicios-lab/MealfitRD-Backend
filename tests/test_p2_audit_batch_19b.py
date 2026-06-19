"""[P2-AUDIT-BATCH-19B · 2026-06-19] Lote de P2 del audit fresco 2026-06-19b (cluster S1 + clínicos + infra).

Cubre: #1 árbitro renal+HTA en prompt · #2 nota fibra renal-aware · #3 consumo del monitor vit-K · #4 doc
SSOT de conflictos inter-motor · #5 trim-ceiling renal usa peso ajustado · #6 veto por sub-frase · #7 regla
IMAO↔tiramina · #8 ave cruda (tartar/carpaccio) · #9 especies RD · #10/#14/#15/#16 anchors source-parse.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import condition_rules as cr
import medication_rules as mr
import micronutrients as mn

_BACKEND = Path(__file__).resolve().parent.parent


def _src(name: str) -> str:
    return (_BACKEND / name).read_text(encoding="utf-8")


class _StubDB:
    def __init__(self, **micros):
        self._m = micros

    def micros_from_ingredient_string(self, s):
        return dict(self._m)


def _plan(n=2):
    return {"days": [{"meals": [{"ingredients": [f"ing{i}" for i in range(n)]}]}]}


def _plan_meal(name):
    return {"days": [{"meals": [{"name": name, "ingredients": []}]}]}


# ── #1 árbitro renal+HTA del potasio en el prompt ──────────────────────────────────────────────
def test_p2_1_hta_renal_prompt_arbiter_moderates_potassium():
    p = cr.build_condition_prompt({"medicalConditions": ["hipertension", "enfermedad renal"]})
    assert "HIPERTENSIÓN + ENFERMEDAD RENAL" in p
    assert "MODERA" in p and "potasio" in p.lower()


def test_p2_1_dm2_renal_arbiter_still_present():
    p = cr.build_condition_prompt({"medicalConditions": ["diabetes", "enfermedad renal"]})
    assert "DIABETES + ENFERMEDAD RENAL" in p  # no regresión del árbitro existente


def test_p2_1_hta_alone_no_renal_arbiter():
    p = cr.build_condition_prompt({"medicalConditions": ["hipertension"]})
    assert "HIPERTENSIÓN + ENFERMEDAD RENAL" not in p


# ── #2 nota de fibra renal-aware (no empuja leguminosas) ───────────────────────────────────────
def test_p2_2_fiber_note_renal_aware():
    rep = mn.build_micronutrient_report(_plan(), _StubDB(grams=100, fiber=2), sex="M",
                                        conditions=["enfermedad renal"])
    fgap = [g for g in rep["gaps"] if g["key"] == "fiber_g"]
    assert fgap, "fibra muy baja → gap presente"
    note = fgap[0]["nota"].lower()
    assert "bajos en potasio" in note and "modera" in note


def test_p2_2_fiber_note_standard_without_renal():
    rep = mn.build_micronutrient_report(_plan(), _StubDB(grams=100, fiber=2), sex="M",
                                        conditions=["Ninguna"])
    fgap = [g for g in rep["gaps"] if g["key"] == "fiber_g"]
    assert fgap and "habichuelas" in fgap[0]["nota"].lower()  # nota estándar


# ── #3 consumo del monitor de consistencia de vit-K ────────────────────────────────────────────
def test_p2_3_vitk_high_variability_marks_degraded():
    import graph_orchestrator as go
    plan = {"micronutrient_report": {"gaps": [], "coverage": 0.9},
            "medication_review": {"vitamin_k_consistency": {"variability": "high"}}}
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is True
    assert plan.get("_quality_degraded_reason") == "vitamin_k_inconsistent"


def test_p2_3_vitk_low_variability_not_degraded():
    import graph_orchestrator as go
    plan = {"micronutrient_report": {"gaps": [], "coverage": 0.9},
            "medication_review": {"vitamin_k_consistency": {"variability": "low"}}}
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False


# ── #4 doc SSOT de conflictos inter-motor: cada marker citado existe en el código ───────────────
def test_p2_4_conflict_doc_markers_exist_in_source():
    doc = _src("docs/clinical_conflict_resolution.md")
    # Las 4 filas de conflicto + invariantes.
    for tok in ("C1", "C2", "C3", "C4", "I-CONF-1", "I-CONF-2", "I-CONF-3"):
        assert tok in doc, tok
    # Markers nuevos de este lote DEBEN existir en su archivo fuente (anti-rename).
    assert "P1-POTASSIUM-PANEL-MED-AWARE" in _src("micronutrients.py")
    assert "P2-RENAL-HTA-POTASSIUM-PROMPT" in _src("condition_rules.py")
    assert "P2-RENAL-FIBER-NOTE" in _src("micronutrients.py")
    assert "P2-WARFARIN-VITK-CONSUME" in _src("graph_orchestrator.py")


# ── #5 trim-ceiling renal usa _renal_weight_basis_kg (no el peso real) ──────────────────────────
def test_p2_5_trim_ceiling_uses_renal_weight_basis(monkeypatch):
    import graph_orchestrator as go
    form = {"medicalConditions": ["enfermedad renal"], "mainGoal": "lose_fat"}
    monkeypatch.setattr(go, "_renal_weight_basis_kg", lambda fd: 70.0)   # base ajustada (sentinel)
    monkeypatch.setattr(go, "_weight_kg_from_form", lambda fd: 120.0)    # peso real (no debe usarse)
    target = 56.0
    pct = go._goal_aware_trim_ceiling_pct(form, target)
    # ceiling absoluto = pct*target debe = RENAL_GKG * 70 (base renal), NO * 120.
    assert abs(pct * target - go.RENAL_PROTEIN_GKG_CEILING * 70.0) < 0.6


# ── #6 veto por SUB-FRASE: fármaco real + frase-condición vetada en un mismo string ─────────────
def test_p2_6_veto_tokenized_detects_drug_co_occurring_with_condition():
    ids = {r.id for r in mr.detect_active_medications(
        {"otherConditions": "tengo resistencia a la insulina y tomo lantus"})}
    assert "insulin_secretagogue" in ids


def test_p2_6_veto_still_suppresses_condition_alone():
    ids = {r.id for r in mr.detect_active_medications({"otherConditions": "tengo resistencia a la insulina"})}
    assert "insulin_secretagogue" not in ids


# ── #7 regla IMAO ↔ tiramina ────────────────────────────────────────────────────────────────────
def test_p2_7_maoi_rule_detected():
    for med in ("fenelzina", "tranilcipromina", "moclobemida", "linezolid"):
        assert "maoi" in {r.id for r in mr.detect_active_medications({"medications": [med]})}, med
    prompt = mr.build_medication_prompt({"medications": ["fenelzina"]}).lower()
    assert "tiramina" in prompt


# ── #8 ave cruda + #9 especies RD en la co-ocurrencia tartar/carpaccio ──────────────────────────
def test_p2_8_raw_poultry_tartar_flagged():
    import graph_orchestrator as go
    assert go._scan_raw_seafood_meat_violations(_plan_meal("Tartar de pollo"))
    assert not go._scan_raw_seafood_meat_violations(_plan_meal("Pollo a la plancha"))


def test_p2_9_raw_seafood_species_flagged():
    import graph_orchestrator as go
    assert go._scan_raw_seafood_meat_violations(_plan_meal("Carpaccio de corvina"))
    assert go._scan_raw_seafood_meat_violations(_plan_meal("Tartar de langosta"))


def test_p2_8_9_word_boundary_no_substring_false_positives():
    """El review cazó: substring 'pollo'⊂repollo, 'ave'⊂avena, 'res'⊂fresa → falsos-positivos. El match por
    límite de palabra debe NO flagear platos VEGETALES seguros aunque lleven tartar/carpaccio."""
    import graph_orchestrator as go
    for safe in ("Carpaccio de repollo morado con vinagreta",          # 'pollo' ⊂ repollo
                 "Carpaccio de remolacha con granola de avena",         # 'ave' ⊂ avena
                 "Tartar de fresa y mango",                             # 'res' ⊂ fresa
                 "Tartar de remolacha"):                                # ningún animal
        assert not go._scan_raw_seafood_meat_violations(_plan_meal(safe)), safe


# ── #10 / #14 / #15 / #16 anchors source-parse ──────────────────────────────────────────────────
def test_p2_10_reconcile_clamp_telemetry_anchor():
    assert "P2-RECONCILE-CLAMP-TELEMETRY" in _src("graph_orchestrator.py")


def test_p2_14_session_secret_fail_loud():
    assert "P2-SESSION-SECRET-FAIL-LOUD" in _src("auth.py")
    import auth
    # precondición del fail-loud: un secreto corto deja la cookie deshabilitada.
    monkey_secret = auth._SESSION_SECRET
    try:
        auth._SESSION_SECRET = "corto"
        assert auth.session_cookies_enabled() is False
    finally:
        auth._SESSION_SECRET = monkey_secret


def test_p2_15_silence_node_knob_uses_env_str():
    src = _src("cron_tasks.py")
    # El knob debe leerse via _env_str (auto-registra) y NO via os.environ.get crudo.
    assert 'P2-SILENCE-NODE-KNOB-REGISTRY' in src
    assert '_env_str("MEALFIT_PIPELINE_METRICS_SILENCE_NODE"' in src
    assert 'os.environ.get("MEALFIT_PIPELINE_METRICS_SILENCE_NODE"' not in src
    # y _env_str debe estar importado.
    assert re.search(r'from graph_orchestrator import .*_env_str', src)


def test_p2_16_pregnancy_gate_uses_goal_adjustments():
    src = _src("nutrition_calculator.py")
    assert "P2-PREGNANCY-GATE-ADJUSTMENT" in src
    # El gate de embarazo ya no usa la igualdad de string 'lose_fat' (usa GOAL_ADJUSTMENTS<0).
    # Verifica que el bloque del gate de embarazo contiene la forma robusta.
    preg_idx = src.find("PREGNANCY-GATE-ADJUSTMENT")
    window = src[preg_idx:preg_idx + 800]
    assert "GOAL_ADJUSTMENTS.get(goal, 0.0) < 0" in window
    # y ya NO la igualdad de string en ese bloque del gate de embarazo.
    assert 'if goal == "lose_fat"' not in window
