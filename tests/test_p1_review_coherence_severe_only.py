"""[P1-REVIEW-COHERENCE-SEVERE-ONLY · 2026-07-09] WHACK-A-MOLE BREAKER del path assemble→review.

Forense (batch logs-VPS 2026-07-09): TODA generación iba a los 3 intentos, rechazada por "COHERENCIA
RECETAS LISTA: 1 divergencia crítica" sobre una proteína ROTATIVA (Res→Cangrejo→Chivo). Cada retry
regenera el plan COMPLETO → aparece OTRA divergencia marginal de UNA proteína distinta → nunca converge →
max_attempts → entrega degradada.

Fix (espeja MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY y _T2_BLOCK_SEVERE_ONLY): en `review_plan_node`, una
sola divergencia MARGINAL (magnitud <SEVERE_DELTA) degrada a warn (entrega + telemetría) en vez de forzar
retry. SÍ rechaza si es SEVERA: ≥MIN_COUNT divergencias o magnitud ≥SEVERE_DELTA.
"""
import asyncio
import os

import graph_orchestrator

_GO_SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "graph_orchestrator.py"), encoding="utf-8").read()


# ─────────────────────── helpers (self-contained) ───────────────────────
def _bypass_form_data():
    return {"user_id": "guest", "allergies": [], "medicalConditions": [], "dislikes": [],
            "dietType": "balanced", "_days_to_generate": 3}


def _minimal_plan(*, with_block=None):
    plan = {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [{"day": 1, "meals": [
            {"meal": "almuerzo", "name": "Pollo con arroz", "ingredients": ["200 g pollo", "150 g arroz"],
             "recipe": ["Mise en place: pesa el pollo y lava el arroz.",
                        "El Toque de Fuego: cocina el pollo 8-10 min y hierve el arroz 15 min.",
                        "Montaje: sirve el pollo sobre el arroz."],
             "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}]}],
    }
    if with_block is not None:
        plan["_shopping_coherence_block"] = with_block
    return plan


def _minimal_state(*, plan_result):
    return {"plan_result": plan_result, "form_data": _bypass_form_data(), "taste_profile": "",
            "attempt": 1, "rejection_reasons": [], "_rejection_severity": "minor", "request_id": "test-severe-only"}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────── estructural ───────────────────────
def test_marker_and_knobs_present():
    assert "P1-REVIEW-COHERENCE-SEVERE-ONLY" in _GO_SRC
    assert "MEALFIT_REVIEW_COHERENCE_BLOCK_SEVERE_ONLY" in _GO_SRC
    assert "MEALFIT_REVIEW_COHERENCE_SEVERE_MIN_COUNT" in _GO_SRC
    assert "MEALFIT_REVIEW_COHERENCE_SEVERE_DELTA" in _GO_SRC


# ─────────────────────── funcional ───────────────────────
def test_single_marginal_magnitude_degrades_to_warn(monkeypatch):
    """1 divergencia de magnitud MARGINAL (|Δ|<0.5) → NO rechaza (warn), flag limpiado."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    block = [{"food": "Res", "magnitude": True, "expected_qty": 1000, "actual_qty": 850, "delta_pct": 0.15}]
    plan = _minimal_plan(with_block=block)
    result = _run(graph_orchestrator.review_plan_node(_minimal_state(plan_result=plan)))
    assert result["review_passed"] is True, "1 divergencia marginal no debe forzar retry"
    assert "_shopping_coherence_block" not in plan, "el flag debe limpiarse (tolerado)"


def test_single_presence_divergence_degrades_to_warn(monkeypatch):
    """1 divergencia de PRESENCIA aislada (proteína rotativa false-positive) → warn, NO retry."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    block = [{"food": "Cangrejo", "side": "expected_only"}]
    plan = _minimal_plan(with_block=block)
    result = _run(graph_orchestrator.review_plan_node(_minimal_state(plan_result=plan)))
    assert result["review_passed"] is True, "1 divergencia de presencia aislada no debe forzar retry"


def test_two_divergences_are_severe_and_reject(monkeypatch):
    """≥MIN_COUNT (2) divergencias = sistemático → SÍ rechaza."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    block = [{"food": "Res", "side": "expected_only"},
             {"food": "Pollo", "magnitude": True, "expected_qty": 1000, "actual_qty": 500, "delta_pct": 0.5}]
    result = _run(graph_orchestrator.review_plan_node(_minimal_state(plan_result=_minimal_plan(with_block=block))))
    assert result["review_passed"] is False, "2+ divergencias deben rechazar (severo por count)"


def test_single_large_magnitude_is_severe_and_rejects(monkeypatch):
    """1 divergencia de magnitud EGREGIA (|Δ|≥0.5) → SÍ rechaza."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    block = [{"food": "Res", "magnitude": True, "expected_qty": 1000, "actual_qty": 200, "delta_pct": 0.8}]
    result = _run(graph_orchestrator.review_plan_node(_minimal_state(plan_result=_minimal_plan(with_block=block))))
    assert result["review_passed"] is False, "magnitud egregia debe rechazar (severo por Δ)"


def test_knob_off_reverts_to_reject_always(monkeypatch):
    """Kill switch: severe_only=false → 1 divergencia marginal vuelve a rechazar (comportamiento previo)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    monkeypatch.setenv("MEALFIT_REVIEW_COHERENCE_BLOCK_SEVERE_ONLY", "false")
    block = [{"food": "Res", "magnitude": True, "expected_qty": 1000, "actual_qty": 850, "delta_pct": 0.15}]
    result = _run(graph_orchestrator.review_plan_node(_minimal_state(plan_result=_minimal_plan(with_block=block))))
    assert result["review_passed"] is False, "con el kill switch OFF, 1 divergencia marginal rechaza"
