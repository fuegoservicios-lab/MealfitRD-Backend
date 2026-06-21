"""[P2-PROTEIN-FLOOR-FAILHARD · 2026-06-21] Garantía DURA del piso de proteína.

Audit presupuesto↔calidad (2026-06-21): el piso de proteína era best-effort — si el LLM no
convergía tras los retries (o se agotaba el budget), un plan bajo el piso se ENTREGABA con
banner (severity 'high', NO 'critical' → sin sustitución). El owner pidió "todo terreno":
NUNCA entregar bajo el piso.

Fix: el gate de review sigue en 'high' (da retries al LLM), pero `_apply_critical_review_guardrails`
ahora tiene un BACKSTOP fail-hard — si el plan FINAL sigue bajo el piso (cualquier ruta de
agotamiento) lo sustituye por el fallback matemático, que es protein-targeted POR CONSTRUCCIÓN
(`_build_fallback_day` asigna protein=target×ratio, ratios suman 1.0 → cada día = target ≥ piso).
Exento renal (techo KDIGO manda). Knob de rollback: MEALFIT_PROTEIN_FLOOR_FAILHARD_GATE.
"""
import importlib

import graph_orchestrator as go


def _plan(protein_per_meal, target="150g", renal=False):
    plan = {
        "main_goal": "gain_muscle",
        "macros": {"protein": target, "carbs": "200g", "fats": "60g"},
        "calories": 2000,
        "days": [
            {"day": 1, "meals": [{"protein": p} for p in protein_per_meal]},
        ],
    }
    if renal:
        plan["renal_protein_cap"] = {"applied": True}
    return plan


# ---------------------------------------------------------------------------
# 1. _protein_floor_shortfall (SSOT)
# ---------------------------------------------------------------------------
def test_shortfall_detecta_dias_bajo_piso():
    # target 150g, piso 0.90*150=135g. Día con 100g → bajo el piso.
    short = go._protein_floor_shortfall(_plan([50, 50], target="150g"), renal_capped=False)
    assert short, "Día con 100g de 150g (piso 135g) debe reportarse bajo el piso."
    assert short[0][1] == 100 and short[0][2] == 150


def test_shortfall_cumple_no_reporta():
    # Día con 150g ≥ piso 135g → no reporta.
    short = go._protein_floor_shortfall(_plan([75, 75], target="150g"), renal_capped=False)
    assert short == []


def test_shortfall_exento_renal():
    # Renal: el techo KDIGO manda, no se aplica el piso aunque esté "bajo".
    short = go._protein_floor_shortfall(_plan([50, 50], target="150g", renal=True), renal_capped=True)
    assert short == [], "Plan renal-capeado está EXENTO del piso de proteína."


# ---------------------------------------------------------------------------
# 2. Backstop en _apply_critical_review_guardrails
# ---------------------------------------------------------------------------
_NUTR = {"target_calories": 2000, "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60}}
_FORM = {"mainGoal": "gain_muscle"}


def test_backstop_sustituye_fallback_cuando_bajo_piso(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_FLOOR_FAILHARD_GATE", True)
    state = {
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": ["DÉFICIT DE PROTEÍNA"],
        "plan_result": _plan([40, 40], target="150g"),  # 80g de 150g → bajo el piso
    }
    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=1)
    out = state["plan_result"]
    assert out.get("_is_fallback") is True, "Plan bajo el piso debe sustituirse por fallback."
    assert out.get("_fallback_reason") == "protein_floor"
    # El fallback sustituido DEBE cumplir el piso (garantía dura, misma medida).
    assert go._protein_floor_shortfall(out, renal_capped=False) == [], (
        "El fallback matemático sustituido debe cumplir el piso de proteína."
    )


def test_backstop_no_toca_plan_renal_bajo_piso(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_FLOOR_FAILHARD_GATE", True)
    renal_plan = _plan([40, 40], target="150g", renal=True)
    state = {
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": ["otro"],
        "plan_result": renal_plan,
    }
    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=1)
    out = state["plan_result"]
    # Renal NO se sustituye por el backstop de proteína (techo KDIGO manda).
    assert out.get("_fallback_reason") != "protein_floor", (
        "Un plan renal bajo el piso NO debe dispararse por el backstop de proteína."
    )


def test_backstop_respeta_knob_off(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_FLOOR_FAILHARD_GATE", False)
    state = {
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": ["DÉFICIT DE PROTEÍNA"],
        "plan_result": _plan([40, 40], target="150g"),
    }
    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=1)
    out = state["plan_result"]
    # Knob OFF → comportamiento previo (entrega marcada, sin fallback por proteína).
    assert out.get("_fallback_reason") != "protein_floor"
    assert out.get("_is_fallback") is not True


# ---------------------------------------------------------------------------
# 3. Anchor
# ---------------------------------------------------------------------------
def test_marker_presente_en_source():
    src = open(go.__file__, encoding="utf-8").read()
    assert "P2-PROTEIN-FLOOR-FAILHARD" in src
    assert "PROTEIN_FLOOR_FAILHARD_GATE" in src
    assert "_protein_floor_shortfall" in src
