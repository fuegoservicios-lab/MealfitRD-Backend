"""[P1-UPDATE-INTELLIGENCE · 2026-06-23] Regresión de los 6 P1 del audit de inteligencia del motor.

Las superficies de UPDATE de platos (swap individual S3, regenerate-day S2, chat modify) reusaban el
LLM crudo + gate de macros pero NO heredaban la inteligencia de S1. Estos 6 P1 la extienden:

  P1-2  MEALFIT_UPDATE_MACRO_REBALANCE   — rebalanceador determinista de macros en swap (S2 vía loop). OFF (A/B).
  P1-3  MEALFIT_REGEN_DAY_RETARGET_TO_GOAL — regenerate-day apunta al objetivo real, no a la suma drifteada.
  P1-4  MEALFIT_UPDATE_SUPERPERS         — súper-personalización (gustos/cocina/religión/equipo) en updates.
  P1-5  MEALFIT_UPDATE_CROSS_DAY_VARIETY — variedad cross-day (no repetir la proteína de otros días).
  P1-6  MEALFIT_MARKER_UNRESOLVED_HONESTY — marca degradado si quedan markers de slot-coherence sin resolver.
  P1-7  MEALFIT_UPDATE_CONDITION_DIRECTIVES + MEALFIT_UPDATE_RECOMPUTE_MICROS — directivas de condición + panel.

Mayormente parser-based (el repo prueba estos puntos de integración por parsing del source de prod, con
tooltip-anchors que fallan ante un renombre). Los tests funcionales que requieren importar `routers.plans`
se saltan en entornos sin `langchain_openai` real (corren en CI).
"""
import ast
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
TOOLS = _read("tools.py")
PLANS = _read("routers/plans.py")
ORCH = _read("graph_orchestrator.py")
DASHBOARD = _read("../frontend/src/pages/Dashboard.jsx")


# ── P1-2: rebalanceador de macros en updates ──────────────────────────────────
def test_p1_2_swap_runs_macro_rebalancer_with_pantry_revert():
    src = _func_src(AGENT, "swap_meal")
    assert "_rebalance_day_macros_to_target" in src, "swap_meal debe correr el rebalanceador de macros"
    assert "MEALFIT_UPDATE_MACRO_REBALANCE" in src
    # default OFF (riesgo pantry): el knob se lee con default 'false'
    assert re.search(r'MEALFIT_UPDATE_MACRO_REBALANCE["\']\s*,\s*["\']false', src), "P1-2 debe ser default OFF"
    # re-validación de pantry + revert si rompe
    assert "validate_ingredients_against_pantry" in src and "_snapshot" in src


# ── P1-3: regenerate-day retarget-to-goal ─────────────────────────────────────
def test_p1_3_regenerate_day_retargets_to_goal():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "get_nutrition_targets" in src, "regenerate-day debe recomputar el objetivo real"
    assert "_meal_scale" in src, "debe escalar los targets per-meal hacia el objetivo"
    assert "MEALFIT_REGEN_DAY_RETARGET_TO_GOAL" in src
    assert "day_quality_warning" in src, "debe avisar si el día quedó bajo en proteína (honestidad)"
    # max por nutriente (no bajar de la meta)
    assert "max(day_target" in src


# ── P1-4: super-personalización en updates ────────────────────────────────────
def test_p1_4_superpers_injected_in_swap_and_modify():
    swap = _func_src(AGENT, "swap_meal")
    modify = _func_src(TOOLS, "execute_modify_single_meal")
    assert "build_super_personalization_context" in swap, "swap debe inyectar súper-personalización"
    assert "build_super_personalization_context" in modify, "modify debe inyectar súper-personalización"
    assert "MEALFIT_UPDATE_SUPERPERS" in swap and "MEALFIT_UPDATE_SUPERPERS" in modify


def test_p1_4_enrich_attaches_superpers_and_conditions():
    enrich = _func_src(PLANS, "_enrich_clinical_from_profile")
    assert "super_personalization" in enrich, "el enrich debe adjuntar super_personalization del perfil"
    assert "medicalConditions" in enrich, "el enrich debe adjuntar condiciones del perfil"


def test_p1_4_regenerate_day_meal_form_carries_superpers():
    src = _func_src(PLANS, "api_regenerate_day")
    assert '"super_personalization": data.get("super_personalization")' in src
    assert '"medicalConditions": data.get("medicalConditions")' in src


# ── P1-5: variedad cross-day ──────────────────────────────────────────────────
def test_p1_5_regenerate_day_seeds_cross_day_proteins():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "MEALFIT_UPDATE_CROSS_DAY_VARIETY" in src
    # siembra day_avoid con proteínas de OTROS días antes del loop
    assert "_main_protein_of_meal(_om)" in src or "_main_protein_of_meal(_om" in src


def test_p1_5_swap_biases_to_fresh_proteins():
    src = _func_src(AGENT, "swap_meal")
    assert "_plan_meals_text_for_variety" in src
    assert "_fresh_proteins" in src
    assert "MEALFIT_UPDATE_CROSS_DAY_VARIETY" in src


# ── P1-6: honestidad de markers sin resolver ──────────────────────────────────
def test_p1_6_should_retry_flags_unresolved_markers():
    src = _func_src(ORCH, "should_retry")
    assert "MARKER_UNRESOLVED_HONESTY" in src
    assert "slot_coherence_unresolved" in src
    assert "_mark_plan_result_quality_degraded" in src


def test_p1_6_knob_and_frontend_copy_exist():
    assert re.search(r"MARKER_UNRESOLVED_HONESTY\s*=\s*_env_bool", ORCH)
    assert "slot_coherence_unresolved" in DASHBOARD, "el frontend debe tener copy para el nuevo reason"


# ── P1-7: condiciones + micros en updates ─────────────────────────────────────
def test_p1_7_condition_directives_in_updates():
    swap = _func_src(AGENT, "swap_meal")
    modify = _func_src(TOOLS, "execute_modify_single_meal")
    assert "build_condition_prompt" in swap and "build_medication_prompt" in swap
    assert "build_condition_prompt" in modify and "build_medication_prompt" in modify
    assert "MEALFIT_UPDATE_CONDITION_DIRECTIVES" in swap


def test_p1_7_micro_report_recompute_helper_exists_and_called():
    assert "def recompute_micronutrient_report_for_plan" in ORCH
    regen = _func_src(PLANS, "api_regenerate_day")
    assert "recompute_micronutrient_report_for_plan" in regen
    assert "MEALFIT_UPDATE_RECOMPUTE_MICROS" in regen


def test_p1_7_recompute_helper_failsafe_on_disabled(monkeypatch):
    """El helper es best-effort: con MICRONUTRIENT_REPORT_ENABLED apagado retorna False sin mutar."""
    import graph_orchestrator as go
    monkeypatch.setattr(go, "MICRONUTRIENT_REPORT_ENABLED", False)
    plan = {"days": [{"meals": [{"name": "x", "ingredients": ["arroz"], "cals": 400}]}]}
    assert go.recompute_micronutrient_report_for_plan(plan, {"gender": "female"}) is False
    assert "micronutrient_report" not in plan


# ── knobs umbrella + marker ───────────────────────────────────────────────────
def test_all_p1_knobs_present():
    for knob in (
        "MEALFIT_UPDATE_MACRO_REBALANCE", "MEALFIT_REGEN_DAY_RETARGET_TO_GOAL",
        "MEALFIT_UPDATE_SUPERPERS", "MEALFIT_UPDATE_CROSS_DAY_VARIETY",
        "MEALFIT_MARKER_UNRESOLVED_HONESTY", "MEALFIT_UPDATE_CONDITION_DIRECTIVES",
        "MEALFIT_UPDATE_RECOMPUTE_MICROS",
    ):
        assert knob in (AGENT + TOOLS + PLANS + ORCH), f"knob {knob} ausente"


def test_marker_bumped():
    assert "P1-UPDATE-INTELLIGENCE" in _read("app.py")


# ── Funcional (guardado por import de routers.plans) ──────────────────────────
try:
    from routers.plans import _enrich_clinical_from_profile as _ENRICH
    _ENRICH_ERR = None
except Exception as _e:  # pragma: no cover
    _ENRICH = None
    _ENRICH_ERR = _e

requires_router = pytest.mark.skipif(
    _ENRICH is None, reason=f"routers.plans no importable (¿falta langchain_openai?): {_ENRICH_ERR}"
)


@requires_router
def test_enrich_attaches_superpers_and_conditions_functional(monkeypatch):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {
        "allergies": ["maní"], "dietType": "balanced",
        "super_personalization": {"foodLikes": ["mangú"]},
        "medicalConditions": ["diabetes"], "medications": ["metformina"],
    }})
    data = {"user_id": "u1"}
    _ENRICH(data, "u1")
    assert data.get("super_personalization") == {"foodLikes": ["mangú"]}
    assert data.get("medicalConditions") == ["diabetes"]
    assert data.get("medications") == ["metformina"]
