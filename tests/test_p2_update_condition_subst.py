"""[P2-UPDATE-CONDITION-SUBST · 2026-06-26] (audit 3-flujos P2) La SUSTITUCIÓN determinista de
ingredientes por condición médica (DM2 azúcar→stevia; HTA embutidos/cubitos→fresco; dislipidemia
mantequilla/tocino→magro) debe correr también en las superficies de UPDATE (swap S3 / regenerate-day
S2 / chat-modify), no solo en el Guard 3 de S1.

Pre-fix: `_apply_condition_substitutions` tenía su ÚNICO callsite en el path de assemble de S1. Los
backstops portados a updates (clinical_backstop = alérgeno+dieta+mercurio; renal trim; food-safety) NO
incluían la sustitución por condición → un swap/modify a 'longaniza'/'mantequilla'/'azúcar' se persistía
contraindicado para la condición del usuario (la única defensa era la directiva-prompt advisory, falible).

Fix: helper `condition_substitution_backstop_for_meal(meal, form_data)` que envuelve el meal en un
mini-plan y reusa `_apply_condition_substitutions` (idempotente, macro-preservante, fail-open), cableado
en swap_meal (→ regenerate-day por herencia del loop) y execute_modify_single_meal. Gateado por
UPDATE_CONDITION_SUBST_ENABLED + DM2_SUGAR_GUARD.

Parser-based para el WIRING (no se puede aislar sin importar agent.py/tools.py completos) + funcional
determinista (monkeypatch del engine) para el contrato del wrapper.
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


ORCH = _read("graph_orchestrator.py")
AGENT = _read("agent.py")
TOOLS = _read("tools.py")
SWAP = _func_src(AGENT, "swap_meal")
MODIFY = _func_src(TOOLS, "execute_modify_single_meal")


# ── Helper + knob existen ──────────────────────────────────────────────────────
def test_helper_and_knob_defined():
    assert "def condition_substitution_backstop_for_meal(" in ORCH, "helper SSOT ausente"
    assert "MEALFIT_UPDATE_CONDITION_SUBST" in ORCH, "knob de rollback dedicado ausente"
    assert "UPDATE_CONDITION_SUBST_ENABLED" in ORCH


def test_helper_wraps_and_reuses_engine():
    helper = _func_src(ORCH, "condition_substitution_backstop_for_meal")
    # reusa el motor de S1 (no duplica lógica) sobre un mini-plan
    assert "_apply_condition_substitutions(" in helper
    assert '"days": [{"meals": [meal]}]' in helper or "'days': [{'meals': [meal]}]" in helper
    # gateado por ambos knobs + fail-open
    assert "UPDATE_CONDITION_SUBST_ENABLED" in helper and "DM2_SUGAR_GUARD" in helper
    assert "isinstance(meal, dict)" in helper


# ── Wiring en las 3 superficies ────────────────────────────────────────────────
def test_imported_in_update_surfaces():
    for src, fname in ((AGENT, "agent.py"), (TOOLS, "tools.py")):
        assert "condition_substitution_backstop_for_meal" in src, f"no importado/llamado en {fname}"


def test_wired_in_swap_after_food_safety():
    # regenerate-day (S2) hereda este call por el loop de swaps.
    assert "condition_substitution_backstop_for_meal(_out" in SWAP, "no cableado en swap_meal"
    assert SWAP.index("food_safety_backstop_for_meal(_out)") < SWAP.index("condition_substitution_backstop_for_meal(_out"), \
        "la sustitución por condición debe correr tras food-safety (orden de S1)"


def test_wired_in_chat_modify_after_food_safety():
    assert "condition_substitution_backstop_for_meal(new_meal_data" in MODIFY, "no cableado en chat-modify"
    assert MODIFY.index("food_safety_backstop_for_meal(new_meal_data)") < MODIFY.index("condition_substitution_backstop_for_meal(new_meal_data"), \
        "la sustitución por condición debe correr tras food-safety"
    # el chat no envía el wizard form → enriquece medicalConditions desde el perfil server-side
    assert "_cond_form" in MODIFY and "medicalConditions" in MODIFY


# ── Funcional determinista del wrapper (monkeypatch del engine) ─────────────────
try:
    import graph_orchestrator as _GO
    _GO_ERR = None
except Exception as _e:  # pragma: no cover
    _GO = None
    _GO_ERR = _e

_needs_go = pytest.mark.skipif(_GO is None, reason=f"graph_orchestrator no importable: {_GO_ERR}")


@_needs_go
def test_wrapper_delegates_miniplan_and_form(monkeypatch):
    monkeypatch.setattr(_GO, "UPDATE_CONDITION_SUBST_ENABLED", True)
    monkeypatch.setattr(_GO, "DM2_SUGAR_GUARD", True)
    captured = {}

    def _spy(plan, form_data):
        captured["plan"] = plan
        captured["form"] = form_data
        return 7

    monkeypatch.setattr(_GO, "_apply_condition_substitutions", _spy)
    meal = {"name": "Merienda", "ingredients": ["100g de Longaniza"]}
    out = _GO.condition_substitution_backstop_for_meal(meal, {"medicalConditions": ["hta"]})
    assert out == 7, "debe propagar el nº de comidas del engine"
    assert captured["plan"] == {"days": [{"meals": [meal]}]}, "debe envolver el meal en un mini-plan"
    assert captured["form"] == {"medicalConditions": ["hta"]}
    # el meal pasado es el MISMO objeto → la mutación in-place del engine se refleja en el caller
    assert captured["plan"]["days"][0]["meals"][0] is meal


@_needs_go
def test_wrapper_gated_off_does_not_delegate(monkeypatch):
    monkeypatch.setattr(_GO, "UPDATE_CONDITION_SUBST_ENABLED", False)
    monkeypatch.setattr(_GO, "DM2_SUGAR_GUARD", True)
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return 5

    monkeypatch.setattr(_GO, "_apply_condition_substitutions", _spy)
    assert _GO.condition_substitution_backstop_for_meal({"ingredients": []}, {}) == 0
    assert called["n"] == 0, "knob off → no debe delegar al engine"


@_needs_go
def test_wrapper_non_dict_meal_is_noop():
    assert _GO.condition_substitution_backstop_for_meal(None, {}) == 0
    assert _GO.condition_substitution_backstop_for_meal("nope", {}) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
