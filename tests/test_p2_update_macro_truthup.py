"""[P2-UPDATE-MACRO-TRUTHUP · 2026-06-24] Regresión del P2-1 de la 2da re-auditoría de inteligencia.

Gap: el validador de banda de las superficies de update leía los macros AUTO-REPORTADOS por el LLM
(`res.model_dump()`), no recomputados desde los ingredientes. Un LLM que emitía `protein:30` con
ingredientes que rinden ~12g pasaba la banda y persistía → Dashboard/PDF/day_quality_warning operaban
sobre la cifra fantasma. S1 ya corría `_truth_up_meal_macros_from_strings` (Guard 8z) pero NINGUNA
superficie de update lo invocaba.

Fix: invocar el helper SSOT ANTES del band-validator en swap_meal (agent.py) y execute_modify_single_meal
(tools.py). regenerate-day queda cubierto TRANSITIVAMENTE — su loop persiste el output de `swap_meal`,
ya truthed-up. Solo toca NÚMEROS (NO strings → lista de compras intacta). Knob MEALFIT_UPDATE_MACRO_TRUTHUP
(default ON). tooltip-anchor: P2-UPDATE-MACRO-TRUTHUP.

Parser-based (corre local + CI) + funcional del helper sobre un plato inflado (guardado por import de
graph_orchestrator en CI).
"""
import ast
import os

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


# ── helper SSOT existe en S1 ──────────────────────────────────────────────────
def test_helper_exists_in_s1():
    assert "def _truth_up_meal_macros_from_strings" in ORCH


# ── wired en swap (agent.py) ──────────────────────────────────────────────────
def test_truthup_wired_in_swap():
    src = _func_src(AGENT, "swap_meal")
    assert "_truth_up_meal_macros_from_strings" in src, "swap debe truth-up macros desde strings"
    assert "MEALFIT_UPDATE_MACRO_TRUTHUP" in src, "debe gatearse por el knob de update"
    assert "P2-UPDATE-MACRO-TRUTHUP" in src, "tooltip-anchor para que un renombre falle el test"
    # el truth-up debe correr ANTES del band-validator (si no, la banda mide la cifra inflada)
    assert src.index("_truth_up_meal_macros_from_strings") < src.index("_validate_macros is not None"), \
        "truth-up debe preceder al band-validator en swap"


# ── wired en chat-modify (tools.py) ───────────────────────────────────────────
def test_truthup_wired_in_chatmodify():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "_truth_up_meal_macros_from_strings" in src, "chat-modify debe truth-up macros desde strings"
    assert "MEALFIT_UPDATE_MACRO_TRUTHUP" in src
    assert "P2-UPDATE-MACRO-TRUTHUP" in src
    assert src.index("_truth_up_meal_macros_from_strings") < src.index("_validate_macros is not None"), \
        "truth-up debe preceder al band-validator en chat-modify"


# ── regenerate-day cubierto transitivamente (su loop persiste el output de swap_meal) ──
def test_regenerate_day_covered_transitively_via_swap():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "swap_meal(" in src, (
        "regenerate-day debe regenerar cada plato vía swap_meal (que ya hace truth-up) → "
        "la corrección de macros se propaga al día persistido sin un callsite extra"
    )


# ── funcional: el helper corrige un plato con proteína inflada ────────────────
class _FakeDB:
    """db mínima: pollo+arroz resuelven con macros reales bajos; 'sal' no resuelve (condimento)."""
    _M = {
        "pollo": {"protein": 12.0, "carbs": 0.0, "fats": 3.0, "kcal": 75.0},
        "arroz": {"protein": 2.0, "carbs": 28.0, "fats": 0.5, "kcal": 125.0},
    }

    def macros_from_ingredient_string(self, s):
        s = str(s).lower()
        for k, v in self._M.items():
            if k in s:
                return v
        return None

    def lookup(self, s):
        s = str(s).lower()
        return {"name": s} if any(k in s for k in self._M) else None


def test_truthup_corrects_inflated_meal():
    from graph_orchestrator import _truth_up_meal_macros_from_strings
    meal = {
        "protein": 40, "carbs": 30, "fats": 10, "cals": 500,
        "ingredients": ["100g pechuga de pollo", "1 taza de arroz", "sal al gusto"],
        "macros": ["P:40g", "C:30g", "G:10g"],
    }
    changed = _truth_up_meal_macros_from_strings(meal, _FakeDB())
    assert changed is True, "debe reescribir: 40g→14g proteína (pollo 12 + arroz 2)"
    assert meal["protein"] == 14, f"proteína real = 14g, no 40g inflados (got {meal['protein']})"
    assert meal["carbs"] == 28, f"carbos real = 28g (got {meal['carbs']})"
    assert meal["cals"] == 200, f"kcal real = 200 (got {meal['cals']})"
    # los strings de ingredientes NO se tocan → la lista de compras queda intacta
    assert meal["ingredients"] == ["100g pechuga de pollo", "1 taza de arroz", "sal al gusto"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
