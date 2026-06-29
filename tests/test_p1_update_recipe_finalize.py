"""[P1-UPDATE-RECIPE-FINALIZE · 2026-06-29] (audit objetivo · paridad updates ↔ form-gen)

Los finalizadores de coherencia de RECETA que `assemble_plan_node` corre en form-gen
(veg-fantasma en los PASOS → ingredients[], 'lonja de queso' → gramos, cap de hojas) NO corrían
en NINGUNA superficie de update → reaparecía el bug ghost-vegetal (un vegetal en los pasos ausente
de la lista no se compra + sub-cuenta macros) y unidades vagas en platos swapeados/modificados.

`finalize_single_meal_recipe_coherence(meal, db)` los aplica per-meal y se cablea en:
  - swap_meal (agent.py) → regenerate-day lo hereda (loop de swap)
  - execute_modify_single_meal (tools.py, chat-modify)
  - /recalculate-shopping-list (plans.py, antes de reconstruir la lista canónica)
  - finalize_plan_data_coherence (graph_orchestrator) para el persist boundary degradado/INSERT

Tests: (1) parser-based del wiring en las 4 superficies; (2) funcional de la orquestación del
helper con sub-funciones mockeadas (sin DB/Neon).
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser-based: helper definido + knob + wiring en las 4 superficies
# ---------------------------------------------------------------------------
def test_helper_and_knob_exist():
    assert hasattr(g, "finalize_single_meal_recipe_coherence"), "falta el helper finalizador"
    assert hasattr(g, "UPDATE_RECIPE_FINALIZE_ENABLED"), "falta el knob UPDATE_RECIPE_FINALIZE_ENABLED"
    assert g.UPDATE_RECIPE_FINALIZE_ENABLED is True, "el finalizador de update debe estar ON por default"


def test_wired_in_swap():
    assert "finalize_single_meal_recipe_coherence" in _AGENT, "swap_meal no invoca el finalizador"
    assert "P1-UPDATE-RECIPE-FINALIZE" in _AGENT


def test_wired_in_chat_modify():
    assert "finalize_single_meal_recipe_coherence" in _TOOLS, "chat-modify no invoca el finalizador"
    assert "P1-UPDATE-RECIPE-FINALIZE" in _TOOLS


def test_wired_in_recalculate():
    assert "finalize_single_meal_recipe_coherence" in _PLANS, "/recalculate no invoca el finalizador"
    assert "P1-UPDATE-RECIPE-FINALIZE" in _PLANS


def test_veg_guard_added_to_persist_boundary():
    """finalize_plan_data_coherence (persist boundary: INSERT/chunk degradado) debe incluir el
    veg-guard, no solo slice/leaf/quantize."""
    # El bloque del veg-guard dentro de finalize_plan_data_coherence.
    assert "_add_missing_recipe_step_vegetables(days)" in _GRAPH, \
        "el veg-guard no se añadió al persist boundary finalize_plan_data_coherence"


# ---------------------------------------------------------------------------
# 2. Funcional: orquestación del helper con sub-funciones mockeadas (sin DB)
# ---------------------------------------------------------------------------
def test_finalizer_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(g, "UPDATE_RECIPE_FINALIZE_ENABLED", False)
    assert g.finalize_single_meal_recipe_coherence({"name": "x", "ingredients": [], "recipe": []}) == 0


def test_finalizer_noop_on_non_dict():
    assert g.finalize_single_meal_recipe_coherence(None) == 0
    assert g.finalize_single_meal_recipe_coherence("nope") == 0


def test_finalizer_orchestrates_subfunctions(monkeypatch):
    """Con el knob ON, el helper corre veg-guard + slice + leaf y SUMA sus conteos; el veg-guard
    dispara un truth-up. Mockeamos las sub-funciones para no tocar Neon."""
    monkeypatch.setattr(g, "UPDATE_RECIPE_FINALIZE_ENABLED", True)
    monkeypatch.setattr(g, "RECIPE_STEP_VEG_GUARD_ENABLED", True)

    calls = {"veg": 0, "slice": 0, "leaf": 0, "truthup": 0}

    def _fake_veg(wrap, **kw):
        calls["veg"] += 1
        # Simula que añadió 1 vegetal al meal envuelto.
        wrap[0]["meals"][0].setdefault("ingredients", []).append("100g Brócoli")
        return 1

    def _fake_slice(wrap, db=None):
        calls["slice"] += 1
        return 2

    def _fake_leaf(wrap, db=None):
        calls["leaf"] += 1
        return 0

    def _fake_truthup(meal, db):
        calls["truthup"] += 1
        return True

    monkeypatch.setattr(g, "_add_missing_recipe_step_vegetables", _fake_veg)
    monkeypatch.setattr(g, "_recipe_slice_units_to_grams", _fake_slice)
    monkeypatch.setattr(g, "_cap_leaf_volume_in_meals", _fake_leaf)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", _fake_truthup)

    meal = {"name": "Revoltillo", "ingredients": ["3 huevos"], "recipe": ["Saltea el brócoli"]}
    # Pasamos un db dummy para que NO construya IngredientNutritionDB() (que tocaría Neon).
    total = g.finalize_single_meal_recipe_coherence(meal, db=object())

    assert total == 3, f"el helper debe sumar veg(1)+slice(2)+leaf(0)=3, dio {total}"
    assert calls["veg"] == 1 and calls["slice"] == 1 and calls["leaf"] == 1
    assert calls["truthup"] == 1, "tras añadir un veg, debe recomputar macros (truth-up)"
    assert "100g Brócoli" in meal["ingredients"], "el veg añadido debe quedar en el meal mutado"
