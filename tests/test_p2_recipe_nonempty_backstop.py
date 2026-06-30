"""[audit objetivo · P2-11 + P2-7] Backstop determinista de receta no-vacía + gate suave de dish-quality.

- P2-11 (P2-RECIPE-NONEMPTY-BACKSTOP): `_ensure_nonempty_recipe` rellena recipe vacía/no-sustantiva con un
  template de 3 pilares sustantivo (derivado de name+ingredients), cubriendo `recipe` ausente Y `[]`. Cableado
  en assemble, el finalizador de updates, y el persist boundary (finalize_plan_data_coherence).
- P2-7 (P2-DISH-QUALITY-GATE): soft gate gated-OFF en review + advisory en updates.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ───────────────────────── P2-11 ─────────────────────────
def test_knob_on_by_default():
    assert g.RECIPE_NONEMPTY_BACKSTOP_ENABLED is True


def test_fills_empty_recipe_with_substantive_steps():
    meal = {"name": "Pollo al horno", "ingredients": ["200g Pechuga de pollo", "1 batata"], "recipe": []}
    assert g._ensure_nonempty_recipe(meal) is True
    assert len(meal["recipe"]) == 3
    # Tras el fill, ya NO es 'receta no sustantiva' (los pasos superan el umbral del detector).
    lo, why = g._meal_dish_quality_issue(meal)
    assert lo is False, f"el plato no debería seguir flagueado: {why}"
    assert meal.get("_dish_quality_degraded") is True


def test_handles_absent_recipe_key():
    meal = {"name": "Ensalada", "ingredients": ["Lechuga", "Tomate"]}  # sin key 'recipe'
    assert g._ensure_nonempty_recipe(meal) is True
    assert len(meal.get("recipe", [])) == 3


def test_idempotent_on_real_recipe():
    meal = {"name": "Sancocho", "ingredients": ["Yuca", "Pollo"], "recipe": [
        "Mise en place: pela y trocea la yuca y limpia el pollo en piezas medianas.",
        "El Toque de Fuego: sofríe el pollo, añade agua y la yuca, hierve 40 minutos hasta espesar.",
        "Montaje: sirve caliente con aguacate y un toque de limón.",
    ]}
    before = list(meal["recipe"])
    assert g._ensure_nonempty_recipe(meal) is False
    assert meal["recipe"] == before


def test_single_generic_step_is_filled():
    # Un solo paso genérico ('Cocinar') NO es sustantivo → se rellena.
    meal = {"name": "Plato", "ingredients": ["Arroz"], "recipe": ["Cocinar"]}
    assert g._ensure_nonempty_recipe(meal) is True
    assert len(meal["recipe"]) == 3


def test_wired_in_persist_boundary_and_assemble():
    assert "recipe_fill=" in _GRAPH, "no se cableó en finalize_plan_data_coherence (persist boundary)"
    assert _GRAPH.count("_ensure_nonempty_recipe(") >= 3, "debe cablearse en assemble + updates + persist"


# ───────────────────────── P2-7 ─────────────────────────
def test_dish_quality_soft_gate_off_by_default():
    # Default OFF (A/B-pending): encenderlo cambia review_plan_node para cualquier receta no-sustantiva (blast
    # radius amplio). Los disparates conocidos los cubren los fixes deterministas + el prompt P1-DISH-PALATABILITY.
    assert g.DISH_QUALITY_SOFT_GATE_ENABLED is False
    assert 0.1 <= g.DISH_QUALITY_REJECT_RATIO <= 1.0


def test_dish_quality_gate_wired_in_review():
    assert "P2-DISH-QUALITY-GATE" in _GRAPH
    assert "DISH_QUALITY_SOFT_GATE_ENABLED" in _GRAPH
    # advisory en updates (paridad de telemetría)
    assert "_meal_dish_quality_issue(meal)" in _GRAPH
