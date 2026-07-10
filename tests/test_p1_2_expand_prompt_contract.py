"""[P1-2-EXPAND-CONTRACT · 2026-07-10] (recipe plausibility roadmap, item P1-2) `RECIPE_EXPANSION_PROMPT`
ya prohíbe alimentos-fantasma (regla 5, P1-RECIPE-EXPAND-COHERENCE) pero NO decía nada sobre: (a)
cantidades inventadas en los pasos (el Chef puede escribir "2 cdas" en el paso mientras `ingredients`
dice "1.25 cdas" — el backend re-sincroniza post-hoc vía `_sync_recipe_step_quantities`, pero es más
barato que el LLM no invente el número primero); (b) "según las instrucciones del paquete" para
alimentos FRESCOS (evidencia viva: "Cocer el Batata según instrucciones del paquete" — una batata fresca
no trae paquete); (c) verbos de cocción absurdos sobre proteína cruda ("Batir ligeramente pechuga de
pollo" — se cocina, no se bate cruda).
"""
from __future__ import annotations

from prompts import meal_operations as mo


def test_prompt_has_quantity_coherence_rule():
    p = mo.RECIPE_EXPANSION_PROMPT
    assert "cantidad" in p.lower()
    assert "ingredients_json" in p or "Ingredientes" in p, "la regla debe anclar contra la lista real"


def test_prompt_forbids_package_instructions_for_fresh_produce():
    p = mo.RECIPE_EXPANSION_PROMPT
    assert "paquete" in p.lower() or "empaquetad" in p.lower()
    # ejemplo concreto que ancla el error real observado (batata fresca)
    assert "fresc" in p.lower()


def test_prompt_forbids_absurd_verb_food_pairings():
    p = mo.RECIPE_EXPANSION_PROMPT
    assert "batir" in p.lower() or "cruda" in p.lower() or "cruda" in p.lower()
