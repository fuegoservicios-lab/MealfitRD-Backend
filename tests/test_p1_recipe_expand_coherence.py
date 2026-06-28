"""[P1-RECIPE-EXPAND-COHERENCE · 2026-06-28] El endpoint /recipe/expand (Chef AI "Masterclass") podía generar pasos que
mencionan alimentos fuera de la lista de ingredientes (proteína-fantasma tipo cap_swallowed_modifier). Fix: (a) el
prompt RECIPE_EXPANSION_PROMPT prohíbe mencionar alimentos no listados (regla 5); (b) validador post-LLM soft-fail
(reusa validate_meal_recipe_ingredients_coherence + knob MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE) → no persiste pasos
incoherentes, HTTP 200 + operation_failed para retry sin error rojo.
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc


def test_prompt_has_coherence_rule():
    from prompts import meal_operations
    p = meal_operations.RECIPE_EXPANSION_PROMPT
    assert "COHERENCIA RECETA" in p or "EXCLUSIVAMENTE los alimentos" in p
    # ejemplos concretos del error a evitar
    assert "queso cottage" in p.lower() or "dorado" in p.lower()


def test_validator_catches_phantom_protein():
    # receta menciona "pollo" pero ingredientes no lo tienen → divergencia
    passed, divs, summary = nc.validate_meal_recipe_ingredients_coherence({
        "ingredients": ["200g de pavo", "1 taza de arroz"],
        "recipe": ["Mise en place: corta el pollo en cubos.", "El Toque de Fuego: dora el pollo."],
    })
    assert passed is False
    assert divs  # hay al menos una proteína fantasma


def test_validator_passes_coherent():
    passed, divs, summary = nc.validate_meal_recipe_ingredients_coherence({
        "ingredients": ["200g de pavo", "1 taza de arroz"],
        "recipe": ["Mise en place: corta el pavo.", "El Toque de Fuego: dora el pavo con el arroz."],
    })
    assert passed is True
    assert not divs


def test_handler_softfail_wired():
    # el handler de expand invoca el validador y retorna soft-fail (coherence_check_failed) sin persistir
    src = (Path(nc.__file__).resolve().parent / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "P1-RECIPE-EXPAND-COHERENCE" in src
    assert "validate_meal_recipe_ingredients_coherence" in src
    assert "coherence_check_failed" in src
    # debe correr ANTES de log_api_usage (no cobrar por receta incoherente)
    i_val = src.index("P1-RECIPE-EXPAND-COHERENCE")
    i_charge = src.index('log_api_usage(user_id, "llm_recipe_expand")')
    assert i_val < i_charge, "el validador debe correr antes de cobrar cuota"


def test_knob_reused_not_new():
    # reusa el knob existente de swap/modify, NO crea uno nuevo
    src = (Path(nc.__file__).resolve().parent / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "_swap_recipe_coherence_enabled" in src
    assert "MEALFIT_RECIPE_EXPAND_COHERENCE" not in src  # no knob nuevo
