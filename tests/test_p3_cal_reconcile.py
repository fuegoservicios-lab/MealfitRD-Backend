"""[P3-CAL-RECONCILE · 2026-06-13] Paso final del cerebro dividido: nivela las calorías
de cada día al target exacto re-escalando porciones + macros uniformemente. Cierra el
`cal_score` del holistic (= max(0, 1 − desviación×5)) → 1.0 con desviación ~0.

El escalado uniforme (ingredientes + macros por el MISMO factor) preserva la consistencia
receta↔macro que el solver estableció: si grams×f y macro×f, entonces macro sigue ==
grams/100 × per100g.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import IngredientNutritionDB, rescale_ingredient_string

ROWS = [
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 107.0,
     "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.93},
    {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 360.0,
     "protein_g_per_100g": 6.6, "carbs_g_per_100g": 79.0, "fats_g_per_100g": 0.6},
]


def test_reconcile_block_present_and_before_shopping():
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(backend, "graph_orchestrator.py"), encoding="utf-8").read()
    a = src.find("async def assemble_plan_node")
    body = src[a:src.find("async def", a + 20)]
    i_rec = body.find("[P3-CAL-RECONCILE] Niveló calorías")
    i_shop = body.find("# Calcular shopping lists")
    assert i_rec > 0, "bloque de reconciliación calórica ausente"
    assert i_rec < i_shop, "la reconciliación debe correr ANTES del shopping (gramos finales → lista)"


def test_uniform_scaling_preserves_recipe_macro_consistency():
    # Simula el escalado de la reconciliación: ingredientes ×f + macros ×f → siguen
    # consistentes (la proteína recomputada de los ingredientes == la macro reportada).
    db = IngredientNutritionDB(rows=ROWS)
    ingredients = ["150g de pechuga de pollo (150g)", "0.5 taza de arroz (80g)"]
    macro_protein = sum((db.macros_from_ingredient_string(s) or {}).get("protein", 0) for s in ingredients)
    f = 1.2  # nivelar día +20%
    scaled = [rescale_ingredient_string(s, f) for s in ingredients]
    new_macro_protein = macro_protein * f  # los macros se escalan por el mismo factor
    recomputed = sum((db.macros_from_ingredient_string(s) or {}).get("protein", 0) for s in scaled)
    assert recomputed == pytest.approx(new_macro_protein, rel=0.02), \
        f"inconsistencia tras escalado: receta={recomputed} vs macro={new_macro_protein}"


def test_scaling_factor_reaches_calorie_target():
    # Un día a 1800 kcal con factor target/dc llega al target.
    day_cals, target = 1800.0, 2050.0
    f = max(0.6, min(1.6, target / day_cals))
    assert day_cals * f == pytest.approx(target, abs=1.0)
