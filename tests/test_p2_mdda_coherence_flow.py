"""[P3-MACRO-SOLVER · 2026-06-13] F3.2 — los ingredientes re-escalados por el solver
fluyen CORRECTAMENTE al coherence guard + shopping aggregator.

El coherence guard (expected_sum_from_recipes) y el shopping aggregator parsean las
MISMAS strings de ingredientes vía shopping_calculator._parse_quantity. Como el solver
re-escribe esa única fuente (cantidad líder + hint), recipe↔shopping quedan consistentes
por construcción. Este test prueba que las strings re-escaladas siguen siendo parseables
por el _parse_quantity REAL y con la cantidad correctamente escalada (no rotas).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import rescale_ingredient_string
import shopping_calculator as sc


@pytest.mark.parametrize("original,factor,exp_qty,exp_unit", [
    ("150g de pechuga de pollo", 2.0, 300.0, "g"),
    ("0.5 taza de avena (50g)", 2.0, 1.0, "taza"),
    ("1 taza de leche descremada (240ml)", 1.5, 1.5, "taza"),
    ("3 cucharadas de aceite de oliva (45ml)", 2.0, 6.0, "cda"),
])
def test_rescaled_string_parses_with_scaled_qty(original, factor, exp_qty, exp_unit):
    rescaled = rescale_ingredient_string(original, factor)
    # El _parse_quantity REAL (que usan coherence guard + shopping aggregator) parsea
    # la string re-escalada y extrae la cantidad escalada — no se rompió el formato.
    qty, unit, _name = sc._parse_quantity(rescaled, apply_yield_multiplier=False)
    assert qty == pytest.approx(exp_qty), f"{rescaled!r}: qty {qty} != {exp_qty}"
    assert unit == exp_unit


def test_no_quantity_item_survives_parse():
    rescaled = rescale_ingredient_string("Sal al gusto", 2.0)
    assert rescaled == "Sal al gusto"
    qty, unit, _ = sc._parse_quantity(rescaled, apply_yield_multiplier=False)
    assert qty == 0.0  # 'al gusto' → nominal 0, listado pero sin alterar magnitud


def test_recipe_shopping_consistency_single_source():
    # Recipe-sum y shopping parsean la MISMA string → escalar la string mantiene
    # ambos lados en lock-step. Verificamos que el ratio de qty == factor exacto.
    original = "120g de pechuga de pollo"
    factor = 1.8
    q0, _, _ = sc._parse_quantity(original, apply_yield_multiplier=False)
    q1, _, _ = sc._parse_quantity(
        rescale_ingredient_string(original, factor), apply_yield_multiplier=False)
    assert q1 / q0 == pytest.approx(factor, abs=0.01)


def test_solver_runs_before_shopping_aggregation_in_assemble():
    """[P3-MACRO-SOLVER ORDER] El solver DEBE re-escalar las porciones ANTES de que
    se agregue la lista de compras en assemble_plan_node — si corre después, la lista
    queda con cantidades pre-solver y diverge de las recetas (bug live 2026-06-13: 60
    divergencias receta↔lista). Ancla el orden por parsing del source de prod.
    Tooltip-anchor: P3-MACRO-SOLVER."""
    import os as _os
    backend = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    src = open(_os.path.join(backend, "graph_orchestrator.py"), encoding="utf-8").read()
    a = src.find("async def assemble_plan_node")
    assert a > 0, "assemble_plan_node no encontrado"
    body = src[a:src.find("async def", a + 20)]
    i_solver = body.find("[P3-MACRO-SOLVER] Re-escaló porciones")
    i_shop = body.find("# Calcular shopping lists")
    i_human = body.find("Humanizar ingredientes a medidas caseras")
    assert i_solver > 0 and i_shop > 0 and i_human > 0, "marcadores de orden ausentes"
    assert i_solver < i_shop, "el solver debe correr ANTES de la agregación de compras"
    assert i_shop < i_human, "la humanización debe correr DESPUÉS del shopping (orden legacy)"
