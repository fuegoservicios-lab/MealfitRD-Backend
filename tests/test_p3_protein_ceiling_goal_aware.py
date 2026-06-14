"""[P3-PROTEIN-CEILING-GOAL-AWARE · 2026-06-13] El techo de proteína ENTREGADA es goal-aware
en g/kg: ≤2.2 g/kg para volumen/mantenimiento (más proteína desplaza carbos útiles), hasta
~2.6 g/kg en DÉFICIT (proteína alta preserva músculo al perder grasa — respaldado por evidencia).

Ajuste fino tras observar que un plan de pérdida de grasa entregaba 2.43 g/kg en un día: para
déficit eso es protector (correcto), pero para volumen NO debe pasar de 2.2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import (
    _protein_gkg_ceiling, _weight_kg_from_form, _trim_day_protein_to_ceiling, _meal_macro_num,
)
from nutrition_db import IngredientNutritionDB

_ROWS = [
    {"name": "Pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165,
     "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    {"name": "Arroz", "aliases": ["arroz"], "kcal_per_100g": 130,
     "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3, "density_g_per_cup": 158},
]


def _db():
    return IngredientNutritionDB(rows=_ROWS)


def test_techo_gkg_por_objetivo():
    assert _protein_gkg_ceiling("gain_muscle") == 2.2
    assert _protein_gkg_ceiling("maintenance") == 2.2
    assert _protein_gkg_ceiling("performance") == 2.2
    assert _protein_gkg_ceiling("lose_fat") == 2.6
    assert _protein_gkg_ceiling("Pérdida de Grasa (Déficit 20%)") == 2.6


def test_weight_kg_convierte_lb():
    assert abs(_weight_kg_from_form({"weight": 120, "weightUnit": "lb"}) - 54.43) < 0.1
    assert _weight_kg_from_form({"weight": 65, "weightUnit": "kg"}) == 65.0
    assert _weight_kg_from_form({}) == 0.0


def _ceil_pct(goal, weight_kg, target_pg):
    mx = _protein_gkg_ceiling(goal) * weight_kg
    return max(1.0, min(1.30, mx / target_pg)) if (mx > 0 and target_pg > 0) else 1.12


def test_volumen_trima_por_encima_de_2_2_gkg():
    # Volumen, 54.4kg, target 120g (2.2 g/kg). Un día a 140g (2.57 g/kg) → trim a ~2.2 g/kg.
    wkg = _weight_kg_from_form({"weight": 120, "weightUnit": "lb"})
    pct = _ceil_pct("gain_muscle", wkg, 120.0)
    meals = [{"name": "A", "protein": 140, "carbs": 40, "fats": 12, "cals": 140*4+40*4+12*9,
              "ingredients": ["450g de pollo (450g)", "1 taza de arroz (158g)"]}]
    trimmed = _trim_day_protein_to_ceiling(meals, 120.0, _db(), ceiling_pct=pct)
    assert trimmed
    P = sum(_meal_macro_num(m["protein"]) for m in meals)
    assert P / wkg <= 2.25, f"volumen quedó en {P/wkg:.2f} g/kg (>2.2)"


def test_helper_ceiling_pct_robusto_a_peso_ausente():
    from graph_orchestrator import _goal_aware_trim_ceiling_pct
    # con peso: volumen 54kg target 119g → ~1.0 (estricto 2.2 g/kg)
    pct_vol = _goal_aware_trim_ceiling_pct({"weight": 54, "weightUnit": "kg", "mainGoal": "gain_muscle"}, 119.0)
    assert abs(pct_vol - 1.0) < 0.02
    # con peso: déficit 65kg target 140g → ~1.21 (permite hasta 2.6 g/kg)
    pct_cut = _goal_aware_trim_ceiling_pct({"weight": 65, "weightUnit": "kg", "mainGoal": "lose_fat"}, 140.0)
    assert 1.18 < pct_cut < 1.25
    # SIN peso: fallback por objetivo (volumen estricto < déficit laxo)
    f_vol = _goal_aware_trim_ceiling_pct({"mainGoal": "gain_muscle"}, 120.0)
    f_cut = _goal_aware_trim_ceiling_pct({"mainGoal": "lose_fat"}, 120.0)
    assert f_vol < f_cut


def test_surgical_regen_reaplica_techo():
    # Ancla: el nodo de surgical regen re-aplica el trim de techo a los días re-generados.
    import ast as _ast
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "graph_orchestrator.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    node = next((n for n in _ast.walk(tree)
                 if isinstance(n, _ast.AsyncFunctionDef) and n.name == "surgical_marker_regen_node"), None)
    assert node is not None
    body = _ast.get_source_segment(src, node)
    assert "_trim_day_protein_to_ceiling" in body and "_goal_aware_trim_ceiling_pct" in body


def test_deficit_permite_proteina_alta_protectora():
    # Déficit, 65kg, target 140g (2.15 g/kg). Un día a 158g (2.43 g/kg) → NO se trima (protector).
    wkg = 65.0
    pct = _ceil_pct("lose_fat", wkg, 140.0)
    meals = [{"name": "A", "protein": 158, "carbs": 30, "fats": 10, "cals": 158*4+30*4+10*9,
              "ingredients": ["500g de pollo (500g)"]}]
    trimmed = _trim_day_protein_to_ceiling(meals, 140.0, _db(), ceiling_pct=pct)
    assert not trimmed, "déficit a 2.43 g/kg NO debe trimarse (preservación muscular)"
    assert _meal_macro_num(meals[0]["protein"]) == 158
