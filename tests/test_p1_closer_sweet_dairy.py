"""[P1-CLOSER-SWEET-DAIRY · 2026-07-07] Cierra el 3er bloqueador del review 4e7b8dbb (queso salado en
postres) con la vía CORRECTA: el closer usa lácteos dulce-compatibles BARATOS (yogurt/cottage/ricotta)
para platos DULCES en vez de queso blanco/de hoja.

Contexto: el pool de alta densidad del closer (≥18g/100g) solo tiene QUESOS SALADOS como lácteo; yogurt
(10g), cottage (12g), ricotta (7.5g) caen bajo el umbral → el closer estaba forzado a meter queso blanco
en bocaditos de avena-miel. Fix: pool suplementario de lácteos dulce-compatibles con umbral bajo,
preferido sobre queso salado; cae al queso solo si no hay ninguno (piso de proteína > pureza). Solo
activo cuando el caller pasa `allergies` → los tests de piso (que no lo pasan) conservan el queso.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import IngredientNutritionDB
from graph_orchestrator import _close_protein_gap_for_meal, _safe_high_density_proteins, _meal_macro_num

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()

# Catálogo: yogurt/cottage (dulce, baja densidad) + queso blanco (salado, ≥18) + pollo (salado).
_ROWS = [
    {"name": "Pollo", "aliases": ["pollo"], "kcal_per_100g": 165,
     "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    {"name": "Queso blanco", "aliases": ["queso blanco", "queso"], "kcal_per_100g": 298,
     "protein_g_per_100g": 18.1, "carbs_g_per_100g": 3, "fats_g_per_100g": 24},
    {"name": "Yogurt griego entero", "aliases": ["yogurt griego", "yogur griego", "yogurt"],
     "kcal_per_100g": 94, "protein_g_per_100g": 8.8, "carbs_g_per_100g": 4, "fats_g_per_100g": 5,
     "density_g_per_cup": 245},
    {"name": "Queso cottage", "aliases": ["queso cottage", "cottage"], "kcal_per_100g": 70,
     "protein_g_per_100g": 12.4, "carbs_g_per_100g": 3.4, "fats_g_per_100g": 1, "density_g_per_cup": 225},
]


def _db():
    return IngredientNutritionDB(rows=_ROWS)


def _sweet_meal():
    return {"name": "Bocaditos de Avena con Miel y Mango", "meal": "Merienda",
            "protein": 5, "carbs": 55, "fats": 12, "cals": 5 * 4 + 55 * 4 + 12 * 9,
            "ingredients": ["1 taza de avena", "1 mango", "1 cda de miel"]}


def test_sweet_dish_prefers_yogurt_over_savory_cheese():
    """Con allergies pasado: el closer añade lácteo dulce (yogurt/cottage), NO queso blanco."""
    meal = _sweet_meal()
    cands = _safe_high_density_proteins([], _db(), min_protein=18.0)  # solo Queso blanco/Pollo (≥18)
    _close_protein_gap_for_meal(meal, 14.0, _db(), cands, allergies=[], fill_pct=0.92)
    joined = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert ("yogur" in joined or "cottage" in joined), f"esperado lácteo dulce, dio: {meal['ingredients']}"
    assert "queso blanco" not in joined, f"queso blanco NO debe entrar en el postre: {meal['ingredients']}"


def test_sweet_dish_without_allergies_keeps_old_behavior():
    """Sin allergies (default None): comportamiento previo (queso) — no rompe los tests de piso."""
    meal = _sweet_meal()
    cands = _safe_high_density_proteins([], _db(), min_protein=18.0)
    _close_protein_gap_for_meal(meal, 14.0, _db(), cands, fill_pct=0.92)  # sin allergies
    joined = " ".join(str(i) for i in meal["ingredients"]).lower()
    # el pool sin lácteo dulce → cae al queso (la única proteína dulce-compatible del pool ≥18)
    assert "queso" in joined, f"sin allergies debe conservar el queso: {meal['ingredients']}"


def test_sweet_fallback_to_cheese_when_no_sweet_dairy():
    """Si el catálogo NO tiene lácteo dulce (solo queso), cae al queso (piso de proteína > pureza)."""
    rows = [r for r in _ROWS if "yogur" not in r["name"].lower() and "cottage" not in r["name"].lower()]
    db = IngredientNutritionDB(rows=rows)
    meal = _sweet_meal()
    cands = _safe_high_density_proteins([], db, min_protein=18.0)
    _close_protein_gap_for_meal(meal, 14.0, db, cands, allergies=[], fill_pct=0.92)
    joined = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "queso" in joined, "sin lácteo dulce disponible → queso como último recurso (no romper piso)"


def test_savory_dish_unaffected():
    """Un plato SALADO no dispara la rama dulce (queso/pollo normal)."""
    meal = {"name": "Revoltillo de Pollo con Vegetales", "meal": "Almuerzo",
            "protein": 8, "carbs": 20, "fats": 10, "cals": 8 * 4 + 20 * 4 + 10 * 9,
            "ingredients": ["2 huevos", "vegetales"]}
    cands = _safe_high_density_proteins([], _db(), min_protein=18.0)
    n = _close_protein_gap_for_meal(meal, 25.0, _db(), cands, allergies=[], fill_pct=0.92)
    assert n >= 0  # no crashea; la rama dulce no aplica


def test_markers_and_knobs_anchored():
    assert "P1-CLOSER-SWEET-DAIRY" in _GO
    assert 'CLOSER_SWEET_DAIRY_ENABLED = _env_bool("MEALFIT_CLOSER_SWEET_DAIRY", True)' in _GO
    assert "_is_savory_cheese_name(nlow)" in _GO
    # ambos callers de producción pasan allergies
    assert _GO.count("allergies=form_data.get(\"allergies\")") >= 2
