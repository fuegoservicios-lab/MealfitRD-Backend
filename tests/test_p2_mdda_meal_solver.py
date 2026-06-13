"""[P2-MDDA · 2026-06-13] Tests offline del layer STRING del solver (F3): parsing de
ingredientes-string del plan, rescaler, gramos hint-aware, y solve_meal_macros con
chequeo de consistencia coherence (recipe re-escalado ↔ macros reportados).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import (
    IngredientNutritionDB, rescale_ingredient_string, _split_qty_unit_name,
)

ROWS = [
    {"name": "Avena", "aliases": ["avena"], "kcal_per_100g": 382.1,
     "protein_g_per_100g": 13.2, "carbs_g_per_100g": 67.7, "fats_g_per_100g": 6.5},
    {"name": "Leche", "aliases": ["leche", "leche descremada"], "kcal_per_100g": 61.0,
     "protein_g_per_100g": 3.15, "carbs_g_per_100g": 4.8, "fats_g_per_100g": 3.25,
     "density_g_per_cup": 244.0},
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 107.0,
     "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.93},
    {"name": "Aceite de oliva", "aliases": ["aceite", "aceite de oliva"], "kcal_per_100g": 900.0,
     "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 100.0},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=ROWS)


# ───────────────────── _split_qty_unit_name ─────────────────────
def test_split_taza_with_hint():
    assert _split_qty_unit_name("0.5 taza de avena (50g)") == (0.5, "taza", "avena")


def test_split_grams_direct():
    assert _split_qty_unit_name("150g de pechuga de pollo") == (150.0, "g", "pechuga de pollo")


def test_split_no_quantity():
    qty, unit, name = _split_qty_unit_name("Sal al gusto")
    assert qty == 0.0 and "Sal" in name


def test_split_unicode_fraction():
    qty, unit, name = _split_qty_unit_name("½ taza de leche (120ml)")
    assert qty == pytest.approx(0.5) and unit == "taza"


# ───────────────────── rescale_ingredient_string ─────────────────────
def test_rescale_scales_lead_and_hint():
    assert rescale_ingredient_string("0.5 taza de avena (50g)", 2.0) == "1 taza de avena (100g)"


def test_rescale_grams_only():
    assert rescale_ingredient_string("150g de pechuga de pollo", 2.0) == "300g de pechuga de pollo"


def test_rescale_factor_one_noop():
    s = "0.5 taza de avena (50g)"
    assert rescale_ingredient_string(s, 1.0) == s


def test_rescale_no_quantity_unchanged():
    assert rescale_ingredient_string("Sal al gusto", 2.5) == "Sal al gusto"


def test_rescale_ml_hint():
    assert rescale_ingredient_string("1 taza de leche descremada (240ml)", 1.5) == \
        "1.5 taza de leche descremada (360ml)"


# ───────────────────── grams_from_ingredient_string ─────────────────────
def test_grams_uses_hint(db):
    assert db.grams_from_ingredient_string("0.5 taza de avena (50g)") == pytest.approx(50.0)


def test_grams_no_hint_uses_weight(db):
    assert db.grams_from_ingredient_string("150g de pechuga de pollo") == pytest.approx(150.0)


def test_macros_from_string(db):
    m = db.macros_from_ingredient_string("0.5 taza de avena (50g)")  # 50g avena
    assert m["protein"] == pytest.approx(13.2 * 0.5, abs=0.1)
    assert m["kcal"] == pytest.approx(382.1 * 0.5, abs=0.5)


# ───────────────────── solve_meal_macros + coherence ─────────────────────
def _recompute_from_strings(db, strings):
    """Suma macros recomputados desde los strings (verifica consistencia recipe↔macros)."""
    tot = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
    for s in strings:
        m = db.macros_from_ingredient_string(s)
        if m:
            for k in tot:
                tot[k] += m[k]
    return tot


def test_solve_meal_closes_protein_and_is_coherent(db):
    from portion_solver import solve_meal_macros
    ings = [
        "150g de pechuga de pollo",          # proteína
        "0.5 taza de avena (50g)",            # carbos
        "1 cucharada de aceite de oliva (14g)",  # grasa
    ]
    target = {"protein": 60, "carbs": 50, "fats": 25, "kcal": 665}
    res = solve_meal_macros(ings, target, db=db)
    # Déficit de proteína cerrado (pechuga escaló).
    assert res["achieved"]["protein"] >= 55
    # CONSISTENCIA COHERENCE: macros recomputados desde los strings re-escalados
    # == macros reportados (lo que verá el coherence guard al parsear la receta).
    recomputed = _recompute_from_strings(db, res["ingredients"])
    for macro in ("protein", "carbs", "fats"):
        assert recomputed[macro] == pytest.approx(res["achieved"][macro], abs=0.6), \
            f"{macro}: recipe={recomputed[macro]} vs reportado={res['achieved'][macro]}"


def test_solve_meal_preserves_format(db):
    from portion_solver import solve_meal_macros
    ings = ["150g de pechuga de pollo", "Sal al gusto"]
    res = solve_meal_macros(ings, {"protein": 45, "carbs": 0, "fats": 0}, db=db)
    # 'Sal al gusto' intacto; pechuga sigue siendo string con 'g'.
    assert "Sal al gusto" in res["ingredients"]
    chicken = [s for s in res["ingredients"] if "pechuga" in s][0]
    assert chicken.endswith("de pechuga de pollo") and "g " in chicken
