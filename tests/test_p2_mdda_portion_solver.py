"""[P2-MDDA · 2026-06-13] Tests offline del lado determinista del cerebro dividido:
nutrition_db (lookup + to_grams) + portion_solver + allocate_macros_per_slot.

100% offline: inyecta `rows` fixture a IngredientNutritionDB (sin DB ni USDA).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import IngredientNutritionDB, NutritionInfo


# Filas fixture con el shape de master_ingredients (SELECT *) + macros poblados.
FIXTURE_ROWS = [
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"],
     "kcal_per_100g": 107.0, "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 1.93, "fiber_g_per_100g": 0.0, "sodium_mg_per_100g": 45,
     "nutrition_source": "usda", "fdc_id": 2646170, "is_dominican_cultivar": False},
    {"name": "Arroz blanco", "aliases": ["arroz"],
     "kcal_per_100g": 360.0, "protein_g_per_100g": 6.6, "carbs_g_per_100g": 79.0,
     "fats_g_per_100g": 0.6, "nutrition_source": "usda", "fdc_id": 169756},
    {"name": "Aceite de oliva", "aliases": ["aceite"],
     "kcal_per_100g": 900.0, "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 100.0, "nutrition_source": "usda"},
    {"name": "Huevo", "aliases": ["huevos", "huevo entero"],
     "kcal_per_100g": 143.0, "protein_g_per_100g": 12.6, "carbs_g_per_100g": 0.7,
     "fats_g_per_100g": 9.5, "density_g_per_unit": 50.0, "nutrition_source": "usda"},
    {"name": "Sal", "aliases": [], "kcal_per_100g": 0.0, "protein_g_per_100g": 0.0,
     "carbs_g_per_100g": 0.0, "fats_g_per_100g": 0.0, "nutrition_source": "manual"},
    # Sin macros poblados (kcal NULL) → lookup debe degradar a None.
    {"name": "Casabe", "aliases": ["casabe de yuca"], "kcal_per_100g": None},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=FIXTURE_ROWS)


# ───────────────────────── nutrition_db: lookup ─────────────────────────
def test_lookup_exact_name(db):
    info = db.lookup("Pechuga de pollo")
    assert info is not None and info.name == "Pechuga de pollo"
    assert info.protein == 22.5 and info.source == "usda" and info.fdc_id == 2646170


def test_lookup_via_alias(db):
    assert db.lookup("pollo").name == "Pechuga de pollo"
    assert db.lookup("arroz").name == "Arroz blanco"


def test_lookup_strips_parens_and_accents(db):
    # "Pechuga de Pollo (a la plancha)" → quita paréntesis, case/acentos.
    assert db.lookup("Pechuga de Pollo (a la plancha)").name == "Pechuga de pollo"


def test_lookup_alias_word_boundary(db):
    # "huevos revueltos" matchea alias "huevos" por word-boundary.
    assert db.lookup("huevos revueltos").name == "Huevo"


def test_lookup_no_match_returns_none(db):
    assert db.lookup("dragonfruit marciano") is None


def test_lookup_no_macros_returns_none(db):
    # Casabe existe pero kcal_per_100g IS NULL → degradar, no inventar.
    assert db.lookup("Casabe") is None


# ───────────────────────── nutrition_db: to_grams ───────────────────────
def test_to_grams_weight_direct(db):
    info = db.lookup("Pechuga de pollo")
    assert db.to_grams(150, "g", info) == pytest.approx(150.0)
    assert db.to_grams(1, "lb", info) == pytest.approx(453.592, rel=1e-4)
    assert db.to_grams(2, "oz", info) == pytest.approx(56.699, rel=1e-3)


def test_to_grams_unidad_uses_density(db):
    info = db.lookup("Huevo")
    assert db.to_grams(2, "unidad", info) == pytest.approx(100.0)  # 2×50g


def test_to_grams_unidad_without_density_returns_none(db):
    info = db.lookup("Pechuga de pollo")  # sin density_g_per_unit
    assert db.to_grams(1, "unidad", info) is None


def test_to_grams_container_without_weight_returns_none(db):
    info = db.lookup("Arroz blanco")  # sin container_weight_g
    assert db.to_grams(1, "paquete", info) is None


def test_macros_for_line_scales_linearly(db):
    m = db.macros_for_line(200, "g", "pollo")  # 200g pechuga
    assert m["protein"] == pytest.approx(45.0)   # 22.5×2
    assert m["kcal"] == pytest.approx(214.0)
    assert m["grams"] == 200.0


# ───────────────────────── portion_solver ───────────────────────────────
def test_solver_closes_protein_deficit(db):
    from portion_solver import solve_portion_macros
    # Slot pobre en proteína: 100g pollo (22.5g P) vs target 50g P.
    ingredients = [
        {"name": "pollo", "quantity": 100, "unit": "g"},
        {"name": "arroz", "quantity": 80, "unit": "g"},
        {"name": "aceite", "quantity": 10, "unit": "g"},
    ]
    target = {"kcal": 700, "protein": 50, "carbs": 60, "fats": 25}
    res = solve_portion_macros(ingredients, target, db=db)
    # Déficit CERRADO: proteína alcanzada >= target. El solver proporcional v1
    # sobre-entrega levemente porque las fuentes de carbo/grasa (arroz) también
    # cargan proteína sobre los 50g del pollo — aceptable (sobre-entrega ~10% es
    # muchísimo mejor que el déficit -30% actual; LP cruzado solo si telemetría lo pide).
    assert res["achieved"]["protein"] >= 50
    assert res["achieved"]["protein"] <= 50 * 1.20  # dentro de +20%
    assert res["report"]["protein"]["applied"] is True
    # El pollo creció (factor>1), el aceite (grasa) también escaló a su target.
    chicken = next(i for i in res["ingredients"] if i["name"] == "pollo")
    assert chicken["quantity"] > 100


def test_solver_leaves_unresolved_untouched(db):
    from portion_solver import solve_portion_macros
    ingredients = [
        {"name": "pollo", "quantity": 100, "unit": "g"},
        {"name": "ingrediente fantasma", "quantity": 1, "unit": "unidad"},
    ]
    res = solve_portion_macros(ingredients, {"protein": 30, "carbs": 0, "fats": 0}, db=db)
    ghost = next(i for i in res["ingredients"] if i["name"] == "ingrediente fantasma")
    assert ghost["quantity"] == 1  # intacto
    assert res["unresolved"] == 1 and res["resolved_count"] == 1


def test_solver_clamps_extreme_factor(db):
    from portion_solver import solve_portion_macros
    # Target absurdo (500g P de 22.5g) → factor clamp a max_scale.
    res = solve_portion_macros(
        [{"name": "pollo", "quantity": 100, "unit": "g"}],
        {"protein": 500, "carbs": 0, "fats": 0}, db=db, max_scale=3.5,
    )
    assert res["report"]["protein"]["factor"] == pytest.approx(3.5)
    assert res["converged"] is False  # no pudo alcanzar el target absurdo


def test_solver_dominant_macro_classification(db):
    from portion_solver import solve_portion_macros
    # aceite es 100% grasa → debe clasificar al grupo 'fats', no tocar con protein.
    res = solve_portion_macros(
        [{"name": "aceite", "quantity": 10, "unit": "g"}],
        {"protein": 100, "carbs": 0, "fats": 20}, db=db,
    )
    oil = res["ingredients"][0]
    assert oil["quantity"] == pytest.approx(20.0)  # 10g × (20/10) fats factor


# ───────────────────────── allocate_macros_per_slot ─────────────────────
def test_allocate_4_meal_split_sums_to_daily():
    from nutrition_calculator import allocate_macros_per_slot
    daily = {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fats_g": 60}
    slots = allocate_macros_per_slot(daily, num_meals=4)
    assert set(slots) == {"desayuno", "almuerzo", "merienda", "cena"}
    assert sum(s["protein"] for s in slots.values()) == pytest.approx(150, abs=0.5)
    assert sum(s["kcal"] for s in slots.values()) == pytest.approx(2000, abs=0.5)
    # almuerzo = 35% del día.
    assert slots["almuerzo"]["protein"] == pytest.approx(150 * 0.35, abs=0.1)


def test_allocate_accepts_aliases_and_default_meals():
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot({"kcal": 1800, "protein": 120, "carbs": 180, "fats": 50})
    assert sum(s["fats"] for s in slots.values()) == pytest.approx(50, abs=0.5)


def test_allocate_custom_splits_override():
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot(
        {"kcal": 1000, "protein": 100, "carbs": 100, "fats": 40},
        splits={"unica": 1.0},
    )
    assert slots["unica"]["protein"] == pytest.approx(100)
