"""[P1-PANTRY-SUFFICIENCY · 2026-06-23] Tests del evaluador de suficiencia de la Nevera.

`evaluate_pantry_sufficiency(user_id, form_data, scope, ...)` decide si la Nevera
alcanza para generar un plato/día acorde al objetivo. La PALANCA es la proteína; los
micros son advisory por default; ítems no resolubles a gramos NO se cuentan; ante
error interno hace fail-open. Estos tests inyectan inventario + master rows + targets
para ser deterministas offline (sin DB, sin Cohere)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nutrition_db import IngredientNutritionDB
import inventory_sufficiency as isuf
from inventory_sufficiency import evaluate_pantry_sufficiency, available_macros_from_inventory


# Master rows inyectados (per-100g). lookup() + to_grams() operan sobre éstos offline.
_MASTER = [
    {"name": "Pollo", "kcal_per_100g": 165, "protein_g_per_100g": 31.0,
     "carbs_g_per_100g": 0.0, "fats_g_per_100g": 3.6, "iron_mg_per_100g": 1.0,
     "potassium_mg_per_100g": 256.0},
    {"name": "Arroz", "kcal_per_100g": 130, "protein_g_per_100g": 2.7,
     "carbs_g_per_100g": 28.0, "fats_g_per_100g": 0.3},
    {"name": "Huevo", "kcal_per_100g": 143, "protein_g_per_100g": 13.0,
     "carbs_g_per_100g": 1.1, "fats_g_per_100g": 9.5, "density_g_per_unit": 50.0,
     "iron_mg_per_100g": 1.8},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=_MASTER)


@pytest.fixture(autouse=True)
def _stub_targets(monkeypatch):
    """Targets deterministas: día = 2100 kcal / 124 g proteína (gain_muscle típico)."""
    monkeypatch.setattr(
        "nutrition_calculator.get_nutrition_targets",
        lambda fd: {"target_calories": 2100, "macros": {"protein_g": 124.0, "carbs_g": 270.0, "fats_g": 58.0}},
    )
    monkeypatch.setattr(
        "micronutrients.dri_targets",
        lambda sex=None, age=None, pregnant=False: {
            "iron_mg": {"floor": 8.0, "unit": "mg"},
            "potassium_mg": {"floor": 3400.0, "unit": "mg"},
            "fiber_g": {"floor": 38.0, "unit": "g"},
        },
    )


_FORM = {"goal": "gain_muscle", "gender": "male", "age": 30}
_MEAL_TARGET = {"kcal": 500, "protein_g": 40.0, "carbs_g": 50.0, "fats_g": 15.0}


# --------------------------------------------------------------------------
# available_macros_from_inventory
# --------------------------------------------------------------------------
def test_macros_summed_from_inventory(db):
    panel, uncountable = available_macros_from_inventory(
        [{"ingredient_name": "Pollo", "available_quantity": 1.0, "unit": "lb"}], db
    )
    # 1 lb ≈ 453.6 g × 31/100 ≈ 140.6 g proteína.
    assert 130 < panel["protein_g"] < 150
    assert uncountable == []


def test_uncountable_items_excluded(db):
    panel, uncountable = available_macros_from_inventory(
        [{"ingredient_name": "Laurel", "available_quantity": 1.0, "unit": "hoja"},
         {"ingredient_name": "Pollo", "available_quantity": 1.0, "unit": "lb"}], db
    )
    assert "Laurel" in uncountable          # sin master row → no contado
    assert panel["protein_g"] > 100         # el pollo SÍ cuenta


def test_reserved_quantity_respected(db):
    # available_quantity (no quantity) es lo que se cuenta.
    panel, _ = available_macros_from_inventory(
        [{"ingredient_name": "Pollo", "quantity": 2.0, "available_quantity": 0.0, "unit": "lb"}], db
    )
    assert panel["protein_g"] == 0.0


# --------------------------------------------------------------------------
# evaluate_pantry_sufficiency — gate de proteína (palanca)
# --------------------------------------------------------------------------
def test_rich_inventory_sufficient(db):
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal", meal_target=_MEAL_TARGET, nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out["sufficient"] is True
    assert out["coverage"]["protein_g"] > 1.0
    assert out["message"] is None


def test_sparse_inventory_insufficient_protein(db):
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal", meal_target=_MEAL_TARGET, nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Arroz", "available_quantity": 0.2, "unit": "lb"}],
    )
    assert out["sufficient"] is False
    prot = [d for d in out["deficits"] if d["nutrient"] == "protein_g"]
    assert prot and prot[0]["advisory"] is False
    assert out["message"] and "proteína" in out["message"].lower()


def test_empty_inventory_insufficient(db):
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal", meal_target=_MEAL_TARGET, nutrition_db=db, inventory_rows=[],
    )
    assert out["sufficient"] is False


def test_day_scope_requires_full_daily(db):
    # 2 lb pollo (~280 g prot, ~1500 kcal) + 2 lb arroz (~1180 kcal) → cubre prot Y kcal del DÍA.
    out_ok = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="day", nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 2.0, "unit": "lb"},
                        {"ingredient_name": "Arroz", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out_ok["sufficient"] is True
    assert out_ok["required"]["protein_g"] == 124.0
    # Medio lb de pollo = ~70 g proteína < 124 × 0.90 → insuficiente para el día.
    out_bad = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="day", nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 0.5, "unit": "lb"}],
    )
    assert out_bad["sufficient"] is False


# --------------------------------------------------------------------------
# Micros advisory + fail-open
# --------------------------------------------------------------------------
def test_micros_advisory_does_not_block(db, monkeypatch):
    monkeypatch.setattr(isuf, "_MICROS_GATE", False)
    # Pollo+arroz cubren proteína Y kcal, pero NO la fibra (38 g) → déficit advisory, NO bloquea.
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="day", nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 2.0, "unit": "lb"},
                        {"ingredient_name": "Arroz", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out["sufficient"] is True
    fib = [d for d in out["deficits"] if d["nutrient"] == "fiber_g"]
    assert fib and fib[0]["advisory"] is True   # surfaceado pero advisory (no bloquea)


def test_fail_open_on_internal_error(db, monkeypatch):
    # Forzar fallo en el cálculo de targets → ejercita el fail-open global del evaluador.
    def _boom(_fd):
        raise RuntimeError("boom en get_nutrition_targets")
    monkeypatch.setattr("nutrition_calculator.get_nutrition_targets", _boom)
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal", nutrition_db=db,  # sin meal_target → usa _daily_targets → throws
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out["sufficient"] is True   # fail-open: nunca bloquear por bug del evaluador
    assert "error" in out
