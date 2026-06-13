"""[P3-PROTEIN-TOPUP · 2026-06-13] Cierre determinista del gap de proteína: si tras el
solver una comida queda corta, añade la proteína MÁS MAGRA del pool aprobado del día.

Test E2E live mostró meriendas a 0-6g de proteína (Casabe con Queso, Maní con Melón) que
el escalado no puede arreglar (no hay proteína que escalar). El top-up la añade.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import IngredientNutritionDB

ROWS = [
    {"name": "Claras de huevo", "aliases": ["claras", "claras de huevo"],
     "kcal_per_100g": 52.0, "protein_g_per_100g": 11.0, "carbs_g_per_100g": 0.7, "fats_g_per_100g": 0.2},
    {"name": "Huevo", "aliases": ["huevos", "huevos enteros"],
     "kcal_per_100g": 143.0, "protein_g_per_100g": 12.6, "carbs_g_per_100g": 0.7, "fats_g_per_100g": 9.5},
    {"name": "Lentejas", "aliases": ["lentejas"],
     "kcal_per_100g": 361.0, "protein_g_per_100g": 24.6, "carbs_g_per_100g": 63.0, "fats_g_per_100g": 1.1},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=ROWS)


def test_topup_adds_protein_to_poor_meal(db):
    from graph_orchestrator import _protein_topup_meal
    meal = {"name": "Casabe con Queso", "protein": 4, "carbs": 30, "fats": 2, "cals": 160,
            "ingredients": ["1 casabe (60g)"], "ingredients_raw": ["1 casabe (60g)"], "recipe": ["paso 1"]}
    added = _protein_topup_meal(meal, target_protein=30,
                                db=db, approved_proteins=["Huevos enteros", "Lentejas", "Claras de huevo"])
    assert added >= 15
    assert meal["protein"] >= 25  # de 4g subió cerca del target
    # añadió un ingrediente con la proteína más MAGRA (claras: mayor proteína/kcal).
    assert any("clara" in str(i).lower() for i in meal["ingredients"])
    assert len(meal["recipe"]) == 2  # nota del nutricionista añadida


def test_no_topup_when_protein_sufficient(db):
    from graph_orchestrator import _protein_topup_meal
    meal = {"name": "Pollo", "protein": 40, "cals": 500, "ingredients": ["200g de pollo"]}
    added = _protein_topup_meal(meal, target_protein=40, db=db, approved_proteins=["Huevo"])
    assert added == 0
    assert meal["protein"] == 40  # intacto


def test_failsafe_empty_pool_no_topup(db):
    from graph_orchestrator import _protein_topup_meal
    meal = {"name": "X", "protein": 2, "cals": 100, "ingredients": ["algo"]}
    # pool vacío → NO añade nada (jamás mete proteína fuera del pool aprobado).
    assert _protein_topup_meal(meal, target_protein=30, db=db, approved_proteins=[]) == 0
    assert meal["protein"] == 2


def test_failsafe_pool_without_resolvable_protein(db):
    from graph_orchestrator import _protein_topup_meal
    meal = {"name": "X", "protein": 2, "cals": 100, "ingredients": ["algo"]}
    # pool con nombres que no resuelven en la DB → fail-safe, no añade.
    assert _protein_topup_meal(meal, target_protein=30, db=db,
                               approved_proteins=["IngredienteInexistente"]) == 0
