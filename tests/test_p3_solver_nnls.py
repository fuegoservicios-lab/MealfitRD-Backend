"""[M2-SOLVER-NNLS · 2026-06-14] Solver de porciones multi-restricción (mínimos cuadrados acotados
por descenso por coordenadas) que reemplaza el greedy por-grupo. El benchmark M2 midió la fuga:
proteína 16% MAPE / solo 48% de días en ±10% / solo 24% con 4/4 macros en banda. La causa
(análisis 6-lentes): el solver greedy escala por grupo de macro dominante y no compensa los
aportes cruzados (pollo=P+grasa, arroz=C+P) → no clava los 4 a la vez. Este test prueba que el LSQ
sí los clava, es determinista, respeta el box, y cae al greedy si se desactiva.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_db import IngredientNutritionDB

ROWS = [
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 107.0,
     "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.93},
    {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 360.0,
     "protein_g_per_100g": 6.6, "carbs_g_per_100g": 79.0, "fats_g_per_100g": 0.6},
    {"name": "Aceite de oliva", "aliases": ["aceite"], "kcal_per_100g": 900.0,
     "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 100.0},
    {"name": "Habichuelas", "aliases": ["habichuelas", "frijoles"], "kcal_per_100g": 127.0,
     "protein_g_per_100g": 8.7, "carbs_g_per_100g": 22.8, "fats_g_per_100g": 0.5},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=ROWS)


def test_method_is_lsq_by_default(db):
    from portion_solver import solve_portion_macros
    res = solve_portion_macros([{"name": "pollo", "quantity": 100, "unit": "g"}],
                               {"protein": 30, "carbs": 0, "fats": 0}, db=db)
    assert res["method"] == "lsq"


def test_lsq_hits_all_four_macros_on_coupled_meal(db):
    """NÚCLEO: con ingredientes ACOPLADOS el LSQ clava kcal+P+C+F a la vez (greedy no puede).
    El benchmark medía proteína 16% MAPE; aquí los 4 quedan ≤12%."""
    from portion_solver import solve_meal_macros
    ings = ["120g de pechuga de pollo", "100g de arroz blanco", "10g de aceite", "80g de habichuelas"]
    target = {"kcal": 700, "protein": 55, "carbs": 65, "fats": 22}
    res = solve_meal_macros(ings, target, db=db)
    assert res["method"] == "lsq"
    a = res["achieved"]
    worst = max(abs(a[m] - target[m]) / target[m] for m in ("kcal", "protein", "carbs", "fats"))
    assert worst <= 0.12, f"LSQ dejó un macro a {worst:.0%} (achieved={a}, target={target})"


def test_lsq_beats_greedy_on_coupled_meal(db, monkeypatch):
    """Comparación directa: el LSQ tiene MENOR error máximo que el greedy en el mismo plato."""
    import importlib
    import portion_solver
    ings = ["120g de pechuga de pollo", "100g de arroz blanco", "10g de aceite"]
    target = {"kcal": 650, "protein": 52, "carbs": 62, "fats": 22}

    def worst_err(method_lsq):
        monkeypatch.setenv("MEALFIT_SOLVER_LSQ", "true" if method_lsq else "false")
        importlib.reload(portion_solver)
        a = portion_solver.solve_meal_macros(ings, target, db=db)["achieved"]
        return max(abs(a[m] - target[m]) / target[m] for m in ("kcal", "protein", "carbs", "fats"))

    greedy_err = worst_err(False)
    lsq_err = worst_err(True)
    importlib.reload(portion_solver)  # restaurar default
    assert lsq_err < greedy_err, f"LSQ {lsq_err:.0%} no mejoró al greedy {greedy_err:.0%}"
    assert lsq_err <= 0.10


def test_box_lsq_deterministic_and_bounded():
    from portion_solver import _box_lsq
    # 10·x = 20 → x=2 sin reg; bounds [0.3,1.5] → clamp a 1.5.
    assert _box_lsq([[10.0]], [20.0], [1.0], 0.3, 1.5, reg=0.0)[0] == pytest.approx(1.5, abs=1e-6)
    # sistema diagonal: [[1,0],[0,1]]·x=[2,0.5] → x=[2,0.5].
    x = _box_lsq([[1.0, 0.0], [0.0, 1.0]], [2.0, 0.5], [1.0, 1.0], 0.1, 5.0, reg=0.0)
    assert x[0] == pytest.approx(2.0, abs=1e-4) and x[1] == pytest.approx(0.5, abs=1e-4)
    # determinista
    assert x == _box_lsq([[1.0, 0.0], [0.0, 1.0]], [2.0, 0.5], [1.0, 1.0], 0.1, 5.0, reg=0.0)


def test_lsq_falls_back_to_greedy_when_disabled(db, monkeypatch):
    import portion_solver
    monkeypatch.setattr(portion_solver, "SOLVER_LSQ", False)
    res = portion_solver.solve_portion_macros([{"name": "pollo", "quantity": 100, "unit": "g"}],
                                              {"protein": 30, "carbs": 0, "fats": 0}, db=db)
    assert res["method"] == "greedy"


def test_coherence_recipe_matches_achieved(db):
    """Invariante coherence: macros recomputados desde los strings re-escalados == achieved."""
    from portion_solver import solve_meal_macros
    ings = ["120g de pechuga de pollo", "100g de arroz blanco", "10g de aceite"]
    res = solve_meal_macros(ings, {"kcal": 650, "protein": 52, "carbs": 62, "fats": 22}, db=db)
    tot = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
    for s in res["ingredients"]:
        m = db.macros_from_ingredient_string(s)
        if m:
            for k in tot:
                tot[k] += m[k]
    for macro in ("protein", "carbs", "fats"):
        assert tot[macro] == pytest.approx(res["achieved"][macro], abs=1.0)
