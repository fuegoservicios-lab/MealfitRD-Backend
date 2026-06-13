"""[P0-MEAL-MACRO-RECOVERY · 2026-06-13] Un meal que el day-gen (gap del LLM) o un
self-critique fallido (cb_open/timeout/None → `_critique_unresolved` + surgical-regen
que falla) dejaba SIN protein/carbs/fats se shippeaba con `protein: '0'` + el
placeholder `macros: ['Plan Matemático']` → el usuario veía 0g de proteína ese día
(observado en prod 2026-06-13, plan 80b33cdf día 1, pese a que el plan PASÓ el review).

`_recover_meal_macros` estima el breakdown faltante desde las cals del meal con el
split objetivo del plan (determinístico, muy superior a 0), preservando los meals que
SÍ tienen macros reales.

Anchor: P0-MEAL-MACRO-RECOVERY.
"""
from graph_orchestrator import _recover_meal_macros, _meal_macro_num


def test_meal_macro_num_parses_variants():
    assert _meal_macro_num("154g") == 154.0
    assert _meal_macro_num("464 kcal") == 464.0
    assert _meal_macro_num("0") == 0.0
    assert _meal_macro_num(None) == 0.0
    assert _meal_macro_num(32) == 32.0
    assert _meal_macro_num("") == 0.0


def test_estimates_macros_from_cals_when_missing():
    """El bug de prod: cals presente pero P/C/F en 0 → estimar, no dejar 0."""
    m = {"cals": "464", "protein": "0", "carbs": "0", "fats": "0", "macros": ["Plan Matemático"]}
    _recover_meal_macros(m, 0.30, 0.45, 0.25)  # split 30/45/25
    assert _meal_macro_num(m["protein"]) > 25   # 464*0.30/4 ≈ 35
    assert _meal_macro_num(m["carbs"]) > 40
    assert _meal_macro_num(m["fats"]) > 8
    assert m["macros"][0].startswith("P:")
    assert "Plan Matemático" not in m["macros"]


def test_real_macros_are_untouched():
    m = {"cals": "500", "protein": 32, "carbs": 48, "fats": 20, "macros": ["P:32g", "C:48g", "G:20g"]}
    _recover_meal_macros(m, 0.30, 0.45, 0.25)
    assert m["protein"] == 32
    assert m["macros"] == ["P:32g", "C:48g", "G:20g"]


def test_no_cals_and_no_macros_keeps_placeholder():
    m = {"protein": "0", "carbs": "0", "fats": "0"}
    _recover_meal_macros(m, 0.30, 0.45, 0.25)
    assert m["macros"] == ["Plan Matemático"]


def test_reconstructs_display_list_when_numeric_present():
    m = {"cals": "500", "protein": 40, "carbs": 50, "fats": 15}  # sin lista de display
    _recover_meal_macros(m, 0.30, 0.45, 0.25)
    assert m["macros"] == ["P:40g", "C:50g", "G:15g"]
