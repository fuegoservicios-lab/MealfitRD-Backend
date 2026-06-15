"""[P2-RAW-EGG-SUBSTITUTE · 2026-06-15] Huevo crudo en batido: mitigación a nivel de COMPOSICIÓN.

Audit P2-10: el caso 'blended' (huevo crudo licuado = vector directo de Salmonella, hallazgo CRÍTICO
original) se mitigaba solo con una NOTA. Ahora se REEMPLAZA el huevo por yogur griego (blend-safe,
resuelve al catálogo) preservando cantidad + delta de macros → el plan ya no contiene el peligro. El
caso 'no_cook' (cocinable) sigue con nota de cocción; el huevo COCIDO (revoltillo) no se toca.

Validación determinista con catálogo stub (sin LLM/créditos).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _StubDB:
    def macros_from_ingredient_string(self, s):
        t = str(s).lower()
        if "huevo" in t or "clara" in t or "yema" in t:
            return {"kcal": 70.0, "protein": 6.0, "carbs": 0.0, "fats": 5.0}
        if "yogur" in t or "yogurt" in t:
            return {"kcal": 60.0, "protein": 10.0, "carbs": 4.0, "fats": 0.0}
        return None


def _meal(name, ings, recipe):
    return {"name": name, "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": list(recipe), "protein": 20, "carbs": 40, "fats": 12, "cals": 350}


def test_blended_raw_egg_is_substituted(go, monkeypatch):
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    plan = {"days": [{"meals": [_meal("Batido proteico",
                                      ["2 huevos crudos", "1 banana", "leche"], ["Licúa todo"])]}]}
    go._apply_food_safety_fixes(plan)
    meal = plan["days"][0]["meals"][0]
    ings = " ".join(str(i).lower() for i in meal["ingredients"])
    assert "huevo" not in ings, ("el huevo crudo debe removerse del batido", meal["ingredients"])
    assert "yogur" in ings, ("debe quedar el reemplazo blend-safe", meal["ingredients"])
    assert meal["_food_safety_fixed"] == "blended_substituted"
    assert any("Seguridad alimentaria" in str(s) for s in meal["recipe"])


def test_no_cook_egg_keeps_note_not_removed(go):
    """Huevo en preparación cocinable (no licuada): nota de cocción, NO se remueve (cocinar es viable)."""
    plan = {"days": [{"meals": [_meal("Wrap de huevo", ["1 huevo", "tortilla integral de maiz"],
                                      ["Arma el wrap con los ingredientes"])]}]}
    go._apply_food_safety_fixes(plan)
    meal = plan["days"][0]["meals"][0]
    assert meal["_food_safety_fixed"] == "no_cook"
    assert any("huevo" in str(i).lower() for i in meal["ingredients"]), "el huevo se conserva (cocinable)"


def test_cooked_egg_not_touched(go):
    """Revoltillo (huevo COCIDO) no es violación → no se toca."""
    plan = {"days": [{"meals": [_meal("Revoltillo de huevo", ["2 huevos"],
                                      ["Revuelve los huevos en el sartén hasta cuajar"])]}]}
    n = go._apply_food_safety_fixes(plan)
    assert n == 0
    assert plan["days"][0]["meals"][0].get("_food_safety_fixed") is None


def test_knob_off_falls_back_to_note(go, monkeypatch):
    monkeypatch.setattr(go, "RAW_EGG_BLENDED_SUBSTITUTE_ENABLED", False)
    plan = {"days": [{"meals": [_meal("Batido de proteína", ["2 huevos crudos", "1 banana"],
                                      ["Licúa"])]}]}
    go._apply_food_safety_fixes(plan)
    meal = plan["days"][0]["meals"][0]
    assert meal["_food_safety_fixed"] == "blended", "con knob off, nota-only (no sustituye)"
    assert any("huevo" in str(i).lower() for i in meal["ingredients"])


def test_marker_present(go):
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-RAW-EGG-SUBSTITUTE" in src
    assert "def _substitute_blended_raw_egg(" in src
