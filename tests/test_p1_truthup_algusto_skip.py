"""[P1-TRUTHUP-ALGUSTO-SKIP · 2026-07-07] Completa P1-TRUTHUP-NEGLIGIBLE-SKIP tras el review VPS del
plan vivo 4e7b8dbb (post-fix). El fix de hierbas (kcal≤40) NO cubría las ESPECIAS densas usadas
"al gusto": "pimienta negra al gusto" (327 kcal/100g) y "orégano dominicano al gusto" (350) SEGUÍAN
abortando el truth-up de meals con aguacate → la grasa del aguacate quedaba sin contar y day1/day2
del plan chunked salían sobre el target de grasa (72/67 vs 58).

Insight: "al gusto"/"opcional"/"una pizca" tienen CERO masa contable por definición → deben saltarse
SIEMPRE, independiente de la densidad por-100g del alimento. "al gusto" NO es masa real como "1 lata".
"""
import os
import re

import graph_orchestrator as g

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _Info:
    def __init__(self, kcal, fats):
        self.kcal = kcal
        self.fats = fats


class _DB:
    """pimienta/orégano = ESPECIA densa (kcal alto) sin cantidad convertible ('al gusto').
    aguacate con hint = convertible."""
    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "gusto" in low or "opcional" in low or "pizca" in low:  # sin masa contable
            return None
        mg = re.search(r"\((\d+(?:\.\d+)?)\s*g", low) or re.match(r"^\s*(\d+(?:\.\d+)?)\s*g\b", low)
        if not mg:
            return None
        gr = float(mg.group(1))
        if "aguacate" in low:
            return {"protein": gr * 0.02, "carbs": gr * 0.09, "fats": gr * 0.15, "kcal": gr * 1.6}
        return {"protein": gr * 0.10, "carbs": gr * 0.10, "fats": gr * 0.01, "kcal": gr * 1.0}

    def lookup(self, s):
        low = str(s).lower()
        if "pimienta" in low:
            return _Info(327, 3.3)   # especia densa
        if "oregano" in low or "orégano" in low:
            return _Info(350, 4.3)
        if "sal" in low:
            return _Info(0, 0)
        if "aguacate" in low:
            return _Info(160, 15)
        if "lata" in low or "atun" in low:
            return _Info(130, 5)
        return None


def test_pepper_al_gusto_no_longer_aborts_counts_avocado():
    """'Pimienta negra al gusto' (327 kcal/100g) ya NO aborta → el aguacate se cuenta."""
    meal = {"name": "Chivo con Ensalada de Aguacate", "meal": "Cena",
            "ingredients": ["115 g de chivo", "1 aguacate (150 g)", "Pimienta negra al gusto",
                            "Sal al gusto"],
            "ingredients_raw": ["115 g de chivo", "1 aguacate (150 g)", "Pimienta negra al gusto",
                                "Sal al gusto"],
            "protein": 30, "carbs": 10, "fats": 8, "cals": 250}  # STALE: aguacate sin contar
    changed = g._truth_up_meal_macros_from_strings(meal, _DB())
    assert changed is True, "truth-up debe reescribir (antes pimienta-al-gusto abortaba)"
    assert meal["fats"] >= 20, f"grasa honesta debe contar el aguacate (150g×0.15=22g), dio {meal['fats']}"


def test_oregano_al_gusto_skipped():
    meal = {"name": "X", "ingredients": ["1 aguacate (150 g)", "Orégano dominicano al gusto"],
            "ingredients_raw": ["1 aguacate (150 g)", "Orégano dominicano al gusto"],
            "protein": 2, "carbs": 5, "fats": 3, "cals": 60}
    assert g._truth_up_meal_macros_from_strings(meal, _DB()) is True
    assert meal["fats"] >= 20


def test_opcional_skipped():
    meal = {"name": "X", "ingredients": ["1 aguacate (150 g)", "1 cdta de aceite (opcional)"],
            "ingredients_raw": ["1 aguacate (150 g)", "1 cdta de aceite (opcional)"],
            "protein": 2, "carbs": 5, "fats": 3, "cals": 60}
    assert g._truth_up_meal_macros_from_strings(meal, _DB()) is True


def test_real_canned_mass_still_aborts():
    """Regresión: '1 lata de atún' (masa real, no 'al gusto') SIGUE abortando (conservador)."""
    meal = {"name": "X", "ingredients": ["1 lata de atún", "1 aguacate (150 g)"],
            "ingredients_raw": ["1 lata de atún", "1 aguacate (150 g)"],
            "protein": 30, "carbs": 5, "fats": 20, "cals": 300}
    before = (meal["protein"], meal["carbs"], meal["fats"])
    assert g._truth_up_meal_macros_from_strings(meal, _DB()) is False
    assert (meal["protein"], meal["carbs"], meal["fats"]) == before


def test_nomass_regex_cases():
    R = g._TRUTHUP_NOMASS_QTY_RE
    for s in ["Pimienta negra al gusto", "Sal al gusto", "Orégano al gusto",
              "1 cdta de aceite (opcional)", "una pizca de canela"]:
        assert R.search(s.lower()), f"debe matchear: {s}"
    for s in ["1 aguacate (221 g)", "65 g de soya texturizada", "2 cebollas morada (199 g)"]:
        assert not R.search(s.lower()), f"NO debe matchear: {s}"


def test_marker_anchored():
    assert "P1-TRUTHUP-ALGUSTO-SKIP" in _GO
    assert "_TRUTHUP_NOMASS_QTY_RE.search(str(ing).lower())" in _GO
