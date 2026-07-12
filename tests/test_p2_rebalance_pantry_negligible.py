"""[P2-REBALANCE-PANTRY-NEGLIGIBLE · 2026-07-12] Un condimento no revierte el rebalance del día.

Caso vivo (regen 08:19Z corr=f37c4bb5, plan df263d1b): el rebalance a nivel-día — que habría
puesto proteína/carbs/kcal en banda — se revirtió COMPLETO por "Ajo: necesita ~22g pero hay
~5g" (excedente 17g ≈ 25 kcal, macro-negligible). El ajo está en todos los platos criollos,
así que el line-clamp nivel-1 (excluir plato violador) tampoco resolvía → band 0.33 con chips
"Macros algo fuera de la banda" evitables.

Fix: `_day_exceeds_pantry` exime violaciones cuyo EXCEDENTE en kcal no puede mover macros
(< MEALFIT_REBALANCE_PANTRY_NEGLIGIBLE_KCAL, default 40; 0 = comportamiento previo). El
gate sigue firme para excedentes sustantivos (yogurt 325g vs 150g ≈ 103 kcal — el caso que
motivó el line-clamp P1-REBALANCE-LINE-CLAMP).
tooltip-anchor: P2-REBALANCE-PANTRY-NEGLIGIBLE
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND)


def _plans():
    import routers.plans as p
    return p


class _DB:
    """Ajo 149 kcal/100g (aromático); Yogurt 59; Arroz 130."""

    _KCAL = {"ajo": ("Ajo", 149.0), "yogurt": ("Yogurt griego", 59.0), "arroz": ("Arroz blanco", 130.0)}

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        for tok, (nm, k100) in self._KCAL.items():
            if tok in low:
                import re
                m = re.search(r"(\d+(?:\.\d+)?)\s*g", low)
                g = float(m.group(1)) if m else 100.0
                return {"name": nm, "grams": g, "kcal": round(k100 * g / 100.0, 1)}
        return None


def _mk_meals(lines):
    return [{"meal": "Almuerzo", "name": "Plato", "ingredients": list(lines)}]


def test_garlic_overage_does_not_gate():
    p = _plans()
    meals = _mk_meals(["22 g de ajo", "200 g de arroz blanco"])
    ledger = {"Ajo": 5.0, "Arroz blanco": 500.0}
    exceeds, why = p._day_exceeds_pantry(meals, ledger, _DB())
    assert exceeds is False, f"excedente de ajo ≈25 kcal es macro-negligible (why={why!r})"


def test_yogurt_overage_still_gates():
    p = _plans()
    meals = _mk_meals(["325 g de yogurt griego"])
    ledger = {"Yogurt griego": 150.0}
    exceeds, why = p._day_exceeds_pantry(meals, ledger, _DB())
    assert exceeds is True and "Yogurt" in why, \
        "excedente sustantivo (~103 kcal) sigue protegiendo la honestidad de macros"


def test_knob_zero_restores_previous_behavior(monkeypatch):
    monkeypatch.setenv("MEALFIT_REBALANCE_PANTRY_NEGLIGIBLE_KCAL", "0")
    p = _plans()
    meals = _mk_meals(["22 g de ajo"])
    exceeds, why = p._day_exceeds_pantry(meals, {"Ajo": 5.0}, _DB())
    assert exceeds is True and "Ajo" in why, "knob=0 → todo excedente gatea (rollback sin redeploy)"


def test_external_ingredient_still_allowed():
    p = _plans()
    meals = _mk_meals(["325 g de yogurt griego"])
    exceeds, _ = p._day_exceeds_pantry(meals, {"Arroz blanco": 100.0}, _DB())
    assert exceeds is False, "ingrediente fuera del ledger sigue siendo externo/permitido"


def test_marker_anchored_in_source():
    src = open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8").read()
    assert "P2-REBALANCE-PANTRY-NEGLIGIBLE" in src
    assert "MEALFIT_REBALANCE_PANTRY_NEGLIGIBLE_KCAL" in src
