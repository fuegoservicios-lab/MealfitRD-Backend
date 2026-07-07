"""[P1-RECIPE-BLOCKERS-2 · 2026-07-07] 2 bloqueadores del review visual del plan vivo 4e7b8dbb:

- P1-VEG-VOLUME-CAP: "545 g de pepino" + "580 g de pepino" en ensaladas — el solver infla vegetales
  ACUOSOS de bajo-caloría sin cap (ni volume-fruit ni leaf-cap cubren pepino). Techo servible 250g.
- P1-UNIT-LINE-CONSOLIDATE: "1 cdta de aceite de oliva" DUPLICADA — la fusión de gramos/conteo no veía
  las UNIDADES de cuchara. Suma unidades idénticas ("1 cdta"+"1 cdta" → "2 cdtas").

(El 3er bloqueador — queso en dulces — se DIFIRIÓ: excluir queso salado del pool dulce rompe el piso de
proteína cuando el queso es la única proteína densa dulce-compatible del pool. El fix real es sumar
yogurt griego/whey/requesón al pool de alta densidad — esfuerzo separado. Ver memoria.)
"""
import os
import re

import graph_orchestrator as g

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"(\d+(?:\.\d+)?)", str(s))
        gr = float(m.group(1)) if m else 0.0
        low = str(s).lower()
        if "pepino" in low:
            return {"protein": gr * 0.01, "carbs": gr * 0.03, "fats": gr * 0.001, "kcal": gr * 0.15}
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


# ───────────────────────── P1-VEG-VOLUME-CAP ─────────────────────────

def test_cucumber_bomb_capped():
    meal = {"name": "Ensalada", "meal": "Cena",
            "ingredients": ["545 g de pepino", "100 g de arroz"],
            "ingredients_raw": ["545 g de pepino", "100 g de arroz"],
            "protein": 5, "carbs": 30, "fats": 2, "cals": 150}
    n = g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    assert n >= 1
    assert meal["ingredients"][0].startswith("250"), meal["ingredients"][0]
    assert meal["ingredients_raw"][0].startswith("250")  # lockstep


def test_cucumber_under_cap_untouched():
    meal = {"name": "Ensalada", "meal": "Cena",
            "ingredients": ["150 g de pepino"], "ingredients_raw": ["150 g de pepino"],
            "protein": 2, "carbs": 5, "fats": 1, "cals": 40}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    assert meal["ingredients"][0].startswith("150")  # < 250 cap → intacto


def test_veg_cap_knob_and_tokens():
    assert "P1-VEG-VOLUME-CAP" in _GO
    assert "pepino" in g._REALISM_VOLUME_VEG_TOKENS
    assert 100 <= g.REALISM_VEG_VOLUME_CAP_G <= 500


# ───────────────────────── P1-UNIT-LINE-CONSOLIDATE ─────────────────────────

def test_duplicate_cdta_oil_merged():
    meal = {"name": "X",
            "ingredients": ["1 cdta de aceite de oliva", "105 g de cangrejo", "1 cdta de aceite de oliva"],
            "ingredients_raw": ["1 cdta de aceite de oliva", "105 g de cangrejo", "1 cdta de aceite de oliva"]}
    n = g._consolidate_duplicate_gram_lines([{"meals": [meal]}])
    assert n >= 1
    joined = " | ".join(meal["ingredients"])
    assert "2 cdtas de aceite de oliva" in joined, joined
    assert meal["ingredients"].count("1 cdta de aceite de oliva") == 0  # ya no duplicada


def test_distinct_units_not_merged():
    """Distinto alimento o distinta unidad → NO fusiona."""
    meal = {"name": "X", "ingredients": ["1 cdta de aceite de oliva", "1 cda de mantequilla de maní"],
            "ingredients_raw": ["1 cdta de aceite de oliva", "1 cda de mantequilla de maní"]}
    before = list(meal["ingredients"])
    g._consolidate_duplicate_gram_lines([{"meals": [meal]}])
    assert meal["ingredients"] == before


def test_unit_consolidate_anchored():
    assert "P1-UNIT-LINE-CONSOLIDATE" in _GO
