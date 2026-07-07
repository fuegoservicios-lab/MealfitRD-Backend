"""[P1-APIO-STALK-CAP · 2026-07-07] Review del plan vivo 4b9291fe (degradado): "9½ tallos de apio
(285g)" en un gratinado para 1 persona. El apio es aromático (cup-cap ≤1 taza) pero por CONTEO de
tallos o en GRAMOS se escapaba de los caps. Cap: ~4 tallos (count-compound) + 250g (veg-volume).
"""
import os
import re

import graph_orchestrator as g

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.search(r"\((\d+)", str(s))
        gr = float(m.group(1)) if m else 0.0
        return ({"protein": gr * 0.007, "carbs": gr * 0.03, "fats": gr * 0.002, "kcal": gr * 0.16}
                if "apio" in str(s).lower() else {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0})

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def _cap(line):
    meal = {"name": "Gratinado de Apio", "meal": "Cena", "ingredients": [line], "ingredients_raw": [line],
            "protein": 5, "carbs": 30, "fats": 2, "cals": 150}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    return meal["ingredients"][0]


def test_apio_stalk_count_capped():
    out = _cap("9½ tallos de apio (285g) cortados en bastones")
    assert out.startswith("4 tallos de apio"), out


def test_apio_grams_capped():
    assert _cap("285 g de apio").startswith("250"), _cap("285 g de apio")


def test_apio_reasonable_portion_untouched():
    assert _cap("2½ tallos de apio (75g)").startswith("2½ tallos"), _cap("2½ tallos de apio (75g)")


def test_anchored():
    assert "P1-APIO-STALK-CAP" in _GO
    assert "apio" in g._REALISM_VOLUME_VEG_TOKENS
    assert any("apio" in t[0] for t in g._REALISM_COMPOUND_COUNT_CAPS)
