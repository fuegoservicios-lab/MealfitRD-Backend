# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-922 · 2026-09-29] El cerrador de banda no deja bajo el piso de 15 g una porción que ya lo cumplía, y las
hierbas frescas son exentas del piso.

Medidor de punto fijo del escudo (913, ia-59): rdv805, «15 g de avena» → «10 g de avena» por
`_rebalance_day_macros_to_target` (corre después del piso); la pasada siguiente la dropeaba. «10 g de cilantro» dropeado.
"""
from __future__ import annotations

import pathlib
import re
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import piso_en_rebalance as per  # noqa: E402

_POR100 = {"avena": (13.0, 66.0, 7.0), "arroz": (2.7, 28.0, 0.3), "cilantro": (2.1, 3.7, 0.5)}


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        k = next((k for k in _POR100 if k in str(s).lower()), None)
        if g is None or k is None:
            return None
        p, c, f = (v * g / 100.0 for v in _POR100[k])
        return {"grams": g, "protein": p, "carbs": c, "fats": f, "kcal": 4 * p + 4 * c + 9 * f}


def _dia():
    db = _DB()
    ings = ["15 g de avena", "300 g de arroz blanco cocido"]
    mc = [db.macros_from_ingredient_string(x) for x in ings]
    return [{"meal": "Desayuno", "name": "Bowl de yogur con frutas", "ingredients": list(ings),
             "protein": round(sum(m["protein"] for m in mc)), "carbs": round(sum(m["carbs"] for m in mc)),
             "fats": round(sum(m["fats"] for m in mc))}]


def _gramos(meal, token):
    return next(_DB().grams_from_ingredient_string(x) for x in meal["ingredients"] if token in x)


def test_bajar_los_carbos_no_deja_la_avena_bajo_el_piso():
    meals = _dia()
    assert go._rebalance_day_macros_to_target(meals, 60.0, 5.0, _DB(), passes=1)
    assert _gramos(meals[0], "avena") == 15.0, meals[0]["ingredients"]
    assert _gramos(meals[0], "arroz") < 300.0, "el resto del grupo absorbe el ajuste"


def test_knob_apagado_conducta_previa(monkeypatch):
    monkeypatch.setenv("MEALFIT_REBALANCE_KEEPS_FLOOR", "false")
    meals = _dia()
    go._rebalance_day_macros_to_target(meals, 60.0, 5.0, _DB(), passes=1)
    assert _gramos(meals[0], "avena") < 15.0, meals[0]["ingredients"]


def test_las_hierbas_frescas_son_exentas_del_piso():
    for h in ("cilantro", "perejil", "cebollin", "albahaca", "culantro"):
        assert h in go._SHRINK_FLOOR_EXEMPT_TOKENS
    assert per.bajo_el_piso("20 g de cilantro fresco", "10 g de cilantro fresco", _DB()) is False
    assert per.bajo_el_piso("15 g de avena", "10 g de avena", _DB()) is True
    assert per.bajo_el_piso("12 g de avena", "8 g de avena", _DB()) is False, "lo que ya estaba bajo el piso no es de este lote"


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("piso_en_rebalance").bajo_el_piso(orig, quant, db)' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-922" in (_BACKEND / "piso_en_rebalance.py").read_text(encoding="utf-8")
