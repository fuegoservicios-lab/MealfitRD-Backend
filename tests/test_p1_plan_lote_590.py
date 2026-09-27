# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-590 · 2026-09-27] Una pizca o el jugo de un limón no abortan el recálculo de macros de la comida.

Baterías de esta semana: 11 de 808 comidas mostraban macros que no cuadraban con sus líneas por ≥10 g de proteína (98 g
mostrados con 26 g reales) porque una línea «1 pizca de sal y pimienta», «jugo de ½ limón» o «Limón» abortaba el recálculo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import truthup_sin_masa as tsm  # noqa: E402


@pytest.mark.parametrize("linea", [
    "1 pizca de sal y pimienta", "1½ pizca de canela en polvo", "2 pizcas de comino", "1 pizca de semillas de linaza",
    "jugo de ½ limón", "Jugo de 1 limón", "el jugo de 2 limas", "jugo de medio limón", "Limón", "Limones",
])
def test_no_pesan(linea):
    assert tsm.sin_masa(linea) is True


@pytest.mark.parametrize("linea", ["½ limón", "Aceite de oliva", "Cebolla", "2 dientes de ajo", "Jugo de naranja",
                                   "Limonada"])
def test_si_pesan_o_ya_se_leen(linea):
    assert tsm.sin_masa(linea) is False


class _Info:
    def __init__(self, kcal, fats):
        self.kcal, self.fats = kcal, fats


class _DB:
    _M = {"150 g de pechuga de pollo": {"protein": 34.0, "carbs": 0.0, "fats": 4.0, "kcal": 180.0},
          "1½ ciruelas": {"protein": 1.0, "carbs": 15.0, "fats": 0.0, "kcal": 60.0},
          "1 limón": {"protein": 0.0, "carbs": 7.0, "fats": 0.0, "kcal": 30.0}}

    def macros_from_ingredient_string(self, s):
        return self._M.get(s)

    def lookup(self, s):
        if re.search(r"pimienta", s):
            return _Info(327, 3.3)
        if re.search(r"lim[oó]n", s, re.IGNORECASE):
            return _Info(46.6, 0.3)
        if re.search(r"ciruela", s, re.IGNORECASE):
            return _Info(46.0, 0.3)
        return None


@pytest.mark.parametrize("otra", ["1 pizca de sal y pimienta", "jugo de ½ limón", "Limón"])
def test_la_comida_se_recalcula(otra):
    import graph_orchestrator as go
    m = {"name": "Pollo al limón", "ingredients": ["150 g de pechuga de pollo", otra],
         "protein": 98, "carbs": 40, "fats": 20, "cals": 700}
    go._truth_up_meal_macros_from_strings(m, _DB())
    assert m["protein"] == 34 and m["cals"] == 180, m


@pytest.mark.parametrize("otra, kcal", [("1–2 ciruelas", 240), ("Limón, 1 unidad", 210)])
def test_la_linea_torcida_se_lee_enderezada(otra, kcal):
    import graph_orchestrator as go
    m = {"name": "Pollo con fruta", "ingredients": ["150 g de pechuga de pollo", otra],
         "protein": 98, "carbs": 40, "fats": 20, "cals": 700}
    go._truth_up_meal_macros_from_strings(m, _DB())
    assert m["cals"] == kcal and m["protein"] in (34, 35), m


def test_ancla():
    import graph_orchestrator as go
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert 'if mc is None and __import__("truthup_sin_masa").legible(ing):  # [P1-PLAN-LOTE-590]' in src
