# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-525 · 2026-09-27] Una hierba o especia SIN cantidad no aborta el truth-up de macros de la comida.

Replay de la compra única (27-sep): «Orégano dominicano» (sin cantidad) hacía abortar
`_truth_up_meal_macros_from_strings` —el nombre está en el catálogo, la cantidad no convierte y el orégano seco no es
«despreciable» (265 kcal/100 g)— y un almuerzo mostraba 938 kcal y 98 g de proteína cuando sus líneas sumaban 762 y 68.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import truthup_sin_masa as tsm  # noqa: E402


class _Info:
    def __init__(self, kcal, fats):
        self.kcal, self.fats = kcal, fats


class _Db:
    _MC = {"150 g de sardinas en lata": {"protein": 37.0, "carbs": 0.0, "fats": 17.0, "kcal": 312.0},
           "195 g de yautía": {"protein": 3.0, "carbs": 54.0, "fats": 0.3, "kcal": 230.0}}
    _INFO = {"orégano": _Info(265, 4.3), "ajo": _Info(149, 0.5), "aceite": _Info(884, 100)}

    def macros_from_ingredient_string(self, s):
        return self._MC.get(str(s))

    def lookup(self, s):
        low = str(s).lower()
        return next((v for k, v in self._INFO.items() if k in low), None)


def _comida(extra):
    return {"name": "Bowl caribeño de sardinas con yautía", "ingredients": ["150 g de sardinas en lata", "195 g de yautía"]
            + extra, "protein": 98, "carbs": 90, "fats": 30, "cals": 938}


def test_oregano_sin_cantidad_no_aborta_el_recalculo():
    m = _comida(["Orégano dominicano", "Ajo"])
    assert go._truth_up_meal_macros_from_strings(m, _Db()) is True
    assert (m["protein"], m["carbs"], m["fats"]) == (40, 54, 17), m


def test_el_aceite_sin_cantidad_sigue_abortando():
    m = _comida(["Aceite de oliva"])
    assert go._truth_up_meal_macros_from_strings(m, _Db()) is False
    assert (m["protein"], m["cals"]) == (98, 938)


def test_la_cabeza_de_la_linea_tiene_que_ser_la_hierba():
    assert tsm.sin_masa("Orégano dominicano") and tsm.sin_masa("Ajo") and tsm.sin_masa("Pimienta negra")
    assert tsm.sin_masa("Sal") and tsm.sin_masa("Comino molido")
    assert not tsm.sin_masa("2 dientes de ajo") and not tsm.sin_masa("Ajo (2 dientes)")
    assert not tsm.sin_masa("Pechuga de pollo al ajo") and not tsm.sin_masa("Salmón")
    assert not tsm.sin_masa("Aceite de oliva") and not tsm.sin_masa("Cebolla") and not tsm.sin_masa("Ajo porro")
