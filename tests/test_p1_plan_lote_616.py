# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-616 · 2026-09-27] Sin sustituto para la fruta de un plato salado, la fruta va al lado.

Producción (27-sep, Nevera exigida): «PAREO CHOCANTE FRUTA+SALADO» en el primer intento de 4 de 5 generaciones y la
autocorrección sin sustituto admitido no hacía nada → dos regeneraciones completas por bloque.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from culinary_context import _meal_has_sweet_savory_clash  # noqa: E402

_SIN_SUSTITUTO = {"dislikes": ["Aguacate", "Tomate", "Batata"]}      # el mismo efecto que una Nevera exigida sin ellos

_MANGU = {
    "meal": "Desayuno",
    "name": "Mangú de guineo verde con queso de hoja y lechosa",
    "ingredients": ["1 guineo verde", "50 g de queso de hoja", "85 g de lechosa en cubos"],
    "recipe": ["Mise en place: pela y corta el guineo verde; corta 85 g de lechosa en cubos.",
               "El Toque de Fuego: hierve el guineo 15-18 min y májalo; dora el queso de hoja 2 min por lado.",
               "Montaje: sirve el mangú con el queso dorado."],
}


def _autofix(meal, form=_SIN_SUSTITUTO):
    dias = [{"day": 1, "meals": [copy.deepcopy(meal)]}]
    return go._fruit_savory_autofix(dias, form, db=object()), dias[0]["meals"][0]


def test_sin_sustituto_la_fruta_va_al_lado():
    n, m = _autofix(_MANGU)
    assert n == 1
    assert m["name"] == "Mangú de guineo verde con queso de hoja y lechosa al lado"
    assert m["recipe"][2] == "Montaje: sirve el mangú con el queso dorado. Sirve la lechosa aparte."
    assert m["ingredients"] == _MANGU["ingredients"], "ni la lista ni los macros cambian"
    assert not _meal_has_sweet_savory_clash(m) and m["_slot_autofix_applied"] == "fruit_side"


def test_el_montaje_que_ya_la_sirve_al_lado_no_se_repite():
    meal = copy.deepcopy(_MANGU)
    meal["recipe"][2] = "Montaje: sirve el mangú con el queso dorado y la lechosa en cubos al lado."
    n, m = _autofix(meal)
    assert n == 1 and m["recipe"][2] == meal["recipe"][2]


def test_la_fruta_cocinada_dentro_del_plato_no_se_separa():
    meal = copy.deepcopy(_MANGU)
    meal["recipe"][1] = "El Toque de Fuego: hierve el guineo y májalo; saltea la lechosa con el queso 2 min."
    n, m = _autofix(meal)
    assert n == 0 and m["name"] == _MANGU["name"]


def test_la_fruta_en_medio_del_nombre_no_se_reescribe():
    meal = copy.deepcopy(_MANGU)
    meal["name"] = "Revoltillo con mango y queso de hoja"
    meal["ingredients"] = ["2 huevos", "80 g de mango", "30 g de queso de hoja"]
    n, m = _autofix(meal)
    assert n == 0 and m["name"] == "Revoltillo con mango y queso de hoja"


def test_con_sustituto_sigue_el_cambio_de_siempre():
    n, m = _autofix(_MANGU, form={"dislikes": []})
    assert n == 1 and "aguacate" in m["name"].lower() and "al lado" not in m["name"]
