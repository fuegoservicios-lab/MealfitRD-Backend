# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-630 · 2026-09-27] Los «panqueques salados» del almuerzo y la cena son tortitas.

Producción (20-27 sep, Nevera exigida): cuatro rechazos «COMIDA FUERA DE HORARIO… comida de desayuno en la cena
(cereal/panqueque…)» sobre «Tilapia a la plancha con panqueques salados de harina y tayota al limón», cada uno una
regeneración completa.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import tortitas_de_noche as tn  # noqa: E402

_CENA = {
    "meal": "Cena",
    "name": "Tilapia a la plancha con panqueques salados de harina y tayota al limón",
    "ingredients": ["150 g de tilapia", "40 g de harina de trigo", "1 tayota", "½ limón"],
    "recipe": ["Mise en place: mezcla la harina con agua y sal hasta una masa fluida.",
               "El Toque de Fuego: cocina los panqueques finos 2-3 min por lado y voltéalos con cuidado; sella la tilapia.",
               "Montaje: sirve la tilapia con los panqueques salados y la tayota."],
}


def _r(meal):
    dias = [{"day": 1, "meals": [copy.deepcopy(meal)]}]
    return tn.renombrar(dias), dias[0]["meals"][0]


def test_la_cena_de_produccion():
    n, m = _r(_CENA)
    assert n == 1
    assert m["name"] == "Tilapia a la plancha con tortitas saladas de harina y tayota al limón"
    assert m["recipe"][1] == ("El Toque de Fuego: cocina las tortitas finas 2-3 min por lado y voltéalas con cuidado; "
                              "sella la tilapia.")
    assert m["recipe"][2] == "Montaje: sirve la tilapia con las tortitas saladas y la tayota."
    assert m["ingredients"] == _CENA["ingredients"]


def test_el_nombre_que_empieza_por_panqueques():
    meal = copy.deepcopy(_CENA)
    meal["name"] = "Panqueques de harina con pescado blanco a la plancha cítrica y vegetales salteados"
    n, m = _r(meal)
    assert n == 1 and m["name"] == "Tortitas de harina con pescado blanco a la plancha cítrica y vegetales salteados"


def test_los_panqueques_dulces_son_desayuno():
    meal = copy.deepcopy(_CENA)
    meal["name"] = "Panqueques de avena con miel y fresas"
    meal["ingredients"] = ["40 g de avena", "1 cdta de miel", "80 g de fresas"]
    n, m = _r(meal)
    assert n == 0 and m["name"] == meal["name"]


def test_el_desayuno_no_se_toca():
    meal = copy.deepcopy(_CENA)
    meal["meal"] = "Desayuno"
    n, m = _r(meal)
    assert n == 0 and m["name"] == _CENA["name"]


def test_ancla_junto_a_la_fritura_de_la_cena():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("_fry_fixed = _dinner_fry_autofix(days)")
    assert '__import__("tortitas_de_noche").renombrar(days)  # [P1-PLAN-LOTE-630]' in src[i:i + 300]
