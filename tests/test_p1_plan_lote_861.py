# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-861 · 2026-09-29] El sustituto de la fruta de un plato salado no es un alimento que el plato ya tiene.

Batería real DO de 6d (29-sep, D3 desayuno): «Tostadas integrales con huevo, aguacate y queso mozzarella» — la fruta se
cambió por aguacate en un plato que ya lo llevaba: «corta el aguacate en láminas y aguacate en cubos… coloca el aguacate
sobre las tostadas… y sirve aguacate aparte». Corpus: ≈30 de 666 sustituciones.
"""
from __future__ import annotations

import copy

import graph_orchestrator as go
import sustituto_de_fruta as sf

_TOSTADAS = {
    "meal": "Desayuno",
    "name": "Tostadas integrales con huevo, aguacate y mango",
    "ingredients": ["1 rebanada de pan integral", "1 huevo", "½ aguacate", "85 g de mango"],
    "recipe": ["Mise en place: corta el aguacate en láminas y el mango en cubos; mide 1 rebanada de pan integral y 1 huevo.",
               "El Toque de Fuego: tuesta el pan 2 minutos por lado; cocina el huevo 4-5 minutos, hasta que cuaje.",
               "Montaje: coloca el aguacate sobre las tostadas, acompaña con el huevo y sirve mango fresco aparte."],
}


def _autofix(meal, dislikes=()):
    dias = [{"day": 1, "meals": [copy.deepcopy(meal)]}]
    n = go._fruit_savory_autofix(dias, {"dislikes": list(dislikes), "allergies": []}, db=object())
    return n, dias[0]["meals"][0]


def test_con_aguacate_ya_en_el_plato_el_sustituto_es_tomate():
    n, m = _autofix(_TOSTADAS)
    assert n == 1
    assert "85 g de tomate" in m["ingredients"], m["ingredients"]
    assert sum("aguacate" in x.lower() for x in m["ingredients"]) == 1, m["ingredients"]
    assert "aguacate y tomate" in m["name"].lower() or "tomate" in m["name"].lower(), m["name"]


def test_sin_aguacate_en_el_plato_sigue_el_aguacate():
    sin = copy.deepcopy(_TOSTADAS)
    sin["ingredients"] = ["1 rebanada de pan integral", "1 huevo", "85 g de mango"]
    n, m = _autofix(sin)
    assert n == 1 and "85 g de aguacate" in m["ingredients"], m["ingredients"]


def test_en_una_avena_no_sale_tomate():
    """Corpus: 11 de 77 aguacates dobles eran avenas o yogures — «avena con tomate» sería peor que el aguacate doble."""
    avena = {"meal": "Desayuno", "name": "Avena cremosa con aguacate, mango y huevo bien cocido",
             "ingredients": ["30 g de avena", "½ aguacate", "85 g de mango", "1 huevo bien cocido"],
             "recipe": ["Mise en place: mide la avena; corta el aguacate y el mango en cubos.",
                        "El Toque de Fuego: cocina la avena con agua 7-9 minutos; hierve el huevo 10-12 minutos.",
                        "Montaje: sirve la avena con el aguacate y acompaña con el huevo bien cocido."]}
    assert sf.elegir(avena, ["Aguacate", "Tomate", "Batata"]) == "Aguacate", "la conducta de siempre"
    n, m = _autofix(avena)
    assert n == 1 and not any("tomate" in x.lower() for x in m["ingredients"]), m["ingredients"]


def test_elegir():
    assert sf.elegir({"ingredients": ["½ aguacate", "1 huevo"]}, ["Aguacate", "Tomate", "Batata"]) == "Tomate"
    assert sf.elegir({"ingredients": ["½ aguacates", "2 tomates", "1 batata"]}, ["Aguacate", "Tomate", "Batata"]) is None
    assert sf.elegir({"ingredients": ["1 huevo"]}, ["Aguacate", "Tomate"]) == "Aguacate"
    assert sf.elegir({"ingredients": ["1 huevo"]}, []) is None
