# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-406 · 2026-09-26] La masa de harina de maíz lleva su agua.

Batería REAL sobre el 399 (perfil del dueño, día 3): «mezcla 130 g de harina de maíz precocida con ½ taza de agua» (0,9
ml/g; la masa pide ~2,3). Corpus: 9 de 54 masas por debajo de 1,2 ml/g, casi todas del perfil del dueño."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_masa_recibe_su_agua():
    m = {"ingredients": ["130 g de harina de maíz precocida", "½ taza de agua", "30 g de queso blanco fresco"],
         "ingredients_raw": ["130 g de harina de maíz precocida", "0.5 taza de agua", "30 g de queso blanco fresco"],
         "recipe": ["Mise en place: mezcla 130 g de harina de maíz precocida con ½ taza de agua y orégano.",
                    "El Toque de Fuego: aplana la masa y cocínala 3-4 minutos por lado."]}
    assert pc.masa_con_su_agua(m) == 2
    assert m["ingredients"][1] == "300 ml de agua"
    assert m["recipe"][0] == "Mise en place: mezcla 130 g de harina de maíz precocida con 300 ml de agua y orégano."
    assert pc.masa_con_su_agua(m) == 0                                         # idempotente
    p = {"ingredients": ["100 g de harina de maíz precocida", "15 ml de agua"],
         "recipe": ["Mise en place: mezcla la harina con 15 ml de agua tibia y forma arepitas."]}
    assert pc.masa_con_su_agua(p) == 2
    assert p["ingredients"][1] == "230 ml de agua" and "con 230 ml de agua tibia" in p["recipe"][0]


def test_lo_que_ya_tiene_agua_o_es_ambiguo_no_se_toca():
    casos = [
        (["60 g de harina de maíz precocida", "140 ml de agua"], ["Mise en place: mezcla con 140 ml de agua."]),  # 2,3 ml/g
        (["60 g de harina de maíz precocida", "20 ml de agua", "200 ml de agua"], ["Mise en place: mezcla."]),    # dos aguas
        (["60 g de harina de trigo", "20 ml de agua"], ["Mise en place: mezcla con 20 ml de agua."]),              # otra harina
        # replay del 406: el agua de la lista era la de la auyama; la masa va «con agua suficiente» o «según el paquete»
        (["100 g de harina de maíz precocida", "2 cdas de agua"],
         ["Mise en place: mide 100 g de harina de maíz precocida y 2 cdas de agua.",
          "El Toque de Fuego: coloca la auyama con el agua restante; mezcla la harina con agua suficiente."]),
        (["110 g de harina de maíz precocida", "1 cucharada de agua"],
         ["El Toque de Fuego: coloca la auyama con la cucharada de agua; forma las arepitas según el paquete."]),
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.masa_con_su_agua(m) == 0 and m["ingredients"] == lista and m["recipe"] == pasos, lista


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").masa_con_su_agua(meal)  # [P1-PLAN-LOTE-406]' in src
