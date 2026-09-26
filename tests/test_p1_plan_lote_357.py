# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-357 · 2026-09-26] El lácteo que la lista compra y ningún paso usa se sirve.

Plan del perfil del dueño (batería sobre el 331): «Pisto criollo de vegetales con queso blanco pochado y casabe tostado»
con «70 g de queso mozzarella» en la lista y ningún paso que lo nombre."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _pisto():
    return {"name": "Pisto criollo de vegetales con queso blanco pochado y casabe tostado",
            "ingredients": ["20 g de queso blanco fresco", "½ torta de casabe", "½ cebolla", "½ cdta de aceite de oliva",
                            "½ limón", "70 g de queso mozzarella"],
            "recipe": ["Mise en place: corta ½ cebolla; ten listos 20 g de queso blanco fresco y ½ torta de casabe.",
                       "El Toque de Fuego: pocha el queso blanco fresco en la sartén con 2 cdas de agua, tapado, 3-4 minutos.",
                       "Montaje: sirve el pisto sobre el casabe tostado, coloca el queso blanco pochado encima y termina "
                       "con unas gotas de limón."]}


def test_el_queso_huerfano_se_sirve():
    m = _pisto()
    assert pc.servir_lo_que_sobra(m) == 1
    assert m["recipe"][2].endswith("termina con unas gotas de limón. Acompaña con queso mozzarella."), m["recipe"][2]
    assert pc.servir_lo_que_sobra(m) == 0                                       # idempotente


def test_lo_que_no_se_toca():
    usado = {"ingredients": ["30 g de queso de hoja", "1 huevo"],
             "recipe": ["Mise en place: corta el queso en láminas.", "Montaje: sirve con el huevo."]}
    sin_montaje = {"ingredients": ["½ taza de yogurt natural"], "recipe": ["Mezcla la avena con la leche."]}
    crudo = {"ingredients": ["150 g de pechuga de pollo", "½ taza de arroz"],
             "recipe": ["Mise en place: mide el arroz.", "Montaje: sirve el arroz."]}
    for m in (usado, sin_montaje, crudo):
        antes = list(m["recipe"])
        assert pc.servir_lo_que_sobra(m) == 0 and m["recipe"] == antes, m


def test_el_yogur_huerfano_tambien():
    m = {"ingredients": ["¾ taza de avena", "¼ taza de yogurt natural entero"],
         "recipe": ["Mise en place: mide la avena.", "Montaje: sirve la avena tibia"]}
    assert pc.servir_lo_que_sobra(m) == 1
    assert m["recipe"][1] == "Montaje: sirve la avena tibia. Acompaña con yogurt natural entero."


def test_la_proteina_ya_cocida_huerfana_tambien():
    # bariátrico sobre el 331: «25 g de pavo molido cocido» en la merienda, sin paso
    m = {"ingredients": ["1 mandarina pequeña (70 g)", "20 g de maní molido", "25 g de pavo molido cocido"],
         "recipe": ["Mise en place: pela la mandarina; muele 20 g de maní.",
                    "Montaje: espolvorea el maní molido sobre los gajos de mandarina y sirve de inmediato."]}
    assert pc.servir_lo_que_sobra(m) == 1
    assert m["recipe"][1].endswith("sirve de inmediato. Acompaña con pavo molido cocido."), m["recipe"][1]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").servir_lo_que_sobra(meal)  # [P1-PLAN-LOTE-357]' in src
