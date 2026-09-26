# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-408 · 2026-09-26] El víver que el paso usa «ya hervido» y la lista compra crudo trae su hervor.

Replay de la cola: «pela y maja la yautía ya hervida», «corta 350 g de batata cocida en rodajas» con el víver crudo en la
lista y ningún paso que lo hierva (perfiles sin tiempo)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_viver_crudo_usado_hervido_trae_su_hervor():
    m = {"ingredients": ["½ pedazo de yautía (125 g)", "2 huevos"],
         "recipe": ["Mise en place: pela y maja la yautía ya hervida con el ajo.", "Montaje: sirve."]}
    assert pc.viver_cocido_de_la_lista(m) == 1
    assert m["recipe"][1] == ("💡 Cocción previa: hierve la yautía pelada en agua 15-20 min, hasta que el cuchillo entre sin "
                              "fuerza."), m["recipe"][1]
    assert pc.viver_cocido_de_la_lista(m) == 0
    y = {"ingredients": ["1 pedazo de yuca (200 g)"], "recipe": ["Mise en place: corta la yuca cocida en bastones."]}
    assert pc.viver_cocido_de_la_lista(y) == 1
    assert y["recipe"][1].endswith("y desecha el agua de cocción (cruda no se come).")


def test_lo_que_ya_se_hierve_no_se_toca():
    casos = [
        (["½ batata mediana"], ["El Toque de Fuego: hierve la batata en agua 15-20 min; corta la batata cocida en rodajas."]),
        (["½ batata mediana"], ["🍠 Cuece la batata de tus ingredientes (hervida o al horno) y sírvela como acompañante."]),
        (["250 g de batata cocida"], ["Mise en place: corta la batata cocida."]),
        (["1 pedazo de yautía"], ["El Toque de Fuego: cocina la yautía en el microondas 6 min; maja la yautía cocida."]),
        (["½ plátano verde"], ["El Toque de Fuego: hierve el plátano hasta que esté bien cocido."]),
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.viver_cocido_de_la_lista(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").viver_cocido_de_la_lista(meal)  # [P1-PLAN-LOTE-408]' in src
