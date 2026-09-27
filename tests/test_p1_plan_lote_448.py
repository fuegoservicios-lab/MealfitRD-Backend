# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-448 · 2026-09-27] El agua que el Mise en place mide, la cocción de la avena también la usa (92 avenas)."""
from __future__ import annotations

import pathlib

import avena_liquido as al

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _avena(toque, ings=None):
    return {"name": "Avena cremosa con lechosa",
            "ingredients": ings or ["1 taza de avena", "5 ml de leche descremada", "455 ml de agua", "60 g de lechosa",
                                    "¼ cdta de canela en polvo"],
            "recipe": ["Mise en place: mide 1 taza de avena (50 g), 5 ml de leche descremada y 455 ml de agua.", toque,
                       "Montaje: sirve la avena tibia con la lechosa."]}


def test_la_leche_y_el_agua_van_juntas_a_la_olla():
    m = _avena("El Toque de Fuego: cocina la avena con la leche descremada y la canela en una olla a fuego medio durante "
               "7-9 min, revolviendo hasta que quede cremosa.")
    assert al.agua_en_la_coccion(m) == 1
    assert m["recipe"][1].startswith("El Toque de Fuego: cocina la avena con la leche descremada, el agua y la canela"), m
    assert al.agua_en_la_coccion(m) == 0, "idempotente"
    m2 = _avena("El Toque de Fuego: calienta la avena con la leche en el microondas 2-3 minutos.")
    assert al.agua_en_la_coccion(m2) == 1 and "con la leche y el agua en el microondas" in m2["recipe"][1]
    # la leche con todos sus adjetivos («pasteurizada»), y tras una coma el agua va en la enumeración
    m3 = _avena("El Toque de Fuego: cocina la avena con la leche pasteurizada y la canela 7-9 min.")
    assert al.agua_en_la_coccion(m3) == 1 and "con la leche pasteurizada, el agua y la canela" in m3["recipe"][1], m3
    m4 = _avena("El Toque de Fuego: cocina la avena con la leche descremada, la canela y la linaza 6 min.")
    assert al.agua_en_la_coccion(m4) == 1 and "la leche descremada, el agua, la canela y la linaza" in m4["recipe"][1]


def test_sin_leche_nombrada_el_agua_va_con_la_avena():
    m = _avena("El Toque de Fuego: cocina la avena a fuego medio 5 minutos, removiendo hasta cremosa.")
    assert al.agua_en_la_coccion(m) == 1 and "cocina la avena con el agua a fuego medio" in m["recipe"][1]
    m2 = _avena("El Toque de Fuego: cocina la avena con la canela a fuego medio 5 minutos.")
    assert al.agua_en_la_coccion(m2) == 1 and "cocina la avena con el agua y la canela" in m2["recipe"][1]


def test_lo_que_no_toca():
    ya = _avena("El Toque de Fuego: cocina la avena con la leche y 455 ml de agua a fuego medio 7 min.")
    assert al.agua_en_la_coccion(ya) == 0
    fria = _avena("El Toque de Fuego: remoja la avena con la leche en la nevera toda la noche.")
    assert al.agua_en_la_coccion(fria) == 0
    sin_agua = _avena("El Toque de Fuego: cocina la avena con la leche 5 min.",
                      ings=["40 g de avena", "200 ml de leche descremada"])
    assert al.agua_en_la_coccion(sin_agua) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("avena_liquido").agua_en_la_coccion(meal)  # [P1-PLAN-LOTE-448]' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-448" in (_BACKEND / "avena_liquido.py").read_text(encoding="utf-8")
