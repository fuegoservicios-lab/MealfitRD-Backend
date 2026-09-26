# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-376 · 2026-09-26] El conteo del paso sigue al de la lista también en piezas, filetes y guineítos.

Replay de la cola real (V7e, 10 en las baterías recientes): «mide 2 piezas de casabe» con «1 pieza de casabe», «pela y
corta los 2 guineítos verdes» con «1 guineíto verde», «corta 1¾ filetes de pescado» con «1½ filetes de pescado»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_conteo_sigue_a_la_lista():
    m = {"ingredients": ["1 pieza de casabe", "30 g de queso blanco fresco", "1 guineíto verde",
                         "1½ filetes de pescado (≈240 g)"],
         "recipe": ["Mise en place: mide 2 piezas de casabe y 30 g de queso blanco fresco; pela y corta los 2 guineítos "
                    "verdes; corta 1¾ filetes de pescado (250 g).",
                    "Montaje: coloca el queso sobre el casabe."]}
    assert pc.conteos_con_unidad(m) == 3
    assert m["recipe"][0] == ("Mise en place: mide 1 pieza de casabe y 30 g de queso blanco fresco; pela y corta el "
                              "guineíto verde; corta 1½ filetes de pescado (250 g)."), m["recipe"][0]
    n = {"ingredients": ["1 pieza de casabe"], "recipe": ["Montaje: unta la mantequilla sobre las 2 piezas de casabe."]}
    assert pc.conteos_con_unidad(n) == 1 and n["recipe"][0] == "Montaje: unta la mantequilla sobre la pieza de casabe."


def test_lo_que_no_se_toca():
    casos = [
        (["1 pieza de casabe", "150 g de pechuga de pollo"], ["Mise en place: separa 2 piezas de pollo."]),        # otro alimento
        (["2 piezas de casabe"], ["Mise en place: tuesta 1 pieza de casabe.", "Montaje: sirve 1 pieza de casabe más."]),  # dos menciones
        (["1 guineíto verde", "2 guineítos maduros"], ["Mise en place: pela los 2 guineítos."]),                   # lista ambigua
        (["2 piezas de casabe"], ["Mise en place: reserva la mitad, 1 pieza de casabe, para la merienda."]),       # reparto
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.conteos_con_unidad(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").conteos_con_unidad(meal)  # [P1-PLAN-LOTE-376]' in src
