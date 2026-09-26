# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-329 · 2026-09-25] Las porciones del paso son las de la lista.

Batería de cierre del 25-sep: «Mise en place: mide 2 porciones de casabe (15 g)» con «1 porción de casabe (15 g)» en la
lista (3 de 330 comidas): el peso se sincronizó y el número no — «porción» no es una unidad del sincronizador."""
from __future__ import annotations

import recipe_contract as rc
import pasos_cantidades as pc


def test_dos_porciones_en_el_paso_con_una_en_la_lista():
    m = {"ingredients": ["1 porción de casabe (15 g)", "60 g de guineo"],
         "recipe": ["Mise en place: mide 2 porciones de casabe (15 g) y 60 g de guineo.", "Montaje: sirve."]}
    assert pc.porciones_de_la_lista(m) == 1
    assert m["recipe"][0] == "Mise en place: mide 1 porción de casabe (15 g) y 60 g de guineo."


def test_plural_y_sin_linea_de_porcion():
    m = {"ingredients": ["2 porciones de casabe (30 g)"],
         "recipe": ["Mise en place: mide 1 porción de casabe; reserva cada porción de casabe aparte.", "Montaje: sirve."]}
    assert pc.porciones_de_la_lista(m) == 1
    assert m["recipe"][0] == "Mise en place: mide 2 porciones de casabe; reserva cada porción de casabe aparte."
    m2 = {"ingredients": ["30 g de casabe"], "recipe": ["Mise en place: mide 2 porciones de casabe.", "Montaje: sirve."]}
    assert pc.porciones_de_la_lista(m2) == 0


def test_corre_dentro_del_contrato():
    assert "porciones_de_la_lista(meal)" in __import__("inspect").getsource(pc.lo_que_dice_la_lista)
    assert rc is not None
