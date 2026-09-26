# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-331 · 2026-09-25] El mismo alimento dos veces en una enumeración.

Batería de cierre del 25-sep (sin gluten ni huevo, día 1): «Mise en place: … mide 50 g de aguacate, 30 g de aguacate, 5 g
de semillas de calabaza» con «50 g de aguacate» en la lista: la cifra vieja se quedó junto a la nueva."""
from __future__ import annotations

import pasos_cantidades as pc


def test_la_cifra_vieja_sale_de_la_enumeracion():
    m = {"ingredients": ["¼ pedazo mediano de yuca (≈120 g)", "50 g de aguacate", "5 g de semillas de calabaza"],
         "recipe": ["Mise en place: pela la yuca; mide 50 g de aguacate, 30 g de aguacate, 5 g de semillas de calabaza y "
                    "1 cdta de aceite de oliva.", "Montaje: sirve."]}
    assert pc.mencion_repetida(m) == 1
    assert m["recipe"][0] == ("Mise en place: pela la yuca; mide 50 g de aguacate, 5 g de semillas de calabaza y 1 cdta "
                              "de aceite de oliva.")


def test_dos_semillas_distintas_o_un_reparto_no_se_tocan():
    pasos = ["Mise en place: mide 15 g de semillas de girasol, 30 g de semillas de chía y 195 ml de agua.",
             "Montaje: usa 30 g de aguacate en la base, 20 g de aguacate encima."]
    m = {"ingredients": ["15 g de semillas de girasol", "30 g de semillas de chía", "50 g de aguacate"],
         "recipe": list(pasos)}
    assert pc.mencion_repetida(m) == 0 and m["recipe"] == pasos
    assert "mencion_repetida(meal)" in __import__("inspect").getsource(pc.lo_que_dice_la_lista)
