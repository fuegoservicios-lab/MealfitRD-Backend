# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-526 · 2026-09-27] El paso mide las tortas de casabe de la lista, no los gramos que el tope ya quitó.

Batería real del dueño (compra única, días 25-28): «2 tortas pequeñas de casabe» en la lista y «mide 175 g de casabe»,
«corta 90 g de casabe», «mide 115 g», «mide 70 g» en los pasos — el humanizador cuenta tortas de 20 g y un tope dejó la
lista en 2.
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def test_los_gramos_del_modelo_pasan_a_las_tortas_de_la_lista():
    m = {"ingredients": ["195 g de atún en agua", "2 tortas pequeñas de casabe", "½ taza de repollo morado finamente cortado",
                         "1 cda de aceite de oliva"],
         "recipe": ["Mise en place: escurre 195 g de atún en agua; corta ½ taza de repollo morado; mide 175 g de casabe y "
                    "1 cda de aceite de oliva.",
                    "El Toque de Fuego: tuesta el casabe en sartén seca a fuego medio durante 1-2 minutos por lado."]}
    assert pc.gramos_de_casabe(m) == 1
    assert "mide 2 tortas pequeñas de casabe (≈40 g) y 1 cda de aceite de oliva." in m["recipe"][0], m["recipe"][0]
    assert pc.gramos_de_casabe(m) == 0                                   # idempotente


def test_una_torta_en_singular_y_el_corte_se_queda():
    m = {"ingredients": ["60 g de sardinas en lata", "1 torta pequeña de casabe"],
         "recipe": ["Mise en place: escurre 60 g de sardinas; corta 90 g de casabe en porciones."]}
    assert pc.gramos_de_casabe(m) == 1
    assert m["recipe"][0] == ("Mise en place: escurre 60 g de sardinas; corta 1 torta pequeña de casabe (≈20 g) en "
                              "porciones."), m["recipe"][0]


def test_el_articulo_concuerda_con_las_tortas():
    # rd4 (sin gluten ni huevo): «calienta los 35 g de casabe» con «1 torta pequeña de casabe» en la lista
    m = {"ingredients": ["1 torta pequeña de casabe", "30 g de queso blanco fresco"],
         "recipe": ["El Toque de Fuego: calienta los 35 g de casabe en una sartén seca 1 minuto por lado."]}
    assert pc.gramos_de_casabe(m) == 1
    assert m["recipe"][0] == ("El Toque de Fuego: calienta la torta pequeña de casabe (≈20 g) en una sartén seca 1 "
                              "minuto por lado."), m["recipe"][0]
    m = {"ingredients": ["2 tortas pequeñas de casabe"], "recipe": ["Montaje: sirve los 90 g de casabe al lado."]}
    assert pc.gramos_de_casabe(m) == 1
    assert m["recipe"][0] == "Montaje: sirve las 2 tortas pequeñas de casabe (≈40 g) al lado.", m["recipe"][0]


def test_si_los_gramos_ya_son_los_de_la_lista_no_se_toca():
    m = {"ingredients": ["225 g de yogur griego entero", "1¾ tortas pequeñas de casabe"],
         "recipe": ["Mise en place: mide 225 g de yogur griego entero y 35 g de casabe."]}
    antes = list(m["recipe"])
    assert pc.gramos_de_casabe(m) == 0 and m["recipe"] == antes


def test_casabe_en_gramos_en_la_lista_no_se_toca():
    m = {"ingredients": ["70 g de casabe"], "recipe": ["Mise en place: mide 175 g de casabe."]}
    assert pc.gramos_de_casabe(m) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").gramos_de_casabe(meal)  # [P1-PLAN-LOTE-526]' in src
