# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-736 · 2026-09-28] Lo que el guiso ya incorpora no se «acompaña» otra vez aunque se nombre más corto.

Batería real sobre el 659 (adulto mayor con HTA, día 1, cena): «Escurre e incorpora atún en agua bajo en sodio (ya viene
cocido) al guiso en los últimos minutos.» y, en el Montaje, «Toma agua con la cena. Acompaña con atún en agua.»
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cerrador as pce  # noqa: E402

_FUEGO = ("El Toque de Fuego: cocina la cebolla y el tomate 5 minutos. Escurre e incorpora atún en agua bajo en sodio (ya "
          "viene cocido) al guiso en los últimos minutos.")


def test_el_atun_del_guiso_no_se_acompana_otra_vez():
    m = {"name": "Yautía majada con atún", "ingredients": ["125 g de atún en agua bajo en sodio", "½ pedazo de yautía"],
         "recipe": [_FUEGO, "Montaje: sirve la yautía majada con el tomate al lado. Toma agua con la cena. Acompaña con "
                            "atún en agua."]}
    assert pce.proteina_servida_una_vez(m) == 1
    assert m["recipe"][1] == "Montaje: sirve la yautía majada con el tomate al lado. Toma agua con la cena."
    assert m["recipe"][0] == _FUEGO


def test_lo_que_va_detras_con_y_se_sigue_acompanando():
    # replay de la cola: «Acompaña con filete de pescado blanco y edamame.» perdía el edamame entero
    m = {"name": "Ensalada de papas", "ingredients": ["1 filete de pescado blanco fresco", "¼ taza de edamame"],
         "recipe": ["El Toque de Fuego: Cocina filete de pescado blanco a la plancha o hervido y sírvelo como "
                    "proteína del plato.",
                    "Montaje: sirve la ensalada fría. Acompaña con filete de pescado blanco y edamame."]}
    assert pce.proteina_servida_una_vez(m) == 1
    assert m["recipe"][1] == "Montaje: sirve la ensalada fría. Acompaña con edamame."


def test_otro_alimento_se_sigue_acompanando():
    m = {"name": "Yautía majada con atún", "ingredients": ["125 g de atún en agua bajo en sodio", "½ aguacate"],
         "recipe": [_FUEGO, "Montaje: sirve la yautía. Acompaña con aguacate."]}
    antes = list(m["recipe"])
    assert pce.proteina_servida_una_vez(m) == 0 and m["recipe"] == antes


def test_una_palabra_parecida_no_basta():
    # «atún» no empareja con «atunes…» ni «pollo» con «pollos»: prefijo por PALABRA completa
    m = {"name": "Guiso", "ingredients": ["125 g de pollo"],
         "recipe": ["El Toque de Fuego: Escurre e incorpora pollo (ya viene cocido) al guiso en los últimos minutos.",
                    "Montaje: sirve. Acompaña con pollos al carbón."]}
    antes = list(m["recipe"])
    assert pce.proteina_servida_una_vez(m) == 0 and m["recipe"] == antes
