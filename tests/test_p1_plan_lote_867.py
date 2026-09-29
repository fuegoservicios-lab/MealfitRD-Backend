# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-867 · 2026-09-29] «…hasta que no quede rosada por dentro y no quede rosada por dentro» → una vez.

Segunda batería real de embarazo del 29-sep (almuerzo D2, tortitas de pavo). Corpus: 1 caso (éste).
"""
from __future__ import annotations

import pasos_cantidades as pc


def test_la_frase_repetida_a_los_dos_lados_del_y_queda_una_vez():
    m = {"recipe": ["El Toque de Fuego: cocina la pechuga de pavo a fuego medio 5-6 min, hasta que no quede rosada por "
                    "dentro y no quede rosada por dentro. Incorpórala al plátano majado."]}
    assert pc.frases_repetidas(m) == 1
    assert m["recipe"][0] == ("El Toque de Fuego: cocina la pechuga de pavo a fuego medio 5-6 min, hasta que no quede "
                              "rosada por dentro. Incorpórala al plátano majado.")
    assert pc.frases_repetidas(m) == 0, "idempotente"


def test_lo_que_no_es_eco_no_se_toca():
    for t in ("Montaje: corta y corta en cubos el mango.",                       # dos palabras: no
              "El Toque de Fuego: cocina la cebolla y el ajo 3 minutos y el tomate 2 minutos."):
        m = {"recipe": [t]}
        pc.frases_repetidas(m)
        assert m["recipe"][0] == t
