# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-339 · 2026-09-25] El mismo alimento no se sirve dos veces, tampoco con el nombre corto y el largo.

Batería real sobre el 331 (perfil del dueño, día 2): «💪 Sirve Yogurt al lado para acompañar.» dos veces y «Montaje: …
Acompaña con yogurt natural entero. Acompaña con yogurt griego entero.»; adulto mayor: «💪 Sirve queso cottage bajo en
sodio al lado para acompañar.» con «Acompaña con queso cottage.»"""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_nombre_corto_y_largo_son_el_mismo_servido():
    m = {"recipe": ["Mise en place: mide 1 casabe.", "💪 Sirve Yogurt al lado para acompañar.",
                    "💪 Sirve Yogurt al lado para acompañar.", "💪 Sirve queso cottage bajo en sodio al lado para acompañar.",
                    "Montaje: unta el casabe. Acompaña con yogurt natural entero. Acompaña con queso cottage."]}
    assert pc.servir_una_vez(m) == 3
    assert m["recipe"] == ["Mise en place: mide 1 casabe.",
                           "Montaje: unta el casabe. Acompaña con yogurt natural entero. Acompaña con queso cottage."]


def test_otro_alimento_con_la_misma_cabeza_no():
    pasos = ["💪 Sirve queso mozzarella al lado para acompañar.", "Montaje: sirve. Acompaña con queso cottage."]
    m = {"recipe": list(pasos)}
    assert pc.servir_una_vez(m) == 0 and m["recipe"] == pasos


def test_el_orden_de_la_cola_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i_var = src.index('variedad_de_la_lista(meal, index)')
    i_srv = src.index('servir_una_vez(meal)')
    i_rep = src.index('frases_repetidas(meal)     # [P1-PLAN-LOTE-339]')
    assert i_var < i_srv < i_rep
