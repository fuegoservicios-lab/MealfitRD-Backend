# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-397 · 2026-09-26] «El filete de 195 g» pesa lo que dice la lista.

Batería REAL sobre el 379 (adulto mayor con HTA): «seca el filete de pescado blanco de 195 g» con «1 filete de pescado
(≈160 g)» y «seca pechuga de pollo de 150 g» con «½ pechuga de pollo (≈100 g)» (corpus: 16 de 22 discrepan >15 %)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_pieza_del_paso_pesa_lo_de_la_lista():
    m = {"ingredients": ["1 filete de pescado (≈160 g)", "½ ají morrón"],
         "recipe": ["Mise en place: corta ½ ají morrón; seca el filete de pescado blanco de 195 g."]}
    assert pc.peso_de_la_pieza_en_el_paso(m) == 1
    assert m["recipe"][0] == "Mise en place: corta ½ ají morrón; seca el filete de pescado blanco de 160 g."
    p = {"ingredients": ["½ pechuga de pollo (≈100 g)"], "recipe": ["Mise en place: seca pechuga de pollo de 150 g."]}
    assert pc.peso_de_la_pieza_en_el_paso(p) == 1 and p["recipe"][0] == "Mise en place: seca pechuga de pollo de 100 g."
    c = {"ingredients": ["2 filetes de pescado (285 g)"], "recipe": ["Mise en place: corta 2 filetes de 220 g cada uno."]}
    assert pc.peso_de_la_pieza_en_el_paso(c) == 1 and c["recipe"][0] == "Mise en place: corta 2 filetes de 142 g cada uno."
    assert pc.peso_de_la_pieza_en_el_paso(c) == 0                                        # idempotente


def test_lo_que_no_se_decide_no_se_toca():
    casos = [
        (["1 filete de pescado (≈160 g)"], "Mise en place: seca el filete de pescado de 165 g."),        # ±10 %
        (["1 filete de pescado", "1 filete de res (150 g)"], "Mise en place: seca el filete de 195 g."),  # dos líneas
        (["1 filete de pescado"], "Mise en place: seca el filete de 195 g."),                             # sin peso
        (["15 g de almendras fileteadas"], "Mise en place: corta el filete de 150 g."),                   # «fileteadas»
        (["½ pechuga de pollo (≈100 g)"], "Mise en place: seca la pechuga de pavo de 150 g."),            # otra proteína
    ]
    for lista, paso in casos:
        m = {"ingredients": list(lista), "recipe": [paso]}
        assert pc.peso_de_la_pieza_en_el_paso(m) == 0 and m["recipe"] == [paso], (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").peso_de_la_pieza_en_el_paso(meal)  # [P1-PLAN-LOTE-397]' in src
