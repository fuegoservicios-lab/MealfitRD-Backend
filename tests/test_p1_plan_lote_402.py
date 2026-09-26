# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-402 · 2026-09-26] El pescado del guiso no se cuece 15 minutos.

Batería REAL sobre el 379 (adulto mayor con HTA, día 2): «Añade filete de pescado blanco al guiso y cocínalo a fuego medio
12-15 minutos, hasta que esté cocido por dentro» — la plantilla del guiso da al pescado el tiempo del pollo (corpus: 13)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_pescado_del_guiso_no_se_deshace():
    m = {"recipe": ["El Toque de Fuego: Agrega yautía al guiso y cocínalos a fuego medio 12-15 minutos, hasta que estén "
                    "cocidos por dentro. Añade filete de pescado blanco al guiso y cocínalo a fuego medio 12-15 minutos, "
                    "hasta que esté cocido por dentro; incorpóralo con cuidado."]}
    assert pc.pescado_del_guiso(m) == 1
    assert ("Añade filete de pescado blanco al guiso y cocínalo a fuego medio 5-7 minutos, hasta que se desmenuce "
            "fácilmente (63 °C al centro)") in m["recipe"][0], m["recipe"][0]
    assert "Agrega yautía al guiso y cocínalos a fuego medio 12-15 minutos" in m["recipe"][0]      # el víver conserva
    c = {"recipe": ["El Toque de Fuego: Añade camarones al guiso y cocínalos a fuego medio 12-15 minutos, hasta que estén "
                    "cocidos por dentro."]}
    assert pc.pescado_del_guiso(c) == 1 and "2-3 minutos, hasta que estén rosados y opacos" in c["recipe"][0]


def test_el_ave_y_la_carne_conservan_su_tiempo():
    pasos = ["El Toque de Fuego: Añade pechuga de pollo al guiso y cocínala a fuego medio 12-15 minutos, hasta que esté "
             "cocida por dentro.",
             "El Toque de Fuego: Añade filete de pollo al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté "
             "cocido por dentro.",
             "El Toque de Fuego: Añade filete de res al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté "
             "cocido por dentro."]
    m = {"recipe": list(pasos)}
    assert pc.pescado_del_guiso(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pescado_del_guiso(meal)  # [P1-PLAN-LOTE-402]' in src
