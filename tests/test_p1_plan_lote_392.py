# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-392 · 2026-09-26] El pollo y el pavo se cocinan a 74 °C.

Batería REAL sobre el 379 (adulto mayor con HTA, día 1): «Marina pechuga de pollo…; cocina el filete 4 minutos por lado,
hasta 63 °C en el centro» (el cambio pescado→pollo dejó el punto del pescado). Corpus: 18 comidas de ave a 63 °C, cinco
en perfiles de embarazo."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_ave_sube_a_74():
    m = {"ingredients": ["½ pechuga de pollo (≈100 g)", "45 g de harina de trigo integral", "1 diente de ajo"],
         "recipe": ["El Toque de Fuego: forma 3 arepitas y cocínalas 3 minutos por lado. Marina pechuga de pollo con el ajo "
                    "5 minutos; calienta el aceite en la plancha a fuego medio-alto y cocina el filete 4 minutos por lado, "
                    "hasta 63 °C en el centro. Hornea las arepitas a 160 °C si las quieres crujientes.",
                    "⚠️ Seguridad alimentaria: cocina pechuga de pollo hasta que la carne esté opaca y firme y se separe en "
                    "lascas (63 °C en el centro).",
                    "⚠️ Seguridad alimentaria: verifica que pechuga de pollo esté opaco y se separe en láminas (≥63°C al centro)."]}
    assert pc.ave_a_74(m) == 3
    assert "cocina el filete 4 minutos por lado, hasta 74 °C en el centro." in m["recipe"][0]
    assert "a 160 °C si" in m["recipe"][0]                                                # el horno no se toca
    assert m["recipe"][1] == ("⚠️ Seguridad alimentaria: cocina pechuga de pollo hasta que la carne esté opaca y firme, "
                              "sin partes rosadas (74 °C en el centro).")
    assert m["recipe"][2] == ("⚠️ Seguridad alimentaria: verifica que pechuga de pollo esté opaco, sin partes rosadas "
                              "(≥74°C al centro).")
    assert pc.ave_a_74(m) == 0                                                             # idempotente


def test_lo_que_no_es_ave_no_se_toca():
    casos = [
        (["1 filete de pescado (≈160 g)"], "El Toque de Fuego: cocina el filete 4 minutos por lado, hasta 63 °C en el centro."),
        (["1 filete de pescado", "1 cubito de caldo de pollo"], "El Toque de Fuego: cocina el filete hasta 63 °C al centro."),
        (["120 g de pechuga de pollo", "2 huevos"], "⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y "
                                                   "clara firmes) antes de servir."),
        (["120 g de pechuga de pollo", "65 g de atún"], "El Toque de Fuego: sella el filete de atún 2 minutos, hasta 52 °C."),
    ]
    for lista, paso in casos:
        m = {"ingredients": list(lista), "recipe": [paso]}
        assert pc.ave_a_74(m) == 0 and m["recipe"] == [paso], (lista, m["recipe"])
    # con pollo y atún en la lista, la frase que nombra el pollo sí sube
    m = {"ingredients": ["120 g de pechuga de pollo", "65 g de atún"],
         "recipe": ["El Toque de Fuego: cocina el pollo 5 minutos por lado hasta 63 °C; sella el atún hasta 52 °C."]}
    assert pc.ave_a_74(m) == 1
    assert m["recipe"][0] == "El Toque de Fuego: cocina el pollo 5 minutos por lado hasta 74 °C; sella el atún hasta 52 °C."


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").ave_a_74(meal)  # [P1-PLAN-LOTE-392]' in src
