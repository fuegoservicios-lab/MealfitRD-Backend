# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-377 · 2026-09-26] Lo crudo nunca «ya viene cocido»: el pescado fresco se cocina.

Batería REAL sobre el 376 (perfil del dueño, día 3, cena): «Escurre e incorpora filete de pescado blanco (ya viene
cocido) a la preparación antes de servir» con «¾ filete de pescado (≈88 g)» — pescado crudo servido sin cocer (el tope de
sodio cambió el atún en lata del cerrador por filete fresco y dejó la frase de enlatado)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_pescado_crudo_se_cocina():
    m = {"ingredients": ["30 g de queso blanco fresco", "1 cdta de aceite de oliva", "¾ filete de pescado (≈88 g)"],
         "ingredients_raw": ["30 g de queso blanco fresco", "1 cdta de aceite de oliva", "¾ filete de pescado (≈95 g)"],
         "recipe": ["El Toque de Fuego: cocina las arepitas; dora el queso blanco fresco (~1-2 min a fuego medio). Escurre e "
                    "incorpora filete de pescado blanco (ya viene cocido) a la preparación antes de servir.",
                    "Montaje: sirve."]}
    assert pc.crudo_no_viene_cocido(m) == 1
    assert "(ya viene cocido)" not in m["recipe"][0]
    assert m["recipe"][0].endswith("Cocina el filete de pescado blanco a la plancha 3-4 min por lado, hasta que se desmenuce "
                                   "fácilmente (63 °C al centro), y agrégalo a la preparación antes de servir."), m["recipe"][0]
    assert m["recipe"][0].startswith("El Toque de Fuego: cocina las arepitas;")                     # el resto del paso intacto


def test_el_pollo_crudo_al_guiso():
    m = {"ingredients": ["150 g de pechuga de pollo"],
         "recipe": ["El Toque de Fuego: guisa los vegetales. Incorpora pollo (ya viene cocido) al guiso en los últimos minutos."]}
    assert pc.crudo_no_viene_cocido(m) == 1
    assert m["recipe"][0].endswith("Cocina el pollo a la plancha 5-7 min por lado, hasta que no quede rosado por dentro "
                                   "(74 °C al centro), y agrégalo al guiso en los últimos minutos."), m["recipe"][0]


def test_la_pechuga_cocida_de_otra_linea_no_hace_cocido_al_pavo():
    # replay del celíaco de producción: la línea «pechuga de pollo cocida» engañó al cerrador («pechuga» es genérico)
    m = {"ingredients": ["¾ pechuga de pollo (≈163 g)", "110 g de pechuga de pavo"],
         "ingredients_raw": ["174.08 g de pechuga de pollo cocida", "108.8 g de pechuga de pavo"],
         "recipe": ["El Toque de Fuego: comprueba que el pollo esté completamente cocido por dentro. Escurre e incorpora "
                    "pechuga de pavo (ya viene cocido) a la preparación antes de servir."]}
    assert pc.crudo_no_viene_cocido(m) == 1
    assert m["recipe"][0].endswith("Cocina la pechuga de pavo a la plancha 5-7 min por lado, hasta que no quede rosado por "
                                   "dentro (74 °C al centro), y agrégala a la preparación antes de servir."), m["recipe"][0]


def test_lo_que_de_verdad_viene_cocido_no_se_toca():
    casos = [
        ["65 g de atún claro en agua"],
        ["150 g de pechuga de pollo cocida"],
        ["1 lata de sardinas"],
        ["100 g de queso"],                              # sin línea del alimento: no se decide
    ]
    frases = {"65 g de atún claro en agua": "Escurre e incorpora atún en agua (ya viene cocido) a la preparación antes de servir.",
              "150 g de pechuga de pollo cocida": "Incorpora pollo (ya viene cocido) al guiso en los últimos minutos.",
              "1 lata de sardinas": "Escurre e incorpora sardinas (ya viene cocido) a la preparación antes de servir.",
              "100 g de queso": "Escurre e incorpora filete de pescado (ya viene cocido) a la preparación antes de servir."}
    for lista in casos:
        pasos = ["El Toque de Fuego: " + frases[lista[0]]]
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.crudo_no_viene_cocido(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").crudo_no_viene_cocido(meal)  # [P1-PLAN-LOTE-377]' in src
