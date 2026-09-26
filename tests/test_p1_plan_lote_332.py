# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-332 · 2026-09-25] La pista de peso de un paso sigue a la de la lista.

Batería de cierre del 25-sep: «Mise en place: mide 1 porción de casabe (15 g), ½ cda de mantequilla de maní natural
(16 g)…» con «½ cda de mantequilla de maní natural sin sal (8 g)» en la lista; «pica la cebolla (20 g), el tomate (40 g)»
con «½ cda de cebolla picada (5 g)». 32 pasos así en el corpus de 308 planes."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_pista_vieja_pasa_a_la_de_la_lista():
    m = {"ingredients": ["1 porción de casabe (15 g)", "½ cda de mantequilla de maní natural sin sal (8 g)", "60 g de guineo"],
         "recipe": ["Mise en place: mide 1 porción de casabe (15 g), ½ cda de mantequilla de maní natural (16 g) y 60 g de "
                    "guineo.", "Montaje: unta y sirve."]}
    assert pc.pistas_de_la_lista(m) == 1
    assert m["recipe"][0] == ("Mise en place: mide 1 porción de casabe (15 g), ½ cda de mantequilla de maní natural (8 g) y "
                              "60 g de guineo.")


def test_el_alimento_mas_cercano_manda():
    m = {"ingredients": ["30 g de lentejas secas", "½ tomate (40 g)", "½ cda de cebolla picada (5 g)"],
         "recipe": ["Mise en place: remoja las lentejas; pica la cebolla (20 g), el tomate (40 g) y el ajo.", "Montaje: sirve."]}
    assert pc.pistas_de_la_lista(m) == 1
    assert m["recipe"][0] == "Mise en place: remoja las lentejas; pica la cebolla (5 g), el tomate (40 g) y el ajo."


def test_reparto_cocido_o_sin_linea_no_se_tocan():
    pasos = ["Mise en place: reserva la mitad del queso (15 g); mide el arroz cocido (150 g) y el perejil (5 g).",
             "Montaje: sirve."]
    m = {"ingredients": ["40 g de queso blanco", "50 g de arroz", "Perejil al gusto"], "recipe": list(pasos)}
    assert pc.pistas_de_la_lista(m) == 0 and m["recipe"] == pasos


def test_el_alimento_cercano_sin_peso_o_con_otra_cantidad_no_toma_la_cifra_de_otro():
    # replay del 25-sep: la primera versión daba a la cebolla los 210 g del jitomate y al yogurt los 275 g de la lechosa
    pasos = ["Mise en place: corta 185 g de nopal en tiras, 210 g de jitomate y ½ cebolla (50 g); mide ½ taza de yogurt "
             "griego sin azúcar (120 g), 15 g de almendras.", "Montaje: sirve."]
    m = {"ingredients": ["185 g de nopal", "210 g de jitomate", "½ cebolla", "½ taza de yogurt griego sin azúcar",
                         "15 g de almendras", "275 g de lechosa"], "recipe": list(pasos)}
    assert pc.pistas_de_la_lista(m) == 0 and m["recipe"] == pasos
    m2 = {"ingredients": ["¾ taza de yogurt natural (180 g)"],
          "recipe": ["Mise en place: mide 1 taza de yogurt natural (240 g).", "Montaje: sirve."]}
    assert pc.pistas_de_la_lista(m2) == 0                          # la taza también es vieja: eso es del sincronizador


def test_ancla():
    src = (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert "+ pistas_de_la_lista(meal)" in src and "tooltip-anchor: P1-PLAN-LOTE-332" in src
