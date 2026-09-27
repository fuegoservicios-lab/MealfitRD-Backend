# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-440 · 2026-09-26] Tras el artículo, el alimento también va en minúscula.

Replay de la cola sobre 322 planes: «licúa la Leche descremada con el Repollo», «calienta el Aceite de oliva y cocina la
Cebolla y el Tomate» — 60 comidas, 315 menciones."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_alimento_de_la_lista_baja_a_minuscula():
    m = {"ingredients": ["5 ml de leche descremada", "55 g de repollo", "1 cdta de aceite de oliva", "½ cebolla", "1 diente de ajo"],
         "recipe": ["El Toque de Fuego: calienta el Aceite de oliva y cocina la Cebolla y el Ajo 2 minutos.",
                    "Montaje: licúa la Leche descremada con el Repollo y sirve."]}
    assert pc.minuscula_tras_articulo(m) == 2
    assert m["recipe"] == ["El Toque de Fuego: calienta el aceite de oliva y cocina la cebolla y el ajo 2 minutos.",
                           "Montaje: licúa la leche descremada con el repollo y sirve."]
    assert pc.minuscula_tras_articulo(m) == 0
    # tras la cantidad también: «pica ½ tomate, ½ Cebolla y ½ diente de ajo; exprime ½ Limón» (vegetariana, replay)
    v = {"ingredients": ["½ cebolla", "½ limón", "½ mango mediano"],
         "recipe": ["Mise en place: pica ½ Cebolla; exprime ½ Limón y corta ½ Mango mediano."]}
    assert pc.minuscula_tras_articulo(v) == 1
    assert v["recipe"][0] == "Mise en place: pica ½ cebolla; exprime ½ limón y corta ½ mango mediano."


def test_lo_que_no_es_de_la_lista_o_es_propio_se_queda():
    pasos = ["El Toque de Fuego: cocina a la manera del Cibao con el Queso Philadelphia y la Bruselas.",
             "⚠️ Nota: la Leche cruda no se usa."]
    m = {"ingredients": ["30 g de queso crema", "1 taza de coles de bruselas"], "recipe": list(pasos)}
    assert pc.minuscula_tras_articulo(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").minuscula_tras_articulo(meal)  # [P1-PLAN-LOTE-440]' in src
