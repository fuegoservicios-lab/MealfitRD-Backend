# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-328 · 2026-09-25] Los gramos de un paso siguen a la PIEZA de la lista.

Batería de cierre del 25-sep (330 comidas re-medidas con el contrato vigente): «Mise en place: corta 205 g de pechuga en
tiras» con «1 pechuga de pollo (≈158 g)» en la lista, «corta 172 g de pechuga de pollo en cubos» con «¾ pechuga de pollo
(≈138 g)» — 11 comidas. Ni el sincronizador (lee líneas «N g de X») ni el lote 310 (el paréntesis tras el MISMO texto)
lo veían."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_paso_pesa_lo_que_dice_la_pieza():
    m = {"ingredients": ["1 pechuga de pollo (≈158 g)", "15 g de berenjena", "30g de vainitas"],
         "recipe": ["Mise en place: corta 205 g de pechuga en tiras, 15 g de berenjena en cubos y 30 g de vainitas.",
                    "El Toque de Fuego: saltea el pollo 5-7 min.", "Montaje: sirve."]}
    assert pc.gramos_de_la_pieza(m) == 1
    assert m["recipe"][0].startswith("Mise en place: corta 158 g de pechuga en tiras, 15 g de berenjena"), m["recipe"][0]


def test_la_misma_porcion_nombrada_dos_veces_sigue_a_la_pieza():
    m = {"ingredients": ["¾ pechuga de pollo (≈138 g)", "½ cebolla"],
         "recipe": ["Mise en place: corta 172 g de pechuga de pollo en cubos; pica ½ cebolla.",
                    "El Toque de Fuego: dora los 172 g de pechuga de pollo 3 minutos.", "Montaje: sirve."]}
    assert pc.gramos_de_la_pieza(m) == 2
    assert "corta 138 g de pechuga de pollo en cubos" in m["recipe"][0]
    assert "dora los 138 g de pechuga de pollo" in m["recipe"][1]


def test_lo_que_no_es_la_misma_base_ni_la_misma_pieza_no_se_toca():
    pasos = ["Mise en place: desmenuza 145 g de pechuga de pollo cocida; mide 110 g de pechuga de pavo.",
             "El Toque de Fuego: reparte la mitad, 80 g de pechuga de pollo, en cada tortilla.", "Montaje: sirve."]
    m = {"ingredients": ["¾ pechuga de pollo (≈163 g)", "110 g de pechuga de pavo"], "recipe": list(pasos)}
    assert pc.gramos_de_la_pieza(m) == 0            # cocida contra cruda, y un reparto: ninguna reescritura
    assert m["recipe"] == pasos
    m2 = {"ingredients": ["½ pechuga de pollo (≈115 g)", "100 g de pechuga de pavo"],
          "recipe": ["Mise en place: corta 90 g de pechuga en tiras.", "Montaje: sirve."]}
    assert pc.gramos_de_la_pieza(m2) == 0           # «pechuga» sola: ¿pollo o pavo? ambiguo


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").lo_que_dice_la_lista(meal)' in src
    assert "gramos_de_la_pieza(meal)" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
