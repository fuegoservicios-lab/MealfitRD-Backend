# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-358 · 2026-09-26] El paso sigue a la pizca que el pulido de la cola escribe DESPUÉS del contrato.

Replay de la cola real (contrato + pulido) sobre el corpus: 12 menciones «½ g de Sal», «0.99 g de Orégano dominicano» en
los pasos con «1 pizca de sal» en la lista — el lote 330 corre dentro del contrato, antes de que el pulido convierta la
migaja de la lista en pizca."""
from __future__ import annotations

import pathlib

import pulido_lineas as pl

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_paso_sigue_a_la_pizca_nueva():
    m = {"name": "Tilapia al horno", "meal": "Cena",
         "ingredients": ["150 g de tilapia", "0.47 g de Sal", "1 cdta de aceite de oliva"],
         "ingredients_raw": ["150 g de tilapia", "0.47 g de Sal", "1 cdta de aceite de oliva"],
         "recipe": ["Mise en place: mide 150 g de tilapia, 1 cdta de aceite de oliva y ½ g de Sal.",
                    "Montaje: sirve la tilapia."]}
    pl.pulir_plan({"days": [{"day": 1, "meals": [m]}]})
    assert "1 pizca de sal" in m["ingredients"], m["ingredients"]
    assert m["recipe"][0].endswith("1 cdta de aceite de oliva y 1 pizca de sal."), m["recipe"][0]


def test_el_cero_a_secas_tambien():
    # rd4 (DM2): «mide 0 g de almendras tostadas sin sal» con «1 pizca de almendras tostadas sin sal» en la lista
    m = {"name": "Mandarina con almendras, queso blanco fresco y huevo", "meal": "Merienda",
         "ingredients": ["1 mandarina mediana", "0.2 g de almendras tostadas sin sal", "5 g de queso blanco fresco"],
         "ingredients_raw": ["1 mandarina mediana", "0.2 g de almendras tostadas sin sal", "5 g de queso blanco fresco"],
         "recipe": ["Mise en place: pela la mandarina; mide 0 g de almendras tostadas sin sal y corta 5 g de queso."]}
    pl.pulir_plan({"days": [{"day": 1, "meals": [m]}]})
    assert "1 pizca de almendras tostadas sin sal" in m["ingredients"], m["ingredients"]
    assert "mide 1 pizca de almendras tostadas sin sal" in m["recipe"][0], m["recipe"][0]


def test_ancla():
    assert "P1-PLAN-LOTE-358" in (_BACKEND / "pulido_lineas.py").read_text(encoding="utf-8")
