# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-375 · 2026-09-26] Lo que la lista compra SECO y el paso usa cocido trae su cocción previa.

Replay de la cola real: 123 comidas con V7c (46 en las baterías recientes): «mide 140 g de garbanzos cocidos» con los
garbanzos SECOS en la lista y ningún paso que los remoje ni los hierva."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc
from culinary_coherence import _v7c_seco_sin_coccion, build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = build_culinary_index([
    {"name": "Garbanzos", "aliases": ["garbanzo"], "category": "Legumbres"},
    {"name": "Arroz blanco", "aliases": ["arroz"], "category": "Granos"},
    {"name": "Tomate", "aliases": ["tomates"], "category": "Vegetales"},
])


def _comida():
    return {"name": "Garbanzos guisados con arroz",
            "ingredients": ["140 g de garbanzos secos", "40 g de arroz blanco crudo", "½ tomate"],
            "recipe": ["Mise en place: mide 385 g de garbanzos cocidos y corta ½ tomate.",
                       "El Toque de Fuego: saltea el tomate y agrega los garbanzos cocidos 5 min.",
                       "Montaje: sirve los garbanzos con el arroz."]}


def test_la_coccion_previa_cierra_la_contradiccion():
    m = _comida()
    assert _v7c_seco_sin_coccion({"day": 0}, m, _IDX)                     # antes: V7c
    assert pc.coccion_previa(m, _IDX) == 2
    assert m["recipe"][1].startswith("💡 Cocción previa: remoja los garbanzos secos 8-12 h y hiérvelos 60-90 min"), m["recipe"]
    assert "tanda de varios días" in m["recipe"][1]
    assert m["recipe"][2] == "💡 Cocción previa: enjuaga el arroz blanco crudo y cuécelo en agua 15-20 min hasta que esté tierno."
    assert m["recipe"][0].startswith("Mise en place") and m["recipe"][3].startswith("El Toque de Fuego")
    assert not _v7c_seco_sin_coccion({"day": 0}, m, _IDX)                 # después: nada
    assert pc.coccion_previa(m, _IDX) == 0                                # idempotente


def test_lo_que_no_se_toca():
    m = {"ingredients": ["140 g de garbanzos secos"],
         "recipe": ["Mise en place: remoja los garbanzos 8-12 h.", "El Toque de Fuego: hierve los garbanzos 60 min."]}
    antes = list(m["recipe"])
    assert pc.coccion_previa(m, _IDX) == 0 and m["recipe"] == antes
    assert pc.coccion_previa(_comida(), None) == 0                        # sin índice


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").coccion_previa(meal, index)  # [P1-PLAN-LOTE-375]' in src
