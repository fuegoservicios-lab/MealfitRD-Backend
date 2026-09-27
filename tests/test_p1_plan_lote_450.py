# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-450 · 2026-09-27] «Cocina el pescado…» usa el «filete de tilapia» de la lista; «jitomate» es el tomate."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _plato(ings, toque):
    return {"name": "Plato", "ingredients": ings,
            "recipe": ["Mise en place: seca y sazona.", toque, "Montaje: sirve caliente."]}


def test_la_clase_del_pescado_cuenta_como_uso():
    m = _plato(["150 g de tilapia", "½ taza de arroz blanco"],
               "El Toque de Fuego: cocina el arroz blanco 15-18 min. Cocina el pescado a la plancha 3-4 min por lado.")
    assert go._ensure_ingredients_used_in_recipe(m) == 0, m["recipe"]
    assert not any("Incorpora también" in p for p in m["recipe"])


def test_el_jitomate_es_el_tomate():
    m = _plato(["1 tomate", "1 cebolla"], "El Toque de Fuego: sofríe la cebolla y el jitomate 4 min.")
    assert go._ensure_ingredients_used_in_recipe(m) == 0, m["recipe"]


def test_lo_que_de_verdad_no_se_usa_se_sigue_incorporando():
    m = _plato(["150 g de tilapia", "½ taza de arroz blanco"], "El Toque de Fuego: cocina el arroz blanco 15-18 min.")
    assert go._ensure_ingredients_used_in_recipe(m) == 1


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("alias_receta").usado_por_alias(_stems, recipe_low):  # [P1-PLAN-LOTE-450]' in src
