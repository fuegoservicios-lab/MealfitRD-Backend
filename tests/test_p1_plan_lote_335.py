# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-335 · 2026-09-25] El nombre y los pasos nombran la variedad que trae la lista.

Baterías del 25-sep: «habichuelas blancas» en los pasos con «habichuelas negras» en la lista, «pica el ají cubanela» con
ají morrón, «tortillas de trigo» con tortilla integral."""
from __future__ import annotations

import pathlib

import culinary_coherence as cc
import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = cc.build_culinary_index([
    {"name": "Habichuelas negras", "aliases": ["frijoles negros"], "category": "Despensa"},
    {"name": "Habichuelas blancas", "aliases": ["frijoles blancos"], "category": "Despensa"},
    {"name": "Tortilla integral", "aliases": ["tortillas integrales"], "category": "Despensa"},
    {"name": "Tortilla de trigo", "aliases": ["tortillas de trigo"], "category": "Despensa"},
    {"name": "Ají morrón", "aliases": ["pimiento"], "category": "Vegetales"},
    {"name": "Ají cubanela", "aliases": ["ajíes cubanela"], "category": "Vegetales"},
    {"name": "Yogurt griego sin azúcar", "aliases": [], "category": "Lácteos"},
    {"name": "Yogurt griego entero", "aliases": ["yogurt griego", "yogur natural entero", "yogurt griego natural"],
     "category": "Lácteos"},
    {"name": "Cebolla", "aliases": [], "category": "Vegetales"},
])


def test_la_variedad_de_los_pasos_y_del_nombre_es_la_de_la_lista():
    m = {"name": "Habichuelas blancas guisadas en tortillas de trigo",
         "ingredients": ["½ taza de habichuelas negras", "2 tortillas integrales", "½ ají morrón", "½ cebolla"],
         "recipe": ["Mise en place: escurre las habichuelas blancas; pica el ají cubanela y la cebolla.",
                    "El Toque de Fuego: guisa las habichuelas blancas 10 min y calienta las tortillas de trigo.",
                    "Montaje: rellena las tortillas de trigo con las habichuelas."]}
    assert pc.variedad_de_la_lista(m, _IDX) == 7
    assert m["name"] == "Habichuelas negras guisadas en tortillas integrales"
    assert m["recipe"][0] == "Mise en place: escurre las habichuelas negras; pica el ají morrón y la cebolla."
    assert m["recipe"][1] == "El Toque de Fuego: guisa las habichuelas negras 10 min y calienta las tortillas integrales."


def test_mencion_generica_alias_del_mismo_alimento_y_notas_no_se_tocan():
    pasos = ["Montaje: sirve con el yogurt griego y la cebolla.",
             "⚠️ Alergia declarada: esta receta nombra habichuelas blancas, omítelo."]
    m = {"name": "Avena con yogurt griego", "recipe": list(pasos),
         "ingredients": ["½ taza de yogurt griego sin azúcar", "½ cebolla", "½ taza de habichuelas negras"]}
    assert pc.variedad_de_la_lista(m, _IDX) == 0 and m["recipe"] == pasos and m["name"] == "Avena con yogurt griego"
    m2 = {"name": "Avena con yogurt natural entero", "recipe": ["Montaje: sirve."],
          "ingredients": ["¼ taza de yogurt griego entero"]}
    assert pc.variedad_de_la_lista(m2, _IDX) == 0          # alias del MISMO alimento en el catálogo
    m3 = {"name": "Avena con yogurt natural entero", "recipe": ["Montaje: sirve."],
          "ingredients": ["½ taza de yogurt griego sin azúcar"]}
    assert pc.variedad_de_la_lista(m3, _IDX) == 1 and m3["name"] == "Avena con yogurt griego sin azúcar"


def test_lo_que_sigue_ya_es_la_hermana():
    # replay del 25-sep: «1 tortilla de trigo integral» → «tortilla integral integral»; «yogurt griego natural sin
    # azúcar» → «yogurt griego sin azúcar sin azúcar»
    pasos = ["Mise en place: ten lista 1 tortilla de trigo integral y el yogurt griego natural sin azúcar.",
             "Montaje: sirve."]
    m = {"name": "Wrap", "recipe": list(pasos),
         "ingredients": ["1 tortilla integral", "½ taza de yogurt griego sin azúcar"]}
    assert pc.variedad_de_la_lista(m, _IDX) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").variedad_de_la_lista(meal, index)' in src
