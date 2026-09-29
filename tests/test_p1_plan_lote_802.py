# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-802 · 2026-09-28] El pan fantasma no se materializa si la lista ya trae pan, con la unidad que sea.

Batería real (embarazo, 28-sep): «Tostadas integrales con queso fresco, mango y yogurt griego entero» salió con
«1 rebanada de pan integral familiar» + «2 rebanadas de pan integral» ×2 porque el pase de carbohidratos fantasma buscaba
«lonjas de pan» (el paso) en una lista que dice «rebanada», y lo hacía en cada pasada (grafo y escudo del guardado).
"""
from __future__ import annotations

import copy

import graph_orchestrator as go


def _panes(meal):
    return [x for x in meal["ingredients"] if "pan" in str(x).lower()]


def _tostadas(ingredientes):
    return {"name": "Tostadas integrales con queso fresco, mango y yogurt griego entero",
            "ingredients": list(ingredientes), "ingredients_raw": list(ingredientes),
            "recipe": ["Mise en place: corta 20 g de queso blanco pasteurizado fresco y 95 g de mango en cubos; mide ¾ cdta "
                       "de aceite de oliva y prepara 2 lonjas de pan integral familiar.",
                       "El Toque de Fuego: tuesta las 2 lonjas de pan integral en una sartén a fuego medio durante 2-3 "
                       "minutos por lado; distribuye el aceite de oliva sobre el pan caliente.",
                       "Montaje: coloca el queso blanco sobre las tostadas y acompaña con el mango recién cortado."]}


def test_la_rebanada_de_la_lista_es_el_pan_de_las_lonjas():
    meal = _tostadas(["1 rebanada de pan integral familiar", "20 g de queso blanco pasteurizado", "95 g de mango"])
    go._add_missing_recipe_step_carbs([{"day": 3, "meals": [meal]}])
    assert _panes(meal) == ["1 rebanada de pan integral familiar"], meal["ingredients"]
    assert [x for x in meal["ingredients_raw"] if "pan" in x] == ["1 rebanada de pan integral familiar"]


def test_el_pan_que_falta_entra_una_sola_vez_aunque_el_humanizador_lo_renombre():
    meal = _tostadas(["20 g de queso blanco pasteurizado", "95 g de mango"])
    go._add_missing_recipe_step_carbs([{"day": 3, "meals": [meal]}])
    assert _panes(meal) == ["2 lonjas de pan integral"], "sin pan en la lista, el fantasma sigue materializándolo"
    # el humanizador lo muestra en rebanadas; la pasada del escudo del guardado no lo vuelve a añadir
    meal["ingredients"] = ["2 rebanadas de pan integral" if x == "2 lonjas de pan integral" else x
                           for x in meal["ingredients"]]
    antes = copy.deepcopy(meal)
    go._add_missing_recipe_step_carbs([{"day": 3, "meals": [meal]}])
    assert meal["ingredients"] == antes["ingredients"] and meal["ingredients_raw"] == antes["ingredients_raw"]


def test_otros_fantasmas_intactos():
    meal = {"name": "Revoltillo con avena", "ingredients": ["2 huevos"], "ingredients_raw": ["2 huevos"],
            "recipe": ["El Toque de Fuego: cocina la avena con agua 5 minutos y revuelve los huevos."]}
    go._add_missing_recipe_step_carbs([{"day": 1, "meals": [meal]}])
    assert any("avena" in x for x in meal["ingredients"]), meal["ingredients"]
