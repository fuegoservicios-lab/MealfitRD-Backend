# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-302 · 2026-09-25] El nombre del alimento de una mención no se traga la cantidad siguiente.

«Mise en place: mide 45 g de avena y 200 ml de leche pasteurizada» con 60 ml en la lista: la mención de la avena se
comía «avena y 200» (las palabras del alimento admitían dígitos) y la de la leche quedaba invisible — el paso seguía
pidiendo 200 ml. En las baterías del 25-sep: 5 comidas así (leche 200/250 ml contra 5-60 ml, queso 30 contra 25 g)."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_mencion_no_incluye_la_cantidad_siguiente():
    ms = [mm.group(0) for mm in go._STEP_QTY_MENTION_RE.finditer("mide 45 g de avena y 200 ml de leche descremada")]
    assert ms == ["45 g de avena y", "200 ml de leche descremada"], ms


def test_la_leche_detras_de_la_avena_sigue_a_la_lista():
    m = {"ingredients": ["45 g de avena", "60 ml de leche pasteurizada", "30 g de aguacate", "3 huevos bien cocido"],
         "recipe": ["Mise en place: mide 45 g de avena y 200 ml de leche pasteurizada; lava y corta 30 g de aguacate; "
                    "prepara 3 huevos.",
                    "El Toque de Fuego: cocina la avena con la leche y canela a fuego medio durante 5-7 minutos."]}
    go._sync_recipe_step_quantities(m)
    assert "mide 45 g de avena y 60 ml de leche pasteurizada;" in m["recipe"][0], m["recipe"][0]
    assert "200 ml" not in m["recipe"][0]


def test_el_queso_detras_del_casabe():
    m = {"ingredients": ["1 torta pequeña de casabe", "25 g de queso blanco", "60 g de guineo"],
         "recipe": ["Mise en place: porciona 15 g de casabe y 30 g de queso blanco bajo en grasa; corta el guineo."]}
    go._sync_recipe_step_quantities(m)
    assert "25 g de queso blanco bajo en grasa" in m["recipe"][0], m["recipe"][0]


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "(?:\\s+[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+){0,2})\")  # [P1-PLAN-LOTE-302]" in src
