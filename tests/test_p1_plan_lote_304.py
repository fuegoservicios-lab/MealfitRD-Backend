# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-304 · 2026-09-25] La misma porción nombrada dos veces no es un reparto.

«Mise en place: … 12 g de queso blanco fresco…» y «Montaje: … desmenuza los 12 g de queso blanco fresco», con 25 g en la
lista: dos menciones del queso ⇒ el sincronizador no tocaba ninguna (para no duplicar un ingrediente repartido) y el
paso pedía la mitad de lo que se compra y se cuenta. Igual el casabe de la bariátrica (15 g dos veces, lista 20)."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go
import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_queso_de_la_mise_en_place_y_del_montaje_sigue_a_la_lista():
    m = {"ingredients": ["3 huevos", "45 g de harina de maíz precocida", "25 g de queso blanco fresco"],
         "recipe": ["Mise en place: mide 45 g de harina de maíz precocida y 12 g de queso blanco fresco bajo en sodio.",
                    "El Toque de Fuego: dora las arepitas 2-3 min por lado.",
                    "Montaje: sirve las arepitas y desmenuza los 12 g de queso blanco fresco por encima."]}
    go._sync_recipe_step_quantities(m)
    assert "25 g de queso blanco fresco bajo en sodio" in m["recipe"][0], m["recipe"][0]
    assert "desmenuza los 25 g de queso blanco fresco" in m["recipe"][2], m["recipe"][2]


def test_el_casabe_medido_y_servido():
    m = {"ingredients": ["2 huevos enteros", "20 g de casabe", "60 g de mandarina"],
         "recipe": ["Mise en place: bate 2 huevos enteros; ten listos 15 g de casabe y pela la mandarina en gajos.",
                    "El Toque de Fuego: revuelve los huevos 2-3 min hasta que cuajen.",
                    "Montaje: sirve el revoltillo con 15 g de casabe a un lado."]}
    go._sync_recipe_step_quantities(m)
    assert "ten listos 20 g de casabe" in m["recipe"][0] and "con 20 g de casabe" in m["recipe"][2], m["recipe"]


def test_la_misma_porcion_igual_a_la_lista_no_se_vuelve_restante():
    # batería renal del 25-sep: el «restante» del reparto escribía «sirve con los la lechosa fresca aparte restante»
    m = {"ingredients": ["2 huevos", "105 g de lechosa"],
         "recipe": ["Mise en place: corta 105 g de lechosa fresca en cubos y bate los huevos.",
                    "Montaje: sirve la tortilla caliente con los 105 g de lechosa fresca aparte."]}
    antes = list(m["recipe"])
    go._sync_recipe_step_quantities(m)
    assert m["recipe"] == antes, m["recipe"]


def test_un_reparto_de_verdad_no_se_toca():
    reparto = {"ingredients": ["24 g de queso blanco"],
               "recipe": ["Mezcla 12 g de queso blanco con la masa.", "Espolvorea 12 g de queso blanco encima."]}
    restante = {"ingredients": ["30 g de queso blanco"],
                "recipe": ["Mise en place: mide 12 g de queso blanco.", "Montaje: añade los 12 g de queso blanco restantes."]}
    mitad = {"ingredients": ["30 g de queso blanco"],
             "recipe": ["Mise en place: mide 12 g de queso blanco.", "Montaje: pon la otra mitad, 12 g de queso blanco."]}
    for caso in (reparto, restante, mitad):
        antes = list(caso["recipe"])
        go._sync_recipe_step_quantities(caso)
        assert caso["recipe"] == antes, caso["recipe"]


def test_la_regla_sola():
    rx = go._STEP_QTY_MENTION_RE
    pasos = ["Mise en place: mide 12 g de queso.", "Montaje: desmenuza los 12 g de queso."]
    assert pc.misma_porcion(pasos, "queso", [(12.0, "g"), (12.0, "g")], (25.0, "g", "25 g"), rx) is True
    assert pc.misma_porcion(pasos, "queso", [(12.0, "g"), (10.0, "g")], (25.0, "g", "25 g"), rx) is False   # distintas
    assert pc.misma_porcion(pasos, "queso", [(12.0, "g"), (12.0, "g")], (12.0, "g", "12 g"), rx) is True    # ya es la lista
    assert pc.misma_porcion(pasos, "queso", [(12.0, "g"), (12.0, "g")], (25.0, "cda", "25 cda"), rx) is False  # otra unidad
    assert pc.misma_porcion(pasos, "queso", [(12.0, "g"), (12.0, "g")], None, rx) is False
    assert pc.misma_porcion(None, "queso", None, None, rx) is False


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'misma_porcion([_s for _s in recipe_work' in src and "# [P1-PLAN-LOTE-304]" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-304" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
