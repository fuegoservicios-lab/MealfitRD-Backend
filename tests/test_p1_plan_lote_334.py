# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-334 · 2026-09-25] El mismo alimento no se sirve dos veces.

Perfil del dueño, batería de cierre, día 2: «El Toque de Fuego: … Sirve queso cottage al lado para acompañar. Sirve
yogurt natural entero al lado para acompañar.» y «Montaje: … Acompaña con queso cottage. Acompaña con yogurt natural
entero.» — 368 de 3.778 comidas del corpus servían así dos veces el mismo alimento."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_frase_de_al_lado_sale_si_el_montaje_ya_lo_acompana():
    m = {"recipe": [
        "Mise en place: mide 35 g de avena y corta 1 guineo.",
        "El Toque de Fuego: cocina la avena 2-3 minutos. Sirve queso cottage al lado para acompañar. Sirve yogurt natural "
        "entero al lado para acompañar.",
        "💪 Sirve el queso blanco al lado para acompañar.",
        "Montaje: sirve la avena con el guineo. Acompaña con queso cottage, queso blanco y yogurt natural entero."]}
    assert pc.servir_una_vez(m) == 3
    assert m["recipe"] == ["Mise en place: mide 35 g de avena y corta 1 guineo.",
                           "El Toque de Fuego: cocina la avena 2-3 minutos.",
                           "Montaje: sirve la avena con el guineo. Acompaña con queso cottage, queso blanco y yogurt "
                           "natural entero."]


def test_sin_acompana_la_frase_de_al_lado_es_la_unica_y_se_queda():
    pasos = ["El Toque de Fuego: cocina la avena. Sirve queso cottage al lado para acompañar.",
             "Montaje: sirve la avena. Acompaña con yogurt griego."]
    m = {"recipe": list(pasos)}
    assert pc.servir_una_vez(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").servir_una_vez(meal)' in src
