# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-449 · 2026-09-27] Lo que el cerrador ya sirve no se acompaña otra vez (119 comidas) y los «Acompaña…»
seguidos van en una frase (176 montajes)."""
from __future__ import annotations

import pathlib

import pasos_cerrador as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_proteina_del_cerrador_se_sirve_una_vez():
    m = {"recipe": ["Mise en place: pela la yautía.",
                    "El Toque de Fuego: hierve la yautía 15-18 minutos. Cocina filete de pescado blanco a la plancha o "
                    "hervido y sírvelo como proteína del plato.",
                    "Montaje: sirve la yautía majada. Acompaña la cena con agua. Acompaña con edamame. Acompaña con filete "
                    "de pescado blanco."]}
    assert pc.proteina_servida_una_vez(m) == 1
    assert m["recipe"][2] == "Montaje: sirve la yautía majada. Acompaña la cena con agua. Acompaña con edamame."
    assert pc.proteina_servida_una_vez(m) == 0


def test_lo_escurrido_al_guiso_tampoco_se_acompana_y_concuerda():
    m = {"recipe": ["Mise en place: pica la cebolla.",
                    "El Toque de Fuego: guisa los garbanzos 6-8 min. Escurre e incorpora sardinas en lata (ya viene cocido) "
                    "al guiso en los últimos minutos.",
                    "Montaje: sirve el majado como base. Acompaña con sardinas en lata."]}
    assert pc.ya_viene_concordado(m) == 1
    assert "Escurre e incorpora sardinas en lata (ya vienen cocidas) al guiso" in m["recipe"][1]
    assert pc.proteina_servida_una_vez(m) == 1 and m["recipe"][2] == "Montaje: sirve el majado como base."
    atun = {"recipe": ["El Toque de Fuego: Escurre e incorpora atún en agua (ya viene cocido) al guiso."]}
    assert pc.ya_viene_concordado(atun) == 0


def test_los_acompanamientos_en_una_frase():
    m = {"recipe": ["Mise en place: x.", "El Toque de Fuego: y.",
                    "Montaje: sirve la yautía majada. Acompaña la cena con agua. Acompaña con edamame."]}
    assert pc.acompanamientos_en_una_frase(m) == 1
    assert m["recipe"][2] == "Montaje: sirve la yautía majada. Acompaña la cena con edamame y agua."
    assert pc.acompanamientos_en_una_frase(m) == 0
    m2 = {"recipe": ["Montaje: sirve la avena. Acompaña con agua. Acompaña con queso cottage. Acompaña con huevo."]}
    assert pc.acompanamientos_en_una_frase(m2) == 1
    assert m2["recipe"][0] == "Montaje: sirve la avena. Acompaña con queso cottage, huevo y agua."


def test_no_junta_lo_que_no_va_seguido_ni_vacia_el_montaje():
    m = {"recipe": ["Montaje: Acompaña con agua. Sirve caliente. Acompaña con edamame."]}
    assert pc.acompanamientos_en_una_frase(m) == 0
    solo = {"recipe": ["El Toque de Fuego: Cocina huevo a la plancha o hervido y sírvelo como proteína del plato.",
                       "Montaje: Acompaña con huevo."]}
    assert pc.proteina_servida_una_vez(solo) == 0 and solo["recipe"][1] == "Montaje: Acompaña con huevo."


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cerrador").proteina_servida_una_vez(meal)  # [P1-PLAN-LOTE-449]' in src
    assert '__import__("pasos_cerrador").acompanamientos_en_una_frase(meal)  # [P1-PLAN-LOTE-449]' in src
    assert '__import__("pasos_cerrador").ya_viene_concordado(meal)  # [P1-PLAN-LOTE-449]' in src
