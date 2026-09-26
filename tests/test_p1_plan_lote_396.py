# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-396 · 2026-09-26] El queso y el cilantro no se hierven «hasta que ablanden».

Batería REAL sobre el 379 (adulto mayor con HTA, día 3): «Incorpora cilantro picado en agua hasta que ablanden e
incorpóralo al plato» — la frase del cerrador para una legumbre con otro alimento dentro. Corpus: 15 (11 de queso)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_lo_que_no_se_hierve_deja_el_hervor():
    m = {"recipe": [
        "Mise en place: pela la yautía; pica ½ cebolla y 1 cda de cilantro.",
        "El Toque de Fuego: hierve la yautía 15-18 minutos y májala. Cocina la cebolla 4 minutos. Incorpora cilantro picado "
        "en agua hasta que ablanden e incorpóralo al plato. Cocina el pescado a la plancha.",
        "Montaje: sirve la yautía majada."]}
    assert pc.no_se_hierve(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: hierve la yautía 15-18 minutos y májala. Cocina la cebolla 4 minutos. "
                              "Cocina el pescado a la plancha."), m["recipe"][1]
    q = {"recipe": ["El Toque de Fuego: maja el ñame. Incorpora queso blanco fresco en agua hasta que ablanden e incorpóralo "
                    "al plato (~12-15 min en agua hirviendo).", "Montaje: sirve el majado."]}
    assert pc.no_se_hierve(q) == 1
    assert q["recipe"][0] == "El Toque de Fuego: maja el ñame. Incorpora queso blanco fresco al plato.", q["recipe"][0]


def test_la_legumbre_conserva_su_hervor():
    pasos = ["El Toque de Fuego: Cocina edamame en agua hasta que ablanden e incorpóralo al plato.",
             "El Toque de Fuego: Cocina lentejas secas en agua hasta que ablanden e incorpóralas al plato."]
    m = {"recipe": list(pasos)}
    assert pc.no_se_hierve(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").no_se_hierve(meal)  # [P1-PLAN-LOTE-396]' in src
