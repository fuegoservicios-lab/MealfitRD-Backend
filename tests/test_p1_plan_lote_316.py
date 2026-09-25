# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-316 · 2026-09-25] La misma frase dos veces seguidas en un paso se deja una vez.

Batería real (nocturno, 25-sep): «Montaje: … sírvelas con la yautía cocida. Acompaña con filete de pescado blanco.
Acompaña con filete de pescado blanco.»"""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc
import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _plato():
    return {"name": "Wrap", "meal": "Almuerzo", "ingredients": ["1 tortilla integral", "2 filetes de pescado"],
            "ingredients_raw": ["1 tortilla integral", "2 filetes de pescado"],
            "recipe": ["El Toque de Fuego: cocina el pescado 4 min por lado.",
                       "Montaje: sirve el wrap con la yautía cocida. Acompaña con filete de pescado blanco. "
                       "Acompaña con filete de pescado blanco."]}


def test_la_frase_repetida_se_deja_una_vez():
    m = _plato()
    assert pc.frases_repetidas(m) == 1
    assert m["recipe"][1] == "Montaje: sirve el wrap con la yautía cocida. Acompaña con filete de pescado blanco.", m["recipe"][1]
    assert pc.frases_repetidas(m) == 0


def test_lo_que_no_se_toca():
    m = {"recipe": ["Mezcla. Revuelve. Mezcla.", "⚠️ Nota. ⚠️ Nota.", "Sirve frío"]}
    assert pc.frases_repetidas(m) == 0 and m["recipe"] == ["Mezcla. Revuelve. Mezcla.", "⚠️ Nota. ⚠️ Nota.", "Sirve frío"]
    assert pc.frases_repetidas(None) == 0


def test_corre_en_el_contrato_final():
    m = _plato()
    rc._aplicar_meal(m, build_culinary_index([{"name": "Filete de pescado blanco", "aliases": ["pescado"], "category": "Proteínas"}]),
                     "repair")
    assert m["recipe"][1].count("Acompaña con filete de pescado blanco.") == 1, m["recipe"][1]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").frases_repetidas(meal)     # [P1-PLAN-LOTE-316]' in src
