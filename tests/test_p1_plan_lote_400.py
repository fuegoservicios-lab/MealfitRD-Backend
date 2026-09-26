# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-400 · 2026-09-26] La pista no repite el calificativo.

Corpus (embarazo, lactancia): «mide ½ taza de yogurt griego sin azúcar pasteurizado (140 g) pasteurizado»; batería real
sobre el 379 (estatina): «⅔ taza de yogurt griego sin azúcar descremado (0-2% de grasa) (135 g) descremado»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_calificativo_una_vez():
    m = {"recipe": ["Mise en place: mide ½ taza de yogurt griego sin azúcar pasteurizado (140 g) pasteurizado, 35 g de mango.",
                    "Mise en place: mide ⅔ taza de yogurt griego sin azúcar descremado (0-2% de grasa) (135 g) descremado, "
                    "lava 175 g de níspero."]}
    assert pc.pista_sin_eco(m) == 2
    assert m["recipe"][0] == "Mise en place: mide ½ taza de yogurt griego sin azúcar pasteurizado (140 g), 35 g de mango."
    assert m["recipe"][1] == ("Mise en place: mide ⅔ taza de yogurt griego sin azúcar descremado (0-2% de grasa) (135 g), "
                              "lava 175 g de níspero."), m["recipe"][1]


def test_sin_eco_no_se_toca():
    pasos = ["Mise en place: mide ½ taza de yogurt griego (140 g) pasteurizado y la fruta.",
             "Mise en place: mide 30 g de queso pasteurizado (30 g de queso) rallado."]
    m = {"recipe": list(pasos)}
    assert pc.pista_sin_eco(m) == 0 and m["recipe"] == pasos, m["recipe"]


def test_ancla():
    src = (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert "descremad[oa]|light|griego|pasteurizad[oa])" in src and "tooltip-anchor: P1-PLAN-LOTE-400" in src
