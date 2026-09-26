# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-359 · 2026-09-26] La pista de gramos no repite la mención ni arrastra calificativos dobles.

Replay de la cola real: «retira del refrigerador 80 g de yogurt griego sin azúcar (80 g) natural sin azúcar» (bariátrico
sobre el 331), «mide 25 g de harina de maíz precocida (25 g)», «yogurt griego natural sin azúcar (160 g) sin azúcar»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_pista_sin_eco():
    m = {"recipe": [
        "Mise en place: retira del refrigerador 80 g de yogurt griego sin azúcar (80 g) natural sin azúcar y sirve.",
        "Mise en place: mide 25 g de harina de maíz precocida (25 g); pica ½ tomate.",
        "Mise en place: mide ⅔ taza de yogurt griego natural sin azúcar (160 g) sin azúcar, extrae la pulpa.",
        "Montaje: sirve el yogurt natural natural sin azúcar (90 g) sin azúcar.",
        "⚠️ Nota: 80 g de yogurt (80 g)."]}
    assert pc.pista_sin_eco(m) == 4
    assert m["recipe"][0] == "Mise en place: retira del refrigerador 80 g de yogurt griego sin azúcar y sirve."
    assert m["recipe"][1] == "Mise en place: mide 25 g de harina de maíz precocida; pica ½ tomate."
    assert m["recipe"][2] == "Mise en place: mide ⅔ taza de yogurt griego natural sin azúcar (160 g), extrae la pulpa."
    assert m["recipe"][3] == "Montaje: sirve el yogurt natural sin azúcar (90 g)."
    assert m["recipe"][4] == "⚠️ Nota: 80 g de yogurt (80 g)."                                   # las notas no se tocan


def test_lo_que_no_se_toca():
    pasos = ["Mise en place: mide 15 g de casabe, 1 cdta de mantequilla de maní (15 g) y la pimienta.",
             "Mise en place: mide 80 g de yogurt (85 g) y ⅓ taza de avena (30 g).",
             "Montaje: sirve el yogurt griego natural (80 g) sin azúcar."]
    m = {"recipe": list(pasos)}
    assert pc.pista_sin_eco(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pista_sin_eco(meal)  # [P1-PLAN-LOTE-359]' in src
