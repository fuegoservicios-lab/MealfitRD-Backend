# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-395 · 2026-09-26] El yogur no se corta en cubos.

Batería REAL sobre el 379 (adulto mayor con HTA, día 2): «corta 1 taza de yogurt natural sin azúcar (231 g) bajo en
sodio en cubos… sirve los bollitos horneados con los cubos de yogurt» — el ajuste de sodio cambió el queso por yogur y
dejó la forma del queso."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_yogur_no_se_corta():
    m = {"recipe": [
        "Mise en place: mide 25 g de harina de trigo y 50 ml de agua; corta 1 taza de yogurt natural sin azúcar (231 g) bajo "
        "en sodio en cubos y pica ½ cebolla.",
        "Montaje: sirve los bollitos horneados con los cubos de yogurt natural sin azúcar bajo en sodio y los vegetales.",
        "Montaje: sirve las arepitas con yogurt natural sin azúcar en láminas y el palmito."]}
    assert pc.yogur_sin_forma(m) == 3
    assert m["recipe"] == [
        "Mise en place: mide 25 g de harina de trigo y 50 ml de agua; mide 1 taza de yogurt natural sin azúcar (231 g) bajo "
        "en sodio y pica ½ cebolla.",
        "Montaje: sirve los bollitos horneados con el yogurt natural sin azúcar bajo en sodio y los vegetales.",
        "Montaje: sirve las arepitas con yogurt natural sin azúcar y el palmito."], m["recipe"]


def test_el_queso_conserva_su_forma():
    pasos = ["Mise en place: corta 30 g de queso blanco en cubos y mide ½ taza de yogurt.",
             "Montaje: corta el queso y el yogurt helado en cubos.",
             "Montaje: sirve el yogurt con el mango en cubos."]
    m = {"recipe": list(pasos)}
    assert pc.yogur_sin_forma(m) == 0 and m["recipe"] == pasos, m["recipe"]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").yogur_sin_forma(meal)  # [P1-PLAN-LOTE-395]' in src
