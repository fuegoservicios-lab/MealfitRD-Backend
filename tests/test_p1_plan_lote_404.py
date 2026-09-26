# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-404 · 2026-09-26] «1 de cebolla» → «1 cebolla».

Batería REAL sobre el 399 (perfil del dueño): «1 de cebolla roja», «1 de cebolla» en la lista (corpus: 26 líneas)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_uno_sin_de():
    m = {"ingredients": ["1 de cebolla roja", "1 de cebolla picada", "½ de cebolla", "11 de mayo"],
         "recipe": ["Mise en place: corta 1 de cebolla roja en plumas; ½ de cebolla en cubos."]}
    assert pc.uno_sin_de(m) == 3
    assert m["ingredients"] == ["1 cebolla roja", "1 cebolla picada", "½ de cebolla", "11 de mayo"]
    assert m["recipe"][0] == "Mise en place: corta 1 cebolla roja en plumas; ½ de cebolla en cubos."
    assert pc.uno_sin_de(m) == 0


def test_lo_que_lleva_de_se_queda():
    pasos = ["Mise en place: separa 1 de cada 2 huevos; 1 de los tomates va al sofrito; 1½ de cebolla."]
    m = {"recipe": list(pasos), "ingredients": ["1½ de cebolla"]}
    assert pc.uno_sin_de(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").uno_sin_de(meal)  # [P1-PLAN-LOTE-404]' in src
