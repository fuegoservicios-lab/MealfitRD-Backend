# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-370 · 2026-09-26] La pieza contada de la lista dice cuánto pesa.

Plan renal real: «¼ filete de pescado» en la lista visible y 55 g en el motor (`ingredients_raw`); un cuarto de un filete
de catálogo son 37 g. En el corpus de 314 planes, 461 líneas visibles contadas (filete, pechuga, muslo, chuleta) sin peso."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _DB:
    _T = {"¼ filete de pescado": (37.5, "Filete de pescado blanco"),
          "55 g de filete de pescado blanco": (55.0, "Filete de pescado blanco"),
          "1½ pechugas de pollo": (255.0, "Pechuga de pollo"),
          "271.78 g de pechuga de pollo cocida y enfriada": (271.78, "Pechuga de pollo")}

    def macros_from_ingredient_string(self, s):
        g, n = self._T.get(s.strip(), (0.0, ""))
        return {"grams": g, "name": n}


def test_la_pieza_contada_dice_su_peso():
    m = {"ingredients": ["¼ filete de pescado", "1 tomate"], "ingredients_raw": ["55 g de filete de pescado blanco", "1 tomate"]}
    assert pc.peso_de_la_pieza(m, _DB()) == 1 and m["ingredients"][0] == "¼ filete de pescado (≈55 g)", m["ingredients"]
    p = {"ingredients": ["1½ pechugas de pollo"], "ingredients_raw": ["1½ pechugas de pollo"]}
    assert pc.peso_de_la_pieza(p, _DB()) == 1 and p["ingredients"][0] == "1½ pechugas de pollo (≈255 g)", p["ingredients"]
    assert p["ingredients_raw"] == ["1½ pechugas de pollo"]                 # lo que mide el motor no se toca


def test_lo_que_no_se_toca():
    ya = {"ingredients": ["1¼ pechugas de pollo (≈228 g)"], "ingredients_raw": ["228 g de pechuga de pollo"]}
    cocida = {"ingredients": ["1½ pechugas de pollo"], "ingredients_raw": ["271.78 g de pechuga de pollo cocida y enfriada"]}
    ambigua = {"ingredients": ["1 pechuga de pollo"], "ingredients_raw": ["100 g de pechuga de pollo", "80 g de pechuga de pollo"]}
    for m in (ya, cocida, ambigua):
        antes = list(m["ingredients"])
        assert pc.peso_de_la_pieza(m, _DB()) == 0 and m["ingredients"] == antes, m
    assert pc.peso_de_la_pieza({"ingredients": ["1 pechuga de pollo"]}, None) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").peso_de_la_pieza(meal, db)  # [P1-PLAN-LOTE-370]' in src
