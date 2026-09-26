# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-403 · 2026-09-26] «tuesta las 1 rebanada» → «tuesta la rebanada».

Batería REAL sobre el 399 (perfil del dueño, día 1): «tuesta las 1 rebanada de pan integral» (corpus: 33)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_con_articulo_y_una_pieza_sin_numero():
    m = {"ingredients": ["1 rebanada de pan integral"],
         "recipe": ["El Toque de Fuego: tuesta las 1 rebanada de pan integral 2-3 minutos por lado.",
                    "Montaje: sirve los 1 pechuga sobre el arroz; Las 1 porción va al lado; unos 1 plátano.",
                    "⚠️ Nota: las 1 rebanada no se toca."]}
    assert pc.articulo_de_uno(m) == 2
    assert m["recipe"][0] == "El Toque de Fuego: tuesta la rebanada de pan integral 2-3 minutos por lado."
    assert m["recipe"][1] == "Montaje: sirve la pechuga sobre el arroz; La porción va al lado; un plátano."
    assert m["recipe"][2] == "⚠️ Nota: las 1 rebanada no se toca."
    assert pc.articulo_de_uno(m) == 0


def test_lo_que_no_es_uno_no_se_toca():
    pasos = ["Mise en place: corta las 11 rebanadas y los 1½ plátanos; mide 1 rebanada."]
    m = {"recipe": list(pasos), "ingredients": []}
    assert pc.articulo_de_uno(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").articulo_de_uno(meal)  # [P1-PLAN-LOTE-403]' in src
