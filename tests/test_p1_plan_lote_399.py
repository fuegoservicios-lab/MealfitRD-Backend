# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-399 · 2026-09-26] «1 rebanadas», «1 tortas pequeñas»: con uno, en singular.

Batería REAL sobre el 379 (estatina): «mide ½ aguacate y 1 rebanadas de pan integral», «1 ajíes morrones»; corpus: 84
pasos «1 rebanadas» y 23 líneas («1 tortas pequeñas de casabe», «1 dátiles sin hueso»)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_con_uno_en_singular():
    m = {"ingredients": ["1 tortas pequeñas de casabe", "1 dátiles sin hueso", "1 ajíes morrones", "1½ tortas pequeñas de casabe"],
         "recipe": ["Mise en place: mide ½ aguacate y 1 rebanadas de pan integral; pica 1 dátiles.",
                    "⚠️ Nota: 1 rebanadas no se toca en las notas."]}
    assert pc.uno_en_singular(m) == 4
    assert m["ingredients"] == ["1 torta pequeña de casabe", "1 dátil sin hueso", "1 ají morrón", "1½ tortas pequeñas de casabe"]
    assert m["recipe"][0] == "Mise en place: mide ½ aguacate y 1 rebanada de pan integral; pica 1 dátil."
    assert m["recipe"][1] == "⚠️ Nota: 1 rebanadas no se toca en las notas."
    assert pc.uno_en_singular(m) == 0
    p = {"ingredients": ["1 dátiles picados"], "recipe": []}                        # replay: el participio también
    assert pc.uno_en_singular(p) == 1 and p["ingredients"] == ["1 dátil picado"]


def test_lo_que_no_es_uno_no_se_toca():
    textos = ["11 rebanadas de pan", "0,1 tazas", "1½ tazas de avena", "2 rebanadas de pan", "21 tomates"]
    m = {"ingredients": list(textos), "recipe": ["Mise en place: corta 11 rebanadas y 1½ tortas."]}
    assert pc.uno_en_singular(m) == 0 and m["ingredients"] == textos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").uno_en_singular(meal)  # [P1-PLAN-LOTE-399]' in src
