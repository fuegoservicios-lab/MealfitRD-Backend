# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-494 · 2026-09-27] El estado que el paso pide al alimento concuerda con el sustituto.

Replay forzado de los días 21+ (perfil estudiante económico): el plátano verde pasó a batata y el paso quedó «hierve la
batata en agua con sal durante 12-15 min hasta que esté tierno»: el participio seguía al plátano.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sustitucion_fresca as sf  # noqa: E402


def test_tierno_pasa_a_tierna():
    m = {"name": "Atún con puré", "ingredients": ["½ plátano verde mediano"], "ingredients_raw": ["½ plátano verde mediano"],
         "recipe": ["El Toque de Fuego: hierve el plátano verde en agua con sal durante 12-15 min hasta que esté tierno; "
                    "escúrrelo y májalo con el ajo."]}
    sf.sustituir_en_plato(m, 0, "½ plátano verde mediano", "140 g de batata", "batata")
    assert m["recipe"][0] == ("El Toque de Fuego: hierve la batata en agua con sal durante 12-15 min hasta que esté "
                              "tierna; escúrrela y májala con el ajo."), m["recipe"][0]


def test_el_numero_del_verbo():
    m = {"name": "Guiso", "ingredients": ["2 plátanos verdes"], "ingredients_raw": ["2 plátanos verdes"],
         "recipe": ["El Toque de Fuego: hierve los plátanos 15 minutos, hasta que estén tiernos."]}
    sf.sustituir_en_plato(m, 0, "2 plátanos verdes", "300 g de batata", "batata")
    assert m["recipe"][0] == "El Toque de Fuego: hierve la batata 15 minutos, hasta que esté tierna.", m["recipe"][0]


def test_otro_sustantivo_no_se_toca():
    m = {"name": "Guiso", "ingredients": ["½ plátano verde"], "ingredients_raw": ["½ plátano verde"],
         "recipe": ["El Toque de Fuego: hierve el plátano con el arroz hasta que el arroz esté tierno."]}
    sf.sustituir_en_plato(m, 0, "½ plátano verde", "75 g de batata", "batata")
    assert m["recipe"][0].endswith("hasta que el arroz esté tierno."), m["recipe"][0]
