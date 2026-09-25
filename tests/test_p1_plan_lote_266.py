# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-266 · 2026-09-25] Bariátrico: frutos secos y semillas siempre molidos o en crema, nunca enteros.

Batería final (bariátrica + SOP): «20 g de maní sin sal» en la merienda nocturna y «20 g de almendras fileteadas» en otra
merienda, contra la regla clínica bariátrica del propio sistema (riesgo de obstrucción del pouch).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import bariatrico_texturas as bt  # noqa: E402


@pytest.mark.parametrize("antes, despues", [
    ("20 g de maní sin sal", "20 g de maní molido sin sal"),
    ("20 g de almendras fileteadas", "20 g de almendras molidas"),
    ("15 g de nueces", "15 g de nueces molidas"),
    ("1 cda de semillas de chía", "1 cda de semillas de chía molidas"),
    ("10 g de linaza", "10 g de linaza molida"),
    ("15 g de almendras tostadas", "15 g de almendras molidas"),
    ("10 g de pistachos enteros", "10 g de pistachos molidos"),
])
def test_a_su_forma_molida(antes, despues):
    assert bt.moler_texto(antes) == despues


@pytest.mark.parametrize("seguro", [
    "1 cda de mantequilla de maní", "250 ml de leche de almendras", "1 cdta de aceite de ajonjolí",
    "10 g de linaza molida", "2 cdas de harina de almendras", "1 cda de tahín", "una pizca de nuez moscada",
    "1 cda de crema de maní", "200 ml de bebida de almendras",
])
def test_lo_seguro_no_se_toca(seguro):
    assert bt.moler_texto(seguro) == seguro


def _dia():
    return [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Yogurt Griego con Mango y Almendras Fileteadas", "protein": 14, "cals": 213,
         "ingredients": ["⅓ taza de yogurt griego sin azúcar", "60 g de mango", "20 g de almendras fileteadas"],
         "ingredients_raw": ["⅓ taza de yogurt griego sin azúcar", "60 g de mango", "20 g de almendras fileteadas"],
         "recipe": ["Mise en place: mide 20 g de almendras fileteadas.",
                    "Montaje: sirve el yogur con el mango y las almendras fileteadas por encima."]},
        {"meal": "Merienda Nocturna", "name": "Lechosa en Cubos con Maní", "protein": 5, "cals": 173,
         "ingredients": ["100 g de lechosa", "20 g de maní sin sal"],
         "recipe": ["Montaje: sirve la lechosa con el maní."]},
    ]}]


def test_la_comida_entera_lista_pasos_y_nombre():
    dias = _dia()
    assert bt.moler_frutos_secos(dias) == 2
    a, b = dias[0]["meals"]
    assert a["ingredients"][2] == "20 g de almendras molidas" == a["ingredients_raw"][2]
    assert a["name"] == "Yogurt Griego con Mango y Almendras Molidas"
    assert all("fileteadas" not in p for p in a["recipe"])
    assert (a["protein"], a["cals"]) == (14, 213)                     # moler no cambia la composición
    assert b["ingredients"][1] == "20 g de maní molido sin sal"
    assert b["name"] == "Lechosa en Cubos con Maní"                   # sin textura en el nombre: se deja


def _form(conds):
    return {"medicalConditions": conds}


class _DB:
    def macros_from_ingredient_string(self, s):
        return None


def test_solo_para_bariatrico_via_el_tope_de_porcion():
    import graph_orchestrator as go
    dias = _dia()
    go.cap_bariatric_portions(dias, _form(["Cirugía Bariátrica"]), db=_DB())
    assert dias[0]["meals"][1]["ingredients"][1] == "20 g de maní molido sin sal"
    otros = _dia()
    antes = copy.deepcopy(otros)
    go.cap_bariatric_portions(otros, _form(["Diabetes T2"]), db=_DB())
    assert otros == antes


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def cap_bariatric_portions(")
    j = src.index("\ndef ", i + 10)
    assert '__import__("bariatrico_texturas").moler_frutos_secos(days)' in src[i:j]
    assert "P1-PLAN-LOTE-266-MOLIDOS" in (_BACKEND / "bariatrico_texturas.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 266
