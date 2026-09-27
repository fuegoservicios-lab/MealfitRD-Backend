# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-613 · 2026-09-27] La fruta del plato salado se cambia por algo que se come crudo.

Batería real (texto libre: no le gusta el aguacate, alergia a la piña, día 3): «Mangú de guineo verde con queso de hoja y
batata… sirve el mangú con el queso dorado y batata fresca cortada en cubos» — la autocorrección fruta-en-plato-salado
cambia la fruta por aguacate y, si no vale, por batata, que hay que cocinar. Corpus: 19 cambios a batata, 9 sin cocción.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_MANGU = {
    "meal": "Desayuno",
    "name": "Mangú de guineo verde con queso de hoja y lechosa",
    "ingredients": ["1 guineo verde", "50 g de queso de hoja", "85 g de lechosa", "1 cdta de aceite de oliva"],
    "recipe": ["Mise en place: pela y corta en trozos 1 guineo verde; mide 50 g de queso de hoja y 85 g de lechosa.",
               "Montaje: sirve el mangú con el queso dorado y lechosa fresca cortada en cubos."],
}


def _autofix(dislikes):
    dias = [{"day": 1, "meals": [copy.deepcopy(_MANGU)]}]
    n = go._fruit_savory_autofix(dias, {"dislikes": dislikes, "allergies": ["Mani", "Sesamo"]}, db=object())
    return n, dias[0]["meals"][0]


def test_sin_aguacate_va_tomate_y_no_batata_cruda():
    n, m = _autofix(["Aguacate"])
    assert n == 1
    assert "85 g de tomate" in m["ingredients"], m["ingredients"]
    assert "tomate" in m["name"].lower() and "batata" not in m["name"].lower(), m["name"]
    assert "tomate fresco cortado en cubos" in m["recipe"][1], m["recipe"][1]             # lote 612: lechosa → tomate
    assert not any("batata" in s.lower() for s in m["recipe"])


def test_con_aguacate_sigue_el_aguacate():
    n, m = _autofix([])
    assert n == 1 and "85 g de aguacate" in m["ingredients"], m["ingredients"]


def test_sin_aguacate_ni_tomate_queda_la_batata():
    n, m = _autofix(["Aguacate", "Tomate"])
    assert n == 1 and "85 g de batata" in m["ingredients"], m["ingredients"]
