# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-582 · 2026-09-27] El Montaje no vuelve a servir lo que ya sirvió.

Batería real: «Sirve la ensalada con el gouda y las arepitas calientes. Acompaña con queso gouda.»
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import montaje_sin_eco as mse  # noqa: E402


@pytest.mark.parametrize("antes, despues", [
    ("Montaje: Sirve la ensalada con el gouda y las arepitas calientes. Acompaña con queso gouda.",
     "Montaje: Sirve la ensalada con el gouda y las arepitas calientes."),
    ("Montaje: sirve el mangú con la cebolla salteada y el queso blanco pasteurizado; acompaña con aguacate. "
     "Acompaña con queso blanco pasteurizado.",
     "Montaje: sirve el mangú con la cebolla salteada y el queso blanco pasteurizado; acompaña con aguacate."),
])
def test_el_eco_se_va(antes, despues):
    m = {"recipe": [antes]}
    assert mse.limpiar(m) == 1 and m["recipe"][0] == despues


@pytest.mark.parametrize("paso", [
    "Montaje: sirve acompañado de yogurt natural entero. Acompaña con yogurt griego entero.",
    "Montaje: coloca la lechosa en un plato. Termina con queso cottage.",
    "Montaje: sirve el revoltillo con el casabe. Acompaña con agua.",
    # replay del corpus: la 1.ª versión quitaba estas dos
    "Montaje: unta la mantequilla de maní natural y sirve el mango. Acompaña con yogurt natural entero.",
    "Montaje: bate el aderezo con agua caliente y sirve el bulgur. Acompaña con agua.",
])
def test_lo_que_no_se_habia_servido_se_queda(paso):
    m = {"recipe": [paso]}
    assert mse.limpiar(m) == 0 and m["recipe"][0] == paso


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("montaje_sin_eco").limpiar(meal)  # [P1-PLAN-LOTE-582]' in src
