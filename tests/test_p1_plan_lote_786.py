# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-786 · 2026-09-28] La cuenta de una pieza vaga sigue a su peso.

Bloque 3 real del plan 6594aae1: «¼ pedazo mediano de yuca (≈173 g)» (y en la lista del motor «0.25 pedazo mediano de
yuca (≈173 g)»). El resolvedor lee el paréntesis —173 g—, así que la fracción mentía: con el pedazo de la tabla casera
(400 g) son 0,43 → «½».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pieza_sigue_al_peso as p  # noqa: E402


@pytest.mark.parametrize("linea, esperado", [
    ("¼ pedazo mediano de yuca (≈173 g)", "½ pedazo mediano de yuca (≈173 g)"),
    ("0.25 pedazo mediano de yuca (≈173 g)", "0.5 pedazo mediano de yuca (≈173 g)"),
    ("1 pedazo de yautía (≈115 g)", "½ pedazo de yautía (≈115 g)"),
    ("½ pedazo de yautía (≈236 g)", "1 pedazo de yautía (≈236 g)"),
    ("1¼ pedazo de yautía (≈192 g)", "¾ pedazo de yautía (≈192 g)"),
])
def test_la_cuenta_sigue_al_peso(linea, esperado):
    assert p.corregir_linea(linea) == esperado


@pytest.mark.parametrize("linea", [
    "½ pedazo mediano de yuca (≈172 g)",     # 0,43 piezas: ½ vale
    "¾ pedazo de yautía (≈155 g)",           # 0,62: ¾ y ½ valen lo mismo; no se toca
    "¼ pedazo de yautía (≈40 g)",            # por debajo de un cuarto: el humanizador no contaría
    "1¼ filetes de pescado (≈199 g)",        # no es una pieza vaga
    "2 pedazos medianos de yuca",            # sin peso
])
def test_lo_que_ya_cuadra_no_se_toca(linea):
    assert p.corregir_linea(linea) == linea


def test_la_lista_y_la_del_motor():
    m = {"name": "Bollitos de yuca", "ingredients": ["¼ pedazo mediano de yuca (≈173 g)", "60 g de queso"],
         "ingredients_raw": ["0.25 pedazo mediano de yuca (≈173 g)", "60 g de queso"]}
    assert p.ajustar(m) == 2
    assert m["ingredients"][0].startswith("½ pedazo") and m["ingredients_raw"][0].startswith("0.5 pedazo")


def test_enganchado_en_la_cola_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("pieza_sigue_al_peso").ajustar(meal)')
    assert i < src.index('__import__("pasos_cantidades").lo_que_dice_la_lista(meal)'), "antes de sincronizar los pasos"
