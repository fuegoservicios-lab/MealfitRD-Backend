# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-780 · 2026-09-28] «½ tortas pequeñas de casabe» → «½ torta pequeña de casabe» en la lista.

Batería real sobre el 665 (familia de 4, día 2, merienda) y plan vivo 125e45b1 («½ plátanos verde», que el re-pulido de
frontera volvía «½ plátanos verdes»).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pulido_lineas as pl  # noqa: E402


@pytest.mark.parametrize("linea, esperado", [
    ("½ tortas pequeñas de casabe", "½ torta pequeña de casabe"),
    ("½ plátanos verdes", "½ plátano verde"),
    ("½ plátanos verde", "½ plátano verde"),
    ("½ rábanos", "½ rábano"),
    ("1 guineos maduros", "1 guineo maduro"),
    # replay del corpus: con dos adjetivos quedaba «½ plátano verde medianos»
    ("½ plátanos verdes medianos", "½ plátano verde mediano"),
    ("½ ajíes cubanela", "½ ají cubanela"),
])
def test_la_fraccion_sola_singulariza_nombre_y_adjetivo(linea, esperado):
    assert pl.pulir_linea(linea) == esperado


@pytest.mark.parametrize("linea", ["2 plátanos verdes", "1½ tortas de casabe", "3 rábanos", "½ taza de arroz blanco"])
def test_con_mas_de_uno_no_se_toca(linea):
    assert pl.pulir_linea(linea) == linea


def test_lo_que_ya_estaba_sigue_igual():
    assert pl.pulir_linea("1 tostadas de casabe") == "1 tostada de casabe"
    assert pl.pulir_linea("½ pechugas de pollo (≈134 g)") == "½ pechuga de pollo (≈134 g)"
