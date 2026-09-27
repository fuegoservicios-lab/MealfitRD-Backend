# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-541 · 2026-09-27] «½ unidad de tomate» → «½ tomate» en la lista.

Batería real del 27-sep (DM2 con insulina, día 3): «½ unidad de tomate», «½ unidad de cebolla», «½ unidad de mandarina»,
«½ unidad de pimentón» en las líneas de la lista.
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pulido_lineas as pl  # noqa: E402


def test_la_unidad_es_la_pieza():
    assert pl.pulir_linea("½ unidad de tomate") == "½ tomate"
    assert pl.pulir_linea("½ unidad de mandarina") == "½ mandarina"
    assert pl.pulir_linea("½ unidad de pimentón") == "½ pimentón"
    assert pl.pulir_linea("2 unidades de tomate") == "2 tomates"
    assert pl.pulir_linea("1 unidad de cebolla mediana") == "1 cebolla mediana"


def test_lo_demas_no_se_toca():
    # replay del corpus: «2 unidades de casabe» → «2 casabe» era peor (el casabe se cuenta en tortas)
    for linea in ("½ tomate", "1 taza de arroz", "Sal al gusto", "2 huevos", "2 unidades de casabe"):
        assert pl.pulir_linea(linea) == linea, linea
