# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-583 · 2026-09-27] «yogur» y «yogurt» son el mismo alimento para la identidad del plato.

Batería real (renal + gota; adulto mayor con HTA): «Bowl fresco de fresas, yogur y almendras» con 10 g de yogurt griego y
«Vasito fresco de lechosa, yogur y maní…» con 5 g. La identidad no veía el yogurt de la lista como el yogur del nombre.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import identidad_plato as idp  # noqa: E402


@pytest.mark.parametrize("nombre, linea", [
    ("Bowl fresco de fresas, yogur y almendras", "10 g de yogurt griego sin azúcar"),
    ("Vasito fresco de lechosa, yogur y maní con semillas de girasol", "5 g de yogurt natural sin azúcar"),
    ("Vasito portátil de yogurt natural con lechosa", "¾ taza de yogur griego"),
    ("Yogures con mango", "½ taza de yogurt natural"),
])
def test_el_yogur_del_nombre_es_el_yogurt_de_la_lista(nombre, linea):
    assert idp.nombrada_en_el_nombre(nombre, linea) is True
    assert idp.protege_linea({"name": nombre}, linea) is True


def test_contiene_cruza_las_grafias():
    assert idp._contiene("yogurt griego", "Yogur griego con mango") is True


@pytest.mark.parametrize("nombre, linea", [
    ("Avena cremosa con mango y canela", "½ taza de yogurt natural"),
    ("Bowl fresco de fresas y almendras", "10 g de yogurt griego sin azúcar"),
])
def test_sin_yogur_en_el_nombre_no_es_identidad(nombre, linea):
    assert idp.nombrada_en_el_nombre(nombre, linea) is False


def test_las_demas_variantes_no_cambian():
    assert idp._variantes("mango") == {"mango", "mangos", "mangoes"}
    assert "yogurt" not in idp._variantes("mango")
