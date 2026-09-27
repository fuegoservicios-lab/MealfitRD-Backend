# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-580 · 2026-09-27] Con «30 min» o «1 hora», el prompt dice qué no cabe en ese tiempo.

Batería real (mujer, perder grasa, 30 min): «Pollo Guisado Criollo con Arroz Integral y Habichuelas Rojas» sumaba 48 min
de pasos; el prompt sólo decía «cada comida en 30 minutos o menos».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402


def test_treinta_minutos_nombra_el_arroz_integral_y_las_legumbres_secas():
    regla = horizon.cooking_time_rule({"cookingTime": "30min"})
    assert "arroz integral" in regla and "legumbres secas" in regla and "30 minutos" in regla, regla


def test_una_hora_nombra_las_legumbres_secas():
    regla = horizon.cooking_time_rule({"cookingTime": "1hour"})
    assert "legumbres secas" in regla and "60 minutos" in regla, regla
    assert "arroz integral" not in regla


def test_sin_tope_sigue_vacio():
    assert horizon.cooking_time_rule({"cookingTime": "plenty"}) == ""
