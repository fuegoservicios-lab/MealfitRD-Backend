# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-321 · 2026-09-25] En la compra ÚNICA un envase no se redondea hacia abajo.

Batería de cierre (perfil del dueño: 30 días de una vez, sin congelador): «Garbanzos: 7 cartones (400 gr · Rica c/u) ·
alcanza ~28 de 30 días — consúmelo en esos primeros días». El selector de envase admitía comprar hasta un 10 % de menos
(`SKU_FLOOR_MAX_UNDER_PCT`) para ahorrar un envase; con una sola compra no hay segunda vuelta: el mes sale corto."""
from __future__ import annotations

import pathlib

import pytest

import shopping_calculator as sc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_ENVASES = [{"grams": 400, "price": 95, "label": "400 gr · Rica"}, {"grams": 800, "price": 180, "label": "800 gr"}]


@pytest.fixture(autouse=True)
def _sin_compra_unica_al_salir():
    yield
    sc.set_single_trip_notes(False)


def test_con_compra_unica_el_mes_sale_cubierto():
    sc.set_single_trip_notes(True)
    sel = sc._select_market_package(2980.0, _ENVASES)
    assert sel["count"] * sel["grams"] >= 2980.0, sel               # antes: 7 × 400 = 2.800 («~28 de 30 días»)
    count, size = sc._find_best_sku(2980.0, [400.0, 800.0], 0.02)
    assert count * size >= 2980.0, (count, size)


def test_fuera_de_la_compra_unica_sigue_el_ahorro_de_siempre():
    sc.set_single_trip_notes(False)
    sel = sc._select_market_package(2980.0, _ENVASES)
    assert sel["count"] == 7 and sel["grams"] == 400, sel             # 6 % de menos, más barato: la nota avisa
    assert sc._tope_de_menos() == sc.SKU_FLOOR_MAX_UNDER_PCT


def test_el_colchon_de_un_envase_sigue_valiendo():
    sc.set_single_trip_notes(True)
    assert sc._tope_de_menos() == 0.0
    sel = sc._select_market_package(2805.0, _ENVASES)                 # falta el 0,3 % de UN envase: 7 cartones bastan
    assert sel["count"] == 7 and sel["grams"] == 400, sel


def test_ancla():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert src.count("g_total * _tope_de_menos()") == 3 and "tooltip-anchor: P1-PLAN-LOTE-321" in src
