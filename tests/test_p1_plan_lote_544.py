# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-544 · 2026-09-27] Un rango en la lista («2–3 guayabas medianas») pasa a su punto medio.

Batería real del 27-sep (rechaza pescado y berenjena, sin tiempo): la merienda «Yogurt natural con guayaba fresca y queso
cottage» con «2–3 guayabas medianas» quedaba en 89 kcal (el recálculo de macros aborta con un rango) teniendo ~190, y el
paso ya decía «lava 2½ guayabas». Corpus: 19 líneas así («1–2 ciruelas», «4.5–6 fresas frescas»).
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import linea_invertida as li  # noqa: E402


def test_el_punto_medio_en_cuartos():
    assert li.rango("2–3 guayabas medianas") == "2½ guayabas medianas"
    assert li.rango("1–2 ciruelas frescas") == "1½ ciruelas frescas"
    assert li.rango("4.5–6 fresas frescas") == "5¼ fresas frescas"
    assert li.rango("1.75–3 ciruelas") == "2½ ciruelas"


def test_lo_que_no_es_un_rango_no_se_toca():
    for linea in ("2 guayabas", "150 g de pollo", "10-12 minutos", "3–2 ciruelas", "Sal al gusto"):
        assert li.rango(linea) is None, linea


def test_la_lista_se_endereza_antes_del_motor():
    days = [{"meals": [{"ingredients": ["80 g de yogurt", "2–3 guayabas medianas", "50 g de queso cottage"],
                        "recipe": ["Mise en place: lava 2–3 guayabas medianas y córtalas en cubos."]}]}]
    assert li.normaliza_dias(days) == 1
    m = days[0]["meals"][0]
    assert m["ingredients"][1] == "2½ guayabas medianas"
    assert m["recipe"][0] == "Mise en place: lava 2½ guayabas medianas y córtalas en cubos.", m["recipe"]
