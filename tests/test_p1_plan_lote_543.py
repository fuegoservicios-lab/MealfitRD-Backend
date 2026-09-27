# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-543 · 2026-09-27] La hoja verde que ningún paso usa: fuera del desayuno y las meriendas, servida en
almuerzo y cena.

Batería real del 27-sep (hipotiroidismo + IMAO, con levotiroxina): «75g de espinacas» en un yogur con sandía, una avena
del desayuno y un casabe con lechosa, que ningún paso nombra — sólo la nota de la levotiroxina, que pide separarlas.
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import hoja_huerfana as hh  # noqa: E402

_NOTA = ("⚕️ Nota clínica: toma la levotiroxina en ayunas y separa estos alimentos (lácteos/soya/linaza/espinacas/café/"
         "toronja) al menos 4 horas de la dosis.")


def test_la_merienda_dulce_pierde_la_espinaca_huerfana():
    days = [{"meals": [{"meal": "Merienda", "name": "Yogurt griego con sandía y semillas de girasol",
                        "ingredients": ["150 g de yogurt griego", "100 g de sandía", "10 g de semillas de girasol",
                                        "75g de espinacas"],
                        "ingredients_raw": ["150 g de yogurt griego", "100 g de sandía", "10 g de semillas de girasol",
                                            "127 g de espinacas"],
                        "recipe": ["Mise en place: corta la sandía en cubos.",
                                   "Montaje: sirve el yogurt con la sandía y las semillas.", _NOTA]}]}]
    assert hh.limpiar(days) == 1
    m = days[0]["meals"][0]
    assert "75g de espinacas" not in m["ingredients"] and not any("espinaca" in x for x in m["ingredients_raw"])
    assert m["_hoja_huerfana_quitada"] == ["75g de espinacas"]
    assert hh.limpiar(days) == 0                                       # idempotente


def test_en_el_almuerzo_se_sirve():
    days = [{"meals": [{"meal": "Almuerzo", "name": "Casabe con queso blanco fresco y calabacín guisado",
                        "ingredients": ["1 torta de casabe", "30 g de queso blanco", "½ calabacín", "75g de espinacas"],
                        "recipe": ["Mise en place: corta el calabacín.", "El Toque de Fuego: guisa el calabacín 8 minutos.",
                                   "Montaje: sirve el casabe con el guiso.", _NOTA]}]}]
    assert hh.limpiar(days) == 1
    m = days[0]["meals"][0]
    assert "75g de espinacas" in m["ingredients"]
    assert m["recipe"][2] == "Montaje: sirve el casabe con el guiso. Acompaña con las espinacas frescas.", m["recipe"]
    assert hh.limpiar(days) == 0                                       # ya la nombra un paso


def test_la_hoja_que_la_receta_usa_o_el_nombre_promete_no_se_toca():
    days = [{"meals": [
        {"meal": "Desayuno", "name": "Revoltillo de espinacas",
         "ingredients": ["2 huevos", "1 taza de espinacas"], "recipe": ["Montaje: sirve el revoltillo."]},
        {"meal": "Merienda", "name": "Batido verde",
         "ingredients": ["1 taza de espinacas", "1 guineo"], "recipe": ["Montaje: licúa las espinacas con el guineo."]},
    ]}]
    assert hh.limpiar(days) == 0


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("hoja_huerfana").limpiar(days, db)  # [P1-PLAN-LOTE-543]' in src
