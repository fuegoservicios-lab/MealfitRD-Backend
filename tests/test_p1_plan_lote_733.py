# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-733 · 2026-09-28] «…suficiente lechosa para obtener ½ taza» dice la taza de la lista.

Batería real sobre el 659 (adulto mayor con HTA): «pela y corta suficiente lechosa para obtener ½ taza» con «1¼ tazas
de lechosa» en la lista; en el corpus, «desgrana la granada hasta obtener ½ taza» con «¼ taza de granada desgranada».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import obtener_de_la_lista as odl  # noqa: E402


@pytest.mark.parametrize("linea, paso, esperado", [
    ("1¼ tazas de lechosa",
     "Mise en place: mide 15 g de yogurt natural, pela y corta suficiente lechosa para obtener ½ taza y pesa 10 g de maní.",
     "Mise en place: mide 15 g de yogurt natural, pela y corta suficiente lechosa para obtener 1¼ tazas y pesa 10 g de maní."),
    ("¼ taza de granada desgranada",
     "Mise en place: corta el queso blanco; desgrana la granada hasta obtener ½ taza.",
     "Mise en place: corta el queso blanco; desgrana la granada hasta obtener ¼ taza."),
    ("⅔ taza de guineo",
     "Mise en place: pela y corta suficiente guineo para obtener ½ taza y mide 8 g de avena.",
     "Mise en place: pela y corta suficiente guineo para obtener ⅔ taza y mide 8 g de avena."),
])
def test_la_cifra_es_la_de_la_lista(linea, paso, esperado):
    m = {"name": "Merienda", "ingredients": [linea, "10 g de maní picado sin sal"], "recipe": [paso, "Montaje: sirve."]}
    assert odl.sincronizar(m) == 1
    assert m["recipe"][0] == esperado


def test_dos_cifras_respectivamente_no_se_toca():
    m = {"name": "Frutas picadas", "ingredients": ["1 taza de melón", "⅓ taza de lechosa"],
         "recipe": ["Mise en place: corta el melón y la lechosa en cubos hasta completar 1 taza y ½ taza "
                    "respectivamente; exprime el limón."]}
    antes = list(m["recipe"])
    assert odl.sincronizar(m) == 0 and m["recipe"] == antes


def test_en_gramos_no_se_toca():
    m = {"name": "Avena", "ingredients": ["60 g de guineo"],
         "recipe": ["Mise en place: pela y corta suficiente guineo para obtener ½ taza."]}
    antes = list(m["recipe"])
    assert odl.sincronizar(m) == 0 and m["recipe"] == antes


def test_dos_lineas_del_alimento_no_se_toca():
    m = {"name": "Bowl", "ingredients": ["1 taza de lechosa", "½ lechosa madura en cubos"],
         "recipe": ["Mise en place: pela y corta suficiente lechosa para obtener ½ taza."]}
    antes = list(m["recipe"])
    assert odl.sincronizar(m) == 0 and m["recipe"] == antes


def test_la_misma_cifra_y_las_notas_no_se_tocan():
    nota = "💡 Tip: corta suficiente lechosa para obtener ½ taza y guarda el resto."
    m = {"name": "Bowl", "ingredients": ["½ taza de lechosa"],
         "recipe": ["Mise en place: pela y corta suficiente lechosa para obtener ½ taza.", nota]}
    antes = list(m["recipe"])
    assert odl.sincronizar(m) == 0 and m["recipe"] == antes


def test_enganchado_tras_lo_que_dice_la_lista():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("obtener_de_la_lista").sincronizar(meal)')
    assert i > src.index('__import__("pasos_cantidades").lo_que_dice_la_lista(meal)')
