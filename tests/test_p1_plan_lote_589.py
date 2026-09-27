# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-589 · 2026-09-27] «Aguacate (0.25 unidad)» → «¼ aguacate» antes del motor.

Batería real del dueño (día 1): «½ aguacate (0.25 unidad)», «1 aceite de oliva (1 cdta)», «30 g de harina de maíz
precocida (17 g)» — la IA escribió la cantidad entre paréntesis y el motor antepuso la suya.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import linea_invertida as li  # noqa: E402


@pytest.mark.parametrize("entrada, esperado", [
    ("Aguacate (0.25 unidad)", "¼ aguacate"),
    ("Aceite de oliva (1 cdta)", "1 cdta de aceite de oliva"),
    ("Harina de maíz precocida (45 g)", "45 g de harina de maíz precocida"),
    ("Cebolla (¼ unidad)", "¼ cebolla"),
    ("Canela en polvo (1 pizca)", "1 pizca de canela en polvo"),
    ("Repollo (1 taza)", "1 taza de repollo"),
    ("Limón (0.5 unidad)", "½ limón"),
    ("Ajo (1 diente)", "1 diente de ajo"),
    ("Lechosa (30 g)", "30 g de lechosa"),
])
def test_la_cantidad_entre_parentesis_pasa_delante(entrada, esperado):
    assert li.enderezar(entrada) == esperado, li.enderezar(entrada)


@pytest.mark.parametrize("linea", [
    "160 g de nabo (½ unidad)",          # la pista del motor, detrás de su cantidad
    "½ pechuga de pollo (≈100 g)",
    "Pechuga de pollo (sin piel)",
    "Pollo (≈150 g)",
    "Sal al gusto",
])
def test_lo_demas_no_se_toca(linea):
    assert li.enderezar(linea) is None, linea


def test_el_dia_del_dueno_entra_derecho_al_motor():
    days = [{"meals": [{"ingredients": ["1 huevo", "Aguacate (0.25 unidad)", "Aceite de oliva (1 cdta)"],
                        "ingredients_raw": ["1 huevo", "Aguacate (0.25 unidad)", "Aceite de oliva (1 cdta)"],
                        "recipe": ["Mise en place: corta el aguacate."]}]}]
    assert li.normaliza_dias(days) == 2
    assert days[0]["meals"][0]["ingredients"] == ["1 huevo", "¼ aguacate", "1 cdta de aceite de oliva"]
