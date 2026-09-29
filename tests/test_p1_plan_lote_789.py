# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-789 · 2026-09-28] El Montaje no re-sirve lo servido, aunque la frase no sea la última ni use el mismo nombre.

Bloque 3 REAL de 6594aae1 (merienda del día 5): «…coloca la mozzarella sobre las tostadas calientes… Acompaña con queso
mozzarella fresco bajo en sodio. Espolvorea las semillas de linaza por encima.» Y en el corpus de la cola 744: «coloca
encima el pescado desmenuzado… Acompaña con filete de pescado blanco.».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import montaje_sin_eco as mse  # noqa: E402


def _uno(paso):
    m = {"name": "x", "recipe": ["Mise en place: corta todo.", paso]}
    mse.limpiar(m)
    return m["recipe"][1]


def test_la_mozzarella_del_caso_real():
    paso = ("Montaje: coloca la mozzarella sobre las tostadas calientes, añade las rodajas de tomate y pimienta negra; "
            "acompaña con agua. Acompaña con queso mozzarella fresco bajo en sodio. Espolvorea las semillas de linaza "
            "por encima.")
    assert _uno(paso) == ("Montaje: coloca la mozzarella sobre las tostadas calientes, añade las rodajas de tomate y "
                          "pimienta negra; acompaña con agua. Espolvorea las semillas de linaza por encima.")


@pytest.mark.parametrize("paso, queda", [
    ("Montaje: sirve el bulgur como base, coloca encima el pescado desmenuzado con el palmito. Acompaña con filete de "
     "pescado blanco.", "Montaje: sirve el bulgur como base, coloca encima el pescado desmenuzado con el palmito."),
    ("Montaje: sirve las tortitas con el pollo desmenuzado encima y la rúcula al lado. Acompaña con pechuga de pollo "
     "cocida y desmenuzada.", "Montaje: sirve las tortitas con el pollo desmenuzado encima y la rúcula al lado."),
])
def test_la_proteina_ya_servida(paso, queda):
    assert _uno(paso) == queda


@pytest.mark.parametrize("paso", [
    "Montaje: sirve el pescado con el arroz. Acompaña con agua.",
    "Montaje: sirve el pescado con el arroz. Acompaña con pechuga de pollo.",
    "Montaje: sirve el pollo con el arroz. Acompaña con pollo y aguacate.",
    "Montaje: unta la mantequilla de maní natural sobre el casabe. Acompaña con yogurt natural.",
    "Montaje: sirve el mangú con el huevo. Termina con queso blanco.",
])
def test_lo_que_no_se_toca(paso):
    assert _uno(paso) == paso
