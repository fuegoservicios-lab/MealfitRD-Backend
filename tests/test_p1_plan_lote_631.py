# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-631 · 2026-09-28] La yema «cremosa/blanda/suave» del paso se alinea con la nota de yema firme.

Validación del 592 (celíaco, día 2): «cuaja el huevo entero con 5 claras hasta que la clara esté firme y la yema siga
cremosa» con «⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes…)» en el mismo plato.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yema_firme as yf  # noqa: E402

_NOTA = ("⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, sin partes líquidas) "
         "antes de servir; evita el huevo crudo o poco cocido.")


def _meal(paso):
    return {"name": "Casabe crujiente con aguacate, tomate aliñado y huevo", "recipe": [paso, "Montaje: sirve.", _NOTA]}


@pytest.mark.parametrize("paso, esperado", [
    ("El Toque de Fuego: cuaja el huevo entero con 5 claras hasta que la clara esté firme y la yema siga cremosa, unos "
     "4-5 minutos.",
     "El Toque de Fuego: cuaja el huevo entero con 5 claras hasta que la clara y la yema estén firmes, unos 4-5 minutos."),
    ("El Toque de Fuego: cocínalo 3-4 minutos hasta que la clara cuaje y la yema quede cremosa.",
     "El Toque de Fuego: cocínalo 3-4 minutos hasta que la clara y la yema cuajen."),
    ("El Toque de Fuego: plancha a fuego medio 3-4 minutos, hasta que la clara esté cuajada y la yema aún cremosa; sazona.",
     "El Toque de Fuego: plancha a fuego medio 3-4 minutos, hasta que la clara y la yema estén firmes; sazona."),
    ("El Toque de Fuego: plancha 2-3 min hasta que la clara cuaje por completo y la yema quede blanda.",
     "El Toque de Fuego: plancha 2-3 min hasta que la clara y la yema cuajen."),
    ("El Toque de Fuego: sírvelo cuando la yema quede suave.",
     "El Toque de Fuego: sírvelo cuando la yema quede firme."),
])
def test_con_la_nota_la_yema_es_firme(paso, esperado):
    m = _meal(paso)
    assert yf.ajustar(m) == 1
    assert m["recipe"][0] == esperado


def test_sin_la_nota_no_se_toca():
    m = {"name": "Huevos pochados", "recipe": ["El Toque de Fuego: cocínalo hasta que la clara cuaje y la yema quede cremosa."]}
    antes = list(m["recipe"])
    assert yf.ajustar(m) == 0 and m["recipe"] == antes
