# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-730 · 2026-09-28] La yema «al punto» / «a tu gusto» se alinea con la nota de yema firme.

Batería real sobre el 659 (adulto mayor con HTA): «hasta que la clara cuaje y la yema quede al punto» y «…quede a tu
gusto», con «⚠️ … yema y clara firmes» en el mismo plato.
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


@pytest.mark.parametrize("paso, esperado", [
    ("El Toque de Fuego: casca 2 huevos y cocínalos 3-4 minutos hasta que la clara cuaje y la yema quede al punto.",
     "El Toque de Fuego: casca 2 huevos y cocínalos 3-4 minutos hasta que la clara y la yema cuajen."),
    ("El Toque de Fuego: tapa y cocina 6-8 minutos hasta que la clara cuaje y la yema quede a tu gusto.",
     "El Toque de Fuego: tapa y cocina 6-8 minutos hasta que la clara y la yema cuajen."),
    ("El Toque de Fuego: cocina el huevo hasta que la yema quede como te guste.",
     "El Toque de Fuego: cocina el huevo hasta que la yema quede firme."),
    ("El Toque de Fuego: cocina el huevo hasta que la yema quede al punto deseado.",
     "El Toque de Fuego: cocina el huevo hasta que la yema quede firme."),
    ("El Toque de Fuego: casca dentro 3 huevos, tapa y cocina 5-6 minutos hasta que la clara cuaje y la yema quede a "
     "gusto.",
     "El Toque de Fuego: casca dentro 3 huevos, tapa y cocina 5-6 minutos hasta que la clara y la yema cuajen."),
    # replay de la cola: sin «deseado» en el patrón salía «…hasta que la clara y la yema estén firmes deseado.»
    ("El Toque de Fuego: cuaja 1 huevo 4-5 min hasta que la clara esté firme y la yema al punto deseado. Cocina el pescado.",
     "El Toque de Fuego: cuaja 1 huevo 4-5 min hasta que la clara y la yema estén firmes. Cocina el pescado."),
    ("El Toque de Fuego: incorpora los huevos, tapa y cocina 6-8 min, hasta que las claras estén completamente cuajadas y "
     "las yemas a tu punto; mezcla suavemente.",
     "El Toque de Fuego: incorpora los huevos, tapa y cocina 6-8 min, hasta que las claras estén completamente cuajadas y "
     "las yemas firmes; mezcla suavemente."),
    ("El Toque de Fuego: cocina los huevos a la plancha hasta que las claras estén cuajadas y las yemas queden a tu gusto.",
     "El Toque de Fuego: cocina los huevos a la plancha hasta que las claras estén cuajadas y las yemas queden firmes."),
])
def test_la_yema_al_gusto_es_firme_con_la_nota(paso, esperado):
    m = {"name": "Huevos guisados", "recipe": [paso, "Montaje: sirve.", _NOTA]}
    assert yf.ajustar(m) == 1
    assert m["recipe"][0] == esperado


def test_sin_la_nota_no_se_toca():
    m = {"name": "Huevos", "recipe": ["El Toque de Fuego: cocina hasta que la clara cuaje y la yema quede al punto."]}
    antes = list(m["recipe"])
    assert yf.ajustar(m) == 0 and m["recipe"] == antes
