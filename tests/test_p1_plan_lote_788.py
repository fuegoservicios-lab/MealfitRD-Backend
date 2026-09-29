# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-788 · 2026-09-28] El punto del sustituto del pescado sólo cambia DETRÁS de su mención.

El 785 cambiaba «63 °C» y «esté opaco» en toda la frase que nombraba al sustituto; si la frase empieza por otra cosa
(«saltea la cebolla hasta que esté opaca y añade la pechuga de pollo…») ese punto no es del ave.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import embarazo_pescado as ep  # noqa: E402


def test_lo_de_antes_de_la_mencion_no_se_toca():
    m = {"name": "x", "recipe": ["El Toque de Fuego: saltea la cebolla hasta que esté opaca, añade la pechuga de pollo y "
                                 "cocina la pechuga 6 min hasta que alcance 63 °C."]}
    ep._punto_del_sustituto(m, "Pechuga de pollo")
    paso = m["recipe"][0]
    assert "la cebolla hasta que esté opaca" in paso, paso
    assert "63 °C" not in paso and "74 °C" in paso, paso


def test_lo_de_despues_si():
    m = {"name": "x", "recipe": ["El Toque de Fuego: cocina la pechuga de pavo 4 min por lado hasta que esté opaca."]}
    assert ep._punto_del_sustituto(m, "Pechuga de pavo") == 1
    assert m["recipe"][0].endswith("hasta que no quede rosada por dentro."), m["recipe"][0]
