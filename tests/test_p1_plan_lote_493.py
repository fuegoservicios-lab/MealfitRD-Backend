# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-493 · 2026-09-27] El tamaño es del fresco: se va con él.

Replay forzado de los días 21+ (perfil DM2 + HTA): «mide 1 tomate mediano» → «mide 60 g de salsa de tomate mediano».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sustitucion_fresca as sf  # noqa: E402


def test_el_tamano_se_va_con_el_fresco():
    m = {"name": "Pasta integral con atún, salsa de tomate y vegetales", "ingredients": ["1 tomate mediano"],
         "ingredients_raw": ["1 tomate mediano"],
         "recipe": ["Mise en place: mide 1 tomate mediano, ½ cebolla y 1 diente de ajo.",
                    "El Toque de Fuego: sofríe la cebolla y guisa el tomate 5 minutos."]}
    sf.sustituir_en_plato(m, 0, "1 tomate mediano", "60 g de salsa de tomate", "salsa de tomate")
    assert m["recipe"][0] == "Mise en place: mide 60 g de salsa de tomate, ½ cebolla y 1 diente de ajo.", m["recipe"][0]
