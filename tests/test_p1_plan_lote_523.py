# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-523 · 2026-09-27] «Cocina 3 huevos y 2 claras de huevo en agua hirviendo» también es hervir.

Batería real del 27-sep (adulto mayor con HTA, día 2): el lote 390 añade «(hiérvelas dentro de su huevo entero, con
cáscara)… quita la yema» cuando un paso hierve claras sueltas, pero sólo reconocía «hierve/cuece».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def test_cocina_en_agua_hirviendo_es_hervir():
    m = {"ingredients": ["3 huevos", "2 claras de huevo"],
         "recipe": ["El Toque de Fuego: Cocina 3 huevos y 2 claras de huevo en agua hirviendo durante 10-12 minutos, hasta "
                    "que estén completamente cocidos."]}
    assert pc.claras_en_su_huevo(m) == 1
    assert "2 claras de huevo (hiérvelas dentro de su huevo entero, con cáscara)" in m["recipe"][0], m["recipe"][0]


def test_cocinar_claras_en_la_sarten_no_se_toca():
    m = {"ingredients": ["2 claras de huevo"],
         "recipe": ["El Toque de Fuego: cocina 2 claras de huevo en la sartén 2-3 minutos, hasta que cuajen."]}
    antes = list(m["recipe"])
    assert pc.claras_en_su_huevo(m) == 0 and m["recipe"] == antes
