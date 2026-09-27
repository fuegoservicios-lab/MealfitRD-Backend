# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-565 · 2026-09-27] El paso no pide yema líquida cuando la nota del plato exige yema firme.

Batería real (perfil tipo dueño): «plancha 2 huevos 2-3 min hasta que la clara cuaje (yema líquida)» con «⚠️ …cocina el
huevo por completo (≥71°C, yema y clara firmes…)» en el mismo plato.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yema_firme as yf  # noqa: E402

_NOTA = ("⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, sin partes líquidas) antes "
         "de servir; evita el huevo crudo o poco cocido.")


def test_plancha_con_yema_liquida_y_nota_de_yema_firme():
    m = {"recipe": ["El Toque de Fuego: plancha 2 huevos 2-3 min hasta que la clara cuaje (yema líquida).", _NOTA]}
    assert yf.ajustar(m) == 1
    assert m["recipe"][0] == "El Toque de Fuego: plancha 2 huevos 2-3 min hasta que la clara y la yema cuajen."
    assert m["recipe"][1] == _NOTA


def test_poche_de_tres_minutos():
    m = {"recipe": ["El Toque de Fuego: desliza el huevo; cocina 3 minutos para una yema líquida.", _NOTA]}
    assert yf.ajustar(m) == 1
    assert "cocina 4-5 minutos, hasta que la yema esté firme." in m["recipe"][0], m["recipe"][0]


def test_sin_nota_o_con_sin_yema_liquida_no_se_toca():
    m = {"recipe": ["El Toque de Fuego: cocina 3 minutos para una yema líquida."]}
    assert yf.ajustar(m) == 0
    m = {"recipe": ["El Toque de Fuego: revuelve 3-4 min hasta que cuajen firmes y sin yema líquida.", _NOTA]}
    assert yf.ajustar(m) == 0


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("yema_firme").ajustar(meal)  # [P1-PLAN-LOTE-565]' in src
