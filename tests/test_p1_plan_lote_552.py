# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-552 · 2026-09-27] La última palabra del escudo ve lo tecleado en «Otra alergia».

Auditoría del formulario: en el merge de un bloque (días 4+) `restricciones_finales.retirar_prohibidos` recibe el
`form_data` crudo del worker, con `otherAllergies` aparte, y sólo leía `allergies` (los chips).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import restricciones_finales as rf  # noqa: E402

_PLAN = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Pollo guisado con arroz",
                                        "ingredients": ["150 g de pechuga de pollo", "100 g de pimiento rojo",
                                                        "80 g de arroz blanco"],
                                        "ingredients_raw": ["150 g de pechuga de pollo", "100 g de pimiento rojo",
                                                            "80 g de arroz blanco"],
                                        "recipe": ["El Toque de Fuego: sofríe el pimiento y guisa el pollo."]}]}]}


def test_la_alergia_tecleada_se_retira_como_la_del_chip():
    plan = copy.deepcopy(_PLAN)
    out = rf.retirar_prohibidos(plan, {"allergies": [], "otherAllergies": "pimiento", "dietType": "balanced",
                                       "dislikes": []})
    meal = plan["days"][0]["meals"][0]
    assert [r["term"] for r in out["retiradas"]] == ["pimiento"], out
    assert meal["ingredients"] == ["150 g de pechuga de pollo", "80 g de arroz blanco"]
    assert "100 g de pimiento rojo" not in meal["ingredients_raw"]


def test_el_contexto_del_llamador_no_se_toca():
    ctx = {"allergies": [], "otherAllergies": "pimiento", "dietType": "balanced", "dislikes": []}
    rf.retirar_prohibidos(copy.deepcopy(_PLAN), ctx)
    assert ctx["allergies"] == []
