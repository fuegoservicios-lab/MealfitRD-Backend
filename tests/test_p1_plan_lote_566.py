# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-566 · 2026-09-27] «Tus básicos» con UNA clave canónica.

Auditoría del formulario: la política (`policy_from_form`) prefería `stapleFoods` aunque fuera `[]` y el motor
(`_raw_staple_foods`) `staple_foods` aunque fuera `[]`; 4 de 12 perfiles reales tienen las dos claves.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import plan_policy as pp  # noqa: E402


def _form(**kw):
    base = {"groceryDuration": "weekly", "mainGoal": "maintenance", "dietType": "balanced"}
    base.update(kw)
    return base


def test_el_basico_de_configuracion_llega_a_la_politica():
    pol = pp.policy_from_form(_form(stapleFoods=[], staple_foods=["Plátano verde"]))
    assert [a["name"] for a in pol["food_anchors"]] == ["Plátano verde"], pol["food_anchors"]


def test_borrado_en_configuracion_es_sin_basicos():
    pol = pp.policy_from_form(_form(staple_foods=[], stapleFoods=["Yuca"]))
    assert pol["food_anchors"] == []


def test_el_motor_lee_lo_mismo():
    import graph_orchestrator as go
    assert go._raw_staple_foods(_form(staple_foods=[], stapleFoods=["Yuca"])) == []
    assert go._raw_staple_foods(_form(stapleFoods=["Yuca"])) == ["Yuca"]
    assert go._raw_staple_foods(_form(stapleFoods=[], staple_foods=["Plátano verde"])) == ["Plátano verde"]
