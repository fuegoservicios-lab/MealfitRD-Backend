# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-548 · 2026-09-27] Sin respuesta de tandas, «Nada» de tiempo no es «cocina por tandas».

Auditoría del formulario: `_batch_from_cooking_time` buscaba «15/rapid/30/45» y el asistente manda
`none|30min|1hour|plenty`, así que «Nada» y «1 hora» caían en «often» → el prompt pedía «COCINA POR TANDAS… un guiso».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402
import plan_policy as pp  # noqa: E402


def _form(**kw):
    base = {"groceryDuration": "biweekly", "mainGoal": "lose_fat", "dietType": "balanced"}
    base.update(kw)
    return base


def test_nada_de_tiempo_cocina_al_dia():
    pol = pp.policy_from_form(_form(cookingTime="none"))
    assert pol["shopping"]["batch_cooking"] == "never"
    eff = {"shopping": pol["shopping"]}
    assert "COCINA POR TANDAS" not in " ".join(horizon.batch_cooking_prompt_lines(eff))


def test_con_tiempo_y_sin_respuesta_no_se_supone_tandas():
    for ct in ("30min", "1hour", "plenty"):
        assert pp.policy_from_form(_form(cookingTime=ct))["shopping"]["batch_cooking"] == "sometimes", ct


def test_la_respuesta_explicita_manda():
    pol = pp.policy_from_form(_form(cookingTime="none", batchCooking="often"))
    assert pol["shopping"]["batch_cooking"] == "often"
