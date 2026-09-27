# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-561 · 2026-09-27] La compra única sin congelador también al cambiar un plato.

Auditoría del formulario: `attach_policy_to_swap_form` guardaba `_policy_day_index` sólo si se lo pasaban, el router
no lo pasaba y nadie lo leía; la sustitución duradera (`_single_trip_fresh_substitute`) vivía sólo en el escudo del
generador. «Mensual + solo la compra grande + no congelo» → cambiar la cena del día 18 → tilapia fresca y lechuga.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402

_EFF = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
        "diet": {"type": "balanced"}}


def _form(idx, eff=_EFF):
    return {horizon.POLICY_EFFECTIVE_KEY: eff, horizon.POLICY_DAY_INDEX_KEY: idx}


def test_dia_18_sin_congelador_lo_dice_el_prompt_del_swap():
    regla = horizon.single_trip_rule_for_swap(_form(17))
    assert "día 18" in regla and "nada de pescado, pollo ni carne frescos" in regla, regla


def test_los_primeros_dias_y_quien_repone_no_llevan_regla():
    assert horizon.single_trip_rule_for_swap(_form(0)) == ""
    repone = {"shopping": dict(_EFF["shopping"], fresh_topup_days=7)}
    assert horizon.single_trip_rule_for_swap(_form(17, repone)) == ""
    assert horizon.single_trip_rule_for_swap({}) == ""


def test_con_congelador_la_proteina_puede_ser_congelada():
    con = {"shopping": dict(_EFF["shopping"], freezer_mode="full")}
    assert "congelada" in horizon.single_trip_rule_for_swap(_form(17, con))


def test_cableado():
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert re.search(r'_attach_policy_f3\(data, user_id, plan_id=data\.get\("plan_id"\),\s*day_index=data\.get\("day_index"\)\)',
                     plans)
    assert plans.count("_stfs561(") == 2
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert '__import__("horizon").single_trip_rule_for_swap(form_data)  # [P1-PLAN-LOTE-561]' in agent
