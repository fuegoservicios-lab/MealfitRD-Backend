# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-551 · 2026-09-27] La política declara lo que SUPUSO.

Auditoría del formulario: si el usuario salta el paso opcional de compras, `policy_from_form` rellena congelador
«limited», tandas y reposición semanal, y el panel lo pinta como elección («Congelas algunos alimentos»). Contrato con el
frontend (lote 417 de la otra sesión): `requested.source.defaulted`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import plan_policy as pp  # noqa: E402


def _form(**kw):
    base = {"groceryDuration": "biweekly", "mainGoal": "lose_fat", "dietType": "balanced", "cookingTime": "30min"}
    base.update(kw)
    return base


def test_sin_respuestas_declara_los_tres_supuestos():
    pol = pp.policy_from_form(_form())
    assert pol["source"]["defaulted"] == ["freezer_mode", "batch_cooking", "fresh_topup_days"]


def test_con_respuestas_no_declara_nada_y_el_hash_no_cambia():
    pol = pp.policy_from_form(_form(freezerMode="none", batchCooking="often", freshTopup="no"))
    assert pol["source"]["defaulted"] == []
    a = pp.policy_from_form(_form(freezerMode="limited", batchCooking="sometimes", freshTopup="yes"))
    b = pp.policy_from_form(_form(freezerMode="limited", freshTopup="yes"))
    b["shopping"]["batch_cooking"] = "sometimes"
    assert a["source"]["defaulted"] == [] and b["source"]["defaulted"] == ["batch_cooking"]
    assert pp.policy_hash(a) == pp.policy_hash(b), "defaulted vive en `source`, volátil para el hash"


def test_con_ciclo_semanal_la_reposicion_no_es_un_supuesto():
    pol = pp.policy_from_form(_form(groceryDuration="weekly", freezerMode="full", batchCooking="never"))
    assert pol["source"]["defaulted"] == []
