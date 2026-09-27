# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-521 · 2026-09-27] El cerrador de proteína añade lo que dura en la compra única sin congelador.

Batería real del 27-sep (alérgico al pescado, días 8-11 de 30 sin congelador): el cerrador añadía «1¼ pechugas de pollo
(≈255 g)» —y «y pechuga de pollo» al nombre— y la sustitución la cambiaba después por claras topadas a 6: el cerrador
había contado 59 g de proteína y quedaban 22.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import graph_orchestrator as go  # noqa: E402

_SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none",
                        "batch_cooking": "never"}, "diet": {"type": "balanced", "allergies": []}}
_CANDS = ["Pechuga de pollo", "Atún en agua", "Claras de huevo", "Huevo", "Sardinas en lata"]


def test_desde_el_dia_4_solo_lo_que_aguanta():
    fd = {"_plan_policy_effective": _SINGLE}
    out = cu.candidatos_del_dia(_CANDS, {"day": 8}, fd)
    assert "Pechuga de pollo" not in out, out
    assert {"Atún en agua", "Claras de huevo", "Huevo", "Sardinas en lata"} <= set(out), out


def test_los_primeros_dias_y_sin_politica_la_lista_de_siempre():
    fd = {"_plan_policy_effective": _SINGLE}
    assert cu.candidatos_del_dia(_CANDS, {"day": 2}, fd) == _CANDS
    assert cu.candidatos_del_dia(_CANDS, {"day": 8}, {}) == _CANDS
    semanal = {"shopping": {"main_cycle_days": 7, "fresh_topup_days": None, "freezer_mode": "full"}}
    assert cu.candidatos_del_dia(_CANDS, {"day": 8}, {"_plan_policy_effective": semanal}) == _CANDS


def test_si_nada_aguanta_no_deja_al_cerrador_sin_candidatos():
    fd = {"_plan_policy_effective": _SINGLE}
    assert cu.candidatos_del_dia(["Pechuga de pollo"], {"day": 8}, fd) == ["Pechuga de pollo"]


def test_los_tres_cierres_de_proteina_filtran_por_el_dia():
    src = inspect.getsource(go)
    assert src.count('__import__("compra_unica").candidatos_del_dia(') == 3
