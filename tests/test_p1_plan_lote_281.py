# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-281 · 2026-09-25] Compra única: el costo del ciclo ES la compra.

Plan de 30 días del dueño (compra única, presupuesto «bajo»): la compra costaba RD$9.668 y la reconciliación decía
«RD$25.149 — tu lista supera tu referencia» (RD$14.332): multiplicaba los perecederos por las 4,3 semanas del ciclo
aunque la lista, sellada `_compra_unica`, ya era el mes entero. El ajustador de presupuesto cambiaba comida por eso.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import shopping_calculator as sc  # noqa: E402


def _item(nombre, costo, perecedero, **kw):
    return {"name": nombre, "estimated_cost_rd": costo, "is_perishable": perecedero, **kw}


_LISTA = [_item("Arroz", 1500.0, False), _item("Habichuelas", 2988.0, False),
          _item("Pechuga", 2200.0, True), _item("Guineo", 980.0, True), _item("Huevos", 2000.0, True)]   # 9.668
_SELLADA = [dict(i, _compra_unica=30) for i in _LISTA]


def _mensual(lista):
    cs = sc.compute_shopping_cost_summary(lista, lista, lista, active_duration="monthly")
    return cs, cs["by_duration"]["monthly"]


def test_compra_unica_el_ciclo_es_la_compra():
    _cs, m = _mensual(_SELLADA)
    assert m["trip_total_rd"] == 9668.0
    assert m["cycle_total_rd"] == m["trip_total_rd"]
    assert m["cycle_trips"] == 1 and m["cycle_repurchase_saving_rd"] == 0.0


def test_sin_sello_los_perecederos_se_recompran():
    _cs, m = _mensual(_LISTA)
    assert m["cycle_total_rd"] > m["trip_total_rd"] * 2     # comportamiento previo intacto para quien repone frescos


def test_la_reconciliacion_ya_no_dice_excedido():
    from nutrition_calculator import reconcile_budget_with_cost
    ref = {"tier": "low", "basis": "low", "currency": "DOP", "reference_rd": 14332, "floor_rd": 13650, "days": 30,
           "household": 1}
    cs, _m = _mensual(_SELLADA)
    br = reconcile_budget_with_cost(ref, cs, active_household=1)
    assert br["status"] == "dentro" and br["estimated_cycle_rd"] == 9668
    cs_viejo, _ = _mensual(_LISTA)
    assert reconcile_budget_with_cost(ref, cs_viejo, active_household=1)["status"] == "excedido"   # lo que decía antes


def test_ancla():
    assert "P1-PLAN-LOTE-281-COMPRA-UNICA-COSTO" in (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 281
