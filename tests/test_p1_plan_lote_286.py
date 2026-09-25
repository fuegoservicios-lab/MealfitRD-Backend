# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-286 · 2026-09-25] Con «Nada» de tiempo la legumbre duradera de la compra única se compra LISTA.

Batería real del 25-sep con el formulario del dueño (30 días, sin congelador, «Nada», nunca por tandas): los 3 días
generados no llevaban garbanzos, pero la lista del mes compraba «1 funda (Secos 800 gr)» — la proyección del ciclo
cambiaba la proteína fresca por «garbanzos cocidos» y la lista, por precio, elegía la funda seca (remojo y una hora de
olla)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compra_unica as cu  # noqa: E402
import shopping_calculator as sc  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced", "allergies": []}}
_REQ = {"need_days": 11, "allow_frozen": False, "freezer_mode": "none"}


@pytest.fixture(autouse=True)
def _limpio():
    yield
    sc.set_single_trip_notes(False)
    cu._MEMO.clear()


def _plan(cooking_time=None):
    p = {"total_days_requested": 30, "_plan_policy": {"effective": SINGLE},
         "days": [{"day": n, "meals": [
             {"meal": "Almuerzo", "name": "Pollo con arroz", "ingredients": ["200 g de pechuga de pollo", "1 taza de arroz blanco"],
              "ingredients_raw": ["200 g de pechuga de pollo", "1 taza de arroz blanco"]},
             {"meal": "Cena", "name": "Pescado con batata", "ingredients": ["150 g de filete de pescado blanco", "100 g de batata"],
              "ingredients_raw": ["150 g de filete de pescado blanco", "100 g de batata"]}]} for n in (1, 2, 3)]}
    if cooking_time:
        p["_cooking_time"] = cooking_time
    return p


def _todas(dias):
    return [x for d in dias for m in d["meals"] for x in m["ingredients_raw"]]


def test_la_linea_del_duradero_dice_de_lata_con_nada_de_tiempo():
    r = cu.sustituir_linea("200 g de pechuga de pollo", 10, _REQ, semilla=2, contexto={"cookingTime": "none"})
    assert r and r[0] == "200 g de garbanzos de lata, escurridos" and r[1] == "garbanzos cocidos", r
    r = cu.sustituir_linea("200 g de pechuga de pollo", 10, _REQ, semilla=1, vegetal=True, contexto={"cookingTime": "none"})
    assert r and r[0] == "200 g de lentejas de lata, escurridas" and r[1] == "lentejas cocidas", r
    r = cu.sustituir_linea("200 g de pechuga de pollo", 10, _REQ, semilla=0, contexto={"cookingTime": "none"})
    assert r and r[0] == "200 g de atun en agua", r                         # lo que ya viene listo, igual


def test_con_tiempo_nada_cambia():
    for ctx in ({"cookingTime": "30min"}, {}, None):
        r = cu.sustituir_linea("200 g de pechuga de pollo", 10, _REQ, semilla=2, contexto=ctx)
        assert r and r[0] == "200 g de garbanzos cocidos", (ctx, r)


def test_la_proyeccion_del_mes_lee_el_sello_del_plan():
    resto = _todas(sc.shopping_source_days(_plan("none"))[7:])
    assert any("garbanzos de lata, escurridos" in x for x in resto), resto[:20]
    assert not any("garbanzos cocidos" in x for x in resto)
    resto30 = _todas(sc.shopping_source_days(_plan("30min"))[7:])
    assert any("garbanzos cocidos" in x for x in resto30) and not any("de lata" in x for x in resto30)
