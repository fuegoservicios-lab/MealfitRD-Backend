# -*- coding: utf-8 -*-
"""[P1-SINGLE-TRIP-ROTATION · 2026-09-05] Bloque 3 de la prueba B (días 8-11, «solo la compra grande», sin congelador)
mandó a comprar lechuga, fresas, manzana y lechosa: el modo rotación completaba los pools con frescos nuevos del sorteo.
Y la avena seguía pudiendo ser base de almuerzo/cena (3 rechazos hoy por arepitas/bowls de avena)."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ai_helpers as ah  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"}}
SINGLE_FREEZER = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "full"}}
WEEKLY = {"shopping": {"main_cycle_days": 7, "fresh_topup_days": None, "freezer_mode": "none"}}
POOL = ["Lechuga", "Fresas", "Manzana", "Batata", "Lentejas", "Filete de pescado blanco", "Pechuga de pollo"]


def test_single_trip_keeps_only_what_lasts_until_the_end_of_the_block():
    out = ah._single_trip_durable_filter(POOL, {"_plan_policy_effective": SINGLE, "_days_offset": 7}, 4)   # días 8-11
    assert "Lechuga" not in out and "Fresas" not in out, out
    assert "Manzana" in out and "Batata" in out and "Lentejas" in out
    assert "Filete de pescado blanco" not in out and "Pechuga de pollo" not in out, "sin congelador la carne fresca no llega al día 11"
    # con congelador, lo congelable sí (dentro de la ventana)
    outf = ah._single_trip_durable_filter(POOL, {"_plan_policy_effective": SINGLE_FREEZER, "_days_offset": 7}, 4)
    assert "Pechuga de pollo" in outf and "Lechuga" not in outf


def test_no_single_trip_policy_leaves_the_pool_untouched():
    assert ah._single_trip_durable_filter(POOL, {"_plan_policy_effective": WEEKLY, "_days_offset": 7}, 4) == POOL
    assert ah._single_trip_durable_filter(POOL, {"_plan_policy_effective": SINGLE, "_days_offset": 0}, 3) == POOL   # semana de frescos
    assert ah._single_trip_durable_filter(POOL, {}, 4) == POOL
    assert ah._single_trip_durable_filter(POOL, None, 4) == POOL


def test_breakfast_cereals_never_pair_as_lunch_dinner_bases():
    assert ah._base_carbs_for_pairs(["Avena", "Papa", "Yuca"]) == ["Papa", "Yuca"]
    assert ah._base_carbs_for_pairs(["Granola", "Avena"]) == ["Granola", "Avena"], "sin alternativa, no se vacía"
    assert ah._base_carbs_for_pairs([]) == []


def test_wiring():
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert src.count("_rotate_pairs(_base_carbs_for_pairs(chosen_carbs), days=_dc)") == 2, "los dos sitios de las parejas"
    i = src.index("_min_p = PANTRY_ROTATION_MIN_PROTEINS")
    assert "unique_carbs = _single_trip_durable_filter(unique_carbs, form_data, _dc)" in src[i:i + 1500]
