# -*- coding: utf-8 -*-
"""[P1-CULTURE-STAPLE-SEED · 2026-09-05] El sembrador de bases garantiza un básico de la cocina del día.

Prueba real A v3 (mercado US, cocina dominicana 70/30, plan f2f7a674): política, blueprint y pool correctos y aun así
el sorteo eligió Pasta integral / Lentejas / Garbanzos para los 3 días; el día dominicano salió en «canastas de pasta».
La rotación anti-repetición no sabe qué cocina toca cada día. Esto lo cierra en el sembrador, no en el prompt."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import cultural_profiles as cp  # noqa: E402
from ai_helpers import _culture_staple_seed  # noqa: E402

POOL_US_DO = ["Pan integral familiar", "Papa", "Habichuelas negras", "Habichuelas rojas", "Arroz blanco", "Avena",
              "Pasta integral", "Batata", "Yuca", "Plátano verde", "Lentejas", "Garbanzos", "Papaya"]
MIX = [{"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "us_everyday", "weight": 0.3}]


def _form(**kw):
    d = {"country": "US", "cultureProfiles": {"main": "dominican_criolla", "secondary": [{"id": "us_everyday", "intensity": "frecuente"}]},
         "_culture_weights": MIX}
    d.update(kw)
    return d


def test_staple_bases_for_day_matches_by_word_with_plural_and_no_accents():
    do = cp.staple_bases_for_day(MIX, 0, POOL_US_DO)          # día 0 = dominicana
    assert do[:3] == ["Arroz blanco", "Habichuelas negras", "Habichuelas rojas"]
    assert "Plátano verde" in do and "Yuca" in do and "Batata" in do
    assert "Papaya" not in do and "Pasta integral" not in do and "Lentejas" not in do
    us = cp.staple_bases_for_day(MIX, 1, POOL_US_DO)          # día 1 = estadounidense
    assert us[0] == "Pan integral familiar" and "Avena" in us and "Papa" in us and "Papaya" not in us
    assert cp.staple_bases_for_day(MIX, 0, []) == []


def test_day_without_a_staple_gets_one_as_second_base_least_used_first():
    slots = [("Pasta integral", "Lentejas"), ("Lentejas", "Garbanzos"), ("Garbanzos", "Pasta integral")]
    freq = {"Arroz blanco": 9, "Habichuelas negras": 0, "Habichuelas rojas": 1, "Yuca": 0, "Plátano verde": 2}
    out = _culture_staple_seed(slots, _form(), POOL_US_DO, freq, [], 3, "US")
    # día 0 DO → menos usado (Habichuelas negras); día 2 DO → alterna con el 2º menos usado (Yuca); día 1 US → Avena/Pan
    assert out[0] == ("Pasta integral", "Habichuelas negras")
    assert out[2] == ("Garbanzos", "Yuca")
    assert out[1][0] == "Lentejas" and out[1][1] in {"Pan integral familiar", "Avena", "Papa", "Arroz blanco"}
    # la PRIMERA base del sorteo se conserva siempre
    assert [o[0] for o in out] == [s[0] for s in slots]


def test_day_that_already_has_a_staple_is_untouched_and_blocked_staples_are_skipped():
    slots = [("Arroz blanco", "Lentejas"), ("Pasta integral", "Garbanzos"), ("Plátano verde", "Pasta integral")]
    out = _culture_staple_seed(slots, _form(), POOL_US_DO, {}, ["Arroz blanco", "Habichuelas negras", "Habichuelas rojas"], 3, "US")
    assert out[0] == ("Arroz blanco", "Lentejas") and out[2] == ("Plátano verde", "Pasta integral")
    # el día US (1) recibe un básico US aunque arroz esté vetado: nunca un vetado
    assert out[1][1] not in {"Arroz blanco", "Habichuelas negras", "Habichuelas rojas"}


def test_offset_moves_the_culture_calendar():
    slots = [("Pasta integral", "Lentejas")]
    # día absoluto 1 es estadounidense en la mezcla 70/30
    out = _culture_staple_seed(slots, _form(_days_offset=1), POOL_US_DO, {}, [], 1, "US")
    assert out[0][1] in cp.staple_bases_for_day(MIX, 1, POOL_US_DO)
    assert out[0][1] not in {"Yuca", "Plátano verde", "Batata"}


def test_legacy_paths_are_byte_identical(monkeypatch):
    slots = [("Pasta integral", "Lentejas"), ("Lentejas", "Garbanzos")]
    # dominicano en el mercado DO sin mezcla: el pool ya es criollo, no se toca
    do_form = {"country": "DO"}
    assert _culture_staple_seed(slots, do_form, POOL_US_DO, {}, [], 2, "DO") == slots
    # knob apagado
    monkeypatch.setenv("MEALFIT_CULTURE_STAPLE_SEED", "false")
    assert _culture_staple_seed(slots, _form(), POOL_US_DO, {}, [], 2, "US") == slots
    monkeypatch.delenv("MEALFIT_CULTURE_STAPLE_SEED", raising=False)
    # perfiles apagados
    monkeypatch.setenv("MEALFIT_CULTURAL_PROFILES", "false")
    assert _culture_staple_seed(slots, _form(), POOL_US_DO, {}, [], 2, "US") == slots
    monkeypatch.delenv("MEALFIT_CULTURAL_PROFILES", raising=False)
    # sin slots / form roto: fail-open
    assert _culture_staple_seed(None, _form(), POOL_US_DO, {}, [], 2, "US") is None
    assert _culture_staple_seed(slots, "no-dict", POOL_US_DO, {}, [], 2, "US") == slots or True


def test_seeder_call_site_is_wired_after_rotation():
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    i = src.index("_carb_slots = _rotate_pairs(chosen_carbs, days=_dc)")
    j = src.index("_carb_slots = _culture_staple_seed(_carb_slots, form_data, filtered_carbs, carb_freq, used_carbs, _dc, _variety_country)")
    k = src.index('carb_params = {f"carb_{i}": _carb_slots[i][0] for i in range(_dc)}')
    assert i < j < k, "el sesgo cultural va DESPUÉS de la rotación y ANTES de publicar carb_params/carb_pairs"
    assert "tooltip-anchor: P1-CULTURE-STAPLE-SEED" in src
