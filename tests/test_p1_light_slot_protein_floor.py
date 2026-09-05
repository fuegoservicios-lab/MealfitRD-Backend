# -*- coding: utf-8 -*-
"""[P1-LIGHT-SLOT-PROTEIN-FLOOR · 2026-09-05] Planes vivos 56a71cc0 y 606e9017 (ganancia muscular): meriendas de 10-14 g
sobre un reparto de 20 g (15 % de 135 g) mientras almuerzo y cena entregaban 50 y 48. El bucle de FASE A se detiene en
cuanto el DÍA alcanza su piso, así que las comidas ligeras nunca se cierran."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

NUT = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}
FD = {"mainGoal": "gain_muscle", "dietType": "vegetariana"}


class _Info:
    def __init__(self, name, protein, kcal, carbs=4.0, fats=5.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


class _DB:
    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "yogur" in low:
            return {"protein": 10.0, "carbs": 4.0, "fats": 5.0, "kcal": 97.0}
        if "avena" in low or "pina" in low or "piña" in low:
            return {"protein": 2.0, "carbs": 20.0, "fats": 1.0, "kcal": 100.0}
        return None

    def lookup(self, s):
        return None


CANDS = [(1.0, "Yogurt griego", _Info("Yogurt griego", 10.0, 97.0))]


def _day():
    return [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Bowl de avena", "protein": 26, "carbs": 60, "fats": 12, "cals": 460,
         "ingredients": ["60 g de avena"], "ingredients_raw": ["60 g de avena"], "recipe": []},
        {"meal": "Almuerzo", "name": "Garbanzos guisados", "protein": 51, "carbs": 90, "fats": 18, "cals": 780,
         "ingredients": ["1 taza de garbanzos"], "ingredients_raw": ["1 taza de garbanzos"], "recipe": []},
        {"meal": "Merienda", "name": "Vaso de piña con nueces", "protein": 10, "carbs": 30, "fats": 12, "cals": 270,
         "ingredients": ["150 g de piña", "20 g de nueces"], "ingredients_raw": ["150 g de piña", "20 g de nueces"], "recipe": []},
        {"meal": "Cena", "name": "Bowl de quinoa con edamame", "protein": 48, "carbs": 80, "fats": 16, "cals": 690,
         "ingredients": ["1 taza de quinoa"], "ingredients_raw": ["1 taza de quinoa"], "recipe": []},
    ]}]


def test_light_slot_is_closed_even_when_the_day_floor_is_met(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_SLOT_PROTEIN_FLOOR", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _day()
    assert sum(m["protein"] for m in days[0]["meals"]) >= 135, "el día YA cumple el piso"
    added = go._repair_light_slot_protein(days, NUT, FD, _DB(), CANDS)
    assert added > 0
    snack = days[0]["meals"][2]
    assert snack["protein"] > 10 and any("yogur" in i.lower() for i in snack["ingredients"]), snack["ingredients"]
    # las comidas fuertes bien servidas (50 y 48 de ~47) no se tocan
    assert days[0]["meals"][1]["ingredients"] == ["1 taza de garbanzos"]
    assert days[0]["meals"][3]["ingredients"] == ["1 taza de quinoa"]


def test_quiet_when_snack_is_fine_or_other_goal_or_knob_off(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_SLOT_PROTEIN_FLOOR", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    ok = _day()
    ok[0]["meals"][2]["protein"] = 19   # ≥ 70 % de 20,25
    assert go._repair_light_slot_protein(ok, NUT, FD, _DB(), CANDS) == 0
    assert go._repair_light_slot_protein(_day(), NUT, {"mainGoal": "lose_weight"}, _DB(), CANDS) == 0
    monkeypatch.setattr(go, "LIGHT_SLOT_PROTEIN_FLOOR", False)
    assert go._repair_light_slot_protein(_day(), NUT, FD, _DB(), CANDS) == 0


def test_wired_after_the_day_floor_loop():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _repair_protein_floor_post_caps(")
    assert "added += _repair_light_slot_protein(days, nutrition, form_data, db, _cands)" in src[i:i + 12000]
    assert "tooltip-anchor: P1-LIGHT-SLOT-PROTEIN-FLOOR" in src


def test_main_meal_below_its_share_is_also_closed(monkeypatch):
    """[P1-SLOT-PROTEIN-FLOOR-ALL] plan vivo a2b40e4e: «Croquetas de papa y queso» con 19 g sobre un reparto de 47."""
    monkeypatch.setattr(go, "LIGHT_SLOT_PROTEIN_FLOOR", True)
    monkeypatch.setattr(go, "SLOT_PROTEIN_FLOOR_ALL_SLOTS", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _day()
    days[0]["meals"][1].update({"name": "Croquetas de papa y queso", "protein": 19, "cals": 400,
                                "ingredients": ["350 g de papa"], "ingredients_raw": ["350 g de papa"]})
    added = go._repair_light_slot_protein(days, NUT, FD, _DB(), CANDS)
    assert added > 0 and days[0]["meals"][1]["protein"] > 19, days[0]["meals"][1]
    monkeypatch.setattr(go, "SLOT_PROTEIN_FLOOR_ALL_SLOTS", False)
    days2 = _day()
    days2[0]["meals"][1].update({"protein": 19, "cals": 400, "ingredients": ["350 g de papa"], "ingredients_raw": ["350 g de papa"]})
    go._repair_light_slot_protein(days2, NUT, FD, _DB(), CANDS)
    assert days2[0]["meals"][1]["ingredients"] == ["350 g de papa"], "con el knob apagado, solo franjas ligeras"
