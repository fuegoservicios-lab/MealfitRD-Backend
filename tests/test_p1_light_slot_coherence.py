# -*- coding: utf-8 -*-
"""[P1-LIGHT-SLOT-COHERENCE · 2026-09-05] Dos restos de los planes vivos 2a2e2516 / c0dc2519:

(a) «40 g de arroz blanco crudo» en una merienda de tortillas con pera: el piso kcal de ganancia muscular añade
    arroz a toda comida «no dulce» y la merienda no era dulce para el léxico. La franja manda.
(b) «coloca 1 yogur griego en una olla con agua»: el swap huevo→yogur solo reconocía la cláusula fósil por
    VERBO de cocción; el huevo hervido deja su utensilio.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from culinary_coherence import step_has_cooking_verb, clause_bounds  # noqa: E402


def _day(meals):
    return [{"day": 1, "meals": meals}]


def _meal(slot, name, cals, carbs=30, protein=20, fats=10, ingredients=None):
    return {"meal": slot, "name": name, "cals": cals, "carbs": carbs, "protein": protein, "fats": fats,
            "ingredients": list(ingredients or ["1 porción"]), "ingredients_raw": list(ingredients or ["1 porción"])}


def test_gainmuscle_floor_never_adds_rice_to_light_slots(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    # solo comidas ligeras bajo el piso: NO hay dónde meter arroz ⇒ 0
    days = _day([_meal("Desayuno", "Tostadas de trigo con pera y mantequilla de maní", 300),
                 _meal("Merienda", "Tortilla de trigo tostada con pera y mantequilla de maní", 250)])
    nutrition = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}   # 2497 kcal
    added = go._repair_gainmuscle_day_kcal(days, nutrition, {"mainGoal": "gain_muscle"})
    assert added == 0
    for m in days[0]["meals"]:
        assert not any("arroz" in i.lower() for i in m["ingredients"]), m
        assert not m.get("_gainmuscle_kcal_floor")


def test_gainmuscle_floor_still_fills_the_lunch(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    days = _day([_meal("Merienda", "Tortilla de trigo tostada con pera y mantequilla de maní", 250),
                 _meal("Almuerzo", "Pollo guisado con habichuelas", 700, carbs=60, protein=50, fats=20)])
    nutrition = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}
    added = go._repair_gainmuscle_day_kcal(days, nutrition, {"mainGoal": "gain_muscle"})
    assert added > 0
    lunch, snack = days[0]["meals"][1], days[0]["meals"][0]
    assert any("arroz blanco cocido" in i for i in lunch["ingredients"])
    assert not any("arroz" in i.lower() for i in snack["ingredients"])


def test_egg_swap_detects_the_fossil_clause_by_utensil():
    assert go._stale_cooking_verb_precedes_egg_swap("coloca 1 yogur griego en una olla con agua", step_has_cooking_verb)
    assert go._stale_cooking_verb_precedes_egg_swap("pon el yogur griego en agua hirviendo 10 min", step_has_cooking_verb)
    # una mención legítima no se toca
    assert not go._stale_cooking_verb_precedes_egg_swap("sirve el yogur griego con la fruta", step_has_cooking_verb)
    assert not go._stale_cooking_verb_precedes_egg_swap("mezcla el yogur griego en un bol con la avena", step_has_cooking_verb)


def test_egg_swap_rewrites_only_the_fossil_clause():
    original = ("Mise en place: mide 40 g de avena, pela y corta el mango en cubos; "
                "coloca 1 huevo en una olla con agua.")
    swapped = ("Mise en place: mide 40 g de avena, pela y corta el mango en cubos; "
               "coloca 1 yogur griego en una olla con agua.")
    out = go._rewrite_stale_cooking_step(original, swapped, step_has_cooking_verb, clause_bounds)
    assert out.startswith("Mise en place: mide 40 g de avena, pela y corta el mango en cubos;")
    assert "olla" not in out and "Incorpora el yogur griego" in out


def test_marker_anchor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-LIGHT-SLOT-COHERENCE" in src
    assert "and not _meal_slot_is_light(m, _sa_gm)" in src
