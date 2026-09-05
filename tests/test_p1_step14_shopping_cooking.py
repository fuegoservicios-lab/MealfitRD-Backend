# -*- coding: utf-8 -*-
"""[P1-STEP14-SHOPPING-COOKING · 2026-09-05] El paso 14 del asistente («¿Repones frescos? ¿Congelas? ¿Cocinas por
tandas?») al 100 %: (1) batch_cooking se guardaba en la política y nadie lo leía; (2) un fresco fuera de horizonte en
compra única solo generaba un aviso; (3) la Nevera propagada a los bloques no envejecía."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import horizon  # noqa: E402
import ai_helpers as ah  # noqa: E402
import dish_registry as dr  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "often"},
          "diet": {"type": "balanced"}}


def _eff(batch, cycle=30, topup=None, freezer="none"):
    return {"shopping": {"main_cycle_days": cycle, "fresh_topup_days": topup, "freezer_mode": freezer, "batch_cooking": batch}}


def test_batch_cooking_reaches_the_prompt_per_mode():
    assert "COCINA POR TANDAS" in " ".join(horizon.batch_cooking_prompt_lines(_eff("often")))
    assert "A VECES COCINA DE MÁS" in " ".join(horizon.batch_cooking_prompt_lines(_eff("sometimes")))
    assert "COCINA AL DÍA" in " ".join(horizon.batch_cooking_prompt_lines(_eff("never")))
    assert horizon.batch_cooking_prompt_lines({}) == [] and horizon.batch_cooking_prompt_lines(None) == []
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert "lines.extend(batch_cooking_prompt_lines(effective))" in src, "entra en policy_prompt_block"


def test_batch_cooking_prefers_batch_friendly_templates():
    kw = horizon._dur_kwargs(_eff("often"), 10)
    assert kw.get("prefer_batch") is True and kw.get("need_days") == 11
    assert "prefer_batch" not in horizon._dur_kwargs(_eff("never"), 10)
    cands_b = dr.template_candidates("DO", "almuerzo", None, k=6, prefer_batch=True)
    cands = dr.template_candidates("DO", "almuerzo", None, k=6)
    assert cands and cands_b
    assert all((c.get("logistics") or {}).get("batch_friendly") for c in cands_b[:1]), cands_b[:2]
    assert sum(1 for c in cands_b if (c.get("logistics") or {}).get("batch_friendly")) >= sum(
        1 for c in cands if (c.get("logistics") or {}).get("batch_friendly"))


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _days(n_before, meal_ings):
    days = [{"day": i + 1, "meals": [{"meal": "Almuerzo", "name": "x", "ingredients": ["1 taza de arroz"],
                                     "ingredients_raw": ["1 taza de arroz"]}]} for i in range(n_before)]
    days.append({"day": n_before + 1, "meals": [{"meal": "Cena", "name": "Ensalada de pescado",
                                                 "ingredients": list(meal_ings), "ingredients_raw": list(meal_ings)}]})
    return days


def test_fresh_beyond_horizon_is_substituted_not_only_warned(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _days(9, ["2 tazas de lechuga", "140 g de fresas", "150 g de filete de pescado blanco", "1 taza de arroz", "Cilantro al gusto"])
    n = go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    assert n >= 3, n
    ings = days[-1]["meals"][0]["ingredients"]
    assert "2 tazas de repollo" in ings and "140 g de manzana" in ings and "150 g de atun en agua" in ings, ings
    assert "1 taza de arroz" in ings
    assert days[-1]["meals"][0]["ingredients_raw"] == ings
    assert days[-1]["meals"][0].get("_fresh_substituted")
    # el día 1 (semana de frescos) no se toca
    assert days[0]["meals"][0]["ingredients"] == ["1 taza de arroz"]


def test_substitution_respects_diet_and_policy(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    veg = {"shopping": SINGLE["shopping"], "diet": {"type": "vegetarian"}}
    days = _days(9, ["150 g de pechuga de pollo"])
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=veg, diet="vegetarian")
    assert days[-1]["meals"][0]["ingredients"] == ["150 g de garbanzos cocidos"]
    weekly = _days(9, ["2 tazas de lechuga"])
    assert go._single_trip_fresh_substitute(weekly, db=_NoopDB(), effective=_eff("never", cycle=7), diet="balanced") == 0
    assert weekly[-1]["meals"][0]["ingredients"] == ["2 tazas de lechuga"]


def test_pantry_ages_under_single_trip():
    fd = {"_plan_policy_effective": SINGLE, "_days_offset": 7}
    out = ah._age_pantry_for_block(["Lechuga", "Fresas", "Papa", "Lentejas", "Filete de pescado blanco"], fd, 4)
    assert "Lechuga" not in out and "Fresas" not in out and "Filete de pescado blanco" not in out
    assert "Papa" in out and "Lentejas" in out
    same = ["Lechuga", "Papa"]
    assert ah._age_pantry_for_block(same, {"_plan_policy_effective": SINGLE, "_days_offset": 0}, 3) == same
    assert ah._age_pantry_for_block(same, {"_plan_policy_effective": _eff("never", cycle=7), "_days_offset": 7}, 4) == same


def test_wiring():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert "_single_trip_fresh_substitute as _stfs" in src and "_pass_n += _stfs(" in src
    src2 = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "current_pantry_ingredients = _age_pantry_for_block(current_pantry_ingredients, form_data, _dc)" in src2
    assert "tooltip-anchor: P1-STEP14-SHOPPING-COOKING" in (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
