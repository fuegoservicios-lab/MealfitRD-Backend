"""[P1-MIN-CALORIE-FLOOR · 2026-06-15] (gap-audit P1-2) Piso mínimo de calorías GENERAL + gate FS9.

Antes el único floor calórico era el de embarazo/lactancia; un perfil válido pero extremo (mujer
pequeña/mayor en déficit → ~850 kcal) caía bajo el mínimo clínico sin floor ni flag. Este P-fix añade:
  - `_min_target_kcal(gender)` (1200 mujer / 1500 hombre) + floor en `get_nutrition_targets`.
  - flag `low_calorie_floored` en el result → gate de revisión profesional (FS9) en el clinical layer
    (Guard 8b de `_apply_deterministic_clinical_layer`, heredado por assemble + fallback).

Validación determinista (sin LLM/DB/créditos) + parser-anchors.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_NC_PATH = Path(__file__).resolve().parent.parent / "nutrition_calculator.py"
_GO_PATH = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# ---------------------------------------------------------------------------
# _min_target_kcal
# ---------------------------------------------------------------------------
def test_min_target_kcal_by_gender(nc):
    assert nc._min_target_kcal("female") == 1200
    assert nc._min_target_kcal("male") == 1500
    assert nc._min_target_kcal("hombre") == 1500
    assert nc._min_target_kcal(None) == 1200  # desconocido → piso más bajo (no sobre-floorea)


# ---------------------------------------------------------------------------
# floor en get_nutrition_targets
# ---------------------------------------------------------------------------
def test_small_older_woman_is_floored(nc):
    form = {
        "weight": 45, "weightUnit": "kg", "height": 150, "age": 70,
        "gender": "female", "activityLevel": "sedentary", "mainGoal": "lose_fat",
    }
    res = nc.get_nutrition_targets(form)
    assert res["target_calories"] >= 1200, "el objetivo no debe quedar bajo el piso clínico"
    lcf = res.get("low_calorie_floored")
    assert isinstance(lcf, dict) and lcf.get("applied") is True
    assert lcf.get("pre_floor_calories") < 1200, "el pre-floor debía estar bajo el piso"
    assert lcf.get("floored_to") == 1200


def test_normal_male_is_not_floored(nc):
    form = {
        "weight": 80, "weightUnit": "kg", "height": 180, "age": 30,
        "gender": "male", "activityLevel": "moderate", "mainGoal": "maintenance",
    }
    res = nc.get_nutrition_targets(form)
    assert res["target_calories"] >= 1500
    assert "low_calorie_floored" not in res, "un objetivo normal no debe activar el floor"


# ---------------------------------------------------------------------------
# FS9 wiring (Guard 8b) en el clinical layer
# ---------------------------------------------------------------------------
def test_clinical_layer_sets_professional_review_on_floor(go):
    plan = {"days": []}
    nutrition = {
        "low_calorie_floored": {"applied": True, "pre_floor_calories": 1050,
                                "floored_to": 1200, "gender": "female"},
        "target_calories": 1200,
        "macros": {"protein_g": 90, "carbs_g": 120, "fats_g": 40},
        "total_daily_macros": {"protein_g": 90, "carbs_g": 120, "fats_g": 40},
    }
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "female"}, nutrition)
    rpr = out.get("requires_professional_review")
    assert isinstance(rpr, dict) and rpr.get("flag") is True
    assert rpr.get("low_calorie_floored") is True
    assert "calórico" in (rpr.get("note") or "").lower() or "calorico" in (rpr.get("note") or "").lower()


def test_clinical_layer_no_floor_no_low_cal_flag(go):
    plan = {"days": []}
    nutrition = {"target_calories": 2000, "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60},
                 "total_daily_macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60}}
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male"}, nutrition)
    rpr = out.get("requires_professional_review")
    # sin floor y sin condiciones médicas → no se setea por low-calorie
    if isinstance(rpr, dict):
        assert not rpr.get("low_calorie_floored")


# ---------------------------------------------------------------------------
# Parser-anchors
# ---------------------------------------------------------------------------
def test_nutrition_calculator_anchors():
    src = _NC_PATH.read_text(encoding="utf-8")
    assert "P1-MIN-CALORIE-FLOOR" in src
    assert "_low_calorie_floored" in src
    assert 'result["low_calorie_floored"]' in src


def test_clinical_layer_anchor():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "P1-MIN-CALORIE-FLOOR" in src
    # Guard 8b debe leer el flag de nutrition y setear el gate profesional
    assert 'nutrition.get("low_calorie_floored")' in src
