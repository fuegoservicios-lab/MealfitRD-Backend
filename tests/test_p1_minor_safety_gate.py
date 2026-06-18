"""[P1-MINOR-SAFETY-GATE · 2026-06-18] (audit fresco P1-A) Gate de seguridad para menores de edad.

El formulario acepta edades 12-17 (`_BIO_RANGES["age"]=(12,100)`) pero el pipeline los trataba como
adultos: BMR Mifflin (no validada en adolescentes), déficit -20% permitido, piso de kcal de adulto, y CERO
gate de revisión profesional por edad. Es la simétrica del gate de embarazo (P1-PREGNANCY-DEFICIT-GATE).
Este P-fix añade:
  - knob `MEALFIT_MINOR_SAFETY_GATE` (default ON) + gate en `get_nutrition_targets`: un menor (<18) nunca
    recibe déficit (lose_fat → maintenance) y queda con `minor_safety` en el result.
  - piso cinturón-y-tirantes: un menor nunca queda bajo mantenimiento (TDEE).
  - FS9 wiring (Guard 8c de `_apply_deterministic_clinical_layer`): `minor_safety` → requires_professional_review.

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
# gate en get_nutrition_targets
# ---------------------------------------------------------------------------
def test_minor_lose_fat_forced_to_maintenance(nc):
    """Un menor que pidió lose_fat NO recibe déficit: meta efectiva = maintenance, sin déficit calórico."""
    form = {
        "weight": 60, "weightUnit": "kg", "height": 165, "age": 15,
        "gender": "male", "activityLevel": "sedentary", "mainGoal": "lose_fat",
    }
    res = nc.get_nutrition_targets(form)
    ms = res.get("minor_safety")
    assert isinstance(ms, dict) and ms.get("applied") is True
    assert ms.get("age") == 15
    assert ms.get("original_goal") == "lose_fat"
    assert ms.get("effective_goal") == "maintenance"
    # Sin déficit: el objetivo no cae al 80% del TDEE (que es lo que daría lose_fat).
    assert res["target_calories"] >= res["tdee"] * 0.95, "un menor no debe quedar en déficit"


def test_minor_never_below_tdee_belt_and_suspenders(nc):
    """El piso cinturón-y-tirantes: un menor jamás queda bajo mantenimiento (TDEE)."""
    form = {
        "weight": 50, "weightUnit": "kg", "height": 158, "age": 14,
        "gender": "female", "activityLevel": "light", "mainGoal": "lose_fat",
    }
    res = nc.get_nutrition_targets(form)
    assert res.get("minor_safety", {}).get("applied") is True
    # round(tdee/50)*50 puede quedar 1-25 kcal bajo tdee por redondeo; permitimos esa tolerancia.
    assert res["target_calories"] >= int(round(res["tdee"] / 50) * 50)


def test_adult_lose_fat_not_gated(nc):
    """Un adulto (age 30) con lose_fat conserva el déficit y NO recibe minor_safety."""
    form = {
        "weight": 80, "weightUnit": "kg", "height": 180, "age": 30,
        "gender": "male", "activityLevel": "moderate", "mainGoal": "lose_fat",
    }
    res = nc.get_nutrition_targets(form)
    assert "minor_safety" not in res
    assert res["target_calories"] < res["tdee"], "un adulto con lose_fat sí recibe déficit"


def test_minor_string_decimal_age_still_gated(nc):
    """[review P2] Un age string-decimal ('17.0'/'17.5') no debe caer al default adulto (25) por
    `int('17.5')` y eludir el gate. El parse es `int(float(...))` (igual que el router)."""
    for age_val in ("17", "17.0", "17.5", 17.5, 16.9):
        form = {
            "weight": 60, "weightUnit": "kg", "height": 165, "age": age_val,
            "gender": "male", "activityLevel": "sedentary", "mainGoal": "lose_fat",
        }
        res = nc.get_nutrition_targets(form)
        assert res.get("minor_safety", {}).get("applied") is True, f"age={age_val!r} debió gatear como menor"


def test_eighteen_is_adult(nc):
    """La frontera: 18 años es adulto (no se aplica el gate)."""
    form = {
        "weight": 70, "weightUnit": "kg", "height": 175, "age": 18,
        "gender": "male", "activityLevel": "moderate", "mainGoal": "lose_fat",
    }
    res = nc.get_nutrition_targets(form)
    assert "minor_safety" not in res


def test_minor_gain_muscle_not_forced_but_flagged(nc):
    """Un menor con gain_muscle conserva el superávit (no es peligroso) PERO igual queda flagueado FS9."""
    form = {
        "weight": 62, "weightUnit": "kg", "height": 170, "age": 16,
        "gender": "male", "activityLevel": "active", "mainGoal": "gain_muscle",
    }
    res = nc.get_nutrition_targets(form)
    ms = res.get("minor_safety")
    assert isinstance(ms, dict) and ms.get("applied") is True
    assert ms.get("original_goal") == "gain_muscle"
    assert ms.get("effective_goal") == "gain_muscle"  # superávit intacto


# ---------------------------------------------------------------------------
# FS9 wiring (Guard 8c) en el clinical layer
# ---------------------------------------------------------------------------
def test_clinical_layer_sets_professional_review_on_minor(go):
    plan = {"days": []}
    nutrition = {
        "minor_safety": {"applied": True, "age": 15, "original_goal": "lose_fat",
                         "effective_goal": "maintenance"},
        "target_calories": 1850,
        "macros": {"protein_g": 120, "carbs_g": 210, "fats_g": 60},
        "total_daily_macros": {"protein_g": 120, "carbs_g": 210, "fats_g": 60},
    }
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male", "age": 15}, nutrition)
    rpr = out.get("requires_professional_review")
    assert isinstance(rpr, dict) and rpr.get("flag") is True
    assert rpr.get("minor") is True
    assert "menor" in (rpr.get("note") or "").lower()


def test_clinical_layer_no_minor_no_flag(go):
    plan = {"days": []}
    nutrition = {"target_calories": 2000, "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60},
                 "total_daily_macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60}}
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male", "age": 30}, nutrition)
    rpr = out.get("requires_professional_review")
    if isinstance(rpr, dict):
        assert not rpr.get("minor")


# ---------------------------------------------------------------------------
# Parser-anchors
# ---------------------------------------------------------------------------
def test_nutrition_calculator_anchors():
    src = _NC_PATH.read_text(encoding="utf-8")
    assert "P1-MINOR-SAFETY-GATE" in src
    assert "MINOR_SAFETY_GATE_ENABLED" in src
    assert "MEALFIT_MINOR_SAFETY_GATE" in src
    assert 'result["minor_safety"]' in src


def test_clinical_layer_anchor():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "P1-MINOR-SAFETY-GATE" in src
    # Guard 8c debe leer el flag de nutrition y setear el gate profesional.
    assert 'nutrition.get("minor_safety")' in src
