"""[P1-GOAL-PACE-DEFICIT · 2026-07-03] El ritmo del wizard (`goalPace`: gradual/
moderado/decidido, step "Tu meta de peso" P1-CLINICAL-INTAKE) modula el déficit/
superávit DETERMINISTA en `get_nutrition_targets`. Antes solo influía vía prompt
(el LLM no fija calorías → el ritmo no cambiaba números).

Contrato:
  - Sin `goalPace` (perfiles viejos / "Sin meta específica") → GOAL_ADJUSTMENTS
    legacy intacto (-20% lose_fat / +15% gain_muscle). CERO regresión.
  - lose_fat: gradual -12% / moderado -17% / decidido -22% (monótono).
  - gain_muscle: gradual +8% / moderado +12% / decidido +15% (NUNCA excede el
    legacy +15% — más superávit = más grasa, no más músculo).
  - Solo lose_fat/gain_muscle: maintenance/performance ignoran el ritmo.
  - Gates de embarazo/menor reescriben `goal` ANTES del lookup → una embarazada/
    menor jamás recibe déficit por ritmo (target >= TDEE).
  - MIN-CALORIE-FLOOR corre DESPUÉS: perfil pequeño + decidido floorea + flag FS9.
  - Rollback: MEALFIT_GOAL_PACE_ENABLED=false (module const GOAL_PACE_ENABLED).
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_SRC = (Path(nc.__file__).resolve().parent / "nutrition_calculator.py").read_text(encoding="utf-8")


def _form(extra=None):
    # Hombre 200 lb / 175 cm / 30 años / moderate → TDEE holgado sobre el piso de
    # 1500 kcal para que los asserts de porcentaje no se contaminen con el floor.
    f = {"weight": 200, "height": 175, "age": 30, "gender": "male", "weightUnit": "lb",
         "activityLevel": "moderate", "mainGoal": "lose_fat"}
    if extra:
        f.update(extra)
    return f


def _expected_target(tdee: float, adj: float) -> int:
    return int(round((tdee * (1 + adj)) / 50) * 50)


# ---------------------------------------------------------------------------
# 1. Anchors + knob
# ---------------------------------------------------------------------------
def test_anchor_present():
    assert "P1-GOAL-PACE-DEFICIT" in _SRC
    assert "MEALFIT_GOAL_PACE_ENABLED" in _SRC
    assert "PACE_ADJUSTMENTS" in _SRC


def test_knob_registered():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_GOAL_PACE_ENABLED" in snap


def test_pace_table_shape():
    # decidido en superávit NO excede el legacy +15%; déficits dentro de [-0.25, -0.08].
    assert set(nc.PACE_ADJUSTMENTS) == {"lose_fat", "gain_muscle"}
    for pace_map in nc.PACE_ADJUSTMENTS.values():
        assert set(pace_map) == {"gradual", "moderado", "decidido"}
    assert nc.PACE_ADJUSTMENTS["gain_muscle"]["decidido"] <= nc.GOAL_ADJUSTMENTS["gain_muscle"]
    assert all(-0.25 <= v <= -0.08 for v in nc.PACE_ADJUSTMENTS["lose_fat"].values())
    assert all(0.05 <= v <= 0.18 for v in nc.PACE_ADJUSTMENTS["gain_muscle"].values())


# ---------------------------------------------------------------------------
# 2. Cero regresión sin goalPace + valores modulados
# ---------------------------------------------------------------------------
def test_no_pace_legacy_intact():
    r = nc.get_nutrition_targets(_form())
    assert r.get("goal_pace_applied") is None
    assert r["target_calories"] == _expected_target(r["tdee"], nc.GOAL_ADJUSTMENTS["lose_fat"])


def test_lose_fat_paces_monotonic_and_exact():
    targets = {}
    for pace in ("gradual", "moderado", "decidido"):
        r = nc.get_nutrition_targets(_form({"goalPace": pace}))
        assert r["goal_pace_applied"]["applied"] is True
        assert r["goal_pace_applied"]["pace"] == pace
        assert r["target_calories"] == _expected_target(
            r["tdee"], nc.PACE_ADJUSTMENTS["lose_fat"][pace]
        )
        targets[pace] = r["target_calories"]
    # Más ritmo = más déficit = menos kcal.
    assert targets["gradual"] > targets["moderado"] > targets["decidido"]


def test_gain_muscle_gradual_leaner_than_legacy():
    legacy = nc.get_nutrition_targets(_form({"mainGoal": "gain_muscle"}))
    lean = nc.get_nutrition_targets(_form({"mainGoal": "gain_muscle", "goalPace": "gradual"}))
    assert lean["target_calories"] < legacy["target_calories"]
    decidido = nc.get_nutrition_targets(_form({"mainGoal": "gain_muscle", "goalPace": "decidido"}))
    assert decidido["target_calories"] == legacy["target_calories"]  # techo = legacy +15%


def test_goal_label_reflects_real_pct():
    r = nc.get_nutrition_targets(_form({"goalPace": "gradual"}))
    assert "12%" in r["goal_label"] and "gradual" in r["goal_label"]
    assert "Ritmo (gradual)" in r["calculation_details"]


# ---------------------------------------------------------------------------
# 3. No-aplica: pace inválido, metas sin cobertura, knob off
# ---------------------------------------------------------------------------
def test_invalid_pace_ignored():
    r = nc.get_nutrition_targets(_form({"goalPace": "yolo"}))
    assert r.get("goal_pace_applied") is None
    assert r["target_calories"] == _expected_target(r["tdee"], nc.GOAL_ADJUSTMENTS["lose_fat"])


def test_maintenance_and_performance_ignore_pace():
    for goal in ("maintenance", "performance"):
        r = nc.get_nutrition_targets(_form({"mainGoal": goal, "goalPace": "decidido"}))
        assert r.get("goal_pace_applied") is None, goal


def test_knob_off_disables(monkeypatch):
    monkeypatch.setattr(nc, "GOAL_PACE_ENABLED", False)
    r = nc.get_nutrition_targets(_form({"goalPace": "decidido"}))
    assert r.get("goal_pace_applied") is None
    assert r["target_calories"] == _expected_target(r["tdee"], nc.GOAL_ADJUSTMENTS["lose_fat"])


# ---------------------------------------------------------------------------
# 4. Seguridad: gates de embarazo/menor ganan; MIN-CALORIE-FLOOR intacto
# ---------------------------------------------------------------------------
def test_pregnancy_gate_beats_pace():
    r = nc.get_nutrition_targets(_form({
        "gender": "female", "medicalConditions": ["Embarazo"], "goalPace": "decidido",
    }))
    assert r.get("goal_pace_applied") is None          # goal ya es maintenance en el lookup
    assert r["target_calories"] >= r["tdee"] - 25       # piso TDEE (redondeo a 50)
    assert r.get("pregnancy_lactation_safety", {}).get("applied") is True


def test_minor_gate_beats_pace():
    r = nc.get_nutrition_targets(_form({"age": 16, "goalPace": "decidido"}))
    assert r.get("goal_pace_applied") is None
    assert r["target_calories"] >= r["tdee"] - 25
    assert r.get("minor_safety", {}).get("applied") is True


def test_min_calorie_floor_still_applies_after_pace():
    # Mujer pequeña sedentaria + decidido → el -22% caería bajo 1200 → floor + flag FS9.
    r = nc.get_nutrition_targets(_form({
        "gender": "female", "weight": 52, "weightUnit": "kg", "height": 152,
        "age": 60, "activityLevel": "sedentary", "goalPace": "decidido",
    }))
    assert r["target_calories"] >= 1200
    assert r.get("low_calorie_floored", {}).get("applied") is True
    # El flag de pace se preserva aunque el floor haya ganado la última palabra.
    assert r.get("goal_pace_applied", {}).get("applied") is True
