"""[P1-GOAL-ETA · 2026-07-03] Plazo estimado honesto hasta la meta de peso.

`get_nutrition_targets` calcula `goal_eta` (semanas, kg/semana, delta) desde
`targetWeight` (P1-CLINICAL-INTAKE) usando la energética de primer orden
~7,700 kcal ≈ 1 kg sobre el gap TDEE↔target FINAL (post ritmo P1-GOAL-PACE-
DEFICIT + pisos/techos clínicos — si el floor recortó el déficit real, el ETA
se alarga o desaparece; nunca promete un ritmo que el plan no entrega).

Contrato:
  - Solo lose_fat/gain_muscle + targetWeight válido + delta ≥ 0.5 kg + gap
    ≥ 100 kcal/día. Fuera de eso → sin `goal_eta` (silencio > ficción).
  - Dirección incoherente (lose_fat con meta > actual) → sin ETA (defensivo;
    el frontend ya bloquea, pero clientes viejos no).
  - Gates de embarazo/menor fuerzan maintenance ANTES → sin ETA.
  - Piso min-kcal que deja el gap < 100 kcal/día → sin ETA (mantenimiento
    efectivo — un "~200 semanas" sería ridículo y desmotivante).
  - `graph_orchestrator` copia `goal_eta` a plan_data (junto a main_goal).
  - Rollback: MEALFIT_GOAL_ETA_ENABLED=false (module const GOAL_ETA_ENABLED).
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc

_SRC = (Path(nc.__file__).resolve().parent / "nutrition_calculator.py").read_text(encoding="utf-8")
_ORCH_SRC = (Path(nc.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _form(extra=None):
    # Hombre 200 lb / 175 cm / 30 años / moderate — TDEE holgado sobre el piso.
    f = {"weight": 200, "height": 175, "age": 30, "gender": "male", "weightUnit": "lb",
         "activityLevel": "moderate", "mainGoal": "lose_fat"}
    if extra:
        f.update(extra)
    return f


def _expected_weeks(r) -> int:
    eta = r["goal_eta"]
    daily_gap = (r["tdee"] - r["target_calories"]) if eta["direction"] == "down" \
        else (r["target_calories"] - r["tdee"])
    return max(1, int(round((eta["delta_kg"] * 7700.0) / (daily_gap * 7))))


# ---------------------------------------------------------------------------
# 1. Anchors + knob + propagación a plan_data
# ---------------------------------------------------------------------------
def test_anchor_present():
    assert "P1-GOAL-ETA" in _SRC
    assert "MEALFIT_GOAL_ETA_ENABLED" in _SRC


def test_knob_registered():
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_GOAL_ETA_ENABLED" in get_knobs_registry_snapshot()


def test_orchestrator_copies_goal_eta_to_plan_data():
    # plan_data expone goal_eta junto a main_goal — el Dashboard lo renderiza.
    assert re.search(
        r'result\["main_goal"\]\s*=\s*nutrition\["goal_label"\].*?'
        r'result\["goal_eta"\]\s*=\s*nutrition\["goal_eta"\]',
        _ORCH_SRC, re.DOTALL,
    ), "graph_orchestrator debe copiar nutrition['goal_eta'] a result['goal_eta'] tras main_goal"


# ---------------------------------------------------------------------------
# 2. Cálculo correcto (lb y kg, lose y gain, con y sin ritmo)
# ---------------------------------------------------------------------------
def test_lose_fat_eta_lb():
    r = nc.get_nutrition_targets(_form({"targetWeight": "160"}))
    eta = r["goal_eta"]
    assert eta["direction"] == "down"
    assert eta["target_weight_kg"] == round(160 / 2.20462, 1)
    assert eta["delta_kg"] > 15
    assert eta["weeks_estimate"] == _expected_weeks(r)
    assert "Meta: 160 lb" in r["calculation_details"]
    assert "semanas" in r["calculation_details"]


def test_lose_fat_eta_kg_with_pace():
    r = nc.get_nutrition_targets(_form({
        "weight": 90, "weightUnit": "kg", "targetWeight": "80", "goalPace": "gradual",
    }))
    eta = r["goal_eta"]
    assert eta["target_weight_kg"] == 80
    assert eta["delta_kg"] == 10
    assert eta["pace"] == "gradual"
    assert eta["weeks_estimate"] == _expected_weeks(r)


def test_pace_changes_eta():
    # Más déficit (decidido) → menos semanas que gradual, para la misma meta.
    gradual = nc.get_nutrition_targets(_form({"targetWeight": "170", "goalPace": "gradual"}))
    decidido = nc.get_nutrition_targets(_form({"targetWeight": "170", "goalPace": "decidido"}))
    assert decidido["goal_eta"]["weeks_estimate"] < gradual["goal_eta"]["weeks_estimate"]


def test_gain_muscle_eta_up():
    r = nc.get_nutrition_targets(_form({"mainGoal": "gain_muscle", "targetWeight": "215"}))
    eta = r["goal_eta"]
    assert eta["direction"] == "up"
    assert eta["weeks_estimate"] == _expected_weeks(r)


# ---------------------------------------------------------------------------
# 3. Silencio > ficción: casos sin ETA
# ---------------------------------------------------------------------------
def test_no_target_weight_no_eta():
    for tw in (None, "", "abc"):
        r = nc.get_nutrition_targets(_form({"targetWeight": tw}))
        assert r.get("goal_eta") is None, repr(tw)


def test_maintenance_no_eta_even_with_target():
    r = nc.get_nutrition_targets(_form({"mainGoal": "maintenance", "targetWeight": "180"}))
    assert r.get("goal_eta") is None


def test_wrong_direction_no_eta():
    # lose_fat con meta MAYOR que el peso actual (cliente viejo / dato corrupto).
    r = nc.get_nutrition_targets(_form({"targetWeight": "220"}))
    assert r.get("goal_eta") is None


def test_pregnancy_gate_kills_eta():
    r = nc.get_nutrition_targets(_form({
        "gender": "female", "medicalConditions": ["Embarazo"], "targetWeight": "160",
    }))
    assert r.get("goal_eta") is None
    assert r.get("pregnancy_lactation_safety", {}).get("applied") is True


def test_floored_deficit_below_gap_threshold_no_eta():
    # Mujer pequeña sedentaria: el MIN-CALORIE-FLOOR deja el gap TDEE↔target en
    # ~11 kcal/día → mantenimiento efectivo → un ETA sería ficción. Sin goal_eta.
    r = nc.get_nutrition_targets(_form({
        "gender": "female", "weight": 52, "weightUnit": "kg", "height": 152,
        "age": 60, "activityLevel": "sedentary", "goalPace": "decidido",
        "targetWeight": "48",
    }))
    assert r.get("low_calorie_floored", {}).get("applied") is True
    assert (r["tdee"] - r["target_calories"]) < 100
    assert r.get("goal_eta") is None


def test_knob_off_disables(monkeypatch):
    monkeypatch.setattr(nc, "GOAL_ETA_ENABLED", False)
    r = nc.get_nutrition_targets(_form({"targetWeight": "160"}))
    assert r.get("goal_eta") is None
