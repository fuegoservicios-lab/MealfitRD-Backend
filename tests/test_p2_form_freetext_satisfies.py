"""[P2-FORM-FREETEXT-SATISFIES · 2026-06-27] El usuario escribió su condición a mano ("cirugía bariátrica")
en "Otra condición médica" sin marcar chip → el botón "Siguiente" del step lo aceptaba, pero el validador
global (frontend findFirstIncompleteField + backend _validate_form_data_min) miraba SOLO el array
`medicalConditions` (vacío) → lo rebotaba en bucle. Fix: ambos validadores aceptan el TEXTO LIBRE companion
(otherConditions↔medicalConditions, otherAllergies↔allergies). El backend ya mergea el companion al array
canónico downstream, así que el guard/reviewer lo ven → seguro.

+ se añadió "Cirugía Bariátrica" como chip predeterminado de condiciones (decide_meals_per_day → 6 comidas).
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


def _base_form():
    return {
        "age": 30, "mainGoal": "maintenance", "weight": 70, "height": 170, "gender": "male",
        "activityLevel": "moderate", "weightUnit": "kg", "householdSize": 1, "groceryDuration": "weekly",
        "motivation": "estar sano", "allergies": ["Ninguna"], "medicalConditions": ["Ninguna"],
        "scheduleType": "standard", "cookingTime": "30min", "budget": 5000, "sleepHours": "7",
        "stressLevel": "low", "dislikes": ["Ninguno"], "struggles": ["Ninguno"],
    }


def test_bariatric_chip_triggers_6_meals():
    assert nc.decide_meals_per_day({"medicalConditions": ["Cirugía Bariátrica"]})["num_meals"] == 6
    # también por texto libre
    assert nc.decide_meals_per_day({"otherConditions": "cirugía bariátrica"})["num_meals"] == 6


def test_validator_accepts_freetext_medical_condition():
    from routers.plans import _validate_form_data_min
    # array vacío PERO escribió a mano → válido
    ok, missing = _validate_form_data_min(dict(_base_form(), medicalConditions=[], otherConditions="cirugía bariátrica"))
    assert ok, missing
    # array vacío y SIN texto libre → falta
    ok2, missing2 = _validate_form_data_min(dict(_base_form(), medicalConditions=[], otherConditions=""))
    assert not ok2 and "medicalConditions" in missing2


def test_validator_accepts_freetext_allergy():
    from routers.plans import _validate_form_data_min
    ok, missing = _validate_form_data_min(dict(_base_form(), allergies=[], otherAllergies="maní"))
    assert ok, missing


def test_anchors_present():
    plans_src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "P2-FORM-FREETEXT-SATISFIES" in plans_src
    assert "_FREE_TEXT_COMPANION_FIELDS" in plans_src
    # chip + validador frontend (repo hermano — solo si está presente, no en CI backend-only)
    iq = _BACKEND.parent / "frontend" / "src" / "components" / "assessment" / "questions" / "QMedical.jsx"
    fv = _BACKEND.parent / "frontend" / "src" / "config" / "formValidation.js"
    if iq.exists():
        assert "Cirugía Bariátrica" in iq.read_text(encoding="utf-8")
    if fv.exists():
        assert "FREE_TEXT_COMPANION" in fv.read_text(encoding="utf-8")
