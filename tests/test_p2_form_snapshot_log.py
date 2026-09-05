# -*- coding: utf-8 -*-
"""[P2-FORM-SNAPSHOT-LOG · 2026-09-05] Lo que el usuario PIDIÓ en el asistente queda legible: en el journal al arrancar
el run (`[FORM-SNAPSHOT]`) y en `plan_generation_runs.input_snapshot.form_choices`. Nace de la prueba B: la dieta
vegetariana estaba en el formulario y quien revisaba el plan tuvo que PREGUNTAR si era intencional."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from plan_policy import form_choices, form_choices_summary  # noqa: E402

FORM_B = {"mainGoal": "gain_muscle", "dietType": "vegetarian", "allergies": [], "country": "US", "totalDays": 30,
          "cultureProfiles": {"main": "dominican_criolla", "secondary": []},
          "mealOrganization": "explore", "groceryDuration": "monthly", "freshTopup": "no", "freezerMode": "none",
          "batchCooking": "never", "budget": "medium", "householdSize": 1, "weight": 135, "age": 21, "_days_offset": 0}


def test_summary_reads_like_the_wizard():
    s = form_choices_summary(FORM_B)
    for frag in ("objetivo=gain_muscle", "dieta=vegetarian", "alergias=ninguna", "país=US", "cocina=dominican_criolla",
                 "recurrencia=explore", "compra=monthly/no/none/never", "presupuesto=medium", "días=30"):
        assert frag in s, (frag, s)
    mix = dict(FORM_B, cultureProfiles={"main": "dominican_criolla", "secondary": [{"profile_id": "us_everyday", "intensity": "frecuente"}]})
    assert "cocina=dominican_criolla+us_everyday(frecuente)" in form_choices_summary(mix)
    assert form_choices_summary(None) == "(formulario vacío)"


def test_choices_keep_wizard_fields_and_drop_body_data():
    c = form_choices(FORM_B)
    assert c["dietType"] == "vegetarian" and c["freezerMode"] == "none" and c["totalDays"] == 30
    assert "weight" not in c and "age" not in c and "_days_offset" not in c
    assert "allergies" not in c, "lista vacía no se guarda"


def test_wiring_in_run_creation():
    src = (_BACKEND / "routers" / "plans_generation.py").read_text(encoding="utf-8")
    assert '"form_choices": _form_choices_fs(data)' in src
    assert "[FORM-SNAPSHOT] run=" in src
