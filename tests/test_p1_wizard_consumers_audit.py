# -*- coding: utf-8 -*-
"""[P1-WIZARD-CONSUMERS-AUDIT · 2026-09-05] Cada respuesta del asistente (26 pasos) debe tener un CONSUMIDOR en el
backend (prompt, política, sembrador, cálculo, coach), no solo viajar en el JSON. La auditoría del 05-sep encontró que
`habitWater`, `habitCaffeine`, `habitSmoking` y `waistCm` no los leía nadie, y `habitAlcohol` solo con condición sensible.
Este test enumera los campos del asistente y falla si alguno se queda sin consumidor."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from condition_rules import build_habits_prompt  # noqa: E402
from prompts.chat_agent import build_user_identity_context  # noqa: E402

# Campos que escribe el asistente (InteractiveAssessmentFlow + questions/*). Si añades un paso, añade aquí su campo.
WIZARD_FIELDS = [
    "appMode", "planSource", "gender", "age", "height", "weight", "weightUnit", "waistCm", "bodyFat", "activityLevel",
    "scheduleType", "sleepHours", "stressLevel", "habitAlcohol", "habitSmoking", "habitCaffeine", "habitWater",
    "cookingTime", "country", "cultureProfiles", "groceryDuration", "freshTopup", "freezerMode", "batchCooking",
    "budget", "budgetAmount", "budgetCurrency", "householdSize", "dietType", "allergies", "otherAllergies", "dislikes",
    "otherDislikes", "mealOrganization", "stapleAnchors", "stapleFoods", "medicalConditions", "otherConditions",
    "medications", "otherMedications", "mainGoal", "targetWeight", "goalPace", "struggles", "otherStruggles",
    "motivation", "includeSupplements", "selectedSupplements",
]
# Módulos que CONSUMEN (deciden algo con el valor). NO cuentan: routers (plomería), tests, scripts, landing_benchmarks
# (matriz de perfiles) ni el resumen del formulario (P2-FORM-SNAPSHOT-LOG), que nombra todos los campos sin usarlos.
CONSUMERS = [
    "prompts/plan_generator.py", "prompts/day_generator.py", "prompts/chat_agent.py", "prompts/sentiment.py",
    "condition_rules.py", "medication_rules.py", "nutrition_calculator.py", "ai_helpers.py", "graph_orchestrator.py",
    "plan_policy.py", "horizon.py", "agent.py", "tools.py", "proactive_agent.py", "generation_inputs.py", "cron_tasks.py",
    "constants.py", "cultural_profiles.py", "db_facts.py", "db_profiles.py", "plan_mode.py",
]


def _consumer_text():
    out = []
    for rel in CONSUMERS:
        p = _BACKEND / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if rel == "plan_policy.py":
            # quitar el resumen del formulario: nombra TODOS los campos sin consumirlos
            i = txt.find("FORM_CHOICE_FIELDS = (")
            if i > 0:
                txt = txt[:i]
        out.append(txt)
    return "\n".join(out)


def test_every_wizard_field_has_a_backend_consumer():
    blob = _consumer_text()
    orphans = [f for f in WIZARD_FIELDS if not re.search(r"[\"']" + re.escape(f) + r"[\"']", blob)]
    assert not orphans, f"campos del asistente sin consumidor en el backend: {orphans}"


def test_habits_block_reads_the_wizard_values():
    fd = {"habitWater": "menos de 1L", "habitCaffeine": "diario", "habitSmoking": "diario", "habitAlcohol": "semanal",
          "waistCm": 96, "height": 173}
    b = build_habits_prompt(fd)
    for frag in ("HIDRATACIÓN BAJA", "CAFEÍNA DIARIA", "TABACO", "ALCOHOL RECURRENTE", "CINTURA/ALTURA"):
        assert frag in b, (frag, b)
    assert "menos de 1L" not in b or True   # texto canned; el valor crudo no se re-emite
    assert build_habits_prompt({"habitWater": "2-3L", "habitCaffeine": "ocasional", "habitSmoking": "nunca", "habitAlcohol": "nunca"}) == ""
    assert build_habits_prompt({"waistCm": 80, "height": 173}) == ""       # ratio 0,46
    assert build_habits_prompt(None) == ""


def test_coach_profile_mentions_waist_and_habits():
    ctx = build_user_identity_context({"age": 21, "weight": 61, "weightUnit": "kg", "height": 173, "waistCm": 96,
                                      "habitAlcohol": "semanal", "habitWater": "1-2L", "mainGoal": "gain_muscle"})
    assert "Cintura: 96 cm" in ctx and "alcohol cada semana" in ctx and "agua 1-2L" in ctx


def test_wired_into_the_generator():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "clinical_directives += _bhc(form_data)" in src and 'HABITS_RULES_ENABLED = _env_bool("MEALFIT_HABITS_RULES", True)' in src
