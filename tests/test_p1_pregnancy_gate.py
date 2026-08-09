"""[P1-PREGNANCY-DEFICIT-GATE + P1-CONDITION-COVERAGE · 2026-06-14] Cobertura de condiciones
comunes que faltaban del modelado clínico (audit P1-de-precisión 2026-06-14).

Dos piezas:
  A. SEGURIDAD fail-hard: una persona embarazada/lactante NUNCA recibe un déficit calórico
     (`nutrition_calculator`). Antes el goal genérico `lose_fat` aplicaba un déficit del 20% sin
     ninguna salvaguarda → potencialmente dañino. El gate fuerza ≥ mantenimiento + flag.
  B. ADVISORY: filas `ConditionRule` (prompt_block + derivación FS9) para embarazo, hipotiroidismo,
     gota, hígado graso y SOP. La guía es estándar-de-cuidado general; la regla fina la valida el
     profesional (NO enforcement determinista sin revisión humana — cautela del audit).

Cubre:
  A. `_is_pregnancy_or_lactation` (medicalConditions + campo dedicado + negativos).
  B. `get_nutrition_targets`: embarazo + lose_fat → sin déficit (≥ ~TDEE) + flag; no-embarazo
     conserva el déficit (regresión: el gate no afecta usuarios normales).
  C. `condition_rules`: las 5 condiciones nuevas se detectan + emiten prompt_block.
  D. Parser-based anchors.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import nutrition_calculator as nc
import condition_rules as cr

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_NC_PATH = _BACKEND_ROOT / "nutrition_calculator.py"
_CR_PATH = _BACKEND_ROOT / "condition_rules.py"

_BASE_FORM = {"weight": 70, "weightUnit": "kg", "height": 165, "age": 30,
              "gender": "female", "activityLevel": "moderate"}


def _form(**over):
    f = dict(_BASE_FORM)
    f.update(over)
    return f


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A. Detección de embarazo/lactancia
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("conds", [
    ["embarazo"], ["Embarazada"], ["gestante"], ["lactancia"],
    ["estoy amamantando"], ["Pregnancy"], ["postparto"],
])
def test_pregnancy_detected_from_conditions(conds):
    assert nc._is_pregnancy_or_lactation(_form(medicalConditions=conds)) is True


def test_pregnancy_detected_from_dedicated_field():
    assert nc._is_pregnancy_or_lactation(_form(isPregnant=True)) is True
    assert nc._is_pregnancy_or_lactation(_form(lactating="true")) is True


@pytest.mark.parametrize("conds", [None, [], ["Ninguna"], ["diabetes"], ["hipertension"]])
def test_non_pregnant_not_detected(conds):
    assert nc._is_pregnancy_or_lactation(_form(medicalConditions=conds)) is False


# ════════════════════════════════════════════════════════════════════════════════════════════════
# B. El gate fuerza ≥ mantenimiento (sin déficit) para embarazo/lactancia
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_pregnant_lose_fat_gets_no_deficit():
    res = nc.get_nutrition_targets(_form(mainGoal="lose_fat", medicalConditions=["embarazo"]))
    tdee = res["tdee"]
    # Sin déficit: target ≈ TDEE (mantenimiento), muy por encima del déficit del 20% (tdee*0.8).
    assert res["target_calories"] >= tdee * 0.95, (res["target_calories"], tdee)
    safety = res.get("pregnancy_lactation_safety")
    assert safety and safety["applied"] is True
    assert safety["original_goal"] == "lose_fat"
    assert safety["effective_goal"] == "maintenance"


def test_lactation_lose_fat_gets_no_deficit():
    res = nc.get_nutrition_targets(_form(mainGoal="lose_fat", isLactating=True))
    assert res["target_calories"] >= res["tdee"] * 0.95
    assert res.get("pregnancy_lactation_safety", {}).get("applied") is True


def test_non_pregnant_lose_fat_keeps_deficit():
    """Regresión: el gate NO afecta a un usuario normal — el déficit del 20% sigue aplicándose."""
    res = nc.get_nutrition_targets(_form(mainGoal="lose_fat", medicalConditions=["Ninguna"]))
    assert res["target_calories"] < res["tdee"], (res["target_calories"], res["tdee"])
    assert "pregnancy_lactation_safety" not in res


def test_pregnant_gain_muscle_not_flagged_as_deficit_override():
    """gain_muscle (superávit) no es un déficit → no se reescribe la meta, pero NO debe bajar de TDEE."""
    res = nc.get_nutrition_targets(_form(mainGoal="gain_muscle", medicalConditions=["embarazada"]))
    assert res["target_calories"] >= res["tdee"]


# ════════════════════════════════════════════════════════════════════════════════════════════════
# C. Las condiciones nuevas se detectan + emiten prompt_block (advisory)
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("cond,rule_id", [
    ("embarazo", "pregnancy"),
    ("hipotiroidismo", "hypothyroid"),
    ("gota", "gout"),
    ("higado graso", "nafld"),
    ("SOP", "pcos"),
])
def test_new_condition_active_and_has_prompt(cond, rule_id):
    active_ids = {r.id for r in cr.detect_active_rules(_form(medicalConditions=[cond]))}
    assert rule_id in active_ids, (cond, active_ids)
    prompt = cr.build_condition_prompt(_form(medicalConditions=[cond]))
    assert prompt and "REGLA CLÍNICA" in prompt


def test_new_conditions_are_referral_not_substitution():
    """Las 5 nuevas son advisory/derivación; 4 sin sustituciones deterministas.

    [reapuntado 2026-07-27] La cautela original del audit ("cero sustituciones") fue superada
    DELIBERADAMENTE por el gate embarazo/lactancia (commit e07bdea, P1-3): pregnancy ganó UNA
    sustitución — pescado alto en mercurio (tiburón/pez espada/marlin/king mackerel) → pescado
    blanco. Es contraindicación de libro en embarazo; sustituir es MÁS seguro que solo derivar.
    [reapuntado 2026-08-09 · P1-PREGNANCY-SAFETY-NOTES] 2ª sustitución deliberada: cundeamor
    (melón amargo, Momordica charantia) → Tayota. Justificación clínica del reapunte: el propio
    reviewer clínico lo rechazó CRITICAL en un plan de embarazo REAL (corr=d395f5c8, 2026-08-08:
    «contraindicado durante todo el embarazo por riesgo abortivo y uterotónico») y es
    contraindicación estándar en obstetricia — mismo criterio que el mercurio: contraindicación
    a nivel de ALIMENTO que el motor puede remover deterministamente. La clasificación sigue
    siendo referral y las otras 4 condiciones permanecen sin sustituciones. Si pregnancy gana
    una 3ª sustitución o alguna de las otras 4 gana la primera, este test debe caer y forzar
    la conversación clínica.
    """
    for rid in ("hypothyroid", "gout", "nafld", "pcos"):
        rule = cr._RULES_BY_ID[rid]
        assert rule.substitutions == (), rid
        assert rule.classification == cr.CLINICAL_REFERRAL, rid
    preg = cr._RULES_BY_ID["pregnancy"]
    assert preg.classification == cr.CLINICAL_REFERRAL
    assert len(preg.substitutions) == 2, (
        "pregnancy debe tener EXACTAMENTE las 2 sustituciones documentadas (mercurio e07bdea + "
        "cundeamor uterotónico P1-PREGNANCY-SAFETY-NOTES); una 3ª requiere revisión clínica "
        "explícita"
    )
    _triggers = " ".join(preg.substitutions[0][0])
    assert "tiburon" in _triggers and "pez espada" in _triggers, preg.substitutions[0][0]
    assert "pescado blanco" in str(preg.substitutions[0][1]).lower()
    _ut_triggers = " ".join(preg.substitutions[1][0])
    assert "cundeamor" in _ut_triggers, preg.substitutions[1][0]
    assert str(preg.substitutions[1][1]).lower() == "tayota"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D. Parser-based anchors
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_markers_present():
    assert "P1-PREGNANCY-DEFICIT-GATE" in _NC_PATH.read_text(encoding="utf-8")
    assert "P1-CONDITION-COVERAGE" in _CR_PATH.read_text(encoding="utf-8")
