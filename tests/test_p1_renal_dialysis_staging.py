"""[P1-RENAL-DIALYSIS-STAGING + P1-RENAL-K-CEILING + P1-GASTRITIS-RULE · 2026-06-26] (auditoría gap #8)
Profundiza el enforcement clínico más allá de alérgenos + cap-renal-fijo:

1. Cap de proteína renal ESTADIFICADO por diálisis (1.2 g/kg HD/PD vs 0.8 no-diálisis) — capear a un
   dializado a 0.8 es iatrogénico (malnutrición proteico-energética).
2. Techo OBSERVABLE de potasio en ERC (hiperkalemia = riesgo AGUDO de arritmia) — el panel marca 'alto'.
3. Gastritis/reflujo: el chip del form ahora tiene ConditionRule (antes solo FS9 genérico advisory).
"""
from __future__ import annotations

import graph_orchestrator as go
import condition_rules as cr
from nutrition_db import IngredientNutritionDB
from micronutrients import build_micronutrient_report


# ── 1. Dialysis staging del cap de proteína ──────────────────────────────────────────────
def test_dialysis_raises_protein_ceiling():
    assert go._renal_protein_gkg_for_profile({"medicalConditions": ["Diálisis por ERC"]}) == go.RENAL_DIALYSIS_PROTEIN_GKG
    assert go.RENAL_DIALYSIS_PROTEIN_GKG >= 1.0, "diálisis debe permitir >=1.0 g/kg (KDIGO HD/PD)"


def test_non_dialysis_renal_keeps_conservative_cap():
    assert go._renal_protein_gkg_for_profile({"medicalConditions": ["Enfermedad renal crónica"]}) == go.RENAL_PROTEIN_GKG_CEILING
    assert go.RENAL_PROTEIN_GKG_CEILING == 0.8


def test_non_renal_profile_uses_goal_ceiling_not_renal():
    # Sin condición renal el helper retorna el cap renal solo si se le llama; el call site lo gatea con
    # _is_renal_condition. Aquí confirmamos que un perfil no-renal/no-diálisis cae al 0.8 conservador (fail-safe).
    assert go._renal_protein_gkg_for_profile({"medicalConditions": ["Hipertensión"]}) == 0.8


# ── 2. Techo de potasio observable en ERC ────────────────────────────────────────────────
def test_renal_potassium_becomes_observable_ceiling():
    db = IngredientNutritionDB(rows=[{
        "name": "Guineo", "aliases": ["guineo"],
        "kcal_per_100g": 89, "protein_g_per_100g": 1.1, "carbs_g_per_100g": 23, "fats_g_per_100g": 0.3,
        "potassium_mg_per_100g": 358,
    }])
    # 1000 g de guineo → ~3580 mg K/día, sobre el techo 3000.
    plan = {"days": [{"meals": [{"ingredients": ["1000g de guineo (1000g)"]}]}]}
    rep = build_micronutrient_report(plan, db, sex="male", age=40, conditions=["enfermedad renal cronica"])
    pot = next((e for e in rep["panel"] if e["key"] == "potassium_mg"), None)
    assert pot is not None, "potasio debe estar en el panel"
    assert "techo" in pot, "para ERC el potasio debe modelarse como TECHO (no piso)"
    assert pot["status"] == "alto", f"~3580mg K debe exceder el techo 3000 → 'alto' (status={pot.get('status')})"


def test_non_renal_potassium_stays_floor():
    db = IngredientNutritionDB(rows=[{
        "name": "Guineo", "aliases": ["guineo"],
        "kcal_per_100g": 89, "protein_g_per_100g": 1.1, "carbs_g_per_100g": 23, "fats_g_per_100g": 0.3,
        "potassium_mg_per_100g": 358,
    }])
    plan = {"days": [{"meals": [{"ingredients": ["100g de guineo (100g)"]}]}]}
    rep = build_micronutrient_report(plan, db, sex="male", age=40, conditions=[])
    pot = next((e for e in rep["panel"] if e["key"] == "potassium_mg"), None)
    assert pot is not None and "piso" in pot, "sin ERC el potasio sigue siendo un PISO (DRI)"


# ── 3. ConditionRule de gastritis ────────────────────────────────────────────────────────
def test_gastritis_rule_active_and_in_prompt():
    fd = {"medicalConditions": ["Gastritis"]}
    rules = cr.detect_active_rules(fd)
    assert any(r.id == "gastritis" for r in rules), "el chip 'Gastritis' debe activar su ConditionRule"
    prompt = cr.build_condition_prompt(fd).upper()
    assert "GASTRITIS" in prompt or "REFLUJO" in prompt, "el prompt debe incluir la guía de gastritis"


def test_gastritis_variants_detected():
    for term in ("reflujo", "ERGE", "acidez estomacal", "úlcera péptica"):
        rules = cr.detect_active_rules({"medicalConditions": [term]})
        assert any(r.id == "gastritis" for r in rules), f"variante no detectada: {term}"
