"""[P3-FALLBACK-CLINICAL-LAYER · 2026-06-14] El plan de fallback matemático
(`_get_extreme_fallback_plan` / `_repair_partial_plan`) BYPASSA assemble_plan_node y por tanto perdía
la capa clínica determinista (FS1 food-safety, sustitución de sodio/azúcar por condición, FS2 quantize,
cap renal per-comida, micros/variedad/proveniencia, gate FS9). `_apply_deterministic_clinical_layer`
extrae esa capa a SSOT reutilizable y se aplica al fallback en el punto único de salida.

Estos tests verifican: el fallback HEREDA el marker + reports + gate FS9 + sustituciones; idempotencia;
que NO se muta el `nutrition` del caller (copia de active_macros); y el cableado del call site.
Los guards que dependen de la DB (micros/proveniencia/enforcement renal) se prueban con un stub o en
la validación en vivo (el offline fail-safea sin DB).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import graph_orchestrator as go


class _StubDB:
    """DB sin datos (offline): lookups vacíos → los guards DB-dependientes no-opean limpio."""
    def __init__(self, *a, **k):
        pass

    def lookup(self, name):
        return None

    def macros_from_ingredient_string(self, s):
        return None

    def micros_from_ingredient_string(self, s):
        return None

    def grams_from_ingredient_string(self, s):
        return None


@pytest.fixture(autouse=True)
def _stub_db(monkeypatch):
    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", _StubDB)


def _fallback(form_data, nutrition, days=3):
    return go._get_extreme_fallback_plan(
        nutrition, form_data.get("mainGoal", "Salud General"), num_days=days)


def _nutrition(protein=150, carbs=200, fats=60, cals=2000, **extra):
    n = {
        "target_calories": cals, "total_daily_calories": cals,
        "goal_label": "Mantenimiento",
        "macros": {"protein_g": protein, "carbs_g": carbs, "fats_g": fats,
                   "protein_str": f"{protein}g", "carbs_str": f"{carbs}g", "fats_str": f"{fats}g"},
        "total_daily_macros": {"protein_g": protein, "carbs_g": carbs, "fats_g": fats,
                               "protein_str": f"{protein}g", "carbs_str": f"{carbs}g", "fats_str": f"{fats}g"},
    }
    n.update(extra)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# El fallback HEREDA la capa (marker + reports + gate)
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_gets_clinical_layer_marker_and_reports():
    form = {"medicalConditions": ["Ninguna"], "gender": "male"}
    plan = _fallback(form, _nutrition())
    assert not plan.get("_clinical_layer_applied")          # antes de la capa
    go._apply_deterministic_clinical_layer(plan, form, _nutrition())
    assert plan["_clinical_layer_applied"] is True
    assert "variety_report" in plan                          # FS5 advisory heredado
    assert "data_provenance" in plan                         # M1 heredado (resuelve 0 con stub, pero presente)
    assert plan.get("_is_fallback") is True                  # sigue marcado como fallback


def test_fallback_renal_profile_inherits_fs9_gate_and_cap():
    form = {"medicalConditions": ["Enfermedad renal crónica"], "gender": "male",
            "weight": 80, "weightUnit": "kg"}
    nutr = _nutrition(protein=150)
    plan = _fallback(form, nutr)
    go._apply_deterministic_clinical_layer(plan, form, nutr)
    rpr = plan.get("requires_professional_review")
    assert isinstance(rpr, dict) and rpr.get("flag") is True
    assert rpr.get("renal_gate") is True                     # FS9 gate nefrólogo heredado
    cap = plan.get("renal_protein_cap")
    assert isinstance(cap, dict) and cap.get("applied") is True
    assert cap["protein_g"] == round(go.RENAL_PROTEIN_GKG_CEILING * 80)  # 0.8 g/kg × 80kg


def test_fallback_hta_substitutes_seeded_embutido_with_quantity():
    """Un fallback con un embutido sembrado + HTA → se sustituye y conserva cantidad (subs-fix)."""
    form = {"medicalConditions": ["Hipertensión"], "gender": "female"}
    nutr = _nutrition()
    plan = _fallback(form, nutr)
    plan["days"][0]["meals"][0]["ingredients"] = ["100g de salami", "arroz blanco"]
    go._apply_deterministic_clinical_layer(plan, form, nutr)
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("salami" in str(i).lower() for i in ings)
    # el reemplazo del staple conserva la cantidad líder
    repl = [i for i in ings if "pollo" in str(i).lower()]
    assert repl and str(repl[0])[0].isdigit()


def test_fallback_dm2_substitutes_seeded_sugar():
    form = {"medicalConditions": ["Diabetes tipo 2"], "gender": "female"}
    nutr = _nutrition()
    plan = _fallback(form, nutr)
    plan["days"][0]["meals"][0]["ingredients"] = ["avena", "1 cda de miel"]
    go._apply_deterministic_clinical_layer(plan, form, nutr)
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("miel" in str(i).lower() for i in ings)


# ─────────────────────────────────────────────────────────────────────────────
# Seguridad: idempotencia + no mutar el nutrition del caller
# ─────────────────────────────────────────────────────────────────────────────
def test_idempotent_second_call_is_noop():
    form = {"medicalConditions": ["Hipertensión"], "gender": "male"}
    nutr = _nutrition()
    plan = _fallback(form, nutr)
    plan["days"][0]["meals"][0]["ingredients"] = ["100g de salami"]
    go._apply_deterministic_clinical_layer(plan, form, nutr)
    import copy
    snapshot = copy.deepcopy(plan)
    go._apply_deterministic_clinical_layer(plan, form, nutr)   # 2ª llamada
    assert plan == snapshot                                     # marker short-circuit → sin cambios


def test_does_not_mutate_caller_nutrition():
    """El cap renal re-derivado debe operar sobre una COPIA de active_macros, NUNCA sobre `nutrition`."""
    form = {"medicalConditions": ["Enfermedad renal crónica"], "gender": "male",
            "weight": 80, "weightUnit": "kg"}
    nutr = _nutrition(protein=150)
    import copy
    nutr_before = copy.deepcopy(nutr)
    plan = _fallback(form, nutr)
    go._apply_deterministic_clinical_layer(plan, form, nutr)
    assert nutr == nutr_before                                  # nutrition intacto (copia, no referencia)


def test_non_dict_plan_is_safe():
    assert go._apply_deterministic_clinical_layer(None, {}, {}) is None
    assert go._apply_deterministic_clinical_layer("x", {}, {}) == "x"


# ─────────────────────────────────────────────────────────────────────────────
# Cableado: el call site en _apply_final_defense_guardrails gatea sobre _is_fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_final_defense_call_site_gates_on_is_fallback():
    src = go._module_source() if hasattr(go, "_module_source") else open(go.__file__, encoding="utf-8").read()
    # el call site existe, gateado por knob + _is_fallback + marker
    assert "_apply_deterministic_clinical_layer(_fcl_plan, actual_form_data, nutrition)" in src
    assert "FALLBACK_CLINICAL_LAYER_ENABLED" in src
    # y se cablea también en el fallback de emergencia P1-5
    assert "_apply_deterministic_clinical_layer(plan_to_return, actual_form_data, nutrition)" in src


def test_knob_registered_and_default_true():
    assert go.FALLBACK_CLINICAL_LAYER_ENABLED is True
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_FALLBACK_CLINICAL_LAYER" in snap
