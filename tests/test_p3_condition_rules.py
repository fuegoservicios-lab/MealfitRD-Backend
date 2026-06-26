"""[P3-CONDITION-RULES · 2026-06-14] Reglas clínicas deterministas por condición del set Pareto DR.

Cubre las DOS condiciones que el piso regulatorio más estricto (Medicare MNT) reembolsa:
  • DM2  → ADA 2025/2026: calidad del carbohidrato (fibra ≥14 g/1000 kcal), NO %carbos/IG.
  • ERC  → KDIGO 2024: cap de SEGURIDAD de proteína ~0.8 g/kg + gate de derivación profesional.

Más M1 (data provenance) anclas parser-based. Si alguien renombra un marker/función de prod, el
test falla ANTES de que el cambio llegue a producción (convención del repo).
"""
import os
import re
import inspect

import pytest

import graph_orchestrator as go
import micronutrients as mn
from prompts import plan_generator as pg


# ----------------------------------------------------------------------------
# 1. DETECCIÓN DE CONDICIONES (graph_orchestrator)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("conds,expect_renal", [
    (["Enfermedad renal crónica"], True),
    (["ERC G3"], True),
    (["Insuficiencia renal"], True),
    (["problema en el riñón"], True),     # acento normalizado
    (["Diálisis"], True),
    (["Ninguna"], False),
    (["Diabetes tipo 2"], False),
    (["Hipertensión"], False),
    ([], False),
])
def test_renal_detection(conds, expect_renal):
    assert go._is_renal_condition({"medicalConditions": conds}) is expect_renal


@pytest.mark.parametrize("conds,expect_dm2", [
    (["Diabetes tipo 2"], True),
    (["DM2"], True),
    (["prediabetes"], True),
    (["resistencia a la insulina"], True),
    (["Ninguna"], False),
    (["Enfermedad renal"], False),
    ([], False),
])
def test_diabetes_detection(conds, expect_dm2):
    assert go._is_diabetes_condition({"medicalConditions": conds}) is expect_dm2


def test_condition_strings_filters_sentinel():
    assert go._condition_strings({"medicalConditions": ["Ninguna"]}) == []
    out = go._condition_strings({"medicalConditions": ["Diabetes T2", "Ninguna"]})
    assert any("diabet" in c for c in out)


# ----------------------------------------------------------------------------
# 2. ERC — techo de proteína renal-aware (KDIGO 0.8 g/kg, NO el goal-aware alto)
# ----------------------------------------------------------------------------
def test_goal_aware_trim_ceiling_renal_caps_at_08gkg():
    # Target = nivel renal (0.8 × 70 = 56g): renal → ceiling_pct ~1.0 (trima AL target renal).
    renal = go._goal_aware_trim_ceiling_pct(
        {"medicalConditions": ["ERC"], "mainGoal": "gain_muscle",
         "weight": 70, "weightUnit": "kg"}, 56.0)
    assert renal == pytest.approx(1.0, abs=0.01)
    # Mismo target con perfil NO-renal de volumen → techo mucho más alto (clamp 1.30).
    non_renal = go._goal_aware_trim_ceiling_pct(
        {"medicalConditions": ["Ninguna"], "mainGoal": "gain_muscle",
         "weight": 70, "weightUnit": "kg"}, 56.0)
    assert non_renal > renal
    assert non_renal == pytest.approx(1.30, abs=0.01)


def test_goal_aware_trim_ceiling_renal_no_weight_returns_strict():
    # Sin peso parseable, ERC retorna techo estricto 1.0 (no el fallback laxo por objetivo).
    assert go._goal_aware_trim_ceiling_pct(
        {"medicalConditions": ["enfermedad renal"], "mainGoal": "lose_fat"}, 0.0) == 1.0


def test_renal_protein_cap_knob_default():
    assert go.RENAL_PROTEIN_GKG_CEILING == pytest.approx(0.8)
    assert go.CONDITION_RULES_ENABLED is True


# ----------------------------------------------------------------------------
# 3. DM2 — piso de fibra ADA 2026 (≥14 g/1000 kcal) en el reporte de micros
# ----------------------------------------------------------------------------
class _StubDB:
    """DB stub: no resuelve micros (fuerza coverage 0) — basta para probar el piso de fibra."""
    def micros_from_ingredient_string(self, s):
        return {}


_PLAN = {"days": [{"meals": [{"ingredients": ["avena 40g", "habichuelas 100g"]}]}]}


def test_dm2_raises_fiber_floor():
    # Mujer + 2000 kcal + DM2 → piso de fibra = max(25 DRI, 14×2) = 28 g.
    rep = mn.build_micronutrient_report(
        _PLAN, _StubDB(), sex="female",
        conditions=["diabetes tipo 2"], daily_kcal=2000.0)
    fiber = next(p for p in rep["panel"] if p["key"] == "fiber_g")
    assert fiber["piso"] == pytest.approx(28.0)
    assert rep["condition_targets"], "DM2 debe poblar condition_targets"
    ct = rep["condition_targets"][0]
    assert "ADA" in ct["guia"]


def test_no_condition_keeps_dri_fiber_floor():
    rep = mn.build_micronutrient_report(
        _PLAN, _StubDB(), sex="female", conditions=[], daily_kcal=2000.0)
    fiber = next(p for p in rep["panel"] if p["key"] == "fiber_g")
    assert fiber["piso"] == pytest.approx(25.0)
    assert rep["condition_targets"] == []


def test_dm2_male_floor_is_max_of_dri_and_ada():
    # Hombre DRI fibra = 38 > 28 (ADA 2000kcal) → se mantiene 38 (max, no baja).
    rep = mn.build_micronutrient_report(
        _PLAN, _StubDB(), sex="male",
        conditions=["dm2"], daily_kcal=2000.0)
    fiber = next(p for p in rep["panel"] if p["key"] == "fiber_g")
    assert fiber["piso"] == pytest.approx(38.0)


def test_build_micronutrient_report_backward_compatible():
    # Sin los nuevos kwargs sigue funcionando (firma con defaults).
    rep = mn.build_micronutrient_report(_PLAN, _StubDB(), sex="female")
    assert "panel" in rep and rep["condition_targets"] == []


# ----------------------------------------------------------------------------
# 4. Directiva de PROMPT por condición (plan_generator)
# ----------------------------------------------------------------------------
def test_prompt_context_dm2():
    txt = pg.build_medical_condition_context({"medicalConditions": ["Diabetes tipo 2"]})
    assert "ADA" in txt
    assert "FIBRA" in txt.upper()
    assert "bebidas azucaradas" in txt.lower()
    # ADA 2026: NO debe vender %carbos ni índice glucémico como feature.
    assert "índice glucémico" not in txt.lower() or "sin obsesionarse" in txt.lower()


def test_prompt_context_renal():
    txt = pg.build_medical_condition_context({"medicalConditions": ["Enfermedad renal crónica"]})
    assert "KDIGO" in txt
    assert "nefrólogo" in txt.lower()
    assert "moderada" in txt.lower()


def test_prompt_context_empty_for_no_condition():
    assert pg.build_medical_condition_context({"medicalConditions": ["Ninguna"]}) == ""
    assert pg.build_medical_condition_context({}) == ""


def test_stress_hint_no_longer_mentions_glycemic_index():
    # ADA-clean: la última referencia a "índice glucémico" (en hint de estrés) fue removida.
    src = inspect.getsource(pg.build_sleep_stress_context)
    assert "índice glucémico" not in src


# ----------------------------------------------------------------------------
# 5. Anclas parser-based en producción (rename → test falla antes que prod)
# ----------------------------------------------------------------------------
def test_source_anchors_present():
    src = inspect.getsource(go)
    for marker in (
        "P3-CONDITION-RULES",          # knobs + helpers
        "renal_protein_cap",           # cap de seguridad ERC en assemble
        "renal_gate",                  # FS9 reforzado
        "P3-DATA-PROVENANCE",          # M1 quick-win
        "data_provenance",             # campo del result
        "build_medical_condition_context",  # directiva de prompt importada
        "meals_enforced",              # review#1/#7/#8: enforcement per-comida solver-independiente
        "reassigned_to",               # review#5: reasignación a grasa si diabético
        "comorbid_diabetes",           # review#2: comorbilidad DM2+ERC
        "_renal_capped_plan",          # review#4: floor-gate renal-aware
        "fiber_per_1000kcal",          # review#6: knob de fibra cableado
    ):
        assert marker in src, f"marker de producción ausente: {marker}"


def test_data_provenance_knob_default():
    assert go.DATA_PROVENANCE_ENABLED is True


# ----------------------------------------------------------------------------
# 6. Fixes del review adversario (P3-CONDITION-RULES)
# ----------------------------------------------------------------------------
def test_detector_terms_are_ssot_shared():
    """review#3: el cap (graph) y la directiva (prompt) usan los MISMOS términos (constants SSOT)."""
    import constants
    assert go._RENAL_CONDITION_TERMS is constants.RENAL_CONDITION_TERMS
    assert go._DIABETES_CONDITION_TERMS is constants.DIABETES_CONDITION_TERMS


def test_prompt_detects_english_renal_terms():
    """review#3: 'dialysis'/'chronic kidney' (solo-inglés) deben disparar el bloque renal en el
    prompt — antes el cap los detectaba pero el prompt no (drift peligroso)."""
    for term in ("dialysis", "chronic kidney disease"):
        txt = pg.build_medical_condition_context({"medicalConditions": [term]})
        assert "KDIGO" in txt, f"prompt renal no disparó para '{term}'"
        # y el cap (graph) también los detecta → paridad
        assert go._is_renal_condition({"medicalConditions": [term]}) is True


def test_prompt_comorbid_dm2_renal_reconciliation():
    """review#2: DM2+ERC juntos → bloque de precedencia clínica (ERC manda sobre fibra DM2)."""
    txt = pg.build_medical_condition_context(
        {"medicalConditions": ["Diabetes tipo 2", "Enfermedad renal crónica"]})
    assert "PRECEDENCIA" in txt
    assert "MANDA" in txt
    # ambos bloques base presentes también
    assert "ADA" in txt and "KDIGO" in txt


def test_dm2_fiber_knob_flows_to_floor():
    """review#6: el coeficiente fibra/1000kcal es parametrizable (knob vivo, no hardcode)."""
    # 20 g/1000kcal × 2000 kcal = 40 g > DRI 25 → piso debe reflejar el parámetro, no 14.
    rep = mn.build_micronutrient_report(
        _PLAN, _StubDB(), sex="female",
        conditions=["dm2"], daily_kcal=2000.0, fiber_per_1000kcal=20.0)
    fiber = next(p for p in rep["panel"] if p["key"] == "fiber_g")
    assert fiber["piso"] == pytest.approx(40.0)
    # y el callsite de prod pasa el knob
    assert "fiber_per_1000kcal=DM2_FIBER_G_PER_1000KCAL" in inspect.getsource(go)


def test_floor_gate_skips_renal_capped_plan():
    """review#4: el validador de piso de proteína (que empuja proteína animal arriba) NO debe
    correr en un plan renal capeado. Ancla la guarda en el source de review_plan_node."""
    src = inspect.getsource(go.review_plan_node)
    assert "_renal_capped_plan" in src
    # [anchor actualizado · 2026-06-26] El skip renal del piso de proteína se movió DENTRO de
    # `_protein_floor_shortfall(plan, renal_capped=_renal_capped_plan)` — el helper exime los días
    # renal-capeados (memoria P1-RENAL-UPDATE-ENFORCE: "_protein_floor_shortfall, que exime renal").
    # La propiedad de seguridad se PRESERVA vía el parámetro; anclamos esa forma en vez del antiguo
    # inline `not _renal_capped_plan` (que el refactor renombró). Verificado: el flag renal fluye al
    # validador de piso, que NO empuja proteína animal en un plan renal capeado.
    assert "renal_capped=_renal_capped_plan" in src, \
        "el flag renal debe fluir al validador de piso de proteína (que exime los planes renal-capeados)"


def test_source_cap_renal_only_reassigns_to_carb():
    """review-live#1 (raíz): el cap renal vive EN LA FUENTE (nutrition) → fluye a generación/
    review/fallback. Renal-only: proteína a 0.8 g/kg, kcal liberadas a carbo."""
    nutr = {"macros": {"protein_g": 154, "carbs_g": 300, "fats_g": 80, "protein_str": "154g",
                       "carbs_str": "300g", "fats_str": "80g"},
            "total_daily_macros": {"protein_g": 154, "carbs_g": 300, "fats_g": 80,
                                   "protein_str": "154g", "carbs_str": "300g", "fats_str": "80g"}}
    go._apply_renal_cap_to_nutrition(nutr, {"weight": 70, "weightUnit": "kg",
                                            "medicalConditions": ["Enfermedad renal cronica"]})
    assert nutr["macros"]["protein_g"] == 56          # 0.8 × 70
    assert nutr["total_daily_macros"]["protein_g"] == 56
    assert nutr["macros"]["carbs_g"] > 300            # kcal liberadas → carbo
    cap = nutr["renal_protein_cap"]
    assert cap["applied"] and cap["reassigned_to"] == "carb" and cap["source"] == "nutrition_target"


def test_source_cap_renal_diabetic_reassigns_to_fat():
    """review#5 en la fuente: diabético-nefropatía → kcal liberadas a GRASA (no carbo)."""
    nutr = {"macros": {"protein_g": 154, "carbs_g": 300, "fats_g": 80, "protein_str": "154g",
                       "carbs_str": "300g", "fats_str": "80g"},
            "total_daily_macros": {"protein_g": 154, "carbs_g": 300, "fats_g": 80}}
    go._apply_renal_cap_to_nutrition(nutr, {"weight": 70, "weightUnit": "kg",
                                            "medicalConditions": ["ERC", "Diabetes tipo 2"]})
    assert nutr["macros"]["protein_g"] == 56
    assert nutr["macros"]["fats_g"] > 80              # kcal → grasa
    assert nutr["macros"]["carbs_g"] == 300           # carbo intacto (no sube glucemia)
    assert nutr["renal_protein_cap"]["reassigned_to"] == "fat"
    assert nutr["renal_protein_cap"]["comorbid_diabetes"] is True


def test_source_cap_noop_when_not_renal_or_no_weight():
    nutr = {"macros": {"protein_g": 154, "protein_str": "154g"}, "total_daily_macros": {}}
    go._apply_renal_cap_to_nutrition(nutr, {"weight": 70, "weightUnit": "kg",
                                            "medicalConditions": ["Diabetes tipo 2"]})
    assert "renal_protein_cap" not in nutr            # diabético-no-renal → no cap
    nutr2 = {"macros": {"protein_g": 154, "protein_str": "154g"}, "total_daily_macros": {}}
    go._apply_renal_cap_to_nutrition(nutr2, {"medicalConditions": ["ERC"]})  # sin peso
    assert "renal_protein_cap" not in nutr2


def test_source_cap_and_safety_net_anchors():
    src = inspect.getsource(go)
    for marker in ("_apply_renal_cap_to_nutrition", "nutrition_target",
                   "Red de seguridad renal", "RED DE SEGURIDAD RENAL",
                   "Gate profesional genérico"):  # gate FS9 en fallback para cualquier condición
        assert marker in src, f"marker ausente: {marker}"
    # el cap en la fuente se invoca tras get_nutrition_targets
    assert "_apply_renal_cap_to_nutrition(nutrition, actual_form_data)" in src


def test_dm2_glycemic_soft_reject_knob_default():
    assert go.DM2_GLYCEMIC_SOFT_REJECT is True


def test_dm2_glycemic_downgrade_is_safe_scoped():
    """task DM2-fallback: el revisor degrada el rechazo glucémico crítico a 'high' para diabéticos
    → entrega el plan real (con fibra ADA) en vez de fallback matemático. La degradación NUNCA
    toca allergen/schema criticals (siguen cayendo a fallback por seguridad). Ancla en source."""
    src = inspect.getsource(go.review_plan_node)
    # el downgrade existe y está acotado a diabetes
    assert "DM2_GLYCEMIC_SOFT_REJECT" in src
    assert "_is_diabetes_condition(form_data)" in src
    assert 'severity = "high"' in src
    # las salvaguardas: NO degradar allergen ni schema-invalid
    assert "_had_allergen_critical" in src
    assert '_schema_invalid' in src
    # el flag de allergen se setea en el guard determinista
    assert "_had_allergen_critical = True" in src


def test_dm2_sugar_guard_replaces_added_sugars():
    """mejora DM2: guard determinista sustituye azúcar añadida/bebidas azucaradas por stevia/agua
    (clínicamente mejor + deja de gatillar el rechazo del revisor). Azúcar natural (fruta) intacta."""
    plan = {"days": [{"meals": [
        {"name": "Desayuno", "ingredients": ["Avena 40g", "1 cda de miel", "Guineo 1 und"],
         "ingredients_raw": ["Avena 40g", "1 cda de miel"], "recipe": ["Mezclar"]},
        {"name": "Merienda", "ingredients": ["Yogur 150g", "2 cda de azúcar", "Refresco de cola"], "recipe": "Frío"},
        {"name": "Almuerzo", "ingredients": ["Pollo 150g", "Arroz integral 100g"], "recipe": ["Guisar"]},
    ]}]}
    n = go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes tipo 2"]})
    assert n == 2
    des = plan["days"][0]["meals"][0]
    assert "Stevia al gusto" in des["ingredients"]
    assert not any("miel" in str(i).lower() for i in des["ingredients"])
    assert "Guineo 1 und" in des["ingredients"]           # azúcar natural NO se toca
    assert any("Ajuste clínico" in str(s) for s in des["recipe"])
    mer = plan["days"][0]["meals"][1]
    assert "Agua" in mer["ingredients"] and not any("refresco" in str(i).lower() for i in mer["ingredients"])
    assert plan["days"][0]["meals"][2].get("_dm2_sugar_fixed") is None  # almuerzo sin azúcar añadida


def test_dm2_sugar_guard_noop_non_diabetic():
    plan = {"days": [{"meals": [{"ingredients": ["1 cda de miel"]}]}]}
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Ninguna"]}) == 0
    assert plan["days"][0]["meals"][0]["ingredients"] == ["1 cda de miel"]


def test_dm2_sugar_guard_idempotent_and_respects_sin_azucar():
    plan = {"days": [{"meals": [{"ingredients": ["Yogur sin azucar 150g", "1 cda de miel"]}]}]}
    go._apply_condition_substitutions(plan, {"medicalConditions": ["dm2"]})
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert "Yogur sin azucar 150g" in ings                # 'sin azúcar' NO gatilla
    assert "Stevia al gusto" in ings
    # segundo run = no-op (ya es stevia)
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["dm2"]}) == 0


def test_dm2_sugar_guard_knob_and_anchors():
    assert go.DM2_SUGAR_GUARD is True
    src = inspect.getsource(go)
    # [P3-FALLBACK-CLINICAL-LAYER Fase B] el guard de sustitución se invoca DENTRO de la capa clínica
    # determinista (SSOT), que assemble consume vía `_apply_deterministic_clinical_layer(result, ...)`.
    assert "_apply_condition_substitutions(plan, form_data)" in src        # call dentro de la capa SSOT
    assert "_apply_deterministic_clinical_layer(result, form_data, nutrition)" in src  # cableado en assemble


def test_renal_enforcement_machinery_trims_meals_to_cap():
    """review#1/#7/#8 (el crítico): la maquinaria que usa el enforcement determinista per-comida
    (independiente del solver) realmente baja la proteína de las comidas al cap renal y refilla
    las kcal con carbo. Smoke funcional con la DB real — el mismo par de funciones que invoca el
    bloque solver-independiente en assemble_plan_node."""
    from nutrition_db import IngredientNutritionDB
    db = IngredientNutritionDB()
    meals = [
        {"name": "Almuerzo", "cals": 600, "protein": 70, "carbs": 50, "fats": 15,
         "ingredients": ["250g de pechuga de pollo", "100g de arroz"]},
        {"name": "Cena", "cals": 550, "protein": 60, "carbs": 45, "fats": 14,
         "ingredients": ["220g de filete de pescado", "150g de batata"]},
    ]
    cap = 56.0  # 0.8 g/kg × 70 kg
    p_before = sum(go._meal_macro_num(m.get("protein")) for m in meals)
    assert p_before > cap * 1.5  # parte de un plan claramente alto en proteína (130g)
    trimmed = go._trim_day_protein_to_ceiling(meals, cap, db, ceiling_pct=1.0)
    go._protein_preserving_day_reconcile(meals, 1700.0, db)
    p_after = sum(go._meal_macro_num(m.get("protein")) for m in meals)
    assert trimmed is True
    assert p_after <= cap * 1.10, f"proteína per-comida {p_after}g no bajó al cap {cap}g"
    assert p_after < p_before
