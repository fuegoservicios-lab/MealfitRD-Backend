"""[P3-MICRONUTRIENTS · 2026-06-13] Panel clínico de micros del plan vs DRI/WHO (FS4).

Cubre: (1) DRI sex-aware (hierro 8 vs 18, fibra 38 vs 25, 'mujer' NO es masculino pese a la M),
(2) azúcares LIBRES solo de ingredientes añadidos (miel), no del azúcar intrínseco, (3) promedio
diario, (4) reporte: gaps con nota de suplemento, techo de sodio alto, cobertura parcial →
estimado_bajo, (5) NO es gate duro (siempre retorna reporte, nunca lanza).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import IngredientNutritionDB
from micronutrients import (
    dri_targets, compute_plan_micronutrient_totals, build_micronutrient_report,
)

_ROWS = [
    {"name": "Yogurt griego", "aliases": ["yogurt griego", "yogur griego"],
     "kcal_per_100g": 59, "protein_g_per_100g": 10, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4,
     "fiber_g_per_100g": 0, "sodium_mg_per_100g": 36, "calcium_mg_per_100g": 111,
     "vitamin_b12_mcg_per_100g": 0.7, "vitamin_d_mcg_per_100g": 0, "iron_mg_per_100g": 0.07,
     "sugars_g_per_100g": 3.2, "potassium_mg_per_100g": 141},
    {"name": "Espinaca", "aliases": ["espinaca", "espinacas"],
     "kcal_per_100g": 23, "protein_g_per_100g": 2.9, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4,
     "fiber_g_per_100g": 2.2, "sodium_mg_per_100g": 79, "calcium_mg_per_100g": 99, "iron_mg_per_100g": 2.7,
     "potassium_mg_per_100g": 558, "vitamin_d_mcg_per_100g": 0, "vitamin_b12_mcg_per_100g": 0,
     "sugars_g_per_100g": 0.42},
    {"name": "Miel", "aliases": ["miel"],
     "kcal_per_100g": 304, "protein_g_per_100g": 0.3, "carbs_g_per_100g": 82, "fats_g_per_100g": 0,
     "fiber_g_per_100g": 0.2, "sodium_mg_per_100g": 4, "sugars_g_per_100g": 82, "calcium_mg_per_100g": 6,
     "iron_mg_per_100g": 0.4, "potassium_mg_per_100g": 52, "vitamin_d_mcg_per_100g": 0,
     "vitamin_b12_mcg_per_100g": 0},
    {"name": "Sal", "aliases": ["sal"], "kcal_per_100g": 0, "protein_g_per_100g": 0,
     "carbs_g_per_100g": 0, "fats_g_per_100g": 0, "sodium_mg_per_100g": 38758},
]


def _db():
    return IngredientNutritionDB(rows=_ROWS)


# ---------- DRI sex-aware ----------

def test_dri_hierro_y_fibra_por_sexo():
    m = dri_targets("male")
    f = dri_targets("female")
    assert m["iron_mg"]["floor"] == 8.0 and f["iron_mg"]["floor"] == 18.0
    assert m["fiber_g"]["floor"] == 38.0 and f["fiber_g"]["floor"] == 25.0


def test_mujer_no_es_masculino_pese_a_la_M():
    # 'mujer' empieza con M pero es femenino → hierro 18.
    assert dri_targets("mujer")["iron_mg"]["floor"] == 18.0
    assert dri_targets("hombre")["iron_mg"]["floor"] == 8.0


def test_sodio_y_azucar_son_techos():
    t = dri_targets("female")
    assert t["sodium_mg"]["ceiling"] == 2000.0
    assert t["free_sugars_g"]["ceiling"] == 25.0


# ---------- Totales ----------

def test_azucar_libre_solo_de_anadidos():
    # Miel (añadida) cuenta como azúcar libre; el azúcar del yogur (intrínseco) NO.
    plan = {"days": [{"meals": [{"ingredients": [
        "30g de miel (30g)", "200g de yogurt griego (200g)"]}]}]}
    tot = compute_plan_micronutrient_totals(plan, _db())
    # miel 30g * 82/100 = 24.6g de azúcar libre; yogur (6.4g azúcar intrínseco) NO cuenta.
    assert abs(tot["daily"]["free_sugars_g"] - 24.6) < 0.5


def test_promedio_diario_divide_por_dias():
    plan = {"days": [
        {"meals": [{"ingredients": ["100g de espinaca (100g)"]}]},
        {"meals": [{"ingredients": ["100g de espinaca (100g)"]}]},
    ]}
    tot = compute_plan_micronutrient_totals(plan, _db())
    # 2 días iguales → el promedio diario = el aporte de 1 día (100g espinaca = 2.7mg hierro).
    assert abs(tot["daily"]["iron_mg"] - 2.7) < 0.2
    assert tot["num_days"] == 2


def test_cobertura_cuenta_resueltos():
    plan = {"days": [{"meals": [{"ingredients": [
        "100g de espinaca (100g)", "1 unidad de ingrediente_inexistente"]}]}]}
    tot = compute_plan_micronutrient_totals(plan, _db())
    assert tot["total_ings"] == 2 and tot["resolved_ings"] == 1
    assert tot["coverage"] == 0.5


# ---------- Reporte advisory ----------

def test_reporte_marca_vitd_baja_con_nota_suplemento():
    # Plan sin fuentes de vit D → gap con nota de suplemento.
    plan = {"days": [{"meals": [{"ingredients": ["200g de yogurt griego (200g)", "100g de espinaca (100g)"]}]}]}
    rep = build_micronutrient_report(plan, _db(), sex="female")
    vitd = next(e for e in rep["panel"] if e["key"] == "vit_d_mcg")
    assert vitd["status"] in ("bajo", "estimado_bajo")
    assert any(g["key"] == "vit_d_mcg" and "suplemento" in g.get("nota", "").lower() for g in rep["gaps"])


def test_reporte_sodio_alto_cuando_excede_techo():
    # 10g de sal ≈ 3876mg sodio > 2000 techo → 'alto'.
    plan = {"days": [{"meals": [{"ingredients": ["10g de sal (10g)"]}]}]}
    rep = build_micronutrient_report(plan, _db(), sex="female")
    sodio = next(e for e in rep["panel"] if e["key"] == "sodium_mg")
    assert sodio["status"] == "alto"
    assert sodio["valor"] > 2000


def test_reporte_cobertura_parcial_es_estimado_bajo():
    # Cobertura <60% → un piso incumplido es 'estimado_bajo' (incierto), no 'bajo'.
    plan = {"days": [{"meals": [{"ingredients": [
        "100g de espinaca (100g)", "1 inexistente_a", "1 inexistente_b"]}]}]}
    rep = build_micronutrient_report(plan, _db(), sex="female")
    assert rep["coverage"] < 0.6
    calcio = next(e for e in rep["panel"] if e["key"] == "calcium_mg")
    assert calcio["status"] == "estimado_bajo"


def test_reporte_nunca_lanza_y_tiene_disclaimer():
    rep = build_micronutrient_report({"days": []}, _db(), sex=None)
    assert "disclaimer" in rep and "panel" in rep
    assert rep["sex"] == "F"  # default conservador
