"""[P1-FOOD-DB-EXTENDED-MICROS · 2026-06-25] Panel exhaustivo: +8 micros (zinc, folato, vit A, vit C,
vit E, vit K, selenio, omega-3). DRI sex/embarazo-aware, cómputo desde el catálogo, panel + steering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import IngredientNutritionDB
from micronutrients import (
    dri_targets, build_micronutrient_report, build_micronutrient_targets_directive, _SUPPLEMENT_NOTE,
)

_NEW = ("zinc_mg", "folate_mcg", "vit_a_mcg", "vit_c_mg", "vit_e_mg", "vit_k_mcg", "selenium_mcg", "omega3_g")

# Espinaca con las 8 columnas nuevas pobladas (valores ~USDA).
_SPINACH = {
    "name": "Espinaca", "aliases": ["espinaca", "espinacas"],
    "kcal_per_100g": 23, "protein_g_per_100g": 2.9, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4,
    "zinc_mg_per_100g": 0.53, "folate_mcg_dfe_per_100g": 194, "vitamin_a_mcg_rae_per_100g": 469,
    "vitamin_c_mg_per_100g": 28, "vitamin_e_mg_per_100g": 2.03, "vitamin_k_mcg_per_100g": 483,
    "selenium_mcg_per_100g": 1.0, "omega3_ala_g_per_100g": 0.138,
}


def test_dri_targets_incluye_los_8_nuevos():
    t = dri_targets("male", 20)
    for k in _NEW:
        assert k in t and t[k].get("floor", 0) > 0, f"falta floor de {k}"


def test_dri_nuevos_sex_y_embarazo_aware():
    m = dri_targets("male", 30)
    f = dri_targets("female", 30)
    p = dri_targets("female", 30, pregnant=True)
    assert m["zinc_mg"]["floor"] == 11 and f["zinc_mg"]["floor"] == 8
    assert m["vit_c_mg"]["floor"] == 90 and f["vit_c_mg"]["floor"] == 75
    assert m["vit_a_mcg"]["floor"] == 900 and f["vit_a_mcg"]["floor"] == 700
    assert p["folate_mcg"]["floor"] == 600 and f["folate_mcg"]["floor"] == 400  # embarazo
    assert p["vit_a_mcg"]["floor"] == 770


def test_panel_computa_y_muestra_los_nuevos():
    db = IngredientNutritionDB(rows=[_SPINACH])
    plan = {"days": [{"meals": [{"ingredients": ["100g de espinaca (100g)"]}]}]}
    rep = build_micronutrient_report(plan, db, sex="male", age=20)
    keys = {e["key"] for e in rep["panel"]}
    for k in _NEW:
        assert k in keys, f"{k} no está en el panel"
    # 100g de espinaca → folato ~194 computado (no 0).
    fol = next(e for e in rep["panel"] if e["key"] == "folate_mcg")
    assert fol["valor"] > 100
    vk = next(e for e in rep["panel"] if e["key"] == "vit_k_mcg")
    assert vk["valor"] > 400  # espinaca es altísima en vit K


def test_steer_incluye_alcanzables_pero_no_los_informativos():
    d = build_micronutrient_targets_directive(sex="male", age=20)
    # Alcanzables con comida y sin conflicto → al steering.
    assert "Zinc" in d and "Folato" in d and "Vitamina C" in d
    assert "Vitamina E" in d and "Omega-3" in d  # [P1-MICRO-STEER-OMEGA3-VITE]
    # vit A/K/selenio quedan informativos en el medidor, NO en el steering (ya suelen alcanzarse;
    # vit K además choca con warfarina → no empujarla).
    assert "Selenio" not in d and "Vitamina K" not in d and "Vitamina A" not in d


def test_vit_k_nota_consciente_de_warfarina():
    nota = _SUPPLEMENT_NOTE["vit_k_mcg"].lower()
    assert "warfarina" in nota or "anticoagulante" in nota
    assert "consistente" in nota  # no empuja a aumentarla de golpe
