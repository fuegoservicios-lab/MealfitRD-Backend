# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-211 · 2026-09-24] Proteína en obesidad sobre peso ajustado aunque no se dé el % de grasa; en diabetes,
lo que libera el techo va a grasa. Ver `proteina_obesidad.py`."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import proteina_obesidad as po  # noqa: E402

_DM2 = {"weight": 88, "weightUnit": "kg", "height": 160, "age": 52, "gender": "female", "activityLevel": "moderate",
        "mainGoal": "lose_fat", "medicalConditions": ["Diabetes tipo 2"]}


def test_deurenberg():
    assert po.estimar_grasa(88, 160, 52, False) == round(1.2 * 88 / 1.6 ** 2 + 0.23 * 52 - 5.4, 1)
    assert po.estimar_grasa(88, 160, 10, False) is None, "fuera de rango adulto: no se estima"
    assert po.estimar_grasa(None, 160, 40, True) is None


def test_solo_con_imc_de_obesidad_y_sin_grasa_declarada():
    assert po.grasa_para_techo(None, 88, _DM2) > 30
    assert po.grasa_para_techo(25.0, 88, _DM2) == 25.0, "la del usuario manda"
    assert po.grasa_para_techo(None, 70, {"height": 175, "age": 30, "gender": "male"}) is None, "IMC 22.9"
    assert po.grasa_para_techo(None, 88, {"age": 52}) is None, "sin altura no se inventa"


def test_destino_de_lo_liberado():
    assert po.destino_liberado(_DM2) == "fats"
    assert po.destino_liberado({"medicalConditions": ["Hipertensión"]}) == "carbs"
    assert po.destino_liberado({"otherConditions": "prediabetes"}) == "fats"


def test_el_caso_de_la_bateria_rd20():
    """Mujer 88 kg / 160 cm / 52 años con DM2 y pérdida de grasa: la proteína baja a ~2,2 g/kg de peso ajustado y las
    calorías liberadas van a grasa, no a carbohidratos."""
    import os
    from nutrition_calculator import get_nutrition_targets
    nuevo = get_nutrition_targets(dict(_DM2))["macros"]
    os.environ["MEALFIT_PROTEIN_OBESITY_BMI_ESTIMATE"] = "false"
    os.environ["MEALFIT_DM2_FREED_KCAL_TO_FAT"] = "false"
    try:
        antes = get_nutrition_targets(dict(_DM2))["macros"]
    finally:
        del os.environ["MEALFIT_PROTEIN_OBESITY_BMI_ESTIMATE"]
        del os.environ["MEALFIT_DM2_FREED_KCAL_TO_FAT"]
    assert antes["protein_g"] >= 150 and nuevo["protein_g"] <= 130, (antes, nuevo)
    assert nuevo["carbs_g"] == antes["carbs_g"], "en diabetes lo liberado NO sube los carbohidratos"
    assert nuevo["fats_g"] > antes["fats_g"]


def test_calculate_macros_conserva_su_contrato():
    from nutrition_calculator import calculate_macros
    assert calculate_macros(2050, "gain_muscle", weight_kg=54) == calculate_macros(2050, "gain_muscle", weight_kg=54,
                                                                                  freed_to="carbs")
    a = calculate_macros(3000, "lose_fat", weight_kg=60, freed_to="fats")
    b = calculate_macros(3000, "lose_fat", weight_kg=60)
    assert a["protein_g"] == b["protein_g"] and a["carbs_g"] < b["carbs_g"] and a["fats_g"] > b["fats_g"]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 211 and m.group(2) >= "2026-09-24"
