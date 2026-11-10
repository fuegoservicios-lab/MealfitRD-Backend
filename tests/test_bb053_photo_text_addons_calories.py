# -*- coding: utf-8 -*-
"""[BB-053 · 2026-11-10] Tests para fusión calórica foto + texto y coherencia física Atwater (4P + 4C + 9F)."""
import pytest
from langchain_core.messages import HumanMessage
from agent import _sanitizar_ingredientes_contra_alucinacion_foto
from tools import _validar_macros_comida


def test_extra_foods_in_user_text_preserve_higher_calories():
    """Caso real Moisés: foto de 558 kcal (huevos + panecillos), pero el texto declara
    alimentos extra («mantequilla de mani y queso cottage»).
    Las calorías (768 kcal) y macros (52P, 59C, 36F) NO deben pisarse hacia las 558 kcal de la foto sola."""
    plate_photos = [{
        "desc": "4 huevos hervidos partidos por la mitad y 2 panecillos redondos en un plato",
        "kcal": 558.0,
        "attachment_id": "att_1",
    }]
    messages = [HumanMessage(content="mantequilla de mani y queso cottage")]
    tool_args = {
        "meal_name": "4 huevos hervidos y 2 panecillos con mantequilla de maní y queso cottage",
        "calories": 768,
        "protein": 52,
        "carbs": 59,
        "healthy_fats": 36,
        "ingredients": [
            "4 huevos hervidos enteros (≈200 g)",
            "2 panecillos redondos (≈100 g)",
            "mantequilla de maní (≈32 g)",
            "queso cottage (≈60 g)",
        ],
    }

    sanitized = _sanitizar_ingredientes_contra_alucinacion_foto(tool_args, plate_photos, messages)
    assert sanitized["calories"] == 768, "Debe preservar 768 kcal por los alimentos añadidos en el texto"
    assert sanitized["protein"] == 52
    assert sanitized["carbs"] == 59
    assert sanitized["healthy_fats"] == 36


def test_hallucinated_calories_without_text_addons_are_clamped_and_macros_scaled():
    """Si el usuario NO añadió alimentos en el texto ni porciones (ej. 'estuvo muy rico'),
    y el LLM infló arbitrariamente las calorías muy por encima de la foto (+20%),
    el guard de visión debe ajustar calorías a kf y escalar los macros proporcionalmente."""
    plate_photos = [{
        "desc": "pizza con queso mozzarella y jamón",
        "kcal": 600.0,
        "attachment_id": "att_pizza",
    }]
    messages = [HumanMessage(content="estuvo muy rico")]
    tool_args = {
        "meal_name": "pizza con queso mozzarella y jamón",
        "calories": 1200,  # 2x de la foto sin alimentos extra
        "protein": 60,
        "carbs": 120,
        "healthy_fats": 40,
    }

    sanitized = _sanitizar_ingredientes_contra_alucinacion_foto(tool_args, plate_photos, messages)
    assert sanitized["calories"] == 600, "Debe ajustar a 600 kcal al no haber alimentos extra en el texto"
    assert sanitized["protein"] == 30  # 60 * (600/1200)
    assert sanitized["carbs"] == 60     # 120 * (600/1200)
    assert sanitized["healthy_fats"] == 20  # 40 * (600/1200)


def test_portion_modifiers_in_text_preserve_custom_calories():
    """Si el usuario indica porción ('solo me comí la mitad'), las calorías bajas no se fuerzan a kf."""
    plate_photos = [{
        "desc": "plato de sancocho con arroz",
        "kcal": 800.0,
        "attachment_id": "att_sancocho",
    }]
    messages = [HumanMessage(content="solo me comí la mitad del plato")]
    tool_args = {
        "meal_name": "media porción de sancocho con arroz",
        "calories": 400,
        "protein": 25,
        "carbs": 45,
        "healthy_fats": 12,
    }

    sanitized = _sanitizar_ingredientes_contra_alucinacion_foto(tool_args, plate_photos, messages)
    assert sanitized["calories"] == 400, "Debe respetar la porción reducida declarada por el usuario"


def test_atwater_consistency_reconciles_underestimated_calories():
    """Si se pasa una comida con calorías físicamente incompatibles con sus propios macros
    (ej. 558 kcal cuando 52P + 59C + 36F da 768 kcal), _validar_macros_comida reconcilia
    las calorías al piso físico de Atwater."""
    out, err = _validar_macros_comida(
        calories=558,
        protein=52,
        carbs=59,
        healthy_fats=36,
    )
    assert err is None
    # 52*4 + 59*4 + 36*9 = 208 + 236 + 324 = 768 kcal
    assert out["calories"] == 768, "Debe elevar calorías a 768 kcal según Atwater"
    assert out["protein"] == 52
    assert out["carbs"] == 59
    assert out["healthy_fats"] == 36


def test_atwater_consistency_preserves_legitimate_meals_with_fiber():
    """Una comida con carbohidratos fibrosos o tolerancia normal (~10% desfase) no debe ser alterada."""
    # Habichuelas: 15P, 40C, 1F -> Atwater floor = 60 + 160 + 9 = 229 kcal.
    # Con fibra dietética se declaran 215 kcal (215 >= 229 * 0.88 = 201.52).
    out, err = _validar_macros_comida(
        calories=215,
        protein=15,
        carbs=40,
        healthy_fats=1,
    )
    assert err is None
    assert out["calories"] == 215, "No debe tocar calorías dentro de la tolerancia de fibra"


def test_atwater_consistency_zero_calories_preserved():
    """Un café solo o agua no debe afectarse."""
    out, err = _validar_macros_comida(
        calories=0,
        protein=0,
        carbs=0,
        healthy_fats=0,
    )
    assert err is None
    assert out["calories"] == 0
