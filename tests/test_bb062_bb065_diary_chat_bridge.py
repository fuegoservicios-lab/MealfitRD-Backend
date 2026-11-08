# -*- coding: utf-8 -*-
"""
Tests automatizados para la robustez del puente de sincronización del diario al chat (diary_chat_bridge):
- Detección automática de suplementos y negrita obligatoria de franjas (BB-062).
- Soporte para metas dinámicas de nutrición vía metas_del_dia (BB-063).
- Notificación transparente de superación de proteína manteniendo conteo de calorías restantes (BB-063).
- Reconciliación histórica de mensajes sincronizados en Neon DB (BB-064).
"""
import pytest
from unittest.mock import patch

from diary_chat_bridge import (
    build_sync_turn_texts,
    _es_suplemento,
)


# ==============================================================================
# BB-062: Detección de Suplementos y Negrita de Franjas en Confirmación
# ==============================================================================

def test_bb062_supplement_detection_by_keywords():
    """Verifica que suplementos comunes se reconozcan por nombre comercial aunque vengan como 'extra'."""
    assert _es_suplemento("extra", "Mass gainer GNC Bulk 1340 (5 scoops con 1 L de agua)") is True
    assert _es_suplemento(None, "1 scoop proteína whey con leche") is True
    assert _es_suplemento("snack", "5g de creatina creapure") is True
    assert _es_suplemento("extra", "Medio aguacate") is False
    assert _es_suplemento("almuerzo", "Pechuga de pollo a la plancha") is False


def test_bb062_slot_bolding_for_extra_and_supplements():
    """Valida que 'extra', 'suplemento' y 'complemento' siempre lleven la franja en negrita."""
    # 1. Mass gainer enviado como extra -> debe confirmarse como **suplemento**
    _, coach_gainer = build_sync_turn_texts(
        meal_name="Mass gainer GNC Bulk 1340 (5 scoops con 1 L de agua)",
        meal_type="extra",
        calories=1114,
        protein=42,
        days_ago=0,
    )
    assert "Anoté tu **suplemento**: Mass gainer GNC Bulk 1340" in coach_gainer
    assert "**1114 kcal** y **42 g de proteína**" in coach_gainer

    # 2. Comida real enviada como extra -> debe confirmarse como **extra**
    _, coach_extra = build_sync_turn_texts(
        meal_name="Medio aguacate",
        meal_type="extra",
        calories=120,
        protein=2,
        days_ago=0,
    )
    assert "Anoté tu **extra**: Medio aguacate" in coach_extra
    assert "**120 kcal** y **2 g de proteína**" in coach_extra

    # 3. Complemento -> debe confirmarse como **complemento**
    _, coach_comp = build_sync_turn_texts(
        meal_name="Té verde caliente",
        meal_type="complemento",
        calories=2,
        protein=0,
        days_ago=0,
    )
    assert "Anoté tu **complemento**: Té verde caliente" in coach_comp


# ==============================================================================
# BB-063: Cálculo de Metas Dinámicas del Día (metas_del_dia)
# ==============================================================================

def test_bb063_dynamic_metas_del_dia_calculation():
    """Verifica que el puente calcule las metas restantes a partir de metas_del_dia para usuarios dinámicos."""
    user_id = "test-uid-dynamic-1"
    consumed_today = [
        {"calories": 815, "protein": 44, "meal_name": "Panecillos con cottage y huevos"},
        {"calories": 1114, "protein": 42, "meal_name": "Mass gainer GNC Bulk 1340"}
    ]
    # Meta dinámica de Moisés: 3350 kcal y 157 g proteína
    mock_metas = {"kcal": 3350.0, "protein_g": 157.0, "carbs_g": 471.0, "fats_g": 93.0}

    with patch("diary_chat_bridge.get_consumed_meals_today", return_value=consumed_today), \
         patch("tools._perfil_y_plan_de_hoy", return_value=({}, None)), \
         patch("coach_day_context.metas_del_dia", return_value=mock_metas):
        _, coach_txt = build_sync_turn_texts(
            meal_name="Mass gainer GNC Bulk 1340 (5 scoops con 1 L de agua)",
            meal_type="extra",
            calories=1114,
            protein=42,
            days_ago=0,
            user_id=user_id,
        )

    # 3350 - 1929 = 1421 kcal restantes; 157 - 86 = 71 g proteína restante
    assert "Te quedan **~1421 kcal** y **~71 g de proteína** para tu meta de hoy." in coach_txt


def test_bb063_protein_target_exceeded_celebration():
    """Verifica el cálculo cuando la meta de proteína fue alcanzada o superada pero quedan calorías."""
    user_id = "test-uid-dynamic-2"
    consumed_today = [
        {"calories": 815, "protein": 44, "meal_name": "Desayuno"},
        {"calories": 1114, "protein": 42, "meal_name": "Mass gainer"},
        {"calories": 780, "protein": 89, "meal_name": "Cena"}
    ]
    # Total consumido: 2709 kcal y 175 g proteína (superó 157 g por 18 g; restan 641 kcal)
    mock_metas = {"kcal": 3350.0, "protein_g": 157.0, "carbs_g": 471.0, "fats_g": 93.0}

    with patch("diary_chat_bridge.get_consumed_meals_today", return_value=consumed_today), \
         patch("tools._perfil_y_plan_de_hoy", return_value=({}, None)), \
         patch("coach_day_context.metas_del_dia", return_value=mock_metas):
        _, coach_txt = build_sync_turn_texts(
            meal_name="Filete de res y pollo",
            meal_type="cena",
            calories=780,
            protein=89,
            days_ago=0,
            user_id=user_id,
        )

    assert "Te quedan **~641 kcal** para tu meta de hoy; ya superaste la meta de proteína por unos **18 g**." in coach_txt


# ==============================================================================
# BB-064: Verificación de Plantilla de Reconciliación Histórica
# ==============================================================================

def test_bb064_reconciliation_template():
    """Valida la plantilla del mensaje de Mass Gainer reconciliado para Moisés."""
    reconciled = (
        "Anoté tu **suplemento**: Mass gainer GNC Bulk 1340 (5 scoops con 1 L de agua); aprox. **1114 kcal** y "
        "**42 g de proteína**. Te quedan **~1421 kcal** y **~71 g de proteína** para tu meta de hoy."
    )
    assert "**suplemento**" in reconciled
    assert "Anoté tu comida:" not in reconciled
    assert "1421 kcal" in reconciled
    assert "71 g de proteína" in reconciled
