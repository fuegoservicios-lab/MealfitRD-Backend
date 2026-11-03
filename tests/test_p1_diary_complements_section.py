"""
Tests para [BB-025 / P1-DIARY-COMPLEMENTS-SECTION · 2026-11-03]
Separación de bebidas ligeras y complementos del horario principal de comidas.
"""

import pytest
from unittest.mock import patch, MagicMock
from tools import _normalize_meal_type, log_consumed_meal, _MEAL_TYPES_CANON
from prompts.chat_agent import _CHAT_VOICE_RULES


def test_complemento_canonical_type_exists():
    assert "complemento" in _MEAL_TYPES_CANON
    assert _normalize_meal_type("complemento") == "complemento"
    assert _normalize_meal_type("infusion") == "complemento"
    assert _normalize_meal_type("te") == "complemento"
    assert _normalize_meal_type("drink") == "complemento"


def test_te_ligero_reclasificado_a_complemento_no_desayuno():
    user_id = "test-uid-bb025"
    with patch("tools.db_log_consumed_meal", return_value="meal-uuid-te") as mock_db, \
         patch("proactive_agent.mark_nudge_logged_from_diary", return_value=None):

        # Usuario registra una taza de té de 2 kcal a las 8 AM con meal_type='desayuno' por horario
        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Taza de té o infusión herbal sin azúcar",
            "calories": 2,
            "protein": 0,
            "carbs": 0,
            "healthy_fats": 0,
            "meal_type": "desayuno"
        })
        assert "REGISTRADO" in res or "registrada" in res.lower()
        mock_db.assert_called_once()
        _, kwargs = mock_db.call_args
        assert kwargs.get("meal_type") == "complemento", f"Esperado 'complemento', obtenido {kwargs.get('meal_type')}"


def test_desayuno_real_con_huevos_mantiene_desayuno():
    user_id = "test-uid-bb025"
    with patch("tools.db_log_consumed_meal", return_value="meal-uuid-huevos") as mock_db, \
         patch("proactive_agent.mark_nudge_logged_from_diary", return_value=None):

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Huevos revueltos con tostadas",
            "calories": 420,
            "protein": 25,
            "carbs": 30,
            "healthy_fats": 15,
            "meal_type": "desayuno"
        })
        mock_db.assert_called_once()
        _, kwargs = mock_db.call_args
        assert kwargs.get("meal_type") == "desayuno"
        mock_db.assert_called_once()
        _, kwargs = mock_db.call_args
        assert kwargs.get("meal_type") == "desayuno"


def test_prompt_contiene_directiva_de_complementos():
    assert "P1-DIARY-COMPLEMENTS-SECTION" in _CHAT_VOICE_RULES
    assert "complemento" in _CHAT_VOICE_RULES
