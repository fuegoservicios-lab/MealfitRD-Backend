"""
Tests para [BB-034 / P1-COMPARATIVE-YESTERDAY-TEMPLATE · 2026-11-05]
Diferenciación semántica de 'ayer'/'anoche' como plantilla comparativa vs. fecha de consumo histórica.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import (
    es_plantilla_comparativa_ayer,
    _days_ago_named_by_user,
    execute_tools,
    _extraer_ultimo_texto_usuario,
)


def test_es_plantilla_comparativa_ayer_positives():
    """Expresiones donde 'ayer' o 'anoche' se usan como plantilla de receta/ingredientes a clonar/adaptar."""
    casos = [
        "desayune el mismo desayuno de ayer pero sin el atun",
        "almorce lo mismo de ayer",
        "comi lo mismo que ayer",
        "cené igual que ayer",
        "desayuné lo de ayer",
        "para la cena fue lo mismo de ayer",
        "el mismo almuerzo de ayer pero con aguacate",
        "comí igual a lo de ayer",
        "la misma cena de ayer",
        "desayune lo mismo de anoche",
        "comi igual que anoche",
        "fue lo de anoche",
        "el mismo plato de anoche",
    ]
    for c in casos:
        assert es_plantilla_comparativa_ayer(c) is True, f"Se esperaba True para '{c}'"


def test_es_plantilla_comparativa_ayer_negatives():
    """Expresiones de eventos históricos reales de ayer o anoche (deben dar False)."""
    casos = [
        "ayer me comi un chimi a las 11 pm",
        "eso fue ayer",
        "ayer en la noche cene pizza",
        "se me olvido registrar el almuerzo de ayer",
        "ayer cene tacos",
        "anoche comi hamburguesa",
        "eso fue anoche",
    ]
    for c in casos:
        assert es_plantilla_comparativa_ayer(c) is False, f"Se esperaba False para '{c}'"


def test_days_ago_named_by_user_ignores_comparative_yesterday():
    """_days_ago_named_by_user no asigna days_ago=1 si el usuario usó 'ayer' como plantilla comparativa."""
    msg = HumanMessage(content="desayune el mismo desayuno de ayer pero sin el atun")
    res = _days_ago_named_by_user([msg], hora_local=12.0)
    assert res is None, f"Esperado None (hoy), obtenido {res}"


def test_days_ago_named_by_user_preserves_genuine_yesterday():
    """_days_ago_named_by_user retorna 1 si el usuario realmente describe una comida de ayer."""
    msg = HumanMessage(content="ayer almorcé un buen sancocho")
    res = _days_ago_named_by_user([msg], hora_local=16.0)
    assert res == 1, f"Esperado 1 (ayer), obtenido {res}"


def test_execute_tools_forces_days_ago_zero_for_comparative_template():
    """Si el modelo mandó erróneamente days_ago=1 ante 'desayune el mismo desayuno de ayer...',

    execute_tools intercepta el tool_call y fuerza days_ago=0 (hoy, sin consumed_at_override)."""
    user_msg = HumanMessage(content="desayune el mismo desayuno de ayer pero sin el atun")
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_123",
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Tostadas integrales con queso cottage y huevos",
                "calories": 700,
                "protein": 58,
                "carbs": 50,
                "healthy_fats": 20,
                "meal_type": "desayuno",
                "days_ago": 1,  # El LLM mandó 1 por error
            }
        }]
    )

    state = {
        "messages": [user_msg, ai_msg],
        "user_id": "00000000-0000-0000-0000-000000000001",
        "session_id": "sess-bb034",
    }

    with patch("tools.db_log_consumed_meal", return_value="meal-bb034-1") as mock_db_log:
        out = execute_tools(state)

        # Verificar que el override determinista forzó days_ago=0 (consumed_at_override es None)
        call_kwargs = mock_db_log.call_args[1]
        assert call_kwargs["consumed_at_override"] is None, (
            f"Esperado consumed_at_override=None (hoy), pero se recibió {call_kwargs['consumed_at_override']}"
        )

        # Verificar que la respuesta de la tool NO contiene aviso de 'fecha de AYER'
        tmsg = out["messages"][0]
        assert "con fecha de AYER" not in str(tmsg.content)


def test_execute_tools_preserves_days_ago_1_for_genuine_yesterday():
    """Si el usuario realmente dice 'ayer cené sancocho', execute_tools mantiene days_ago=1 (con consumed_at_override)."""
    user_msg = HumanMessage(content="ayer cené un sancocho con arroz")
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_456",
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Sancocho con arroz",
                "calories": 650,
                "protein": 40,
                "carbs": 60,
                "healthy_fats": 20,
                "meal_type": "cena",
                "days_ago": 1,
            }
        }]
    )

    state = {
        "messages": [user_msg, ai_msg],
        "user_id": "00000000-0000-0000-0000-000000000001",
        "session_id": "sess-bb034",
    }

    with patch("tools.db_log_consumed_meal", return_value="meal-bb034-2") as mock_db_log:
        out = execute_tools(state)

        call_kwargs = mock_db_log.call_args[1]
        assert call_kwargs["consumed_at_override"] is not None, (
            "Esperado consumed_at_override con timestamp de ayer, pero se recibió None"
        )

        tmsg = out["messages"][0]
        assert "con fecha de AYER" in str(tmsg.content)
