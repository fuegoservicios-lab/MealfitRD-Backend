# -*- coding: utf-8 -*-
"""[P1-ANTI-CLONE-GUARD · 2026-10-29] Pruebas de protección anti-clonación de comidas en el diario.

Incidente del 29-oct:
El usuario subió una foto de su plato (Plátano maduro hervido con 2 huevos fritos y queso derretido)
a las 11:34 AM con el texto "Plátano maduro · 2 huevos".
El LLM lo anotó inicialmente como 'almuerzo'.
Al recibir el resultado de la herramienta indicando que el desayuno seguía sin registrarse,
el LLM llamó nuevamente a `log_consumed_meal` en el segundo loop del turno y clonó exactamente
el mismo plato como 'desayuno', duplicando las calorías y los macros en el diario del día.

Garantías probadas aquí:
1. `franja_por_hora` a las 11:34 AM con desayuno pendiente devuelve 'desayuno'.
2. `execute_tools` bloquea un segundo `log_consumed_meal` en el mismo turno cuando el usuario solo envió un plato/foto.
3. El dedup de `db_facts.log_consumed_meal` rechaza comidas con mismos macros e ingredientes en ventana corta aunque el nombre varíe.
4. `db_facts.log_consumed_meal` rechaza registrar un plato con un `attachment_id` que ya fue registrado hoy.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from coach_day_context import franja_por_hora
from agent import execute_tools


def test_franja_por_hora_late_breakfast_defaults_to_desayuno():
    # 11:34 AM = 11.566 h
    # Con desayuno pendiente, antes del mediodía debe ser desayuno
    assert franja_por_hora(11.566, faltan=["desayuno", "almuerzo", "cena"]) == "desayuno"
    assert franja_por_hora(11.0, faltan=["desayuno", "almuerzo"]) == "desayuno"
    # Si el desayuno ya está registrado, a las 11:34 es almuerzo
    assert franja_por_hora(11.566, faltan=["almuerzo", "cena"]) == "almuerzo"
    # A las 12:30 PM (mediodía), pasa a almuerzo aunque falte desayuno
    assert franja_por_hora(12.5, faltan=["desayuno", "almuerzo", "cena"]) == "almuerzo"


def test_execute_tools_blocks_cloned_meal_in_same_turn():
    # Simulamos el segundo loop del agente donde ya se creó una comida en este turno
    prev_human = HumanMessage(content="Plátano maduro · 2 huevos")
    prev_ai = AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "call-1"}])
    prev_tool = ToolMessage(
        content="¡Éxito! [ID_REGISTRO_DIARIO: meal-uuid-1] Registrado como almuerzo.",
        tool_call_id="call-1",
    )
    # El LLM intenta llamar a log_consumed_meal de nuevo en el mismo turno
    duplicate_ai = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Desayuno con plátano maduro hervido",
                "calories": 480,
                "protein": 21,
                "carbs": 50,
                "healthy_fats": 22,
                "meal_type": "desayuno",
            },
            "id": "call-2",
        }],
    )

    state = {
        "messages": [prev_human, prev_ai, prev_tool, duplicate_ai],
        "user_id": "test-user-uuid",
        "session_id": "test-session-uuid",
        "turn_plate_photos": [{"desc": "Plátano con huevos", "attachment_id": "att-1"}],
    }

    result = execute_tools(state)
    tool_msgs = result["messages"]
    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0], ToolMessage)
    assert "ERROR: Ya registraste la comida de este turno" in tool_msgs[0].content
    assert "TERMINANTEMENTE PROHIBIDO registrar el mismo plato o foto en una segunda franja" in tool_msgs[0].content


def test_db_facts_dedup_rejects_reused_photo_attachment():
    from db_facts import log_consumed_meal
    with patch("db_facts.execute_sql_query") as mock_query:
        # Simulamos que la query encuentra que este attachment_id ya existe
        mock_query.return_value = {"id": "existing-meal-id", "meal_name": "Plátano maduro"}
        res = log_consumed_meal(
            "fake-user", "Desayuno con plátano", 480, 21, 50, 22,
            ingredients=["plátano (≈180 g)"], meal_type="desayuno",
            nutrition={"attachment_ids": ["att-1234"]},
        )
        assert res == "deduped"
        # Verificar que la query buscó por attachment_ids
        sql_call = mock_query.call_args[0][0]
        assert "nutrition->'attachment_ids'" in sql_call


def test_db_facts_dedup_rejects_identical_macros_even_with_different_name():
    from db_facts import log_consumed_meal
    with patch("db_facts.execute_sql_query") as mock_query:
        # Primera llamada no es por attachment_ids, simulamos que encuentra fila idéntica por macros
        mock_query.return_value = {"id": "existing-meal-id"}
        res = log_consumed_meal(
            "fake-user", "Otro nombre para la misma comida", 480, 21, 50, 22,
            ingredients=["plátano (≈180 g)"], meal_type="desayuno",
        )
        assert res == "deduped"
        sql_call = mock_query.call_args[0][0]
        assert "calories = %s AND protein = %s" in sql_call
        assert "meal_name = %s" not in sql_call
