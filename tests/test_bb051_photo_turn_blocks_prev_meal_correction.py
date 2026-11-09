# -*- coding: utf-8 -*-
"""[BB-051 · 2026-11-09] Regresión: Una foto de plato/bebida nueva NO puede alterar comidas de turnos previos.

Incidente Angelo (15:00):
El usuario registró su almuerzo (arroz con pollo y maduro, ID `02b2f9b8...`).
Una hora después envió una foto de jugo de guanábana con azúcar.
El LLM llamó primero a `correct_consumed_meal` sobre el almuerzo del turno previo.
Como el almuerzo estaba en `_meals_created_in_previous_turn`, el guard anterior no lo bloqueó.
Al ejecutarse `correct_consumed_meal`, emitió `[ID_REGISTRO_DIARIO: 02b2f9b8...]`, lo que causó
que `_meals_created_this_turn` creyera falsamente que se acababa de crear una comida en este turno.
Cuando el LLM intentó llamar a `log_consumed_meal` para registrar la guanábana, el guard anti-clones
lo bloqueó diciendo "Ya registraste la comida de este turno", dejando la guanábana sin registrar
y al coach respondiendo: "La pechuga de pollo ya está incluida... la guanábana no quedó registrada".

Garantías BB-051:
1. En turnos con foto de comida (`turn_plate_photos` o `turn_photo_to_log`), `correct_consumed_meal`
   sobre una comida de un turno previo queda TERMINANTEMENTE BLOQUEADO salvo solicitud explícita humana en texto.
2. `_meals_created_this_turn` solo cuenta creaciones reales de `log_consumed_meal`, nunca actualizaciones de `correct_consumed_meal`.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import (
    _meals_created_this_turn,
    execute_tools,
)


def test_meals_created_this_turn_ignores_correct_consumed_meal():
    """`_meals_created_this_turn` no debe contar llamadas de `correct_consumed_meal`."""
    messages = [
        HumanMessage(content="Guanábana · 1 cucharadita de azúcar"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "correct_consumed_meal",
                "args": {"meal_id": "prev-meal-uuid-123"},
                "id": "call_correct_1",
            }],
        ),
        ToolMessage(
            content="¡Corregido! Actualicé el registro existente ('Pollo con arroz') [ID_REGISTRO_DIARIO: prev-meal-uuid-123 — uso interno tuyo]",
            tool_call_id="call_correct_1",
        ),
    ]
    # No fue una creación: fue una corrección
    assert _meals_created_this_turn(messages) == set()


def test_photo_turn_blocks_correct_consumed_meal_on_previous_turn_meal():
    """En un turno con foto de comida, `correct_consumed_meal` sobre la comida del turno anterior debe ser bloqueado."""
    state = {
        "messages": [
            # Turno anterior: almuerzo registrado
            HumanMessage(content="Pechuga de pollo · 1.5 tazas"),
            AIMessage(
                content="",
                tool_calls=[{"name": "log_consumed_meal", "args": {"meal_name": "Pollo"}, "id": "call_lunch"}],
            ),
            ToolMessage(
                content="¡Éxito! [ID_REGISTRO_DIARIO: lunch-meal-uuid-999]",
                tool_call_id="call_lunch",
            ),
            AIMessage(content="Anoté tu almuerzo: arroz con pollo."),
            # Turno actual: foto de guanábana
            HumanMessage(content="Guanábana · 1 cucharadita de azúcar"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "correct_consumed_meal",
                    "args": {
                        "meal_id": "lunch-meal-uuid-999",
                        "meal_name": "Arroz con pollo",
                        "calories": 650,
                    },
                    "id": "call_erroneous_correct",
                }],
            ),
        ],
        "turn_photo_to_log": True,
        "turn_plate_photos": [{"desc": "Vaso corto con jugo de guanábana", "kcal": 145}],
        "user_id": "61a13831-2a70-4437-a084-0d3e09b653e4",
        "session_id": "test-session-id",
    }

    res = execute_tools(state)
    tool_msgs = res.get("messages", [])
    assert len(tool_msgs) == 1
    assert "TERMINANTEMENTE PROHIBIDO" in tool_msgs[0].content
    assert "FOTO DE PLATO" in tool_msgs[0].content
    assert "log_consumed_meal" in tool_msgs[0].content
