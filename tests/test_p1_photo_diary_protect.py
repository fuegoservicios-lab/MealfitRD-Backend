# -*- coding: utf-8 -*-
"""[P1-PHOTO-DIARY-PROTECT · 2026-10-26] Protección de comidas anteriores contra respuestas a dudas de fotos.

Incidente: El usuario tenía un desayuno registrado (Plátano maduro con 2 huevos). Al mediodía subió una foto de almuerzo
(arroz, habichuelas, pollo) y contestó «Frito» a la pregunta sobre cómo estaba preparado el pollo.
El coach interpretó «Frito» como una continuación de los huevos del desayuno y llamó a `correct_consumed_meal` sobre el
desayuno de la mañana, en lugar de registrar el almuerzo como comida nueva.

Garantías:
1. `_es_solicitud_explicita_de_corregir_diario`: Respuestas cortas o de dudas («Frito», «2 huevos», «Salmón») NO son peticiones de corregir comidas pasadas.
2. `_meals_created_this_turn`: Solo las comidas creadas en este mismo turno pueden ser corregidas/ajustadas durante un turno con foto.
3. `execute_tools`: Bloquea deterministamente cualquier llamada a `correct_consumed_meal` sobre comidas previas si el turno trae foto de plato y el usuario no lo pidió expresamente.
4. `route_tools`: Una llamada a `correct_consumed_meal` NO satisface `turn_photo_to_log`: si no hubo `log_consumed_meal`, fuerza `nudge_photo_to_log`.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import (
    _es_solicitud_explicita_de_corregir_diario,
    _meals_created_this_turn,
    execute_tools,
    route_tools,
)
from langgraph.graph import END


def test_es_solicitud_explicita_de_corregir_diario():
    # Respuestas a dudas de foto o rótulos: NO son solicitudes de corrección de diario
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Frito")])
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Guisado")])
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="2 huevos")])
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Salmón")])
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Mi almuerzo")])
    assert not _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="📷")])

    # Solicitudes explícitas humanas de corrección/borrado: SÍ son
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Corrección eran dos huevos")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Borra el desayuno")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Elimina lo que anotaste")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="No me comí eso, era un ejemplo")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Quita la cena")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Cambia el desayuno por favor")])


def test_meals_created_this_turn():
    # ToolMessage del turno previo no cuenta
    msgs_old = [
        HumanMessage(content="Desayuné"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! [ID_REGISTRO_DIARIO: old-bf-1234]", tool_call_id="t1"),
        HumanMessage(content="Frito"),  # nuevo turno
    ]
    assert _meals_created_this_turn(msgs_old) == set()

    # ToolMessage creado dentro de este turno sí cuenta
    msgs_cur = msgs_old + [
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "t2"}]),
        ToolMessage(content="¡Éxito! [ID_REGISTRO_DIARIO: new-lunch-5678]", tool_call_id="t2"),
    ]
    assert _meals_created_this_turn(msgs_cur) == {"new-lunch-5678"}


def test_execute_tools_blocks_correct_consumed_meal_on_prior_meal_during_photo_turn():
    state = {
        "messages": [
            HumanMessage(content="Frito"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "correct_consumed_meal",
                    "args": {
                        "meal_id": "edb55c02-f910-422f-af41-c6f05ec69a8c",
                        "meal_name": "Plátano maduro con 2 huevos fritos y tostones cubiertos de queso",
                        "calories": 435,
                    },
                    "id": "call_test_123",
                }],
            ),
        ],
        "turn_photo_to_log": True,
        "turn_plate_photos": [{"desc": "Arroz blanco con habichuelas y pollo", "kcal": 630}],
        "user_id": "test-user-id",
        "session_id": "test-session-id",
    }

    res = execute_tools(state)
    tool_msgs = res.get("messages", [])
    assert len(tool_msgs) == 1
    assert "TERMINANTEMENTE PROHIBIDO" in tool_msgs[0].content
    assert "FOTO DE PLATO NUEVA" in tool_msgs[0].content
    assert "log_consumed_meal" in tool_msgs[0].content


def test_route_tools_nudges_if_only_correct_consumed_meal_called_on_photo_turn():
    # Turno con foto para anotar donde el modelo solo llamó a correct_consumed_meal:
    # Debe disparar nudge_photo_to_log porque no se llamó a log_consumed_meal
    state = {
        "messages": [
            HumanMessage(content="Frito"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "correct_consumed_meal",
                    "args": {"meal_id": "some-id", "meal_name": "algo"},
                    "id": "c1",
                }],
            ),
            ToolMessage(content="resultado", tool_call_id="c1"),
            AIMessage(content="Actualicé tu desayuno."),
        ],
        "turn_photo_to_log": True,
        "photo_log_retried": False,
        "plate_photo_retried": True,
        "diary_claim_retried": True,
    }

    assert route_tools(state) == "nudge_photo_to_log"

    # Si se llamó a log_consumed_meal, route_tools termina en END
    state["messages"] = [
        HumanMessage(content="Frito"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "log_consumed_meal",
                "args": {"meal_name": "Arroz con habichuelas y pollo frito", "calories": 630},
                "id": "c2",
            }],
        ),
        ToolMessage(content="¡Éxito! [ID_REGISTRO_DIARIO: new-id]", tool_call_id="c2"),
        AIMessage(content="Anoté tu almuerzo."),
    ]
    assert route_tools(state) == END
