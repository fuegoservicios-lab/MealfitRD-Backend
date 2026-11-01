# [P1-DELETE-GUARD · 2026-10-31]
# Tests para el guardián de borrado en diario: prohíbe correct_consumed_meal(delete=True)
# sin solicitud explícita del usuario (ej. erratas como "Tsurus blanca" o respuestas a preguntas de slot).

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import _es_solicitud_explicita_de_borrado_diario, execute_tools


def test_es_solicitud_explicita_de_borrado_positivos():
    ejemplos_positivos = [
        "bórralo por favor",
        "borralo ya",
        "elimina el desayuno de hoy",
        "no me comí eso",
        "no me lo comi",
        "era un ejemplo, no comí nada",
        "es un ejemplo nada más",
        "quita eso del diario",
        "quítalo por favor",
        "deshaz el último registro",
        "cancela el registro de la cena",
        "era broma, no lo anotes",
    ]
    for frase in ejemplos_positivos:
        msgs = [HumanMessage(content=frase)]
        assert _es_solicitud_explicita_de_borrado_diario(msgs) is True, f"Falló en positivo: {frase}"


def test_es_solicitud_explicita_de_borrado_negativos():
    ejemplos_negativos = [
        "Huevo revuelto y Tsurus blanca",
        "Almuerzo",
        "Desayuno",
        "Sindesayuno",
        "Huevos fritos con tostadas",
        "Huevo revuelto sin sal",
        "Comí pollo guisado con ensalada verde",
        "San chocho de lenteja y chuleta",
        "2 huevos revueltos",
    ]
    for frase in ejemplos_negativos:
        msgs = [HumanMessage(content=frase)]
        assert _es_solicitud_explicita_de_borrado_diario(msgs) is False, f"Falló en negativo: {frase}"


def test_execute_tools_bloquea_delete_sin_solicitud_del_usuario():
    # Simula el turno exacto donde el usuario dijo "Huevo revuelto y Tsurus blanca"
    # y el LLM intentó llamar a correct_consumed_meal(delete=True)
    messages = [
        HumanMessage(content="Huevo revuelto y Tsurus blanca"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "correct_consumed_meal",
                "args": {"meal_id": "01e384f4-fake-id", "delete": True},
                "id": "call_test_delete_1",
                "type": "tool_call",
            }]
        )
    ]
    state = {
        "messages": messages,
        "user_id": "test-user-id",
    }
    result = execute_tools(state)
    out_msgs = result.get("messages") or []
    assert len(out_msgs) == 1
    tool_msg = out_msgs[0]
    assert isinstance(tool_msg, ToolMessage)
    assert "ERROR: El usuario NO ha pedido borrar ni eliminar ninguna comida del diario" in tool_msg.content
    assert tool_msg.tool_call_id == "call_test_delete_1"
