# -*- coding: utf-8 -*-
"""[BB-014 · 2026-11-01] Pruebas de regresión para:
1. Registro de múltiples comidas declaradas en un mismo mensaje de texto
   (ej. «De comida un bowl de pasta con pollo y de cena un steak de res»).
2. Supresión de preguntas repetidas sobre comidas faltantes (ej. «¿Qué comiste en el desayuno, o estuviste en ayunas?»)
   cuando el asistente ya hizo esa pregunta en un turno previo de la conversación.
"""
from __future__ import annotations

from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from meal_registration_time import _RE_EXPLICIT_MEAL_SLOT, explicit_slot_named_by_user
from agent import (
    _es_segundo_plato_distinto_del_turno,
    _neutralizar_nota_comida_faltante_ya_preguntada,
    _strip_repeated_missing_meal_question,
    execute_tools,
)


def test_re_explicit_meal_slot_matches_de_comida_and_de_cena():
    msg = "De comida un bowl de pasta con pollo y de cena un steak de res"
    slots = _RE_EXPLICIT_MEAL_SLOT.findall(msg)
    assert len(slots) == 2
    assert explicit_slot_named_by_user([HumanMessage(content="De comida un bowl de pasta con pollo")])


def test_execute_tools_allows_second_distinct_meal_in_multi_meal_message():
    human = HumanMessage(content="De comida un bowl de pasta con pollo y de cena un steak de res")
    ai_1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Bowl de pasta con pollo",
                "calories": 650,
                "protein": 45,
                "carbs": 75,
                "healthy_fats": 15,
                "meal_type": "almuerzo",
                "explicit_slot": True,
            },
            "id": "call-1",
        }],
    )
    tool_1 = ToolMessage(
        content="¡Éxito! [ID_REGISTRO_DIARIO: ccb17945-b770-40e4-9d47-06285500e62e] Registrado como almuerzo.",
        tool_call_id="call-1",
    )
    ai_2 = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Steak de res",
                "calories": 400,
                "protein": 45,
                "carbs": 0,
                "healthy_fats": 22,
                "meal_type": "cena",
                "explicit_slot": True,
            },
            "id": "call-2",
        }],
    )

    state = {
        "messages": [human, ai_1, tool_1, ai_2],
        "user_id": "7061dbfa-d364-44ef-86de-569e7b4bff78",
        "session_id": "db1ae7e9-6186-465f-8a5d-45b3ddd99a90",
        "turn_plate_photos": [],
    }

    fake_tool = patch("agent.agent_tools")
    with fake_tool as mock_tools:
        from unittest.mock import MagicMock
        t = MagicMock()
        t.name = "log_consumed_meal"
        t.invoke.return_value = "¡Éxito! Registrado Steak de res como cena."
        mock_tools.__iter__.return_value = [t]
        res = execute_tools(state)
        assert t.invoke.called
        assert "ERROR: Ya registraste la comida de este turno" not in res["messages"][0].content


def test_es_segundo_plato_distinto_del_turno_even_without_two_explicit_slot_keywords():
    user_text = "Al mediodía un bowl de pasta con pollo y por la noche un steak de res"
    human = HumanMessage(content=user_text)
    ai_1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Bowl de pasta con pollo",
                "meal_type": "almuerzo",
            },
            "id": "call-1",
        }],
    )
    tool_1 = ToolMessage(
        content="¡Éxito! [ID_REGISTRO_DIARIO: id-1]",
        tool_call_id="call-1",
    )
    second_args = {
        "meal_name": "Steak de res",
        "meal_type": "cena",
    }
    assert _es_segundo_plato_distinto_del_turno(
        second_args, [human, ai_1, tool_1], user_text, current_tool_id="call-2"
    )


def test_neutralizar_and_strip_repeated_missing_meal_question():
    messages = [
        HumanMessage(content="De comida un bowl de pasta con pollo y de cena un steak de res"),
        AIMessage(
            content=(
                "Anoté tu **almuerzo** —el bowl de pasta con pollo— con aprox. **650 kcal** "
                "y **45 g de proteína**. ¿Qué comiste en el desayuno, o estuviste en ayunas?"
            )
        ),
        HumanMessage(content="Registra un filete de carne"),
    ]
    raw_tool_result = (
        "¡Éxito! Se ha registrado el consumo de 'Filete de carne' como cena en tu diario."
        " (Para el asistente: son las 9:10 PM. hoy sigue sin registrar desayuno (ya pasó su hora) "
        "— cierra ofreciendo agregar LA QUE FALTA por su nombre, en una pregunta corta sobre qué comió "
        "para anotarlo o si estuvo en ayunas (nunca «cuando la comas», ni «¿te propongo algo para comer?», "
        "ni la pregunta genérica de si le falta algo). Una sola vez: si ya se lo preguntaste en esta "
        "conversación y no lo contestó, no lo repitas. Si ya preguntaste por esa comida antes en este "
        "chat, cierra sin volver a nombrarla.) [ID_REGISTRO_DIARIO: e607ce61]"
    )

    updated = _neutralizar_nota_comida_faltante_ya_preguntada(raw_tool_result, messages)
    assert "PROHIBIDO volver a preguntar qué comió en desayuno o si estuvo en ayunas" in updated
    assert "cierra ofreciendo agregar LA QUE FALTA" not in updated

    repeated_resp = (
        "Anoté tu **cena**: filete de carne, aprox. **400 kcal** y **45 g de proteína**. "
        "¿Qué comiste en el desayuno, o estuviste en ayunas? [UI_ACTION: REFRESH_INVENTORY]"
    )
    cleaned = _strip_repeated_missing_meal_question(repeated_resp, messages)
    assert "¿Qué comiste en el desayuno" not in cleaned
    assert cleaned == "Anoté tu **cena**: filete de carne, aprox. **400 kcal** y **45 g de proteína**. [UI_ACTION: REFRESH_INVENTORY]"


def test_bb015_almuerzo_missing_does_not_say_estuviste_en_ayunas():
    msg_history = [HumanMessage(content="Cené NY steak con pechuga de pollo")]
    resp = (
        "Anoté tu **cena** de NY steak, pechuga de pollo a la parrilla, zanahorias y puré de papa: "
        "aprox. **780 kcal** y **71 g de proteína**. ¿Qué comiste en el **almuerzo** de hoy, o estuviste en ayunas?"
    )
    cleaned = _strip_repeated_missing_meal_question(resp, msg_history)
    assert "estuviste en ayunas" not in cleaned
    assert "¿Qué comiste en el **almuerzo** de hoy, o no almorzaste?" in cleaned


def test_bb015_past_diary_block_includes_protein_when_present():
    from datetime import date
    from chat_history_context import build_past_diary_block

    hoy = date(2026, 11, 1)
    rows = [
        {
            "meal_name": "1 lata de atún Genova (estimado, escurrida)",
            "meal_type": "almuerzo",
            "calories": 210,
            "protein": 29,
            "consumed_at": "2026-10-30T16:52:03+00:00",
        }
    ]
    block = build_past_diary_block(rows, hoy, days_back=3, max_chars=3000)
    assert "(210 kcal, 29g prot)" in block

