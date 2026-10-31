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
    # Fusión / agregado a comida existente: SÍ son
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Fue parte del almuerzo")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Agrégalo al almuerzo")])
    assert _es_solicitud_explicita_de_corregir_diario([HumanMessage(content="Se me quedó de la cena")])


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


def test_fusion_intent_detection():
    from agent import _detectar_intencion_parte_de_comida
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="Fue parte del almuerzo")]) == "almuerzo"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="agrégalo al desayuno porfa")]) == "desayuno"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="esto fue parte de la cena")]) == "cena"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="se me quedó del almuerzo")]) == "almuerzo"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="lo comí junto con el almuerzo")]) == "almuerzo"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="Me comí otro guineo más en el almuerzo")]) == "almuerzo"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="Hola qué tal")]) is None
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="Quiero comer pollo para el almuerzo")]) is None


def test_sumar_unidades_nombre():
    from agent import _sumar_unidades_nombre, _fusionar_ingredientes
    # 1 banana + 1 banana -> 2 bananas
    res = _sumar_unidades_nombre("Arroz blanco con habichuelas rojas, pollo frito y 1 guineo maduro", "1 guineo maduro")
    assert "2 guineos maduros" in res
    assert "1 guineo maduro y 1 guineo maduro" not in res

    # 2 tostadas + 1 tostada -> 3 tostadas
    res2 = _sumar_unidades_nombre("Huevos revueltos y 2 tostadas", "1 tostada")
    assert "3 tostadas" in res2

    # Different item -> joins with "y"
    res3 = _sumar_unidades_nombre("Arroz con pollo", "1 manzana")
    assert res3 == "Arroz con pollo y 1 manzana"

    # Full meal title passed by model -> does NOT duplicate title
    res4 = _sumar_unidades_nombre(
        "Arroz blanco con habichuelas rojas, pollo frito y 2 guineos maduros",
        "Arroz blanco con habichuelas rojas, pollo frito y 3 guineos maduros"
    )
    assert res4 == "Arroz blanco con habichuelas rojas, pollo frito y 3 guineos maduros"
    assert "y 1 Arroz blanco" not in res4

    # Fusionar ingredientes: caso delta individual
    orig_ing = [
        "arroz blanco (≈150 g)",
        "habichuelas rojas guisadas (≈100 g)",
        "2 piezas de pollo frito (≈150 g)",
        "2 guineos maduros medianos (≈238 g)"
    ]
    new_ing = ["1 guineo maduro (≈120 g)"]
    fused_ing = _fusionar_ingredientes(orig_ing, new_ing)
    assert len(fused_ing) == 4
    assert any("3 guineos maduros" in item for item in fused_ing)
    assert any("≈358 g" in item for item in fused_ing)

    # Fusionar ingredientes: caso reemplazo completo (NO duplicar arroz ni guineos)
    full_new_ing = [
        "arroz blanco (≈150 g)",
        "habichuelas rojas guisadas (≈100 g)",
        "2 piezas de pollo frito (≈150 g)",
        "4 guineos maduros medianos (≈476 g)"
    ]
    fused_full = _fusionar_ingredientes(orig_ing, full_new_ing)
    assert len(fused_full) == 4
    assert any("4 guineos maduros" in item for item in fused_full)
    assert not any("7 guineos" in item for item in fused_full)
    assert not any("arrozs" in item for item in fused_full)


def test_fusion_into_existing_meal_slot(monkeypatch):
    from agent import _intentar_fusionar_alimento_en_comida_existente
    import db_facts

    mock_meals = [{
        "id": "62b04607-952e-49b1-ab92-b8461a2d736f",
        "meal_name": "Arroz blanco con habichuelas rojas y pollo frito",
        "meal_type": "almuerzo",
        "calories": 630,
        "protein": 41,
        "carbs": 65,
        "healthy_fats": 20,
        "ingredients": ["arroz blanco (≈150 g)", "habichuelas rojas guisadas (≈100 g)", "pollo frito (≈150 g)"],
        "nutrition": {"attachment_ids": ["dd32eeb3-104e-492f-859d-cfa1410fcb0f"]},
        "created_at": "2026-10-26T17:08:27Z",
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda user_id: mock_meals)

    tool_args = {
        "meal_name": "1 guineo maduro mediano",
        "meal_type": "almuerzo",
        "calories": 105,
        "protein": 1,
        "carbs": 27,
        "healthy_fats": 0,
        "ingredients": ["1 guineo maduro mediano (≈118 g)"],
    }
    messages = [HumanMessage(content="Fue parte del almuerzo")]
    plate_photos = [{"attachment_id": "0054ec80-9496-4c7d-b7a3-8ba74e94662a"}]

    fused = _intentar_fusionar_alimento_en_comida_existente("user-123", tool_args, messages, plate_photos)
    assert fused is not None
    assert fused["meal_id"] == "62b04607-952e-49b1-ab92-b8461a2d736f"
    assert "guineo" in fused["meal_name"]
    assert fused["calories"] == 735
    assert fused["protein"] == 42
    assert fused["carbs"] == 92
    assert fused["healthy_fats"] == 20
    assert "1 guineo maduro mediano (≈118 g)" in fused["ingredients"]
    assert "0054ec80-9496-4c7d-b7a3-8ba74e94662a" in fused["nutrition"]["attachment_ids"]
    assert "dd32eeb3-104e-492f-859d-cfa1410fcb0f" in fused["nutrition"]["attachment_ids"]


def test_independent_meals_same_slot_never_fused():
    """Dos comidas distintas en la misma franja (ej. cenó a las 5pm y luego a las 10pm manda otra cena)
    NUNCA deben fusionarse si el usuario no dice explícitamente que fue parte de la comida previa."""
    from agent import _detectar_slot_para_fusion, _intentar_fusionar_alimento_en_comida_existente
    # El usuario responde "si, como cena" a una pregunta o manda un plato
    messages = [HumanMessage(content="si, como cena")]
    tool_args = {"meal_name": "Mangú con salchicha", "meal_type": "cena", "calories": 490}
    # No debe detectar slot para fusión
    assert _detectar_slot_para_fusion(messages, tool_args) is None
    assert _intentar_fusionar_alimento_en_comida_existente("user-123", tool_args, messages) is None


def test_meals_created_in_previous_turn_allows_rectification():
    """Si el usuario mandó una foto y 1 minuto después aclara el ingrediente (ej. 'Bistec de res'),
    _meals_created_in_previous_turn debe capturar el ID de la comida recién anotada."""
    from agent import _meals_created_in_previous_turn
    messages = [
        HumanMessage(content="Pechuga de pollo · Pechuga de pollo"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! [ID_REGISTRO_DIARIO: meal-abc-123] registrada", tool_call_id="t1"),
        AIMessage(content="Anoté tu cena."),
        HumanMessage(content="Bistec de res · Pechuga de pollo"),
    ]
    prev_ids = _meals_created_in_previous_turn(messages)
    assert "meal-abc-123" in prev_ids


def test_foto_para_anotar_identifies_composite_dish_names():
    """Textos con nombres de platos o ingredientes sobre fotos de plato deben marcarse como fotos para anotar."""
    from respuestas_de_la_foto import foto_para_anotar
    vision = {"kind": "multi", "items": [{"kind": "plato", "description": "Plato de mangú con salchicha"}]}
    assert foto_para_anotar(vision, "Guineito verde · Salchicha") is True
    assert foto_para_anotar(vision, "Pechuga de pollo · Pechuga de pollo") is True
    assert foto_para_anotar(vision, "Cerdo") is True
    # Consultas o preguntas siguen siendo NO para anotar
    assert foto_para_anotar(vision, "¿Cuánto engorda esto?") is False


def test_extra_food_immediate_addition_inherits_previous_meal_slot():
    """Si el usuario registra desayuno y 1 minuto después escribe 'Extra lata de atun genova',
    el slot destino debe heredar 'desayuno' de forma inmediata."""
    from agent import _detectar_slot_para_fusion, _slot_comida_turno_previo
    messages = [
        HumanMessage(content="Desayune 1 rodaja de pan con queso cottage y mantequilla de mani"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "desayuno", "meal_name": "1 rodaja de pan con queso cottage"}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! Se ha registrado el consumo de '1 rodaja de pan con queso cottage' (230 kcal, 11g proteína) como desayuno en tu diario. [ID_REGISTRO_DIARIO: meal-desayuno-789]", tool_call_id="t1"),
        AIMessage(content="Anoté tu desayuno..."),
        HumanMessage(content="Extra lata de atun genova"),
    ]
    slot, mid = _slot_comida_turno_previo(messages)
    assert slot == "desayuno"
    assert mid == "meal-desayuno-789"

    tool_args = {"meal_name": "lata de atun genova", "meal_type": "almuerzo", "calories": 190, "protein": 24}
    assert _detectar_slot_para_fusion(messages, tool_args) == "desayuno"


def test_extra_food_fuses_into_previous_meal_sums_calories(monkeypatch):
    """La lata de atún extra (190 kcal, 24g prot) debe sumarse al desayuno previo (230 kcal, 11g prot)
    totalizando 420 kcal y 35g proteína en la misma fila."""
    from agent import _intentar_fusionar_alimento_en_comida_existente
    import db_facts

    mock_meals = [{
        "id": "meal-desayuno-789",
        "meal_name": "1 rodaja de pan con queso cottage y mantequilla de maní",
        "meal_type": "desayuno",
        "calories": 230,
        "protein": 11,
        "carbs": 25,
        "healthy_fats": 9,
        "ingredients": ["1 rodaja de pan", "queso cottage (30 g)", "mantequilla de maní (15 g)"],
        "nutrition": {},
        "created_at": "2026-10-30T12:51:00Z",
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda user_id: mock_meals)

    messages = [
        HumanMessage(content="Desayune 1 rodaja de pan con queso cottage y mantequilla de mani"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "desayuno", "meal_name": "1 rodaja de pan con queso cottage y mantequilla de maní"}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! Se ha registrado el consumo como desayuno en tu diario. [ID_REGISTRO_DIARIO: meal-desayuno-789]", tool_call_id="t1"),
        AIMessage(content="Anoté tu desayuno..."),
        HumanMessage(content="Extra lata de atun genova"),
    ]
    tool_args = {
        "meal_name": "lata de atún Genova",
        "meal_type": "almuerzo",  # El LLM o reloj intentó marcarlo como almuerzo
        "calories": 190,
        "protein": 24,
        "carbs": 0,
        "healthy_fats": 10,
        "ingredients": ["1 lata de atún Genova (140 g)"],
    }
    fused = _intentar_fusionar_alimento_en_comida_existente("user-123", tool_args, messages)
    assert fused is not None
    assert fused["meal_id"] == "meal-desayuno-789"
    assert fused["meal_type"] == "desayuno"
    assert fused["calories"] == 420  # 230 + 190
    assert fused["protein"] == 35   # 11 + 24
    assert "atún" in fused["meal_name"].lower() or "atun" in fused["meal_name"].lower()



