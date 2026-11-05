# -*- coding: utf-8 -*-
"""[BB-036 · 2026-11-05] Tests de supresión de confirmaciones absurdas al registrar foto tras nudge o con porción.

Verifica:
1. `detectar_slot_de_nudge_reciente`: hereda la franja de comida (merienda, almuerzo, cena, desayuno)
   del último mensaje proactivo o pregunta del coach.
2. `marcar_rotulo`: ante foto + porción/detalle («1.5 a 2 cucharadas») tras un recordatorio de merienda,
   asigna automáticamente rotulo='merienda'.
3. `_registro_es_la_foto`: ante ajuste calórico por porción aclarada (ej. 215 kcal de tortita con mantequilla de maní
   vs 450 kcal visuales del plato), NO descarta el registro si los alimentos coinciden.
4. Preservación del guard contra snacks aislados (P1-PLATE-PHOTO-INTEGRITY: 143 kcal de huevos en plato de 670 kcal se descarta).
5. Eliminación de la cláusula restrictiva tóxica en `nudge_plate_photo` y `NOTA_REINTENTO`.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent
from respuestas_de_la_foto import (
    detectar_slot_de_nudge_reciente,
    es_detalle_o_porcion_de_comida,
    marcar_rotulo,
    foto_para_anotar,
    NOTA_REINTENTO,
)


def test_detectar_slot_de_nudge_reciente():
    # Caso 1: Nudge de merienda de la tarde (caso real del usuario BB-036)
    msgs_merienda = [
        {"role": "user", "content": "hola"},
        {"role": "model", "content": "Haz una pausa para merendar ahora y seguir con buen ritmo durante la tarde; luego puedes dejarla registrada por aquí."},
    ]
    assert detectar_slot_de_nudge_reciente(msgs_merienda) == "merienda"

    # Caso 2: Pregunta de almuerzo
    msgs_almuerzo = [
        {"role": "model", "content": "Una batata asada con carne de res guisada te caería de lo mejor para almorzar ahora."},
    ]
    assert detectar_slot_de_nudge_reciente(msgs_almuerzo) == "almuerzo"

    # Caso 3: Sugerencia de cena
    msgs_cena = [
        {"role": "model", "content": "Para cenar ahora, ¿te provoca un filete de pescado con arroz y habichuelas?"},
    ]
    assert detectar_slot_de_nudge_reciente(msgs_cena) == "cena"

    # Caso 4: Desayuno
    msgs_desayuno = [
        AIMessage(content="Una avena cremosa con pavo te vendría buenísima para desayunar ahora."),
    ]
    assert detectar_slot_de_nudge_reciente(msgs_desayuno) == "desayuno"

    # Caso 5: Mensaje genérico sin franja
    msgs_generico = [
        {"role": "model", "content": "¡Hola! ¿Cómo te sientes hoy para entrenar?"},
    ]
    assert detectar_slot_de_nudge_reciente(msgs_generico) is None


def test_es_detalle_o_porcion_de_comida():
    assert es_detalle_o_porcion_de_comida("1.5 a 2 cucharadas") is True
    assert es_detalle_o_porcion_de_comida("1 cucharada") is True
    assert es_detalle_o_porcion_de_comida("2 rebanadas") is True
    assert es_detalle_o_porcion_de_comida("con mantequilla de maní") is True
    assert es_detalle_o_porcion_de_comida("solo la carne") is True
    assert es_detalle_o_porcion_de_comida("¿puedo comer esto?") is False
    assert es_detalle_o_porcion_de_comida("¿cuántas calorías tiene esto?") is False


def test_marcar_rotulo_con_nudge_y_porcion():
    vision = {
        "kind": "plato",
        "description": "Tortita de arroz con mantequilla de maní (Estimación: Calorías: 450, Proteína: 12g, Carbohidratos: 40g, Grasas Saludables: 20g)",
    }
    recent_merienda = [
        {"role": "model", "content": "Haz una pausa para merendar ahora y seguir con buen ritmo durante la tarde; luego puedes dejarla registrada por aquí."},
    ]

    # Foto con porción «1.5 a 2 cucharadas» tras nudge de merienda
    v_out = marcar_rotulo(vision, "1.5 a 2 cucharadas", recent_messages=recent_merienda)
    assert v_out.get("rotulo") == "merienda"

    # Foto sola (sin texto) tras nudge de merienda
    v_solo = marcar_rotulo(vision, "", recent_messages=recent_merienda)
    assert v_solo.get("solo_foto") is True
    assert v_solo.get("rotulo") == "merienda"


def test_registro_es_la_foto_con_ajuste_calorico_porcion():
    foto = {
        "desc": "Tortita de arroz con mantequilla de maní",
        "kcal": 450.0,
        "attachment_id": "att-123",
    }
    # Registro del modelo ajustando la porción a 215 kcal (menos del 50% de 450 kcal)
    # pero manteniendo los alimentos clave de la foto
    reg_porcion = {
        "meal_name": "Tortita de arroz con 1.5 a 2 cucharadas de mantequilla de maní",
        "calories": 215,
    }
    assert agent._registro_es_la_foto(reg_porcion, foto) is True

    reg_simple = {
        "meal_name": "Tortita de arroz con mantequilla de maní",
        "calories": 210,
    }
    assert agent._registro_es_la_foto(reg_simple, foto) is True


def test_registro_es_la_foto_preserva_guard_plato_integro():
    # Regresión P1-PLATE-PHOTO-INTEGRITY: Si el plato tiene 670 kcal de mangú, huevo y salami
    # y el registro solo anota 2 huevos (143 kcal), DEBE retornar False para forzar el plato completo
    foto_completa = {
        "desc": "Plátano maduro hervido con 2 huevos fritos y salami",
        "kcal": 670.0,
        "attachment_id": "att-456",
    }
    reg_aislado = {
        "meal_name": "2 huevos",
        "calories": 143,
    }
    assert agent._registro_es_la_foto(reg_aislado, foto_completa) is False


def test_nudge_plate_photo_sin_clausula_toxica():
    state = {
        "turn_plate_photos": [{"desc": "Tortita de arroz con mantequilla de maní", "kcal": 450.0, "attachment_id": "att-1"}],
        "messages": [
            HumanMessage(content="1.5 a 2 cucharadas"),
            AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_name": "Manzana", "calories": 80}, "id": "call_1"}]),
        ],
        "turn_photo_to_log": True,
        "plate_photo_retried": False,
        "user_id": "test-user-123",
    }
    resultado = agent.nudge_plate_photo(state)
    msg_content = resultado["messages"][0].content

    # Verificar que NO incluye la cláusula que inducía al modelo a preguntar si se la comió
    assert "él NO dijo que se la comiera, no registres nada" not in msg_content
    # Verificar que prohíbe pedir confirmaciones absurdas
    assert "PROHIBIDO TERMINANTEMENTE pedir confirmaciones absurdas" in msg_content or "PROHIBIDO preguntar si se lo comió" in msg_content
    # Verificar que NOTA_REINTENTO prohíbe confirmaciones redundantes
    assert "ESTÁ TERMINANTEMENTE PROHIBIDO pedir confirmación redundante" in NOTA_REINTENTO
