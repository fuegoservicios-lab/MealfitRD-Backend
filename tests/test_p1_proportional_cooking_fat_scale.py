"""
Tests para [BB-028 / P1-PROPORTIONAL-COOKING-FAT-SCALE · 2026-11-04]
Escalamiento proporcional del buffer de grasa de cocción criolla (sofrito/aceite)
al reducir porciones, fracciones globales o descartar el vehículo absorbente
(moro, arroz, víveres fritos, etc.).
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage

from agent import _ajustar_calorias_por_delta_ingrediente
from prompts.chat_agent import (
    _CHAT_VOICE_RULES,
    build_tools_instructions,
    build_tools_instructions_stream,
    _PLATO_INSTRUCCION,
)


def test_escala_global_mitad_plato():
    """
    Caso escala global 50%: Moro con pollo guisado en 800 kcal (45g prot).
    Usuario: 'Solo me comí la mitad del plato'.
    El LLM propuso 650 kcal (inflado por mantener 180 kcal de aceite fijo).
    El corrector proporcional debe ajustar a 400 kcal (23g prot),
    escalando el buffer de grasa de 180 a 90 kcal.
    """
    user_id = "test-uid-bb028-global-mitad"
    meal_id = "meal-uuid-moro-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado",
        "calories": 800,
        "protein": 45,
        "carbs": 80,
        "healthy_fats": 25,
        "ingredients": ["Moro de habichuelas", "Pollo guisado"],
        "created_at": "2026-11-04T12:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado",
        "calories": 650,  # LLM retuvo grasa estática
        "protein": 38,
    }

    messages = [
        HumanMessage(content="Anoté mi almuerzo"),
        AIMessage(content="Anoté tu almuerzo: Moro con pollo, ~800 kcal."),
        HumanMessage(content="Solo me comí la mitad del plato"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None, "Debe interceptar la reducción global"
    assert res["calories"] == 400, f"Esperado 400 kcal, obtenido {res['calories']}"
    assert res["protein"] == 23, f"Esperado 23g prot, obtenido {res['protein']}"
    assert "media porción" in res["meal_name"].lower()

    fat_scale = res["nutrition"]["cooking_fat_scale"]
    assert fat_scale["reduction_type"] == "global_fraction"
    assert fat_scale["scale_factor"] == 0.5
    assert fat_scale["orig_implicit_cooking_kcal"] == 180
    assert fat_scale["remaining_implicit_cooking_kcal"] == 90
    assert fat_scale["scaled_oil_delta"] == 90


def test_escala_parcial_vehiculo_moro():
    """
    Caso reducción del vehículo absorbente al 50%:
    Moro con pollo guisado en 800 kcal (45g prot).
    Usuario: 'Solo me comí la mitad del moro'.
    El moro base aporta 260 kcal y retiene 70% de la grasa de cocción (126 kcal).
    Al dejar la mitad del moro:
    - Delta base = -130 kcal
    - Delta aceite absorbido = -63 kcal
    - Total = 800 - 193 = 607 kcal (42g prot).
    El buffer de grasa restante pasa de 180 a 117 kcal (sin inflación).
    """
    user_id = "test-uid-bb028-moro-mitad"
    meal_id = "meal-uuid-moro-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado",
        "calories": 800,
        "protein": 45,
        "ingredients": ["Moro de habichuelas", "Pollo guisado"],
        "created_at": "2026-11-04T12:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado",
        "calories": 710,  # LLM restó arroz crudo sin aceite
        "protein": 44,
    }

    messages = [
        HumanMessage(content="Solo me comí la mitad del moro"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 607, f"Esperado 607 kcal, obtenido {res['calories']}"
    assert res["protein"] == 43, f"Esperado 43g prot, obtenido {res['protein']}"
    assert "media porción de moro" in res["meal_name"].lower()

    fat_scale = res["nutrition"]["cooking_fat_scale"]
    assert fat_scale["reduction_type"] == "carrier_scale"
    assert fat_scale["carrier_item"] == "moro"
    assert fat_scale["scale_factor"] == 0.5
    assert fat_scale["remaining_implicit_cooking_kcal"] == 117
    assert fat_scale["scaled_oil_delta"] == 63


def test_descarte_casi_todo_el_arroz_solo_carne():
    """
    Caso 'Dejé casi todo el arroz, solo me comí la carne' (scale_factor=0.1):
    Moro con pollo guisado en 800 kcal (45g prot).
    Delta base moro (90% fuera) = -234 kcal.
    Delta grasa moro (90% fuera) = -113 kcal.
    Delta total = -347 kcal -> 453 kcal (40g prot).
    """
    user_id = "test-uid-bb028-casi-todo-arroz"
    meal_id = "meal-uuid-arroz-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Arroz blanco con pollo guisado",
        "calories": 800,
        "protein": 45,
        "ingredients": ["Arroz blanco", "Pollo guisado"],
        "created_at": "2026-11-04T12:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Pollo guisado",
        "calories": 650,  # LLM infló buffer
        "protein": 38,
    }

    messages = [
        HumanMessage(content="Dejé casi todo el arroz, solo me comí la carne"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 453, f"Esperado 453 kcal, obtenido {res['calories']}"
    assert res["protein"] == 41, f"Esperado 41g prot, obtenido {res['protein']}"

    fat_scale = res["nutrition"]["cooking_fat_scale"]
    assert fat_scale["reduction_type"] == "carrier_scale"
    assert fat_scale["carrier_item"] == "arroz"
    assert fat_scale["remaining_implicit_cooking_kcal"] == 67


def test_descarte_total_arroz_solo_pollo():
    """
    Caso 'Dejé todo el arroz, solo me comí la carne' (scale_factor=0.0):
    Moro con pollo en 800 kcal (45g prot).
    Todo el moro fuera (-260 kcal) y toda la grasa del moro fuera (-126 kcal).
    Total = 800 - 386 = 414 kcal (40g prot).
    Grasa residual preservada: 54 kcal (el sofrito/salsa de la carne).
    """
    user_id = "test-uid-bb028-todo-arroz"
    meal_id = "meal-uuid-moro-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Moro con pollo guisado",
        "calories": 800,
        "protein": 45,
        "ingredients": ["Moro", "Pollo guisado"],
        "created_at": "2026-11-04T12:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Pollo guisado",
        "calories": 620,
        "protein": 38,
    }

    messages = [
        HumanMessage(content="Dejé todo el arroz, solo me comí la carne"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 414, f"Esperado 414 kcal, obtenido {res['calories']}"
    assert res["protein"] == 40, f"Esperado 40g prot, obtenido {res['protein']}"
    fat_scale = res["nutrition"]["cooking_fat_scale"]
    assert fat_scale["remaining_implicit_cooking_kcal"] == 54


def test_escala_porcentaje_global():
    """
    Caso porcentaje global 'Solo comí el 60%':
    800 * 0.60 = 480 kcal (27g prot).
    """
    user_id = "test-uid-bb028-pct"
    meal_id = "meal-uuid-moro-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Locrio de pollo con tostones",
        "calories": 800,
        "protein": 45,
        "created_at": "2026-11-04T12:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Locrio de pollo con tostones",
        "calories": 700,
    }

    messages = [
        HumanMessage(content="Solo me comí el 60%"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 480
    assert res["protein"] == 27
    assert "60%" in res["meal_name"]


def test_prompts_contienen_directiva_bb028():
    """
    Verifica que las directivas de BB-028 / P1-PROPORTIONAL-COOKING-FAT-SCALE
    estén presentes en las instrucciones del coach.
    """
    assert "BB-028" in _CHAT_VOICE_RULES
    assert "P1-PROPORTIONAL-COOKING-FAT-SCALE" in _CHAT_VOICE_RULES

    tools_inst = build_tools_instructions("uid123", "America/Santo_Domingo", "RD")
    assert "BB-028" in tools_inst
    assert "P1-PROPORTIONAL-COOKING-FAT-SCALE" in tools_inst

    tools_stream = build_tools_instructions_stream("uid123", "America/Santo_Domingo", "RD")
    assert "BB-028" in tools_stream
    assert "P1-PROPORTIONAL-COOKING-FAT-SCALE" in tools_stream

    assert "BB-028" in _PLATO_INSTRUCCION
    assert "P1-PROPORTIONAL-COOKING-FAT-SCALE" in _PLATO_INSTRUCCION
