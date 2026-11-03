"""
Tests para [BB-023 / P1-DIARY-DELTA-CORRECTION · 2026-11-03]
Recálculo de calorías ocultas al editar un ingrediente por diferencial (BB-002 + BB-011).
Verifica que correcciones parciales de ingredientes no reseteen el plato a crudo/hervido
borrando el margen criollo de aceite, sofrito y salsa (~700-850 kcal).
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import _ajustar_calorias_por_delta_ingrediente
from prompts.chat_agent import (
    _CHAT_VOICE_RULES,
    build_tools_instructions,
    build_tools_instructions_stream,
    _PLATO_INSTRUCCION,
)


def test_delta_correction_maduros_preserva_sofrito_criollo():
    """
    Caso emblemático BB-023: Moro con pollo guisado y 3 maduros registrado en 800 kcal.
    El usuario aclara: "No eran 3 maduros, era solo 1".
    El LLM tiende a sumar en crudo (arroz hervido + pollo sin grasa + 1 maduro = 490 kcal).
    El guard por delta debe forzar 800 - 2*140 = 520 kcal (nunca 490 kcal).
    """
    user_id = "test-uid-bb023-maduros"
    meal_id = "meal-uuid-moro-800"

    mock_row = {
        "id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado y 3 maduros",
        "calories": 800,
        "protein": 45,
        "ingredients": ["Moro de habichuelas", "Pollo guisado", "3 maduros"],
        "created_at": "2026-11-03T13:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Moro de habichuelas con pollo guisado y 1 maduro",
        "calories": 490,  # LLM colapsó a cálculo en crudo
        "protein": 42,
    }

    messages = [
        HumanMessage(content="Anoté mi almuerzo"),
        AIMessage(content="Anoté tu almuerzo: Moro con pollo y 3 maduros, ~800 kcal."),
        HumanMessage(content="No eran 3 maduros, era solo 1"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None, "Debe interceptar y ajustar la corrección"
    assert res["calories"] == 520, f"Esperado 520 kcal (800 - 280), obtenido {res['calories']}"
    assert "1 maduro" in res["meal_name"]
    assert "delta_correction" in res["nutrition"]
    delta_meta = res["nutrition"]["delta_correction"]
    assert delta_meta["old_units"] == 3
    assert delta_meta["new_units"] == 1
    assert delta_meta["delta_calories"] == -280
    assert delta_meta["adjusted_calories"] == 520
    assert delta_meta["llm_proposed_calories"] == 490


def test_delta_correction_reduccion_huevos():
    """
    Desayuno con 4 huevos (650 kcal, 40g prot).
    Usuario aclara: "eran 2 huevos, no 4".
    LLM propone 350 kcal.
    Delta: -2 huevos * 75 kcal = -150 kcal. Total esperado = 500 kcal.
    """
    user_id = "test-uid-bb023-huevos"
    meal_id = "meal-uuid-huevos-650"

    mock_row = {
        "id": meal_id,
        "meal_name": "Tostadas con mantequilla de maní y 4 huevos",
        "calories": 650,
        "protein": 40,
        "ingredients": ["Tostadas", "Mantequilla de maní", "4 huevos"],
        "created_at": "2026-11-03T09:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Tostadas con mantequilla de maní y 2 huevos",
        "calories": 350,
        "protein": 28,
    }

    messages = [
        HumanMessage(content="Anoté mi desayuno"),
        AIMessage(content="Anoté tu desayuno: Tostadas y 4 huevos, ~650 kcal."),
        HumanMessage(content="eran 2 huevos, no 4"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 500  # 650 - 150
    assert "2 huevos" in res["meal_name"]


def test_delta_correction_aumento_unidades():
    """
    Aumento de unidades: "fueron 3 tostones en vez de 1".
    Fila previa: 400 kcal.
    Delta: +2 tostones * 60 = +120 kcal -> 520 kcal.
    Si el LLM propone 400 kcal o menos, debe corregirse hacia arriba.
    """
    user_id = "test-uid-bb023-aumento"
    meal_id = "meal-uuid-tostones"

    mock_row = {
        "id": meal_id,
        "meal_name": "Pechuga con 1 toston",
        "calories": 400,
        "protein": 35,
        "created_at": "2026-11-03T14:00:00",
    }

    tool_args = {
        "meal_id": meal_id,
        "meal_name": "Pechuga con 3 tostones",
        "calories": 390,  # LLM bajó por error al recalcular
        "protein": 35,
    }

    messages = [
        HumanMessage(content="3 tostones en vez de 1"),
    ]

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is not None
    assert res["calories"] == 520  # 400 + 120


def test_delta_correction_no_interfiere_en_cambio_total_de_plato():
    """
    Si el usuario cambió el plato completo ("no comí moro, comí avena"),
    la función debe retornar None y dejar que el LLM calcule normalmente.
    """
    user_id = "test-uid-bb023-cambio-total"
    tool_args = {
        "meal_id": "meal-uuid-moro",
        "meal_name": "Avena con leche descremada",
        "calories": 280,
    }
    messages = [
        HumanMessage(content="No comí moro con pollo, me comí una avena con leche"),
    ]

    mock_row = {
        "id": "meal-uuid-moro",
        "meal_name": "Moro con pollo",
        "calories": 800,
        "created_at": "2026-11-03T13:00:00",
    }

    with patch("tools._leer_fila_diario", return_value=mock_row):
        res = _ajustar_calorias_por_delta_ingrediente(user_id, tool_args, messages)

    assert res is None, "No debe interferir en cambios totales de plato"


def test_delta_correction_no_interfiere_si_delete_true():
    """
    Si delete=True, la función no debe alterar nada.
    """
    res = _ajustar_calorias_por_delta_ingrediente(
        "uid", {"meal_id": "m1", "delete": True}, [HumanMessage(content="no eran 3 maduros")]
    )
    assert res is None


def test_prompts_contienen_directiva_delta_correction():
    """
    Verifica que las 4 superficies de prompts contengan las directivas de
    recálculo por diferencial (BB-023 / P1-DIARY-DELTA-CORRECTION).
    """
    assert "P1-DIARY-DELTA-CORRECTION" in _CHAT_VOICE_RULES
    assert "DIFERENCIAL" in _CHAT_VOICE_RULES

    tools_inst = build_tools_instructions("uid123", "America/Santo_Domingo", "RD")
    assert "P1-DIARY-DELTA-CORRECTION" in tools_inst
    assert "DIFERENCIAL" in tools_inst

    tools_stream = build_tools_instructions_stream("uid123", "America/Santo_Domingo", "RD")
    assert "P1-DIARY-DELTA-CORRECTION" in tools_stream
    assert "DIFERENCIAL" in tools_stream

    assert "P1-DIARY-DELTA-CORRECTION" in _PLATO_INSTRUCCION
    assert "DIFERENCIAL" in _PLATO_INSTRUCCION
