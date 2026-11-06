# -*- coding: utf-8 -*-
"""
Tests automatizados para soporte multi-comida con marcadores temporales,
tolerancia morfológica para jerga y supresión de reclamos de desayuno tardío (BB-046 a BB-048).
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from meal_registration_time import count_meal_indicators_in_text
from agent import (
    _token_match,
    _tokens_coinciden,
    _es_segundo_plato_distinto_del_turno,
)
from tools import _nota_comidas_sin_registrar


# ==============================================================================
# BB-046: Detección de Múltiples Comidas con Marcadores Temporales
# ==============================================================================

def test_bb046_count_meal_indicators_in_text():
    """Detecta múltiples momentos de comida cuando el usuario usa horas o partes del día."""
    # Frase exacta de Darian
    t1 = "A las 12 me comí un plato de sancocho y a la noche un chofa"
    assert count_meal_indicators_in_text(t1) >= 2

    # Otros ejemplos dominicanos comunes
    t2 = "Al mediodía arroz con pollo y más tarde una batida de proteína"
    assert count_meal_indicators_in_text(t2) >= 2

    t3 = "En la mañana 3 tostadas y por la noche mangú con salami"
    assert count_meal_indicators_in_text(t3) >= 2

    # Una sola comida no debe contar como múltiples
    t_single = "Me comí un sancocho bien bueno"
    assert count_meal_indicators_in_text(t_single) < 2


def test_bb046_token_match_morfologico():
    """Valida coincidencia flexible por prefijo/raíz común de al menos 4 letras."""
    # 'chofa' (slang) coincide con 'chofan'
    assert _token_match("chofa", "chofan") is True
    assert _token_match("chofan", "chofa") is True

    # Plurales y singulares
    assert _token_match("salchicha", "salchichas") is True
    assert _token_match("tostada", "tostadas") is True

    # Alimentos distintos no coinciden
    assert _token_match("sancocho", "chofan") is False
    assert _token_match("arroz", "habichuela") is False


def test_bb046_segundo_plato_distinto_darian_sancocho_y_chofa():
    """El sancocho y el chofán son comidas distintas y no deben ser bloqueadas por el anti-clone guard."""
    user_text = "A las 12 me comí un plato de sancocho y a la noche un chofa"
    
    # Simular que el primer tool_call del turno ya registró el sancocho como almuerzo
    messages = [
        HumanMessage(content=user_text),
        AIMessage(content="Anoté el sancocho", tool_calls=[{
            "id": "tc1",
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Sancocho dominicano, porción regular",
                "meal_type": "almuerzo",
                "calories": 580,
                "protein": 29,
            }
        }]),
    ]

    # Segunda llamada para el chofán como cena
    chofan_args = {
        "meal_name": "Chofán dominicano",
        "meal_type": "cena",
        "calories": 650,
        "protein": 28,
    }

    es_distinto = _es_segundo_plato_distinto_del_turno(
        tool_args=chofan_args,
        messages=messages,
        user_text=user_text,
        current_tool_id="tc2"
    )
    assert es_distinto is True


def test_bb046_bloquea_clonacion_real_del_mismo_plato():
    """Si el usuario menciona una sola comida y el modelo intenta registrar el mismo plato dos veces, lo bloquea."""
    user_text = "Me comí un sancocho"
    messages = [
        HumanMessage(content=user_text),
        AIMessage(content="Anoté", tool_calls=[{
            "id": "tc1",
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Sancocho dominicano",
                "meal_type": "almuerzo",
                "calories": 580,
            }
        }]),
    ]
    # Intento de duplicar el mismo sancocho en otra franja
    duplicado_args = {
        "meal_name": "Sancocho dominicano",
        "meal_type": "cena",
        "calories": 580,
    }
    es_distinto = _es_segundo_plato_distinto_del_turno(
        tool_args=duplicado_args,
        messages=messages,
        user_text=user_text,
        current_tool_id="tc2"
    )
    assert es_distinto is False


# ==============================================================================
# BB-048: Supresión de Reclamos de Desayuno a Deshoras en la Noche
# ==============================================================================

def test_bb048_no_pregunta_por_desayuno_tarde_en_la_noche():
    """A las 10:55 PM (22.9h), no pregunta por el desayuno ni dice 'estuviste en ayunas'."""
    with patch("tools._diary_date_str_for_user", return_value="2026-11-05"), \
         patch("tools.user_tz_offset_min", return_value=0), \
         patch("tools._hora_local_float", return_value=22.9), \
         patch("chat_history_context.comidas_sin_registrar", return_value=["desayuno"]):
        
        # Simular llamada a las 10:55 PM (22.9) donde la única comida que falta es desayuno
        res = _nota_comidas_sin_registrar("u1", days_ago=0, ahora_local=22.9)
        assert "estuviste en ayunas" not in res
        assert "Hoy sigue sin registrar desayuno" not in res
        assert "el día ya cerró" in res


# ==============================================================================
# BB-051 / BB-052: Protocolo de Catch-up Nocturno Cálido y Supresión con Múltiples Faltantes
# ==============================================================================

def test_bb051_no_pregunta_por_desayuno_ni_merienda_con_multiples_faltantes_en_noche():
    """A las 10:55 PM, aun si faltaban desayuno y merienda en el plan, no interroga sobre ellos."""
    with patch("tools._diary_date_str_for_user", return_value="2026-11-05"), \
         patch("tools.user_tz_offset_min", return_value=0), \
         patch("tools._hora_local_float", return_value=22.9), \
         patch("db_facts.get_consumed_meals_today", return_value=[]), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("chat_history_context.comidas_sin_registrar", return_value=["desayuno", "merienda"]):
        
        res = _nota_comidas_sin_registrar("u1", days_ago=0, ahora_local=22.9)
        assert "estuviste en ayunas" not in res
        assert "qué comiste en el desayuno" not in res
        assert "sigue sin registrar" not in res
        assert "el día ya cerró" in res


def test_bb051_cierre_nocturno_si_ya_ceno():
    """Si la cena ya fue registrada en el día y son más de las 21:00, no insiste en comidas pasadas."""
    cena_row = {"meal_type": "cena", "meal_name": "Chofán", "calories": 650, "protein": 28}
    with patch("tools._diary_date_str_for_user", return_value="2026-11-05"), \
         patch("tools.user_tz_offset_min", return_value=0), \
         patch("tools._hora_local_float", return_value=22.9), \
         patch("db_facts.get_consumed_meals_today", return_value=[cena_row]), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("chat_history_context.comidas_sin_registrar", return_value=["desayuno", "almuerzo"]):
        
        res = _nota_comidas_sin_registrar("u1", days_ago=0, ahora_local=22.9)
        assert "estuviste en ayunas" not in res
        assert "qué comiste en el desayuno" not in res
        assert "sigue sin registrar" not in res
        assert "el día ya cerró" in res


def test_bb051_prompt_instrucciones_catchup_nocturno_calido():
    """El prompt del chat agent contiene la directiva BB-051 de empatía y prohibición de interrogatorio tardío."""
    from prompts.chat_agent import _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "[BB-051 / P1-CATCHUP-NOCTURNO-CALIDO" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Apertura empática y natural" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Supresión total de interrogatorios matutinos" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Cierre de descanso" in _REGLA_UNA_LLAMADA_POR_COMIDA
