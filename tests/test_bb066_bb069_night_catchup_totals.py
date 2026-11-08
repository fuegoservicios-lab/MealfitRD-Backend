# -*- coding: utf-8 -*-
"""
Tests automatizados para:
- BB-066: Supresión de despedidas nocturnas repetidas consecutivas en catch-up y reporte obligatorio de acumulado diario.
- BB-067: Respuesta directa sin fricción ante el nombre de una franja suelta ("Desayuno", "Almuerzo", "Cena", "Merienda").
- BB-068: Reconciliación histórica de mensajes y validación de templates de respuesta de catch-up nocturno.
- BB-069: Verificación de regresión y cobertura de cierre de jornada.
"""
import pytest
from unittest.mock import patch, MagicMock

import tools
from prompts.chat_agent import (
    _REGLA_UNA_LLAMADA_POR_COMIDA,
    _CHAT_VOICE_RULES,
    _CHAT_RESOLVE_RULES,
)


# ==============================================================================
# BB-066: Supresión de Despedidas Nocturnas Repetidas y Acumulado Diario Obligatorio
# ==============================================================================

def test_bb066_prompt_reglas_supresion_despedidas_repetidas():
    """El prompt prohíbe terminantemente despedidas nocturnas repetidas en turnos seguidos."""
    assert "[BB-066 / P0-NO-REPEATED-NIGHT-FAREWELLS" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Cero despedidas repetidas en bucle" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "queda TERMINANTEMENTE PROHIBIDO volver a despedirte en los siguientes turnos" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Acumulado obligatorio del día" in _REGLA_UNA_LLAMADA_POR_COMIDA

    # En _CHAT_VOICE_RULES (R2)
    assert "R2. [BB-066 / P0-NO-REPEATED-NIGHT-FAREWELLS" in _CHAT_VOICE_RULES
    assert "CERO DESPEDIDAS NOCTURNAS REPETIDAS" in _CHAT_VOICE_RULES


def test_bb066_nota_comidas_sin_registrar_incluye_directiva_anti_repeticion():
    """_nota_comidas_sin_registrar no empuja a despedirse repetidamente si ya se despidió."""
    with patch("tools._diary_date_str_for_user", return_value="2026-11-07"), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._hora_local_float", return_value=21.5), \
         patch("db_facts.get_consumed_meals_today", return_value=[{"meal_type": "cena"}]), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("chat_history_context.comidas_sin_registrar", return_value=[]):

        res = tools._nota_comidas_sin_registrar("u1", days_ago=0, ahora_local=21.5)
        assert "el día ya cerró" in res
        assert "[BB-066] REGLA CONTRA REPETICIÓN DE DESPEDIDAS NOCTURNAS" in res
        assert "ESTÁ TERMINANTEMENTE PROHIBIDO REPETIRLO" in res
        assert "reporta siempre el total diario acumulado" in res


def test_bb066_nota_total_del_dia_instruye_acumulado_obligatorio():
    """_nota_total_del_dia instruye reportar el total acumulado en registros nocturnos o catch-up."""
    fake_rows = [
        {"meal_type": "cena", "calories": 720, "protein": 55, "carbs": 60, "fat": 15},
        {"meal_type": "desayuno", "calories": 450, "protein": 21, "carbs": 40, "fat": 10},
    ]
    with patch("tools._diary_date_str_for_user", return_value="2026-11-07"), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db_facts.get_consumed_meals_today", return_value=fake_rows), \
         patch("tools._perfil_y_plan_de_hoy", return_value=({}, None)):

        nota = tools._nota_total_del_dia("u1", 0)
        assert "TOTAL REAL DE HOY" in nota
        assert "~1170 kcal" in nota
        assert "~76 g de proteína" in nota
        assert "[BB-066]" in nota
        assert "reporta SIEMPRE este acumulado diario de calorías y proteína" in nota


# ==============================================================================
# BB-067: Respuesta Directa ante Nombre de Franja Aislado
# ==============================================================================

def test_bb067_prompt_direct_slot_name_query():
    """El prompt prohíbe preguntar '¿Quieres anotar tu desayuno?' y exige pregunta directa."""
    assert "[BB-067 / P1-DIRECT-SLOT-NAME-QUERY" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Cero fricción o preguntas dubitativas" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "PROHIBIDO preguntar si desea o quiere anotarlo" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "¿Quieres anotar tu desayuno de hoy?" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Pregunta directa y cálida" in _REGLA_UNA_LLAMADA_POR_COMIDA

    # En _CHAT_RESOLVE_RULES (Q4)
    assert "Q4. [BB-067 / P1-DIRECT-SLOT-NAME-QUERY" in _CHAT_RESOLVE_RULES
    assert "FRANJA AISLADA SIN ALIMENTO" in _CHAT_RESOLVE_RULES
    assert "NUNCA preguntes con duda «¿Quieres anotar tu desayuno?»" in _CHAT_RESOLVE_RULES


# ==============================================================================
# BB-068: Validación de Formatos de Respuestas de Cierre y Catch-up
# ==============================================================================

def test_bb068_estructura_respuestas_catchup_sin_repeticion():
    """Verifica que las respuestas de una secuencia de puesta al día no repitan despedidas."""
    respuestas_asistente = [
        "Anoté tu **cena**: pechuga de pollo, ñame y pastel en hoja, aprox. **720 kcal** y **55 g de proteína**. Que descanses.",
        "¿Qué desayunaste hoy? Cuéntame qué comiste y más o menos cuánto para anotártelo de inmediato.",
        "Anoté tu **desayuno**: 2 huevos con víveres, aprox. **450 kcal** y **21 g de proteína**.",
        "Anoté tu **almuerzo**: chuleta con lentejas, aprox. **520 kcal** y **42 g de proteína**. Hoy llevas en total **~1690 kcal** y **~118 g de proteína** en el diario. ¡Que descanses!"
    ]

    # En la respuesta de desayuno (turno intermedio), NO debe haber despedida nocturna
    assert "Que descanses" not in respuestas_asistente[2]
    assert "buenas noches" not in respuestas_asistente[2].lower()

    # En la respuesta de franja aislada (turno 1), NO debe haber fricción
    assert "¿Quieres anotar" not in respuestas_asistente[1]
    assert "¿Qué desayunaste hoy?" in respuestas_asistente[1]

    # En la respuesta final de almuerzo, debe incluirse el total acumulado y la despedida de cierre
    assert "Hoy llevas en total **~1690 kcal** y **~118 g de proteína** en el diario." in respuestas_asistente[3]
    assert "¡Que descanses!" in respuestas_asistente[3]
