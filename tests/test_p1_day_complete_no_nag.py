"""Regresión: cuando el día principal ya está registrado (Desayuno+Almuerzo+Cena) o la meta
calórica ya está cubierta (p. ej. 1956/2050 kcal a la 1:00 AM), el coach no debe preguntar
«¿Qué comiste hoy que todavía no esté anotado?» ni tratar la merienda opcional como pendiente.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import coach_day_context as cdc
from prompts.chat_agent import _CHAT_VOICE_RULES


_COMIDAS_DEL_DIA = [
    {"meal_type": "Desayuno", "calories": 86, "protein": 1, "carbs": 21, "fat": 0, "description": "Bowl de lechoza, sandía y piña"},
    {"meal_type": "Almuerzo", "calories": 630, "protein": 33, "carbs": 68, "fat": 24, "description": "Arroz blanco con cerdo guisado y vegetales"},
    {"meal_type": "Cena", "calories": 1240, "protein": 61, "carbs": 140, "fat": 45, "description": "4 sándwiches y jugo"},
]


def test_build_day_gap_context_madrugada_with_all_main_meals_forbids_nagging():
    out = cdc.build_day_gap_context(
        form_data={"goal": "maintain"},
        plan_vigente={"calories": 2050, "protein": 134, "carbs": 230, "fat": 68},
        consumed_today=_COMIDAS_DEL_DIA,
        hora_local=1.0,
    )
    assert "DÍA PRINCIPAL YA REGISTRADO" in out
    assert "MADRUGADA (jornada biológica en curso)" in out
    assert "PROHIBIDO preguntarle" in out
    assert "le falta REGISTRAR" not in out


def test_estado_de_las_comidas_noche_treats_merienda_as_optional_when_main_meals_logged():
    out = cdc.estado_de_las_comidas(_COMIDAS_DEL_DIA, 22.0)
    assert "merienda — SIN anotar y su hora ya pasó" in out
    assert "opcional; ya registró las 3 comidas principales" in out
    assert "NO preguntes por ella" in out


def test_estado_de_las_comidas_madrugada_contract_preserved():
    assert cdc.estado_de_las_comidas(_COMIDAS_DEL_DIA, 2.0) == ""


def test_today_remaining_context_does_not_claim_merienda_pending_at_95_pct():
    plan = {
        "calories": 2050,
        "days": [
            {
                "day": 1,
                "day_name": "Martes",
                "meals": [
                    {"meal": "Desayuno", "calories": 400},
                    {"meal": "Almuerzo", "calories": 650},
                    {"meal": "Merienda", "calories": 300},
                    {"meal": "Cena", "calories": 700},
                ],
            }
        ],
    }
    out = agent._build_today_remaining_context(
        plan,
        _COMIDAS_DEL_DIA,
        target_cal_int=2050,
        total_consumed=1956,
        local_date_str="2026-07-28",
    )
    assert "Hoy te quedan 1 comida(s) del plan" not in out
    assert "Ya registró sus comidas principales de hoy" in out
    assert "NO preguntes qué falta por anotar" in out


def test_chat_voice_rules_rule_p_forbids_asking_whats_unlogged_when_covered():
    assert "P. SI SOLO SALUDA" in _CHAT_VOICE_RULES
    assert "PROHIBIDO preguntar «¿qué comiste hoy?»" in _CHAT_VOICE_RULES
    assert "la próxima comida del plan POR SU NOMBRE" in _CHAT_VOICE_RULES
    assert "techo, no meta" in _CHAT_VOICE_RULES


def test_rechazar_autolog_disponibilidad_blocks_manzana_verde_after_tienes_a_mano():
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = [
        AIMessage(content="Si tienes hambre, elige algo pequeño y ligero; ¿qué alimentos tienes a mano?"),
        HumanMessage(content="Manzana verde"),
    ]
    res = agent._rechazar_autolog_disponibilidad(msgs, plate_photos=[])
    assert res is not None and "BLOQUEADO" in res


def test_rechazar_autolog_disponibilidad_allows_explicit_consumption_verb():
    from langchain_core.messages import AIMessage, HumanMessage

    msgs_eat = [
        AIMessage(content="Si tienes hambre, elige algo pequeño y ligero; ¿qué alimentos tienes a mano?"),
        HumanMessage(content="Me comí una manzana verde"),
    ]
    assert agent._rechazar_autolog_disponibilidad(msgs_eat, plate_photos=[]) is None

    msgs_log_q = [
        AIMessage(content="¿Qué cenaste hoy?"),
        HumanMessage(content="Manzana verde"),
    ]
    assert agent._rechazar_autolog_disponibilidad(msgs_log_q, plate_photos=[]) is None


def test_1_kcal_over_target_does_not_fire_critical_excess():
    out_rem = agent._build_today_remaining_context(
        None,
        consumed_today=[{"meal_type": "cena", "calories": 2051}],
        target_cal_int=2050,
        total_consumed=2051,
        local_date_str="2026-07-28",
    )
    assert "ALERTA CRÍTICA" not in out_rem
    assert "NO digas que «superaste tu meta»" in out_rem

    out_gap = cdc.build_day_gap_context(
        form_data={"goal": "maintain"},
        plan_vigente={"calories": 2050, "protein": 134, "carbs": 230, "fat": 68},
        consumed_today=[{"meal_type": "cena", "calories": 2051, "protein": 134}],
        hora_local=1.2,
    )
    assert "ya se pasó de su meta por ~1 kcal" not in out_gap
    assert "en rango, sin exceso" in out_gap


def test_night_cookies_380_kcal_with_prior_cena_classified_as_merienda_nocturna():
    from unittest.mock import patch
    from tools import log_consumed_meal

    mock_cena = {"id": "cena-01", "meal_type": "cena", "meal_name": "4 sándwiches", "calories": 1240}
    with patch("tools._hora_local_float", return_value=1.22), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._buscar_cena_principal_hoy", return_value=mock_cena), \
         patch("tools.db_log_consumed_meal", return_value="meal-cookies-1") as mock_db_log:
        res = log_consumed_meal.invoke({
            "user_id": "user-test-cookies",
            "meal_name": "4 galletas pequeñas caseras de avena con mantequilla de maní y miel",
            "calories": 380,
            "protein": 8,
            "carbs": 42,
            "healthy_fats": 18,
            "meal_type": "cena",
            "explicit_slot": False,
        })
        assert mock_db_log.call_args[1]["meal_type"] == "merienda"
        assert "merienda nocturna / extra de la noche" in res


def test_rechazar_autolog_sin_evidencia_blocks_tengo_hambre_and_phantom_clone():
    from langchain_core.messages import AIMessage, HumanMessage

    # Caso real 03:45 AM: usuario dice "Tengo hambre" y el LLM intenta clonar su almuerzo previo
    msgs_hambre = [
        AIMessage(content="no te convence ninguna opción 😊"),
        HumanMessage(content="Tengo hambre"),
    ]
    res_hambre = agent._rechazar_autolog_sin_evidencia(
        msgs_hambre,
        plate_photos=[],
        tool_args={"meal_name": "Arroz blanco con cerdo guisado y vegetales", "calories": 630},
        consumed_today=[{"meal_name": "Arroz blanco con cerdo guisado y vegetales"}],
    )
    assert res_hambre is not None and "BLOQUEADO" in res_hambre

    # Charlita sin consumo ni comida ("Soy tonto") tampoco puede disparar log_consumed_meal
    msgs_chat = [
        AIMessage(content="¿Qué tipo de almuerzo te apetece más: arroz, pasta o algo sin arroz?"),
        HumanMessage(content="Soy tonto"),
    ]
    assert agent._rechazar_autolog_sin_evidencia(msgs_chat, plate_photos=[]) is not None

    # Clonación fantasma del diario sin pedir repetición ni nombrar ese plato
    msgs_vague = [
        AIMessage(content="Otra opción para el almuerzo de hoy es moro de guandules."),
        HumanMessage(content="Mmm no sé"),
    ]
    assert agent._rechazar_autolog_sin_evidencia(
        msgs_vague,
        plate_photos=[],
        tool_args={"meal_name": "Arroz blanco con cerdo guisado y vegetales"},
        consumed_today=[{"meal_name": "Arroz blanco con cerdo guisado y vegetales"}],
    ) is not None

    # Consumo explícito SÍ pasa ("Tenía hambre y me comí 2 huevos", "Lo mismo de ayer")
    assert agent._rechazar_autolog_sin_evidencia(
        [HumanMessage(content="Tenía hambre y me comí 2 huevos")],
        plate_photos=[],
        tool_args={"meal_name": "2 huevos", "calories": 150},
    ) is None
    assert agent._rechazar_autolog_sin_evidencia(
        [HumanMessage(content="Me comí lo mismo de ayer")],
        plate_photos=[],
        tool_args={"meal_name": "Arroz blanco con cerdo guisado y vegetales"},
        consumed_today=[{"meal_name": "Arroz blanco con cerdo guisado y vegetales"}],
    ) is None
    # Confirmar pregunta directa de registro también pasa ("Sí" tras "¿Te comiste la lata de atún?")
    assert agent._rechazar_autolog_sin_evidencia(
        [AIMessage(content="¿Te comiste la lata de atún? Cuéntame en qué comida fue para anotártela."), HumanMessage(content="Sí, en el almuerzo")],
        plate_photos=[],
    ) is None

