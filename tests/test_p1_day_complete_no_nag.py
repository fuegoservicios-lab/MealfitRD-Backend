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
