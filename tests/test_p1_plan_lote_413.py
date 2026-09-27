# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-413 · 2026-09-27] Los avisos de comida hablan del día REAL y dejan de sonar a plantilla.

El dueño: «los mensajes genéricos del agente, ¿puedes hacerlos más versátiles y que se sientan más humanos e
inteligentes?». Los avisos que escribe la IA salían así: «Es el momento perfecto para desayunar: arma un plato
balanceado con proteína, fibra y fruta para apoyar tu objetivo de ganar músculo…», «Aprovecha para almorzar ahora…
para no frenar tu objetivo de ganar músculo. ¿Qué está fallando con este almuerzo: el tiempo, las opciones…?».
Dos causas en el prompt: recibía el objetivo sin más (y lo metía en cada aviso) y no sabía NADA del día.

Ahora el aviso recibe un bloque con lo que la persona lleva hoy (qué registró, calorías y proteína contra su meta,
cuando el perfil alcanza para calcularla) y reglas de estilo: el objetivo solo de pasada, nada de interrogatorios,
arranque variado, tono de amigo nutricionista por WhatsApp.
Tooltip-anchor: P1-PLAN-LOTE-413
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

_PERFIL = {
    "gender": "male", "age": 30, "height": 175, "weight": 80, "weightUnit": "kg",
    "activityLevel": "moderate", "mainGoal": "gain_muscle", "medicalConditions": ["none"],
}
_DESAYUNO = {"meal_type": "desayuno", "meal_name": "Omelet de huevo", "calories": 990, "protein": 59, "carbs": 64, "healthy_fats": 54}


def test_el_bloque_dice_lo_que_lleva_hoy_con_su_meta():
    import aviso_del_dia as ad
    b = ad.bloque_del_dia([_DESAYUNO], _PERFIL)
    assert "desayuno (990 kcal)" in b
    assert "990 de " in b and "le quedan" in b
    assert "59 de " in b and "le faltan" in b
    assert "no los recites todos" in b


def test_sin_registros_lo_dice():
    import aviso_del_dia as ad
    b = ad.bloque_del_dia([], _PERFIL)
    assert "Todavía no registró nada hoy." in b


def test_sin_perfil_suficiente_no_inventa_metas():
    import aviso_del_dia as ad
    b = ad.bloque_del_dia([_DESAYUNO], {"mainGoal": "gain_muscle"})
    assert "desayuno (990 kcal)" in b
    assert " de " not in b.split("desayuno (990 kcal)")[1]      # sin «X de Y»: no hay meta que comparar
    assert ad.bloque_del_dia(None, None).startswith("\nLo que lleva hoy")


def test_el_prompt_pide_sonar_humano_y_no_repetir_el_objetivo():
    from prompts.proactive import PROACTIVE_PROMPT
    assert "amigo nutricionista" in PROACTIVE_PROMPT
    assert "NO cierres con «para apoyar tu objetivo" in PROACTIVE_PROMPT
    assert "¿qué está fallando?" in PROACTIVE_PROMPT
    assert "Empieza de forma distinta" in PROACTIVE_PROMPT
    # sin huecos nuevos: los llamadores que formatean con la lista fija de siempre siguen valiendo
    PROACTIVE_PROMPT.format(missing_meal="Almuerzo", verbo="almorzaste", infinitivo="almorzar", trigger_time="12:45 PM",
                            diet_type="balanceada", goals="ganar músculo", tone_instruction="", style_instruction="")


def test_el_bucle_anade_el_bloque_del_dia():
    src = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
    i = src.index("prompt = PROACTIVE_PROMPT.format(")
    assert "prompt += bloque_del_dia(consumed, health)" in src[i:i + 1500]
