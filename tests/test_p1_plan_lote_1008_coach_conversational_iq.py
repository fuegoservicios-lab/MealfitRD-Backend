"""Regresion P1: identidad Bioboros, ruido/comandos, anti-eco y cierre «ya comi» del coach."""

from __future__ import annotations

import os
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import agent  # noqa: E402
import diario_afirmacion as da  # noqa: E402
from prompts import chat_agent as ca  # noqa: E402


PROMPT_VARIANTS = [
    ("CHAT_AGENT_INLINE_PROMPT", ca.CHAT_AGENT_INLINE_PROMPT),
    ("CHAT_STREAM_INLINE_PROMPT", ca.CHAT_STREAM_INLINE_PROMPT),
    ("CHAT_SYSTEM_PROMPT_BASE", ca.CHAT_SYSTEM_PROMPT_BASE),
    ("CHAT_STREAM_SYSTEM_PROMPT_BASE", ca.CHAT_STREAM_SYSTEM_PROMPT_BASE),
    ("CHAT_VOICE_MODE_PROMPT", ca.CHAT_VOICE_MODE_PROMPT),
]


@pytest.mark.parametrize("name,prompt", PROMPT_VARIANTS, ids=[p[0] for p in PROMPT_VARIANTS])
def test_prompts_incluyen_identidad_bioboros_y_reglas_conversacionales(name: str, prompt: str):
    """Todos los prompts del coach incluyen identidad Bioboros, anti-eco/ruido y «ya comí»."""
    assert "IDENTIDAD BIOBOROS" in prompt, f"{name} perdió R3"
    assert "NUNCA menciones a OpenAI" in prompt, f"{name} no prohíbe atribuirse OpenAI"
    assert "RUIDO, CHARLA Y ANTI-ECO" in prompt, f"{name} perdió R4"
    assert "«YA COMÍ» SIN ALIMENTO NUEVO" in prompt, f"{name} perdió Q5"


@pytest.mark.parametrize(
    "texto",
    [
        "uuuuu",
        "aaaaaa",
        "/fluidez",
        "/help",
        "Todo lo que registre hoy fue lo que comi",
        "todo lo que registré hoy fue lo que comí",
        "ya no comí nada más",
        "eso fue todo lo que comí",
        "ya está todo anotado",
        "Pss ya comi",
        "ya comí",
    ],
)
def test_sin_nada_que_registrar_filtra_ruido_slash_y_cierres(texto: str):
    """Ruido repetido, /comandos, «ya comí» sin plato y cierres de diario no disparan nudge."""
    assert da.mensaje_sin_nada_que_registrar(texto) is True


@pytest.mark.parametrize(
    "texto",
    [
        "sí",
        "ok",
        "no",
        "ajá",
        "me comí un mangú",
        "ya comí arroz con salmón",
        "pan con queso",
        "/fluidez me comi dos huevos",
    ],
)
def test_sin_nada_que_registrar_preserva_afirmaciones_y_comidas(texto: str):
    """Invariante Lote 904: respuestas breves ('sí', 'ok', 'ajá') y comidas reales siguen protegidas."""
    assert da.mensaje_sin_nada_que_registrar(texto) is False


@pytest.mark.parametrize(
    "texto",
    [
        "uuuuu",
        "Bien",
        "Magnifico",
        "Magnífico!!",
        "/fluidez",
        "Pss ya comi",
        "Todo lo que registre hoy fue lo que comi",
        "ok",
        "sí",
    ],
)
def test_rag_es_casual_evita_rag_y_sentimiento_en_charla_ruido_y_slash(texto: str):
    """Ruido, /comandos y elogios cortos saltan sentiment_classifier y RAG."""
    assert agent._rag_es_casual(texto) is True


@pytest.mark.parametrize(
    "texto",
    [
        "Quien es tu creador",
        "me comí un mangú con queso",
        "cuántas calorías tiene el salmón",
    ],
)
def test_rag_es_casual_no_oculta_preguntas_ni_comidas_reales(texto: str):
    """Preguntas sustantivas o comidas reales mantienen su flujo completo."""
    assert agent._rag_es_casual(texto) is False


@pytest.mark.parametrize(
    "user_msg,ai_reply",
    [
        (
            "/fluidez",
            "La comida de hoy que ya registraste sigue anotada; no agregué ningún alimento nuevo. Hoy llevas **3114 kcal**.",
        ),
        (
            "/fluidez",
            "Sí, entendido: lo que registraste hoy fue lo que comiste. No agregué nada más.",
        ),
        (
            "Todo lo que registre hoy fue lo que comi",
            "Entendido: todo lo que registraste hoy quedó anotado tal como lo comiste.",
        ),
        (
            "Pss ya comi",
            "Tu cena de hoy ya está anotada.",
        ),
        (
            "Pss ya comi",
            "Ya tienes anotada tu cena de hoy.",
        ),
        (
            "Pss ya comi",
            "Tu cena de hoy sigue anotada con 595 kcal.",
        ),
    ],
)
def test_route_tools_no_dispara_falso_nudge_en_estado_previo_o_slash(user_msg: str, ai_reply: str):
    """'sigue anotada', /comandos y 'Pss ya comi' sin comida no deben activar nudge_diary_tool."""
    state = {"messages": [HumanMessage(content=user_msg), AIMessage(content=ai_reply)]}
    assert agent.route_tools(state) == "__end__"


def test_adelgazar_preserva_reglas_identidad_y_conversacion():
    """El podador lean no debe eliminar R3, R4 ni Q5 del prefijo estático."""
    from prompts.coach_lean import adelgazar

    static_full = ca.CHAT_STREAM_INLINE_PROMPT + "\n\n" + ca.build_tools_instructions_stream_estaticas()
    lean = adelgazar(static_full, ["add_meal_to_diary", "update_meal_in_diary", "update_form_field"])
    assert "IDENTIDAD BIOBOROS" in lean
    assert "NUNCA menciones a OpenAI" in lean
    assert "RUIDO, CHARLA Y ANTI-ECO" in lean
    assert "«YA COMÍ» SIN ALIMENTO NUEVO" in lean


def test_route_tools_sigue_protegiendo_comida_real_sin_tool():
    """Cuando el usuario sí reporta un plato y el modelo finge anotarlo, el nudge sigue activo."""
    state = {
        "messages": [
            HumanMessage(content="me comí un mangú"),
            AIMessage(content="Listo, anoté tu mangú en el desayuno."),
        ]
    }
    assert agent.route_tools(state) == "nudge_diary_tool"
