"""[P1-DIARY-CLAIM-PERFECTIVE · 2026-09-15] El «hola» del dueño que recibió «Aclarado: no registré…».

Checkpoint real (sesión 734a5820):
  Human «hola» · Human «hola» (sembrado dos veces) · AI saludo con el menú que termina en
  «¿En qué te ayudo — con el plan, registro de comidas o tu lista de compras?» · System nudge ·
  AI «Aclarado: no registré ni hay nada pendiente de registrar — solo saludaste 😄…».

Tres defectos encadenados:
  1. el detector tomaba el SUSTANTIVO «registro» por una afirmación de registro;
  2. el nudge (visible desde P1-DIARY-NUDGE-VISIBLE) no le decía al modelo que su respuesta
     nueva SUSTITUYE a la anterior, y el modelo le contestó a la nota;
  3. el texto final sumaba las dos pasadas y el filtro de deliberación (>300 chars) se comía
     el saludo, dejando solo la aclaración.
Y de paso: el hilo nuevo sembraba el mensaje del usuario dos veces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import agent as A  # noqa: E402

SALUDO = (
    "¡Qué tal, Angelo! ¿Cómo vas? Hoy es martes y el plan te trae un día bastante sabroso: "
    "**avena con leche evaporada, maní y queso cottage** para arrancar, **locrio de pescado "
    "blanco** de almuerzo y **salami guisado con yuca y camarones** de cena.\n\n¿En qué te "
    "ayudo — con el plan, registro de comidas o tu lista de compras?"
)


@pytest.mark.parametrize("texto", [
    SALUDO,
    "Cuando me digas que comiste algo, lo anoto en el diario de inmediato.",
    "¿Quieres que lo deje registrado como cena?",
    "¿Quieres que lo anote?",
    "Puedo registrar tu almuerzo si me dices qué comiste.",
    "No pude registrarlo todavía.",
])
def test_no_es_una_afirmacion_de_registro(texto):
    assert A._reply_claims_diary_write(texto) is False, texto


@pytest.mark.parametrize("texto", [
    "Cena registrada. Asumo 2 panes de molde con queso.",
    "Listo, registré tu almuerzo: 520 kcal.",
    "Quedó anotada como tu cena de hoy.",
    "Te lo apunté en el diario.",
])
def test_si_es_una_afirmacion_de_registro(texto):
    assert A._reply_claims_diary_write(texto) is True, texto


def test_el_saludo_del_dueno_no_dispara_el_nudge():
    estado = {"messages": [HumanMessage(content="hola"), AIMessage(content=SALUDO)]}
    assert A.route_tools(estado) != "nudge_diary_tool"


def test_tras_un_nudge_la_respuesta_nueva_sustituye_a_la_anterior():
    msgs = [
        HumanMessage(content="cené pan"),
        AIMessage(content="Cena registrada. Asumo 2 panes."),
        SystemMessage(content="ALTO. Acabas de afirmar…"),
        AIMessage(content="Te lo anoto ahora mismo si me confirmas cuántos panes fueron."),
    ]
    final = A._build_final_content_from_messages(msgs)
    assert "Cena registrada" not in final, "la respuesta rechazada por el guard sigue en el final"
    assert "si me confirmas" in final


def test_sin_nudge_la_narracion_sigue_sumandose():
    """Anti-oscilación con P1-CHAT-NARRATION-KEPT."""
    msgs = [HumanMessage(content="x"), AIMessage(content="Lo anoto y te digo."),
            AIMessage(content="Listo, quedó anotado.")]
    final = A._build_final_content_from_messages(msgs)
    assert "Lo anoto" in final and "quedó anotado" in final


def test_el_nudge_dice_que_la_nota_es_interna_y_que_se_sustituye():
    texto = A.nudge_diary_tool({"user_id": "u"})["messages"][0].content
    assert "no la menciones" in texto and "sustituye" in texto


def test_el_hilo_nuevo_no_siembra_el_mensaje_dos_veces():
    recent = [{"role": "model", "content": "hola de antes"}, {"role": "user", "content": "hola"}]
    msgs = A._seed_thread_messages(recent, "hola")
    assert [m.content for m in msgs].count("hola") == 1
    # si el prompt es otro, sí se añade
    assert A._seed_thread_messages(recent, "¿y mi cena?")[-1].content == "¿y mi cena?"


def test_el_stream_vacia_el_buffer_en_una_segunda_pasada_sin_tool():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "if _step is not None and _buf_step is not None and _step != _buf_step:" in src
    assert src.count("_seed_thread_messages(memory[\"recent_messages\"], prompt)") == 2
