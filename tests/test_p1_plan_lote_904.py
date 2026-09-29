"""[P1-PLAN-LOTE-904 · 2026-09-29] La guarda del diario ya no salta con el agua, los estados ni el ruido.

Modo voz del dueño (29-sep 19:14-19:15 UTC, leído del checkpoint del hilo): `P1-DIARY-CLAIM-VERIFY` saltó en dos de
tres turnos seguidos — otra llamada al modelo (~1 s) para escribir casi la misma frase:
  «Ah»       → «Te anoté un vaso de agua, ya llevas 1 de tus 9 del día.»
  «Hey hola» → «Con el desayuno anotado y un vaso de agua, lo que sigue es el almuerzo.»
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent
from diario_afirmacion import mensaje_sin_nada_que_registrar


def _salta(usuario: str, respuesta: str) -> bool:
    estado = {"messages": [HumanMessage(content=usuario), AIMessage(content=respuesta)], "diary_claim_retried": False}
    return agent.route_tools(estado) == "nudge_diary_tool"


# ── 1. Los dos casos reales ya no reintentan ─────────────────────────────────────────────────────────────────

def test_los_dos_turnos_del_duenio():
    assert not _salta("Ah", "Te anoté un vaso de agua, ya llevas 1 de tus 9 del día.\n\nCuando almuerces, cuéntame qué comiste y lo registro.")
    assert not _salta("Hey hola", "Con el desayuno anotado y un vaso de agua, lo que sigue es el almuerzo. Cuéntame qué comas y te lo registro.")


@pytest.mark.parametrize("frase", [
    "Te anoté un vaso de agua, ya llevas 3 de 8.",
    "Tu hidratación quedó registrada: 2 de 9.",
])
def test_el_agua_no_es_el_diario(frase):
    assert not agent._reply_claims_diary_write(frase)


@pytest.mark.parametrize("frase", [
    "Con el desayuno anotado, lo que sigue es el almuerzo.",
    "Ya tienes tu cena registrada, vas bien hoy.",
    "Con tus dos comidas registradas, te quedan 600 kcal.",
])
def test_un_estado_no_es_un_registro_de_este_turno(frase):
    assert not agent._reply_claims_diary_write(frase)


@pytest.mark.parametrize("texto", ["Ah", "mmm", "Eh", "Hey hola", "hola", "Cómo estás", ""])
def test_ruido_y_charla_corta(texto):
    assert mensaje_sin_nada_que_registrar(texto)


# ── 2. Lo que la guarda debe seguir cazando ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("frase", [
    "Cena registrada: 450 kcal.",
    "Te anoté un vaso de jugo de chinola, 120 kcal.",
    "Te anoté un vaso de agua y el mangú del desayuno.",
    "Listo, lo anoté como almuerzo.",
])
def test_una_afirmacion_de_comida_sigue_siendo_afirmacion(frase):
    assert agent._reply_claims_diary_write(frase)


def test_si_el_usuario_conto_una_comida_la_guarda_salta():
    assert _salta("me comí dos huevos con pan", "Listo, te anoté los huevos con pan en el desayuno.")


@pytest.mark.parametrize("texto", ["sí", "ok", "no", "ajá", "me comí un mangú", "pan con queso"])
def test_una_respuesta_o_una_comida_no_es_ruido(texto):
    assert not mensaje_sin_nada_que_registrar(texto)


def test_con_foto_no_se_da_por_charla():
    estado = {"messages": [HumanMessage(content=[{"type": "text", "text": "hola"}, {"type": "image_url", "image_url": {"url": "x"}}]),
                           AIMessage(content="Cena registrada: 450 kcal.")], "diary_claim_retried": False}
    assert agent.route_tools(estado) == "nudge_diary_tool"
