# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1171 · 2026-11-23] El día real de un usuario (21-nov): el coach contestó en inglés a un mensaje en
español, y a las 00:29 anotó su bistec como «almuerzo de ayer» sin que él nombrara ningún día."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biological_date_resolver import corregir_days_ago_madrugada
from idioma_del_turno import idioma_del_mensaje, nota_de_idioma


def test_idioma_de_los_mensajes_reales():
    assert idioma_del_mensaje("agrega al desayuno 4 huevos, hervidos, 4 rodajas de pan con mantequilla de mané y queso cottage") == "es"
    assert idioma_del_mensaje("whats the name of the yogurt I sent u sometimes that has 25 grams of protein") == "en"
    assert idioma_del_mensaje("Solo con leche descremada / ligero · Bistec de res / filete") == "es"
    # sin idioma claro no se dice nada: manda la regla F
    for t in ("ok", "Grilled chicken breast", "Mangú", "", "Add it on yesterday"):
        assert idioma_del_mensaje(t) in (None, "en")
    assert idioma_del_mensaje("ok") is None


def test_la_nota_va_junto_al_ultimo_mensaje(monkeypatch):
    import agent
    msgs = [
        HumanMessage(content="whats the name of the yogurt I sent you"),
        AIMessage(content="It's Ratio Protein yogurt."),
        HumanMessage(content="agrega al desayuno 4 huevos con pan y queso cottage"),
    ]
    out = agent._llm_messages_from_state(msgs, "SISTEMA")
    ultimo = out[-1]
    assert isinstance(ultimo, HumanMessage)
    assert "español" in ultimo.content and "agrega al desayuno" in ultimo.content
    # el primer mensaje (inglés) no se toca
    assert out[1].content == "whats the name of the yogurt I sent you"
    monkeypatch.setenv("MEALFIT_COACH_IDIOMA_DEL_TURNO", "false")
    assert nota_de_idioma(msgs[-1]) is None


def test_madrugada_sin_dia_nombrado_queda_en_la_jornada_abierta():
    msgs = [HumanMessage(content="Solo con leche descremada / ligero · Bistec de res / filete")]
    assert corregir_days_ago_madrugada(msgs, 1, hora_local=0.5, slot_pedido="almuerzo") == 0
    # fuera de la madrugada no se toca
    assert corregir_days_ago_madrugada(msgs, 1, hora_local=18.0, slot_pedido="almuerzo") == 1


def test_madrugada_con_otro_dia_dicho_o_preguntado_se_respeta():
    dicho = [HumanMessage(content="esto fue hace 2 días")]
    assert corregir_days_ago_madrugada(dicho, 2, hora_local=1.0) == 2
    preguntado = [
        HumanMessage(content="me comí un sándwich de pollo"),
        AIMessage(content="¿Fue ayer o hoy?"),
        HumanMessage(content="sí"),
    ]
    assert corregir_days_ago_madrugada(preguntado, 1, hora_local=1.0) == 1


def test_sin_mensajes_humanos_no_rompe():
    assert agent_ok([SystemMessage(content="x")])


def agent_ok(msgs):
    import agent
    return isinstance(agent._llm_messages_from_state(msgs, "S"), list)
