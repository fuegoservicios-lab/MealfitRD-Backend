# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-694 · 2026-09-28] El registro de la foto deja de depender de que el modelo obedezca.

Caso vivo: «Mi cena» + foto → el coach propuso OTRA cena; y con las dudas contestadas terminó «¿Ya te los comiste?».
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import respuestas_de_la_foto as rf
from prompts.chat_agent import build_vision_context

_DESC = ("Plátano verde hervido en trozos con huevos revueltos y salami. (Estimación: Calorías: 550, Proteína: 20g, "
         "Carbohidratos: 57g, Grasas Saludables: 28g)")


def _vision(**extra):
    v = {"kind": "multi", "items": [{"kind": "plato", "description": _DESC}], "has_text": True}
    v.update(extra)
    return v


@pytest.mark.parametrize("texto,tipo", [
    ("Mi cena", "cena"), ("Esta fue mi cena", "cena"), ("el almuerzo de hoy", "almuerzo"), ("Cena", "cena"),
    ("Mi cena de anoche", "cena"), ("desayuno de ayer", "desayuno"), ("my dinner", "cena"), ("comida", ""),
])
def test_rotulos(texto, tipo):
    assert rf.rotulo_de_comida(texto) == tipo


@pytest.mark.parametrize("texto", [
    "¿Esto sirve para la cena?", "para la cena", "voy a cenar esto", "¿Qué tal mi cena?", "2 huevos · Maduro",
    "Mi cena\n2 huevos · Maduro", "", "hola", "esta cena tiene mucha grasa verdad",
])
def test_no_son_rotulos(texto):
    assert rf.rotulo_de_comida(texto) is None


def test_foto_para_anotar():
    assert rf.foto_para_anotar(_vision(), "Mi cena") is True
    assert rf.foto_para_anotar(_vision(respuestas="2 huevos · Maduro"), "Mi cena\n2 huevos · Maduro") is True
    assert rf.foto_para_anotar(_vision(), "¿esto está bien?") is False
    assert rf.foto_para_anotar({"kind": "multi", "items": [{"kind": "items", "description": "compra"}]}, "Mi cena") is False
    assert rf.foto_para_anotar(None, "Mi cena") is False


def test_el_rotulo_llega_al_contexto_de_la_foto():
    v = rf.marcar_rotulo(_vision(), "Mi cena")
    assert v["rotulo"] == "cena"
    ctx = build_vision_context(v)
    assert "RÓTULO DE LA FOTO" in ctx and "ESTO es su cena" in ctx and "No le propongas otro plato" in ctx
    assert "`meal_type` cena" in ctx
    # sin rótulo, nada cambia
    assert "RÓTULO" not in build_vision_context(rf.marcar_rotulo(_vision(), "¿esto está bien?"))
    # con respuestas manda su propia regla (no se marca rótulo)
    assert "rotulo" not in rf.marcar_rotulo(_vision(respuestas="2 huevos"), "Mi cena\n2 huevos")


def _estado(msgs, **kw):
    base = {"messages": msgs, "turn_plate_photos": [], "plate_photo_retried": False, "diary_claim_retried": True,
            "turn_photo_to_log": True, "photo_log_retried": False}
    base.update(kw)
    return base


def test_el_grafo_reintenta_una_vez_si_no_se_registro():
    from agent import route_tools, nudge_photo_to_log
    from langgraph.graph import END
    turno = [HumanMessage(content="Mi cena"), AIMessage(content="¿Ya te lo comiste o es lo que vas a cenar?")]
    assert route_tools(_estado(turno)) == "nudge_photo_to_log"
    out = nudge_photo_to_log(_estado(turno))
    assert out["photo_log_retried"] is True and "log_consumed_meal" in out["messages"][0].content
    # una sola vez
    assert route_tools(_estado(turno, photo_log_retried=True)) == END
    # si no era un turno de anotar, nada
    assert route_tools(_estado(turno, turn_photo_to_log=False)) == END


def test_con_el_registro_hecho_no_reintenta():
    from agent import route_tools
    from langgraph.graph import END
    turno = [
        HumanMessage(content="Mi cena"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_name": "Plátano con huevos"}, "id": "t1"}]),
        AIMessage(content="Anotada tu cena: ~550 kcal."),
    ]
    assert route_tools(_estado(turno)) == END


def test_el_estado_y_el_grafo_lo_declaran():
    import agent
    assert "turn_photo_to_log" in agent.ChatState.__annotations__
    assert "photo_log_retried" in agent.ChatState.__annotations__
    import inspect
    src = inspect.getsource(agent)
    assert 'chat_builder.add_edge("nudge_photo_to_log", "call_model")' in src
    assert '"turn_photo_to_log": _foto_para_anotar(vision, prompt),' in src
    assert "vision = marcar_rotulo(vision, prompt)" in src


def test_697_respuesta_escrita_pide_recalcular_y_la_tocada_registrar_tal_cual():
    """[P1-PLAN-LOTE-697] Batería en seco del 694, caso P5: con «4 huevos con 3 yemas» ESCRITO el modelo registró las
    550 kcal de la foto, porque la regla le decía que la estimación ya incluía las respuestas."""
    escrita = build_vision_context(_vision(respuestas="4 huevos con 3 yemas · Verde"))
    assert "es la de ANTES de sus respuestas: recalcula" in escrita and "YA incluye sus respuestas" not in escrita
    tocada = build_vision_context(_vision(respuestas="2 huevos · Maduro", ajuste={"calories": 18, "carbs": 5}))
    assert "YA incluye sus respuestas: registra esas cifras" in tocada and "Calorías: 568" in tocada
    # con varias fotos de plato el ajuste no se aplica: tampoco se afirma que esté incluido
    v = _vision(respuestas="2 huevos", ajuste={"calories": 90})
    v["items"].append({"kind": "plato", "description": _DESC})
    assert "recalcula" in build_vision_context(v)
    assert "ya incluye sus respuestas" not in rf.NOTA_REINTENTO
