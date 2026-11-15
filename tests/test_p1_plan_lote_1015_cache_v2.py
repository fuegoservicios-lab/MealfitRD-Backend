"""[P1-PLAN-LOTE-1015 · A-01] Caché v2: bloque de sistema único y estable por usuario, contexto del turno en el mensaje de usuario,
catálogo de herramientas fijo con `allowed_tools`.

Sondas contra la API real (14-nov, gpt-6-luna, ~US$ 0,02 en total) que fijan el diseño:
  · un 2.º mensaje system/developer tras el bloque de sistema ⇒ 0 tokens en caché (incluso con herramientas y cebador);
  · bloque de sistema idéntico + mensaje de usuario con el contexto ⇒ 9.544 de 9.575 SIN cebador, también en el turno siguiente;
  · `tool_choice.allowed_tools` no forma parte de lo cacheado.
"""
from __future__ import annotations

import os
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MANUAL = "MANUAL ESTATICO DEL COACH. " * 50


def _prompt(*, hora="10:01", diario="vacío", clinico="Alergias: maní", nombre="Angelo", meta=2050, uid="U1", idioma=True):
    from agent import _VOLATILE_SYS_SPLIT_MARKER as M

    cola = f"El user_id actual es: {uid}"
    cola += f"\n\n🕒 CONTEXTO TEMPORAL ACTUAL: son las {hora}"
    cola += "\n⚠️ RITMO CIRCADIANO: mañana"
    cola += f"\n\n--- 👤 PERFIL DEL USUARIO\n- Nombre: {nombre}"
    cola += f"\n\n🛑 PERFIL CLÍNICO DEL USUARIO\n{clinico}"
    cola += f"\n\n🎯 META DIARIA: {meta} kcal"
    if idioma:
        cola += "\n\n🌐 RESPONSE LANGUAGE — NON-NEGOTIABLE: Write your ENTIRE reply in English."
    cola += f"\n\nDIARIO DE HOY: {diario}"
    cola += "\n\n🧊 NEVERA FÍSICA AHORA (2 items): huevos 6, pan 1"
    return MANUAL + M + cola, M


def test_reordenar_mueve_solo_lo_estable_y_conserva_cada_caracter():
    from coach_capa_estable import reordenar

    p, M = _prompt()
    r = reordenar(p, M)
    assert len(r) == len(p) and sorted(r) == sorted(p)          # solo cambia el orden de tramos
    estatico, volatil = r.split(M, 1)
    for estable in ("El user_id actual es: U1", "PERFIL CLÍNICO DEL USUARIO", "Alergias: maní", "PERFIL DEL USUARIO", "META DIARIA: 2050",
                    "RESPONSE LANGUAGE"):
        assert estable in estatico, estable
    for volatil_ in ("CONTEXTO TEMPORAL", "RITMO CIRCADIANO", "DIARIO DE HOY", "NEVERA FÍSICA"):
        assert volatil_ in volatil and volatil_ not in estatico, volatil_


def test_el_bloque_de_sistema_es_identico_entre_turnos_y_distinto_entre_usuarios():
    from coach_capa_estable import partes, reordenar

    a, M = _prompt(hora="10:01", diario="vacío")
    b, _ = _prompt(hora="14:35", diario="2 huevos, pan")
    sa, va = partes(reordenar(a, M), M)
    sb, vb = partes(reordenar(b, M), M)
    assert sa == sb and va != vb                                # lo que cambia por turno NO toca el bloque de sistema
    c, _ = _prompt(uid="U2", nombre="Otra", clinico="Condición: diabetes")
    assert partes(reordenar(c, M), M)[0] != sa                  # otro usuario ⇒ otro bloque (propio, estable)
    d, _ = _prompt(clinico="Alergias: maní, mariscos")          # un cambio de perfil SÍ lo cambia (y es lo correcto)
    assert partes(reordenar(d, M), M)[0] != sa


def test_reordenar_es_idempotente_y_no_toca_un_prompt_sin_marcador_ni_bloques():
    from coach_capa_estable import reordenar

    p, M = _prompt()
    r = reordenar(p, M)
    assert reordenar(r, M) == r
    assert reordenar("sin marcador", M) == "sin marcador" and reordenar("", M) == "" and reordenar(None, M) is None
    solo_volatil = MANUAL + M + "El user_id actual es: U1\n\n🕒 CONTEXTO TEMPORAL ACTUAL: x\n\nDIARIO DE HOY: y"
    assert reordenar(solo_volatil, M) == solo_volatil


def test_el_bloque_clinico_queda_en_rol_sistema_con_cache_v2():
    import agent
    from coach_capa_estable import reordenar

    p, M = _prompt()
    msgs = agent._llm_messages_from_state([HumanMessage(content="hola")], reordenar(p, M), forzar_split=True)
    assert isinstance(msgs[0], SystemMessage)
    assert "PERFIL CLÍNICO DEL USUARIO" in msgs[0].content and "Alergias: maní" in msgs[0].content
    assert "CONTEXTO TEMPORAL" not in msgs[0].content
    # lo volátil viaja en el mensaje de usuario del turno, nunca en un 2.º mensaje de sistema
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1
    ultimo = msgs[-1]
    assert isinstance(ultimo, HumanMessage) and "CONTEXTO TEMPORAL" in ultimo.content and "hola" in ultimo.content


def test_sin_cache_v2_el_armado_de_mensajes_es_el_de_siempre():
    import agent

    p, M = _prompt()
    legacy = agent._llm_messages_from_state([HumanMessage(content="hola")], p)
    assert isinstance(legacy[0], SystemMessage) and M in legacy[0].content     # el marcador sigue dentro (knob de split apagado)
    assert sum(isinstance(m, SystemMessage) for m in legacy) == 1


def test_el_turno_siguiente_repite_el_mismo_bloque_de_sistema_con_el_historial_creciendo():
    import agent
    from coach_capa_estable import reordenar

    a, M = _prompt(hora="10:01")
    b, _ = _prompt(hora="11:30", diario="1 desayuno")
    m1 = agent._llm_messages_from_state([HumanMessage(content="anota un vaso")], reordenar(a, M), forzar_split=True)
    m2 = agent._llm_messages_from_state(
        [HumanMessage(content="anota un vaso"), AIMessage(content="Listo"), HumanMessage(content="¿cuánto llevo?")],
        reordenar(b, M), forzar_split=True)
    assert m1[0].content == m2[0].content


# ── herramientas ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_allowed_tools_en_el_orden_del_catalogo_y_none_si_es_todo():
    from coach_capa_estable import nombres_de_herramientas_permitidas as perm, tool_choice_permitidas as tcp

    class T:
        def __init__(self, n): self.name = n

    todas = [T(n) for n in ("a", "b", "c", "d")]
    assert perm([T("d"), T("b")], todas) == ["b", "d"]
    assert perm(todas, todas) is None and perm([], todas) is None
    assert tcp(["b", "d"]) == {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [
        {"type": "function", "function": {"name": "b"}}, {"type": "function", "function": {"name": "d"}}]}}


def test_la_carga_http_lleva_las_19_herramientas_y_el_tool_choice(monkeypatch):
    import agent
    import llm_provider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    llm = llm_provider.ChatGLM(model="gpt-6-luna", temperature=0.7, timeout=5)
    state = {"messages": [HumanMessage(content="me comí 2 huevos con pan")]}
    sel = agent._select_tools_for_turn(agent.agent_tools, state)
    from coach_capa_estable import nombres_de_herramientas_permitidas as perm, tool_choice_permitidas as tcp

    nombres = perm(sel, agent.agent_tools)
    assert nombres and "update_form_field" in nombres and len(nombres) < len(agent.agent_tools)
    b = llm.bind_tools(list(agent.agent_tools)).bind(tool_choice=tcp(nombres))
    payload = llm._get_request_payload([SystemMessage("s"), HumanMessage("h")], **b.kwargs)
    assert len(payload["tools"]) == len(agent.agent_tools)
    assert [t["function"]["name"] for t in payload["tools"]] == [t.name for t in agent.agent_tools]   # catálogo y orden fijos
    assert payload["tool_choice"]["type"] == "allowed_tools"
    assert [t["function"]["name"] for t in payload["tool_choice"]["allowed_tools"]["tools"]] == nombres


def test_el_catalogo_es_byte_identico_entre_turnos_distintos(monkeypatch):
    import json

    import agent
    import llm_provider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    llm = llm_provider.ChatGLM(model="gpt-6-luna", temperature=0.7, timeout=5)
    from coach_capa_estable import nombres_de_herramientas_permitidas as perm, tool_choice_permitidas as tcp

    cargas = []
    for texto in ("me comí 2 huevos con pan", "¿cuánta proteína me falta hoy?", "tengo diabetes"):
        sel = agent._select_tools_for_turn(agent.agent_tools, {"messages": [HumanMessage(content=texto)]})
        n = perm(sel, agent.agent_tools)
        b = llm.bind_tools(list(agent.agent_tools))
        if n:
            b = b.bind(tool_choice=tcp(n))
        cargas.append(json.dumps(llm._get_request_payload([SystemMessage("s"), HumanMessage(texto)], **b.kwargs)["tools"], sort_keys=True))
    assert len(set(cargas)) == 1


# ── canario y decisión ───────────────────────────────────────────────────────────────────────────────────────────
def test_el_canario_decide_y_anota_el_brazo(monkeypatch):
    import agent
    from chat_phase_metrics import ChatTurnPhaseTracker

    monkeypatch.setattr(agent, "_chat_agent_model_name", lambda uid=None: "gpt-6-luna")
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_USER_IDS", "u-dueno")
    t = ChatTurnPhaseTracker(user_id="u-dueno", session_id="s", corr="abc12345")

    monkeypatch.delenv("MEALFIT_CHAT_CANARY_CACHE_V2", raising=False)
    assert agent._cache_v2_para_turno("u-dueno", tracker=t) is False                         # off por defecto
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CACHE_V2", "canary")
    assert agent._cache_v2_para_turno("u-dueno", tracker=t) is True
    assert agent._cache_v2_para_turno("u-otro", tracker=t) is False
    assert agent._cache_v2_para_turno("u-dueno", modo_voz=True, tracker=t) is False          # voz: otra pila
    monkeypatch.setattr(agent, "_chat_agent_model_name", lambda uid=None: "gemini-4-hydrogen")
    assert agent._cache_v2_para_turno("u-dueno", tracker=t) is False                          # solo OpenAI
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CACHE_V2", "on")
    monkeypatch.setattr(agent, "_chat_agent_model_name", lambda uid=None: "gpt-6-luna")
    assert agent._cache_v2_para_turno("u-otro", tracker=t) is True
    assert "cache_v2" in t.to_metadata()["arm"]


def test_adelgazar_con_cache_v2_da_el_mismo_manual_en_todos_los_turnos():
    import agent

    base = "preámbulo\n" + "- `log_consumed_meal`: registra\n- `analizar_suplemento`: suplementos\n- `cambiar_ajuste_de_la_app`: ajustes\n"
    a = agent._adelgazar_prompt_coach(base, "me comí 2 huevos con pan", cache_v2=True)
    b = agent._adelgazar_prompt_coach(base, "activa el modo oscuro y mis suplementos", cache_v2=True)
    assert a == b


# ── cableado ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _src():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")


def test_los_dos_constructores_y_el_estado_estan_cableados():
    s = _src()
    assert s.count("_cache_v2 = _cache_v2_para_turno(") == 2
    assert s.count("_split_volatile = _chat_split_volatile_sys_prompt() or _cache_v2") == 2
    assert s.count("_cache_v2 = False   # [P1-PLAN-LOTE-1015 · A-01] definido ANTES del try") == 2
    assert s.count('"cache_v2": _cache_v2,') == 2
    assert s.count("system_prompt = _reordenar_capa(system_prompt, _VOLATILE_SYS_SPLIT_MARKER)") == 2
    assert 'forzar_split=bool(state.get("cache_v2"))' in s


def test_el_turno_escalado_a_terra_no_usa_allowed_tools():
    """La forma de `allowed_tools` en la Responses API es otra y no está verificada: terra conserva el subconjunto de siempre."""
    s = _src()
    assert 'if state.get("cache_v2") and not _hard_info.get("escalated"):' in s
