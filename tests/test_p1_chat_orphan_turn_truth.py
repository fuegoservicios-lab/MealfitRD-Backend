"""[P1-CHAT-ORPHAN-TURN-TRUTH · 2026-09-03] Un turno del coach murió en el servidor (timeout del
modelo tras el reintento de claim-verify); el usuario recargó y el cliente se quedó
«Recuperando tu respuesta…» sondeando /history hasta 30 veces (~4 min) por una respuesta que
el servidor ya sabía que no vendría. Ahora el backend registra los turnos vivos por proceso y
/history devuelve `turn_active`; el cliente abandona en el primer sondeo y ofrece reintentar.
De paso, el timeout por intento del LLM del chat sube de 15 s a 30 s (GLM + tool + reintento
superaba los 15 s y agotaba los 3 intentos del cliente OpenAI).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_ROUTER = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


def test_registro_de_turnos_vivos_con_tope_de_edad():
    assert "_ACTIVE_TURNS: dict[str, float] = {}" in _AGENT
    assert "_ACTIVE_TURN_MAX_AGE_S = 300.0" in _AGENT
    assert "def is_turn_active(session_id: str) -> bool:" in _AGENT
    # zombi: una entrada más vieja que el tope no cuenta como viva y se retira
    body = _AGENT[_AGENT.index("def is_turn_active("):_AGENT.index("def chat_with_agent_stream(")]
    assert "_ACTIVE_TURN_MAX_AGE_S" in body and "_ACTIVE_TURNS.pop(session_id, None)" in body


def test_el_stream_registra_al_entrar_y_retira_en_el_finally():
    i = _AGENT.index("def chat_with_agent_stream(")
    head = _AGENT[i:i + 1500]
    assert "_ACTIVE_TURNS[session_id] = time.time()" in head, "el turno no se registra al entrar"
    # el pop vive en el `finally` del stream (todos los exits: normal, excepción, GeneratorExit)
    m = re.search(r"\n    finally:\r?\n        _ACTIVE_TURNS\.pop\(session_id, None\)", _AGENT)
    assert m, "el finally del stream no retira el turno"
    assert m.start() > i


def test_history_devuelve_turn_active():
    assert "is_turn_active" in _ROUTER.split("\n", 40)[21] or ", is_turn_active" in _ROUTER
    assert 'return {"messages": filtered_messages, "turn_active": is_turn_active(session_id)}' in _ROUTER


def test_timeout_del_llm_del_chat_es_30s_por_intento():
    m = re.search(r'"MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S",\s*30\.0,', _AGENT)
    assert m, "el default del timeout del agente debe ser 30.0"


def test_marker_bumpeado():
    assert '_LAST_KNOWN_PFIX = "P1-CHAT-ORPHAN-TURN-TRUTH · 2026-09-03"' in _APP
