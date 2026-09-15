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
    """[P1-CHAT-ACTIVE-TURN-WHOLE · 2026-09-14] El registro y la retirada viven en el
    decorador `_tracks_active_turn`, que envuelve el generador ENTERO (el `finally` del
    bucle no cubría el preámbulo) y solo retira el turno si sigue siendo el suyo."""
    i = _AGENT.index("def _tracks_active_turn(")
    body = _AGENT[i:_AGENT.index("def _chat_stream_error_payload(")]
    assert "_ACTIVE_TURNS[session_id] = token" in body, "el turno no se registra al entrar"
    assert re.search(r"finally:\r?\n\s+_release_active_turn\(session_id, token\)", body), (
        "el finally del decorador no retira el turno"
    )
    assert "@_tracks_active_turn\ndef chat_with_agent_stream(" in _AGENT.replace("\r\n", "\n")


def test_el_decorador_retira_en_todos_los_exits_y_solo_su_turno():
    import sys
    sys.path.insert(0, str(_BACKEND))
    import agent as A

    @A._tracks_active_turn
    def _gen(session_id, falla=False):
        yield "a"
        if falla:
            raise RuntimeError("boom")
        yield "b"

    list(_gen(session_id="s-ok"))
    assert "s-ok" not in A._ACTIVE_TURNS

    try:
        list(_gen("s-err", falla=True))
    except RuntimeError:
        pass
    assert "s-err" not in A._ACTIVE_TURNS, "una excepción dejó el turno registrado"

    g = _gen(session_id="s-cut")
    next(g)
    assert A.is_turn_active("s-cut")
    g.close()  # el cliente corta
    assert "s-cut" not in A._ACTIVE_TURNS

    # Dos turnos en la misma sesión: el primero que termina no borra al segundo.
    g1 = _gen(session_id="s-dup")
    next(g1)
    A._ACTIVE_TURNS["s-dup"] = A._ACTIVE_TURNS["s-dup"] + 1.0  # otro turno lo pisó
    g1.close()
    assert "s-dup" in A._ACTIVE_TURNS, "el turno viejo borró el marcador del nuevo"
    A._ACTIVE_TURNS.pop("s-dup", None)


def test_history_devuelve_turn_active():
    assert "is_turn_active" in _ROUTER.split("\n", 40)[21] or ", is_turn_active" in _ROUTER
    assert 'return {"messages": filtered_messages, "turn_active": is_turn_active(session_id)}' in _ROUTER


def test_timeout_del_llm_del_chat_es_30s_por_intento():
    m = re.search(r'"MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S",\s*30\.0,', _AGENT)
    assert m, "el default del timeout del agente debe ser 30.0"


def test_marker_bumpeado():
    assert 'P1-CHAT-ORPHAN-TURN-TRUTH · 2026-09-03' in _APP
