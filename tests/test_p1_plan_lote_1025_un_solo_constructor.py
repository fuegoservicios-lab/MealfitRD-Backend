"""[P1-PLAN-LOTE-1025 · A-26] `chat_with_agent` (POST /api/chat) es una envoltura del stream: un solo constructor."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent  # noqa: E402


def _ev(d):
    return f"data: {json.dumps(d)}\n\n"


def _con_stream(monkeypatch, eventos, capturado=None):
    def falso(session_id, prompt, **kw):
        if capturado is not None:
            capturado.update(kw, session_id=session_id, prompt=prompt)
        yield from eventos
    monkeypatch.setattr(agent, "chat_with_agent_stream", falso)


def test_devuelve_lo_del_done_y_pasa_los_mismos_argumentos(monkeypatch):
    vistos = {}
    _con_stream(monkeypatch, [
        _ev({"type": "progress", "phase": "analizando"}), _ev({"type": "chunk", "text": "Hola"}),
        _ev({"type": "done", "response": "Hola, anoté tu cena.", "updated_fields": {"weight": "80"}, "new_plan": {"x": 1}}),
    ], vistos)
    r = agent.chat_with_agent("s1", "cené arroz", current_plan={"p": 1}, user_id="u1", form_data={"a": 1},
                              local_date="2026-11-15", tz_offset=240)
    assert r == ("Hola, anoté tu cena.", {"weight": "80"}, {"x": 1})
    assert vistos == {"session_id": "s1", "prompt": "cené arroz", "current_plan": {"p": 1}, "user_id": "u1",
                      "form_data": {"a": 1}, "local_date": "2026-11-15", "tz_offset": 240}


def test_sin_response_en_el_done_junta_los_trozos(monkeypatch):
    _con_stream(monkeypatch, [_ev({"type": "chunk", "text": "a"}), _ev({"type": "chunk", "text": "b"}),
                              _ev({"type": "done", "response": "", "updated_fields": None})])
    assert agent.chat_with_agent("s", "x") == ("ab", {}, None)


@pytest.mark.parametrize("codigo,excepcion", [
    ("rate_limited", agent.LLMRateLimitedError), ("unavailable", agent.LLMCircuitBreakerOpen),
    ("timeout", TimeoutError), ("internal", RuntimeError),
])
def test_los_errores_del_stream_vuelven_a_ser_las_excepciones_que_traduce_el_router(monkeypatch, codigo, excepcion):
    _con_stream(monkeypatch, ["basura\n\n", _ev({"type": "error", "code": codigo, "message": "m"})])
    with pytest.raises(excepcion):
        agent.chat_with_agent("s", "x")


def test_ya_no_queda_un_segundo_constructor():
    src = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = src.index("def chat_with_agent(session_id: str")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "for evento in chat_with_agent_stream(" in cuerpo
    for viejo in ("chat_graph_app.invoke", "build_language_directive", "_build_past_days_context", "rag_query_router"):
        assert viejo not in cuerpo, viejo
    assert len(cuerpo.splitlines()) < 80
