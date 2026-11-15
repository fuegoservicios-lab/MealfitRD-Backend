"""[P1-PLAN-LOTE-1023 · A-32] El margen del límite de OpenAI se lee de las cabeceras, se avisa al 70 % y las cabeceras
nunca llegan al mensaje guardado."""
from __future__ import annotations

import logging
import os
import sys

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import coach_limites_proveedor as clp  # noqa: E402

H = lambda lt, rt, lr="5000", rr="4990": {"X-RateLimit-Limit-Tokens": lt, "x-ratelimit-remaining-tokens": rt,
                                          "x-ratelimit-limit-requests": lr, "x-ratelimit-remaining-requests": rr,
                                          "content-type": "text/event-stream"}


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    clp.reset_para_tests()
    monkeypatch.delenv("MEALFIT_OPENAI_RATELIMIT_ALERT_PCT", raising=False)
    escritas = []
    monkeypatch.setattr(clp, "_abrir_alerta", lambda foto: escritas.append(foto))
    yield escritas
    clp.reset_para_tests()


def test_lee_el_margen_y_no_avisa_por_debajo_del_umbral(limpio):
    foto = clp.registrar(H("2000000", "1700000"), "gpt-6-luna")
    assert foto["uso_tokens_pct"] == 15.0 and foto["uso_peticiones_pct"] == 0.2
    assert limpio == [] and clp.snapshot()["uso_tokens_pct"] == 15.0


def test_avisa_al_pasar_del_70_y_solo_una_alerta_por_hora(limpio, caplog):
    with caplog.at_level(logging.WARNING):
        clp.registrar(H("2000000", "500000"), "gpt-6-luna")      # 75 %
        clp.registrar(H("2000000", "100000"), "gpt-6-luna")      # 95 %
    assert len(limpio) == 1 and limpio[0]["uso_tokens_pct"] == 75.0
    assert caplog.text.count("[A-32 RATELIMIT]") == 1
    assert clp.snapshot()["uso_tokens_pct"] == 95.0              # la foto sí se actualiza siempre


def test_sin_cabeceras_de_margen_no_hace_nada_y_nunca_lanza(limpio):
    assert clp.registrar({"content-type": "x"}) is None
    assert clp.registrar(None) is None and clp.registrar("basura") is None
    assert clp.snapshot() == {}


def test_las_cabeceras_se_quitan_del_trozo_y_del_resultado(limpio):
    import llm_provider as lp

    trozo = ChatGenerationChunk(message=AIMessageChunk(content="hola"), generation_info={"headers": H("100", "10")})
    lp._leer_y_quitar_cabeceras(trozo, "gpt-6-luna")
    assert "headers" not in (trozo.generation_info or {}) and clp.snapshot()["uso_tokens_pct"] == 90.0

    msg = AIMessage(content="ok", response_metadata={"headers": {"a": "b"}, "finish_reason": "stop"})
    res = ChatResult(generations=[ChatGeneration(message=msg, generation_info={"headers": H("100", "50")})])
    lp._quitar_cabeceras_del_resultado(res, "gpt-6-luna")
    assert "headers" not in res.generations[0].generation_info
    assert res.generations[0].message.response_metadata == {"finish_reason": "stop"}


def test_el_cliente_de_openai_pide_las_cabeceras_salvo_en_responses_api(monkeypatch):
    import llm_provider as lp

    monkeypatch.setattr(lp, "_openai_api_key", lambda: "sk-test")
    assert lp.build_chat_llm("gpt-6-luna").include_response_headers is True
    assert lp.build_chat_llm("gpt-6.1-terra").include_response_headers is False      # Responses API: no aplica
    monkeypatch.setenv("MEALFIT_OPENAI_RATELIMIT_WATCH", "false")
    assert lp.build_chat_llm("gpt-6-luna").include_response_headers is False
