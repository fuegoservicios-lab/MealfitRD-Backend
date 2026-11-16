# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1043 · A-20] Luna por Responses con esfuerzo, detrás de un knob apagado; la caché lee `input`."""
from pathlib import Path

import coach_luna_razona as clr
import openai_prompt_cache as opc

_BACKEND = Path(__file__).resolve().parent.parent


def test_apagado_por_defecto(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_LUNA_RESPONSES_EFFORT", raising=False)
    assert clr.esfuerzo() is None
    assert clr.kwargs_cliente("gpt-6-luna") == {}


def test_solo_luna_y_solo_el_turno_normal(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_LUNA_RESPONSES_EFFORT", "low")
    assert clr.kwargs_cliente("gpt-6-luna") == {"use_responses_api": True, "reasoning_effort": "low"}
    assert clr.kwargs_cliente("gpt-6.1-terra") == {}
    assert clr.kwargs_cliente("gemini-4-hydrogen") == {}
    assert clr.kwargs_cliente("gpt-6-luna", escalado=True) == {}
    assert clr.kwargs_cliente("gpt-6-luna", modo_voz=True) == {}


def test_valor_desconocido_no_activa(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_LUNA_RESPONSES_EFFORT", "max")
    assert clr.esfuerzo() is None


def test_allowed_tools_en_forma_de_responses():
    tc = clr.tool_choice_responses(["log_consumed_meal", "proponer_comida"])
    assert tc == {"type": "allowed_tools", "mode": "auto",
                  "tools": [{"type": "function", "name": "log_consumed_meal"},
                            {"type": "function", "name": "proponer_comida"}]}


def test_agent_lo_usa_en_el_turno_normal():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "_luna_kw = _clr.kwargs_cliente(_chat_agent_model_name(_model_uid)" in src
    assert "**(_luna_kw or _kwargs_de_razonamiento_modo_voz(state))" in src
    assert "_clr.tool_choice_responses(_perm) if _luna_kw else _tc_perm(_perm)" in src


def test_la_clave_de_cache_lee_input_de_responses():
    sistema = "MANUAL " * 900
    chat = {"model": "gpt-6-luna", "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": "hola"}]}
    resp = {"model": "gpt-6-luna", "input": [
        {"role": "system", "content": [{"type": "input_text", "text": sistema}]},
        {"role": "user", "content": [{"type": "input_text", "text": "hola"}]}]}
    s_chat, _ = opc._extract_system_and_tool_names(chat)
    s_resp, _ = opc._extract_system_and_tool_names(resp)
    assert s_chat == s_resp == sistema
    assert opc._extract_system_and_tool_names({"model": "x", "instructions": sistema, "input": "hola"})[0] == sistema
