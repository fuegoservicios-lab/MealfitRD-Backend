# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-995 · 2026-11-13] Solo proveedores occidentales; Gemini 4 Hydrogen en el coach y el escáner.

Lo que ancla:
  1. Sin `MEALFIT_PERMITIR_PROVEEDORES_CHINA`, ningún camino llega a DeepSeek ni a Z.ai: ni el knob del proveedor, ni el
     país forzado, ni un ID `deepseek-*`/`glm-*` fijado por knob.
  2. Un modelo `gemini-*` va a la capa compatible con OpenAI de Google, con la key del escáner si es la que hay, y con
     el esfuerzo en el vocabulario de Gemini.
  3. La firma de razonamiento de cada tool_call (medido: sin ella, la segunda vuelta del coach da 400) se guarda de la
     respuesta —entera y en streaming, donde Gemini manda `index: None`— y vuelve en la siguiente petición.
  4. El coach usa Hydrogen con key de Gemini y el router de siempre sin ella; el escáner baja el esfuerzo.
  5. El permiso `ia-2026-11` no nombra a DeepSeek ni pide la casilla de China, y el de la anterior sigue valiendo.

tooltip-anchor: P1-PLAN-LOTE-995
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import gemini_hydrogen as gh
import llm_provider as lp

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _entorno(monkeypatch):
    for k in ("MEALFIT_PERMITIR_PROVEEDORES_CHINA", "MEALFIT_LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "VISION_API_KEY", "MEALFIT_VISION_BASE_URL", "MEALFIT_CHAT_AGENT_MODEL", "MEALFIT_EU_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-real")


# ── 1. ningún proveedor en China ───────────────────────────────────────────────────────────────────
def test_el_default_es_openai():
    assert lp.llm_provider_name() == "openai"
    llm = lp.ChatGLM(model=lp.GLM_FLASH)
    assert "api.openai.com" in llm.openai_api_base and llm.model_name == lp.GPT6_LUNA


@pytest.mark.parametrize("valor", ["deepseek", "zai"])
def test_el_knob_viejo_a_china_cae_a_openai(monkeypatch, valor):
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", valor)
    assert lp.llm_provider_name() == "openai"
    assert "api.openai.com" in lp.ChatGLM(model=lp.GLM_PRO).openai_api_base


@pytest.mark.parametrize("modelo", ["deepseek-flash", "deepseek-v4-pro", "deepseek-v9-algo", "glm-5v-turbo"])
def test_un_id_chino_fijado_va_a_openai(modelo):
    llm = lp.ChatGLM(model=modelo)
    assert "api.openai.com" in llm.openai_api_base
    assert llm.model_name == lp.GPT6_LUNA
    assert "deepseek" not in llm.openai_api_base and "z.ai" not in llm.openai_api_base


def test_el_pais_forzado_tampoco_lleva_a_china(monkeypatch):
    monkeypatch.setattr("region_ia.proveedor_forzado", lambda: "deepseek")
    assert lp.llm_provider_name() == "openai"


def test_con_el_permiso_a_sabiendas_vuelve_la_fontaneria(monkeypatch):
    monkeypatch.setenv("MEALFIT_PERMITIR_PROVEEDORES_CHINA", "1")
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert lp.llm_provider_name() == "deepseek"
    assert "api.deepseek.com" in lp.ChatGLM(model=lp.GLM_FLASH).openai_api_base


# ── 2. Gemini por la capa compatible con OpenAI ────────────────────────────────────────────────────
def test_gemini_va_a_google_con_la_key_del_escaner_y_su_esfuerzo(monkeypatch):
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", lp.GEMINI_OPENAI_BASE_URL)
    monkeypatch.setenv("VISION_API_KEY", "clave-google")
    llm = lp.ChatGLM(model=gh.GEMINI_HYDROGEN, extra_body={"thinking": {"type": "disabled"}})
    assert llm.openai_api_base == lp.GEMINI_OPENAI_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "clave-google"
    assert llm.model_name == gh.GEMINI_HYDROGEN
    assert llm.reasoning_effort == "low" and not (llm.extra_body or {}).get("thinking")
    assert lp.ChatGLM(model=gh.GEMINI_HYDROGEN, reasoning_effort="medium").reasoning_effort == "medium"
    assert lp.ChatGLM(model=gh.GEMINI_HYDROGEN, reasoning_effort="none").reasoning_effort == "low"


def test_la_key_del_escaner_solo_si_el_escaner_va_a_google(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "clave-de-otro")
    assert gh.clave_gemini() == "" and gh.hay_key_gemini() is False
    monkeypatch.setenv("GEMINI_API_KEY", "propia")
    assert gh.clave_gemini() == "propia"


@pytest.mark.parametrize("valor, esperado", [("low", "low"), ("medium", "medium"), ("high", "high"),
                                             ("xhigh", "xhigh"), ("max", "xhigh"), ("none", "low"), (None, "low")])
def test_esfuerzo_gemini(valor, esperado):
    assert gh.esfuerzo_gemini(valor) == esperado


def test_precio_de_hydrogen_registrado():
    from db_profiles import compute_llm_cost_micros
    assert compute_llm_cost_micros(gh.GEMINI_HYDROGEN, 1_000_000, 1_000_000) == 1_500_000 + 7_500_000


# ── 3. la firma de razonamiento de los tool_calls ──────────────────────────────────────────────────
_FIRMA = {"google": {"thought_signature": "EpoC-firma"}}


def _resultado(tool_calls):
    from langchain_core.messages import AIMessage
    return SimpleNamespace(generations=[SimpleNamespace(message=AIMessage(content="", tool_calls=tool_calls))])


def test_captura_de_la_respuesta_entera():
    crudo = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}, "extra_content": _FIRMA}]}}]}
    res = _resultado([{"name": "f", "args": {}, "id": "c1"}])
    gh.capturar_de_respuesta(crudo, res)
    assert res.generations[0].message.additional_kwargs[gh.CLAVE_FIRMAS] == {"0": _FIRMA}


def test_captura_en_streaming_con_index_none():
    from langchain_core.messages import AIMessageChunk
    gen = SimpleNamespace(message=AIMessageChunk(content=""))
    gh.capturar_de_trozo({"choices": [{"delta": {"tool_calls": [{"index": None, "id": "c1", "extra_content": _FIRMA,
                                                                 "function": {"name": "f", "arguments": "{}"}}]}}]}, gen)
    assert gen.message.additional_kwargs[gh.CLAVE_FIRMAS] == {"0": _FIRMA}


def test_reinyecta_la_firma_y_la_ajena_lleva_el_valor_documentado():
    from langchain_core.messages import AIMessage
    propio = AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "c1"}],
                       additional_kwargs={gh.CLAVE_FIRMAS: {"0": _FIRMA}})
    ajeno = AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "c2"}])
    payload = {"messages": [{"role": "assistant", "tool_calls": [{"id": "c1"}]},
                            {"role": "assistant", "tool_calls": [{"id": "c2"}]},
                            {"role": "user", "content": "hola"}]}
    gh.reinyectar_firmas(payload, [propio, ajeno])
    assert payload["messages"][0]["tool_calls"][0]["extra_content"] == _FIRMA
    assert payload["messages"][1]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "skip_thought_signature_validator"}}


def test_el_cliente_reinyecta_solo_hacia_gemini(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    hist = [HumanMessage("kcal?"),
            AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "c1"}],
                      additional_kwargs={gh.CLAVE_FIRMAS: {"0": _FIRMA}}),
            ToolMessage(content="130", tool_call_id="c1")]
    pg = lp.ChatGLM(model=gh.GEMINI_HYDROGEN)._get_request_payload(hist)
    asis = [m for m in pg["messages"] if m.get("role") == "assistant"][0]
    assert asis["tool_calls"][0]["extra_content"] == _FIRMA
    po = lp.ChatGLM(model=lp.GLM_FLASH)._get_request_payload(hist)
    assert "extra_content" not in [m for m in po["messages"] if m.get("role") == "assistant"][0]["tool_calls"][0]


# ── 4. el coach y el escáner ───────────────────────────────────────────────────────────────────────
def test_el_coach_usa_hydrogen_con_key_y_el_router_sin_ella(monkeypatch):
    import agent
    assert agent._chat_agent_model_name("u") != gh.GEMINI_HYDROGEN
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert agent._chat_agent_model_name("u") == gh.GEMINI_HYDROGEN
    monkeypatch.setenv("MEALFIT_CHAT_AGENT_MODEL", "gpt-6-luna")       # el knob sigue mandando (rollback)
    assert agent._chat_agent_model_name("u") == "gpt-6-luna"


def test_el_respaldo_del_coach_es_de_otro_proveedor():
    import agent
    assert agent._chat_agent_fallback_model(gh.GEMINI_HYDROGEN) == lp.GPT6_LUNA


def test_el_escaner_con_gemini_baja_el_esfuerzo():
    src = io.open(_BACKEND / "vision_agent.py", encoding="utf-8").read()
    assert 'reasoning_effort=esfuerzo_gemini(_env_str("MEALFIT_VISION_REASONING_EFFORT", "low"))' in src


# ── 5. el permiso ──────────────────────────────────────────────────────────────────────────────────
def test_el_permiso_nuevo_no_nombra_a_china():
    import consentimientos as cs
    assert cs.AI_CONSENT_VERSION == "ia-2026-11" and cs.AI_CONSENT_VERSION_ANTERIOR == "ia-2026-10-voz"
    assert cs.CLAVES_IA == ("ai_processing",)
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        md = io.open(_BACKEND / "docs" / "consentimientos" / "ia" / "ia-2026-11" / f"{loc}.md", encoding="utf-8").read()
        texto = md.split("<!-- texto:inicio -->")[1].split("<!-- texto:fin -->")[0]
        assert "DeepSeek" not in texto and "China" not in texto and "Chine" not in texto and "Cina" not in texto


def test_marker():
    src = io.open(_BACKEND / "app.py", encoding="utf-8").read()
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+)\b', src)
    assert m is not None and int(m.group(1)) >= 995
