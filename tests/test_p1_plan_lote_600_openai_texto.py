# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-600 · 2026-09-27] GPT-6 como proveedor de TEXTO por knob (`MEALFIT_LLM_PROVIDER=openai`).

El dueño: «¿usa GPT-6 para texto también? ¿cuál es mejor?». DeepSeek era el único proveedor de texto activo (saldo de
US$3,23; Z.ai sin saldo) y GPT-6 Luna ya generaba la mayoría de los días, a menor precio por token ($0,10/$0,50 frente
a $0,15/$0,60 de deepseek-flash fuera de pico). Contrato:
  · con el knob en `zai`/`deepseek` NADA cambia;
  · con `openai`, los IDs GLM/DeepSeek de los ~12 defaults por feature van a OpenAI TRADUCIDOS (flash/pro → gpt-6-luna,
    cada uno con su knob), sin el `thinking` propio de GLM/DeepSeek (OpenAI rechaza el campo desconocido con 400) y con
    el esfuerzo en el vocabulario de OpenAI (`thinking.type=disabled` ⇒ `none`);
  · un ID `deepseek-*` FIJADO va a DeepSeek siempre: así la red post-fallo (`MEALFIT_PRO_MODEL`) puede quedar en OTRO
    proveedor — la diversidad es su razón de ser (P1-NET-LUNA);
  · una instancia apuntada explícitamente a otra base no se toca.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def lp(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key-not-real")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-not-real")
    for k in ("MEALFIT_LLM_PROVIDER", "MEALFIT_OPENAI_BASE_URL", "MEALFIT_OPENAI_FLASH_MODEL",
              "MEALFIT_OPENAI_PRO_MODEL", "MEALFIT_DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    import llm_provider as _lp
    return _lp


@pytest.fixture
def openai(lp, monkeypatch):
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "openai")
    return lp


def test_openai_es_un_proveedor_valido(openai):
    assert openai.llm_provider_name() == "openai"


def test_los_ids_glm_van_a_openai_traducidos(openai):
    flash = openai.ChatGLM(model=openai.GLM_FLASH)
    assert "api.openai.com" in (flash.openai_api_base or "")
    assert flash.model_name == openai.GPT6_LUNA
    assert openai.ChatGLM(model=openai.GLM_PRO).model_name == openai.GPT6_LUNA
    assert flash.openai_api_key.get_secret_value() == "test-openai-key-not-real"


def test_los_modelos_de_openai_se_eligen_por_knob(openai, monkeypatch):
    monkeypatch.setenv("MEALFIT_OPENAI_PRO_MODEL", "gpt-5.6-terra")
    assert openai.ChatGLM(model=openai.GLM_PRO).model_name == "gpt-5.6-terra"
    assert openai.ChatGLM(model=openai.GLM_FLASH).model_name == openai.GPT6_LUNA


def test_sin_thinking_y_con_el_esfuerzo_de_openai(openai):
    llm = openai.ChatGLM(model=openai.GLM_FLASH)
    assert "thinking" not in (llm.extra_body or {})
    assert llm.reasoning_effort == "low"                                              # default del knob
    assert openai.ChatGLM(model=openai.GLM_FLASH, reasoning_effort="medium").reasoning_effort == "medium"
    assert openai.ChatGLM(model=openai.GLM_FLASH, extra_body={"thinking": {"effort": "max"}}).reasoning_effort == "max"
    apagado = openai.ChatGLM(model=openai.GLM_FLASH, extra_body={"thinking": {"type": "disabled"}})
    assert apagado.reasoning_effort == "none" and "thinking" not in (apagado.extra_body or {})


def test_un_id_deepseek_fijado_va_a_deepseek(openai):
    red = openai.ChatGLM(model="deepseek-flash")
    assert "api.deepseek.com" in (red.openai_api_base or "")
    assert red.model_name == "deepseek-flash"
    assert red.openai_api_key.get_secret_value() == "test-deepseek-key-not-real"
    assert (red.extra_body or {}).get("thinking") == {"type": "enabled"}              # la rama DeepSeek de siempre


def test_con_el_knob_por_defecto_nada_cambia(lp):
    llm = lp.ChatGLM(model=lp.GLM_FLASH)
    assert "api.z.ai" in (llm.openai_api_base or "") and llm.model_name == lp.GLM_FLASH


def test_una_instancia_explicita_no_se_toca(openai):
    llm = openai.ChatGLM(model="gpt-5.6-luna", api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert llm.model_name == "gpt-5.6-luna" and not getattr(llm, "reasoning_effort", None)


def test_la_key_de_openai_solo_del_entorno_con_placeholder(openai, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = openai.ChatGLM(model=openai.GLM_FLASH)                                      # el boot no cae
    assert llm.openai_api_key.get_secret_value() == "MISSING_OPENAI_API_KEY"
