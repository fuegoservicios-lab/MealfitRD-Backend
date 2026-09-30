# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-937 · 2026-09-30] Los datos de un usuario de la Unión Europea no van al proveedor que los trata en China: su proveedor de
texto es OpenAI (EE. UU.), decidido por el PAÍS del formulario o del perfil, no por un knob global.

Auditoría legal para la App Store (29-sep): la transferencia a China se ampara hoy en el consentimiento explícito
(art. 49.1.a RGPD), que el Comité Europeo interpreta de forma restrictiva; lo robusto era enrutar la UE a otro
proveedor. El dueño (30-sep) delegó la decisión: se enruta. El único país de la UE que la app sirve hoy es España.

Dónde se decide: `llm_provider.llm_provider_name()`, que ya leen la base, la clave y el modelo del wrapper. La región
sale (1) del formulario de la corrida que `nevera_exigida` fija en su ContextVar (generación y bloques) o (2) de
`region_ia.fijar_pais`, que el chat fija tras fundir el formulario con el perfil. Sin país conocido, el proveedor de
siempre. Knobs `MEALFIT_EU_LLM_ROUTING` (True), `MEALFIT_EU_LLM_PROVIDER` (openai), `MEALFIT_EU_COUNTRIES` (ES).
"""
from __future__ import annotations

import pathlib

import pytest

import llm_provider as lp
import nevera_exigida as ne
import region_ia as ri

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _proveedor_china(monkeypatch):
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", lp.PROVEEDOR_EN_CHINA)
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    tok = ri.fijar_pais(None)
    yield
    ri.restaurar(tok)


def test_sin_pais_conocido_el_proveedor_de_siempre():
    assert lp.llm_provider_name() == lp.PROVEEDOR_EN_CHINA
    assert lp.PROVEEDOR_EN_CHINA in lp._default_base_url()


def test_el_formulario_de_la_corrida_manda_espana_a_openai():
    tok = ne.fijar({"country": "ES", "age": "30"})
    try:
        assert lp.llm_provider_name() == "openai"
        assert "openai" in lp._default_base_url()
        assert lp.provider_razona_largo() is False
    finally:
        ne._FD.reset(tok)
    tok = ne.fijar({"country": "DO"})
    try:
        assert lp.llm_provider_name() == lp.PROVEEDOR_EN_CHINA
    finally:
        ne._FD.reset(tok)


def test_el_pais_fijado_por_el_chat_tambien():
    tok = ri.fijar_pais("es")
    try:
        assert ri.es_ue("es") and lp.llm_provider_name() == "openai"
    finally:
        ri.restaurar(tok)
    tok = ri.fijar_pais("MX")
    try:
        assert lp.llm_provider_name() == lp.PROVEEDOR_EN_CHINA
    finally:
        ri.restaurar(tok)


def test_el_pais_del_perfil_pasa_por_el_canonizador():
    """«España», «es-ES» o «Spain» son España; basura, ninguno."""
    assert ri.es_ue("España") and ri.es_ue("ES") and ri.es_ue(" es ")
    assert not ri.es_ue("DO") and not ri.es_ue(None) and not ri.es_ue("")


def test_con_el_knob_apagado_nada_cambia(monkeypatch):
    monkeypatch.setenv("MEALFIT_EU_LLM_ROUTING", "false")
    tok = ri.fijar_pais("ES")
    try:
        assert lp.llm_provider_name() == lp.PROVEEDOR_EN_CHINA
    finally:
        ri.restaurar(tok)


def test_otro_proveedor_para_la_ue_por_knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_EU_LLM_PROVIDER", "zai")
    tok = ri.fijar_pais("ES")
    try:
        assert lp.llm_provider_name() == "zai"
    finally:
        ri.restaurar(tok)
    monkeypatch.setenv("MEALFIT_EU_LLM_PROVIDER", lp.PROVEEDOR_EN_CHINA)   # un knob que devuelve a China no vale
    tok = ri.fijar_pais("ES")
    try:
        assert lp.llm_provider_name() == "openai"
    finally:
        ri.restaurar(tok)


def test_el_chat_fija_el_pais_tras_fundir_el_perfil():
    src = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    ocurrencias = [i for i in range(len(src)) if src.startswith("form_data = merge_form_data_with_profile(", i)]
    assert len(ocurrencias) >= 2
    for i in ocurrencias:
        assert 'region_ia").fijar_pais((form_data or {}).get("country"))' in src[i:i + 700], src[i:i + 400]
    assert 'region_ia").proveedor_forzado()' in (_BACKEND / "llm_provider.py").read_text(encoding="utf-8")
    mod = _BACKEND / "region_ia.py"
    assert "tooltip-anchor: P1-PLAN-LOTE-937" in mod.read_text(encoding="utf-8")
    assert b"\x08" not in mod.read_bytes() and b"\r" not in mod.read_bytes()
