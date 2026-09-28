# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-654 · 2026-09-27] Un proveedor de IA sin saldo deja una alerta, no solo un 429 en el log.

El 17-sep Z.ai respondió `429 code 1113 "Insufficient balance"`: cinco errores en el coach proactivo y uno en el chat,
el tráfico acabó en DeepSeek y NADA lo avisó — el backend no conocía la frase («Insufficient balance», «1113»,
«insufficient_quota») y lo trataba como un rate-limit que se pasa solo. Con OpenAI a US$6 sin recarga y DeepSeek a
~US$2,5, el siguiente agotamiento habría sido igual de mudo. Ahora la subclase `ChatOpenAI` de `llm_provider` —por la
que pasan Z.ai, DeepSeek y OpenAI— reconoce el agotamiento, emite `system_alert`
`llm_provider_balance_exhausted:<proveedor>` (una vez cada 10 min por proveedor y proceso) y re-lanza el error: el
breaker y la red cruzada siguen haciendo su trabajo como hoy.

tooltip-anchor: P1-PLAN-LOTE-654
"""
import pytest


class _Err(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize("exc,modelo,esperado", [
    (_Err("Error code: 429 - {'error': {'code': '1113', 'message': 'Insufficient balance or no resource package. "
          "Please recharge.'}}", 429), "glm-5.3-flash", "zai"),
    (_Err("Error code: 429 - {'error': {'message': 'You exceeded your current quota', 'type': 'insufficient_quota'}}",
          429), "gpt-6-luna", "openai"),
    (_Err("Error code: 402 - {'error': {'message': 'Insufficient Balance'}}", 402), "deepseek-flash", "deepseek"),
    (_Err("429 Your project has exceeded its monthly spending cap. https://ai.studio/spend"), "gemini-3.8-flash", "gemini"),
])
def test_reconoce_el_saldo_agotado_de_cada_proveedor(exc, modelo, esperado):
    from saldo_proveedor import proveedor_sin_saldo
    assert proveedor_sin_saldo(exc, modelo) == esperado


@pytest.mark.parametrize("exc", [
    _Err("Error code: 429 - Rate limit reached for requests", 429),
    _Err("Error code: 500 - internal error", 500),
    TimeoutError("timed out"),
])
def test_un_rate_limit_o_un_5xx_no_es_falta_de_saldo(exc):
    from saldo_proveedor import proveedor_sin_saldo
    assert proveedor_sin_saldo(exc, "glm-5.3-flash") is None


def test_la_alerta_se_escribe_una_vez_por_ventana(monkeypatch):
    import saldo_proveedor as sp
    escritos = []
    monkeypatch.setattr(sp, "_escribir_alerta", lambda prov, det: escritos.append((prov, det)))
    monkeypatch.setattr(sp, "_ultimo_aviso", {})
    exc = _Err("code 1113 Insufficient balance", 429)
    assert sp.avisar_si_saldo_agotado(exc, "glm-5.3") == "zai"
    assert sp.avisar_si_saldo_agotado(exc, "glm-5.3") == "zai"
    assert [p for p, _ in escritos] == ["zai"]
    assert sp.avisar_si_saldo_agotado(_Err("rate limit", 429), "glm-5.3") is None


def test_la_clave_de_la_alerta_nombra_al_proveedor(monkeypatch):
    import db_core
    import saldo_proveedor as sp
    llamadas = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: llamadas.append((q, p)))
    sp._escribir_alerta("openai", "insufficient_quota")
    q, p = llamadas[0]
    assert "INSERT INTO system_alerts" in q and p[0] == "llm_provider_balance_exhausted:openai"


def test_el_cliente_avisa_y_relanza(monkeypatch):
    import llm_provider
    import saldo_proveedor as sp
    vistos = []
    monkeypatch.setattr(sp, "avisar_si_saldo_agotado", lambda exc, modelo="": vistos.append(modelo) or "zai")

    def falla(self, *a, **k):
        raise _Err("code 1113 Insufficient balance", 429)
    monkeypatch.setattr(llm_provider._LangChainChatOpenAI, "_generate", falla)
    llm = llm_provider.ChatOpenAI(model="glm-5.3-flash", api_key="x", base_url="http://127.0.0.1:9")
    with pytest.raises(_Err):
        llm._generate([])
    assert vistos == ["glm-5.3-flash"]
