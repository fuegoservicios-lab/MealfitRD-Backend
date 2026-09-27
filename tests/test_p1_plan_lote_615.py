# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-615 · 2026-09-27] El revisor clínico no se queda sin modelo cuando OpenAI falla.

Producción, 27-sep 16:53-16:59 UTC: el breaker del revisor de OpenAI abierto (saldo agotado) → tres «Error TRANSITORIO
del reviewer (LLMCircuitOpenError)», dos regeneraciones completas pagadas y el bloque entregado degradado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import revisor_respaldo as rr  # noqa: E402

_OPENAI = "gpt-6-luna"


class RateLimitError(Exception):          # el nombre es el del cliente de OpenAI («insufficient_quota», 429)
    pass


class AuthenticationError(Exception):     # clave inválida o revocada (401)
    pass


@pytest.fixture(autouse=True)
def _openai(monkeypatch):
    monkeypatch.setattr(go, "is_openai_model", lambda m: str(m).startswith("gpt-"))


@pytest.mark.parametrize("exc", [go.LLMCircuitOpenError("Circuit Breaker OPEN para gpt-6-luna"),
                                 RateLimitError("Error code: 429 - insufficient_quota"),
                                 AuthenticationError("Error code: 401 - invalid_api_key"),
                                 TimeoutError("reviewer timeout")])
def test_openai_caido_pasa_al_respaldo_y_lo_anota(exc):
    plan = {}
    assert rr.respaldo(_OPENAI, exc, plan) == go._REVIEWER_RISK_TIER_DEFAULT
    assert plan["_reviewer_fallback"] == {"from": _OPENAI, "to": go._REVIEWER_RISK_TIER_DEFAULT,
                                          "error": type(exc).__name__}


def test_el_revisor_de_respaldo_no_tiene_otro_respaldo():
    assert rr.respaldo(go._REVIEWER_RISK_TIER_DEFAULT, go.LLMCircuitOpenError("abierto")) is None


def test_un_rechazo_que_no_es_del_proveedor_no_cambia_de_modelo():
    assert rr.respaldo(_OPENAI, ValueError("respuesta clínica inesperada")) is None


def test_el_tope_de_gasto_del_plan_no_es_un_fallo_del_proveedor(monkeypatch):
    monkeypatch.setattr(go, "_is_plan_spend_cap_error", lambda e: True)
    assert rr.respaldo(_OPENAI, RateLimitError("tope")) is None


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_REVIEWER_CROSS_PROVIDER_FALLBACK", "false")
    assert rr.respaldo(_OPENAI, go.LLMCircuitOpenError("abierto")) is None


def test_ancla_en_el_revisor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('if (_resp615 := __import__("revisor_respaldo").respaldo(_reviewer_model, _thk_e, plan)):')
    j = src.index("if not (_rev_thinking and not _is_plan_spend_cap_error(_thk_e)):", i)
    assert 0 < j - i < 900, "el respaldo va justo antes de la decisión de relanzar el error"
    assert "_reviewer_model, _rev_is_openai, _reviewer_cb, _rev_thinking = _resp615, False, _get_circuit_breaker(_resp615), True" \
        in src[i:j]
