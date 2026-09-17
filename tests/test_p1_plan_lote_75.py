# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-75 · 2026-09-17] Dos cabos de la noche de DeepSeek, vistos en la captura y en la base:

1. La cuota del COACH no tenía fila `admin` y caía a la de gratis (60): el dueño, probando, iba 57/60. Admin es
   ilimitado en `_TIER_LIMITS` (planes) y ahora también aquí.
2. `log_llm_usage_event` anotaba el ID que le pasaba el callsite (el de su knob, `glm-5.3-flash`) en llamadas que
   salieron por `deepseek-flash`, y el costo se calculaba con la tarifa de GLM. Anota el modelo EFECTIVO.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


# ── 1. admin en la cuota del coach ────────────────────────────────────────────────────────────────

def test_admin_is_unlimited_in_the_coach_quota(monkeypatch):
    import auth
    assert auth._COACH_LIMITS["admin"] == 999999
    assert auth._COACH_LIMITS["gratis"] == 60          # el resto no se mueve
    monkeypatch.setattr(auth, "get_monthly_api_usage", lambda uid, kind=None: 57)
    monkeypatch.setattr(auth, "get_user_profile", lambda uid: {"plan_tier": "admin"})
    snap = auth.coach_quota_snapshot("u1")
    assert snap["limit"] == 999999 and snap["remaining"] == 999999 - 57 and snap["tier"] == "admin"
    monkeypatch.setattr(auth, "get_user_profile", lambda uid: {"plan_tier": "gratis"})
    assert auth.coach_quota_snapshot("u1")["limit"] == 60


# ── 2. el uso LLM anota el modelo efectivo ────────────────────────────────────────────────────────

@pytest.fixture
def captura(monkeypatch):
    import db_core
    import db_profiles
    monkeypatch.setenv("ZAI_API_KEY", "test-not-real")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    monkeypatch.setenv("MEALFIT_LLM_COST_TRACKING_ENABLED", "1")
    monkeypatch.setattr(db_core, "connection_pool", object())          # que no salte el INSERT
    escritos = []
    monkeypatch.setattr(db_profiles, "execute_sql_write", lambda sql, params=None, **kw: escritos.append(params))
    return db_profiles, escritos


def test_usage_event_records_the_effective_model_under_deepseek(captura, monkeypatch):
    db_profiles, escritos = captura
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    db_profiles.log_llm_usage_event(model="glm-5.3-flash", node="fact_extractor_router", input_tokens=1_000_000, output_tokens=0)
    assert escritos and "deepseek-flash" in escritos[-1] and "glm-5.3-flash" not in escritos[-1]
    assert 150_000 in escritos[-1]                                       # tarifa de DeepSeek, no la de GLM (150k igual… la salida distingue)
    db_profiles.log_llm_usage_event(model="glm-5.3-flash", node="x", input_tokens=0, output_tokens=1_000_000)
    assert 600_000 in escritos[-1] and 500_000 not in escritos[-1]      # $0,60/M de DeepSeek, no $0,50/M de GLM


def test_usage_event_keeps_the_model_under_zai_and_for_openai_ids(captura, monkeypatch):
    db_profiles, escritos = captura
    monkeypatch.delenv("MEALFIT_LLM_PROVIDER", raising=False)
    db_profiles.log_llm_usage_event(model="glm-5.3-flash", node="x", input_tokens=10, output_tokens=10)
    assert "glm-5.3-flash" in escritos[-1]
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    db_profiles.log_llm_usage_event(model="gpt-5.6-luna", node="x", input_tokens=10, output_tokens=10)
    assert "gpt-5.6-luna" in escritos[-1]


def test_marker():
    assert 'P1-PLAN-LOTE-75 · 2026-09-17' in (_BACKEND / "app.py").read_text(encoding="utf-8")
