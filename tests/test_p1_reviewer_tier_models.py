"""[P1-REVIEWER-TIER-MODELS · 2026-07-31] Reviewer clínico enrutado por TIER
tras el recorte de precios de OpenAI (luna -80% → $0.20/$1.20; terra -20% →
$2.00/$12.00 por 1M tokens).

Decisión del owner: "las cuentas gratis deben usar deepseek flash y Luna como
revisor médico; Terra como revisor en plus, y mira si en basic es rentable".
Rentabilidad medida con datos de producción (158 calls/30d, promedio 2.371 tok
in / 213 out): Terra en basic worst-case absoluto (cap de 50 planes, todos con
perfil clínico, 2 calls/plan) = $0.73/mes = 7.3% del revenue ($9.99) → rentable.
Escalera final: free/guest → Luna · basic/plus/ultra → Terra.

Contratos que ancla:
  A. Tier map: risk profile free → `gpt-5.6-luna`; basic/plus/ultra →
     `gpt-5.6-terra`. Sin user en contexto → free (fail-cheap simétrico).
  B. Knobs per-tier `MEALFIT_REVIEWER_RISK_MODEL_{FREE,PAID}` ganan sobre el
     default; `MEALFIT_REVIEWER_RISK_TIER_MODEL` (global) gana sobre el map y
     dispara la alerta de desvío.
  C. Fail-safe: modelo OpenAI sin OPENAI_API_KEY → fallback flash + alerta
     (test en test_p1_deepseek_only_restore.py, mismo marker).
  D. Construcción con dispatch por proveedor (ChatOpenAIInstrumented para
     gpt-*) y thinking DeepSeek-only excluido para OpenAI.
  E. Pricing table actualizada (luna/terra nuevos; el fact-checker NO cambia:
     sigue en `_REVIEWER_RISK_TIER_DEFAULT` = flash).
  F. Marker bumpeado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")

_RISK_FORM = {"allergies": ["maní"], "medicalConditions": ["Diabetes tipo 2"]}


@pytest.fixture()
def _go(monkeypatch):
    import graph_orchestrator as go

    go._CLINICAL_MODEL_GUARD_WARNED.clear()
    for knob in (
        "MEALFIT_REVIEWER_MODEL",
        "MEALFIT_REVIEWER_RISK_TIER_MODEL",
        "MEALFIT_REVIEWER_RISK_MODEL_FREE",
        "MEALFIT_REVIEWER_RISK_MODEL_PAID",
    ):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    yield go
    go._CLINICAL_MODEL_GUARD_WARNED.clear()


def _mute_alert_writes(monkeypatch):
    import db

    writes = []
    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: writes.append(a) or 1)
    return writes


# ------------------------------------------------------------------
# A. Tier map
# ------------------------------------------------------------------

def test_a_free_tier_risk_goes_to_luna(_go, monkeypatch):
    writes = _mute_alert_writes(monkeypatch)
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "gratis")
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-luna"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED and not writes


@pytest.mark.parametrize("tier", ["basic", "plus", "ultra"])
def test_a2_paid_tiers_risk_go_to_terra(_go, monkeypatch, tier):
    # [P1-REVIEWER-SOL-HARD] Se fija dificultad=False para testear el MAP BASE
    # (el escalón sol-difícil tiene su propio archivo test_p1_reviewer_sol_hard.py).
    writes = _mute_alert_writes(monkeypatch)
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: tier)
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: False)
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-terra"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED and not writes


def test_a3_no_user_context_fails_cheap_to_free(_go, monkeypatch):
    # Sin contexto (`user_id_var` default) → gratis → Luna, simétrico al router.
    _mute_alert_writes(monkeypatch)
    assert _go._reviewer_risk_model_for_tier() == "gpt-5.6-luna"


def test_a4_no_risk_profile_stays_flash(_go, monkeypatch):
    _mute_alert_writes(monkeypatch)
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "plus")
    resolved = _go._reviewer_model_name({"allergies": [], "medicalConditions": []})
    assert resolved == "deepseek-v4-flash"


# ------------------------------------------------------------------
# B. Precedencia de knobs
# ------------------------------------------------------------------

def test_b_per_tier_knobs_win_over_defaults(_go, monkeypatch):
    _mute_alert_writes(monkeypatch)
    monkeypatch.setenv("MEALFIT_REVIEWER_RISK_MODEL_PAID", "gpt-5.6-sol")
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "ultra")
    # El knob per-tier ES el esperado del tier → no alerta.
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-sol"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED


def test_b2_global_risk_knob_wins_and_alerts_desvio(_go, monkeypatch):
    writes = _mute_alert_writes(monkeypatch)
    monkeypatch.setenv("MEALFIT_REVIEWER_RISK_TIER_MODEL", "deepseek-v4-pro")
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "gratis")
    resolved = _go._reviewer_model_name(_RISK_FORM)
    assert resolved == "deepseek-v4-pro", "el knob global sigue ganando (observacional)"
    assert ("reviewer", "deepseek-v4-pro") in _go._CLINICAL_MODEL_GUARD_WARNED
    assert len(writes) == 1


# ------------------------------------------------------------------
# D. Construcción: dispatch por proveedor + thinking DeepSeek-only
# ------------------------------------------------------------------

def test_d_reviewer_builds_with_provider_dispatch():
    assert _GO_SRC.count(
        "(ChatOpenAIInstrumented if _rev_is_openai else ChatDeepSeek)("
    ) >= 2, (
        "el reviewer (rama estándar + fallback de thinking) debe despachar por "
        "proveedor — ChatDeepSeek mandaría gpt-5.6-* al base_url de DeepSeek "
        "con la key equivocada (lección P1-DAYGEN-LUNA-CANARY)"
    )


def test_d2_thinking_gate_excludes_openai():
    m = re.search(
        r"_rev_thinking = bool\(REVIEWER_THINKING_ENABLED and _profile_has_medical_risk\(form_data\)\s*\n\s*and not _rev_is_openai\)",
        _GO_SRC,
    )
    assert m, "el gate de thinking debe excluir modelos OpenAI (extra_body es DeepSeek-only)"


def test_d3_fact_checker_unchanged():
    # El fact-checker sigue anclado al risk-tier DeepSeek (no cambió de provider).
    assert '_env_str("MEALFIT_FACT_CHECKER_RISK_TIER_MODEL", _REVIEWER_RISK_TIER_DEFAULT)' in _GO_SRC


# ------------------------------------------------------------------
# E. Pricing actualizado
# ------------------------------------------------------------------

def test_e_pricing_table_post_recorte():
    from db_profiles import _DEFAULT_LLM_PRICING_MICROS_PER_M as TABLA

    assert TABLA["gpt-5.6-luna"] == {"input": 200_000, "output": 1_200_000, "cached": 20_000}
    assert TABLA["gpt-5.6-terra"] == {"input": 2_000_000, "output": 12_000_000, "cached": 200_000}


# ------------------------------------------------------------------
# F. Marker + anchors
# ------------------------------------------------------------------

def test_f_marker_and_anchors():
    assert "P1-REVIEWER-TIER-MODELS" in _GO_SRC
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-REVIEWER-TIER-MODELS" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-31"
