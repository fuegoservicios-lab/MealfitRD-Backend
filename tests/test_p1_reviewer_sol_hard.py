"""[P1-REVIEWER-SOL-HARD · 2026-07-31] Escalón superior del reviewer clínico:
plus/ultra + perfil clínicamente DIFÍCIL → `gpt-5.6-sol` ($5/$30 por 1M,
~$0.018/llamada con los 2.371 in / 213 out reales del reviewer).

Decisión owner: "en el plus para los casos más difíciles usa gpt 5.6 sol".
Definición determinista de "difícil" (SSOT `condition_rules.detect_active_rules`,
cero listas de keywords nuevas): regla bariátrica activa O ≥2 reglas clínicas.

Escalera completa del reviewer risk-tier:
  free/guest → Luna · basic → Terra · plus/ultra → Terra (normal) / SOL (difícil)
Basic NUNCA escala a sol (worst-case ~18% del revenue — no rentable).

Contratos que ancla:
  A. plus/ultra + difícil → sol; plus/ultra + no-difícil → terra;
     basic + difícil → terra; free + difícil → luna.
  B. `_is_hard_clinical_profile`: bariátrico → True; ≥2 reglas → True;
     1 regla → False; excepción de detección → False (fail-safe, nunca crash).
  C. Knob `MEALFIT_REVIEWER_RISK_MODEL_PAID_HARD` override sin redeploy.
  D. El fail-safe sin OPENAI_API_KEY cubre sol igual que luna/terra (flash).
  E. Marker bumpeado.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

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
        "MEALFIT_REVIEWER_RISK_MODEL_PAID_HARD",
    ):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    import db

    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: 1)
    yield go
    go._CLINICAL_MODEL_GUARD_WARNED.clear()


def _rules(n, with_bariatric=False):
    out = [SimpleNamespace(id=f"rule_{i}") for i in range(n)]
    if with_bariatric:
        out.append(SimpleNamespace(id="bariatric"))
    return out


# ------------------------------------------------------------------
# A. Escalera por tier × dificultad
# ------------------------------------------------------------------

@pytest.mark.parametrize("tier,expected", [
    ("plus", "gpt-5.6-sol"),
    ("ultra", "gpt-5.6-sol"),
    ("basic", "gpt-5.6-terra"),   # basic jamás escala a sol
])
def test_a_hard_profile_by_tier(_go, monkeypatch, tier, expected):
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: tier)
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: True)
    assert _go._reviewer_model_name(_RISK_FORM) == expected
    # Resolver al esperado del tier+dificultad jamás alerta desvío.
    assert not _go._CLINICAL_MODEL_GUARD_WARNED


def test_a2_plus_normal_stays_terra(_go, monkeypatch):
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "plus")
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: False)
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-terra"


def test_a3_free_hard_stays_luna(_go, monkeypatch):
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "gratis")
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: True)
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-luna"


# ------------------------------------------------------------------
# B. Detector de dificultad (SSOT detect_active_rules)
# ------------------------------------------------------------------

def test_b_bariatric_is_hard(_go, monkeypatch):
    import condition_rules

    monkeypatch.setattr(condition_rules, "detect_active_rules",
                        lambda fd: _rules(0, with_bariatric=True))
    assert _go._is_hard_clinical_profile(_RISK_FORM) is True


def test_b2_two_rules_are_hard(_go, monkeypatch):
    import condition_rules

    monkeypatch.setattr(condition_rules, "detect_active_rules", lambda fd: _rules(2))
    assert _go._is_hard_clinical_profile(_RISK_FORM) is True


def test_b3_single_rule_is_not_hard(_go, monkeypatch):
    import condition_rules

    monkeypatch.setattr(condition_rules, "detect_active_rules", lambda fd: _rules(1))
    assert _go._is_hard_clinical_profile(_RISK_FORM) is False


def test_b4_detection_failure_is_failsafe(_go, monkeypatch):
    import condition_rules

    def _boom(fd):
        raise RuntimeError("boom")

    monkeypatch.setattr(condition_rules, "detect_active_rules", _boom)
    assert _go._is_hard_clinical_profile(_RISK_FORM) is False, (
        "la duda jamás encarece ni rompe la resolución del gate clínico"
    )


# ------------------------------------------------------------------
# C. Knob override
# ------------------------------------------------------------------

def test_c_hard_knob_override(_go, monkeypatch):
    monkeypatch.setenv("MEALFIT_REVIEWER_RISK_MODEL_PAID_HARD", "gpt-5.6-terra")
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "plus")
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: True)
    assert _go._reviewer_model_name(_RISK_FORM) == "gpt-5.6-terra"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED


# ------------------------------------------------------------------
# D. Fail-safe sin key cubre sol
# ------------------------------------------------------------------

def test_d_sol_without_key_falls_back_to_flash(_go, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(_go, "get_user_tier", lambda uid: "plus")
    monkeypatch.setattr(_go, "_is_hard_clinical_profile", lambda fd: True)
    resolved = _go._reviewer_model_name(_RISK_FORM)
    assert resolved == "glm-5.3-flash"
    assert ("reviewer", "glm-5.3-flash") in _go._CLINICAL_MODEL_GUARD_WARNED


# ------------------------------------------------------------------
# E. Marker + SSOT
# ------------------------------------------------------------------

def test_e_marker_and_ssot():
    assert "P1-REVIEWER-SOL-HARD" in _GO_SRC
    assert "_REVIEWER_SOL_HARD_TIERS = frozenset({\"plus\", \"ultra\"})" in _GO_SRC
    from llm_provider import GPT56_SOL

    assert GPT56_SOL == "gpt-5.6-sol"
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-REVIEWER-SOL-HARD" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-31"
