"""[P1-DEEPSEEK-ONLY-RESTORE · 2026-07-04] Test ancla del cierre "DeepSeek único
provider + guard del gate clínico".

Contexto (audit v7 · P1-1): el mecanismo multi-provider de llm_provider.py
(override global `MEALFIT_LLM_MODEL_OVERRIDE` + detección de host Ollama con
inyección de `think:false`) permitía colapsar TODOS los modelos —incluido el
reviewer médico risk-tier— a un provider de test (Gemini flash-lite / Gemma
local). Caso medido 2026-07-04: con el reviewer en un modelo débil, un plan DM2
pasó con carbos +99% sin rechazo. El owner pidió eliminar todo lo relacionado a
Gemini/Gemma y volver a DeepSeek como único provider.

Qué ancla este test:
  A. llm_provider.py NO contiene el plumbing multi-provider (override global
     funcional, helper Ollama, inyección `think`).
  B. El stack DeepSeek queda intacto (constantes, router por tier, inyección
     `thinking` gated por `_is_deepseek_provider`).
  C. El test ancla del fix Ollama eliminado ya no existe (borrado junto al fix).
  D. graph_orchestrator.py tiene el guard `_warn_if_clinical_model_downgraded`
     cableado en AMBOS resolvers clínicos (`_reviewer_model_name` y
     `_fact_checker_model_name`), con el alert_key documentado.
  E. Funcional: un hard-override que degrada el reviewer para un perfil con
     riesgo médico emite WARN + system_alert idempotente (dedup in-process),
     y el path sano (risk-tier pro) NO emite nada.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_LLM_SRC = _read("llm_provider.py")
_GO_SRC = _read("graph_orchestrator.py")
_ALERTS_DOC = _read(os.path.join("docs", "system_alerts_resolution_table.md"))


# ---------------------------------------------------------------------------
# A. Plumbing multi-provider eliminado
# ---------------------------------------------------------------------------

def test_no_functional_model_override():
    """El override global funcional ya no existe (solo puede quedar mencionado
    en comentarios que documentan su eliminación)."""
    assert '_env_str("MEALFIT_LLM_MODEL_OVERRIDE"' not in _LLM_SRC, (
        "el override global MEALFIT_LLM_MODEL_OVERRIDE volvió a llm_provider.py — "
        "permite colapsar el reviewer clínico a un provider de test (P1-DEEPSEEK-ONLY-RESTORE)"
    )


def test_no_ollama_helper():
    assert "def _is_ollama_provider" not in _LLM_SRC
    assert "11434" not in _LLM_SRC


def test_no_ollama_think_injection():
    """La inyección del toggle `think` de Ollama (distinto del `thinking` de
    DeepSeek) fue eliminada de __init__ y de with_structured_output."""
    assert 'setdefault("think",' not in _LLM_SRC


def test_old_multi_provider_ollama_test_deleted():
    assert not os.path.exists(
        os.path.join(_HERE, "test_p3_multi_provider_ollama.py")
    ), "el test ancla del fix Ollama eliminado debe borrarse junto al fix"


# ---------------------------------------------------------------------------
# B. Stack DeepSeek intacto
# ---------------------------------------------------------------------------

def test_deepseek_stack_intact():
    assert 'DEEPSEEK_FLASH = "deepseek-v4-flash"' in _LLM_SRC
    assert 'DEEPSEEK_PRO = "deepseek-v4-pro"' in _LLM_SRC
    assert "def _is_deepseek_provider(" in _LLM_SRC
    assert "def resolve_model_for_tier(" in _LLM_SRC
    # La inyección `thinking` (DeepSeek-specific) sigue gated por provider.
    assert 'setdefault("thinking", {"type": "disabled"})' in _LLM_SRC
    assert "_is_deepseek_provider(base_url)" in _LLM_SRC


# ---------------------------------------------------------------------------
# D. Guard del gate clínico — estructural
# ---------------------------------------------------------------------------

def test_guard_exists_and_wired_in_both_resolvers():
    assert "def _warn_if_clinical_model_downgraded(" in _GO_SRC

    def _fn_body(src: str, name: str) -> str:
        start = src.index(f"def {name}(")
        nxt = src.find("\ndef ", start + 1)
        return src[start:nxt if nxt != -1 else len(src)]

    rev = _fn_body(_GO_SRC, "_reviewer_model_name")
    fc = _fn_body(_GO_SRC, "_fact_checker_model_name")
    assert rev.count("_warn_if_clinical_model_downgraded(") >= 2, (
        "_reviewer_model_name debe invocar el guard en la rama override Y en la risk-tier"
    )
    assert fc.count("_warn_if_clinical_model_downgraded(") >= 2, (
        "_fact_checker_model_name debe invocar el guard en la rama override Y en la risk-tier"
    )


def test_alert_key_emitted_and_documented():
    assert 'alert_key = f"llm_clinical_reviewer_downgraded:{node}"' in _GO_SRC
    assert "llm_clinical_reviewer_downgraded" in _ALERTS_DOC, (
        "el alert_key debe tener row en docs/system_alerts_resolution_table.md "
        "(contrato P2-AUDIT-4)"
    )


# ---------------------------------------------------------------------------
# E. Guard del gate clínico — funcional
# ---------------------------------------------------------------------------

_RISK_FORM = {"allergies": ["maní"], "medicalConditions": []}


@pytest.fixture()
def _go(monkeypatch):
    import graph_orchestrator as go

    go._CLINICAL_MODEL_GUARD_WARNED.clear()
    monkeypatch.delenv("MEALFIT_REVIEWER_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_FACT_CHECKER_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_REVIEWER_RISK_TIER_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_FACT_CHECKER_RISK_TIER_MODEL", raising=False)
    yield go
    go._CLINICAL_MODEL_GUARD_WARNED.clear()


def _capture_writes(monkeypatch):
    import db

    writes = []

    def _fake_write(sql, params=None, **kwargs):
        writes.append((sql, params))
        return 1

    monkeypatch.setattr(db, "execute_sql_write", _fake_write)
    return writes


def test_healthy_risk_tier_no_alert(_go, monkeypatch):
    """Perfil con riesgo + config default → pro, sin warn ni alert."""
    writes = _capture_writes(monkeypatch)
    resolved = _go._reviewer_model_name(_RISK_FORM)
    assert resolved == "deepseek-v4-pro"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED
    assert not writes


def test_override_downgrade_emits_alert_once(_go, monkeypatch):
    """Hard-override a flash con perfil de riesgo → WARN + system_alert una
    sola vez por proceso por (nodo, modelo)."""
    writes = _capture_writes(monkeypatch)
    monkeypatch.setenv("MEALFIT_REVIEWER_MODEL", "deepseek-v4-flash")

    resolved = _go._reviewer_model_name(_RISK_FORM)
    assert resolved == "deepseek-v4-flash", "el knob sigue ganando (guard observacional)"
    assert ("reviewer", "deepseek-v4-flash") in _go._CLINICAL_MODEL_GUARD_WARNED
    assert len(writes) == 1
    sql, params = writes[0]
    assert "system_alerts" in sql
    assert params[0] == "llm_clinical_reviewer_downgraded:reviewer"

    # Segunda resolución idéntica → dedup, cero writes nuevos.
    _go._reviewer_model_name(_RISK_FORM)
    assert len(writes) == 1


def test_risk_tier_knob_downgrade_also_alerts(_go, monkeypatch):
    """El knob risk-tier apuntado a un modelo débil también dispara el guard
    (no solo el hard-override)."""
    writes = _capture_writes(monkeypatch)
    monkeypatch.setenv("MEALFIT_FACT_CHECKER_RISK_TIER_MODEL", "deepseek-v4-flash")
    resolved = _go._fact_checker_model_name(_RISK_FORM)
    assert resolved == "deepseek-v4-flash"
    assert ("fact_checker", "deepseek-v4-flash") in _go._CLINICAL_MODEL_GUARD_WARNED
    assert len(writes) == 1
    assert writes[0][1][0] == "llm_clinical_reviewer_downgraded:fact_checker"


def test_no_risk_profile_never_warns(_go, monkeypatch):
    """Perfil SIN riesgo médico: flash es el default correcto — jamás alert."""
    writes = _capture_writes(monkeypatch)
    monkeypatch.setenv("MEALFIT_REVIEWER_MODEL", "deepseek-v4-flash")
    resolved = _go._reviewer_model_name({"allergies": [], "medicalConditions": []})
    assert resolved == "deepseek-v4-flash"
    assert not _go._CLINICAL_MODEL_GUARD_WARNED
    assert not writes


def test_alert_emit_failure_is_best_effort(_go, monkeypatch):
    """Fallo del INSERT jamás rompe la resolución de modelo."""
    import db

    def _boom(*a, **k):
        raise RuntimeError("db caída")

    monkeypatch.setattr(db, "execute_sql_write", _boom)
    monkeypatch.setenv("MEALFIT_REVIEWER_MODEL", "deepseek-v4-flash")
    resolved = _go._reviewer_model_name(_RISK_FORM)
    assert resolved == "deepseek-v4-flash"
