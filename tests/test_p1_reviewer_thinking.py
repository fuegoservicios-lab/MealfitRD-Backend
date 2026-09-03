"""[P1-REVIEWER-THINKING · 2026-07-05] Thinking mode V4 en las 2 superficies de juicio clínico.

Contexto: GLM-5.3 trae razonamiento nativo ON de fábrica; el repo lo apaga globalmente
(P1-PROVIDER-THINKING-DEFAULT — en day-gen multiplicaba latencia >170s → fallback matemático).
Pedido del owner 2026-07-05: usarlo donde aporte veracidad. Decisión: SOLO juicio clínico de
bajo volumen — (1) reviewer médico para perfiles risk-tier, (2) escalada a Pro del corrector
quirúrgico (path post-fallo). JAMÁS day-gen/planner (numérica = motor determinista).

Restricción del API calibrada en vivo (smoke 2026-07-05, v4-pro real): thinking NO soporta el
tool_choice forzado de function_calling → structured output vía method="json_mode" (17.5s,
~2.6k reasoning-tokens ≈ $0.0025/llamada; veredicto ERC correcto con 4 hallazgos KDIGO).
Ambos knobs NACEN OFF (convención medir→actuar). Fail-open al reviewer estándar (nunca a
aprobar) si la rama thinking rompe.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_LP = _read("llm_provider.py")


# ---------------------------------------------------------------------------
# knobs: nacen OFF
# ---------------------------------------------------------------------------

def test_knobs_born_off():
    assert '_env_bool("MEALFIT_REVIEWER_THINKING", False)' in _GO
    assert '_env_bool("MEALFIT_SURGICAL_PRO_THINKING", False)' in _GO
    assert '_env_int("MEALFIT_REVIEWER_THINKING_TIMEOUT_S", 90' in _GO
    # [P2-THINKING-EFFORT] knob de effort del quirúrgico (espejo del reviewer).
    assert '_env_str("MEALFIT_SURGICAL_PRO_THINKING_EFFORT", "")' in _GO
    # [P1-FACTCHECKER-THINKING] tercera superficie (fact-checker FASE 1).
    assert '_env_bool("MEALFIT_FACT_CHECKER_THINKING", False)' in _GO
    assert '_env_str("MEALFIT_FACT_CHECKER_THINKING_EFFORT", "")' in _GO
    assert '_env_int("MEALFIT_FACT_CHECKER_THINKING_TIMEOUT_S", 60' in _GO


# ---------------------------------------------------------------------------
# superficie 1: reviewer clínico risk-tier
# ---------------------------------------------------------------------------

def test_reviewer_thinking_gated_to_medical_risk():
    # [P1-REVIEWER-TIER-MODELS · 2026-07-31] El gate añade `and not _rev_is_openai`:
    # `extra_body.thinking` es GLM-only — con Luna/Terra (reviewer por tier) la rama
    # se salta y el modelo OpenAI razona nativo. El anchor sigue la línea que DECIDE.
    i = _GO.index("_rev_thinking = bool(REVIEWER_THINKING_ENABLED and _profile_has_medical_risk(form_data)")
    win = _GO[i:i + 1100]
    assert "and not _rev_is_openai)" in _GO[i:i + 220], \
        "el gate de thinking debe excluir modelos OpenAI (extra_body GLM-only)"
    # [P2-THINKING-EFFORT] el body de thinking se construye en `_think_body` para
    # poder inyectar `effort` opcional; extra_body lo referencia por variable.
    assert '_think_body = {"type": "enabled"}' in win
    assert 'extra_body={"thinking": _think_body}' in win
    assert 'method="json_mode"' in win, \
        "thinking no soporta tool_choice forzado → structured output vía json_mode"
    # la rama estándar (sin riesgo) queda intacta.
    assert ".with_structured_output(ReviewResult)\n" in _GO[i:i + 1400]


def test_reviewer_thinking_timeout_branch():
    assert "if _rev_thinking else 70.0" in _GO, \
        "el cap de _safe_ainvoke sube SOLO en la rama thinking (estándar sigue 70s)"


def test_reviewer_thinking_fail_open_to_standard():
    i = _GO.index("Fail-open hacia el reviewer ESTÁNDAR")
    win = _GO[i:i + 1200]
    assert "_rev_thinking = False" in win
    assert ".with_structured_output(ReviewResult)" in win
    assert "result = await invoke_with_retry()" in win, \
        "tras el fallback se re-invoca el reviewer estándar (gate clínico nunca se salta)"


def test_reviewer_prompt_contract_exists():
    """json_mode depende del contrato JSON literal del system prompt del reviewer."""
    _MR = _read(os.path.join("prompts", "medical_reviewer.py"))
    assert '"approved": true/false' in _MR
    assert '"severity"' in _MR


# ---------------------------------------------------------------------------
# superficie 2: escalada Pro del corrector quirúrgico
# ---------------------------------------------------------------------------

def test_surgical_pro_thinking_branch():
    # [P1-NET-LUNA · 2026-07-31] El gate añade `and not _net_is_openai`: la red
    # default es gpt-5.6-luna (OpenAI) y `extra_body.thinking` es GLM-only.
    i = _GO.index("if SURGICAL_PRO_THINKING_ENABLED and not _net_is_openai:")
    win = _GO[i:i + 1400]
    # [P2-THINKING-EFFORT] body en `_surg_think_body` para inyectar effort opcional.
    assert '_surg_think_body = {"type": "enabled"}' in win
    assert 'extra_body={"thinking": _surg_think_body}' in win
    assert 'with_structured_output(SingleDayPlanModel, method="json_mode")' in win
    # rama por defecto intacta (function_calling implícito).
    assert ".with_structured_output(SingleDayPlanModel)\n" in win


# NOTA: la 3ra superficie (fact-checker clínico, P1-FACTCHECKER-THINKING) tiene su
# propio archivo de regresión: `test_p1_factchecker_thinking.py` (cross-link del marker).


# ---------------------------------------------------------------------------
# contrato del wrapper que las ramas asumen
# ---------------------------------------------------------------------------

def test_wrapper_callsite_override_wins(monkeypatch):
    """[P0-GLM-MIGRATION · 2026-09-02] GLM razona SIEMPRE; lo que el callsite gobierna es el
    ESFUERZO. El `extra_body.thinking.effort` heredado se traduce a `reasoning_effort` y
    gana sobre el default del wrapper (low), que protege day-gen/aux."""
    monkeypatch.setenv("ZAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("MEALFIT_GLM_REASONING_EFFORT", raising=False)
    from llm_provider import ChatGLM
    llm = ChatGLM(model="glm-5.3",
                  extra_body={"thinking": {"type": "enabled", "effort": "high"}})
    assert (llm.extra_body or {}).get("thinking", {}).get("type") == "enabled"
    assert llm.reasoning_effort == "high"
    llm_def = ChatGLM(model="glm-5.3-flash")
    assert (llm_def.extra_body or {}).get("thinking", {}).get("type") == "enabled"
    assert llm_def.reasoning_effort == "low"


def test_wrapper_json_mode_becomes_function_calling():
    """En Z.ai `json_mode`/`json_schema` IGNORAN el esquema (medido 2026-09-02); el wrapper
    reencamina `json_mode` a `function_calling`, que sí lo respeta con razonamiento activo."""
    i = _LP.index('if kwargs["method"] == "json_mode" and _is_glm_provider(')
    assert 'kwargs["method"] = "function_calling"' in _LP[i:i + 200]


def test_daygen_planner_not_touched():
    """La feature JAMÁS activa thinking en day-gen/planner (lección P1-PROVIDER-THINKING-DEFAULT)."""
    i = _GO.index("def _build_day_llm")
    win = _GO[i:i + 1200]
    assert '"type": "enabled"' not in win


def test_marker_anchored_in_source():
    assert "P1-REVIEWER-THINKING" in _GO
