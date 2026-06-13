"""[P1-TOOLS-LLM-HARDENING · 2026-05-20] Extiende las defensas P0-CHAT-LLM-TIMEOUT
+ P1-CHAT-CB-EXTEND + P1-CHAT-LLM-429 + P2-CHAT-TOKEN-TELEMETRY a los 2
callsites Gemini de `backend/tools.py` que estaban sin defensa:

  1. `analyze_preferences_agent` (línea ~56 pre-fix): construía
     `ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", ...)` sin
     `timeout=`, sin CB gate, sin telemetría tokens. Bajo Gemini hang:
     worker thread starvation indefinida.

  2. `execute_modify_single_meal` (línea ~446 pre-fix): mismo patrón +
     dentro de tenacity retry 3× → peor caso 3 cuelgues encadenados.

Adicionalmente, este P-fix cierra el blind-spot del `chat_with_agent`
(non-stream) que NO emitía `chat_stream_total_duration` — solo el stream
lo hacía (P1-CHAT-STREAM-DURATION).

El test es parser-based (mismo patrón que test_p1_chat_cb_extend.py):
escanea source de prod buscando los tokens canónicos del patrón
(`_get_circuit_breaker`, `record_success`, `record_failure`,
`_is_rate_limit_error` lazy via helper, `_emit_llm_usage_event_best_effort`,
`timeout=`). Si alguien revierte un fix sin actualizar el test, falla
en CI.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_tools_llm_hardening`
matchea este archivo. Tooltip-anchor: P1-TOOLS-LLM-HARDENING.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


def _function_body(src: str, signature_re: str, max_chars: int = 12000) -> str:
    """Extrae cuerpo de función (heurística): desde fin del match hasta el
    siguiente `\\ndef ` / `\\nclass ` top-level. Cap `max_chars` defensivo."""
    m = re.search(signature_re, src)
    assert m is not None, f"signature no encontrada: {signature_re!r}"
    body_start = m.end()
    next_def = re.search(r"\ndef\s|\nclass\s", src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(max_chars, len(src) - body_start)
    )
    return src[body_start:body_end]


# ============================================================================
# Sección 0 — Knobs auto-registrados (P3-PREVIEW-MODEL-KNOB compliance)
# ============================================================================

def test_tools_pref_agent_knobs_registered(tools_src: str):
    """`tools.py` define helpers `_tools_pref_agent_model_name()` y
    `_tools_pref_agent_llm_timeout_s()` via `_env_str`/`_env_float`. El
    auto-registro en `_KNOBS_REGISTRY` (convención P3-NEW-D) permite a SRE
    verificar el model + timeout vivos via `/health/version` sin releer
    source. Tooltip-anchor: P1-TOOLS-LLM-HARDENING."""
    assert "def _tools_pref_agent_model_name" in tools_src, (
        "P1-TOOLS-LLM-HARDENING regresión: helper de model name removido."
    )
    assert 'MEALFIT_TOOLS_PREF_AGENT_MODEL' in tools_src, (
        "P1-TOOLS-LLM-HARDENING regresión: env var del knob removida."
    )
    assert "def _tools_pref_agent_llm_timeout_s" in tools_src, (
        "P1-TOOLS-LLM-HARDENING regresión: helper de timeout removido."
    )
    assert 'MEALFIT_TOOLS_PREF_AGENT_LLM_TIMEOUT_S' in tools_src


def test_tools_modify_meal_knobs_registered(tools_src: str):
    """Mismos requisitos que `test_tools_pref_agent_knobs_registered` aplicados
    al callsite `execute_modify_single_meal`. Tooltip-anchor: P1-TOOLS-LLM-HARDENING."""
    assert "def _tools_modify_meal_model_name" in tools_src
    assert 'MEALFIT_TOOLS_MODIFY_MEAL_MODEL' in tools_src
    assert "def _tools_modify_meal_llm_timeout_s" in tools_src
    assert 'MEALFIT_TOOLS_MODIFY_MEAL_LLM_TIMEOUT_S' in tools_src


# ============================================================================
# Sección 1 — analyze_preferences_agent: timeout + CB + 429 disc + telemetry
# ============================================================================

def test_analyze_preferences_has_timeout_kwarg(tools_src: str):
    """`ChatDeepSeek(...)` dentro de `analyze_preferences_agent`
    DEBE pasar `timeout=` (no hardcode). Pre-fix: sin timeout, `.invoke()`
    podía bloquear el worker thread indefinidamente bajo provider hang.
    [P0-DEEPSEEK-MIGRATION · 2026-06-12] constructor renombrado."""
    body = _function_body(
        tools_src, r"def\s+analyze_preferences_agent\s*\(", max_chars=4000
    )
    assert "ChatDeepSeek(" in body
    # timeout= debe ser un kwarg literal en la construcción del cliente.
    assert "timeout=_tools_pref_agent_llm_timeout_s()" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `analyze_preferences_agent` "
        "construye CGGA sin `timeout=_tools_pref_agent_llm_timeout_s()`. "
        "Sin timeout, `.invoke()` puede colgar al worker thread."
    )


def test_analyze_preferences_has_cb_gate(tools_src: str):
    """CB gate `_get_circuit_breaker(...).can_proceed()` ANTES del invoke.
    Si breaker abierto: raise `LLMCircuitBreakerOpen` (resuelto vía lazy
    import en `_tools_get_chat_safety_helpers()`)."""
    body = _function_body(
        tools_src, r"def\s+analyze_preferences_agent\s*\(", max_chars=4000
    )
    assert "_get_circuit_breaker(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `analyze_preferences_agent` "
        "no consulta `_get_circuit_breaker(...)` — sin CB gate, cada "
        "request paga Gemini call incluso con provider degradado."
    )
    assert "can_proceed()" in body
    # El raise va vía la variable `_CBOpen` extraída del helper lazy import.
    assert "_CBOpen(" in body


def test_analyze_preferences_records_success_and_failure(tools_src: str):
    """Post-invoke: `record_success()`. En except: `record_failure()` solo
    si NO es rate-limit (mismo patrón que `call_model` en `agent.py`)."""
    body = _function_body(
        tools_src, r"def\s+analyze_preferences_agent\s*\(", max_chars=4000
    )
    assert "_pref_cb.record_success()" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `_pref_cb.record_success()` "
        "ausente — CB queda atascado en el último fail state."
    )
    assert "_pref_cb.record_failure()" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `_pref_cb.record_failure()` "
        "ausente — provider degradado no abrirá el breaker."
    )


def test_analyze_preferences_discriminates_rate_limit(tools_src: str):
    """429/ResourceExhausted NO debe contar como CB failure. Detecta via
    `_is_rl_err(...)` (extraído del helper lazy) y re-emit como
    `LLMRateLimitedError` que el caller mapea a HTTP 429."""
    body = _function_body(
        tools_src, r"def\s+analyze_preferences_agent\s*\(", max_chars=4000
    )
    assert "_is_rl_err(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `analyze_preferences_agent` no "
        "discrimina rate-limit antes de marcar CB failure. Resultado: 3 "
        "bursts de 429 abren el breaker → 503 falso-positivo."
    )
    assert "_RLErr(" in body
    assert "_emit_rl_metric(" in body


def test_analyze_preferences_emits_usage_event(tools_src: str):
    """Telemetría tokens post-success via `_emit_usage_event(...)` con
    `node='tool_analyze_preferences'`. Pre-fix: costo de esta tool era
    invisible en `llm_usage_events` → cuenta-fantasma del SRE."""
    body = _function_body(
        tools_src, r"def\s+analyze_preferences_agent\s*\(", max_chars=4000
    )
    assert "_emit_usage_event(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: emit de token telemetry "
        "ausente — costos invisibles en `llm_usage_events`."
    )
    assert "node='tool_analyze_preferences'" in body


# ============================================================================
# Sección 2 — execute_modify_single_meal: mismo patrón aplicado
# ============================================================================

def test_modify_meal_has_timeout_kwarg(tools_src: str):
    """Mismo contract que sección 1 aplicado al callsite del modify_meal LLM."""
    body = _function_body(
        tools_src, r"def\s+execute_modify_single_meal\s*\("
    )
    assert "ChatDeepSeek(" in body
    assert "timeout=_tools_modify_meal_llm_timeout_s()" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `execute_modify_single_meal` "
        "construye CGGA sin `timeout=_tools_modify_meal_llm_timeout_s()`. "
        "Dentro de tenacity 3× retry el peor caso es 3 cuelgues encadenados."
    )


def test_modify_meal_has_cb_gate_before_retry(tools_src: str):
    """CB gate ANTES del retry loop (mismo patrón que `swap_meal` en agent.py:
    P1-CHAT-CB-EXTEND). Verifica `_get_circuit_breaker(` aparece antes del
    primer uso de `invoke_with_retry` o `@retry`."""
    body = _function_body(
        tools_src, r"def\s+execute_modify_single_meal\s*\("
    )
    assert "_get_circuit_breaker(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `execute_modify_single_meal` "
        "sin CB gate antes del retry loop."
    )
    cb_pos = body.index("_get_circuit_breaker(")
    # `current_prompt = [modify_prompt]` es la línea que precede al retry
    # decorator post-fix; aseguramos que el CB gate aparece ANTES.
    retry_pos = body.index("@retry")
    assert cb_pos < retry_pos, (
        "P1-TOOLS-LLM-HARDENING regresión: el CB gate DEBE aparecer "
        "ANTES del decorator `@retry` para evitar pagar 3× attempts "
        "cuando el breaker está abierto."
    )


def test_modify_meal_records_success_and_failure(tools_src: str):
    body = _function_body(
        tools_src, r"def\s+execute_modify_single_meal\s*\("
    )
    assert "_modify_cb.record_success()" in body
    assert "_modify_cb.record_failure()" in body


def test_modify_meal_discriminates_rate_limit(tools_src: str):
    body = _function_body(
        tools_src, r"def\s+execute_modify_single_meal\s*\("
    )
    assert "_is_rl_err(" in body
    assert "_RLErr(" in body


def test_modify_meal_emits_usage_event(tools_src: str):
    body = _function_body(
        tools_src, r"def\s+execute_modify_single_meal\s*\("
    )
    assert "_emit_usage_event(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: emit telemetry post-success "
        "ausente en execute_modify_single_meal."
    )
    assert "node='tool_modify_single_meal'" in body


# ============================================================================
# Sección 3 — chat_with_agent (non-stream): emit total duration
# ============================================================================

def test_chat_with_agent_non_stream_emits_total_duration(agent_src: str):
    """`chat_with_agent` (non-stream) DEBE emitir
    `_emit_chat_stream_total_duration_best_effort(...)` con outcome.
    Pre-fix: solo `chat_with_agent_stream` lo emitía. Endpoint
    `/api/chat` (non-stream) sin P99 graphable. Reusamos el helper SSOT
    en lugar de crear node distinto — diferenciable por queries SRE si
    se necesita."""
    body = _function_body(
        agent_src, r"def\s+chat_with_agent\s*\(\s*session_id\s*:", max_chars=20000
    )
    assert "_chat_total_started_at" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: wall-clock start del non-stream "
        "removido."
    )
    assert "_chat_total_outcome" in body
    assert "_emit_chat_stream_total_duration_best_effort(" in body, (
        "P1-TOOLS-LLM-HARDENING regresión: `chat_with_agent` non-stream "
        "no emite total-duration. Sin esto, P99 latency E2E del path "
        "non-stream queda invisible en queries de SRE."
    )
    # outcome 'timeout' cubierto por el except de TimeoutError.
    assert '_chat_total_outcome = "timeout"' in body
    # outcome 'error' cubierto por el except genérico antes de re-raise.
    assert '_chat_total_outcome = "error"' in body


# ============================================================================
# Sección 4 — Helper lazy import (cierra ciclo agent ↔ tools)
# ============================================================================

def test_lazy_import_helper_present(tools_src: str):
    """`_tools_get_chat_safety_helpers()` permite a `tools.py` acceder a
    helpers de `agent.py` SIN crear ciclo al top-level (agent ya importa
    de tools al top-level)."""
    assert "def _tools_get_chat_safety_helpers" in tools_src, (
        "P1-TOOLS-LLM-HARDENING regresión: helper lazy import removido. "
        "Sin esto, import top-level desde agent crearía ciclo."
    )
    # El helper debe importar lazy desde agent (NO al top-level).
    helper_body = _function_body(
        tools_src, r"def\s+_tools_get_chat_safety_helpers\s*\(",
        max_chars=2000,
    )
    assert "from agent import" in helper_body
    assert "_is_rate_limit_error" in helper_body
    assert "LLMCircuitBreakerOpen" in helper_body
    assert "LLMRateLimitedError" in helper_body
    assert "_emit_llm_usage_event_best_effort" in helper_body
