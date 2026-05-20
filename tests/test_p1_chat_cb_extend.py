"""[P1-CHAT-CB-EXTEND · 2026-05-20] Extiende el LLMCircuitBreaker per-modelo
a los 3 callsites adicionales de `ChatGoogleGenerativeAI(...).invoke(...)`
en `backend/agent.py` que estaban sin gate:

  1. `swap_meal` (línea ~559): swap síncrono con tenacity retry 3×. Pre-fix:
     con Gemini degradado, cada swap pagaba 3 attempts × 2-8s exp backoff
     amplificando la condición. Ahora fail-fast con `LLMCircuitBreakerOpen`
     ANTES del retry loop (router → HTTP 503).

  2. `rag_query_router` (línea ~1283): preprocessing hot-path (UNO por turn
     del chat). Pre-fix: cada chat pagaba `MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S`
     (default 12s) antes del fallback. Ahora con breaker abierto retornamos
     fallback de inmediato (NO raise: rag_query_router es preprocessing y
     nunca debe abortar el chat upstream).

  3. `generate_chat_title_background` (línea ~1186): fire-and-forget thread.
     Pre-fix: 1 invoke Gemini por turn × N concurrent chats amplificaba
     condición durante incidente. Ahora skip silente si breaker abierto
     (title se queda en "Nuevo chat" hasta próximo turn).

El patrón es el mismo en los 3 callsites (espejo de `call_model` /
P1-CHAT-CB · 2026-05-19): gate `can_proceed()` antes del invoke;
`record_success()` post-invoke; discriminación rate-limit (NO cuenta como
CB failure, emit metric + raise `LLMRateLimitedError`) vs failure genuino
(`record_failure()`).

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_chat_cb_extend` matchea
este archivo `test_p1_chat_cb_extend.py`.

Tooltip-anchor: P1-CHAT-CB-EXTEND.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


def _function_body(src: str, signature_re: str, max_chars: int = 8000) -> str:
    """Extrae el cuerpo de una función (heurística): desde el final del match
    de la signature hasta el siguiente `\\ndef ` / `\\nclass ` top-level.
    Cap `max_chars` para no leer la mitad del archivo si el siguiente def
    está muy lejos."""
    m = re.search(signature_re, src)
    assert m is not None, f"signature no encontrada: {signature_re!r}"
    body_start = m.end()
    next_def = re.search(r"\ndef\s|\nclass\s", src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(max_chars, len(src) - body_start)
    )
    return src[body_start:body_end]


# ===========================================================================
# Sección 1 — swap_meal: CB gate + record_success + rate-limit discrimination
# ===========================================================================

def test_swap_meal_has_cb_gate_before_retry(agent_src: str):
    """`swap_meal` consulta `_get_circuit_breaker(...).can_proceed()` ANTES
    del retry loop de tenacity. Con breaker abierto, raise
    `LLMCircuitBreakerOpen` SIN pagar 3× attempts contra Gemini degradado.

    Verifica que el gate aparece ANTES del primer uso de `invoke_with_retry`
    (la función decorada con `@retry`)."""
    body = _function_body(agent_src, r"def\s+swap_meal\s*\(\s*form_data\s*:")

    # CB gate present.
    assert "_get_circuit_breaker(" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no llama a "
        "`_get_circuit_breaker(...)`. Sin gate, retry tenacity 3× golpea "
        "Gemini degradado por cada swap × N concurrent → amplifica condición."
    )
    assert "_swap_cb.can_proceed()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no chequea "
        "`_swap_cb.can_proceed()`. Sin el gate, el breaker registra failures "
        "pero no impide nuevas llamadas mientras está abierto."
    )

    # Order: gate before invoke_with_retry()
    gate_idx = body.find("can_proceed()")
    invoke_idx = body.find("invoke_with_retry()")
    assert 0 <= gate_idx < invoke_idx, (
        "P1-CHAT-CB-EXTEND regresión: el gate `can_proceed()` debe aparecer "
        "ANTES de `invoke_with_retry()`. Si va después, no previene el "
        "retry loop bajo breaker abierto."
    )

    # Raise LLMCircuitBreakerOpen sobre el breaker.
    assert "LLMCircuitBreakerOpen" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no raises "
        "`LLMCircuitBreakerOpen` cuando el gate falla. Sin esa señal, el "
        "router no puede mapear a HTTP 503."
    )


def test_swap_meal_records_success_post_invoke(agent_src: str):
    """`swap_meal` invoca `_swap_cb.record_success()` tras
    `invoke_with_retry()` retornar OK. Reset_timeout se renueva ahí."""
    body = _function_body(agent_src, r"def\s+swap_meal\s*\(\s*form_data\s*:")
    assert "_swap_cb.record_success()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no invoca "
        "`_swap_cb.record_success()` post-invoke. Sin record_success, el "
        "breaker no resetea el contador de failures tras una respuesta OK "
        "→ falsamente sigue abierto."
    )


def test_swap_meal_discriminates_rate_limit(agent_src: str):
    """`swap_meal` discrimina rate-limit (NO cuenta como CB failure) vs
    failure genuino. Para rate-limit: emit metric + raise
    `LLMRateLimitedError`. Para failure: `record_failure()` + fallback."""
    body = _function_body(agent_src, r"def\s+swap_meal\s*\(\s*form_data\s*:")

    assert "_is_rate_limit_error(" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no usa el helper "
        "`_is_rate_limit_error(...)` para discriminar 429. Sin él, los "
        "rate-limits del provider abren el CB falsamente (3 bursts × 30s "
        "→ usuarios legítimos ven 503 falso-positivo)."
    )
    assert "LLMRateLimitedError" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no re-emite como "
        "`LLMRateLimitedError`. El caller necesita distinguir 429 (Retry-"
        "After) de 503 (provider degradado)."
    )
    assert "_swap_cb.record_failure()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `swap_meal` no marca `record_failure()` "
        "en el branch de failure genuino. Sin él, el CB nunca abre y la "
        "defensa-en-profundidad es no-op."
    )


# ===========================================================================
# Sección 2 — rag_query_router: CB gate + graceful fallback (NO raise)
# ===========================================================================

def test_rag_query_router_has_cb_gate(agent_src: str):
    """`rag_query_router` consulta `_get_circuit_breaker(...).can_proceed()`
    ANTES de instanciar `ChatGoogleGenerativeAI`. Con breaker abierto
    retorna fallback `{"skip": False, "query": prompt}` SIN raise — el
    rag_query_router es preprocessing y nunca debe abortar el chat upstream."""
    body = _function_body(agent_src, r"def\s+rag_query_router\s*\(\s*prompt\s*:")

    assert "_router_cb.can_proceed()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `rag_query_router` no chequea "
        "`_router_cb.can_proceed()`. Cada chat paga "
        "`MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S` (default 12s) antes del "
        "fallback bajo provider degradado."
    )

    # NO raise sobre el gate — debe retornar fallback graceful.
    gate_idx = body.find("_router_cb.can_proceed()")
    assert gate_idx >= 0
    # Tras el gate (pseudocódigo: if not _router_cb.can_proceed(): logger.warning(); return {...})
    # buscamos `return {"skip": False, "query": prompt}` en los ~600 chars siguientes.
    post_gate = body[gate_idx : gate_idx + 600]
    assert (
        'return {"skip": False, "query": prompt}' in post_gate
        or "return {'skip': False, 'query': prompt}" in post_gate
    ), (
        "P1-CHAT-CB-EXTEND regresión: tras el gate de `rag_query_router` "
        "debe retornarse el fallback graceful, NO raise. "
        "`rag_query_router` es preprocessing — un raise rompería el chat "
        "upstream cada vez que el provider degrada."
    )


def test_rag_query_router_records_success_post_invoke(agent_src: str):
    """`rag_query_router` invoca `_router_cb.record_success()` tras
    `router_llm.invoke(...)` retornar OK."""
    body = _function_body(agent_src, r"def\s+rag_query_router\s*\(\s*prompt\s*:")
    assert "_router_cb.record_success()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `rag_query_router` no marca "
        "`record_success()` post-invoke. El CB no resetea failures."
    )


def test_rag_query_router_discriminates_rate_limit(agent_src: str):
    """`rag_query_router` discrimina rate-limit antes de `record_failure()`."""
    body = _function_body(agent_src, r"def\s+rag_query_router\s*\(\s*prompt\s*:")
    assert "_is_rate_limit_error(" in body, (
        "P1-CHAT-CB-EXTEND regresión: `rag_query_router` no usa "
        "`_is_rate_limit_error(...)`. 429 abriría el CB falsamente y cada "
        "chat se degradaría sin RAG durante 30s tras 3 bursts del provider."
    )
    assert "_router_cb.record_failure()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `rag_query_router` no marca "
        "`record_failure()` en el branch de failure genuino."
    )


# ===========================================================================
# Sección 3 — generate_chat_title_background: CB gate + skip silente
# ===========================================================================

def test_title_background_has_cb_gate(agent_src: str):
    """`generate_chat_title_background` consulta
    `_title_cb.can_proceed()` ANTES de instanciar `title_llm`. Si gate
    falla, return silente (NO raise — corre en background thread, raise
    solo se loguea sin afectar nada)."""
    body = _function_body(
        agent_src,
        r"def\s+generate_chat_title_background\s*\(",
        max_chars=12000,
    )

    assert "_title_cb.can_proceed()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `generate_chat_title_background` no "
        "chequea `_title_cb.can_proceed()`. Bajo incidente, 1 invoke × N "
        "concurrent chats amplifica la condición sin necesidad (title es "
        "cosmético, NO crítico para el chat-flow)."
    )

    # Tras el gate, debe haber un `return` (skip silente) ANTES del invoke.
    gate_idx = body.find("_title_cb.can_proceed()")
    invoke_idx = body.find("title_llm.invoke(")
    assert gate_idx >= 0 and invoke_idx > gate_idx, (
        "P1-CHAT-CB-EXTEND regresión: el gate debe aparecer ANTES de "
        "`title_llm.invoke(...)`."
    )
    # Entre el gate y el invoke, debe haber un `return` (skip path).
    between = body[gate_idx:invoke_idx]
    assert "return" in between, (
        "P1-CHAT-CB-EXTEND regresión: tras el gate de "
        "`generate_chat_title_background` debe haber `return` (skip "
        "silente) ANTES del invoke. Sin él, el gate no previene la llamada."
    )


def test_title_background_records_success_post_invoke(agent_src: str):
    """`generate_chat_title_background` invoca `_title_cb.record_success()`
    tras `title_llm.invoke(...)` retornar OK."""
    body = _function_body(
        agent_src,
        r"def\s+generate_chat_title_background\s*\(",
        max_chars=12000,
    )
    assert "_title_cb.record_success()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `generate_chat_title_background` no "
        "marca `record_success()` post-invoke. El CB no resetea failures."
    )


def test_title_background_discriminates_rate_limit(agent_src: str):
    """`generate_chat_title_background` discrimina rate-limit del provider
    (NO cuenta como CB failure) antes de `record_failure()`."""
    body = _function_body(
        agent_src,
        r"def\s+generate_chat_title_background\s*\(",
        max_chars=12000,
    )
    assert "_is_rate_limit_error(" in body, (
        "P1-CHAT-CB-EXTEND regresión: `generate_chat_title_background` no "
        "usa `_is_rate_limit_error(...)`. 429 abriría el CB falsamente."
    )
    assert "_title_cb.record_failure()" in body, (
        "P1-CHAT-CB-EXTEND regresión: `generate_chat_title_background` no "
        "marca `record_failure()` en el branch de failure genuino."
    )


# ===========================================================================
# Sección 4 — tooltip-anchor presente ≥3× (uno por callsite)
# ===========================================================================

def test_tooltip_anchor_present(agent_src: str):
    """El marker `P1-CHAT-CB-EXTEND` aparece ≥3 veces en agent.py (uno por
    callsite cubierto: swap_meal, rag_query_router, generate_chat_title_background).
    Convención del repo: tooltip-anchor en el código fuente permite que un
    rename del slug rompa este test ANTES de cambiar producción."""
    count = agent_src.count("P1-CHAT-CB-EXTEND")
    assert count >= 3, (
        f"P1-CHAT-CB-EXTEND regresión: tooltip-anchor aparece {count}× "
        f"en agent.py, esperado ≥3 (uno por callsite cubierto). Si un "
        f"rename del slug ocurrió sin actualizar este test, restaurar el "
        f"marker en los 3 callsites: swap_meal, rag_query_router, "
        f"generate_chat_title_background."
    )
