"""[P1-CHAT-ERROR-DIFF · 2026-05-19] Test parser-based: el frontend del
Agente diferencia los status HTTP devueltos por el backend (504 timeout LLM
P0-CHAT-LLM-TIMEOUT, 503 circuit breaker abierto P1-CHAT-CB, 402 quota,
401/403 auth, 429 rate-limit, network status=0) con copy es-DO específico Y
expone botón "Reintentar" en errores retryables.

Por qué este test:
    Pre-fix todos los fallos del chat mostraban la misma cadena
    "❌ Error al comunicarse con la IA: ..." sin distinguir entre los
    códigos del backend. El gap fue flagueado en el audit production-
    readiness del Agente (2026-05-19) como P0: el frontend NO ofrecía al
    usuario claridad sobre si era saturación transitoria (esperar) vs
    timeout (reintentar) vs problema de su red. Tests acá anclan:
        1. Helper `_buildAgentErrorMessage` existe en AgentPage.jsx.
        2. Map `_AGENT_ERROR_COPY` tiene entries diferenciadas para los
           status canónicos del backend (504/503/429/402/401/403/0).
        3. handleSend invoca el helper en el bloque !response.ok Y en el
           catch outer (network errors).
        4. El string legacy "❌ Error al comunicarse con la IA" ya no se
           emite hardcodeado en handleSend.
        5. MessageBubble.jsx renderiza el componente `ErrorRetryButton`
           para bubbles con `_isErrorBubble + retryable`.
        6. Los bubbles de error reciben `role="alert"` (a11y defensa-en-
           profundidad mientras el `role="log" aria-live` container-level
           sigue pendiente del audit 2026-05-19).

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_chat_error_diff`
matchea este archivo `test_p1_chat_error_diff.py`.

Tooltip-anchor: P1-CHAT-ERROR-DIFF | audit 2026-05-19 P0 cierre
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_MESSAGE_BUBBLE_JSX = (
    _REPO_ROOT / "frontend" / "src" / "components" / "agent" / "MessageBubble.jsx"
)


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def message_bubble_src() -> str:
    return _MESSAGE_BUBBLE_JSX.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# AgentPage.jsx — helper + invocations
# -----------------------------------------------------------------------------


def test_anchor_present_agent_page(agent_page_src: str):
    assert "P1-CHAT-ERROR-DIFF" in agent_page_src, (
        "P1-CHAT-ERROR-DIFF regresión: anchor textual perdido en AgentPage.jsx."
    )


def test_helper_build_agent_error_message_defined(agent_page_src: str):
    """`_buildAgentErrorMessage` debe declararse en AgentPage.jsx."""
    pattern = re.compile(r"const\s+_buildAgentErrorMessage\s*=\s*\(")
    assert pattern.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: `_buildAgentErrorMessage` no encontrado. "
        "Este helper centraliza el mapeo status→copy + retryable flag. Si lo "
        "renombraste, actualiza este test."
    )


def test_error_copy_map_covers_canonical_status_codes(agent_page_src: str):
    """`_AGENT_ERROR_COPY` debe declarar entries para 504, 503, 429, 402,
    401, 403 y 0 (network).

    Razón: cada uno representa una clase semántica distinta del backend:
      - 504: P0-CHAT-LLM-TIMEOUT (timeout per-LLM o total-graph)
      - 503: P1-CHAT-CB (circuit breaker abierto)
      - 429: rate-limit
      - 402: cuota mensual agotada (NO retryable)
      - 401/403: auth (NO retryable)
      - 0: network/offline (status=0 sentinela en catch)
    """
    map_match = re.search(
        r"const\s+_AGENT_ERROR_COPY\s*=\s*\{([\s\S]*?)\n\};",
        agent_page_src,
    )
    assert map_match, (
        "P1-CHAT-ERROR-DIFF regresión: `_AGENT_ERROR_COPY` no encontrado."
    )
    body = map_match.group(1)
    required_keys = ["504", "503", "429", "402", "401", "403", "0"]
    missing = [
        k for k in required_keys
        if not re.search(rf"(?m)^\s*{k}\s*:", body)
    ]
    assert not missing, (
        f"P1-CHAT-ERROR-DIFF regresión: faltan entries en _AGENT_ERROR_COPY "
        f"para status: {missing}. Cada uno representa una clase semántica "
        f"distinta del backend — la diferenciación es el contrato del P-fix."
    )


def test_error_copy_distinct_per_status(agent_page_src: str):
    """Los textos de 504 vs 503 vs 402 deben ser textualmente distintos.

    Cierra regresión "alguien copió todos los entries con el mismo texto"
    — si pasa eso, perdemos la diferenciación que el P-fix intenta dar.
    """
    map_match = re.search(
        r"const\s+_AGENT_ERROR_COPY\s*=\s*\{([\s\S]*?)\n\};",
        agent_page_src,
    )
    assert map_match
    body = map_match.group(1)
    texts = {}
    for key in ["504", "503", "402"]:
        m = re.search(
            rf"(?ms)^\s*{key}\s*:\s*\{{[^}}]*?text\s*:\s*['\"]([^'\"]+)['\"]",
            body,
        )
        assert m, f"P1-CHAT-ERROR-DIFF: text para status {key} no encontrado."
        texts[key] = m.group(1).strip()
    assert texts["504"] != texts["503"], (
        "P1-CHAT-ERROR-DIFF: copy de 504 (timeout) idéntico a 503 (saturación) "
        f"— diferenciación perdida. Texts: {texts}"
    )
    assert texts["504"] != texts["402"], (
        "P1-CHAT-ERROR-DIFF: copy de 504 idéntico a 402 (quota) — "
        f"diferenciación perdida. Texts: {texts}"
    )


def test_retryable_flag_quota_and_auth_are_false(agent_page_src: str):
    """402 (quota), 401 y 403 (auth) NO deben ser retryable (reintentar
    no resuelve el problema; mostraría el mismo error)."""
    map_match = re.search(
        r"const\s+_AGENT_ERROR_COPY\s*=\s*\{([\s\S]*?)\n\};",
        agent_page_src,
    )
    assert map_match
    body = map_match.group(1)
    for key in ["402", "401", "403"]:
        entry_re = re.compile(
            rf"(?ms)^\s*{key}\s*:\s*\{{([^}}]*?)\}}"
        )
        m = entry_re.search(body)
        assert m, f"P1-CHAT-ERROR-DIFF: entry {key} no encontrado."
        entry_body = m.group(1)
        retryable_match = re.search(r"retryable\s*:\s*(true|false)", entry_body)
        assert retryable_match, (
            f"P1-CHAT-ERROR-DIFF: entry {key} sin campo `retryable`."
        )
        assert retryable_match.group(1) == "false", (
            f"P1-CHAT-ERROR-DIFF: entry {key} marca retryable=true — "
            f"reintentar 402/401/403 no resuelve el problema (cuota agotada / "
            f"sesión expirada). Esto re-introduciría loops de error."
        )


def test_retryable_flag_timeout_and_saturated_are_true(agent_page_src: str):
    """504 (timeout) y 503 (CB abierto) DEBEN ser retryable; el botón
    da al usuario control para reintentar tras esperar."""
    map_match = re.search(
        r"const\s+_AGENT_ERROR_COPY\s*=\s*\{([\s\S]*?)\n\};",
        agent_page_src,
    )
    assert map_match
    body = map_match.group(1)
    for key in ["504", "503", "0"]:
        entry_re = re.compile(
            rf"(?ms)^\s*{key}\s*:\s*\{{([^}}]*?)\}}"
        )
        m = entry_re.search(body)
        assert m
        entry_body = m.group(1)
        retryable_match = re.search(r"retryable\s*:\s*(true|false)", entry_body)
        assert retryable_match and retryable_match.group(1) == "true", (
            f"P1-CHAT-ERROR-DIFF: entry {key} debería marcar retryable=true."
        )


def test_handlesend_uses_helper_in_response_not_ok(agent_page_src: str):
    """En el bloque `else` del `if (response.ok)`, handleSend debe llamar
    `_buildAgentErrorMessage({ status: response.status, ...})` —
    NO emitir hardcodeado el string legacy."""
    # Busca el patrón post-fix
    pattern = re.compile(
        r"_buildAgentErrorMessage\s*\(\s*\{[\s\S]*?status\s*:\s*response\.status",
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: handleSend no invoca "
        "`_buildAgentErrorMessage({ status: response.status, ... })` en el "
        "bloque !response.ok. Sin esto, el frontend pierde la diferenciación "
        "504/503/etc."
    )


def test_handlesend_uses_helper_in_catch_network(agent_page_src: str):
    """En el `catch (error)` outer (network/offline), handleSend debe
    llamar `_buildAgentErrorMessage({ status: 0, ...})`."""
    pattern = re.compile(
        r"_buildAgentErrorMessage\s*\(\s*\{[\s\S]*?status\s*:\s*0",
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: handleSend no invoca "
        "`_buildAgentErrorMessage({ status: 0, ... })` en el catch outer. "
        "Sin esto, network errors no muestran el copy 'Sin conexión' ni el "
        "botón retry."
    )


def test_legacy_generic_error_string_not_present(agent_page_src: str):
    """El string legacy `❌ Error al comunicarse con la IA: ${errData.detail
    || ''}` NO debe seguir hardcodeado dentro de `setMessages` en handleSend.
    Si vuelve a aparecer, perdemos la diferenciación.

    Nota: el string puede aparecer en COMENTARIOS o documentación; este test
    sólo bloquea su uso en una template literal pasada a setMessages.
    """
    forbidden = re.compile(
        r"setMessages\s*\(\s*prev\s*=>\s*\[\.\.\.prev,\s*\{[^}]*?"
        r"content\s*:\s*`❌\s+Error al comunicarse con la IA",
    )
    assert not forbidden.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: el string legacy "
        "'❌ Error al comunicarse con la IA: ${errData.detail || \\'\\'}' "
        "volvió a aparecer hardcodeado en handleSend. Usar "
        "`_buildAgentErrorMessage(...)` que diferencia por status."
    )


def test_legacy_network_error_string_not_present(agent_page_src: str):
    """El string legacy `❌ Error de conexión al servidor.` NO debe seguir
    hardcodeado en el catch outer."""
    forbidden = re.compile(
        r"setMessages\s*\(\s*prev\s*=>\s*\[\.\.\.prev,\s*\{[^}]*?"
        r"content\s*:\s*['\"]❌\s+Error de conexión al servidor",
    )
    assert not forbidden.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: el string legacy "
        "'❌ Error de conexión al servidor.' volvió a aparecer hardcodeado "
        "en el catch outer. Usar `_buildAgentErrorMessage({status:0,...})`."
    )


def test_onerrorretry_wired_to_messagebubble(agent_page_src: str):
    """El `<MemoizedMessageBubble>` en el map de mensajes debe recibir la
    prop `onErrorRetry`. Sin ese wiring el botón "Reintentar" no funciona."""
    pattern = re.compile(r"onErrorRetry\s*=\s*\{")
    assert pattern.search(agent_page_src), (
        "P1-CHAT-ERROR-DIFF regresión: prop `onErrorRetry` no se pasa al "
        "MemoizedMessageBubble. El botón Reintentar quedaría sin handler."
    )


# -----------------------------------------------------------------------------
# MessageBubble.jsx — UI components
# -----------------------------------------------------------------------------


def test_anchor_present_message_bubble(message_bubble_src: str):
    assert "P1-CHAT-ERROR-DIFF" in message_bubble_src, (
        "P1-CHAT-ERROR-DIFF regresión: anchor textual perdido en MessageBubble.jsx."
    )


def test_error_retry_button_component_defined(message_bubble_src: str):
    """`ErrorRetryButton` debe declararse en MessageBubble.jsx."""
    pattern = re.compile(r"const\s+ErrorRetryButton\s*=\s*\(")
    assert pattern.search(message_bubble_src), (
        "P1-CHAT-ERROR-DIFF regresión: componente `ErrorRetryButton` no "
        "encontrado en MessageBubble.jsx."
    )


def test_error_retry_button_rendered_conditionally(message_bubble_src: str):
    """El botón solo se renderiza si `isErrorBubble && msg.retryable && onErrorRetry`."""
    # Match flexible — los tres flags presentes alrededor del render
    pattern = re.compile(
        r"isErrorBubble\s*&&\s*msg\.retryable[\s\S]*?<ErrorRetryButton",
    )
    assert pattern.search(message_bubble_src), (
        "P1-CHAT-ERROR-DIFF regresión: el render de <ErrorRetryButton> no "
        "está gated por `isErrorBubble && msg.retryable`. Sin el gate, el "
        "botón podría aparecer en bubbles de éxito o en errores no-retryables "
        "(402 quota, 401/403 auth)."
    )


def test_role_alert_on_error_bubble(message_bubble_src: str):
    """Bubbles de error deben tener `role="alert"` (a11y baseline — anuncio
    a screen readers). Defensa-en-profundidad mientras el aria-live container
    sigue pendiente."""
    # Match el spread condicional
    pattern = re.compile(
        r"isErrorBubble\s*\?\s*\{\s*role\s*:\s*['\"]alert['\"]",
    )
    assert pattern.search(message_bubble_src), (
        "P1-CHAT-ERROR-DIFF regresión: bubbles de error perdieron `role=\"alert\"`. "
        "Sin esto, screen readers no anuncian el error al usuario."
    )


def test_message_actions_hidden_on_error_bubble(message_bubble_src: str):
    """`MessageActions` (thumbs up/down/regenerate/copy) NO debe renderizarse
    en bubbles de error — esas acciones no aplican a un mensaje de error."""
    # El render de MessageActions debe estar gated por `!isErrorBubble`
    pattern = re.compile(
        r"!msg\.isStreaming\s*&&\s*!isErrorBubble[\s\S]*?<MessageActions",
    )
    assert pattern.search(message_bubble_src), (
        "P1-CHAT-ERROR-DIFF regresión: <MessageActions> ya no está gated "
        "por `!isErrorBubble`. Esto haría aparecer thumbs/regenerate en "
        "bubbles de error, lo cual es confuso UX."
    )
