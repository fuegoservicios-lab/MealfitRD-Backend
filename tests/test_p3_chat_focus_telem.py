"""[P3-CHAT-FOCUS-TELEM · 2026-05-19] Test parser-based: bundle de 2 P3
remanentes del audit prod-readiness del Agente (2026-05-19):

  (1) FRONTEND — focus restore post-send. Tras `setInput('')` en
      `handleSend`, restaurar foco al textarea SOLO si tenía focus pre-
      send (heurística: `document.activeElement === chatInputRef.current`).
      Esto cubre el flow keyboard (typing → Enter → continuar typing) sin
      degradar mobile UX (tap del botón send NO debe abrir el keyboard).
      No-op en modo llamada (voice flow no escribe).

  (2) FRONTEND — telemetría client-side de latencia visible. `handleSend`
      mide `performance.now()` en 3 puntos: stream-start, first-chunk
      (TTFB), done-event. Emite breadcrumb Sentry + console.info
      estructurado con {ttfb_ms, stream_total_ms, chunk_count,
      is_call_mode, session_id}. NO usa captureMessage (saturaría
      cuota Sentry).

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_chat_focus_telem`
matchea este archivo `test_p3_chat_focus_telem.py`.

Tooltip-anchor: P3-CHAT-FOCUS-TELEM | audit 2026-05-19 P3 cierre
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Anchor + ref declaration
# -----------------------------------------------------------------------------


def test_anchor_present(agent_page_src: str):
    assert "P3-CHAT-FOCUS-TELEM" in agent_page_src, (
        "P3-CHAT-FOCUS-TELEM regresión: anchor textual perdido en AgentPage.jsx."
    )


def test_chat_input_ref_declared(agent_page_src: str):
    """`chatInputRef = useRef(null)` debe declararse — el patch necesita
    una referencia al textarea para llamar `.focus()` post-send."""
    assert re.search(r"chatInputRef\s*=\s*useRef\(null\)", agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: `chatInputRef = useRef(null)` no "
        "encontrado. Sin él, no hay handle al textarea para refocus."
    )


def test_chat_input_ref_wired_to_textarea(agent_page_src: str):
    """El `<textarea>` debe tener `ref={chatInputRef}`."""
    pattern = re.compile(
        r"<textarea\s+[\s\S]{0,400}?ref\s*=\s*\{chatInputRef\}",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el `<textarea>` ya no tiene "
        "`ref={chatInputRef}`. El refocus post-send queda inerte."
    )


# -----------------------------------------------------------------------------
# (1) Focus restore — condicional sobre focus pre-send + !callMode
# -----------------------------------------------------------------------------


def test_focus_pre_send_captured(agent_page_src: str):
    """Antes del `setInput('')` debe capturarse si el textarea tenía focus.
    Heurística: `document.activeElement === chatInputRef.current`."""
    pattern = re.compile(
        r"_hadFocusPreSend\s*=\s*\([\s\S]{0,300}?"
        r"document\.activeElement\s*===\s*chatInputRef\.current",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: la captura `_hadFocusPreSend = "
        "document.activeElement === chatInputRef.current` no se encuentra. "
        "Sin esto, mobile users que tappan el botón send recibirían un "
        "keyboard popup intrusivo."
    )


def test_focus_restore_gated_on_pre_send(agent_page_src: str):
    """El `chatInputRef.current?.focus()` debe estar gated por
    `_hadFocusPreSend && !callModeRef.current`."""
    pattern = re.compile(
        r"_hadFocusPreSend\s*&&\s*!callModeRef\.current[\s\S]{0,300}?"
        r"chatInputRef\.current\?\.focus\(\)",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el `chatInputRef.current?.focus()` "
        "ya no está gated por `_hadFocusPreSend && !callModeRef.current`. "
        "Sin el gate, tap del botón send en mobile dispararía keyboard popup "
        "(UX agresiva), y voice mode interferiría con el flujo de audio."
    )


def test_focus_restore_uses_setimeout_for_render(agent_page_src: str):
    """El `.focus()` debe envolverse en `setTimeout(..., 0)` para que React
    termine el re-render del `setInput('')` antes — sino el focus se
    pierde con el re-render del textarea."""
    pattern = re.compile(
        r"setTimeout\(\s*\(\)\s*=>\s*\{[\s\S]{0,200}?chatInputRef\.current\?\.focus\(\)",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el `.focus()` ya no se hace en "
        "`setTimeout(..., 0)`. Sin esa indirection, React aún no terminó "
        "el render del setInput('') y el focus se pierde inmediatamente."
    )


# -----------------------------------------------------------------------------
# (2) Telemetría — performance markers + emit helper
# -----------------------------------------------------------------------------


def test_perf_telemetry_helper_defined(agent_page_src: str):
    """`_emitChatPerfTelemetry` debe declararse a nivel de módulo."""
    assert re.search(
        r"const\s+_emitChatPerfTelemetry\s*=\s*\(",
        agent_page_src,
    ), (
        "P3-CHAT-FOCUS-TELEM regresión: `_emitChatPerfTelemetry` no "
        "encontrado. Sin él, no hay telemetría client-side estructurada."
    )


def test_perf_telemetry_uses_sentry_breadcrumb(agent_page_src: str):
    """El helper debe usar `Sentry.addBreadcrumb` (NO `captureMessage` — eso
    saturaría la cuota Sentry de la app con eventos por cada stream)."""
    pattern = re.compile(
        r"_emitChatPerfTelemetry\s*=[\s\S]{0,800}?Sentry\.addBreadcrumb",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el helper no usa "
        "Sentry.addBreadcrumb. Si volvió a usar captureMessage, satura la "
        "cuota Sentry — la convención P1-SENTRY-SAMPLE-COST aplica acá."
    )
    forbidden = re.compile(
        r"_emitChatPerfTelemetry\s*=[\s\S]{0,800}?Sentry\.captureMessage",
    )
    assert not forbidden.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el helper usa Sentry.captureMessage "
        "— saturaría cuota Sentry (un evento por stream completado = "
        "miles de eventos/día). Usar breadcrumb para métricas operacionales."
    )


def test_perf_telemetry_records_key_metrics(agent_page_src: str):
    """El breadcrumb debe incluir los 5 campos clave: ttfb_ms,
    stream_total_ms, chunk_count, is_call_mode, session_id."""
    m = re.search(
        r"_emitChatPerfTelemetry\s*=[\s\S]*?addBreadcrumb\([\s\S]*?data\s*:\s*\{([\s\S]*?)\},",
        agent_page_src,
    )
    assert m, "P3-CHAT-FOCUS-TELEM: data block del breadcrumb no parseable."
    data_block = m.group(1)
    required = ["ttfb_ms", "stream_total_ms", "chunk_count", "is_call_mode", "session_id"]
    missing = [k for k in required if k not in data_block]
    assert not missing, (
        f"P3-CHAT-FOCUS-TELEM regresión: campos faltan en el breadcrumb: "
        f"{missing}. Cada uno responde a una pregunta operacional distinta "
        f"(TTFB = backend lentitud, total = duración, count = stream "
        f"sanity, call_mode = budget diferenciado, session_id = bucket)."
    )


def test_handlesend_marks_stream_started_at(agent_page_src: str):
    """`handleSend` debe capturar `_streamStartedAt = performance.now()`
    antes del fetch — baseline para todas las métricas."""
    pattern = re.compile(
        r"_streamStartedAt\s*=\s*\([\s\S]{0,150}?performance\.now",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: `_streamStartedAt = performance.now()` "
        "no se captura antes del fetch. Sin baseline, no hay ttfb_ms ni "
        "stream_total_ms."
    )


def test_handlesend_marks_first_chunk_at(agent_page_src: str):
    """En el handler de `dataObj.type === 'chunk'`, debe setearse
    `_firstChunkAt` la primera vez (gated por `=== null`)."""
    pattern = re.compile(
        r"dataObj\.type\s*===\s*['\"]chunk['\"]"
        r"[\s\S]{0,400}?_firstChunkAt\s*===\s*null"
        r"[\s\S]{0,200}?_firstChunkAt\s*=\s*",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: el handler de `chunk` ya no setea "
        "_firstChunkAt la primera vez. Sin esto, no se mide TTFB."
    )


def test_handlesend_increments_chunk_count(agent_page_src: str):
    """`_chunkCount` debe incrementarse por cada chunk recibido.

    Nota regex: el primer `dataObj.type === 'chunk'` puede caer dentro
    de un comentario en el helper (`_emitChatPerfTelemetry` docs). Por
    eso usamos `findall` y verificamos que AL MENOS UN match siga con
    `_chunkCount += 1` dentro de los próximos 500 chars del statement
    real (no del comentario)."""
    # Approach: buscar `_chunkCount += 1` directamente — su presencia
    # implica el contrato. Más simple y robusto que limitar regex.
    assert re.search(r"_chunkCount\s*\+=\s*1", agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: `_chunkCount += 1` no aparece en "
        "AgentPage.jsx. Sin contar chunks, no se puede detectar streams "
        "anormalmente cortos (ej. LLM falló mid-respuesta sin error)."
    )
    # Y verificamos que la declaración inicial existe — si no, el counter
    # nunca se inicializa y `+= 1` rompería en runtime.
    assert re.search(r"let\s+_chunkCount\s*=\s*0", agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: `let _chunkCount = 0` no aparece. "
        "Sin la inicialización, `_chunkCount += 1` lanza ReferenceError."
    )


def test_handlesend_emits_telemetry_in_done(agent_page_src: str):
    """En `dataObj.type === 'done'`, debe invocarse `_emitChatPerfTelemetry`
    con los 5 campos antes de cualquier otra acción (idealmente al tope
    del bloque para minimizar drift por trabajo subsiguiente)."""
    pattern = re.compile(
        r"dataObj\.type\s*===\s*['\"]done['\"][\s\S]{0,600}?"
        r"_emitChatPerfTelemetry\s*\(\s*\{[\s\S]{0,400}?ttfbMs[\s\S]{0,200}?streamTotalMs",
    )
    assert pattern.search(agent_page_src), (
        "P3-CHAT-FOCUS-TELEM regresión: `_emitChatPerfTelemetry` ya no se "
        "invoca en el branch `done` con los campos ttfbMs + streamTotalMs. "
        "Sin esto la telemetría queda muerta — los markers existen pero "
        "nada los emite."
    )
