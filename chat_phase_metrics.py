# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1001 · G-01 · 2026-11-14] Telemetría de fases del turno y TTFB real (servidor + cliente).

Cierra el gap G-01 del plan de arquitectura del coach:
- Emite UNA sola fila por turno en `pipeline_metrics` (`node='chat_turn_phases'`) con el `corr`
  del request y las marcas de tiempo: recibido, preflight/contexto listo (incluyendo
  sentimiento+router y RAG), primera palabra emitida por SSE, cada llamada al modelo, cada
  herramienta (nombre y duración en ms), `done` y persistencia post-done.
- Recibe la medición real del navegador/teléfono (`envío → primer texto pintado`) desde
  `POST /api/chat/client-metrics` correlacionada con el mismo `corr` (`node='chat_client_perf'`).
- Ofrece `query_chat_phase_percentiles(hours)` (SQL directo sobre `pipeline_metrics`) para
  consultar p50/p90 de cada fase y primera palabra en servidor y cliente.

Fail-open absoluto: ningún fallo de DB o formato puede romper un turno del chat.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from correlation import get_correlation_id, new_correlation_id
from knobs import _env_bool

logger = logging.getLogger("graph_orchestrator")

_NODE_TURN_PHASES = "chat_turn_phases"
_NODE_CLIENT_PERF = "chat_client_perf"
_CORR_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_MAX_LATENCY_MS = 600_000

_ACTIVE_TRACKER_VAR: contextvars.ContextVar[Optional["ChatTurnPhaseTracker"]] = (
    contextvars.ContextVar("mealfit_chat_phase_tracker", default=None)
)

_session_lock = threading.Lock()
_ACTIVE_BY_SESSION: dict[str, "ChatTurnPhaseTracker"] = {}


def phase_metrics_enabled() -> bool:
    """Knob `MEALFIT_CHAT_PHASE_METRICS` (default True)."""
    return _env_bool("MEALFIT_CHAT_PHASE_METRICS", True)


def _db_safe_user_id(user_id: Optional[str], session_id: Optional[str]) -> Optional[str]:
    if not user_id or user_id == "guest" or user_id == session_id:
        return None
    return str(user_id)


def _clamp_ms(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    try:
        iv = int(round(float(val)))
    except Exception:
        return None
    if iv < 0:
        return 0
    if iv > _MAX_LATENCY_MS:
        return _MAX_LATENCY_MS
    return iv


def sanitize_corr(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned if _CORR_RE.match(cleaned) else None


class ChatTurnPhaseTracker:
    """Acumulador thread-safe por turno para emitir una sola fila `chat_turn_phases`."""

    def __init__(
        self,
        *,
        user_id: Optional[str],
        session_id: str,
        corr: Optional[str] = None,
        owned_by_router: bool = False,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id or "default_session"
        self.corr = sanitize_corr(corr) or sanitize_corr(get_correlation_id()) or new_correlation_id()
        self.owned_by_router = owned_by_router
        self.t0_mono = time.monotonic()
        self.received_at = datetime.now(timezone.utc).isoformat()
        self._lock = threading.Lock()
        self._emitted = False

        self.preflight_ms: Optional[int] = None
        self.sentiment_rag_router_ms: Optional[int] = None
        self.rag_ms: Optional[int] = None
        self.context_ready_ms: Optional[int] = None
        self.first_word_ms: Optional[int] = None
        self.first_word_source: Optional[str] = None
        self.done_ms: Optional[int] = None
        self.persist_ms: Optional[int] = None
        self.chunks_yielded: int = 0
        self.outcome: str = "ok"
        self.model: Optional[str] = None
        self.llm_calls: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def elapsed_ms(self) -> int:
        return max(0, int((time.monotonic() - self.t0_mono) * 1000))

    def set_phase_ms(self, field: str, duration_ms: int) -> None:
        clamped = _clamp_ms(duration_ms)
        if clamped is None:
            return
        with self._lock:
            if hasattr(self, field):
                setattr(self, field, clamped)

    def mark_context_ready(self) -> None:
        ms = self.elapsed_ms()
        with self._lock:
            if self.context_ready_ms is None:
                self.context_ready_ms = ms

    def mark_first_word(self, source: str = "stream_chunk") -> None:
        ms = self.elapsed_ms()
        with self._lock:
            self.chunks_yielded += 1
            if self.first_word_ms is None:
                self.first_word_ms = ms
                self.first_word_source = str(source or "stream_chunk")[:32]

    def record_llm_call(
        self,
        *,
        model: str,
        duration_ms: int,
        fallback_used: bool = False,
        input_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        with self._lock:
            entry: dict[str, Any] = {
                "idx": len(self.llm_calls) + 1,
                "model": str(model or "unknown")[:80],
                "duration_ms": _clamp_ms(duration_ms) or 0,
                "fallback_used": bool(fallback_used),
            }
            if input_tokens is not None:
                entry["input_tokens"] = int(input_tokens)
            if cached_tokens is not None:
                entry["cached_tokens"] = int(cached_tokens)
            if output_tokens is not None:
                entry["output_tokens"] = int(output_tokens)
            self.llm_calls.append(entry)

    def record_tool(
        self,
        name: str,
        duration_ms: int,
        *,
        guard_blocked: bool = False,
    ) -> None:
        with self._lock:
            entry: dict[str, Any] = {
                "name": str(name or "unknown")[:80],
                "duration_ms": _clamp_ms(duration_ms) or 0,
            }
            if guard_blocked:
                entry["guard_blocked"] = True
            self.tools.append(entry)

    def mark_done(self, outcome: Optional[str] = None, model: Optional[str] = None) -> None:
        ms = self.elapsed_ms()
        with self._lock:
            self.done_ms = ms
            if outcome:
                self.outcome = str(outcome)[:32]
            if model:
                self.model = str(model)[:80]

    def set_outcome(self, outcome: str, model: Optional[str] = None) -> None:
        with self._lock:
            if outcome:
                self.outcome = str(outcome)[:32]
            if model:
                self.model = str(model)[:80]

    def to_metadata(self) -> dict[str, Any]:
        with self._lock:
            total_ms = self.done_ms if self.done_ms is not None else self.elapsed_ms()
            llm_total = sum(c.get("duration_ms", 0) for c in self.llm_calls)
            tools_total = sum(t.get("duration_ms", 0) for t in self.tools)
            return {
                "corr": self.corr,
                "model": self.model,
                "outcome": self.outcome,
                "received_at": self.received_at,
                "preflight_ms": self.preflight_ms,
                "sentiment_rag_router_ms": self.sentiment_rag_router_ms,
                "rag_ms": self.rag_ms,
                "context_ready_ms": self.context_ready_ms,
                "first_word_ms": self.first_word_ms,
                "first_word_source": self.first_word_source,
                "done_ms": total_ms,
                "persist_ms": self.persist_ms,
                "llm_calls_count": len(self.llm_calls),
                "llm_total_ms": llm_total,
                "llm_calls": list(self.llm_calls),
                "tools_count": len(self.tools),
                "tools_total_ms": tools_total,
                "tools": list(self.tools),
                "chunks_yielded": self.chunks_yielded,
            }

    def emit_once(
        self,
        *,
        outcome: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if self._emitted:
                return False
            self._emitted = True
            if outcome and self.outcome == "ok":
                self.outcome = str(outcome)[:32]
            elif outcome and outcome != "ok":
                self.outcome = str(outcome)[:32]
            if model_name and not self.model:
                self.model = str(model_name)[:80]
            if self.done_ms is None:
                self.done_ms = max(0, int((time.monotonic() - self.t0_mono) * 1000))

        release_turn_tracker(self)
        if not phase_metrics_enabled():
            return False

        meta = self.to_metadata()
        total_dur = int(meta.get("done_ms") or 0)
        try:
            from db_core import execute_sql_write

            execute_sql_write(
                """
                INSERT INTO pipeline_metrics
                    (user_id, session_id, node, duration_ms, retries,
                     tokens_estimated, confidence, metadata)
                VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)
                """,
                (
                    _db_safe_user_id(self.user_id, self.session_id),
                    self.session_id,
                    _NODE_TURN_PHASES,
                    total_dur,
                    json.dumps(meta, ensure_ascii=False),
                ),
            )
            return True
        except Exception as exc:
            logger.debug("[G-01] emit chat_turn_phases falló (best-effort): %r", exc)
            return False


def start_turn_tracker(
    *,
    user_id: Optional[str],
    session_id: str,
    corr: Optional[str] = None,
    owned_by_router: bool = False,
) -> tuple[ChatTurnPhaseTracker, contextvars.Token]:
    """Crea un tracker por turno y lo enlaza al ContextVar y al índice por `session_id`."""
    tracker = ChatTurnPhaseTracker(
        user_id=user_id,
        session_id=session_id,
        corr=corr,
        owned_by_router=owned_by_router,
    )
    tok = _ACTIVE_TRACKER_VAR.set(tracker)
    with _session_lock:
        _ACTIVE_BY_SESSION[tracker.session_id] = tracker
    return tracker, tok


def get_active_turn_tracker(session_id: Optional[str] = None) -> Optional[ChatTurnPhaseTracker]:
    """Obtiene el tracker del ContextVar activo, o hace fallback seguro por `session_id`."""
    tracker = _ACTIVE_TRACKER_VAR.get()
    if tracker is not None:
        if session_id is None or tracker.session_id == session_id:
            return tracker
    if session_id:
        with _session_lock:
            return _ACTIVE_BY_SESSION.get(str(session_id))
    return None


def release_turn_tracker(
    tracker: Optional[ChatTurnPhaseTracker],
    token: Optional[contextvars.Token] = None,
) -> None:
    if token is not None:
        try:
            _ACTIVE_TRACKER_VAR.reset(token)
        except Exception:
            pass
    if tracker is not None:
        with _session_lock:
            if _ACTIVE_BY_SESSION.get(tracker.session_id) is tracker:
                _ACTIVE_BY_SESSION.pop(tracker.session_id, None)


def extract_usage_counts(response: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extrae `(input_tokens, cached_tokens, output_tokens)` de un AIMessage sin lanzar."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None, None, None
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    cached_tok = None
    details = usage.get("input_token_details")
    if isinstance(details, dict):
        cached_tok = details.get("cache_read")
    return (
        int(in_tok) if isinstance(in_tok, (int, float)) else None,
        int(cached_tok) if isinstance(cached_tok, (int, float)) else None,
        int(out_tok) if isinstance(out_tok, (int, float)) else None,
    )


def record_client_chat_perf(
    *,
    user_id: Optional[str],
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persiste la medición del cliente (`envío → primer texto pintado`) ligada por `corr`."""
    corr = sanitize_corr(payload.get("corr"))
    first_chunk_ms = _clamp_ms(payload.get("first_chunk_ms") if "first_chunk_ms" in payload else payload.get("ttfb_ms"))
    first_paint_ms = _clamp_ms(payload.get("first_paint_ms"))
    stream_total_ms = _clamp_ms(payload.get("stream_total_ms"))
    chunk_count = max(0, min(100_000, int(payload.get("chunk_count") or 0))) if not isinstance(payload.get("chunk_count"), bool) else 0
    is_call_mode = bool(payload.get("is_call_mode"))
    raw_platform = str(payload.get("platform") or "").strip()[:32] or None

    primary_ms = first_paint_ms if first_paint_ms is not None else (first_chunk_ms if first_chunk_ms is not None else (stream_total_ms or 0))
    meta = {
        "corr": corr,
        "first_chunk_ms": first_chunk_ms,
        "first_paint_ms": first_paint_ms,
        "stream_total_ms": stream_total_ms,
        "chunk_count": chunk_count,
        "is_call_mode": is_call_mode,
        "platform": raw_platform,
    }
    if not phase_metrics_enabled():
        return {"recorded": False, "corr": corr}

    try:
        from db_core import execute_sql_write

        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (
                _db_safe_user_id(user_id, session_id),
                session_id,
                _NODE_CLIENT_PERF,
                int(primary_ms),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        return {"recorded": True, "corr": corr}
    except Exception as exc:
        logger.debug("[G-01] emit chat_client_perf falló (best-effort): %r", exc)
        return {"recorded": False, "corr": corr}


def query_chat_phase_percentiles(hours: int = 168) -> dict[str, Any]:
    """Calcula p50 y p90 de cada fase en servidor y del primer texto pintado en cliente."""
    window_h = max(1, min(720, int(hours) if hours else 168))
    from db_core import execute_sql_query

    srv_row = execute_sql_query(
        """
        SELECT
            COUNT(*)::int AS turns,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'context_ready_ms', '')::numeric))::numeric)::int AS context_ready_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'context_ready_ms', '')::numeric))::numeric)::int AS context_ready_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'sentiment_rag_router_ms', '')::numeric))::numeric)::int AS sentiment_rag_router_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'sentiment_rag_router_ms', '')::numeric))::numeric)::int AS sentiment_rag_router_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'rag_ms', '')::numeric))::numeric)::int AS rag_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'rag_ms', '')::numeric))::numeric)::int AS rag_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_word_ms', '')::numeric))::numeric)::int AS server_first_word_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_word_ms', '')::numeric))::numeric)::int AS server_first_word_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'llm_total_ms', '')::numeric))::numeric)::int AS llm_total_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'llm_total_ms', '')::numeric))::numeric)::int AS llm_total_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'tools_total_ms', '')::numeric))::numeric)::int AS tools_total_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'tools_total_ms', '')::numeric))::numeric)::int AS tools_total_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_ms))::numeric)::int AS done_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY duration_ms))::numeric)::int AS done_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'persist_ms', '')::numeric))::numeric)::int AS persist_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'persist_ms', '')::numeric))::numeric)::int AS persist_p90_ms
        FROM public.pipeline_metrics
        WHERE node = %s
          AND created_at > NOW() - make_interval(hours => %s)
        """,
        (_NODE_TURN_PHASES, window_h),
        fetch_one=True,
    ) or {}

    cli_row = execute_sql_query(
        """
        SELECT
            COUNT(*)::int AS samples,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_chunk_ms', '')::numeric))::numeric)::int AS client_first_chunk_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_chunk_ms', '')::numeric))::numeric)::int AS client_first_chunk_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_paint_ms', '')::numeric))::numeric)::int AS client_first_paint_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'first_paint_ms', '')::numeric))::numeric)::int AS client_first_paint_p90_ms,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(metadata->>'stream_total_ms', '')::numeric))::numeric)::int AS client_total_p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(metadata->>'stream_total_ms', '')::numeric))::numeric)::int AS client_total_p90_ms
        FROM public.pipeline_metrics
        WHERE node = %s
          AND created_at > NOW() - make_interval(hours => %s)
        """,
        (_NODE_CLIENT_PERF, window_h),
        fetch_one=True,
    ) or {}

    tool_rows = execute_sql_query(
        """
        SELECT
            t->>'name' AS tool_name,
            COUNT(*)::int AS calls,
            ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY NULLIF(t->>'duration_ms', '')::numeric))::numeric)::int AS p50_ms,
            ROUND((percentile_cont(0.90) WITHIN GROUP (ORDER BY NULLIF(t->>'duration_ms', '')::numeric))::numeric)::int AS p90_ms
        FROM public.pipeline_metrics,
             LATERAL jsonb_array_elements(metadata->'tools') AS t
        WHERE node = %s
          AND created_at > NOW() - make_interval(hours => %s)
          AND jsonb_typeof(metadata->'tools') = 'array'
        GROUP BY 1
        ORDER BY calls DESC, p90_ms DESC
        LIMIT 30
        """,
        (_NODE_TURN_PHASES, window_h),
        fetch_all=True,
    ) or []

    return {
        "window_hours": window_h,
        "server": srv_row,
        "client": cli_row,
        "by_tool": tool_rows,
    }
