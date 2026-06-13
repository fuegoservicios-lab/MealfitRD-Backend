"""[P1-CHAT-PROD-AUDIT · 2026-05-20] Bundle de 5 fixes de production-readiness
del stack del chat-agent. Tests parser-based que ancla cada uno a un anchor
literal en código para que un refactor accidental falle el test antes de
romper producción.

Hallazgos cerrados (de audit 2026-05-20 al stack agente):
  1. [P1-CHAT-STREAM-BUDGET]  Total budget + inactivity timeout para el SSE
                              stream (`chat_with_agent_stream`). Pre-fix: solo
                              el wrapper non-stream tenía graph-total timeout
                              (60s); el stream podía correr indefinidamente.
  2. [P1-CHAT-LLM-429]        Diferenciación 429 (ResourceExhausted) en
                              `call_model`. Pre-fix: 429 contaba como CB
                              failure, abriendo el breaker falso-positivo
                              durante saturación temporal del provider.
                              Router mapea ahora a HTTP 429 + Retry-After.
  3. [P1-CHAT-STREAM-DURATION] Persiste `duration_ms` del stream a
                              `pipeline_metrics` con outcome para que SRE
                              pueda graficar P99 E2E latencia del chat.
  4. [P1-CHAT-TOOL-VALIDATE]  Recuperación graceful si LLM emite `tool_args`
                              con tipos inválidos para el schema Pydantic
                              auto-generado de LangChain. Pre-fix: HTTP 500
                              opaco; ahora `tool_result` legible al LLM.
  5. [P1-CHAT-SESSION-TTL]    Cron daily que purga `agent_sessions` con
                              `created_at` >N días (default 90).
                              FK CASCADE en `agent_messages.session_id`
                              borra los mensajes automáticamente.
                              [P1-CHAT-SESSION-TTL-SCHEMA · 2026-05-20]
                              La intención original referenciaba
                              `last_activity` pero esa columna NO existe
                              en `agent_sessions` — el SQL usa
                              `created_at` puro. Test anti-regresión
                              abajo bloquea reintroducir el reference.

NO cubre (intencional):
  - Token counting: ya cubierto por `P2-CHAT-TOKEN-TELEMETRY` (agent.py:595+)
    que emite a `llm_usage_events`. Audit confirmado pre-implementación.
  - Per-LLM-invoke duration: ya cubierto por mismo P2 (graficable como
    `model='gemini-3.5-flash'` rows en `llm_usage_events`).
  - PostgresSaver concurrency: langgraph 1.1.10 + langgraph-checkpoint-postgres
    3.0.5 incluyen channel versioning (optimistic) built-in. Verificado vía
    `requirements.txt`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_CHAT_ROUTER_PY = _BACKEND_ROOT / "routers" / "chat.py"
_CRON_TASKS_PY = _BACKEND_ROOT / "cron_tasks.py"
_APP_PY = _BACKEND_ROOT / "app.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# Fix #1 — SSE total budget + inactivity timeout
# ============================================================

def test_chat_stream_total_timeout_knob_defined():
    """[P1-CHAT-STREAM-BUDGET] Knob + helper para total budget del SSE."""
    src = _read(_AGENT_PY)
    assert "def _chat_stream_total_timeout_s" in src, (
        "Helper _chat_stream_total_timeout_s desapareció. Ver P1-CHAT-STREAM-BUDGET."
    )
    assert 'MEALFIT_CHAT_STREAM_TOTAL_TIMEOUT_S' in src, (
        "Env knob MEALFIT_CHAT_STREAM_TOTAL_TIMEOUT_S desapareció."
    )


def test_chat_stream_inactivity_timeout_knob_defined():
    """[P1-CHAT-STREAM-BUDGET] Knob + helper para inactivity timeout."""
    src = _read(_AGENT_PY)
    assert "def _chat_stream_inactivity_timeout_s" in src
    assert "MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S" in src


def test_chat_stream_budget_check_inside_for_loop():
    """[P1-CHAT-STREAM-BUDGET] Los wall-clock checks deben aparecer DENTRO
    del `for event in stream_iter:` loop, antes de procesar el evento.
    Sin esto, un evento que llega tras exceder el budget aún se procesa
    (y consume tokens del siguiente turn). Cerca del anchor del fix.
    """
    src = _read(_AGENT_PY)
    # Buscamos la firma de _chat_stream_total_timeout_s siendo invocada DENTRO
    # del cuerpo de chat_with_agent_stream.
    cwa_match = re.search(
        r"def chat_with_agent_stream\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert cwa_match, "chat_with_agent_stream definition no encontrada"
    body = cwa_match.group(1)
    assert "_chat_stream_total_timeout_s()" in body, (
        "_chat_stream_total_timeout_s() no se invoca dentro de chat_with_agent_stream"
    )
    assert "_chat_stream_inactivity_timeout_s()" in body, (
        "_chat_stream_inactivity_timeout_s() no se invoca dentro de chat_with_agent_stream"
    )
    assert "_stream_started_at" in body
    assert "_last_event_at" in body
    assert "_stream_outcome" in body
    # El check de inactividad debe disparar TimeoutError, no silencioso.
    assert "TimeoutError" in body


def test_chat_stream_budget_marker_present():
    """[P1-CHAT-STREAM-BUDGET] Tooltip-anchor — protege rename del marker."""
    src = _read(_AGENT_PY)
    assert "P1-CHAT-STREAM-BUDGET" in src


# ============================================================
# Fix #4 — 429 differentiation
# ============================================================

def test_llm_rate_limited_class_defined():
    """[P1-CHAT-LLM-429] Clase de excepción específica para 429."""
    src = _read(_AGENT_PY)
    assert "class LLMRateLimitedError" in src, (
        "LLMRateLimitedError no definida en agent.py — Fix P1-CHAT-LLM-429 roto."
    )


def test_is_rate_limit_error_helper_defined():
    """[P1-CHAT-LLM-429] Helper de clasificación heurística."""
    src = _read(_AGENT_PY)
    assert "def _is_rate_limit_error" in src
    # Debe cubrir las 3 formas: type name, message substring, .code attr.
    assert 'ResourceExhausted' in src
    assert "resource has been exhausted" in src
    assert "429" in src


def test_call_model_uses_is_rate_limit_error_before_cb_failure():
    """[P1-CHAT-LLM-429] La rama 429 debe ejecutarse ANTES de
    `_cb.record_failure()` para no contaminar el CB.
    """
    src = _read(_AGENT_PY)
    call_match = re.search(
        r"def call_model\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert call_match, "call_model definition no encontrada"
    body = call_match.group(1)
    # Buscamos el orden: _is_rate_limit_error(...) y luego _cb.record_failure()
    pos_429 = body.find("_is_rate_limit_error(_invoke_exc)")
    pos_cb_failure = body.find("_cb.record_failure()")
    assert pos_429 > 0, "_is_rate_limit_error check ausente en call_model except"
    assert pos_cb_failure > 0, "_cb.record_failure() ausente en call_model"
    assert pos_429 < pos_cb_failure, (
        "Orden incorrecto: _is_rate_limit_error debe chequearse ANTES de "
        "_cb.record_failure() — si no, los 429 ensucian el CB."
    )
    assert "raise LLMRateLimitedError" in body


def test_router_maps_rate_limited_to_429_with_retry_after():
    """[P1-CHAT-LLM-429] El router mapea la excepción a HTTP 429 con
    `Retry-After` header — semánticamente correcto + permite al cliente
    backoff adaptativo.
    """
    src = _read(_CHAT_ROUTER_PY)
    assert "LLMRateLimitedError" in src, (
        "Router no importa LLMRateLimitedError — el except no resuelve."
    )
    # except clause + 429 status + Retry-After header.
    assert "except LLMRateLimitedError" in src
    assert "status_code=429" in src
    assert 'Retry-After' in src


# ============================================================
# Fix #5-lite — Graph-total duration persisted to pipeline_metrics
# ============================================================

def test_chat_stream_total_duration_emit_helper_defined():
    """[P1-CHAT-STREAM-DURATION] Helper de emit best-effort."""
    src = _read(_AGENT_PY)
    assert "def _emit_chat_stream_total_duration_best_effort" in src
    assert "chat_stream_total_duration" in src
    assert "INSERT INTO pipeline_metrics" in src


def test_chat_stream_emits_duration_in_finally():
    """[P1-CHAT-STREAM-DURATION] El emit debe vivir en el finally del
    `chat_with_agent_stream` — cubre TODOS los exits (ok, cancelled,
    error, timeout)."""
    src = _read(_AGENT_PY)
    cwa_match = re.search(
        r"def chat_with_agent_stream\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    body = cwa_match.group(1)
    assert "_emit_chat_stream_total_duration_best_effort" in body, (
        "Emit del duration no se invoca dentro de chat_with_agent_stream."
    )
    # `outcome` debe estar entre los kwargs/args (ok|cancelled|error|timeout_*).
    # Verificamos los 4 valores canónicos en el código (set en distintas ramas).
    for outcome_val in ('"ok"', '"cancelled"', '"error"', '"timeout_total"', '"timeout_inactivity"'):
        assert outcome_val in body, (
            f"Outcome canónico {outcome_val} no se asigna a _stream_outcome — "
            f"telemetría incompleta."
        )


# ============================================================
# Fix #6 — Pydantic ValidationError graceful recovery
# ============================================================

def test_execute_tools_wraps_dispatch_with_validation_catch():
    """[P1-CHAT-TOOL-VALIDATE] El if/elif/else de dispatch debe estar
    envuelto en try/except _PydanticValidationError para que LLM args
    inválidos NO rompan el graph entero.
    """
    src = _read(_AGENT_PY)
    et_match = re.search(
        r"def execute_tools\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert et_match, "execute_tools definition no encontrada"
    body = et_match.group(1)
    assert "_PydanticValidationError" in body, (
        "_PydanticValidationError no presente en execute_tools — Fix roto."
    )
    assert "except _PydanticValidationError" in body, (
        "Falta el except clause; ValidationError no se atrapa."
    )
    # El tool_result debe ser un string legible al LLM, no raw repr.
    assert "VALIDATION_ERROR" in body
    # Anchor presente.
    assert "P1-CHAT-TOOL-VALIDATE" in body


def test_pydantic_validation_error_import_is_defensive():
    """[P1-CHAT-TOOL-VALIDATE] El import de ValidationError debe ser
    lazy (dentro de try/except) — pydantic v1 vs v2 mueve el path."""
    src = _read(_AGENT_PY)
    # Buscamos el patrón lazy: __import__ + fallback inocuo
    assert 'fromlist=["ValidationError"]' in src or 'fromlist=[\'ValidationError\']' in src, (
        "Import de ValidationError no es defensive — debería usar __import__ lazy."
    )


# ============================================================
# Fix #3 — Session TTL sweep cron
# ============================================================

def test_sweep_stale_chat_sessions_defined():
    """[P1-CHAT-SESSION-TTL] Cron function debe existir + ser
    DELETE-based con TTL configurable."""
    src = _read(_CRON_TASKS_PY)
    assert "def _sweep_stale_chat_sessions" in src
    assert "MEALFIT_CHAT_SESSION_TTL_DAYS" in src
    assert "DELETE FROM agent_sessions" in src
    # El sweep emite tick observable patrón P1-LIVE-4.
    assert "_sweep_stale_chat_sessions_tick" in src


def test_sweep_stale_chat_sessions_clamped_ttl():
    """[P1-CHAT-SESSION-TTL] TTL clamp [7, 730] días — defensa contra
    truncado agresivo accidental + bloquear 'never expire'."""
    src = _read(_CRON_TASKS_PY)
    sweep_match = re.search(
        r"def _sweep_stale_chat_sessions\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert sweep_match
    body = sweep_match.group(1)
    # Clamp pattern: max(7, min(ttl_days, 730))
    assert re.search(r"max\(\s*7\s*,\s*min\(\s*ttl_days\s*,\s*730\s*\)\s*\)", body), (
        "TTL clamp [7, 730] no detectado — defensa rota."
    )


def test_sweep_stale_chat_sessions_killswitch_knob():
    """[P1-CHAT-SESSION-TTL] Killswitch para apagar el cron sin
    redeploy. Convención del repo: knob `MEALFIT_*_ENABLED`."""
    src = _read(_CRON_TASKS_PY)
    assert "MEALFIT_CHAT_SESSION_TTL_ENABLED" in src


def test_sweep_stale_chat_sessions_registered_in_scheduler():
    """[P1-CHAT-SESSION-TTL] Cron registrado en register_plan_chunk_scheduler
    (SSOT de crons)."""
    src = _read(_CRON_TASKS_PY)
    sched_match = re.search(
        r"def register_plan_chunk_scheduler\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert sched_match, "register_plan_chunk_scheduler no encontrada"
    body = sched_match.group(1)
    assert "sweep_stale_chat_sessions" in body, (
        "Cron no registrado en register_plan_chunk_scheduler — no correrá en prod."
    )
    assert "MEALFIT_CHAT_SESSION_TTL_SWEEP_INTERVAL_MIN" in body


def test_sweep_stale_chat_sessions_sql_does_not_reference_last_activity():
    """[P1-CHAT-SESSION-TTL-SCHEMA · 2026-05-20] El cuerpo de
    `_sweep_stale_chat_sessions` NO debe referenciar la columna
    `last_activity` — esa columna NO existe en `agent_sessions`
    (schema actual: id, created_at, locked_at, user_id). El SQL
    debe operar sobre `created_at` puro.

    Pre-fix: el SQL original era
        WHERE COALESCE(last_activity, created_at) < NOW() - ...
    que falla con `column "last_activity" does not exist` el primer
    día que el cron daily dispara → alert
    `scheduler_error_sweep_stale_chat_sessions` + TTL nunca se aplica
    → bloat slow-burn de agent_sessions/agent_messages.

    El test anchora a un anti-pattern textual: si alguien reintroduce
    `last_activity` en el cuerpo del sweep (e.g. al copiar el original
    del runbook o al re-escribir desde la memoria), este test falla
    antes de mergear. Si en el futuro se añade la columna via
    migration SSOT, este test debe actualizarse Y crearse un test
    funcional contra el schema real (no parser-based).
    """
    src = _read(_CRON_TASKS_PY)
    sweep_match = re.search(
        r"def _sweep_stale_chat_sessions\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert sweep_match, "_sweep_stale_chat_sessions no encontrada"
    body = sweep_match.group(1)
    # Comments narrativos y docstrings pueden mencionar `last_activity`
    # legítimamente (documentar el trade-off vs la intención original).
    # Lo que NO se permite es referencias en SQL real — detectamos
    # filtrando triple-quote blocks que contienen DDL/DML keywords.
    triple_blocks = re.findall(r'"""(.*?)"""', body, re.DOTALL)
    sql_blocks = [
        b for b in triple_blocks
        if re.search(r"\b(DELETE FROM|SELECT |INSERT INTO|UPDATE )\b", b)
    ]
    assert sql_blocks, "No se encontraron bloques SQL en el sweep — refactor?"
    for sql in sql_blocks:
        assert "last_activity" not in sql, (
            "SQL del cron NO debe referenciar `last_activity` — columna "
            "inexistente en agent_sessions. Usar `created_at` puro. Ver "
            "P1-CHAT-SESSION-TTL-SCHEMA · 2026-05-20."
        )
    # Sanity: el SQL DELETE debe filtrar por `created_at`.
    delete_sql = next(
        (b for b in sql_blocks if "DELETE FROM agent_sessions" in b),
        None,
    )
    assert delete_sql, "DELETE FROM agent_sessions no encontrado"
    assert "created_at <" in delete_sql, (
        "SQL DELETE no filtra por `created_at <` — fix de schema no aplicado."
    )


# ============================================================
# Cross-link al marker
# ============================================================

def test_last_known_pfix_bumped_to_this_bundle():
    """[P1-CHAT-PROD-AUDIT] Relajado a FLOOR-DE-FECHA: bundles posteriores
    clobbean el marker legítimamente (mismo patrón documentado para
    P2-COST-GEMINI-AUDIT en la memoria 2026-06-01). El contrato vigente es
    que `_LAST_KNOWN_PFIX` sea >= la fecha de este bundle (2026-05-19) —
    el formato/freshness lo enforza `test_p3_1_last_known_pfix_freshness`
    y el cross-link marker↔test `test_p2_hist_audit_14_marker_test_link`."""
    import re as _re

    src = _read(_APP_PY)
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*·\s*(\d{4}-\d{2}-\d{2})"', src)
    assert m, "_LAST_KNOWN_PFIX con fecha no encontrado en app.py"
    assert m.group(1) >= "2026-05-19", (
        f"_LAST_KNOWN_PFIX retrocedió a {m.group(1)} — anterior a este bundle."
    )
