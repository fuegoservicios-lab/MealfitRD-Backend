"""[P2-AUDIT-NEW-2 · 2026-05-12] SSE chat/stream factura quota en `finally`.

Bug original (audit comprehensivo 2026-05-12):
    `/api/chat/stream` event_generator solo llamaba
    `log_api_usage(user_id, "gemini_chat")` DENTRO del callback
    `bg_tasks()` que se ejecutaba SOLO en el path `type=="done"`.
    Si el SSE:
      - Se abortaba a mitad (AbortController, tab close, network drop).
      - El LLM fallaba mid-stream (línea 380 emite `type=error`).
      - GeneratorExit por timeout/cancel.
    → el LLM YA había consumido tokens reales (chunks de texto emitidos)
    pero la quota mensual del usuario NO se decrementaba.

    Vector de explotación: usuario malicioso aborta cada SSE deliberadamente
    tras recibir el ~80% útil del output → tokens del owner gastados sin
    cobrar al user. Mismo gap que P2-LIVE-7 cerró para 5 endpoints
    (analyze/stream, shift-plan, retry-chunk, regenerate-simplified,
    regen-degraded) pero `/chat/stream` quedó fuera.

Fix:
    Refactor del event_generator con `try/except/finally` + 2 flags:

    - `_billed` (False inicial): idempotencia. El finally solo cobra una
      vez aunque haya re-entries (defensivo — finally corre una sola
      vez por ejecución, pero el flag protege contra refactors futuros
      que añadan otra llamada a `log_api_usage`).

    - `_chunk_observed` (False inicial): se activa solo cuando llega
      `type=="chunk"` (texto del LLM principal) o `type=="done"` con
      `response` no-vacío. Evita facturar cuando solo se enviaron
      `progress`/`sentiment` (preamble fast antes de invocar el LLM
      principal; sentiment usa modelo separado de costo marginal y no
      justifica cobrar cuota completa).

    Tres exit paths cubiertos por el finally:
      1. `done` OK → finally factura (cubierto antes por `bg_tasks()`
         pero ahora movido para consistencia).
      2. `GeneratorExit` (cliente abortó) → finally factura si
         `_chunk_observed` (LLM ya emitió texto antes del abort).
      3. `Exception` mid-stream → finally factura si `_chunk_observed`.

    `bg_tasks()` se preserva pero solo cubre summarization + facts
    extraction. `log_api_usage` ya NO vive ahí.

Lo que este test enforza:
    A) `_billed = False` y `_chunk_observed = False` declarados en
       `api_chat_stream` ANTES del `def event_generator()`.
    B) `event_generator` declara `nonlocal _billed, _chunk_observed`
       para poder mutarlas desde el closure.
    C) Existe un branch que activa `_chunk_observed = True` cuando
       `_chunk_type == "chunk"`.
    D) Existe un branch `except GeneratorExit:` que loguea + re-raise
       (NO suprime — el StreamingResponse necesita propagación).
    E) `finally:` block del event_generator contiene la llamada
       idempotente a `log_api_usage(user_id, "gemini_chat")` bajo guard
       `not _billed and _chunk_observed`.
    F) `bg_tasks()` interna YA NO contiene `log_api_usage(user_id,
       "gemini_chat")` — se eliminó de ahí. Si reaparece, hay double-billing.

Tooltip-anchor: P2-AUDIT-NEW-2-SSE-FINALLY-BILLING
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CHAT_PY = _BACKEND_ROOT / "routers" / "chat.py"


@pytest.fixture(scope="module")
def chat_src() -> str:
    return _CHAT_PY.read_text(encoding="utf-8")


def _extract_api_chat_stream_body(src: str) -> str:
    """Extrae el cuerpo del handler `api_chat_stream` (función decorada
    con `@router.post("/stream")`).

    Buscamos directamente `def api_chat_stream(` (más robusto que matchear
    el decorador y luego saltar al `def` siguiente — esto último fallaba
    cuando el `def` viene inmediatamente después del decorador sin newline
    extra, y antes acertaba por accidente con `def api_chat` aguas abajo).
    """
    fn_match = re.search(r"(?:async\s+)?def\s+api_chat_stream\s*\(", src)
    assert fn_match, "No se encontró `def api_chat_stream(` en chat.py."
    body_start = fn_match.end()
    # Cortar al próximo `@router.` o EOF.
    next_decorator = re.search(r"\n@router\.", src[body_start:])
    body_end = (
        body_start + next_decorator.start() if next_decorator else len(src)
    )
    return src[body_start:body_end]


def test_billed_and_chunk_observed_flags_declared(chat_src: str) -> None:
    """A) `_billed = False` y `_chunk_observed = False` declarados en
    `api_chat_stream` ANTES del `def event_generator():`.
    """
    body = _extract_api_chat_stream_body(chat_src)

    event_gen_pos = body.find("def event_generator")
    assert event_gen_pos > 0, "No se encontró `def event_generator` en api_chat_stream."

    preamble = body[:event_gen_pos]
    assert re.search(r"_billed\s*=\s*False", preamble), (
        "P2-AUDIT-NEW-2 violation: `_billed = False` no declarado antes de "
        "`def event_generator():`. Sin esto, `nonlocal _billed` dentro del "
        "generator falla con SyntaxError."
    )
    assert re.search(r"_chunk_observed\s*=\s*False", preamble), (
        "P2-AUDIT-NEW-2 violation: `_chunk_observed = False` no declarado "
        "antes de `def event_generator():`. Sin este flag, el finally no "
        "sabría si el LLM emitió tokens reales antes del abort/error."
    )


def test_event_generator_uses_nonlocal(chat_src: str) -> None:
    """B) `event_generator` declara `nonlocal _billed, _chunk_observed`
    para mutarlas desde el closure.
    """
    body = _extract_api_chat_stream_body(chat_src)
    event_gen_pos = body.find("def event_generator")
    assert event_gen_pos >= 0
    gen_body = body[event_gen_pos:]

    assert re.search(
        r"nonlocal\s+_billed\s*,\s*_chunk_observed|nonlocal\s+_chunk_observed\s*,\s*_billed",
        gen_body,
    ), (
        "P2-AUDIT-NEW-2 violation: `event_generator` no declara "
        "`nonlocal _billed, _chunk_observed` (o el order inverso). "
        "Sin nonlocal, las asignaciones dentro del generator crean "
        "variables locales nuevas — el finally outer nunca ve el flag."
    )


def test_chunk_observed_activated_on_chunk_type(chat_src: str) -> None:
    """C) Algún branch dentro del generator activa
    `_chunk_observed = True` cuando se observa `type == "chunk"`.
    """
    body = _extract_api_chat_stream_body(chat_src)
    event_gen_pos = body.find("def event_generator")
    gen_body = body[event_gen_pos:]

    # Buscar la asignación dentro de un contexto que mencione "chunk".
    # Regex: `_chunk_observed\s*=\s*True` debe aparecer al menos una vez
    # cerca de una comparación con `"chunk"`.
    chunk_observed_matches = list(re.finditer(
        r"_chunk_observed\s*=\s*True",
        gen_body,
    ))
    assert chunk_observed_matches, (
        "P2-AUDIT-NEW-2 violation: el generator no contiene "
        "`_chunk_observed = True` en ningún branch. Sin activación, el "
        "finally nunca cobrará y restauraríamos el bug original."
    )

    # Al menos uno debe estar a ≤300 chars de una comparación con "chunk"
    # (heurística: el branch que activa el flag debe estar contextualizado
    # por la detección del tipo).
    found_chunk_context = False
    for m in chunk_observed_matches:
        window = gen_body[max(0, m.start() - 300):m.start()]
        if re.search(r'_chunk_type\s*==\s*[\"\']chunk[\"\']|type[\"\']\s*\)\s*==\s*[\"\']chunk[\"\']', window):
            found_chunk_context = True
            break
    assert found_chunk_context, (
        "P2-AUDIT-NEW-2 violation: ninguna asignación "
        "`_chunk_observed = True` está bajo un branch que detecte "
        "`type == \"chunk\"`. Si se activa sobre `progress`/`sentiment`, "
        "facturaríamos quota completa por costo marginal (sentiment LLM "
        "es separado y barato — no justifica cobro)."
    )


def test_generator_exit_handler_present(chat_src: str) -> None:
    """D) `except GeneratorExit:` está presente y NO suprime la
    excepción (`raise` debe aparecer en su cuerpo).
    """
    body = _extract_api_chat_stream_body(chat_src)
    event_gen_pos = body.find("def event_generator")
    gen_body = body[event_gen_pos:]

    gen_exit_match = re.search(
        r"except\s+GeneratorExit\s*:",
        gen_body,
    )
    assert gen_exit_match, (
        "P2-AUDIT-NEW-2 violation: el generator no maneja "
        "`except GeneratorExit:`. Sin esto, los logs no distinguen "
        "client-abort vs exception genuina, y el finally NO recibe "
        "señal del path 'abort'."
    )

    # El cuerpo del except debe contener un `raise` (no suprimir).
    # Tomamos los siguientes ~500 chars del except como heurística.
    handler_window = gen_body[gen_exit_match.end(): gen_exit_match.end() + 500]
    # Buscar `raise` antes del siguiente `except` o `finally`.
    next_clause = re.search(r"\n\s+(except|finally|else)\s*", handler_window)
    handler_body = (
        handler_window[: next_clause.start()] if next_clause else handler_window
    )
    assert re.search(r"\braise\b", handler_body), (
        "P2-AUDIT-NEW-2 violation: `except GeneratorExit:` debe terminar "
        "con `raise` para propagar al StreamingResponse. Suprimir oculta "
        "el cierre limpio de la conexión y puede causar resource leaks."
    )


def test_finally_block_bills_idempotently(chat_src: str) -> None:
    """E) El `finally:` del generator factura `log_api_usage(user_id,
    "gemini_chat")` bajo guard `not _billed and _chunk_observed`.
    """
    body = _extract_api_chat_stream_body(chat_src)
    event_gen_pos = body.find("def event_generator")
    gen_body = body[event_gen_pos:]

    finally_match = re.search(r"\n\s+finally\s*:", gen_body)
    assert finally_match, (
        "P2-AUDIT-NEW-2 violation: el generator no tiene clausula "
        "`finally:`. Sin finally, el billing no se ejecuta en path "
        "de abort ni en exception."
    )

    # Cortar el cuerpo del finally al cierre del def event_generator o
    # cierre del scope outer.
    finally_window = gen_body[finally_match.end():]
    # Aproximación: tomar los siguientes 2000 chars (suficiente para
    # cualquier finally razonable).
    finally_body = finally_window[:2000]

    # Guards requeridos
    assert "not _billed" in finally_body, (
        "P2-AUDIT-NEW-2 violation: finally no chequea `not _billed`. "
        "Sin idempotencia, refactors futuros que añadan otra llamada a "
        "log_api_usage podrían double-billing."
    )
    assert "_chunk_observed" in finally_body, (
        "P2-AUDIT-NEW-2 violation: finally no chequea `_chunk_observed`. "
        "Sin este guard, facturaríamos en path donde el LLM NO emitió "
        "tokens (e.g., network drop pre-LLM)."
    )
    assert "log_api_usage(user_id" in finally_body and '"gemini_chat"' in finally_body, (
        "P2-AUDIT-NEW-2 violation: finally no invoca "
        "`log_api_usage(user_id, \"gemini_chat\")`. Sin esto, la quota "
        "nunca se decrementa."
    )
    # Guest exemption preservada (mismo patrón que el código original).
    assert (
        'user_id != "guest"' in finally_body
        and "user_id != session_id" in finally_body
    ), (
        "P2-AUDIT-NEW-2 violation: finally debe preservar guards anti-guest "
        "(`user_id != \"guest\"`) y anti-session-only "
        "(`user_id != session_id`). Sin esto facturaríamos a guests que "
        "no tienen cuenta paga."
    )


def test_bg_tasks_no_longer_bills(chat_src: str) -> None:
    """F) La función interna `bg_tasks()` ya NO contiene
    `log_api_usage(user_id, "gemini_chat")`. Si reaparece allí, hay
    double-billing (finally cobra + bg_tasks cobra).
    """
    body = _extract_api_chat_stream_body(chat_src)
    event_gen_pos = body.find("def event_generator")
    gen_body = body[event_gen_pos:]

    # Localizar la función bg_tasks
    bg_match = re.search(r"def\s+bg_tasks\s*\(\s*\)\s*:", gen_body)
    assert bg_match, "No se encontró `def bg_tasks():` dentro del generator."

    # Cortar el cuerpo: del def bg_tasks hasta el siguiente boundary anchor
    # (call que lanza bg_tasks como fire-and-forget). Acepta tanto el patrón
    # legacy `threading.Thread(target=bg_tasks` como el patrón canónico
    # P1-BG-THREAD-TIMEOUT `submit_bg_task(bg_tasks`.
    bg_body_start = bg_match.end()
    next_anchor = re.search(
        r"threading\.Thread\(target=bg_tasks|submit_bg_task\(\s*bg_tasks",
        gen_body[bg_body_start:],
    )
    assert next_anchor, (
        "No se encontró boundary anchor post `def bg_tasks():` — esperaba "
        "`threading.Thread(target=bg_tasks` (legacy) o `submit_bg_task(bg_tasks` "
        "(P1-BG-THREAD-TIMEOUT)."
    )
    bg_body = gen_body[bg_body_start: bg_body_start + next_anchor.start()]

    assert "gemini_chat" not in bg_body, (
        "P2-AUDIT-NEW-2 violation: `bg_tasks()` aún contiene billing "
        "`log_api_usage(..., \"gemini_chat\")`. Eso causa DOUBLE-BILLING "
        "(finally outer también cobra). Mover toda la lógica de billing "
        "al finally y dejar bg_tasks solo con summarization + facts."
    )
