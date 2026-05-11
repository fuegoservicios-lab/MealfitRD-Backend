"""[P3-NEW-3 · 2026-05-10] Lock-the-contract: docstring de
`_escalate_unrecoverable_chunk` documenta el modelo de idempotency
de los 4 pasos (UPDATE chunk, UPDATE plan_data, INSERT system_alerts,
push notification).

Bug temido (audit 2026-05-10):
  La función NO tenía guard `escalation_notified_at` explícito. Bajo
  race (worker normal vs zombie rescue), podía invocarse dos veces para
  el mismo chunk → push notification duplicada al usuario.

Verificación post-audit:
  La idempotency está cubierta defensivamente por upstream (CAS de
  P1-NEW-2 + COALESCE en el UPDATE + ON CONFLICT en system_alerts +
  filtro `dead_lettered_at IS NULL` en los call sites). Lo que faltaba
  era documentación: futuros devs viendo la función sin contexto no
  sabían qué partes son idempotentes y cuáles best-effort.

Fix:
  Bloque `[P3-NEW-3 · 2026-05-10] Modelo de idempotency` añadido al
  docstring documentando los 4 pasos + diseño consciente del trade-off
  (aceptar push potencialmente duplicado vs añadir flag dedicado).

Cobertura:
  1. Bloque `[P3-NEW-3` presente.
  2. Documenta los 4 pasos (UPDATE plan_chunk_queue, UPDATE
     meal_plans.plan_data, INSERT system_alerts, push).
  3. Nombra `COALESCE` (mecanismo idempotency step 1).
  4. Nombra `ON CONFLICT` (mecanismo idempotency step 3).
  5. Menciona el caso "best-effort" del push (transparencia sobre
     el trade-off aceptado).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read_escalate_docstring() -> str:
    """Extrae el docstring de `_escalate_unrecoverable_chunk`."""
    src = _CRON_PY.read_text(encoding="utf-8")
    fn_match = re.search(
        r"def\s+_escalate_unrecoverable_chunk\s*\(.+?\)\s*->\s*None\s*:\s*\n\s*\"\"\"",
        src,
        re.DOTALL,
    )
    assert fn_match is not None, (
        "No encuentro `def _escalate_unrecoverable_chunk` con signature esperada."
    )
    start = fn_match.end()
    end_match = re.search(r'\"\"\"', src[start:])
    assert end_match is not None
    return src[start:start + end_match.start()]


def test_p3_new_3_block_present():
    docstring = _read_escalate_docstring()
    assert "[P3-NEW-3" in docstring, (
        "Falta anchor `[P3-NEW-3` en el docstring de "
        "_escalate_unrecoverable_chunk."
    )


def test_documents_four_steps_idempotency():
    """El bloque P3-NEW-3 debe documentar los 4 pasos:
    UPDATE plan_chunk_queue, UPDATE meal_plans, INSERT system_alerts, push."""
    docstring = _read_escalate_docstring()
    block = docstring[docstring.find("[P3-NEW-3"):]
    for anchor in (
        "UPDATE plan_chunk_queue",
        "UPDATE meal_plans",
        "INSERT system_alerts",
        "Push notification",
    ):
        assert anchor in block, (
            f"El bloque P3-NEW-3 perdió la mención de `{anchor}` — los "
            f"4 pasos deben quedar enumerados explícitamente."
        )


def test_names_coalesce_mechanism_step_1():
    """Step 1 (UPDATE plan_chunk_queue) debe nombrar `COALESCE` como
    mecanismo de idempotency."""
    docstring = _read_escalate_docstring()
    block = docstring[docstring.find("[P3-NEW-3"):]
    assert "COALESCE" in block, (
        "El bloque P3-NEW-3 no menciona `COALESCE` — el dev no entiende "
        "cómo el step 1 es idempotente sin esa palabra clave."
    )


def test_names_on_conflict_mechanism_step_3():
    """Step 3 (INSERT system_alerts) debe nombrar `ON CONFLICT` como
    mecanismo de dedupe."""
    docstring = _read_escalate_docstring()
    block = docstring[docstring.find("[P3-NEW-3"):]
    assert "ON CONFLICT" in block, (
        "El bloque P3-NEW-3 no menciona `ON CONFLICT` — el dev no "
        "entiende cómo system_alerts deduplica."
    )


def test_documents_push_best_effort_trade_off():
    """El bloque debe documentar transparentemente el trade-off del push
    como best-effort (no idempotente) + las mitigaciones upstream."""
    docstring = _read_escalate_docstring()
    block = docstring[docstring.find("[P3-NEW-3"):]
    # Anchor textual de la diferenciación.
    assert "BEST-EFFORT" in block or "best-effort" in block, (
        "El bloque P3-NEW-3 no diferencia el push como `BEST-EFFORT`."
    )
    # Mitigación upstream debe mencionar el CAS de P1-NEW-2.
    assert "P1-NEW-2" in block, (
        "El bloque P3-NEW-3 perdió la referencia al CAS de P1-NEW-2 "
        "como mitigación del race vs zombie rescue."
    )
    # Diseño consciente debe nombrar `escalation_notified_at` como la
    # alternativa rechazada — para que un futuro dev no la reimplemente
    # sin leer el rationale.
    assert "escalation_notified_at" in block, (
        "El bloque P3-NEW-3 no nombra `escalation_notified_at` como "
        "alternativa rechazada — futuros devs podrían añadirla sin leer "
        "el rationale."
    )
