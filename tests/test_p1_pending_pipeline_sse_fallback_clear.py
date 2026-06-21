"""[P1-PENDING-PIPELINE-SSE-FALLBACK-CLEAR · 2026-06-20] Cuando el pipeline
devuelve un plan de emergencia (`_is_fallback=True` — circuit breaker abierto,
LLM upstream caído, o rechazo crítico de restricción), el FALLBACK-GUARD del
SSE generator hace `break` SIN persistir el plan. El problema: ese path ya
seteó `_sse_completed_naturally=True` (justo antes), así que el `done-callback`
—que SÍ marca el KV `failed` para fallbacks (P2-PIPELINE-FALLBACK-GUARD-DONE)—
se SALTA (race-fix: sentinel=True → return).

Sin marcar el KV en el SSE generator, `pending_pipeline:<user>` queda
`status='generating'` FOREVER. El frontend, al cargar /plan, pollea
`/pending-status`, ve 'generating', y muestra "Diseñando tu plan" colgado
indefinidamente (el timer cuenta pero no hay nada generándose).

Incidente real 2026-06-20 (user 9b686868): la generación falló por circuit
breaker a las 23:11, el flag quedó 'generating', y ~29 min después el usuario
seguía viendo el loading screen (14:30 en el contador) — porque las 2 defensas
existentes no lo cubrían: el DELETE-al-startup (app.py) solo corre en restart
(no hubo en 16h) y el sweep cron NG-6B solo limpia entradas de horas.

Fix: el SSE FALLBACK-GUARD marca `pending_pipeline` failed antes de los breaks,
cubriendo ambos sub-casos (rechazo crítico + LLM caído). Este test ancla la
adición; un refactor que la borre falla aquí antes de tocar producción.
"""
from __future__ import annotations

import re
from pathlib import Path

_PLANS_PY = Path(__file__).resolve().parent.parent / "routers" / "plans.py"
_ANCHOR = "P1-PENDING-PIPELINE-SSE-FALLBACK-CLEAR"


def test_anchor_present():
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert _ANCHOR in src, (
        f"Falta el anchor {_ANCHOR} — el FALLBACK-GUARD/SSE debe marcar "
        "pending_pipeline failed para no dejar el frontend colgado en "
        "'Diseñando tu plan' tras un fallo de breaker."
    )


def test_marks_pending_pipeline_failed_in_block():
    """El bloque del anchor debe llamar upsert_pending_pipeline(status='failed')."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    idx = src.index(_ANCHOR)
    block = src[idx:idx + 1500]
    assert "upsert_pending_pipeline" in block, (
        "El bloque del anchor no llama upsert_pending_pipeline — el flag "
        "'generating' quedaría pegado."
    )
    assert 'status="failed"' in block, (
        "El bloque del anchor debe marcar status='failed' (no 'complete' ni "
        "dejarlo 'generating')."
    )


def test_clear_is_inside_is_fallback_guard():
    """El mark-failed debe vivir DENTRO del `if ..._is_fallback:` del SSE
    generator (no en un path no relacionado), para cubrir crítico + LLM caído."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    idx = src.index(_ANCHOR)
    pre = src[max(0, idx - 220):idx]
    assert "_is_fallback" in pre, (
        "El anchor debe estar inmediatamente dentro del guard "
        "`if isinstance(result, dict) and result.get('_is_fallback'):` del "
        "SSE generator."
    )


def test_mark_runs_before_break():
    """El upsert failed debe ejecutarse ANTES del primer `break` del guard —
    si no, el break sale del loop SSE sin limpiar el KV."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    idx = src.index(_ANCHOR)
    window = src[idx:idx + 3200]
    upsert_pos = window.find('status="failed"')
    # `\n\s+break\b` matchea el STATEMENT break, no la palabra "breaks" del comentario.
    _m = re.search(r"\n\s+break\b", window)
    break_pos = _m.start() if _m else -1
    assert upsert_pos != -1, "No se encontró el mark status='failed' tras el anchor."
    assert break_pos != -1, "No se encontró el break del FALLBACK-GUARD tras el anchor."
    assert upsert_pos < break_pos, (
        "El mark-failed debe correr ANTES del break — si el break sale primero, "
        "el KV nunca se limpia y el frontend queda colgado."
    )
