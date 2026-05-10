"""[P0-HIST-FIX-6 · 2026-05-09] Cross-link backend del fix UX al
cómputo de days range del tab Métricas — ajuste por shift_plan.

Bug reportado en producción 2026-05-09:
    Plan de 7 días con first_chunk que originalmente generó 3 días
    (Vie+Sáb+Dom). Tras shift_plan trimmar Vie expirado, el backend
    mutó el chunk: `days_count` 3 → 2, manteniendo `days_offset = 0`.
    El tab Métricas mostraba "Días 1-2" — usuario respondió "debe
    ser del 1-3 porque son 3 días iniciales".

    Para chunks posteriores (rolling_refill), `days_offset` también
    se decrementó cuando el array `plan_data.days` se re-indexó tras
    perder Vie. Sin ajuste, el refill mostraba "Días 3-6" (offset 2)
    en vez de "Días 4-7" (offset 3 original).

Fix (frontend-only):
    Cómputo de `_planExpiredDays` (= legacy_totalDays - active_total)
    a nivel del modal. Ajuste per-chunk:
      - Chunks con `chunk_kind in ('first_chunk', 'initial_plan')`:
        `_adjustedCount = c.days_count + _planExpiredDays` (los días
        expirados venían de este chunk).
      - Otros chunks: `_adjustedOffset = c.days_offset + _planExpiredDays`
        (su offset retrocedió por el re-index del array).

    Math que cuadra ahora:
      - first_chunk: offset 0, count 2+1=3 → "Días 1-3" ✓
      - rolling_refill: offset 2+1=3, count 4 → "Días 4-7" ✓
      - Total: 3 + 4 = 7 (display total) ✓

Cobertura backend (cross-link del marker — P2-HIST-AUDIT-14
requiere tests/test_p0_hist_fix_6*.py):
    1. Anchor del marker en History.jsx.
    2. Frontend declara `_planExpiredDays` con la misma fórmula que
       el missing-days block (consistencia).
    3. Frontend declara `_isFirstKind` cubriendo ambos aliases
       (first_chunk + initial_plan).
    4. Endpoint /chunk-metrics expone `chunk_kind`, `days_offset`,
       `days_count` (inputs del cómputo).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-6" in text


def test_history_jsx_computes_plan_expired_days():
    """`_planExpiredDays = max(0, legacyTotalDays - activeTotal)` —
    misma fórmula que el missing-days block. Sin esto, el ajuste
    no sabría cuántos días expiraron."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_planExpiredDays\s*=\s*Math\.max\(\s*0\s*,",
        text,
    ), (
        "Frontend debe declarar _planExpiredDays con Math.max(0, ...) "
        "para defender contra cómputos negativos."
    )


def test_history_jsx_recognizes_both_first_kind_aliases():
    """`first_chunk` y `initial_plan` son aliases del kind inicial.
    El cómputo debe cubrir ambos."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_isFirstKind\s*=\s*c\.chunk_kind\s*===\s*['\"]first_chunk['\"]\s*\|\|\s*c\.chunk_kind\s*===\s*['\"]initial_plan['\"]",
        text,
    ), (
        "El cómputo debe reconocer first_chunk Y initial_plan como "
        "ambos kinds del primer chunk del plan."
    )


def test_chunk_metrics_endpoint_exposes_inputs():
    """El frontend usa `chunk_kind`, `days_offset`, `days_count` —
    endpoint debe seguir exponiéndolos."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.chunk_kind" in src
    assert "q.days_offset" in src
    assert "q.days_count" in src
