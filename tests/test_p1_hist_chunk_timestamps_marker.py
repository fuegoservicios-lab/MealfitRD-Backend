"""[P1-HIST-CHUNK-TIMESTAMPS · 2026-05-09] Backend sentinel del
cross-link marker `_LAST_KNOWN_PFIX` ↔ tests vitest del frontend
para el render de `escalated_at` y `learning_persisted_at` en el
tab Métricas del modal del Historial.

Contexto:
    El audit Historial 2026-05-09 (gap P1-4) identificó que el
    endpoint `/chunk-metrics` (P2-HIST-AUDIT-10) ya devolvía dos
    timestamps clave para post-mortem:
      - `escalated_at`: cuándo el cron escalator marcó el chunk
        como no-recoverable (`_escalate_unrecoverable_chunk`).
      - `learning_persisted_at`: cuándo el T2 de `cron_tasks.py`
        commiteó `plan_data._last_chunk_learning` (señal de que
        chunks N+1 tienen el aprendizaje del N).
    Pero el frontend los descartaba en el render. Para diagnosticar
    "¿cuándo escaló este chunk?" o "¿se commiteó learning antes
    del fail?" había que ir directo a SQL.

Fix:
    Helper `_fmtRelTime(iso)` formatea ISO 8601 a tiempo relativo
    legible ("hace 2h 15m") con ISO absoluto en `title=` tooltip.
    Render por chunk añade dos chips:
      - "Escalado: hace Xh" (warn) si `escalated_at` non-null.
      - "Learning: hace Xh" (neutral) si `learning_persisted_at`
        non-null.
      - "Sin learning" (warn) edge-case: chunk completed pero
        learning_persisted_at null → T2 commit fail.

Cobertura:
    Este test backend cumple DOS roles:

    1. Cross-link: `_LAST_KNOWN_PFIX = "P1-HIST-CHUNK-TIMESTAMPS ·
       2026-05-09"` requiere un archivo `tests/test_p1_hist_chunk_timestamps*.py`.

    2. Drift detection del shape backend → frontend: el endpoint
       `chunk-metrics` debe seguir devolviendo `escalated_at` y
       `learning_persisted_at` como ISO strings. Si un futuro
       refactor los renombra, el render del frontend queda mudo
       sin warning. Aserción del SELECT del endpoint cubre eso.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_VITEST_DIR = _REPO_ROOT / "frontend" / "src" / "__tests__"


# ---------------------------------------------------------------------------
# 1. Marker presente en frontend
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P1-HIST-CHUNK-TIMESTAMPS · 2026-05-09]" in text


def test_vitest_file_exists():
    """El test vitest dedicado al P1-HIST-CHUNK-TIMESTAMPS debe
    existir. Cubre el helper _fmtRelTime + render de los dos chips
    + edge case T2 fail."""
    path = _VITEST_DIR / "History.p1_chunk_timestamps.test.js"
    assert path.exists(), (
        f"Falta {path}. Cubre el render de escalated_at + "
        "learning_persisted_at en el tab Métricas."
    )


# ---------------------------------------------------------------------------
# 2. Backend sigue devolviendo los timestamps
# ---------------------------------------------------------------------------
def test_chunk_metrics_endpoint_returns_escalated_at():
    """El SELECT del endpoint `chunk-metrics` debe seguir extrayendo
    `escalated_at` de `plan_chunk_queue`. Si un refactor lo elimina,
    el chip "Escalado" queda mudo sin error visible."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # SELECT incluye `q.escalated_at`.
    assert "q.escalated_at" in src
    # Y se propaga al response dict.
    assert '"escalated_at"' in src


def test_chunk_metrics_endpoint_returns_learning_persisted_at():
    """Mismo contrato para `learning_persisted_at`. Marca el commit
    final de T2 — su ausencia en chunks completed indica T2 fail."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.learning_persisted_at" in src
    assert '"learning_persisted_at"' in src


def test_chunk_metrics_serializes_iso_format():
    """Los dos timestamps deben serializarse vía `_iso(...)` (helper
    interno que llama `.isoformat()`). Sin esto, llegarían como
    `datetime` raw — `Date(...)` del frontend no parsea ese shape
    y `_fmtRelTime` retornaría null silenciosamente."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # Helper `_iso` o `.isoformat()` aplicado a los dos campos.
    assert "_iso(r.get(\"escalated_at\"))" in src or '_iso(r.get("escalated_at"))' in src
    assert (
        '_iso(r.get("learning_persisted_at"))' in src
        or "_iso(r.get(\"learning_persisted_at\"))" in src
    )


# ---------------------------------------------------------------------------
# 3. plan_chunk_queue tiene las columnas (drift detection del schema)
# ---------------------------------------------------------------------------
def test_columns_documented_in_p2_hist_audit_10_docstring():
    """El docstring del endpoint declara los dos timestamps en el
    response shape. Cualquier remoción del docstring sin update
    rompe la documentación operacional."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # Ambos en el docstring del Returns.
    assert "escalated_at" in src
    assert "learning_persisted_at" in src
