"""[P0-HIST-IN-PROGRESS · 2026-05-09] Backend sentinel del cross-link
marker `_LAST_KNOWN_PFIX` ↔ tests vitest del frontend.

Contexto:
    El audit Historial 2026-05-09 identificó 3 P0 visibles en el
    frontend:

      1. P0-HIST-IN-PROGRESS — `getStatusInfo` no tenía bucket
         `in_progress` para `generation_status='generating' |
         'generating_next' | 'rolling'`. Planes sanos generándose en
         background mostraban "Parcial X/Y" idéntico a planes
         abandonados.

      2. P0-HIST-METRICS-FAILED — Tab "Métricas" se ocultaba si
         `chunk_completed_count = 0` aunque hubiera chunks `failed`
         con `dead_letter_reason`. Post-mortem invisible para planes
         que se cayeron sin ningún completed.

      3. P0-HIST-CACHE-INVALIDATION — Singletons de cache
         (P2-HIST-AUDIT-11) con TTL 30 min sin invalidación en
         mutaciones (rename/delete/restore) ni en re-foreground del
         tab. Listado y modal del Historial mostraban estado pre-
         mutación.

Los 3 fixes viven 100% en el frontend (`History.jsx`, `History.module.css`,
`utils/historyCaches.js`) con cobertura vitest:

  - `frontend/src/__tests__/History.p0_in_progress_bucket.test.js`
  - `frontend/src/__tests__/History.p0_metrics_failed_visible.test.js`
  - `frontend/src/__tests__/History.p0_cache_invalidation.test.js`

Este test backend cumple DOS roles:

  1. Cross-link con `test_p2_hist_audit_14_marker_test_link.py` —
     `_LAST_KNOWN_PFIX = "P0-HIST-IN-PROGRESS · 2026-05-09"` requiere
     un archivo `tests/test_p0_hist_in_progress*.py`. Sin esto, el
     cross-link falla y el bump del marker queda bloqueado.

  2. Drift detection cross-archivo: si un futuro refactor remueve
     CUALQUIERA de los 3 markers `[P0-HIST-*-*]` del source frontend,
     este test falla loud con un mensaje accionable. Cubre el gap
     "el marker dice que el fix está pero el código ya no lo
     contiene" sin importar el módulo en runtime.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_HISTORY_JSX = _FRONTEND_SRC / "pages" / "History.jsx"
_HISTORY_CSS = _FRONTEND_SRC / "pages" / "History.module.css"
_HISTORY_CACHES = _FRONTEND_SRC / "utils" / "historyCaches.js"
_VITEST_DIR = _FRONTEND_SRC / "__tests__"


@pytest.fixture(scope="module")
def history_jsx() -> str:
    return _HISTORY_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def history_css() -> str:
    return _HISTORY_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def history_caches_js() -> str:
    return _HISTORY_CACHES.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Markers de los 3 P0 presentes en los archivos correctos
# ---------------------------------------------------------------------------
def test_p0_in_progress_marker_in_history_jsx(history_jsx: str):
    """P0-1: bucket in_progress se documenta con el marker
    `[P0-HIST-IN-PROGRESS · 2026-05-09]` en `History.jsx`."""
    assert "[P0-HIST-IN-PROGRESS · 2026-05-09]" in history_jsx, (
        "Falta marker [P0-HIST-IN-PROGRESS · 2026-05-09] en "
        "frontend/src/pages/History.jsx. Si removiste el fix, "
        "actualiza también _LAST_KNOWN_PFIX en app.py."
    )


def test_p0_in_progress_marker_in_history_css(history_css: str):
    """P0-1: la clase CSS `.statusInProgress` debe existir y el
    marker debe estar en el bloque CSS para drift detection."""
    assert "[P0-HIST-IN-PROGRESS · 2026-05-09]" in history_css
    assert ".statusInProgress" in history_css


def test_p0_metrics_failed_marker_in_history_jsx(history_jsx: str):
    """P0-2: tab Métricas visible si hay failed/exhausted, marker
    `[P0-HIST-METRICS-FAILED · 2026-05-09]` en `History.jsx`."""
    assert "[P0-HIST-METRICS-FAILED · 2026-05-09]" in history_jsx
    # Drift detection del shape del fix: la unión OR entre
    # _metricsTabCount y _exhaustedCount.
    assert "_hasMetrics = _metricsTabCount > 0 || _exhaustedCount > 0" in history_jsx


def test_p0_cache_invalidation_marker_in_caches_module(history_caches_js: str):
    """P0-3: helper invalidateCachesForPlan exportado del módulo de
    caches singleton."""
    assert "[P0-HIST-CACHE-INVALIDATION · 2026-05-09]" in history_caches_js
    assert "export const invalidateCachesForPlan" in history_caches_js


def test_p0_cache_invalidation_marker_in_history_jsx(history_jsx: str):
    """P0-3: History.jsx importa y usa invalidateCachesForPlan."""
    assert "[P0-HIST-CACHE-INVALIDATION · 2026-05-09]" in history_jsx
    # Import explícito del helper.
    import_line_present = any(
        "invalidateCachesForPlan" in line and "from '../utils/historyCaches'" in line
        for line in history_jsx.splitlines()
        if line.lstrip().startswith("import")
    )
    assert import_line_present, (
        "History.jsx debe importar `invalidateCachesForPlan` desde "
        "'../utils/historyCaches'."
    )


def test_p0_visibility_refresh_marker_in_history_jsx(history_jsx: str):
    """P0-3 (auxiliar): marker [P0-HIST-VIS-REFRESH] del listener de
    visibilitychange. Aparece 3 veces (declaración del ref, useEffect,
    bump en fetchHistory) — cualquier <3 indica drift parcial."""
    count = history_jsx.count("[P0-HIST-VIS-REFRESH · 2026-05-09]")
    assert count >= 3, (
        f"Marker [P0-HIST-VIS-REFRESH] aparece {count} veces en "
        "History.jsx; esperaba >=3 (ref + useEffect + bump). "
        "Drift parcial — algún punto del fix fue removido."
    )


# ---------------------------------------------------------------------------
# 2. Tests vitest correspondientes existen en el frontend
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vitest_filename", [
    "History.p0_in_progress_bucket.test.js",
    "History.p0_metrics_failed_visible.test.js",
    "History.p0_cache_invalidation.test.js",
])
def test_vitest_test_file_exists(vitest_filename: str):
    """Cada uno de los 3 P0 tiene su archivo vitest de regresión.
    Si alguien borra uno por accidente, este test falla y avisa
    qué cobertura se perdió."""
    path = _VITEST_DIR / vitest_filename
    assert path.exists(), (
        f"Falta archivo de tests vitest: {path}. "
        "Cubre los 3 P0 del audit Historial 2026-05-09."
    )
