"""[P1-FRONTEND-FINAL-1 · 2026-05-24] Stub Python para cross-link
P2-HIST-AUDIT-14 + anchor del bundle frontend.

Bundle 3-en-1 que cierra los 3 P1 frontend residuales del audit
prod-readiness 2026-05-24 (post P1-PROD-FINAL-4 backend):

  GAP-1 P1-ERROR-BOUNDARY-SENTRY-CAPTURE:
    `GlobalErrorBoundary.componentDidCatch` solo logueaba en DEV; @sentry/react
    NO auto-captura errores swalloweados por error boundaries. Crashes de
    render genuinos eran invisibles para SRE post-incidente + el copy
    "Actualizando App..." engañaba al user. Fix: `captureException` named
    import (preserva tree-shake P2-SENTRY-TREESHAKE) + copy diferenciado.

  GAP-2 P1-PLAN-CHUNK-POLL-ABORT:
    `Plan.jsx` polling cada 5s sin AbortController → setters post-unmount.
    Plan.jsx era el último polling sin abort tras P1-PROD-FINAL-1 (Dashboard)
    y P1-HISTORY-ABORT (History). Fix: AbortController scoped + signal a
    getPlanChunkStatus + guards aborted + AbortError silenciado.

  GAP-3 P1-AGENT-LAZY-INIT-PRIVATE-MODE:
    `AgentPage.jsx:469` lazy initializer usaba `localStorage.getItem` raw
    sin try/catch → SecurityError en iOS Private Mode rompía mount entero.
    Mismo modo de fallo que P1-PROD-FINAL-1 cerró en Settings/Dashboard.
    Fix: migrar a `safeLocalStorageGet`.

Verificaciones detalladas en los 3 archivos vitest:
  - `frontend/src/__tests__/GlobalErrorBoundary.p1_sentry_capture.test.js`
  - `frontend/src/__tests__/Plan.p1_chunk_poll_abort.test.js`
  - `frontend/src/__tests__/AgentPage.p1_lazy_init_private_mode.test.js`

Este stub cubre:
  - Marker `_LAST_KNOWN_PFIX` bumpeado a P1-FRONTEND-FINAL-1 (date-floor).
  - 3 archivos vitest del bundle existen.
  - Anchors P1-ERROR-BOUNDARY-SENTRY-CAPTURE, P1-PLAN-CHUNK-POLL-ABORT,
    P1-AGENT-LAZY-INIT-PRIVATE-MODE presentes en sus files target.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_PY = _REPO_ROOT / "backend" / "app.py"
_FRONTEND = _REPO_ROOT / "frontend" / "src"
_GLOBAL_BOUNDARY = _FRONTEND / "components" / "GlobalErrorBoundary.jsx"
_PLAN_JSX = _FRONTEND / "pages" / "Plan.jsx"
_AGENT_JSX = _FRONTEND / "pages" / "AgentPage.jsx"
_VITEST_DIR = _FRONTEND / "__tests__"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — Anchors en los 3 files target
# ---------------------------------------------------------------------------
def test_anchor_sentry_capture_in_boundary():
    src = _read(_GLOBAL_BOUNDARY)
    assert "P1-ERROR-BOUNDARY-SENTRY-CAPTURE" in src, (
        "Falta anchor `P1-ERROR-BOUNDARY-SENTRY-CAPTURE` en "
        "frontend/src/components/GlobalErrorBoundary.jsx."
    )


def test_anchor_plan_poll_abort_in_plan_jsx():
    src = _read(_PLAN_JSX)
    assert "P1-PLAN-CHUNK-POLL-ABORT" in src, (
        "Falta anchor `P1-PLAN-CHUNK-POLL-ABORT` en frontend/src/pages/Plan.jsx."
    )


def test_anchor_agent_lazy_init_in_agentpage_jsx():
    src = _read(_AGENT_JSX)
    assert "P1-AGENT-LAZY-INIT-PRIVATE-MODE" in src, (
        "Falta anchor `P1-AGENT-LAZY-INIT-PRIVATE-MODE` en "
        "frontend/src/pages/AgentPage.jsx."
    )


# ---------------------------------------------------------------------------
# Sección 2 — Vitest test files del bundle existen
# ---------------------------------------------------------------------------
def test_vitest_files_exist():
    expected = [
        "GlobalErrorBoundary.p1_sentry_capture.test.js",
        "Plan.p1_chunk_poll_abort.test.js",
        "AgentPage.p1_lazy_init_private_mode.test.js",
    ]
    missing = [f for f in expected if not (_VITEST_DIR / f).exists()]
    assert not missing, (
        f"Faltan los siguientes archivos vitest del bundle: {missing}"
    )


# ---------------------------------------------------------------------------
# Sección 3 — Marker bumped + date-floor
# ---------------------------------------------------------------------------
def test_marker_bumped_to_p1_frontend_final_1():
    """[Relajado por P2-PROD-FINAL-3 · 2026-05-24] Sibling date-floor:
    el marker debe tener fecha >= 2026-05-24 (día del cierre del Bundle
    frontend). Exact-match removido tras supersede del Bundle #3 (P2s
    del mismo día) — patrón emergente desde P1-PROD-FINAL-1."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor 2026-05-24 "
        f"(día del cierre P1-FRONTEND-FINAL-1)."
    )


def test_marker_date_meets_p1_frontend_final_1_floor():
    """Date-floor sibling para futuros supersedes."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor}."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Cross-link guard P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    """El slug `p1_frontend_final_1` matchea el marker `P1-FRONTEND-FINAL-1`.
    El cross-link enforcer `test_p2_hist_audit_14_marker_test_link.py`
    verifica que el slug del marker DEBE matchear al menos un archivo
    `tests/test_<slug>*.py`."""
    src = _read(Path(__file__))
    assert "P1-FRONTEND-FINAL-1" in src
