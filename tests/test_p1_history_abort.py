"""[P1-HISTORY-ABORT · 2026-05-23] Cross-link test entre el marker
`_LAST_KNOWN_PFIX` y el test parser-based del frontend.

Bug original (audit production-readiness 2026-05-23 → P1-2):
    `frontend/src/pages/History.jsx::fetchHistory()` y los dos
    siblings del mount useEffect (`_fetchLessonsCounts()` y el
    inline `getHistoryStatusSummary()`) corren con Promise.race +
    timeout 12s. Si el usuario navega fuera de /history mientras
    las 3 fetches están en flight, las .then resuelven post-unmount:
      1. React warning "Can't perform a state update on an
         unmounted component" (visible en console + ruido Sentry).
      2. Memory leak suave: body parseado + closures retenidos
         hasta GC del resolved promise.
      3. Bandwidth desperdiciado: el server completa la response
         que nadie lee.

    El audit cuantificó el impacto como P1 (degradación UX silenciosa
    + Sentry noise visible en usuarios premium tier Plus/Ultra que
    navegan rápido entre /history y /dashboard).

Fix:
    AbortController component-scoped (`_abortControllerRef`):
      - Creado en el mount useEffect.
      - Signal pasado a las 3 fetches (mount + re-fetches del
        listener visibilitychange via el mismo ref).
      - `.abort()` invocado en el cleanup return del useEffect →
        cancela TODOS los in-flight sincronamente.
      - Catch silencioso para AbortError + guards `signal.aborted`
        antes de cada setter previenen state-on-unmounted aunque
        la fetch haya recibido body parcial pre-abort.

    El cambio toca:
      - `frontend/src/config/api.js`: 3 helpers ahora aceptan
        `options = {}` y forwardean a `fetchWithAuth` (backward-compat:
        `getX()` raw sigue funcionando — solo añade pass-through de
        signal).
      - `frontend/src/pages/History.jsx`: mount useEffect crea el
        controller + cleanup abort + visibilitychange handler reusa
        el signal del ref + 3 call sites pasan `{ signal }` + catch
        silencia AbortError.

Cobertura del test (estructural del backend):
    Este test backend es un STUB del cross-link enforcer
    (`test_p2_hist_audit_14_marker_test_link.py`) que valida que el
    slug del marker actual matchee `tests/test_<slug>*.py`. El fix
    real es frontend-only — la cobertura E2E vive en
    `frontend/src/__tests__/History.p1_history_abort_on_unmount.test.js`
    (12 assertions parser-based: anchor + ref declaration + mount
    controller + cleanup abort + signature changes + visibilitychange
    reuse + config/api.js forwarding).

    Aquí solo validamos:
      1. El marker textual del fix existe en `app.py`.
      2. El archivo de test parser-based del frontend existe + tiene
         el marker (proof of regression coverage en CI frontend).

Tooltip-anchor: P1-HISTORY-ABORT | regression guard 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_FRONTEND_TEST = (
    _REPO_ROOT
    / "frontend"
    / "src"
    / "__tests__"
    / "History.p1_history_abort_on_unmount.test.js"
)


def test_last_known_pfix_bumped_to_p1_history_abort():
    """`_LAST_KNOWN_PFIX` en app.py refleja el P-fix más reciente.
    Si alguien revierte el bump sin revertir el código del fix,
    `/health/version` reportará un marker stale → operador no
    puede confirmar que el AbortController está vivo en el binary
    productivo."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m is not None, "_LAST_KNOWN_PFIX no encontrado en app.py"
    marker = m.group(1)
    assert marker.startswith("P1-HISTORY-ABORT"), (
        f"Marker esperado `P1-HISTORY-ABORT · 2026-05-23`, encontrado `{marker}`. "
        "Si bumpeaste a un P-fix posterior, este test sigue siendo válido "
        "como anchor histórico — moverlo a un nombre con sufijo de fecha "
        "si necesitas que falle solo durante la ventana activa."
    )


def test_frontend_regression_test_exists():
    """El test parser-based del frontend que cubre el fix DEBE
    existir. Sin esto el marker es cosmético — un revert de
    History.jsx no fallaría ningún test."""
    assert _FRONTEND_TEST.exists(), (
        f"Test frontend de regresión NO encontrado en `{_FRONTEND_TEST}`. "
        "El fix P1-HISTORY-ABORT toca solo frontend (History.jsx + "
        "config/api.js); la cobertura E2E debe vivir como vitest "
        "parser-based en `frontend/src/__tests__/`."
    )


def test_frontend_regression_test_has_marker():
    """El test del frontend debe contener el marker textual para
    grepability cross-codebase (saltar de marker → test → fix)."""
    text = _FRONTEND_TEST.read_text(encoding="utf-8")
    assert re.search(r"\[P1-HISTORY-ABORT\s*·\s*2026-05-23\]", text), (
        "El test frontend existe pero no contiene el marker "
        "`[P1-HISTORY-ABORT · 2026-05-23]`. Sin el marker el grep "
        "cross-codebase falla — añadir el anchor al docstring."
    )


def test_history_jsx_has_marker_anchor():
    """`History.jsx` debe tener el marker — ancla del fix en la
    fuente productiva. Sin esto un dev puede borrar el
    AbortController creyendo que es cleanup defensivo no esencial."""
    history_jsx = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
    text = history_jsx.read_text(encoding="utf-8")
    assert re.search(r"\[P1-HISTORY-ABORT\s*·\s*2026-05-23\]", text), (
        f"`{history_jsx}` no contiene el marker `[P1-HISTORY-ABORT · 2026-05-23]`. "
        "Añadirlo en los 3 sitios clave: ref declaration, mount useEffect, "
        "cleanup return — sirve de tooltip-anchor."
    )


def test_api_js_helpers_have_marker_anchor():
    """`config/api.js` debe tener el marker — los 3 helpers
    (getHistoryList, getLessonsCounts, getHistoryStatusSummary)
    cambiaron firma para soportar options.signal."""
    api_js = _REPO_ROOT / "frontend" / "src" / "config" / "api.js"
    text = api_js.read_text(encoding="utf-8")
    assert re.search(r"\[P1-HISTORY-ABORT\s*·\s*2026-05-23\]", text), (
        f"`{api_js}` no contiene el marker. El forwarding de options "
        "a fetchWithAuth es el contract layer del fix — sin anchor "
        "alguien puede revertir la firma."
    )
