"""[P2-PDF-OBS-2 · 2026-05-14] Anchors parser-based del bundle P2-PDF-OBS.

Cierre del audit PDF 2026-05-14 (post P3-PDF-POLISH-4): 2 gaps P2
identificados al re-auditar el flujo "lista de compras PDF":

  P2-PDF-OBS-1   `trackEvent('pdf_prefetch_drift_corrected', ...)` tras
                 sincronizar localStorage en `handleDownloadShoppingList`
                 cuando el prefetch detecta `_plan_modified_at` divergente.
                 Sin este event, el `console.warn` previo se elimina por
                 `esbuild.pure=['console.warn']` en producción —
                 operadores quedaban ciegos al drift corregido.

  P2-PDF-OBS-2   `Promise.race` con timeout configurable
                 (`VITE_PDF_RENDER_TIMEOUT_MS`, default 60s, clamp
                 [15s, 180s]) alrededor de `html2pdf().set(opt).from(element).save()`.
                 Antes, si html2canvas colgaba indefinido (raro pero
                 reproducible en iOS Safari con `column-count: 4` en
                 hyper-dense ≥60 items), la promise nunca resolvía →
                 `pdfLock.current = false` en `finally` nunca corría →
                 usuario debía refrescar la página para volver a intentar.
                 Tras el timeout, el `Error` con `name=PdfRenderTimeout`
                 propaga al catch existente que ya emite
                 `pdf_download_failed`, distinguiendo timeouts de errores
                 de render reales para análisis post-mortem.

Cross-link convention (P2-HIST-AUDIT-14): el slug del marker
`P2-PDF-OBS-2` → `p2_pdf_obs_2` matchea este archivo
`test_p2_pdf_obs_2*.py`. Test único para el bundle (ambos P2 son del
mismo audit y comparten file de prod modificado: Dashboard.jsx).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_VITE_CONFIG = _REPO_ROOT / "frontend" / "vite.config.js"


@pytest.fixture(scope="module")
def dashboard_src() -> str:
    return _DASHBOARD_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def vite_config_src() -> str:
    return _VITE_CONFIG.read_text(encoding="utf-8")


def _extract_handler_body(src: str) -> str:
    """Devuelve el cuerpo de `handleDownloadShoppingList` hasta el siguiente
    `const handle...` top-level del componente. Cota suficiente para
    aserciones de presencia.

    Tooltip-anchor: P2-PDF-OBS-EXTRACTOR.
    """
    anchor = re.search(
        r"const\s+handleDownloadShoppingList\s*=\s*async\s*\(\s*\)\s*=>\s*\{",
        src,
    )
    assert anchor is not None, (
        "No se encontró `const handleDownloadShoppingList = async () => {` "
        "en Dashboard.jsx. ¿Renombrado? Si es intencional, actualizar este test."
    )
    start = anchor.end()
    rest = src[start:]
    # Buscamos el próximo `const handle<algo>` o `const <name> = useMemo` que
    # marca el inicio del siguiente bloque top-level.
    next_handler = re.search(
        r"\n    const\s+(?:handle\w+|[a-z]\w*\s*=\s*useMemo)",
        rest,
    )
    end = start + (next_handler.start() if next_handler else len(rest))
    return src[start:end]


# ---------------------------------------------------------------------------
# P2-PDF-OBS-1 · trackEvent en drift detection
# ---------------------------------------------------------------------------
def test_obs_1_drift_emits_track_event(dashboard_src: str):
    """Tras `effectivePlanData = fresh;` debe haber un `trackEvent(
    'pdf_prefetch_drift_corrected', ...)` para que el drift corregido sea
    observable en producción (donde `console.warn` se elimina por esbuild).
    """
    body = _extract_handler_body(dashboard_src)
    assert "pdf_prefetch_drift_corrected" in body, (
        "P2-PDF-OBS-1 regresión: `trackEvent('pdf_prefetch_drift_corrected', ...)` "
        "desapareció del handler PDF. Sin él, los operadores no pueden medir "
        "drift en producción (`console.warn` se elimina por esbuild)."
    )
    # El trackEvent debe estar DESPUÉS de `effectivePlanData = fresh;`
    # (orden semántico: primero sincronizar, luego instrumentar).
    fresh_assign = body.find("effectivePlanData = fresh")
    track_call = body.find("pdf_prefetch_drift_corrected")
    assert fresh_assign > -1 and track_call > -1, (
        "Anclas de orden semántico ausentes."
    )
    assert track_call > fresh_assign, (
        f"`trackEvent('pdf_prefetch_drift_corrected')` (idx={track_call}) "
        f"debe aparecer DESPUÉS de `effectivePlanData = fresh` "
        f"(idx={fresh_assign}). Orden invertido → telemetría emitida antes "
        f"del sync, semántica rota."
    )


def test_obs_1_drift_track_event_includes_user_and_plan(dashboard_src: str):
    """El payload del `trackEvent` debe incluir `user_id` y `plan_id` para
    correlación cross-canal (Sentry/PostHog/GA). Patrón consistente con los
    otros `trackEvent` del handler PDF."""
    body = _extract_handler_body(dashboard_src)
    # Buscar el bloque que contiene el trackEvent.
    drift_idx = body.find("pdf_prefetch_drift_corrected")
    assert drift_idx > -1
    # Cota ±400 chars alrededor del trackEvent para inspeccionar el payload.
    payload_window = body[max(0, drift_idx - 50):drift_idx + 400]
    assert "user_id" in payload_window, (
        "P2-PDF-OBS-1 regresión: payload del trackEvent drift no incluye `user_id`. "
        "Patrón consistente con `pdf_stale_inventory_fallback` y "
        "`pdf_download_success`."
    )
    assert "plan_id" in payload_window, (
        "P2-PDF-OBS-1 regresión: payload del trackEvent drift no incluye `plan_id`."
    )
    # Tanto local como latest timestamps deben aparecer (truncados a 32 chars).
    assert "local_modified_at" in payload_window
    assert "latest_modified_at" in payload_window


def test_obs_1_drift_track_event_is_best_effort(dashboard_src: str):
    """El `trackEvent` debe estar dentro de `try/catch` — analytics SDKs
    pueden fallar (Sentry blocked, PostHog SDK no cargado en edge bundles)
    y eso NO debe abortar la generación del PDF."""
    body = _extract_handler_body(dashboard_src)
    # Cota razonable alrededor de pdf_prefetch_drift_corrected.
    drift_idx = body.find("pdf_prefetch_drift_corrected")
    window = body[max(0, drift_idx - 200):drift_idx + 500]
    assert re.search(r"try\s*\{[^}]*trackEvent\([^)]*pdf_prefetch_drift_corrected", window), (
        "P2-PDF-OBS-1 regresión: `trackEvent('pdf_prefetch_drift_corrected', ...)` "
        "ya no está envuelto en `try/catch`. Un fallo del analytics SDK puede "
        "romper la generación del PDF."
    )


# ---------------------------------------------------------------------------
# P2-PDF-OBS-2 · timeout html2pdf con liberación de pdfLock
# ---------------------------------------------------------------------------
def test_obs_2_html2pdf_wrapped_in_promise_race(dashboard_src: str):
    """`html2pdf().set(opt).from(element).save()` debe estar dentro de un
    `Promise.race(...)` contra un timeout, NO un await pelado."""
    body = _extract_handler_body(dashboard_src)
    # Localizar la región del await sobre html2pdf.
    h2p_idx = body.find("html2pdf().set(opt).from(element).save()")
    assert h2p_idx > -1, (
        "P2-PDF-OBS-2 regresión: la llamada canónica `html2pdf().set(opt)"
        ".from(element).save()` no aparece en el handler. ¿Refactor que rompió "
        "el patrón Promise.race? Verificar manualmente."
    )
    window = body[max(0, h2p_idx - 600):h2p_idx + 400]
    assert "Promise.race" in window, (
        "P2-PDF-OBS-2 regresión: `html2pdf().save()` ya no está envuelto en "
        "`Promise.race`. Sin timeout, un hang del render deja `pdfLock` "
        "colgado permanente."
    )


def test_obs_2_timeout_knob_read_with_clamp(dashboard_src: str):
    """El timeout debe leerse desde `VITE_PDF_RENDER_TIMEOUT_MS` con clamps
    `[15s, 180s]` y default 60s. Mismo patrón que otros knobs del repo."""
    body = _extract_handler_body(dashboard_src)
    assert "VITE_PDF_RENDER_TIMEOUT_MS" in body, (
        "P2-PDF-OBS-2 regresión: knob `VITE_PDF_RENDER_TIMEOUT_MS` desapareció. "
        "El timeout debe ser configurable sin redeploy (escape hatch SRE)."
    )
    # Default 60s.
    assert re.search(r"_pdfRenderTimeoutMs\s*=\s*Number\.isFinite\([^)]+\)\s*\?\s*[^:]+:\s*60000", body), (
        "P2-PDF-OBS-2 regresión: default 60000ms perdido."
    )
    # Clamp inferior 15s y superior 180s.
    assert "15000" in body and "180000" in body, (
        "P2-PDF-OBS-2 regresión: clamps `[15000, 180000]` ms perdidos en el "
        "handler PDF."
    )


def test_obs_2_timeout_error_has_distinct_name(dashboard_src: str):
    """El error del timeout debe tener `name = 'PdfRenderTimeout'` para que
    el catch existente lo emita en `pdf_download_failed.error_name` y los
    operadores puedan filtrar timeouts vs errores reales del render."""
    body = _extract_handler_body(dashboard_src)
    assert "PdfRenderTimeout" in body, (
        "P2-PDF-OBS-2 regresión: error name `PdfRenderTimeout` perdido. "
        "Sin él, `pdf_download_failed.error_name='Error'` se mezcla con "
        "errores reales del html2canvas — operador no puede discriminar."
    )


def test_obs_2_settimeout_cleared_in_finally(dashboard_src: str):
    """El `setTimeout` que arma el reject del timeout debe limpiarse en
    `finally` para no leak handles (ej: caso `save()` resuelve antes del
    timeout). Sin clearTimeout el reject dispara fuera del `await` y se
    propaga como unhandled rejection."""
    body = _extract_handler_body(dashboard_src)
    # Cota ±200 alrededor de la primera aparición de `_pdfTimeoutHandle`.
    handle_idx = body.find("_pdfTimeoutHandle")
    assert handle_idx > -1, (
        "P2-PDF-OBS-2 regresión: handle `_pdfTimeoutHandle` perdido."
    )
    window = body[handle_idx:handle_idx + 1500]
    assert "clearTimeout" in window, (
        "P2-PDF-OBS-2 regresión: `clearTimeout(_pdfTimeoutHandle)` ausente. "
        "Sin él, el handle leak puede disparar reject post-resolve → "
        "unhandled promise rejection en el browser."
    )
    # El clearTimeout debe estar en `finally` (no en ramas condicionales).
    assert re.search(
        r"finally\s*\{[^}]*clearTimeout\(_pdfTimeoutHandle\)",
        window,
    ), (
        "P2-PDF-OBS-2 regresión: `clearTimeout(_pdfTimeoutHandle)` debe estar "
        "en bloque `finally` para que se ejecute siempre (incluso si Promise.race "
        "lanza)."
    )


def test_obs_2_finally_block_releases_pdf_lock(dashboard_src: str):
    """Anchor regression: el `finally` outer del handler sigue liberando
    `pdfLock.current = false`. Sin esto, el timeout aún dejaría el lock
    colgado al usuario."""
    body = _extract_handler_body(dashboard_src)
    # Buscamos `finally { ... pdfLock.current = false ... }` en el handler.
    assert re.search(
        r"finally\s*\{[^}]*pdfLock\.current\s*=\s*false",
        body,
        re.DOTALL,
    ), (
        "P2-PDF-OBS-2 regresión: el `finally` outer dejó de liberar "
        "`pdfLock.current = false`. Sin esta liberación, un timeout (o "
        "cualquier excepción) deja el lock colgado y bloquea re-intentos "
        "del usuario."
    )


# ---------------------------------------------------------------------------
# Sanity: vite.config sigue marcando console.warn como pure (premisa del fix)
# ---------------------------------------------------------------------------
def test_premise_console_warn_is_pruned_in_production(vite_config_src: str):
    """Premisa del bundle: `vite.config.js` declara `console.warn` como pure
    en `mode='production'`. Si esta línea se elimina, P2-PDF-OBS-1 deja de
    ser estrictamente necesario (el `console.warn` sobreviviría) y este test
    debe actualizarse junto con la decisión.

    Si el operador retira `console.warn` del array `pure`, debe actualizar
    también CLAUDE.md "E2E tests" / "Convenciones" y este test.
    """
    assert "pure:" in vite_config_src, (
        "vite.config.js ya no declara `pure: [...]`. Si es intencional, "
        "actualizar este test y la racional del P2-PDF-OBS-1."
    )
    assert "console.warn" in vite_config_src, (
        "vite.config.js ya no incluye `console.warn` en el array de funciones "
        "pure. Si es intencional, el `console.warn` del drift detection ahora "
        "sobreviviría en producción — pero la telemetría del `trackEvent` "
        "sigue siendo el canal autoritativo. Actualizar este test."
    )
