"""[P1-AUDIT-2 · 2026-05-15] Anchors parser-based para el segundo callsite
de `html2pdf().save()` en el frontend: `Recipes.jsx::handleDownloadPDF`.

Contexto:
    P2-PDF-OBS-2 (2026-05-14) aplicó timeout `Promise.race` + knob
    `VITE_PDF_RENDER_TIMEOUT_MS` + error `name='PdfRenderTimeout'` al
    handler PDF de `Dashboard.jsx::handleDownloadShoppingList`. El audit
    cerró el flujo "lista de compras PDF" 100% — pero olvidó el SEGUNDO
    callsite de `html2pdf().save()` en el frontend: el handler de descarga
    de recetas individuales en `Recipes.jsx`.

    Síntoma del bug original (mismo modo de fallo que cerró P2-PDF-OBS-2
    en Dashboard, ahora en Recipes): html2canvas cuelga indefinido en iOS
    Safari cuando la receta es hyper-densa (≥20 pasos + ingredients
    largos), o si la pestaña pierde foco durante un render largo. Sin
    timeout, la promise nunca resuelve → `toast.dismiss(toastId)` nunca
    corre → el toast loading queda visible permanente y el usuario debe
    refrescar la página para retry.

Fix:
    Mismo patrón canónico de P2-PDF-OBS-2 replicado en
    `Recipes.jsx::handleDownloadPDF`:
      - `Promise.race([html2pdf().set(opt).from(htmlString, 'string').save(),
        _pdfTimeoutPromise])`.
      - Knob `VITE_PDF_RENDER_TIMEOUT_MS` (default 60s, clamp [15s, 180s])
        — mismo knob que Dashboard, ningún SRE necesita aprender uno nuevo.
      - Error con `name='PdfRenderTimeout'` para discriminar timeouts
        vs errores de render reales.
      - `clearTimeout(_pdfTimeoutHandle)` en bloque `finally` para evitar
        leak de handles (reject post-resolve → unhandled rejection).

Cross-link convention (P2-HIST-AUDIT-14): el slug del marker
`P1-AUDIT-2` → `p1_audit_2` matchea este archivo
`test_p1_audit_2_recipes_pdf_timeout.py`.

Tooltip-anchor: P1-AUDIT-2-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"


@pytest.fixture(scope="module")
def recipes_src() -> str:
    return _RECIPES_JSX.read_text(encoding="utf-8")


def _extract_handler_body(src: str) -> str:
    """Devuelve el cuerpo de `handleDownloadPDF` desde `const
    handleDownloadPDF = async (meal) => {` hasta el siguiente `const
    <name>` top-level del componente."""
    anchor = re.search(
        r"const\s+handleDownloadPDF\s*=\s*async\s*\(\s*meal\s*\)\s*=>\s*\{",
        src,
    )
    assert anchor is not None, (
        "P1-AUDIT-2 regresión: `const handleDownloadPDF = async (meal) => {` "
        "ya no aparece en Recipes.jsx. ¿Renombrado? Actualizar este test."
    )
    start = anchor.end()
    rest = src[start:]
    # Buscar la siguiente declaración `const <name> = ...` top-level (4
    # espacios de indent) o `return (` que marca el render del componente.
    next_block = re.search(
        r"\n    (?:const\s+\w+\s*=|return\s*\()",
        rest,
    )
    end = start + (next_block.start() if next_block else len(rest))
    return src[start:end]


# ---------------------------------------------------------------------------
# 1. html2pdf().save() envuelto en Promise.race
# ---------------------------------------------------------------------------
def test_html2pdf_wrapped_in_promise_race(recipes_src: str):
    """`html2pdf().set(opt).from(htmlString, 'string').save()` debe estar
    dentro de un `Promise.race(...)` contra un timeout, NO un await pelado.
    """
    body = _extract_handler_body(recipes_src)
    # En Recipes la llamada usa `.from(htmlString, 'string')` (vs
    # `.from(element)` en Dashboard) — aceptamos ambas formas con regex.
    h2p_re = re.compile(
        r"html2pdf\(\)\s*\.set\(opt\)\s*\.from\(\s*\w+",
    )
    h2p_m = h2p_re.search(body)
    assert h2p_m is not None, (
        "P1-AUDIT-2 regresión: la llamada canónica "
        "`html2pdf().set(opt).from(htmlString, ...)` no aparece en el handler. "
        "¿Refactor que rompió el patrón Promise.race? Verificar manualmente."
    )
    window = body[max(0, h2p_m.start() - 600):h2p_m.start() + 400]
    assert "Promise.race" in window, (
        "P1-AUDIT-2 regresión: `html2pdf().save()` ya no está envuelto en "
        "`Promise.race`. Sin timeout, un hang del render deja el toast "
        "loading colgado permanente y el usuario no puede retry sin refresh."
    )


# ---------------------------------------------------------------------------
# 2. Knob VITE_PDF_RENDER_TIMEOUT_MS con clamp
# ---------------------------------------------------------------------------
def test_timeout_knob_read_with_clamp(recipes_src: str):
    """El timeout debe leerse desde `VITE_PDF_RENDER_TIMEOUT_MS` con clamps
    `[15s, 180s]` y default 60s. MISMO knob que Dashboard.jsx (P2-PDF-OBS-2)
    para que SRE no aprenda 2 knobs distintos."""
    body = _extract_handler_body(recipes_src)
    assert "VITE_PDF_RENDER_TIMEOUT_MS" in body, (
        "P1-AUDIT-2 regresión: knob `VITE_PDF_RENDER_TIMEOUT_MS` ausente en "
        "Recipes.jsx::handleDownloadPDF. El timeout debe ser configurable "
        "sin redeploy (escape hatch SRE)."
    )
    assert re.search(
        r"_pdfRenderTimeoutMs\s*=\s*Number\.isFinite\([^)]+\)\s*\?\s*[^:]+:\s*60000",
        body,
    ), (
        "P1-AUDIT-2 regresión: default 60000ms perdido. Mismo default que "
        "Dashboard.jsx P2-PDF-OBS-2."
    )
    assert "15000" in body and "180000" in body, (
        "P1-AUDIT-2 regresión: clamps `[15000, 180000]` ms perdidos. Sin "
        "clamps, SRE puede setear valores absurdos sin protección."
    )


# ---------------------------------------------------------------------------
# 3. Error name = 'PdfRenderTimeout'
# ---------------------------------------------------------------------------
def test_timeout_error_has_distinct_name(recipes_src: str):
    """El error del timeout debe tener `name = 'PdfRenderTimeout'` (mismo
    name que Dashboard) para que analytics/Sentry filtren timeouts vs
    errores reales del render de manera uniforme cross-handlers."""
    body = _extract_handler_body(recipes_src)
    assert "PdfRenderTimeout" in body, (
        "P1-AUDIT-2 regresión: error name `PdfRenderTimeout` perdido en "
        "Recipes.jsx. Sin él, los timeouts se mezclan con errores reales "
        "del html2canvas en analytics — operador no puede discriminar."
    )


# ---------------------------------------------------------------------------
# 4. clearTimeout en finally
# ---------------------------------------------------------------------------
def test_settimeout_cleared_in_finally(recipes_src: str):
    """El `setTimeout` del reject debe limpiarse en `finally` para evitar
    leak de handles (caso `save()` resuelve antes del timeout)."""
    body = _extract_handler_body(recipes_src)
    handle_idx = body.find("_pdfTimeoutHandle")
    assert handle_idx > -1, (
        "P1-AUDIT-2 regresión: handle `_pdfTimeoutHandle` perdido en "
        "Recipes.jsx."
    )
    window = body[handle_idx:handle_idx + 1500]
    assert "clearTimeout" in window, (
        "P1-AUDIT-2 regresión: `clearTimeout(_pdfTimeoutHandle)` ausente. "
        "Sin él, el handle puede disparar reject post-resolve → unhandled "
        "promise rejection en el browser."
    )
    assert re.search(
        r"finally\s*\{[^}]*clearTimeout\(_pdfTimeoutHandle\)",
        window,
    ), (
        "P1-AUDIT-2 regresión: `clearTimeout(_pdfTimeoutHandle)` debe estar "
        "en bloque `finally` para garantizar limpieza incluso si Promise.race "
        "lanza."
    )


# ---------------------------------------------------------------------------
# 5. El catch existente sigue dismisseando el toast (anclar el flujo de UX)
# ---------------------------------------------------------------------------
def test_catch_dismisses_toast(recipes_src: str):
    """El timeout debe propagar al catch existente, que sigue dismisseando
    el toast loading. Sin esto, el toast queda colgado aunque el timeout
    salve el lock."""
    body = _extract_handler_body(recipes_src)
    catch_re = re.compile(
        r"catch\s*\(\s*\w+\s*\)\s*\{[^}]*toast\.dismiss\(toastId\)",
        re.DOTALL,
    )
    assert catch_re.search(body), (
        "P1-AUDIT-2 regresión: el bloque `catch` ya no llama "
        "`toast.dismiss(toastId)`. Sin esto, un PdfRenderTimeout deja el "
        "toast loading colgado aunque el lock se libere."
    )


# ---------------------------------------------------------------------------
# 6. Anchor textual P1-AUDIT-2 presente (grep rápido del fix)
# ---------------------------------------------------------------------------
def test_anchor_present(recipes_src: str):
    """Comment inline `[P1-AUDIT-2 · ...]` cerca del Promise.race para
    `grep -r P1-AUDIT-2` localizar el fix sin abrir el archivo."""
    body = _extract_handler_body(recipes_src)
    assert "P1-AUDIT-2" in body, (
        "P1-AUDIT-2 regresión: anchor textual `[P1-AUDIT-2 · ...]` perdido. "
        "Restaurar para que `grep -r P1-AUDIT-2 frontend/src` localice el "
        "callsite directamente."
    )
