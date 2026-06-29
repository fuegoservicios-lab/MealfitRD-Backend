"""[P3-RECIPES-CHUNK-LOAD-FAIL · 2026-05-15] Anchor + regression guard.

Pre-fix: `frontend/src/pages/Recipes.jsx::handleDownloadPDF` y
`frontend/src/pages/Dashboard.jsx::handleDownloadPDF` hacían
`const html2pdf = (await import('html2pdf.js')).default` sin try/catch
dedicado. Si el CDN dropea el chunk (red intermitente, rotación de
build invalidando hashes mientras la pestaña vive), el outer try/catch del
handler reporta un toast genérico "Error al descargar PDF" — usuario no
sabe que fue un fail de red y no intenta refresh + retry (que arreglaría
el caso).

Fix: wrap dedicado del `await import('html2pdf.js')` con check para
`ChunkLoadError` y mensaje específico que sugiere refresh.

Defensas que este test enforza:
  1. Anchor `P3-RECIPES-CHUNK-LOAD-FAIL` en ambos archivos.
  2. CADA `await import('html2pdf.js')` está dentro de un try/catch que
     menciona `ChunkLoadError` (textualmente, como nombre del error).
  3. El catch dismisses el toast de loading + (Dashboard) libera el pdfLock.
  4. El mensaje específico textual ("Error de red al cargar el PDF.
     Refresca la página e intenta de nuevo.") está presente — anclamos
     copy exacto porque cualquier rephrase rompe el reconocimiento del
     mensaje en soporte y métricas de Sentry.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchors
# ---------------------------------------------------------------------------
def test_anchor_present_in_recipes():
    assert "P3-RECIPES-CHUNK-LOAD-FAIL" in _read(_RECIPES_JSX), (
        "Falta anchor `P3-RECIPES-CHUNK-LOAD-FAIL` en Recipes.jsx."
    )


def test_anchor_present_in_dashboard():
    assert "P3-RECIPES-CHUNK-LOAD-FAIL" in _read(_DASHBOARD_JSX), (
        "Falta anchor `P3-RECIPES-CHUNK-LOAD-FAIL` en Dashboard.jsx."
    )


# ---------------------------------------------------------------------------
# 2. await import('html2pdf.js') está dentro de try { ... } catch
# ---------------------------------------------------------------------------
def _has_try_wrap_around_html2pdf_import(src: str) -> bool:
    """Verifica que existe el bloque sintáctico canónico

        try {
            html2pdf = (await import('html2pdf.js')).default;
        } catch (importErr) {
            ...

    en el archivo, sin importar comentarios o whitespace previo. La regex
    es tolerante a indentación variable + line breaks dentro del try body.
    Adicionalmente exige la mención textual `ChunkLoadError` en el archivo
    (anchor del modo de fallo detectado en el catch).
    """
    canonical_pat = re.compile(
        r"try\s*\{\s*\n"
        r"\s*html2pdf\s*=\s*\(\s*await\s+import\(\s*['\"]html2pdf\.js['\"]\s*\)\s*\)\s*\.\s*default\s*;\s*\n"
        r"\s*\}\s*catch\s*\(\s*importErr\s*\)\s*\{",
        re.DOTALL,
    )
    if not canonical_pat.search(src):
        return False
    return "ChunkLoadError" in src


def test_recipes_wraps_html2pdf_import():
    src = _read(_RECIPES_JSX)
    assert _has_try_wrap_around_html2pdf_import(src), (
        "Recipes.jsx::handleDownloadPDF debe envolver "
        "`await import('html2pdf.js')` en `try { ... } catch (importErr) { ... }` "
        "con detección explícita de `ChunkLoadError`."
    )


def test_dashboard_wraps_html2pdf_import():
    src = _read(_DASHBOARD_JSX)
    assert _has_try_wrap_around_html2pdf_import(src), (
        "Dashboard.jsx::handleDownloadPDF debe envolver "
        "`await import('html2pdf.js')` en `try { ... } catch (importErr) { ... }` "
        "con detección explícita de `ChunkLoadError`."
    )


# ---------------------------------------------------------------------------
# 3. El mensaje específico textual está presente
# ---------------------------------------------------------------------------
_EXPECTED_NETWORK_MSG = "Error de red al cargar el PDF. Refresca la página e intenta de nuevo."
_EXPECTED_FALLBACK_MSG = "No se pudo cargar el generador de PDF. Refresca la página e intenta de nuevo."


def test_recipes_uses_specific_messages():
    src = _read(_RECIPES_JSX)
    assert _EXPECTED_NETWORK_MSG in src, (
        f"Recipes.jsx debe usar mensaje específico de red: {_EXPECTED_NETWORK_MSG!r}. "
        "Si se rephrase, romperá métricas de Sentry/PostHog que ya filtran por copy."
    )
    assert _EXPECTED_FALLBACK_MSG in src, (
        f"Recipes.jsx debe usar mensaje fallback: {_EXPECTED_FALLBACK_MSG!r}."
    )


def test_dashboard_uses_specific_messages():
    src = _read(_DASHBOARD_JSX)
    assert _EXPECTED_NETWORK_MSG in src
    assert _EXPECTED_FALLBACK_MSG in src


# ---------------------------------------------------------------------------
# 4. Dashboard libera el pdfLock en el branch del catch
# ---------------------------------------------------------------------------
def test_dashboard_releases_pdf_lock_on_chunk_fail():
    """Si el chunk-load fail no libera `pdfLock.current = false`, el usuario
    queda bloqueado hasta refresh — peor UX que el bug original. El catch
    DEBE liberarlo antes de `return`."""
    src = _read(_DASHBOARD_JSX)
    # Localizar el bloque del catch (importErr) y verificar que tiene
    # `pdfLock.current = false` antes de `return`.
    m = re.search(
        r"catch\s*\(\s*importErr\s*\)\s*\{(.+?)return;",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró el catch del chunk import en Dashboard.jsx."
    catch_body = m.group(1)
    assert "pdfLock.current = false" in catch_body, (
        "Dashboard.jsx catch del chunk-load fail debe liberar `pdfLock.current = false` "
        "antes de `return`. Sin esto, usuario queda bloqueado hasta refresh."
    )
