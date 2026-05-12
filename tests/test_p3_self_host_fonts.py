"""[P3-SELF-HOST-FONTS · 2026-05-12] Anchor + regression guard.

Outfit + Plus Jakarta Sans deben servirse desde `frontend/public/fonts/`
(self-hosted) — NO desde `fonts.googleapis.com` / `fonts.gstatic.com`.
Pre-fix index.html cargaba CSS + woff2 desde Google:
  - Privacy leak: IP del visitante a Google en cada page-load.
  - LCP variable: dependía de latencia CDN externo.
  - CSP whitelist obligatoria para fonts.googleapis.com + fonts.gstatic.com.

Defensas que el test enforza:
  1. Anchor `P3-SELF-HOST-FONTS` en index.html + index.css.
  2. `frontend/public/fonts/` existe y contiene ≥4 archivos .woff2.
  3. index.html NO contiene `<link>` a fonts.googleapis.com (en líneas
     no-comentadas — los comments con la URL como referencia histórica
     están permitidos).
  4. index.html NO contiene `dns-prefetch` ni `preconnect` a fonts.*.
  5. index.css contiene declaraciones `@font-face` con `src: url(/fonts/...)`.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_HTML = _REPO_ROOT / "frontend" / "index.html"
_INDEX_CSS = _REPO_ROOT / "frontend" / "src" / "index.css"
_FONTS_DIR = _REPO_ROOT / "frontend" / "public" / "fonts"


def test_anchor_present_in_index_html():
    src = _INDEX_HTML.read_text(encoding="utf-8")
    assert "P3-SELF-HOST-FONTS" in src


def test_anchor_present_in_index_css():
    src = _INDEX_CSS.read_text(encoding="utf-8")
    assert "P3-SELF-HOST-FONTS" in src


def test_fonts_dir_exists_with_woff2_files():
    assert _FONTS_DIR.is_dir(), (
        f"Falta directorio {_FONTS_DIR.relative_to(_REPO_ROOT)}. "
        "Self-hosted fonts deben vivir ahí."
    )
    woff2s = list(_FONTS_DIR.glob("*.woff2"))
    assert len(woff2s) >= 4, (
        f"Se esperaban >=4 archivos .woff2 en {_FONTS_DIR.relative_to(_REPO_ROOT)}, "
        f"encontrados: {len(woff2s)}. Outfit + Plus Jakarta Sans necesitan "
        f"al menos latin + latin-ext per family."
    )


def _strip_html_comments(html: str) -> str:
    """Elimina `<!-- ... -->` para validar contenido activo."""
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def test_no_active_link_to_google_fonts_css():
    """En LÍNEAS NO-COMENTADAS, no debe haber `<link href="https://fonts.googleapis.com">`.
    Comments con la URL como referencia histórica/explicación están OK."""
    src = _INDEX_HTML.read_text(encoding="utf-8")
    active = _strip_html_comments(src)
    pat = re.compile(
        r"<link[^>]*\bhref\s*=\s*[\"']https://fonts\.googleapis\.com",
        re.IGNORECASE,
    )
    bad = pat.findall(active)
    assert not bad, (
        f"index.html aún tiene `<link>` activo a fonts.googleapis.com: "
        f"{bad[0][:120]!r}. Self-hosted fonts deben servirse desde /fonts/, "
        "no via CDN externo."
    )


def test_no_dns_prefetch_or_preconnect_to_google_fonts():
    """`dns-prefetch` y `preconnect` para fonts.* perdieron sentido tras
    self-host — son señales al browser de "vas a necesitar conectar a este
    host pronto". Si nunca conectamos, son ruido."""
    src = _INDEX_HTML.read_text(encoding="utf-8")
    active = _strip_html_comments(src)
    bad_patterns = [
        r"rel\s*=\s*[\"']dns-prefetch[\"'][^>]*href\s*=\s*[\"']https://fonts\.googleapis",
        r"rel\s*=\s*[\"']preconnect[\"'][^>]*href\s*=\s*[\"']https://fonts\.googleapis",
        r"rel\s*=\s*[\"']dns-prefetch[\"'][^>]*href\s*=\s*[\"']https://fonts\.gstatic",
        r"rel\s*=\s*[\"']preconnect[\"'][^>]*href\s*=\s*[\"']https://fonts\.gstatic",
    ]
    for pat in bad_patterns:
        m = re.search(pat, active, re.IGNORECASE)
        assert m is None, (
            f"index.html aún tiene `{m.group(0)[:80]}...` activo. "
            "Tras self-host, dns-prefetch/preconnect a fonts.* son ruido."
        )


def test_css_has_local_font_face_declarations():
    src = _INDEX_CSS.read_text(encoding="utf-8")
    # Debe haber al menos un @font-face con src: url(/fonts/...)
    pat = re.compile(
        r"@font-face\s*\{[^}]*src\s*:\s*url\(/fonts/[^)]+\.woff2\)",
        re.DOTALL,
    )
    matches = pat.findall(src)
    assert len(matches) >= 8, (
        f"Se esperaban >=8 declaraciones `@font-face` con `src: url(/fonts/...)` "
        f"en index.css (Outfit 5 weights + PJS 4 weights × subsets). "
        f"Encontradas: {len(matches)}."
    )


def test_css_no_remote_gstatic_url_in_font_face():
    """En @font-face declarations, NO debe quedar referencia a fonts.gstatic.com.
    Si quedó alguna, el browser sigue pegándole al CDN externo."""
    src = _INDEX_CSS.read_text(encoding="utf-8")
    # Buscar dentro de @font-face blocks
    font_face_blocks = re.findall(r"@font-face\s*\{[^}]*\}", src, re.DOTALL)
    bad: list[str] = []
    for block in font_face_blocks:
        if "fonts.gstatic.com" in block:
            bad.append(block[:120])
    assert not bad, (
        f"@font-face block(s) referencian fonts.gstatic.com aún: "
        f"{bad[:2]}. Reemplazar con /fonts/ path local."
    )


def test_preload_hints_present():
    """index.html debe tener `<link rel="preload">` para los 2 archivos
    woff2 más críticos (latin subset, Outfit + Plus Jakarta Sans).
    Sin preload, browser descubre las fuentes solo cuando parsea index.css
    → flash of unstyled text en LCP."""
    src = _INDEX_HTML.read_text(encoding="utf-8")
    active = _strip_html_comments(src)
    pat = re.compile(
        r'<link\s+rel\s*=\s*["\']preload["\'][^>]*href\s*=\s*["\']/fonts/[^"\']+\.woff2["\'][^>]*as\s*=\s*["\']font["\']',
    )
    matches = pat.findall(active)
    assert len(matches) >= 2, (
        f"Se esperaban >=2 `<link rel=preload as=font>` para /fonts/*.woff2 "
        f"(Outfit + Plus Jakarta Sans latin subset). Encontrados: {len(matches)}. "
        "Sin preload, FOUT/CLS en LCP."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-SELF-HOST-FONTS" in src
