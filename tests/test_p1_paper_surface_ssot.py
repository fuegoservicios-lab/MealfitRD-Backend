"""[P1-PAPER-SURFACE-SSOT · 2026-08-01] La lista de rutas con superficie
`paper` vive en TRES sitios que no pueden importarse entre sí:

    1. frontend/src/utils/paperSurface.js   — SSOT de JS
    2. frontend/index.html                  — boot script inline (no puede importar)
    3. frontend/src/utils/marketingRoutes.js — la lista del HEADER, que hoy
       coincide pero gobierna un alcance distinto (19 rutas vs 6)

Antes de este P-fix la duplicación (1)↔(2) estaba sostenida SOLO por un
comentario: cero tests. Tocar una sola producía flash de tema en carga
directa o refresh, que es justo el fallo que el boot script existe para evitar.

Este test ancla las tres copias y, además, que la separación de
responsabilidades no se deshaga: `Header.jsx` debe seguir usando
`isMarketingRoute` (19 rutas), NO `isPaperSurface` (6 rutas).

Tooltip-anchor: P1-PAPER-SURFACE-SSOT
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_PAPER_JS = _FRONTEND / "src" / "utils" / "paperSurface.js"
_MARKETING_JS = _FRONTEND / "src" / "utils" / "marketingRoutes.js"
_INDEX_HTML = _FRONTEND / "index.html"
_APP_JSX = _FRONTEND / "src" / "App.jsx"
_HEADER_JSX = _FRONTEND / "src" / "components" / "layout" / "Header.jsx"

_EXPECTED = ["/", "/precios", "/como-funciona", "/funciones", "/precision", "/motor"]


def _routes_from_array(text: str, const_name: str) -> list[str]:
    """Extrae los literales de un `export const NAME = ['/a', '/b'];`."""
    m = re.search(
        const_name + r"\s*=\s*\[(?P<body>[^\]]*)\]",
        text,
        re.DOTALL,
    )
    assert m is not None, f"No se encontró el array `{const_name}`."
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group("body"))


def test_paper_surface_module_exists_and_lists_the_six_routes():
    assert _PAPER_JS.exists(), (
        "P1-PAPER-SURFACE-SSOT: falta `frontend/src/utils/paperSurface.js`."
    )
    text = _PAPER_JS.read_text(encoding="utf-8")
    assert _routes_from_array(text, "PAPER_SURFACE_ROUTES") == _EXPECTED
    assert "export const isPaperSurface" in text, (
        "P1-PAPER-SURFACE-SSOT: `paperSurface.js` debe exportar `isPaperSurface`."
    )


def test_boot_script_copy_matches_the_ssot():
    """El boot script de index.html no puede importar el módulo, así que
    lleva una copia literal. Si divergen, la carga en frío parpadea."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert _routes_from_array(html, r"var\s+PAPER") == _EXPECTED, (
        "P1-PAPER-SURFACE-SSOT: la copia del boot script en index.html "
        "divergió de paperSurface.js. Las dos tienen que viajar juntas."
    )


def test_theme_call_sites_use_paper_surface_not_marketing_routes():
    """Los 3 call sites del TEMA usan `isPaperSurface`. Si vuelven a
    `isMarketingRoute`, la separación se deshizo."""
    app = _APP_JSX.read_text(encoding="utf-8")
    assert "isPaperSurface" in app, (
        "P1-PAPER-SURFACE-SSOT: App.jsx debe usar `isPaperSurface` para el tema."
    )
    assert "isMarketingRoute" not in app, (
        "P1-PAPER-SURFACE-SSOT: App.jsx ya no debe consultar `isMarketingRoute` — "
        "esa lista gobierna el HEADER (19 rutas), no el tema (6)."
    )


def test_header_still_uses_marketing_routes():
    """El HEADER conserva su alcance de 19 rutas. Si alguien lo repunta a
    `isPaperSurface`, las legales y /supermercado pierden nav y CTA."""
    header = _HEADER_JSX.read_text(encoding="utf-8")
    assert "isMarketingRoute" in header, (
        "P1-PAPER-SURFACE-SSOT: Header.jsx debe seguir usando `isMarketingRoute`."
    )
    assert "isPaperSurface" not in header, (
        "P1-PAPER-SURFACE-SSOT: Header.jsx NO debe usar `isPaperSurface` — "
        "el header cubre 19 rutas, la superficie papel solo 6."
    )


def test_marketing_routes_still_lists_the_same_six_today():
    """Hoy ambas listas coinciden. Cuando dejen de coincidir (p.ej. si
    /supermercado pasa a papel), este test hay que RELAJARLO a propósito,
    no borrarlo: es el aviso de que la separación empezó a importar."""
    text = _MARKETING_JS.read_text(encoding="utf-8")
    assert _routes_from_array(text, "MARKETING_ROUTES") == _EXPECTED


# ---------------------------------------------------------------------------
# 2. Los lectores que comparan 'dark' en duro
# ---------------------------------------------------------------------------
_USE_THEME_COLOR = _FRONTEND / "src" / "components" / "common" / "useThemeColor.js"
_MANIFEST = _FRONTEND / "public" / "manifest.json"

_PAPER_HEX = "#FBFBFA"
_OLD_BRAND_INDIGO = "#4F46E5"


def test_use_theme_color_has_a_paper_branch():
    """`isDarkActive()` compara `=== 'dark'` en duro (theme.js:96), así que
    bajo `paper` devuelve False y las 5 rutas de marketing que no son `/`
    caen al `else` final → #4F46E5 indigo en la barra de estado de Android
    y en el PWA standalone de iOS, sobre una página blanco y negro."""
    text = _USE_THEME_COLOR.read_text(encoding="utf-8")
    assert "isPaperSurface" in text, (
        "P1-PAPER-THEME: useThemeColor.js debe consultar `isPaperSurface` y "
        "devolver el papel para las 6 rutas de marketing."
    )
    assert _PAPER_HEX in text, (
        f"P1-PAPER-THEME: useThemeColor.js debe emitir {_PAPER_HEX} en la rama papel."
    )


def test_splash_has_paper_rules():
    """El splash vive FUERA de #root, así que no lo alcanza ningún CSS de
    React. Sin reglas propias cae a su base: dos radiales indigo + rosa
    sobre #F8FAFC, en cada carga directa o refresh de las 6 rutas."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'html[data-theme="paper"] #pwa-splash' in html, (
        "P1-PAPER-THEME: falta la regla del splash para la superficie papel."
    )
    assert f'<meta name="theme-color" content="{_PAPER_HEX}"' in html, (
        f"P1-PAPER-THEME: el theme-color por defecto debe ser {_PAPER_HEX}."
    )


def test_no_brand_indigo_left_in_pwa_surfaces():
    html = _INDEX_HTML.read_text(encoding="utf-8")
    manifest = _MANIFEST.read_text(encoding="utf-8")
    assert _OLD_BRAND_INDIGO not in manifest, (
        f"P1-PAPER-THEME: manifest.json sigue con {_OLD_BRAND_INDIGO} — el splash "
        "nativo de Android y el chrome de la PWA instalada quedarian indigo."
    )
    assert _OLD_BRAND_INDIGO not in html, (
        f"P1-PAPER-THEME: index.html sigue con {_OLD_BRAND_INDIGO}."
    )
