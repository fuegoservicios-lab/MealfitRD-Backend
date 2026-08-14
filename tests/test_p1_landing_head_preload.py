"""[P1-LANDING-HEAD-PRELOAD · 2026-08-14] El `<head>` deja de ser el mismo para
los dos hosts, y hay que comprobarlo contra el HTML COMPILADO.

POR QUÉ ESTE TEST EXISTE ADEMÁS DEL DE VITEST. El contrato de
`LandingHead.p1_landing_head_preload.test.js` prueba la lógica con un bundle
falso: qué se elige y qué se descarta. Lo que NO puede ver es el modo de fallo
que de verdad da miedo aquí — **el silencioso**.

Los chunks llevan hash de contenido. Un `modulepreload` que apunta a una URL que
ya no existe NO rompe la página: el navegador se come el 404 y sigue. La
optimización simplemente deja de funcionar, sin un solo síntoma, y nadie se
entera hasta que alguien vuelve a medir meses después. Es el mismo patrón que en
este repo dejó los sourcemaps subiéndose bajo un release que no casaba y el
guard de `--release` creído durante semanas.

Por eso, cuando hay `dist/`, este test abre el HTML compilado y comprueba que
CADA fichero referenciado existe en disco.

Tooltip-anchor: P1-LANDING-HEAD-PRELOAD
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_INDEX_HTML = _FRONTEND / "index.html"
_VITE_CONFIG = _FRONTEND / "vite.config.js"
_MAIN_JSX = _FRONTEND / "src" / "main.jsx"
_HOME_JSX = _FRONTEND / "src" / "pages" / "Home.jsx"
_PLUGIN = _FRONTEND / "scripts" / "landingHead.mjs"
_DIST = _FRONTEND / "dist"
_DIST_INDEX = _DIST / "index.html"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P1-LANDING-HEAD-PRELOAD] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Fuente: el plugin está cableado y el preconnect ya no es incondicional
# ---------------------------------------------------------------------------

def test_el_plugin_esta_cableado_en_vite():
    contenido = _read(_VITE_CONFIG)
    assert "landingHeadSnippet" in contenido and "transformIndexHtml" in contenido, (
        "[P1-LANDING-HEAD-PRELOAD] `vite.config.js` ya no invoca el plugin del "
        "`<head>`. Sin él, el chunk del landing vuelve a descubrirse sólo después "
        "de descargar Y parsear el entry: las dos olas (629.800 B + 226.434 B) "
        "vuelven a ir en serie para el 100% de los visitantes del apex."
    )


def test_el_preconnect_de_auth_ya_no_esta_fijo_en_el_html():
    """Un HTML, dos audiencias: el apex nunca contacta ese origen."""
    contenido = _read(_INDEX_HTML)
    enlaces = re.findall(r"<link[^>]*rel=[\"'](?:preconnect|dns-prefetch)[\"'][^>]*>", contenido)
    culpables = [e for e in enlaces if "neonauth" in e or "neon.tech" in e]
    assert not culpables, (
        "[P1-LANDING-HEAD-PRELOAD] Volvió un `preconnect`/`dns-prefetch` fijo al "
        f"host de autenticación: {culpables}\n"
        "El apex NO lo contacta jamás — `AssessmentContext` corta la sesión en seco "
        "con `isApexHost()` (P3-APEX-NO-SESSION) — así que la portada pagaría un "
        "DNS+TCP+TLS contra us-east-1 en su critical path a cambio de nada.\n"
        "Sigue emitiéndose, pero desde el plugin y sólo fuera del apex."
    )


def test_el_host_de_auth_no_esta_escrito_a_mano():
    """Se deriva de VITE_NEON_AUTH_URL: una segunda copia fue el origen del bug."""
    assert "VITE_NEON_AUTH_URL" in _read(_VITE_CONFIG), (
        "[P1-LANDING-HEAD-PRELOAD] El origen de Neon Auth volvió a estar escrito a "
        "mano. Derivarlo del env evita la clase entera: el preconnect anterior "
        "quedó apuntando a un host equivocado porque nadie recordó actualizar la "
        "copia del HTML."
    )


# ---------------------------------------------------------------------------
# 2. El splash espera al contenido, pero con techo
# ---------------------------------------------------------------------------

def test_la_portada_avisa_de_que_ya_monto():
    assert "mealfit:landing-ready" in _read(_HOME_JSX), (
        "[P1-LANDING-HEAD-PRELOAD] `Home.jsx` ya no emite `mealfit:landing-ready`.\n"
        "Sin esa señal el splash se descarta con `mealfit:app-ready`, que en el apex "
        "resuelve SÍNCRONO — o sea mientras el chunk de la portada todavía viene por "
        "la red — y el usuario ve splash → hueco vacío → contenido."
    )


def test_main_espera_esa_senal_solo_en_la_portada():
    contenido = _read(_MAIN_JSX)
    assert "mealfit:landing-ready" in contenido, (
        "[P1-LANDING-HEAD-PRELOAD] `main.jsx` no escucha `mealfit:landing-ready`: "
        "la señal se emite y nadie la recoge."
    )
    assert "pathname === '/'" in contenido, (
        "[P1-LANDING-HEAD-PRELOAD] El gate de la señal dejó de acotarse a `/`.\n"
        "Las demás rutas de papel (precios, legales, novedades) NO emiten "
        "`landing-ready`: esperarla ahí dejaría su splash colgado hasta el fallback."
    )


def test_el_fallback_del_splash_sigue_vivo():
    """Es el techo que hace seguro esperar a una señal que podría no llegar."""
    assert re.search(r"setTimeout\(hideSplash,\s*2500\)", _read(_MAIN_JSX)), (
        "[P1-LANDING-HEAD-PRELOAD] Desapareció el fallback de 2,5 s del splash.\n"
        "Es justo lo que permite condicionar el descarte a un evento: si la portada "
        "fallara en montar, sin ese techo el usuario se queda mirando el splash para "
        "siempre."
    )


# ---------------------------------------------------------------------------
# 3. Contra el HTML COMPILADO: nada apunta al vacío
# ---------------------------------------------------------------------------

def _bloque_inyectado(html: str) -> str:
    m = re.search(r"var esApex[^\n]*\n(.*?)\n\}\)\(\);", html, re.DOTALL)
    return m.group(0) if m else ""


@pytest.mark.skipif(not _DIST_INDEX.exists(), reason="sin dist/ (corre `npm run build`)")
def test_dist_lleva_el_bloque_gateado_por_host():
    bloque = _bloque_inyectado(_read(_DIST_INDEX))
    assert bloque, (
        "[P1-LANDING-HEAD-PRELOAD] El `dist/index.html` construido NO lleva el "
        "bloque del `<head>` por host. El plugin corrió sin efecto."
    )
    assert "bioboros.com" in bloque, (
        "[P1-LANDING-HEAD-PRELOAD] El bloque no comprueba el host."
    )


@pytest.mark.skipif(not _DIST_INDEX.exists(), reason="sin dist/ (corre `npm run build`)")
def test_dist_no_precarga_el_landing_de_forma_incondicional():
    """Un preload suelto le metería 226 kB de landing a app.bioboros.com."""
    html = _read(_DIST_INDEX)
    bloque = _bloque_inyectado(html)
    fuera = html.replace(bloque, "") if bloque else html
    sueltos = re.findall(r"<link[^>]*(?:Home-[A-Za-z0-9_-]+\.(?:js|css))[^>]*>", fuera)
    assert not sueltos, (
        f"[P1-LANDING-HEAD-PRELOAD] Hay `<link>` al chunk del landing FUERA del "
        f"bloque gateado: {sueltos}\n"
        "Con un solo `index.html` para los dos hosts, eso le da al subdominio de la "
        "app el peso del landing eager — exactamente lo que P3-APP-SUBDOMAIN-BUILD-SEP "
        "quitó de ahí."
    )


@pytest.mark.skipif(not _DIST_INDEX.exists(), reason="sin dist/ (corre `npm run build`)")
def test_todo_lo_que_se_precarga_existe_en_disco():
    """EL modo de fallo silencioso: un preload a un hash viejo no rompe nada."""
    bloque = _bloque_inyectado(_read(_DIST_INDEX))
    referencias = sorted(set(re.findall(r"/(assets/[A-Za-z0-9._-]+\.(?:js|css))", bloque)))
    assert referencias, (
        "[P1-LANDING-HEAD-PRELOAD] El bloque no referencia ningún asset: el plugin "
        "no encontró el chunk de la portada (¿se renombró `src/pages/Home.jsx`?). "
        "No rompe el build a propósito, pero la optimización quedó inerte."
    )
    fantasmas = [r for r in referencias if not (_DIST / r).exists()]
    assert not fantasmas, (
        f"[P1-LANDING-HEAD-PRELOAD] Se precargan ficheros que NO existen: {fantasmas}\n"
        "Este es el fallo silencioso que este test existe para cazar: el navegador se "
        "come el 404 sin romper la página, la optimización deja de servir, y no hay "
        "ningún síntoma que lo delate."
    )
