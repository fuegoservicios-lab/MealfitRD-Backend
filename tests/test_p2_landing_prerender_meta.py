"""[P2-LANDING-PRERENDER-META · 2026-08-14] Un `<head>` de verdad por cada ruta
pública.

EL DEFECTO. `index.html` era el ÚNICO HTML que existía, con `canonical`,
`og:url`, `og:title` y `og:description` apuntando todos a la portada.
`RouteTitle.jsx` los corrige en el cliente, pero eso sólo sirve a quien ejecuta
JavaScript — y los unfurlers sociales no lo hacen. El propio comentario de
`RouteTitle` declaraba el hueco por escrito.

Consecuencia: compartir `/precios`, `/motor` o un artículo de `/novedades` por
WhatsApp —el canal #1 en RD, y de donde viene el crecimiento de este producto—
desplegaba la tarjeta de la PORTADA. Y por la especificación de Open Graph,
`og:url` es el identificador permanente del objeto: las 20 URLs resolvían al
mismo objeto social.

⚠️ ESTO NO ES SSR. El `<body>` sigue siendo el shell vacío; sólo cambia el
`<head>`. Es deliberado: resuelve el problema real sin meter Chromium en el build
que corre en el VPS ni reescribir el árbol de rutas.

LAS TRES TRAMPAS QUE LO VOLVERÍAN INERTE O DAÑINO, y dónde se cierra cada una:
  1. `try_files` sin `$uri/` ⇒ nginx ignora estos ficheros EN SILENCIO.
     Verificado contra producción el 2026-08-14: lo incluye.
  2. `location = /index.html` (exacto) ⇒ los HTML nuevos no heredan el
     `no-cache`, y un HTML cacheado que referencia hashes ya borrados es una
     página rota hasta limpiar caché a mano. Cambiado a regex en el mismo P-fix.
  3. `globPatterns` incluye `**/*.html` ⇒ las ~18 copias al precache.

Tooltip-anchor: P2-LANDING-PRERENDER-META
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_SRC = _FRONTEND / "src"
_META = _SRC / "data" / "routeMeta.js"
_ROUTE_TITLE = _SRC / "components" / "layout" / "RouteTitle.jsx"
_SCRIPT = _FRONTEND / "scripts" / "build-route-meta.mjs"
_PKG = _FRONTEND / "package.json"
_NGINX = _REPO_ROOT / "backend" / "infra" / "nginx" / "mealfit.conf"
_DIST = _FRONTEND / "dist"


def _read(p: Path) -> str:
    if not p.exists():
        pytest.fail(f"[P2-LANDING-PRERENDER-META] No existe {p}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. El copy vive en UN sitio
# ---------------------------------------------------------------------------

def test_el_copy_por_ruta_tiene_un_solo_dueno():
    meta = _read(_META)
    for tabla in ("TITLES", "DESCRIPTIONS"):
        assert f"export const {tabla}" in meta, (
            f"[P2-LANDING-PRERENDER-META] `{tabla}` no está en `data/routeMeta.js`."
        )
    rt = _read(_ROUTE_TITLE)
    assert not re.search(r"^const (?:TITLES|DESCRIPTIONS)\s*=", rt, re.MULTILINE), (
        "[P2-LANDING-PRERENDER-META] `RouteTitle.jsx` volvió a declarar las tablas "
        "en local. Ahora tienen DOS consumidores —el runtime y el build— y dos "
        "copias del copy es la duplicación que este P-fix cierra."
    )
    assert "routeMeta" in rt, (
        "[P2-LANDING-PRERENDER-META] `RouteTitle.jsx` ya no importa el SSOT."
    )


def test_las_paginas_self_managed_tienen_su_titulo_en_el_ssot():
    """Sin esto, su título no existía en ninguna tabla: no se podía estampar."""
    meta = _read(_META)
    for ruta in ("/motor", "/como-funciona", "/funciones", "/precision", "/research"):
        assert f"'{ruta}':" in meta, (
            f"[P2-LANDING-PRERENDER-META] Falta el título de `{ruta}` en el SSOT. "
            "Esas cinco páginas fijan su `document.title` con un `useEffect` propio, "
            "así que si no está aquí el HTML servido se queda con el de la portada."
        )


def test_el_titulo_del_ssot_y_el_de_la_pagina_no_pueden_divergir():
    """Cada self-managed repone su título al montar: tienen que decir lo mismo."""
    meta = _read(_META)
    paginas = {
        "/motor": "Engine.jsx",
        "/como-funciona": "HowItWorksPage.jsx",
        "/funciones": "FeaturesPage.jsx",
        "/precision": "PrecisionPage.jsx",
        "/research": "ResearchPage.jsx",
    }
    for ruta, fichero in paginas.items():
        m = re.search(rf"'{re.escape(ruta)}':\s*'([^']+)'", meta)
        assert m, f"[P2-LANDING-PRERENDER-META] Sin título para {ruta} en el SSOT."
        fuente = _read(_SRC / "pages" / fichero)
        assert m.group(1) in fuente, (
            f"[P2-LANDING-PRERENDER-META] El título de `{ruta}` divergió: el SSOT dice "
            f"«{m.group(1)}» y `{fichero}` fija otro. El HTML servido y el que ve el "
            "usuario al navegar dirían cosas distintas."
        )


# ---------------------------------------------------------------------------
# 2. El prerender está cableado y no puede nacer inerte
# ---------------------------------------------------------------------------

def test_el_prerender_corre_en_cada_build():
    pkg = _read(_PKG)
    assert '"postbuild"' in pkg and "build-route-meta" in pkg, (
        "[P2-LANDING-PRERENDER-META] El prerender no está en `postbuild`. Un script "
        "que hay que acordarse de correr a mano no corre."
    )


def test_nginx_sirve_los_html_por_ruta_SIN_redirigir():
    """`$uri/index.html`, y la diferencia con `$uri/` no es cosmética.

    Con `$uri/` nginx encuentra el DIRECTORIO que crea el prerender y emite su
    redirect de barra final. Medido en producción el 2026-08-14, antes de
    corregirlo: `GET /precios` devolvía `301 → /precios/`. El contenido estaba
    detrás y funcionaba, pero cobraba un salto a cada enlace interno y dejaba la
    URL servida distinta de la canónica que el propio HTML declara — la
    incoherencia que este P-fix venía a cerrar.
    """
    conf = _read(_NGINX)
    assert "try_files $uri $uri/index.html /index.html" in conf, (
        "[P2-LANDING-PRERENDER-META] El fallback del SPA no pide el fichero.\n"
        "Con `$uri/` (a secas) nginx redirige 301 a la ruta con barra; sin `$uri` "
        "ni `$uri/index.html` no encuentra los HTML por ruta y sirve la portada, "
        "dejando el prerender inerte en silencio."
    )
    assert "try_files $uri $uri/ /index.html" not in conf, (
        "[P2-LANDING-PRERENDER-META] Volvió `$uri/`: reintroduce el 301 de barra final."
    )


def test_los_html_por_ruta_no_se_cachean_para_siempre():
    conf = _read(_NGINX)
    assert re.search(r"location\s+~\s+/index\\?\.html\$", conf), (
        "[P2-LANDING-PRERENDER-META] La regla de `no-cache` volvió a ser exacta "
        "(`location = /index.html`).\n"
        "Los HTML por ruta no la heredarían, y un HTML cacheado que referencia "
        "hashes que el siguiente deploy borra es una página rota hasta que el "
        "usuario limpie caché a mano."
    )


# ---------------------------------------------------------------------------
# 3. Contra el artefacto: que cada HTML diga lo suyo
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_DIST / "precios" / "index.html").exists(),
                    reason="sin dist/ prerenderizado (corre `npm run build`)")
@pytest.mark.parametrize("ruta,fragmento", [
    ("precios", "Planes y Precios"),
    ("motor", "el motor de Bioboros"),
    ("supermercado", "Supermercados RD"),
    ("privacy", "Política de Privacidad"),
])
def test_cada_html_lleva_su_propio_titulo(ruta, fragmento):
    html = _read(_DIST / ruta / "index.html")
    m = re.search(r"<title>([^<]*)</title>", html)
    assert m and fragmento in m.group(1), (
        f"[P2-LANDING-PRERENDER-META] `/{ruta}` sirve el título «{m.group(1) if m else '?'}» "
        f"en vez de uno que contenga «{fragmento}»."
    )


@pytest.mark.skipif(not (_DIST / "precios" / "index.html").exists(), reason="sin dist/ prerenderizado")
def test_cada_html_se_declara_canonico_de_SI_MISMO():
    """El fallo original: 20 URLs declarando ser la portada."""
    for ruta in ("precios", "motor", "supermercado", "about"):
        html = _read(_DIST / ruta / "index.html")
        for etiqueta, patron in (
            ("canonical", r'<link rel="canonical" href="([^"]*)"'),
            ("og:url", r'<meta property="og:url" content="([^"]*)"'),
        ):
            m = re.search(patron, html)
            assert m, f"[P2-LANDING-PRERENDER-META] `/{ruta}` sin {etiqueta}."
            assert m.group(1).rstrip("/").endswith(ruta), (
                f"[P2-LANDING-PRERENDER-META] `/{ruta}` declara {etiqueta}="
                f"«{m.group(1)}». Por especificación de Open Graph, `og:url` es el "
                "ID permanente del objeto: si todas apuntan a la portada, para "
                "Facebook y WhatsApp las 20 rutas son el MISMO objeto."
            )


@pytest.mark.skipif(not (_DIST / "custom-sw.js").exists(), reason="sin dist/")
def test_los_html_por_ruta_no_entran_al_precache():
    sw = _read(_DIST / "custom-sw.js")
    htmls = [u for u in re.findall(r'"url":"([^"]+)"', sw) if u.endswith(".html")]
    assert htmls == ["index.html"], (
        f"[P2-LANDING-PRERENDER-META] El precache lleva HTML de más: {htmls}.\n"
        "Sólo el `index.html` de la raíz debe precacharse (es el fallback offline "
        "del SPA). Los de ruta suman ~250 KB por visitante y su único consumidor "
        "son unfurlers y crawlers, que no instalan service worker."
    )
