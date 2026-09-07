"""[P2-LANDING-SITEMAP-SSOT + P2-LANDING-MANIFEST-SHORTCUT · 2026-08-14] Dos
listas de rutas escritas a mano que ya habían derivado de las rutas reales.

EL SITEMAP. `public/sitemap.xml` era una CUARTA copia a mano de la lista de rutas
públicas (las otras tres: `paperSurface.js`, el boot script de `index.html`, y
`marketingRoutes.js`), sin ningún test que la cruzara con `App.jsx`. Y ya le
faltaba `/supermercado` — la página con más contenido único del sitio. Lo que sí
estaba era `/novedades/base-datos-supermercados-rd`, o sea **entró el anuncio del
catálogo y no el catálogo**. La cabecera del fichero decía «act. 2026-07-01», un
día ANTES de que la página existiera.

> Sin exagerar el daño: Google ya descubre `/supermercado` por el enlace del
> footer, presente en el DOM de las 19 rutas. Lo que compra el arreglo es
> prioridad de rastreo y, sobre todo, el guard de deriva — porque la 5ª copia
> llegará igual que llegó ésta.

EL SHORTCUT DEL MANIFEST. `manifest.json` declara un atajo de sistema operativo
«Lista del Súper» → `/dashboard/shopping`. Grep en todo `frontend/`: esa línea es
la ÚNICA aparición de esa ruta. No existe en `App.jsx`, así que cae en el
catch-all 404 — el mismo modo de fallo que P1-PANTRY-ROUTE-ALIAS documenta haber
cerrado en vivo para `/pantry` y `/mi-nevera`.

⚠️ Y por qué el test viejo no lo vio: `test_p3_audit_3_manifest_icons.py` afirma
`len(shortcuts) >= 3`. Congeló la CUENTA sin validar nunca la URL, que es justo
la deriva a cerrar. *Un guard que cuenta elementos no dice nada sobre si apuntan
a alguna parte.*

Tooltip-anchor: P2-LANDING-SITEMAP-SSOT
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_SRC = _FRONTEND / "src"
_APP = _SRC / "App.jsx"
_MANIFEST = _FRONTEND / "public" / "manifest.json"
_SITEMAP = _FRONTEND / "public" / "sitemap.xml"
_GEN = _FRONTEND / "scripts" / "build-sitemap.mjs"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P2-LANDING-SITEMAP-SSOT] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def _rutas_declaradas_en_app() -> set[str]:
    """Los `path=` de todos los `<Route>`, incluidos los que sólo redirigen."""
    return set(re.findall(r'<Route\s+path="([^"]+)"', _read(_APP)))


# ---------------------------------------------------------------------------
# 1. El atajo del manifest apunta a una ruta que existe
# ---------------------------------------------------------------------------

def test_los_atajos_del_manifest_resuelven_a_una_ruta_real():
    manifest = json.loads(_read(_MANIFEST))
    rutas = _rutas_declaradas_en_app()
    rotos = []
    for atajo in manifest.get("shortcuts", []):
        url = (atajo.get("url") or "").split("?")[0].rstrip("/") or "/"
        if url not in rutas:
            rotos.append(f"{atajo.get('name')} → {atajo.get('url')}")
    assert not rotos, (
        f"[P2-LANDING-MANIFEST-SHORTCUT] Atajos del manifest sin ruta que los reciba: {rotos}\n"
        "Un atajo del sistema operativo que cae en el 404 es peor que no existir: el "
        "usuario lo ancló a su pantalla de inicio y le responde «esta página no existe».\n"
        "Es el mismo modo de fallo que P1-PANTRY-ROUTE-ALIAS cerró para /pantry y "
        "/mi-nevera. Si la sección no es una página propia, añade un `<Route>` que "
        "redirija (el manifest sólo se relee cuando el navegador lo refresca)."
    )


# ---------------------------------------------------------------------------
# 2. El sitemap se genera, no se escribe
# ---------------------------------------------------------------------------

def test_existe_el_generador_del_sitemap():
    assert _GEN.exists(), (
        "[P2-LANDING-SITEMAP-SSOT] No existe `frontend/scripts/build-sitemap.mjs`.\n"
        "Un sitemap a mano es la 4ª copia de la lista de rutas públicas y ya "
        "demostró derivar: le faltaba /supermercado mientras sí listaba el "
        "artículo que la anunciaba."
    )


def test_el_sitemap_incluye_el_supermercado():
    assert "<loc>https://bioboros.com/supermercado</loc>" in _read(_SITEMAP), (
        "[P2-LANDING-SITEMAP-SSOT] Falta `/supermercado` del sitemap — la página "
        "con más contenido único del sitio."
    )


def test_el_sitemap_no_lista_redirecciones():
    """Indexar una URL que redirige gasta rastreo y produce un duplicado."""
    sitemap = _read(_SITEMAP)
    for ruta, motivo in [
        ("/cookies", "es `<Navigate to=\"/privacy\">` (P3-COOKIES-MERGE)"),
        ("/login", "en el apex sólo redirige por JS a app.* (P3-APP-SUBDOMAIN-ROUTING)"),
    ]:
        assert f"<loc>https://bioboros.com{ruta}</loc>" not in sitemap, (
            f"[P2-LANDING-SITEMAP-SSOT] El sitemap lista `{ruta}`, que {motivo}."
        )


def test_el_sitemap_no_indexa_noticias_que_son_enlaces():
    """Una noticia con `href` apunta a otra página: su slug sería un duplicado."""
    noticias = _read(_SRC / "data" / "news.js")
    # Un slug es "de enlace" si su bloque declara `href`.
    bloques = re.split(r"\n    \{", noticias)
    sitemap = _read(_SITEMAP)
    for bloque in bloques:
        m = re.search(r"slug:\s*'([^']+)'", bloque)
        if m and "href:" in bloque:
            assert f"/novedades/{m.group(1)}</loc>" not in sitemap, (
                f"[P2-LANDING-SITEMAP-SSOT] La noticia `{m.group(1)}` lleva `href` "
                "(apunta a otra página): indexar su slug crea un duplicado."
            )


def test_el_html_ofrece_enlaces_sin_javascript():
    """[P2-LANDING-HEAD-CLIENT] Un cliente sin JS no veía NI UN enlace.

    El `<body>` servido es `<div id="root">` vacío más un splash con
    `role="status"` y la palabra «Cargando». Para un crawler que no renderiza
    —Bing, GPTBot, ClaudeBot, PerplexityBot— y para quien navega con JS
    desactivado, el sitio entero es un indicador de carga sin salida.

    Un `<noscript>` con las rutas principales no arregla el SEO (eso es el
    prerender del gap 7), pero convierte «nada» en «algo navegable» por el coste
    de cinco líneas.
    """
    # Los comentarios HTML se quitan ANTES de buscar la etiqueta: el comentario de
    # P3-SELF-HOST-FONTS menciona literalmente «<noscript> fallback a Google» al
    # explicar qué se retiró, y la búsqueda empezaba ahí — tragándose medio <head>
    # y dando por enlaces del bloque los preloads de las fuentes. Tercera vez hoy
    # que una prosa que DESCRIBE código confunde a un guard que lo busca.
    html = re.sub(r"<!--.*?-->", "", _read(_FRONTEND / "index.html"), flags=re.DOTALL)
    m = re.search(r"<noscript>(.*?)</noscript>", html, re.DOTALL)
    assert m, (
        "[P2-LANDING-HEAD-CLIENT] `index.html` no tiene `<noscript>`. Sin él, un "
        "cliente que no ejecuta JS recibe un `<div id=\"root\">` vacío y un splash "
        "sin un solo enlace."
    )
    enlaces = set(re.findall(r'href="([^"]+)"', m.group(1)))
    rutas = _rutas_declaradas_en_app()
    rotos = [h for h in enlaces if h.startswith("/") and h.rstrip("/") not in rutas and h != "/"]
    assert not rotos, (
        f"[P2-LANDING-HEAD-CLIENT] El `<noscript>` enlaza rutas inexistentes: {rotos}"
    )
    assert len(enlaces) >= 4, (
        "[P2-LANDING-HEAD-CLIENT] El `<noscript>` debe ofrecer una salida real "
        f"(al menos 4 enlaces), no un cartel. Tiene {len(enlaces)}."
    )


def _rutas_servidas_por_el_apex() -> set:
    """Las rutas que nginx sirve desde el sitio estático, leídas de su PROPIA config.

    [P1-ABOUT-UNA-SOLA-COPIA · 2026-09-07] El sitemap es de `bioboros.com`, y ahí una ruta puede
    estar servida por el sitio estático en vez de por la SPA — `mealfit.conf:104` tiene la lista
    explícita. Antes coincidían todas con rutas de `App.jsx`, así que la diferencia no se notaba;
    `/about` es la primera que existe SOLO en el apex, y el guard la llamó «404 ofrecido a Google»
    cuando en realidad se sirve perfectamente.

    Se lee de `mealfit.conf` y no se copia aquí a mano: una segunda lista es exactamente cómo se
    empieza a no saber cuál manda (la lección de `canonicalize_country` y de la 4ª tabla de dietas).
    """
    conf = (Path(__file__).resolve().parents[1] / "infra" / "nginx" / "mealfit.conf")
    if not conf.exists():
        return set()
    m = re.search(r"location\s+~\s+\^/\(([a-z0-9|\-]+)\)\$\s*\{[^}]*?bioboros-v2",
                  conf.read_text(encoding="utf-8", errors="ignore"), re.S)
    return {f"/{r}" for r in m.group(1).split("|")} if m else set()


def test_toda_ruta_del_sitemap_existe_en_la_app():
    """El cruce que nunca existió: sitemap ↔ (árbol de rutas ∪ estáticas del apex)."""
    sitemap = _read(_SITEMAP)
    rutas = _rutas_declaradas_en_app()
    apex = _rutas_servidas_por_el_apex()
    assert apex, ("no se pudo leer la lista de estáticas de mealfit.conf: sin ella este test "
                  "aprobaría por no encontrar nada, que es peor que fallar")
    fantasmas = []
    for loc in re.findall(r"<loc>https://bioboros\.com([^<]*)</loc>", sitemap):
        ruta = loc or "/"
        if ruta.startswith("/novedades/"):
            continue  # dinámica: la cubre el test de arriba
        if ruta not in rutas and ruta not in apex:
            fantasmas.append(ruta)
    assert not fantasmas, (
        f"[P2-LANDING-SITEMAP-SSOT] El sitemap anuncia rutas que NADIE sirve —ni la app ni el "
        f"sitio estático del apex—: {fantasmas}. Cada una es un 404 ofrecido a Google en bandeja."
    )
