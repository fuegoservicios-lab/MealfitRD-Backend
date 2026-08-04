"""[P1-PAPER-SURFACE-SSOT · 2026-08-01] La lista de rutas con superficie
`paper` vive en TRES sitios que no pueden importarse entre sí:

    1. frontend/src/utils/paperSurface.js   — SSOT de JS
    2. frontend/index.html                  — boot script inline (no puede importar)
    3. frontend/src/utils/marketingRoutes.js — la lista del HEADER, que
       gobierna un alcance distinto (19 patrones vs las de papel)

[P1-PAPER-SURFACE-EXTEND · 2026-08-02] Las listas (1)/(2) y (3) YA NO coinciden:
la superficie papel creció a 10 rutas + 1 prefijo dinámico (`/novedades/<slug>`)
y `MARKETING_ROUTES` se quedó en 6. Eso no es drift, es el motivo por el que
existen dos listas — y desde hoy el test que las comparaba comprueba la relación
correcta (marketing ⊆ papel) en vez de la igualdad.

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

# [P1-PAPER-SURFACE-EXTEND · 2026-08-02] La superficie papel dejó de coincidir
# con `MARKETING_ROUTES`, que es exactamente el escenario que la separación de
# listas anticipaba (ver docstring de `paperSurface.js`). Cuatro rutas más:
#   /research      — comparte `HowItWorksPage.module.css` con las 3 páginas de
#                    detalle que ya eran papel; era la única consumidora fuera.
#   /novedades     — el landing la enlaza desde «05 / REGISTRO» (tabla B/N);
#                    cierra la deuda con fecha del spec §10.3.
#   /supermercado  — enlazada desde el footer de todas las rutas de papel.
#   /about         — ídem.
# `MARKETING_ROUTES` NO se tocó: gobierna el header de 19 patrones de ruta y
# repuntarla dejaría 13 rutas sin nav ni CTA.
# [P1-PAPER-LEGAL · 2026-08-02] +9 legales. El footer de las 10 rutas de papel
# enlaza a TODAS ellas: el pliego terminaba en una tarjeta glass sobre un
# degradado azulado. `/cookies` entra aunque NO renderice `LegalPages` (es un
# `<Navigate to="/privacy">`, P3-COOKIES-MERGE, y por eso tampoco está en
# `LEGAL_PATHS`): el boot script corre ANTES de que React resuelva el redirect,
# así que sin ella una carga en frío de /cookies pinta un frame oscuro.
_EXPECTED_PAPER = [
    "/", "/precios", "/como-funciona", "/funciones", "/precision", "/motor",
    "/research", "/novedades", "/supermercado", "/about",
    "/privacy", "/terms", "/cookies", "/medical", "/data-protection",
    "/ai-policy", "/refunds", "/acceptable-use", "/responsible-disclosure",
]
# `/novedades/:slug` es DINÁMICA: ninguna lista exacta puede cubrirla sin
# drift contra `data/news.js`. Va por prefijo, igual que `newsRoutes.js` ya
# hacía para el header.
_EXPECTED_PAPER_PREFIXES = ["/novedades/"]
_EXPECTED_MARKETING = ["/", "/precios", "/como-funciona", "/funciones", "/precision", "/motor"]


def _routes_from_array(text: str, const_name: str) -> list[str]:
    """Extrae los literales de un `export const NAME = ['/a', '/b'];`."""
    m = re.search(
        const_name + r"\s*=\s*\[(?P<body>[^\]]*)\]",
        text,
        re.DOTALL,
    )
    assert m is not None, f"No se encontró el array `{const_name}`."
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group("body"))


def test_paper_surface_module_lists_every_paper_route():
    assert _PAPER_JS.exists(), (
        "P1-PAPER-SURFACE-SSOT: falta `frontend/src/utils/paperSurface.js`."
    )
    text = _PAPER_JS.read_text(encoding="utf-8")
    assert _routes_from_array(text, "PAPER_SURFACE_ROUTES") == _EXPECTED_PAPER
    assert "export const isPaperSurface" in text, (
        "P1-PAPER-SURFACE-SSOT: `paperSurface.js` debe exportar `isPaperSurface`."
    )


def test_paper_surface_module_lists_the_dynamic_prefixes():
    """[P1-PAPER-SURFACE-EXTEND · 2026-08-02] `/novedades/:slug` no cabe en la
    lista exacta. Sin este caso, alguien podría añadir la ruta estática
    `/novedades` y creer que el artículo queda cubierto: no lo estaría, y el
    síntoma sería un parpadeo a oscuro SOLO al abrir un anuncio — el borde que
    nadie prueba a mano."""
    text = _PAPER_JS.read_text(encoding="utf-8")
    assert _routes_from_array(text, "PAPER_SURFACE_PREFIXES") == _EXPECTED_PAPER_PREFIXES
    assert "startsWith" in text, (
        "P1-PAPER-SURFACE-SSOT: `isPaperSurface` debe consultar los prefijos "
        "(startsWith), no solo el `includes` de la lista exacta."
    )


def test_boot_script_copy_matches_the_ssot():
    """El boot script de index.html no puede importar el módulo, así que
    lleva una copia literal. Si divergen, la carga en frío parpadea."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert _routes_from_array(html, r"var\s+PAPER") == _EXPECTED_PAPER, (
        "P1-PAPER-SURFACE-SSOT: la copia del boot script en index.html "
        "divergió de paperSurface.js. Las dos tienen que viajar juntas."
    )
    assert _routes_from_array(html, r"var\s+PAPER_PFX") == _EXPECTED_PAPER_PREFIXES, (
        "P1-PAPER-SURFACE-SSOT: la copia de PREFIJOS del boot script divergió. "
        "Una carga directa de /novedades/<slug> parpadearía a oscuro."
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
    """El HEADER conserva su alcance de 19 rutas: `isLandingLike` — el
    predicado que decide nav completa + CTA sticky — se computa desde
    `isMarketingRoute`, NO desde `isPaperSurface`.

    [fix1 · 2026-08-02] Esta aserción era `'isPaperSurface' not in header`:
    prohibía una CADENA en vez de comprobar el INVARIANTE, y se volvió falso
    positivo en cuanto la Task 7 empezó a gatear tres ORNAMENTOS de papel
    desde el mismo archivo (el cajetín `ES-DO / V1`, la numeración del menú
    móvil y el glifo de 2 trazos del toggle). Ese uso es legítimo: son
    elementos que solo existen en el vocabulario papel, y las 13 rutas
    no-marketing que también montan este header (10 legales + 2 de novedades
    + /supermercado) siguen en su propio claro/oscuro — heredar el cajetín
    `ES-DO / V1` en la política de cookies sería el bug, no la defensa.

    Lo que de verdad no puede pasar es que `isLandingLike` se repunte a la
    lista de 6: eso deja esas 13 rutas SIN nav y SIN CTA. Así que el guard
    ahora parsea la asignación de `isLandingLike` y comprueba su fuente.
    Un uso ornamental de `isPaperSurface` pasa; repuntar el chrome falla.
    """
    header = _HEADER_JSX.read_text(encoding="utf-8")
    assert "from '../../utils/marketingRoutes'" in header, (
        "P1-PAPER-SURFACE-SSOT: Header.jsx debe seguir importando "
        "`isMarketingRoute` desde marketingRoutes.js (19 rutas)."
    )

    m = re.search(
        r"const\s+isLandingLike\s*=\s*(?P<expr>.*?);",
        header,
        re.DOTALL,
    )
    assert m is not None, (
        "P1-PAPER-SURFACE-SSOT: no se encontró `const isLandingLike = …;` en "
        "Header.jsx. Si lo renombraste, actualiza este guard — es el predicado "
        "que decide qué rutas reciben nav completa + CTA sticky."
    )
    expr = m.group("expr")
    assert "isMarketingRoute(" in expr, (
        "P1-PAPER-SURFACE-SSOT: `isLandingLike` debe computarse desde "
        f"`isMarketingRoute(...)`. Expresión encontrada: {expr!r}"
    )
    assert "isPaperSurface" not in expr and "isPaper" not in expr, (
        "P1-PAPER-SURFACE-SSOT: `isLandingLike` NO puede depender de la "
        "superficie papel — el header cubre 19 rutas (marketing + legales + "
        "novedades + /supermercado) y el papel solo 6. Repuntarlo dejaría 13 "
        f"rutas sin nav y sin CTA. Expresión encontrada: {expr!r}"
    )


def test_marketing_routes_did_not_move_when_the_paper_surface_grew():
    """[P1-PAPER-SURFACE-EXTEND · 2026-08-02] Este test decía «hoy ambas listas
    coinciden» y se anotó a sí mismo: *«cuando dejen de coincidir (p.ej. si
    /supermercado pasa a papel), hay que RELAJARLO a propósito, no borrarlo»*.
    Ese día llegó — /supermercado, /research, /novedades y /about pasaron a
    papel — así que la aserción cambia de forma sin perder su trabajo.

    Lo que comprueba ahora es lo que de verdad importa: que mover la superficie
    NO movió el header. `MARKETING_ROUTES` sigue siendo las 6 de siempre y sigue
    siendo un SUBCONJUNTO de la superficie papel (si dejara de serlo, habría una
    ruta con nav de landing y tema de app, o al revés)."""
    text = _MARKETING_JS.read_text(encoding="utf-8")
    marketing = _routes_from_array(text, "MARKETING_ROUTES")
    assert marketing == _EXPECTED_MARKETING, (
        "P1-PAPER-SURFACE-SSOT: `MARKETING_ROUTES` cambió. Esa lista gobierna el "
        "HEADER (19 patrones vía isLandingLike), no el tema: repuntarla deja 13 "
        "rutas sin nav y sin CTA. Si querías mover la superficie papel, el "
        "fichero es `paperSurface.js`."
    )
    assert set(marketing).issubset(set(_EXPECTED_PAPER)), (
        "P1-PAPER-SURFACE-SSOT: una ruta de marketing quedó fuera de la "
        "superficie papel. Marketing ⊆ papel es invariante desde el flip."
    )


# ---------------------------------------------------------------------------
# 2. Los lectores que comparan 'dark' en duro
# ---------------------------------------------------------------------------
_USE_THEME_COLOR = _FRONTEND / "src" / "components" / "common" / "useThemeColor.js"
_MANIFEST = _FRONTEND / "public" / "manifest.json"

_PAPER_HEX = "#FBFBFA"
_OLD_BRAND_INDIGO = "#4F46E5"


def test_use_theme_color_asks_the_theme_not_the_route():
    """[fix1 · 2026-08-01] `isPaperSurface(path)` pregunta si la RUTA es
    elegible para papel, no si el papel está ACTIVO. `PublicThemeLock`
    (App.jsx) fuerza `data-theme="dark"` en esas mismas 6 rutas mientras
    nadie emite `'paper'` — con `isPaperSurface` la rama disparaba HOY,
    pisando #0B1120 con #FBFBFA antes de que exista ningún flip. La rama
    correcta pregunta por el atributo del DOM, simétrica a `isDarkActive`."""
    text = _USE_THEME_COLOR.read_text(encoding="utf-8")
    assert "isPaperActive" in text, (
        "P1-PAPER-THEME: useThemeColor.js debe consultar `isPaperActive` "
        "(lee data-theme del DOM), no `isPaperSurface` (lee la ruta)."
    )
    assert "isPaperSurface" not in text, (
        "P1-PAPER-THEME: useThemeColor.js NO debe volver a importar "
        "`isPaperSurface` — esa función responde 'ruta elegible', no "
        "'papel activo', y PublicThemeLock fuerza dark en las 6 rutas "
        "elegibles mientras 'paper' no se emite."
    )
    assert _PAPER_HEX in text, (
        f"P1-PAPER-THEME: useThemeColor.js debe emitir {_PAPER_HEX} en la rama papel."
    )
    assert text.index("isPaperActive()") < text.index("else if (dark)"), (
        "P1-PAPER-THEME: la rama papel debe evaluarse ANTES que `else if (dark)` "
        "— si el orden se invierte, el tema oscuro real vuelve a ganarle al papel."
    )


def test_theme_exposes_a_paper_reader_without_touching_valid_prefs():
    """`isPaperActive` es simétrica de `isDarkActive`: ambas leen
    `data-theme` del DOM. `'paper'` no es una preferencia persistida en
    localStorage, así que NO debe colarse en `VALID_PREFS`."""
    theme = (_FRONTEND / "src" / "utils" / "theme.js").read_text(encoding="utf-8")
    assert "export function isPaperActive" in theme, (
        "P1-PAPER-THEME: theme.js debe exportar `isPaperActive`, simétrica de "
        "`isDarkActive`, para que los lectores de tema pregunten al DOM y no a la ruta."
    )
    assert "'paper'" not in theme.split("VALID_PREFS")[1].split("]")[0], (
        "P1-PAPER-THEME: 'paper' NO es una preferencia persistida en localStorage "
        "— no debe añadirse a VALID_PREFS. Eso lo gobierna un flip aparte (Task 5)."
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


def test_theme_call_sites_write_paper_not_dark():
    """Los 3 sitios tienen que viajar juntos. Si el boot script y el SPA
    discrepan, la carga en frío parpadea a negro un frame en las 6 rutas."""
    app = _APP_JSX.read_text(encoding="utf-8")
    html = _INDEX_HTML.read_text(encoding="utf-8")

    paper_writes = re.findall(r"setAttribute\(\s*['\"]data-theme['\"]\s*,\s*['\"]paper['\"]", app)
    assert len(paper_writes) == 2, (
        f"P1-PAPER-THEME: App.jsx debe escribir 'paper' en 2 sitios "
        f"(PublicThemeLock y el useEffect de arranque); encontrados {len(paper_writes)}."
    )
    # [P1-PAPER-SURFACE-EXTEND · 2026-08-02] La aserción anterior exigía el
    # literal `PAPER.indexOf(location.pathname) !== -1) theme = 'paper'`. Ese
    # `indexOf` no puede ver `/novedades/<slug>`, así que el boot script pasó a
    # un predicado único (`paperPath`) consultado por el camino normal Y por el
    # `catch` — que es además lo que impide que los dos vuelvan a divergir.
    assert re.search(r"paperPath\(location\.pathname\)\)\s*theme\s*=\s*['\"]paper['\"]", html), (
        "P1-PAPER-THEME: el boot script de index.html debe fijar 'paper' para "
        "las rutas de la superficie. Si se queda en 'dark', la carga en frío "
        "parpadea antes de que el SPA corrija."
    )
    assert re.search(r"paperPath\(location\.pathname\)\s*\?\s*['\"]paper['\"]\s*:\s*['\"]dark['\"]", html), (
        "P1-PAPER-THEME: el `catch` del boot script debe consultar el MISMO "
        "predicado. Si se queda con su propia copia de la condición, un fallo de "
        "localStorage (Safari privado, ITP, webviews) pinta el splash en oscuro "
        "sobre una ruta de papel."
    )
