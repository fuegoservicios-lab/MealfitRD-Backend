"""[P2-WORDMARK-BIOBOROS · 2026-07-30 · ampliado 2026-07-31] El wordmark tiene UN dueño.

Historia corta: la marca estaba escrita a mano en 12 sitios. El rebrand a "Bioboros"
buscaba los tres fragmentos `Mealfit<span>R</span><span>D</span>` ADYACENTES, así que
no alcanzó los archivos donde estaban en líneas separadas. Cayeron dos:

  - `Logo.jsx` (el compartido) → el usuario vio "Mealfit" en la app ya rebrandeada.
  - `Plan.jsx`, la pantalla "Diseñando tu plan" → sobrevivió hasta el 31 de julio
    mostrando "Bioboros" + una "R" indigo con puntito + una "D" rosa. El sufijo "RD"
    venía de "MealfitRD", donde era el país y separarlo SIGNIFICABA algo; sobre
    "Bioboros" quedaba un sufijo sin referente, y encima bicolor — el recurso que el
    owner ya había descartado dos veces (ver el comentario de `Wordmark.jsx`).

Los dos fallaron por la MISMA razón, con dos meses de diferencia. Por eso el guard no
persigue la cadena "RD": persigue el patrón estructural que la produce.

⚠️ [ampliado 2026-07-31, tercera vez] Y aun así se le escapó una: el **splash de
`index.html`** — lo PRIMERO que ve todo usuario al arrancar — seguía bicolor. Falló los
tres anclajes del regex a la vez:

  1. sólo se escaneaba `frontend/src/**/*.jsx`, e `index.html` no es ninguna de las dos;
  2. el color venía de una CLASE (`.splash-r { color: … }`), no de un `style` inline;
  3. la pieza no era una letra mayúscula suelta, sino `b` minúscula + `oros`.

El patrón "letra suelta con color inline" se había generalizado a partir de los dos
casos JSX conocidos, y no cubría el que estaba escrito de otra forma. La lección se
repite: la exclusión que hace tratable un barrido es exactamente donde se queda ciego.

El splash no puede usar `<Wordmark/>` (corre antes de que React monte), así que es la
única copia legítima de la marca fuera del SSOT. Se vigila aparte y por ESTRUCTURA: la
marca va en una pieza, sin markup dentro — así da igual el color, el número de letras o
si son mayúsculas.
"""
import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"
_WORDMARK = _FRONT / "components" / "common" / "Wordmark.jsx"
_INDEX_HTML = _FRONT.parent / "index.html"


def _jsx_files():
    return sorted(p for p in _FRONT.rglob("*.jsx"))


# ------------------------------------------------- el patrón que produjo el bug

# Un <span> que fija un color y cuyo ÚNICO contenido es una letra suelta. Es la firma
# del "acento por letra": no depende de que la letra sea R o D, ni de que esté pegada
# a la marca — dos anclajes que caducarían al primer cambio de branding.
_LETRA_ACENTUADA = re.compile(
    r"<span[^>]*\bcolor\s*:\s*['\"]?#[0-9A-Fa-f]{3,8}[^>]*>\s*([A-ZÁÉÍÓÚÑ])\s*</span>",
    re.DOTALL,
)


@pytest.mark.parametrize("ruta", _jsx_files(), ids=lambda p: p.name)
def test_ninguna_letra_suelta_coloreada_como_acento_de_marca(ruta: Path):
    """Ninguna letra individual lleva color propio.

    `Wordmark.jsx` documenta que la marca es MONOCROMO por decisión de producto, tomada
    tras descartar en vivo dos versiones con acento (bicolor indigo+rosa, y las tres
    "o" en verde). Un acento por letra reintroduce justo eso, y además se salta el SSOT.
    """
    src = ruta.read_text(encoding="utf-8")
    hits = _LETRA_ACENTUADA.findall(src)
    assert not hits, (
        f"P2-WORDMARK-BIOBOROS regresión en {ruta.name}: letra(s) sueltas con color "
        f"propio {hits!r}. El wordmark es monocromo por decisión de producto y se "
        f"renderiza SOLO vía `<Wordmark/>`. Si de verdad quieres un acento, cámbialo "
        f"en Wordmark.jsx y habla la decisión — es la tercera versión y las dos "
        f"anteriores se descartaron por esto."
    )


# ------------------------------------------------------- el SSOT sigue siendo uno

def test_wordmark_es_monocromo():
    """El propio componente no puede llevar color hardcodeado."""
    src = _WORDMARK.read_text(encoding="utf-8")
    cuerpo = src[src.index("export const Wordmark"):]
    assert not re.search(r"color\s*:\s*['\"]?#", cuerpo), (
        "P2-WORDMARK-BIOBOROS regresión: `Wordmark.jsx` fija un color hex. Debe heredar "
        "la tinta del contexto (`--text-main`) para funcionar igual en claro y oscuro."
    )
    assert "Bioboros" in cuerpo, "el wordmark dejó de renderizar la marca"


def test_la_pantalla_de_carga_del_plan_usa_el_componente():
    """`Plan.jsx` fue el último sitio con la marca a mano; que no vuelva a escribirla.

    Anclado al render, no a una ventana de bytes: exige el import Y el uso.
    """
    plan = (_FRONT / "pages" / "Plan.jsx").read_text(encoding="utf-8")
    assert re.search(r"^import\s+Wordmark\s+from\s+['\"].*Wordmark['\"]", plan, re.M), (
        "P2-WORDMARK-BIOBOROS regresión: Plan.jsx ya no importa `Wordmark`."
    )
    assert "<Wordmark />" in plan or "<Wordmark/>" in plan, (
        "P2-WORDMARK-BIOBOROS regresión: Plan.jsx importa `Wordmark` pero no lo renderiza "
        "— la pantalla 'Diseñando tu plan' volvería a dibujar la marca a mano."
    )


# ------------------------------------------------- el splash, la copia legítima

_SPLASH_BRAND = re.compile(r'<div class="splash-brand">(.*?)</div>', re.DOTALL)


def _splash_brand_inner() -> str:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    m = _SPLASH_BRAND.search(html)
    # Sanity del vehículo: si el div se renombra, los asserts de abajo pasarían
    # en vacío y este guard dejaría de proteger sin ponerse rojo.
    assert m, (
        "P2-WORDMARK-BIOBOROS: no encuentro `<div class=\"splash-brand\">` en index.html. "
        "Si lo renombraste, actualiza este test — mientras tanto el splash está sin vigilar."
    )
    return m.group(1).strip()


def test_el_splash_escribe_la_marca_en_UNA_pieza():
    """Sin markup dentro: ni spans, ni letras sueltas, ni bicolor.

    Es una comprobación ESTRUCTURAL a propósito. La versión anterior del guard
    buscaba "letra mayúscula suelta con color inline" y no vio este splash, que
    partía `b` + `oros` coloreando por clase CSS. Exigir que no haya markup no
    depende del color, ni del número de letras, ni de si son mayúsculas.
    """
    inner = _splash_brand_inner()
    assert "<" not in inner, (
        f"P2-WORDMARK-BIOBOROS regresión: el splash de index.html volvió a partir la "
        f"marca en trozos ({inner!r}). Va en una pieza: `<div class=\"splash-brand\">"
        f"Bioboros</div>`. Este splash corre antes de React y no puede usar "
        f"`<Wordmark/>`, así que es la única copia a mano — y por eso es la que se "
        f"desincroniza."
    )
    assert inner == "Bioboros", (
        f"P2-WORDMARK-BIOBOROS: el splash muestra {inner!r} en vez de 'Bioboros'."
    )


def test_ninguna_regla_css_pinta_un_trozo_del_wordmark_del_splash():
    """El color por CLASE fue el agujero: `.splash-r`/`.splash-d` no eran inline.

    Una regla que apunta a un descendiente de `.splash-brand` sólo tiene sentido
    si alguien volvió a meter markup dentro. El test de arriba ya lo cazaría,
    pero una regla huérfana con nombre de letra es una invitación a rehacerlo.
    """
    html = _INDEX_HTML.read_text(encoding="utf-8")
    hits = re.findall(r"\.splash-brand\s+\.[\w-]+", html)
    assert not hits, (
        f"P2-WORDMARK-BIOBOROS regresión: reglas CSS sobre trozos del wordmark del "
        f"splash {hits!r}. El wordmark es monocromo por decisión de producto (tercera "
        f"versión; las dos con acento se descartaron en vivo)."
    )


# ------------------------------------------- la imagen de compartir (og:image)

_GENERADOR = _FRONT.parent / "scripts" / "generate-og-image.mjs"


def test_el_og_image_existe_y_lo_produce_el_generador():
    """El `og:image` es un RASTER: ningún grep ni guard de copy lo alcanza.

    La v3 siguió diciendo "MealfitRD" semanas después del rebrand — con el
    comentario de al lado ya rebrandeado a «wordmark "Bioboros"». Se descubrió
    porque el owner compartió el enlace desde el móvil y lo vio.

    No se puede leer el JPEG desde aquí, pero sí cerrar la deriva REALISTA: el
    nombre del fichero se bumpea en cada cambio (iOS y WhatsApp cachean el
    og:image por URL e ignoran el `?v=`), y ese bump vive en DOS sitios —
    el meta y el generador. Si sólo se toca uno, la preview se queda vieja o
    da 404, que es exactamente el fallo silencioso que este guard evita.
    """
    html = _INDEX_HTML.read_text(encoding="utf-8")
    og = re.findall(r'<meta property="og:image" content="[^"]*/([\w.-]+\.jpe?g)"', html)
    tw = re.findall(r'<meta name="twitter:image" content="[^"]*/([\w.-]+\.jpe?g)"', html)
    # Sanity del vehículo antes de afirmar nada.
    assert len(og) == 1 and len(tw) == 1, (
        f"P2-WORDMARK-BIOBOROS: esperaba un og:image y un twitter:image en index.html, "
        f"encontré og={og!r} twitter={tw!r}. Si cambió el markup, actualiza este test."
    )
    assert og == tw, (
        f"P2-WORDMARK-BIOBOROS: og:image ({og[0]}) y twitter:image ({tw[0]}) apuntan a "
        f"imágenes distintas — X mostraría una y WhatsApp otra."
    )

    fichero = _FRONT.parent / "public" / og[0]
    assert fichero.exists(), (
        f"P2-WORDMARK-BIOBOROS: index.html anuncia /{og[0]} pero no está en public/. "
        f"La preview al compartir daría 404. Corre `node scripts/generate-og-image.mjs`."
    )

    gen = _GENERADOR.read_text(encoding="utf-8")
    assert og[0] in gen, (
        f"P2-WORDMARK-BIOBOROS: index.html usa /{og[0]} pero el generador "
        f"(`scripts/generate-og-image.mjs`) escribe otro nombre. El bump del sufijo "
        f"vive en dos sitios y hay que moverlo en los dos: si no, se publica la imagen "
        f"vieja bajo el nombre nuevo — o al revés."
    )
    assert re.search(r"const\s+MARCA\s*=\s*'Bioboros'", gen), (
        "P2-WORDMARK-BIOBOROS: el generador del og:image ya no escribe 'Bioboros'. "
        "Es la única fuente de la marca en la imagen de compartir."
    )
