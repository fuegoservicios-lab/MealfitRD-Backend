"""[P2-SEO-HREFLANG · 2026-08-21] Los datos estructurados declaraban un área de servicio que dejó
de ser cierta.

`index.html` afirmaba en su JSON-LD que Bioboros sirve a **un** país:

    "areaServed": { "@type": "Country", "name": "República Dominicana" }

…y lo repetía en la descripción del producto («…para República Dominicana») y en el `og:image:alt`.
Desde el flip del 2026-08-18 el selector del propio producto ofrece SEIS países. Una afirmación
factual en datos estructurados —lo que los buscadores leen como declaración de la empresa— que el
producto contradice en su primera pantalla.

QUÉ SE CORRIGE Y QUÉ NO, porque la línea importa. Se corrige lo que es **falso**: el área de
servicio y las dos frases que la repetían. Este P-fix no tocó la moneda de la oferta, ni el copy de
marketing, ni añadió una promesa nueva — esas son decisiones del dueño que la auditoría lista aparte
como P1-27 y P1-28, y meterlas aquí habría sido colar una decisión de producto dentro de un arreglo
técnico. [P1-PLAN-LOTE-793 · 2026-09-28] La de la moneda (P1-28) ya la tomó el dueño (G95): la
oferta va en USD, como el apex; el test que fijaba la moneda vieja se invirtió y vive en su propia
sección, más abajo.

POR QUÉ NO SE AÑADE `hreflang`, que es la otra mitad del gap. `hreflang` declara que existe una URL
ALTERNATIVA por idioma o región. Aquí no existe: la app es una SPA servida desde una sola URL y el
idioma lo elige el usuario en Configuración, guardado en `localStorage`. Declarar alternativas que
resuelven todas a la misma página no mejora nada y le dice al buscador algo que no es verdad — el
mismo error que este P-fix está corrigiendo, cometido de nuevo por el otro lado. Cuando existan
rutas por idioma, el `hreflang` viene con ellas.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_INDEX = (Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html")


@pytest.fixture(scope="module")
def html() -> str:
    if not _INDEX.is_file():
        pytest.skip("index.html no está en este árbol")
    return _INDEX.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def bloques_ld(html) -> list:
    return [json.loads(m.group(1)) for m in
            re.finditer(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]


def _plano(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ── El JSON-LD sigue siendo JSON ────────────────────────────────────────────────────────────────

def test_los_bloques_de_datos_estructurados_parsean(bloques_ld):
    """El primero de todos, y no es teórico: al editar este bloque metí un comentario HTML DENTRO
    del `<script type="application/ld+json">`, que lo habría dejado sin parsear — y un JSON-LD roto
    no da error visible en la página, simplemente deja de existir para el buscador."""
    assert len(bloques_ld) >= 3, f"aparecieron/desaparecieron bloques JSON-LD: {len(bloques_ld)}"


# ── El área de servicio ─────────────────────────────────────────────────────────────────────────

def test_el_area_de_servicio_no_declara_un_solo_pais(bloques_ld):
    org = next((b for b in bloques_ld if b.get("@type") == "Organization"), None)
    assert org, "desapareció el bloque Organization"
    area = org.get("areaServed")
    assert isinstance(area, list), (
        f"`areaServed` volvió a declarar un país único: {area!r}. El selector del producto ofrece "
        f"seis desde el flip"
    )


def test_estan_los_seis_paises_que_el_producto_ofrece(bloques_ld):
    """La lista no se inventa: son los seis del selector. Si el producto añade un séptimo país y
    aquí no aparece, este test lo dice."""
    org = next(b for b in bloques_ld if b.get("@type") == "Organization")
    nombres = {c.get("name") for c in org["areaServed"]}
    esperados = {"República Dominicana", "España", "México", "Colombia", "Puerto Rico",
                 "Estados Unidos"}
    assert nombres == esperados, f"el área de servicio no coincide con el selector: {nombres}"


def test_las_descripciones_no_acotan_el_servicio_a_un_pais(html):
    """La misma afirmación se repetía en la descripción del producto y en el texto alternativo de
    la imagen social — los tres sitios donde un buscador la lee."""
    for patron in (r'"description":\s*"[^"]*para República Dominicana',
                   r'og:image:alt"\s+content="[^"]*para República Dominicana'):
        assert not re.search(patron, html), (
            f"sigue habiendo una descripción que acota el servicio a un país: {patron}"
        )


# ── Lo que decidió el dueño aparte (P1-28 → G95, P1-PLAN-LOTE-793) ─────────────────────────────

def test_la_moneda_del_offer_es_la_del_apex_usd(html):
    """[P1-PLAN-LOTE-793 · 2026-09-28] Antes este test fijaba DOP: P1-28 era una decisión del dueño
    pendiente y colarla dentro de un arreglo técnico habría sido decidir por él. El dueño ya decidió
    (G95, 28-sep): la oferta va en USD, la misma moneda que declara el apex. Invertido, no borrado:
    el motivo de no tocarla aquí sigue valiendo para el copy de marketing, que este P-fix no toca."""
    assert '"priceCurrency": "USD"' in html, (
        "la moneda del Offer ya no es USD: el apex declara la misma oferta en USD (G95)"
    )
    assert '"priceCurrency": "DOP"' not in html, (
        "volvió el DOP al Offer: dos datos estructurados del mismo producto con dos monedas"
    )


# ── Lo que NO se toca ───────────────────────────────────────────────────────────────────────────

def test_no_se_anadio_hreflang_sin_urls_por_idioma(html):
    """`hreflang` declara que existe una URL ALTERNATIVA por idioma. Aquí no existe: SPA de una
    sola URL, idioma elegido en Configuración y guardado en localStorage. Declarar alternativas que
    resuelven todas a la misma página le dice al buscador algo que no es verdad — el mismo error
    que este P-fix corrige, cometido por el otro lado."""
    alternates = re.findall(r'rel="alternate"[^>]*hreflang="([^"]+)"', html)
    if not alternates:
        return
    # Si algún día aparecen, que sea porque hay rutas distintas de verdad.
    urls = set(re.findall(r'rel="alternate"[^>]*href="([^"]+)"', html))
    assert len(urls) == len(alternates) and len(urls) > 1, (
        f"hay {len(alternates)} hreflang apuntando a {len(urls)} URL(s) distintas: alternativas "
        f"que resuelven a la misma página no declaran nada cierto"
    )


def test_el_lang_estatico_sigue_siendo_es_DO(html):
    """`lang="es-DO"` es el fallback correcto y load-bearing: el bootstrap de idioma sólo reescribe
    el atributo cuando el usuario eligió OTRO, así que tocarlo aquí cambiaría el idioma con el que
    se lee la página antes de que React monte."""
    assert re.search(r'<html\s+lang="es-DO"', html)
