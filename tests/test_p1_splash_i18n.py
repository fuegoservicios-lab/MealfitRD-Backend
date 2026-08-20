"""[P1-SPLASH-I18N · 2026-08-20] La pantalla de carga seguia en espanol.

Reportado con captura: la app en ingles y el splash diciendo «Nutricion personalizada
con IA». Era la ULTIMA superficie sin traducir y la mas facil de no ver, porque dura un
segundo -- pero es literalmente lo primero que se lee al abrir la app.

POR QUE NO BASTABA `t()`. El splash vive en `index.html` y se pinta ANTES de que React
y el motor de i18n existan. No es que nadie lo hubiera envuelto: es que ahi no hay `t`.

EL PATRON YA ESTABA EN CASA. `index.html` tiene dos boot scripts sincronos --el del
idioma (`<html lang>`) y el del tema-- por la misma razon: hay cosas que deben estar
puestas antes del primer paint. Este es el tercero, y reutiliza la lectura de
`localStorage('mealfit_locale')` que el primero ya hacia (se publica en
`window.__mfLocale` en vez de leer localStorage dos veces).

DOS DECISIONES QUE MERECEN QUEDAR ESCRITAS

1. EL TEXTO NO SALE DE LOS CATALOGOS. Nadie llama a `t()` con estas cadenas --no
   pueden, el motor aun no existe--, asi que meterlas en los `.json` las volveria
   HUERFANAS y `npm run i18n:check` empezaria a avisar de algo correcto. Un guard que
   avisa de lo correcto se acaba ignorando. El SSOT es el mapa inline, y este test lo
   ancla contra el HTML estatico para que no puedan divergir.

2. EL SCRIPT VA DESPUES DE LOS NODOS QUE LEE, no en un `DOMContentLoaded`. Corre
   durante el parseo, antes del primer paint, asi que no hay parpadeo de espanol. La
   primera version lo puso ENTRE el tagline y los puntos: `querySelector('.splash-dots')`
   devolvia `null` y el `aria-label` se habria quedado en espanol EN SILENCIO --el
   caso que un lector de pantalla si nota y un ojo no--. Por eso hay un test de ORDEN.

tooltip-anchor: P1-SPLASH-I18N
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_INDEX = _ROOT / "frontend" / "index.html"

_LOCALES = ["es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"]


def _html() -> str:
    return io.open(_INDEX, encoding="utf-8").read()


def _mapa(nombre: str) -> dict:
    """Extrae un mapa `var <nombre> = { 'x': 'y', ... };` del script inline."""
    m = re.search(rf"var {nombre} = \{{(.*?)\}};", _html(), re.S)
    assert m, f"no encuentro el mapa {nombre} en index.html"
    return dict(re.findall(r"'([^']+)':\s*'([^']*)'", m.group(1)))


# ───────────────────────── el mapa cubre los 5 idiomas ─────────────────────────

@pytest.mark.parametrize("nombre", ["TAGLINE", "CARGANDO"])
def test_el_mapa_cubre_los_cinco_idiomas(nombre):
    """Un idioma que falte cae al espanol EN SILENCIO. La lista es la misma que la del
    boot de `lang` de arriba, que ya esta anclada por test_p1_i18n_dashboard."""
    mapa = _mapa(nombre)
    faltan = [loc for loc in _LOCALES if loc not in mapa]
    assert not faltan, f"{nombre} no cubre {faltan}"
    vacias = [k for k, v in mapa.items() if not v.strip()]
    assert not vacias, f"{nombre} tiene entradas vacias: {vacias}"


def test_el_espanol_del_mapa_es_IDENTICO_al_del_marcado():
    """El fallback y el mapa son dos copias de la misma frase. Si divergen, el usuario
    en es-DO ve una y el de en-US traduce la otra -- y nadie se entera, porque en
    es-DO el script no llega a tocar el nodo."""
    html = _html()
    m = re.search(r'<div class="splash-tagline">([^<]+)</div>', html)
    assert m, "no encuentro el tagline estatico"
    assert _mapa("TAGLINE")["es-DO"] == m.group(1).strip()

    m2 = re.search(r'class="splash-dots" aria-label="([^"]+)"', html)
    assert m2, "no encuentro el aria-label estatico"
    assert _mapa("CARGANDO")["es-DO"] == m2.group(1).strip()


def test_el_ingles_no_se_quedo_en_espanol():
    """Guard tonto a proposito: un copy-paste que deje el espanol en las 4 casillas
    pasaria todos los tests de arriba."""
    for nombre in ("TAGLINE", "CARGANDO"):
        mapa = _mapa(nombre)
        for loc in ("en-US", "fr-FR", "it-IT"):
            assert mapa[loc] != mapa["es-DO"], (
                f"{nombre}[{loc}] es identico al espanol: copy-paste sin traducir")


# ───────────────────────── el orden es load-bearing ─────────────────────────

def test_el_script_va_DESPUES_de_los_dos_nodos_que_lee():
    """La primera version lo puso entre el tagline y los puntos: el `querySelector` de
    `.splash-dots` devolvia null y el aria-label se quedaba en espanol en silencio."""
    html = _html()
    i_tag = html.index('class="splash-tagline"')
    i_dots = html.index('class="splash-dots"')
    i_script = html.index("var TAGLINE = {")
    assert i_script > i_tag, "el script corre antes del tagline"
    assert i_script > i_dots, "el script corre antes de los puntos: aria-label sin traducir"


def test_no_espera_a_DOMContentLoaded():
    """Un listener diferido pinta espanol y lo cambia despues: parpadeo. El script
    inline sincrono es justo lo que lo evita, igual que sus dos hermanos."""
    html = _html()
    bloque = html[html.index("var TAGLINE = {"):]
    bloque = bloque[:bloque.index("</script>")]
    assert "DOMContentLoaded" not in bloque
    assert "setTimeout" not in bloque


# ───────────────────────── contrato con el boot de idioma ─────────────────────────

def test_reutiliza_la_lectura_del_boot_de_idioma():
    """Una segunda lectura de localStorage seria una segunda fuente de verdad: dos
    sitios que pueden discrepar sobre que idioma es. El boot de `lang` ya lo leyo."""
    html = _html()
    assert "window.__mfLocale = stored" in html, "el boot de idioma ya no publica el locale"
    bloque = html[html.index("var TAGLINE = {"):]
    bloque = bloque[:bloque.index("</script>")]
    assert "window.__mfLocale" in bloque
    assert "localStorage" not in bloque, "el splash lee localStorage por su cuenta"


def test_es_defensivo_como_sus_hermanos():
    """Cualquier excepcion debe dejar el espanol del marcado. Los otros dos boot
    scripts hacen lo mismo y por el mismo motivo: un splash en blanco es peor que un
    splash en el idioma equivocado."""
    html = _html()
    bloque = html[html.index("var TAGLINE = {"):]
    bloque = bloque[:bloque.index("</script>")]
    assert "catch" in bloque
