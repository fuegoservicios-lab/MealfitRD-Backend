"""[P2-I18N-MANIFEST-MONOLINGUE · 2026-08-21] La app invita a instalar en tu idioma y se
instalaba en español.

`public/manifest.json` trae `name`, `description` y los seis literales de los tres
`shortcuts` en español, con `lang: "es-DO"`. Un francés instala la PWA desde una interfaz
que ya está en francés y el icono de su escritorio dice «Bioboros | Nutrición con IA»,
con atajos «Nuevo Chat» y «Lista del Súper».

No es cosmético: el manifiesto es lo ÚNICO que el sistema operativo recuerda de la app.
El usuario ve ese nombre cada vez que abre el móvil, mucho después de haber olvidado en
qué idioma configuró nada — y no hay ninguna pantalla donde corregirlo.

POR QUÉ UN FICHERO POR IDIOMA Y NO UNO CON `lang` DINÁMICO: el manifiesto se descarga y
se cachea POR URL. Un solo fichero significa un solo idioma para todo el que ya lo tenga
cacheado, y el navegador no lo vuelve a pedir al cambiar de idioma. Una URL distinta es
lo único que entiende como «esto es otra cosa».

EL ESPAÑOL NO SE GENERA, y ese es el fail-safe: `manifest.json` sigue siendo el de es-DO
tal cual, y el boot solo reescribe el `href` cuando el locale NO es el base. Si el
generador no corre —o falla— la conducta es exactamente la de hoy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_MARKER = "P2-I18N-MANIFEST-MONOLINGUE"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_FRONT = _ROOT / "frontend"


def _leer(rel: str) -> str:
    if not (_ROOT / "backend").is_dir() or not _FRONT.is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"{rel} no existe en este checkout")
    return p.read_text(encoding="utf-8")


def _generador() -> str:
    return _leer("scripts/build-manifests-i18n.mjs")


def test_el_generador_existe_y_esta_en_el_build() -> None:
    _generador()
    pkg = json.loads(_leer("package.json"))
    post = pkg.get("scripts", {}).get("postbuild", "")
    assert "build-manifests-i18n" in post, (
        f"el generador no está encadenado al build. Un script que hay que acordarse de "
        f"invocar es una costumbre, no una defensa. [{_MARKER}]"
    )


def test_cubre_los_cuatro_idiomas_traducidos() -> None:
    """El español no se genera A PROPÓSITO: `manifest.json` ya es el suyo."""
    src = _generador()
    for code in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        assert f"'{code}'" in src, (
            f"el generador no cubre {code}: ese usuario instala la PWA en español. "
            f"[{_MARKER}]"
        )
    assert "'es-DO':" not in src, (
        f"el generador escribe un `manifest.es-DO.json`. El español tiene que seguir "
        f"usando `manifest.json` tal cual — es lo que hace que un fallo del generador "
        f"degrade a la conducta de hoy en vez de romper la instalación. [{_MARKER}]"
    )


def test_traduce_los_tres_campos_que_ve_el_usuario() -> None:
    src = _generador()
    for campo in ("name:", "description:", "shortcuts:"):
        assert campo in src, f"el generador no toca `{campo}` [{_MARKER}]"
    assert "lang: code" in src, (
        f"el manifiesto generado no declara su propio `lang`. Sin él dice `es-DO` "
        f"mientras el contenido está en francés. [{_MARKER}]"
    )


def test_la_marca_no_se_traduce() -> None:
    """`short_name` de la app es «Bioboros»: es lo que el usuario busca en su lanzador.
    Traducirlo sería cambiarle el nombre al producto."""
    src = _generador()
    assert "short_name` de la app NO se traduce" in src or "no se traduce: es la marca" in src, (
        f"falta la nota de por qué `short_name` de la app se queda igual. Sin ella, el "
        f"siguiente que pase «completa» la traducción. [{_MARKER}]"
    )
    # Y de hecho no puede estar entre las claves que sobrescribe a nivel raíz.
    m = re.search(r"const salida = \{ \.\.\.base,([^}]*)\}", src)
    assert m, "no encontré la construcción del manifiesto de salida"
    assert "short_name" not in m.group(1), (
        f"el generador sobrescribe `short_name` a nivel raíz: eso traduce la marca. "
        f"[{_MARKER}]"
    )


def test_los_atajos_se_traducen_por_posicion_y_degradan() -> None:
    """Si alguien añade un cuarto atajo al manifiesto base y no aquí, ese atajo tiene que
    quedarse en español — no desaparecer. Degradación, no pérdida."""
    src = _generador()
    assert "base.shortcuts.map" in src, (
        f"los atajos no se mapean desde el manifiesto BASE. Construirlos desde la "
        f"traducción haría desaparecer el atajo que falte. [{_MARKER}]"
    )
    assert "return t ? {" in src and ": sc;" in src, (
        f"falta el fallback por atajo: sin traducción, se conserva el original. "
        f"[{_MARKER}]"
    )


def test_el_boot_reescribe_el_href_solo_fuera_del_espanol() -> None:
    html = _leer("index.html")
    assert "link[rel=\"manifest\"]" in html, (
        f"el boot no reescribe el `<link rel=\"manifest\">`. Los manifiestos por idioma "
        f"existirían y nadie los pediría. [{_MARKER}]"
    )
    m = re.search(r"if \(loc && loc !== 'es-DO'[^)]*\)", html)
    assert m, (
        f"la reescritura no está condicionada a que el locale NO sea el base. Sin esa "
        f"condición, un es-DO pediría `manifest.es-DO.json`, que no se genera — 404 y "
        f"PWA sin manifiesto. [{_MARKER}]"
    )
    i_boot = html.index("window.__mfLocale")
    i_href = html.index('link[rel="manifest"]')
    assert i_boot < i_href, (
        f"la reescritura ocurre antes de que el boot resuelva el locale. [{_MARKER}]"
    )


def test_el_manifiesto_base_sigue_en_espanol() -> None:
    """MUTACIÓN DE CONTROL. Si alguien «arregla» esto traduciendo el base al inglés, los
    hispanohablantes —la mayoría— pierden el suyo, y este test lo dice."""
    man = json.loads(_leer("public/manifest.json"))
    assert man.get("lang") == "es-DO", (
        f"`manifest.json` dejó de ser el español. Es el fallback de todos: si el "
        f"generador falla, esto es lo que se instala. [{_MARKER}]"
    )
    assert "Nutrición" in man.get("name", ""), (
        f"el nombre del manifiesto base ya no está en español: {man.get('name')!r} "
        f"[{_MARKER}]"
    )
