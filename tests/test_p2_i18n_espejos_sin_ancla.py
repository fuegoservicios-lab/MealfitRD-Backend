"""[P2-I18N-ESPEJOS-SIN-ANCLA · 2026-08-21] La lista de idiomas vive en 16 sitios y el
test anclaba 5.

`src/i18n/locales.js` es el SSOT y su propio comentario dice que la lista «vive en cinco
lados» y que `test_p1_i18n_dashboard.py` «falla si esos cuatro sitios divergen». Faltaban
NUEVE espejos, y ninguno estaba anclado:

  frontend  · `i18n/index.js` → `LOADERS` (el mapa de `import()` por idioma)
            · `i18n/index.js` → el mapa de `Intl` / formato
            · `index.html`    → DOS mapas más en el boot síncrono, aparte de `SUPPORTED`
  backend   · `prompts/chat_agent.py` → `_COACH_LANGUAGE_NAMES`
            · `prompts/chat_agent.py` → `_TITLE_LANGUAGE_DIRECTIVES`
            · `plan_display_i18n.py`  → `_DISPLAY_LANGUAGE_DIRECTIVES`
            · `plan_display_i18n.py`  → `_PLAN_NAME_ADDENDUM` / `_INSIGHTS_ADDENDUM`
  migración · el `NOT IN` del sanity SQL, aparte del CHECK que sí estaba anclado

Es la misma clase de drift que cerró `P1-DIET-CANON-SSOT`: tres tablas de dieta escritas
a mano driftearon y a la del filtro se le olvidó `'vegetariana'` — el sistema servía Pollo
a vegetarianas. Aquí el fallo es más callado todavía, porque un idioma que falta en un
espejo no rompe nada: simplemente esa superficie sale en español.

CADA ESPEJO TIENE SU PROPIO TEST, y eso es deliberado. Un único test que compare los 16
conjuntos dice «algo divergió» y te deja buscando; uno por espejo dice CUÁL y —más
importante— **qué se ve si falta**, que es lo que un lector necesita para decidir si es
urgente. La consecuencia va en el mensaje de cada aserción, no en un comentario.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_MARKER = "P2-I18N-ESPEJOS-SIN-ANCLA"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_FRONTEND = _ROOT / "frontend"


def _hermanos_disponibles() -> bool:
    """Este checkout es de verdad `<raíz>/backend` con su `frontend/` al lado.

    Sin el check, un worktree en `C:/tmp/...` resuelve `_ROOT` a `C:/tmp` y puede
    encontrarse un `frontend/` viejo y ajeno — leerlo es medir código que nadie
    despliega.
    """
    return (_ROOT / "backend").is_dir() and _FRONTEND.is_dir()


def _leer(p: Path) -> str:
    if not _hermanos_disponibles():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# El SSOT
# ---------------------------------------------------------------------------

def _ssot() -> set[str]:
    src = _leer(_FRONTEND / "src" / "i18n" / "locales.js")
    m = re.search(r"export\s+const\s+LOCALES\s*=\s*\[(.*?)\];", src, re.S)
    assert m, (
        f"no encontré `export const LOCALES = [...]` en locales.js. Si cambió su forma, "
        f"actualiza este extractor: es del que cuelgan los 16 espejos. [{_MARKER}]"
    )
    codigos = set(re.findall(r"code:\s*'([a-z]{2}-[A-Z]{2})'", m.group(1)))
    assert len(codigos) >= 4, f"solo extraje {codigos} del SSOT — ¿cambió el estilo?"
    m_def = re.search(r"DEFAULT_LOCALE\s*=\s*'([a-z]{2}-[A-Z]{2})'", src)
    assert m_def, "no encontré DEFAULT_LOCALE"
    return codigos | {m_def.group(1)}


def _traducidos() -> set[str]:
    """Los idiomas CON catálogo: el SSOT menos el base, que es el fallback y no tiene
    archivo. Los espejos que enumeran traducciones usan este conjunto, no el completo."""
    src = _leer(_FRONTEND / "src" / "i18n" / "locales.js")
    m_def = re.search(r"DEFAULT_LOCALE\s*=\s*'([a-z]{2}-[A-Z]{2})'", src)
    return _ssot() - {m_def.group(1)}


def _codigos_en(texto: str) -> set[str]:
    return set(re.findall(r"['\"]([a-z]{2}-[A-Z]{2})['\"]", texto))


def _bloque(src: str, nombre: str, apertura: str = "{", cierre: str = "}") -> str:
    """El cuerpo de una asignación `nombre = {...}` con balanceo real."""
    m = re.search(re.escape(nombre) + r"\s*(?::[^=]*)?=\s*" + re.escape(apertura), src)
    if not m:
        pytest.fail(
            f"no encontré `{nombre} = {apertura}…{cierre}`. Si se renombró, actualiza "
            f"este guard — no lo borres: es un espejo del SSOT de idiomas. [{_MARKER}]"
        )
    i = m.end() - 1
    prof = 0
    while i < len(src):
        if src[i] == apertura:
            prof += 1
        elif src[i] == cierre:
            prof -= 1
            if prof == 0:
                return src[m.end():i]
        i += 1
    pytest.fail(f"no pude cerrar el bloque de `{nombre}` [{_MARKER}]")


# ---------------------------------------------------------------------------
# Espejos del frontend
# ---------------------------------------------------------------------------

def test_el_mapa_de_loaders_cubre_todos_los_idiomas() -> None:
    src = _leer(_FRONTEND / "src" / "i18n" / "index.js")
    vistos = _codigos_en(_bloque(src, "LOADERS"))
    faltan = _traducidos() - vistos
    assert not faltan, (
        f"`LOADERS` no tiene entrada para {sorted(faltan)}. LO QUE SE VE: el usuario "
        f"elige ese idioma, el selector se lo acepta, y el catálogo no se descarga "
        f"nunca — la app entera se queda en español sin un solo error en consola. "
        f"[{_MARKER}]"
    )


def test_el_boot_sincrono_cubre_todos_los_idiomas() -> None:
    src = _leer(_FRONTEND / "index.html")
    m = re.search(r"var\s+SUPPORTED\s*=\s*\[([^\]]+)\]", src)
    assert m, f"no encontré `SUPPORTED = [...]` en el boot de index.html [{_MARKER}]"
    faltan = _ssot() - _codigos_en(m.group(1))
    assert not faltan, (
        f"el boot síncrono no conoce {sorted(faltan)}. LO QUE SE VE: `<html lang>` "
        f"arranca en es-DO hasta que React monta — parpadeo, y un lector de pantalla "
        f"leyendo francés con voz española durante el arranque en frío. [{_MARKER}]"
    )


# ---------------------------------------------------------------------------
# Espejos del backend
# ---------------------------------------------------------------------------

def test_los_nombres_de_idioma_del_coach_cubren_el_ssot() -> None:
    src = _leer(_BACKEND / "prompts" / "chat_agent.py")
    faltan = _traducidos() - _codigos_en(_bloque(src, "_COACH_LANGUAGE_NAMES"))
    assert not faltan, (
        f"`_COACH_LANGUAGE_NAMES` no cubre {sorted(faltan)}. LO QUE SE VE: el usuario "
        f"tiene la app en ese idioma y el coach le contesta en español. [{_MARKER}]"
    )


def test_las_directivas_de_titulo_cubren_el_ssot() -> None:
    src = _leer(_BACKEND / "prompts" / "chat_agent.py")
    faltan = _traducidos() - _codigos_en(_bloque(src, "_TITLE_LANGUAGE_DIRECTIVES"))
    assert not faltan, (
        f"`_TITLE_LANGUAGE_DIRECTIVES` no cubre {sorted(faltan)}. LO QUE SE VE: los "
        f"títulos de conversación del chat nacen en español dentro de una app que ese "
        f"usuario tiene en otro idioma. [{_MARKER}]"
    )


def test_las_directivas_de_display_cubren_el_ssot() -> None:
    src = _leer(_BACKEND / "plan_display_i18n.py")
    faltan = _traducidos() - _codigos_en(_bloque(src, "_DISPLAY_LANGUAGE_DIRECTIVES"))
    assert not faltan, (
        f"`_DISPLAY_LANGUAGE_DIRECTIVES` no cubre {sorted(faltan)}. LO QUE SE VE: "
        f"`_build_prompt` devuelve None para ese locale y el enriquecimiento se salta "
        f"ENTERO — plan y recetas en español, sin error. [{_MARKER}]"
    )


@pytest.mark.parametrize("mapa", ["_PLAN_NAME_ADDENDUM", "_INSIGHTS_ADDENDUM"])
def test_los_addenda_del_display_cubren_el_ssot(mapa: str) -> None:
    src = _leer(_BACKEND / "plan_display_i18n.py")
    faltan = _traducidos() - _codigos_en(_bloque(src, mapa))
    assert not faltan, (
        f"`{mapa}` no cubre {sorted(faltan)}. LO QUE SE VE: los meals de ese plan salen "
        f"traducidos y el nombre del plan (o su razonamiento) se queda en español — "
        f"media pantalla en cada idioma. [{_MARKER}]"
    )


def test_el_whitelist_de_locales_del_backend_cubre_el_ssot() -> None:
    src = _leer(_BACKEND / "routers" / "user_data.py")
    # `frozenset({...})`, `{...}` o `(...)`: se toma la ASIGNACION entera hasta el fin
    # de linea en vez de adivinar el envoltorio. Mi primer regex exigia que tras el `=`
    # viniera un parentesis o una llave, y `frozenset(` empieza por `f`.
    m = re.search(r"^_LOCALE_VALUES\s*=\s*(.+)$", src, re.M)
    assert m, f"no encontré `_LOCALE_VALUES` en user_data.py [{_MARKER}]"
    faltan = _ssot() - _codigos_en(m.group(1))
    assert not faltan, (
        f"`_LOCALE_VALUES` no acepta {sorted(faltan)}. LO QUE SE VE: el selector cambia "
        f"el idioma en el navegador, el PATCH lo rechaza, y al recargar vuelve al "
        f"anterior — el usuario cree que la app «no guarda» su elección. [{_MARKER}]"
    )


# ---------------------------------------------------------------------------
# El catálogo en disco
# ---------------------------------------------------------------------------

def test_cada_idioma_traducido_tiene_su_catalogo() -> None:
    loc = _FRONTEND / "src" / "i18n" / "locales"
    if not _hermanos_disponibles() or not loc.exists():
        pytest.skip("catálogos no disponibles en este checkout")
    faltan = sorted(c for c in _traducidos() if not (loc / f"{c}.json").exists())
    assert not faltan, (
        f"sin archivo de catálogo: {faltan}. LO QUE SE VE: el `import()` de `LOADERS` "
        f"revienta, `loadLocale` se traga la excepción y devuelve false, y la app se "
        f"queda en español. [{_MARKER}]"
    )
    for c in _traducidos():
        datos = json.loads((loc / f"{c}.json").read_text(encoding="utf-8"))
        assert isinstance(datos, dict) and datos, f"{c}.json vacío o mal formado"


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def test_el_extractor_del_ssot_no_esta_vacio() -> None:
    """MUTACIÓN DE CONTROL. Si `_ssot()` devolviera el conjunto vacío, TODOS los tests
    de arriba pasarían —`vacío - X` es vacío— sin comprobar nada. Es el modo de fallo
    que ya se pagó en `P1-CULINARY-METADATA-BETA`: el guard en verde, pasando en vacío.
    """
    s = _ssot()
    assert len(s) >= 5, f"el SSOT solo dio {sorted(s)}; el producto declara 5 idiomas"
    assert "es-DO" in s, "falta el idioma base en el SSOT extraído"
    assert _traducidos() and "es-DO" not in _traducidos(), (
        "el conjunto de traducidos tiene que excluir el base (no tiene catálogo)"
    )
