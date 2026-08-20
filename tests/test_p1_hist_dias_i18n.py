"""[P1-HIST-DIAS-I18N · 2026-08-19] El modal del Historial se leia en espanol con la
interfaz en ingles.

Lo reporto el dueno con la app en en-US: los chips de comida (BREAKFAST/LUNCH) y las
etiquetas de macros SI estaban traducidos —el trabajo de P1-PLAN-DISPLAY-I18N habia
aterrizado—, pero los tabs de dia decian «Martes», el titulo «Menu — Martes» y la fecha
«martes, 18 de agosto de 2026».

Que los NOMBRES DE PLATO sigan en espanol es de diseno (los escribe el LLM, y traducir
nombres de alimento rompe `pantry_names_match`, el guard de coherencia y el backstop de
alergias). Los nombres de DIA y la FECHA no son contenido: son interfaz.

LO QUE ESTE P-FIX ENSENA

1. Era el UNICO sitio que quedaba asi. `Recipes.jsx`, `Dashboard.jsx` y
   `DiaryHistory.jsx` ya pasaban esos mismos siete literales por `t()`, y los cuatro
   catalogos ya traian la traduccion. No faltaba maquinaria: faltaba pedirla. El
   diagnostico correcto no era «hay que internacionalizar el Historial».

2. El motor ya habia anticipado la mitad del bug: `formatDate` existe en
   `i18n/index.js` con un comentario que dice literalmente que es el reemplazo de los
   `toLocaleDateString('es-DO')` fijos repartidos por el repo, «un menu en frances con
   fechas en espanol es exactamente el descuido que delata una traduccion a medias».

3. LA TRAMPA DEL CONGELADO. Envolver el array tal cual en `t()` habria sido un bug
   distinto y mas dificil de ver: un `const _X = [t('Domingo'), ...]` a nivel de modulo
   se evalua UNA vez al importar —antes de que `initLocale()` cargue el catalogo— y se
   queda en espanol para siempre ademas de no reaccionar al cambio de idioma. En es-DO
   parece correcto. Tiene que ser una FUNCION llamada en render.

4. Y un comentario derroto al guard, otra vez. `scripts/i18n-check.mjs` vigila
   justamente el punto 3, pero su matcher lee la fuente CRUDA: el `t('Domingo')` citado
   DENTRO del comentario que explica la trampa se reportaba como llamada en ambito de
   modulo. Se arreglo el checker en vez de censurar la prosa —un guard que obliga a no
   documentarlo es un guard que alguien acaba desactivando— y el filtro se limito a
   comentarios: excluir tambien las cadenas se habria tragado los `${t('Foo')}` de un
   template literal, que son llamadas REALES. Cambiar un falso positivo por un falso
   negativo no es arreglar nada.

tooltip-anchor: P1-HIST-DIAS-I18N
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY = _ROOT / "frontend" / "src" / "pages" / "History.jsx"
_CHECKER = _ROOT / "frontend" / "scripts" / "i18n-check.mjs"
_LOCALES = _ROOT / "frontend" / "src" / "i18n" / "locales"

_DIAS = ["Domingo", "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


# ─────────────── los nombres de dia salen del catalogo, no del codigo ───────────────

def test_history_no_conserva_el_array_crudo_en_espanol():
    src = _leer(_HISTORY)
    assert not re.search(r"const\s+_DIAS_SEMANA\s*=\s*\[", src), (
        "vuelve a haber un array literal de dias: el modal se leera en espanol "
        "sea cual sea el idioma elegido")


def test_los_siete_nombres_pasan_por_t():
    src = _leer(_HISTORY)
    assert re.search(
        r"t\(['\"]Domingo['\"]\)\s*,\s*t\(['\"]Lunes['\"]\)\s*,\s*t\(['\"]Martes", src)


def test_es_una_funcion_y_no_una_constante_de_modulo():
    """La trampa del congelado (punto 3 del docstring). El cuerpo va con LLAVES
    ademas porque `i18n-check.mjs` decide el ambito contando llaves abiertas y un
    cuerpo conciso `=> [...]` le sale a profundidad 0."""
    src = _leer(_HISTORY)
    m = re.search(r"const\s+_diaSemana\s*=\s*\(\s*idx\s*\)\s*=>\s*\{", src)
    assert m, "«_diaSemana» debe ser una funcion con cuerpo entre llaves"


def _sin_comentarios_de_bloque(src: str) -> str:
    """Quita los `/* ... */` (incluidos los `{/* ... */}` de JSX).

    Existe porque este mismo test cayo en el punto 4 de su docstring: la version
    original comparaba contra la fuente CRUDA y la disparaba el comentario que el
    propio P-fix dejo en History.jsx citando `toLocaleDateString('es-DO')`. La
    respuesta correcta no era reescribir el comentario.

    Se filtran SOLO los bloques, no los `//`: un `//` dentro de una cadena (una URL,
    sin ir mas lejos) haria que el filtro se comiera codigo real, y en una asercion
    de tipo «no debe aparecer» comerse codigo es un falso VERDE. Por eso ademas se
    exige el punto de la llamada (`.toLocaleDateString`), que la prosa no lleva.
    """
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)


def test_la_fecha_del_modal_usa_el_locale_activo():
    src = _leer(_HISTORY)
    codigo = _sin_comentarios_de_bloque(src)
    assert ".toLocaleDateString('es-DO'" not in codigo, (
        "vuelve a haber un locale cableado: la fecha saldra en espanol en las 4 "
        "traducciones")
    assert re.search(r"formatDate\(\s*selectedPlan\.created_at", src)
    assert re.search(r"import \{[^}]*\bformatDate\b[^}]*\} from '\.\./i18n'", src)


@pytest.mark.parametrize("locale", ["en-US", "fr-FR", "it-IT", "pt-BR"])
def test_los_cuatro_catalogos_traen_los_siete_dias(locale):
    """Test de DATO, no de parseo: si una traduccion desaparece, `t()` cae al espanol
    en silencio y el sintoma es exactamente el que este P-fix cierra."""
    cat = json.loads(_leer(_LOCALES / f"{locale}.json"))
    faltan = [d for d in _DIAS if not cat.get(d)]
    assert not faltan, f"{locale} no traduce {faltan}"


# ─────────────── el guard que vigila la trampa, y su propia trampa ───────────────

def test_el_checker_ignora_los_comentarios_pero_no_las_cadenas():
    """Punto 4 del docstring. Las dos mitades importan: sin la primera, documentar la
    trampa dispara el guard; con la segunda de mas, un `${t('Foo')}` en un template
    literal a nivel de modulo dejaria de reportarse."""
    src = _leer(_CHECKER)
    m = re.search(r"function isModuleScopeCode\(src, index\)\s*\{(.*?)\n\}", src, re.S)
    assert m, "falta isModuleScopeCode: el filtro de comentarios se perdio"
    cuerpo = m.group(1)
    assert "!inComment" in cuerpo, "el checker volvio a reportar la prosa"
    assert "inStr" not in cuerpo, (
        "excluir las cadenas convierte el falso positivo en un falso negativo: se "
        "tragaria los ${t('Foo')} de un template literal")
    assert "isModuleScope(src, m.index)" not in src, "quedo un call site del filtro viejo"
