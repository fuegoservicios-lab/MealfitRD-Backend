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
import shutil
import subprocess
import tempfile
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
    literal a nivel de modulo dejaria de reportarse.

    [P1-I18N-EXTRACTOR-AST · 2026-08-21] Reescrito para medir CONDUCTA.

    Antes buscaba el cuerpo de `function isModuleScopeCode(src, index)` y comprobaba
    que dentro pusiera `!inComment` y no `inStr`. Eso anclaba una IMPLEMENTACION, y
    cuando el guard paso de contar llaves a usar un AST —porque el contador era ciego
    al ejemplo canonico de la doc, `const TABS = [{ label: t('X') }]`— el test se puso
    rojo sin que la propiedad que dice defender se hubiera roto: con AST el filtro de
    comentarios es NATIVO (una llamada citada en prosa no es un CallExpression) y la
    firma gano un tercer parametro.

    Un test que se rompe al mejorar el codigo que vigila estaba midiendo la forma, no
    la propiedad. Ahora ejecuta el checker real contra las dos mitades.
    """
    if not shutil.which("node"):
        pytest.skip("node no esta en PATH")

    def _reporta_ambito(fuente: str, tmp: Path) -> bool:
        (tmp / "scripts").mkdir(parents=True, exist_ok=True)
        # [P3-I18N-ENTRADAS-DUPLICADAS · 2026-08-22] Se copia TODO `scripts/*.mjs`, no una
        # tupla enumerada.
        #
        # La lista a mano ya se quedó corta una vez y volvió a quedarse corta hoy, cuando
        # `ENTRADAS` pasó a vivir en `scripts/entradas.mjs`: los seis tests que ejecutan el
        # checker REAL murieron en ERR_MODULE_NOT_FOUND y fallaron por una razón que no
        # tenía nada que ver con lo que miden.
        #
        # Enumerar las dependencias de un programa que se ejecuta de verdad es mantener una
        # segunda copia de su grafo de imports. El glob no puede quedarse corto.
        for rel in sorted(p.name for p in _CHECKER.parent.glob("*.mjs")):
            origen = _CHECKER.parent / rel
            if origen.exists():
                shutil.copy2(origen, tmp / "scripts" / rel)
        lib = _CHECKER.parent / "lib"
        if lib.exists():
            (tmp / "scripts" / "lib").mkdir(parents=True, exist_ok=True)
            for f in lib.glob("*.mjs"):
                shutil.copy2(f, tmp / "scripts" / "lib" / f.name)
        src_dir = tmp / "src"
        (src_dir / "i18n" / "locales").mkdir(parents=True, exist_ok=True)
        (src_dir / "i18n" / "locales.js").write_text(
            "export const DEFAULT_LOCALE = 'es-DO';\n"
            "export const LOCALES = [{ code: 'es-DO' }, { code: 'en-US' }];\n",
            encoding="utf-8",
        )
        (src_dir / "i18n" / "locales" / "en-US.json").write_text("{}\n", encoding="utf-8")
        (src_dir / "c").mkdir(parents=True, exist_ok=True)
        (src_dir / "c" / "D.jsx").write_text(fuente, encoding="utf-8")
        enlace = tmp / "node_modules"
        real = _CHECKER.parent.parent / "node_modules"
        if real.exists() and not enlace.exists():
            try:
                enlace.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("no puedo enlazar node_modules en esta plataforma")
        r = subprocess.run(
            ["node", str(tmp / "scripts" / "i18n-check.mjs")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return "ÁMBITO DE MÓDULO" in (r.stdout + r.stderr)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        # Mitad 1: la prosa que DOCUMENTA la trampa no puede dispararla.
        prosa = (
            "import { t } from '../i18n';\n"
            "// Ojo: `const DIAS = [t('Lunes')]` en ambito de modulo se congela.\n"
            "export function f() { return t('Lunes'); }\n"
        )
        assert not _reporta_ambito(prosa, Path(d1)), (
            "el checker volvio a reportar la prosa: documentar la trampa dispara el "
            "guard, y un guard que obliga a censurar su propia documentacion acaba "
            "desactivado"
        )

        # Mitad 2: y una llamada REAL dentro de un template literal a nivel de modulo
        # sigue reportandose. Es la mutacion de control de la mitad 1: sin ella,
        # apagar el guard entero pasaria este test.
        template = (
            "import { t } from '../i18n';\n"
            "const AVISO = `${t('Lunes')} es el primer dia`;\n"
            "export default AVISO;\n"
        )
        assert _reporta_ambito(template, Path(d2)), (
            "un ${t('...')} en un template literal a nivel de modulo dejo de "
            "reportarse: el falso positivo se cambio por un falso negativo"
        )
