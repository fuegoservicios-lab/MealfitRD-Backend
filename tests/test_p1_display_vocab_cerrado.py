"""[P1-DISPLAY-VOCAB-CERRADO · 2026-08-21] El LLM traducía el vocabulario que los
parsers de pantalla casan literalmente.

Las recetas no son prosa lisa: cada paso empieza por una etiqueta de sección
(«Mise en place:», «El Toque de Fuego:», «Montaje:») y algunos son ANOTACIONES en vez
de acciones («Nota del nutricionista:», «Seguridad alimentaria:»). Los tres parsers del
frontend —`RecipesView.jsx`, `MobileRecipes.jsx`, `utils/recipeSteps.js`— casan **español
literal**:

    /^mise en place:\\s*/i · /^(el\\s+)?toque de fuego:\\s*/i · /^montaje:\\s*/i
    /nota del nutricionista/i · /seguridad alimentaria\\s*:/i · /ajustamos ligeramente…/i

Las directivas de `plan_display_i18n` sólo exceptúan el **nombre canónico del alimento**,
y el validador de `recipe` sólo miraba cifras y longitud. Así que el LLM traducía las
etiquetas y el paso dejaba de reconocerse.

MEDIDO sobre los 1.904 pasos de receta vivos (2026-08-21): **1.816 llevan vocabulario
cerrado, el 95,4 %** — 599 «Montaje», 598 «Mise en place», 484 «Toque de fuego», y 135
anotaciones (71 seguridad alimentaria, 62 nota del nutricionista, 2 porciones).

LO QUE DUELE NO ES EL FORMATO, SON LAS 135 ANOTACIONES. Un paso que pierde su etiqueta
de nota deja de ser anotación y pasa a NUMERARSE como acción de cocina: la numeración de
un meal real va de `[1, null, 2]` en es-DO a `[1, 2, 3]` en los demás idiomas, y una nota
nutricional aparece como «Step 2». Es exactamente el defecto que `P2-RECIPE-NOTES-NOT-STEPS`
cerró, resucitado para 4 de los 5 idiomas.

EL CRITERIO, que es el mismo que ya rige para los nombres de alimento: **esto no es prosa,
es un IDENTIFICADOR**. Se conserva literal en el dato y se traduce en la PANTALLA. Por eso
el arreglo tiene dos mitades y ninguna sirve sola:

  1. Backend: la directiva pide conservarlos exactos, y el validador cae al español
     **para esa línea** si la traducción perdió el prefijo (mismo fallback per-línea que
     ya existía para las cifras).
  2. Frontend: `parseStep`/`isRecipeAnnotation` reconocen el prefijo español y pintan el
     título con `t()`. Sin esta mitad, un inglés leería «El Toque de Fuego» — que es lo
     que ya le pasaba al 33 % que sí parseaba en en-US.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MARKER = "P1-DISPLAY-VOCAB-CERRADO"

_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT = _ROOT / "frontend" / "src"


@pytest.fixture()
def mod():
    import plan_display_i18n

    return importlib.reload(plan_display_i18n)


# ============================================================
# 1 · El vocabulario cerrado es UN SSOT, no tres listas
# ============================================================


def test_existe_el_ssot_del_vocabulario(mod) -> None:
    assert hasattr(mod, "_VOCAB_CERRADO"), (
        f"No existe `_VOCAB_CERRADO`. El vocabulario que los parsers casan literalmente "
        f"tiene que estar en UN sitio: es la lección de P1-DIET-CANON-SSOT, donde tres "
        f"tablas a mano driftearon y una olvidó 'vegetariana'. [{_MARKER}]"
    )
    assert len(mod._VOCAB_CERRADO) >= 6, (
        f"`_VOCAB_CERRADO` sólo tiene {len(mod._VOCAB_CERRADO)} entradas; los parsers "
        f"del frontend casan 3 secciones + 3 anotaciones. [{_MARKER}]"
    )


@pytest.mark.parametrize(
    "linea",
    [
        "Mise en place: pica la cebolla y el ajo.",
        "El Toque de Fuego: sella el pollo a fuego alto.",
        "Toque de Fuego: sella el pollo.",
        "Montaje: sirve el arroz al lado.",
        "🔬 Nota del nutricionista: esta comida cubre el 30 % del hierro diario.",
        "Seguridad alimentaria: cocina el pollo hasta 74 grados.",
        "Ajustamos ligeramente las porciones para cuadrar tus macros.",
    ],
)
def test_el_detector_reconoce_las_formas_reales(mod, linea: str) -> None:
    """Las formas EXACTAS que trae el corpus, incluida la variante sin «El» y la que
    lleva emoji delante (los parsers limpian la cabeza antes de casar)."""
    assert mod._marca_de_vocab_cerrado(linea) is not None, (
        f"no reconoció «{linea[:45]}…». Si el detector no la ve, el validador no puede "
        f"protegerla. [{_MARKER}]"
    )


def test_el_detector_no_marca_un_paso_de_cocina_normal(mod) -> None:
    """MUTACIÓN DE CONTROL. Un detector que devuelva algo para todo haría pasar lo de
    arriba sin medir nada, y de paso congelaría en español las recetas enteras."""
    for normal in (
        "Cuece el arroz blanco según el paquete y sírvelo caliente.",
        "Agrega yogurt a la licuadora y licúa hasta integrar.",
        "Cocina huevo a la plancha y sírvelo como proteína del plato.",
        "",
    ):
        assert mod._marca_de_vocab_cerrado(normal) is None, (
            f"marcó como vocabulario cerrado un paso normal: «{normal[:45]}». Con esto, "
            f"el validador devolvería el español para media receta. [{_MARKER}]"
        )


# ============================================================
# 2 · El validador cae al español POR LÍNEA si se perdió el prefijo
# ============================================================


def _original(pasos: list) -> dict:
    return {
        "day_idx": 0, "meal_idx": 0,
        "name": "Pollo guisado", "description": "Plato criollo.",
        "recipe": pasos,
        "ingredients": ["180 g de Pechuga de pollo"],
    }


def _traducido(pasos: list) -> dict:
    return {
        "i": 0, "name": "Stewed chicken", "description": "Creole dish.",
        "recipe": pasos,
        "ingredients": ["180 g chicken breast (Pechuga de pollo)"],
    }


def test_un_paso_que_pierde_el_prefijo_cae_al_espanol(mod) -> None:
    orig = ["Mise en place: pica la cebolla.", "Montaje: sirve caliente."]
    trad = ["Prep work: chop the onion.", "Plating: serve hot."]

    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d is not None, "el meal entero se descartó; el fallback es POR LÍNEA"
    assert d["recipe"] == orig, (
        f"las líneas que perdieron el prefijo no cayeron al español: {d['recipe']!r}. "
        f"Sin el prefijo, el parser de pantalla no reconoce la sección. [{_MARKER}]"
    )


def test_una_anotacion_que_pierde_su_etiqueta_cae_al_espanol(mod) -> None:
    """El caso que más duele: sin la etiqueta, la nota se numera como paso de cocina."""
    orig = [
        "Mise en place: pica la cebolla.",
        "Nota del nutricionista: cubre el 30 % del hierro diario.",
    ]
    trad = [
        "Mise en place: chop the onion.",
        "Nutritionist note: covers 30 % of your daily iron.",
    ]
    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d["recipe"][0] == "Mise en place: chop the onion.", (
        "la línea que SÍ conservó el prefijo tenía que quedarse traducida"
    )
    assert d["recipe"][1] == orig[1], (
        f"la anotación traducida se persistió: {d['recipe'][1]!r}. Sin «Nota del "
        f"nutricionista», `isRecipeAnnotation` devuelve false y la nota pasa a "
        f"numerarse como acción de cocina — el defecto que P2-RECIPE-NOTES-NOT-STEPS "
        f"cerró. [{_MARKER}]"
    )


def test_el_prefijo_conservado_deja_pasar_la_traduccion(mod) -> None:
    """El caso bueno tiene que seguir funcionando: sólo el prefijo es identificador; el
    resto de la línea se traduce y se persiste traducido."""
    orig = ["Mise en place: pica la cebolla y el ajo."]
    trad = ["Mise en place: chop the onion and the garlic."]
    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d["recipe"] == trad, (
        f"con el prefijo intacto la traducción tiene que pasar: {d['recipe']!r}. Si cae "
        f"al español, el fallback está midiendo la línea entera y no el prefijo. "
        f"[{_MARKER}]"
    )


def test_la_puntuacion_francesa_no_cuenta_como_prefijo_conservado(mod) -> None:
    """La tipografía francesa mete un espacio antes de los dos puntos («Mise en place :»)
    y ESO es lo que ponía a fr-FR en 0/12: el regex del frontend exige `place:` pegado.
    """
    orig = ["Mise en place: pica la cebolla."]
    trad = ["Mise en place : hache l'oignon."]
    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d["recipe"] == orig, (
        f"«Mise en place :» con espacio antes de los dos puntos se dio por bueno: "
        f"{d['recipe']!r}. El parser exige `place:` pegado, así que esa línea no se "
        f"reconoce — es el caso que dejaba fr-FR en 0/12. [{_MARKER}]"
    )


def test_un_paso_sin_vocabulario_cerrado_no_se_toca(mod) -> None:
    """El guard sólo protege las líneas etiquetadas. Un paso de prosa normal se traduce
    y punto — el 4,6 % restante del corpus."""
    orig = ["Cuece el arroz blanco según el paquete."]
    trad = ["Cook the white rice according to the package."]
    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d["recipe"] == trad, (
        f"un paso sin etiqueta cayó al español: {d['recipe']!r}. El guard mira el "
        f"prefijo, no la línea. [{_MARKER}]"
    )


# ============================================================
# 3 · Las 4 directivas lo piden explícitamente
# ============================================================


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_la_directiva_pide_conservar_el_vocabulario(mod, locale: str) -> None:
    """El validador es la red; la directiva es lo que evita gastar la llamada para nada.
    Sin pedirlo, cada línea etiquetada vuelve en español y se paga la traducción de un
    texto que se descarta."""
    d = mod._DISPLAY_LANGUAGE_DIRECTIVES[locale]
    assert "Mise en place" in d, (
        f"la directiva de {locale} no nombra las etiquetas de sección. Pedirlo por "
        f"ejemplo concreto es lo que ya funcionó con el nombre canónico del alimento "
        f"(instrucción Y demostración, P1-COACH-LANGUAGE-NATIVE). [{_MARKER}]"
    )
    assert "Nota del nutricionista" in d, (
        f"la directiva de {locale} no nombra las etiquetas de ANOTACIÓN, que son las "
        f"que al perderse convierten una nota en un paso numerado. [{_MARKER}]"
    )


# ============================================================
# 4 · El frontend traduce el rótulo, nunca el identificador
# ============================================================


def _leer_front(rel: str) -> str:
    """Lee un fichero del repo hermano `frontend/`, o SKIPea.

    El `_ROOT / "backend"` del guard no es paranoia decorativa: corriendo desde un
    worktree en `C:/tmp/mf-be-i18n`, `_ROOT` sale `C:/tmp` — donde había un directorio
    `frontend/` viejo, de otra época y sin relación. Sin este check, el test leía ESE
    árbol y reportaba fallos sobre código que nadie despliega. Exigir que el checkout
    sea de verdad `<raíz>/backend` es lo que distingue «el hermano» de «un directorio
    que se llama igual».
    """
    if not (_ROOT / "backend").is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?): no leo un frontend ajeno")
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"{rel} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def test_el_ssot_del_frontend_sigue_casando_espanol() -> None:
    """La otra dirección del contrato, y la que más fácil se rompe «arreglando».

    El dato conserva el prefijo ESPAÑOL, así que el regex no puede migrar a inglés para
    «que case con lo traducido»: dejaría de reconocer los 1.816 pasos que existen hoy y
    las 135 anotaciones volverían a numerarse como acciones de cocina.

    Se mira el SSOT (`utils/recipeSteps.js`), que es donde vive el vocabulario desde
    este P-fix; los componentes ya no llevan copia.
    """
    s = _leer_front("utils/recipeSteps.js")
    for rx in ("mise en place:", "toque de fuego:", "montaje:"):
        assert rx in s.lower(), (
            f"desapareció el patrón «{rx}» de `RECIPE_SECTIONS`. El prefijo español es "
            f"el identificador; lo que se traduce es el rótulo. [{_MARKER}]"
        )
    for rx in ("nota del nutricionista", "seguridad alimentaria"):
        assert rx in s.lower(), (
            f"desapareció el patrón de anotación «{rx}». Sin él, la nota se numera "
            f"como paso de cocina. [{_MARKER}]"
        )


@pytest.mark.parametrize(
    "rel",
    [
        "components/recipes/RecipesView.jsx",
        "components/recipes/MobileRecipes.jsx",
    ],
)
def test_el_titulo_de_seccion_se_pinta_traducido(rel: str) -> None:
    """Hasta ahora el `title` era el literal español, así que incluso el paso que SÍ
    parseaba en inglés mostraba «El Toque de Fuego» en una interfaz inglesa.

    Se ancla la FORMA del arreglo —destructurar `titleKey` y pasarlo por `t()` en el
    render— y no su ausencia: un test que sólo exigiera «no hay literal» pasaría también
    si alguien borrase el título entero.
    """
    s = _leer_front(rel)
    assert "parseRecipeStep(" in s, (
        f"{rel}: ya no usa `parseRecipeStep`. El vocabulario volvió a duplicarse o el "
        f"parseo desapareció. [{_MARKER}]"
    )
    assert re.search(r"titleKey\s*\?\s*t\(\s*titleKey\s*\)", s), (
        f"{rel}: el rótulo no pasa por `t(titleKey)` en el render. Si se traduce en el "
        f"módulo, se congela en el idioma de arranque; si no se traduce, un anglófono "
        f"lee «El Toque de Fuego». [{_MARKER}]"
    )
    assert "const SECTIONS = [" not in s, (
        f"{rel}: reapareció una copia local de `SECTIONS`. El vocabulario cerrado vive "
        f"en `utils/recipeSteps.js`. [{_MARKER}]"
    )


def test_el_ssot_del_frontend_declara_sus_claves() -> None:
    """`titleKey: 'Montaje'` es invisible para el extractor —solo ve `t('literal')`— y
    sale como clave HUÉRFANA en los 4 catálogos, invitando a borrar la traducción que sí
    hace falta. `i18nKey()` la declara sin leer el catálogo, así que puede vivir en
    ámbito de módulo sin congelar nada."""
    s = _leer_front("utils/recipeSteps.js")
    assert s.count("i18nKey(") >= 3, (
        f"las claves de `RECIPE_SECTIONS` no están declaradas con `i18nKey()`: el gate "
        f"las reporta como huérfanas. [{_MARKER}]"
    )
    assert "RECIPE_SECTIONS" in s, "desapareció el SSOT de secciones del frontend"


# «Mise en place» es un préstamo del francés que la cocina profesional usa tal cual en
# inglés, portugués e italiano: dejarlo igual es la traducción correcta, no un olvido.
# Una whitelist sin razón es indistinguible de un descuido, así que va con la suya.
_ROTULOS_IGUALES_A_PROPOSITO = {
    "Mise en place": "préstamo del francés, se usa igual en los 4 idiomas",
}


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_las_etiquetas_de_seccion_estan_en_los_catalogos(locale: str) -> None:
    """El rótulo SÍ se traduce; lo que no se toca es el prefijo del dato."""
    import json

    cat = json.loads(_leer_front(f"i18n/locales/{locale}.json"))

    faltan = [k for k in ("Mise en place", "El Toque de Fuego", "Montaje") if k not in cat]
    assert not faltan, (
        f"{locale}: sin entrada en el catálogo para {faltan}. Sin clave, el rótulo cae "
        f"al español — que es el estado del que venimos. [{_MARKER}]"
    )

    sin_traducir = [
        k for k in ("El Toque de Fuego", "Montaje")
        if k not in _ROTULOS_IGUALES_A_PROPOSITO and cat.get(k) == k
    ]
    assert not sin_traducir, (
        f"{locale}: {sin_traducir} siguen idénticos al español. [{_MARKER}]"
    )
