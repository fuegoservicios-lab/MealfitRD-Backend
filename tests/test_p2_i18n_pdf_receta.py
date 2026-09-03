"""[P2-I18N-PDF-RECETA · 2026-08-21] El PDF salía en español mientras su propia pantalla
salía traducida.

`Recipes.jsx` calcula `_activeDisplay = mealDisplay(activeMealRaw, locale)` y pinta ESO.
`handleDownloadPDF` recibía `activeMealRaw` — el meal español crudo. Así que el usuario
leía la receta en francés, pulsaba «Descargar PDF» en el mismo botón de esa pantalla, y
el documento salía en español.

El diferimiento que lo justificaba apuntaba a una task que solo cubría el PDF de la LISTA
DE COMPRAS: una nota correcta sobre otra cosa, que aquí se leyó como permiso.

DÓNDE VA LA TRADUCCIÓN, y esto es lo que el test ancla: DENTRO del handler, no en el call
site. El handler es el ACTO, así que cualquier botón futuro que lo invoque queda cubierto
sin wiring — la misma lección que `P2-DISPLAY-POP-VECINO`. Un `handleDownloadPDF(activeMeal)`
en el call site arreglaría el botón de hoy y dejaría el siguiente roto.

`recipe` TAMBIÉN, y ahí está la diferencia con la vista: la pantalla pasa los pasos
aparte (`activeRecipeSteps`), pero `generateRecipeHTML` los lee de `meal.recipe`. Sin esa
línea el PDF saldría con nombre e ingredientes traducidos y la receta en español — peor
que el estado de partida, porque parece un fallo en vez de una decisión.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P2-I18N-PDF-RECETA"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_RECIPES = _ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"


def _src() -> str:
    if not (_ROOT / "backend").is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    if not _RECIPES.exists():
        pytest.skip("Recipes.jsx no existe en este checkout (repos hermanos)")
    return _RECIPES.read_text(encoding="utf-8")


def _sin_comentarios(js: str) -> str:
    """Quita `//…` y `/*…*/`.

    NO es cosmética. La primera versión de este guard comprobaba
    `"mealDisplay(" in cuerpo` y pasaba aunque el código estuviera mutado, porque el
    bloque explicativo que vive DENTRO del handler nombra `mealDisplay(activeMealRaw,
    locale)` al contar cuál era el defecto. Comentario-vence-guard: este repo lo ha
    pagado siete veces en dos días, y varias con el comentario escrito por quien
    escribía el guard.

    Basta con un barrido léxico: aquí no hay literales de cadena con `//` dentro (se
    comprobó), y un parser JS sería desproporcionado para lo que este fichero mide.
    """
    js = re.sub(r"/\*.*?\*/", " ", js, flags=re.S)
    return re.sub(r"//[^\n]*", " ", js)


def _cuerpo_del_handler(src: str) -> str:
    i = src.find("const handleDownloadPDF")
    assert i != -1, f"no encontré `handleDownloadPDF` en Recipes.jsx [{_MARKER}]"
    # Hasta la primera llamada a `generateRecipeHTML`: es donde el meal ya tiene que
    # estar traducido, porque a partir de ahí se compone el documento.
    j = src.index("generateRecipeHTML(", i)
    return _sin_comentarios(src[i:j])


def test_el_handler_traduce_el_meal_antes_de_componer_el_html() -> None:
    cuerpo = _cuerpo_del_handler(_src())
    assert "mealDisplay(" in cuerpo, (
        f"`handleDownloadPDF` no llama a `mealDisplay`: sigue componiendo el PDF con el "
        f"meal español crudo mientras la pantalla de la que cuelga el botón está "
        f"traducida. [{_MARKER}]"
    )
    assert "locale" in cuerpo, (
        f"`mealDisplay` se llama sin el locale activo. [{_MARKER}]"
    )


@pytest.mark.parametrize("campo", ["name", "desc", "ingredients", "recipe"])
def test_los_cuatro_campos_viajan_traducidos(campo: str) -> None:
    """`recipe` es el que se olvida: la pantalla pasa los pasos aparte
    (`activeRecipeSteps`) pero `generateRecipeHTML` los lee de `meal.recipe`. Sin él, el
    PDF sale con el nombre traducido y la receta en español — parece un fallo."""
    cuerpo = _cuerpo_del_handler(_src())
    assert re.search(rf"\b{campo}:\s*_d\.", cuerpo), (
        f"`{campo}` no se toma de la capa `_display` dentro del handler. [{_MARKER}]"
    )


def test_la_traduccion_vive_en_el_handler_y_no_en_el_call_site() -> None:
    """El handler es el ACTO. Traducir en el call site arregla el botón de hoy y deja
    roto el siguiente — la lección de `P2-DISPLAY-POP-VECINO`, y antes la de
    `P1-COUNTRY-SYSTEM-F1`."""
    src = _sin_comentarios(_src())
    llamadas = re.findall(r"handleDownloadPDF\((\w+)\)", src)
    assert llamadas, f"no encontré ninguna invocación de `handleDownloadPDF` [{_MARKER}]"
    for arg in llamadas:
        assert "Raw" in arg or arg == "meal", (
            f"un call site pasa `{arg}`, que parece ya traducido. La traducción tiene "
            f"que hacerla el handler: si la hace el caller, el próximo botón que se "
            f"añada nace en español. [{_MARKER}]"
        )


def test_el_rotulo_de_seccion_del_pdf_se_traduce() -> None:
    """Con el documento ya traducido, dejar «El Toque de Fuego» en medio de un PDF en
    francés es el documento mestizo que la nota anterior quería evitar."""
    src = _sin_comentarios(_src())
    assert re.search(r"const sectionTitle = titleKey \? t\(titleKey\)", src), (
        f"el rótulo de sección del PDF no pasa por `t()`. [{_MARKER}]"
    )


def test_el_prefijo_del_dato_sigue_siendo_espanol() -> None:
    """LA MITAD QUE NO SE MUEVE. El PDF reconoce la sección por el prefijo ESPAÑOL del
    dato (`utils/recipeSteps.js`); lo que se traduce es el rótulo. Si alguien migra el
    reconocimiento al texto traducido, deja de reconocer los 1.816 pasos que existen."""
    src = _sin_comentarios(_src())
    assert "parseRecipeStep(step)" in src, (
        f"el PDF ya no resuelve la sección con `parseRecipeStep`, que casa el prefijo "
        f"español del SSOT. [{_MARKER}]"
    )
    m = re.search(r"const _COLOR_SECCION = \{(.*?)\}", src, re.S)
    assert m, f"no encontré el mapa de colores del PDF [{_MARKER}]"
    for clave in ("Mise en place", "El Toque de Fuego", "Montaje"):
        assert f"'{clave}'" in m.group(1), (
            f"el mapa de colores perdió «{clave}»: está indexado por la CLAVE española, "
            f"no por el rótulo traducido. Si se traduce, el color se cae al de por "
            f"defecto en los cuatro idiomas. [{_MARKER}]"
        )
