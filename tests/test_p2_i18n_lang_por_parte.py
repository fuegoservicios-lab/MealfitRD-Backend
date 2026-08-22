"""[P2-I18N-LANG-POR-PARTE · 2026-08-21] WCAG 3.1.2 — el contenido español sin marcar
bajo `<html lang="fr-FR">`.

MEDIDO: **un solo** `lang=` en todo `frontend/src`, y es el de los nombres nativos del
selector de idioma. CERO en cualquier CONTENIDO.

Así que un lector de pantalla en francés sintetiza «Pollo guisado con arroz blanco» con
fonética francesa. No es que suene raro: es ininteligible — el usuario ciego no puede
leer su propio plan. Y axe no puede detectarlo: no existe forma automática de saber que
un texto no está en el idioma que declara su ancestro. Por eso hace falta un test que
mire el código.

EL MATIZ ES LOAD-BEARING, y es lo que hace que esto no sea «meter `lang="es"` por todas
partes»: se marca POR PARTE. Un `lang` de bloque sería INCORRECTO en la línea bilingüe de
la lista de compras («Black beans (Habichuelas negras)»), donde haría pronunciar «Black
beans» a la española — el mismo defecto del revés.

Y sólo se marca lo que HACE FALTA. Cuando el campo sí vino traducido, heredar
`<html lang>` es lo correcto; un `lang` redundante es ruido que además se queda obsoleto
en cuanto cambie la traducción.

LO QUE NO ENTRA, y por qué no es un olvido: el PDF. `html2pdf` rasteriza con html2canvas
y embebe una imagen — no hay capa de texto ni árbol de accesibilidad, así que un `lang`
ahí no lo lee nadie. Es también la razón de que `partesDeLineaDeCompra` se retirara antes
de nacer: su único consumidor posible vivía en ese generador.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P2-I18N-LANG-POR-PARTE"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_SRC = _ROOT / "frontend" / "src"


def _leer(rel: str) -> str:
    if not (_ROOT / "backend").is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    p = _SRC / rel
    if not p.exists():
        pytest.skip(f"{rel} no existe en este checkout")
    return p.read_text(encoding="utf-8")


# ============================================================
# 1 · El SSOT decide, y decide lo mismo que `mealDisplay`
# ============================================================

def test_existe_el_helper() -> None:
    src = _leer("utils/displayMeal.js")
    assert "export function langDeCampo(" in src, (
        f"No existe `langDeCampo`. Sin un SSOT, cada pantalla decide por su cuenta en "
        f"qué idioma está su texto — y las que no decidan seguirán mintiendo. [{_MARKER}]"
    )


def test_el_helper_no_marca_cuando_la_interfaz_ya_esta_en_espanol() -> None:
    """En español la interfaz y el contenido coinciden: marcar no aporta nada y ensucia
    el DOM de todos los usuarios dominicanos, que son la mayoría."""
    src = _leer("utils/displayMeal.js")
    cuerpo = src[src.index("export function langDeCampo("):]
    assert "locale.startsWith('es')" in cuerpo, (
        f"`langDeCampo` no corta en seco para los locales españoles. [{_MARKER}]"
    )


def test_el_helper_usa_la_misma_regla_de_aceptacion_que_mealDisplay() -> None:
    """LA MITAD QUE IMPORTA. Si `langDeCampo` y `mealDisplay` divergen, el `lang` dice una
    cosa y el texto pintado es otra — y eso es PEOR que no marcar, porque el lector de
    pantalla obedece la marca.

    Las dos reglas: un string no vacío para `name`/`description`; un array de la MISMA
    longitud para `recipe`/`ingredients` (el espejo es por índice).
    """
    src = _leer("utils/displayMeal.js")
    cuerpo = src[src.index("export function langDeCampo("):]
    assert "_isNonEmptyString(entry.name)" in cuerpo, (
        f"`langDeCampo` no aplica a `name` la misma regla que `mealDisplay`. [{_MARKER}]"
    )
    assert "traducido.length === original.length" in cuerpo, (
        f"`langDeCampo` no comprueba la LONGITUD de los arrays. `mealDisplay` sí, así que "
        f"habría casos donde el texto cae al español y el `lang` dice que no. [{_MARKER}]"
    )


# ============================================================
# 2 · Está aplicado donde un lector de pantalla lee
# ============================================================

@pytest.mark.parametrize(
    "rel,nodo",
    [
        ("pages/Dashboard.jsx", "name"),
        ("components/recipes/RecipesView.jsx", "name"),
        ("components/recipes/RecipesView.jsx", "desc"),
        ("components/recipes/RecipesView.jsx", "recipe"),
        ("components/recipes/MobileRecipes.jsx", "name"),
        ("components/recipes/MobileRecipes.jsx", "desc"),
        ("components/recipes/MobileRecipes.jsx", "recipe"),
    ],
)
def test_las_superficies_de_pantalla_marcan_su_idioma(rel: str, nodo: str) -> None:
    src = _leer(rel)
    patron = (rf"lang=\{{langDeCampo\([^)]*'{nodo}'" if "Dashboard" in rel
              else rf"lang=\{{langs\?\.{nodo}")
    assert re.search(patron, src), (
        f"{rel}: el nodo de `{nodo}` no declara su idioma. Bajo `<html lang=\"fr-FR\">` "
        f"un lector de pantalla lo sintetiza con fonética francesa. [{_MARKER}]"
    )


def test_el_lang_es_condicional_y_nunca_pinta_undefined() -> None:
    """`lang={x}` con `x === null` pinta el atributo con la cadena «null» en algunas
    versiones de React, y `undefined` lo omite. La diferencia la ve el lector de
    pantalla."""
    for rel in ("pages/Dashboard.jsx", "components/recipes/RecipesView.jsx",
                "components/recipes/MobileRecipes.jsx"):
        src = _leer(rel)
        for m in re.finditer(r"lang=\{([^}]+)\}", src):
            expr = m.group(1)
            assert "undefined" in expr, (
                f"{rel}: `lang={{{expr}}}` no cae a `undefined`. Un `lang` con valor nulo "
                f"puede acabar pintado como atributo literal. [{_MARKER}]"
            )


def test_el_caller_calcula_los_langs_porque_la_vista_no_puede() -> None:
    """Las vistas de recetas reciben el meal YA traducido, así que desde dentro no hay
    forma de saber qué campo cayó al español. Si alguien mueve el cálculo ahí dentro, la
    marca deja de corresponderse con el texto."""
    src = _leer("pages/Recipes.jsx")
    assert "langs: {" in src and "langDeCampo(activeMealRaw" in src, (
        f"`Recipes.jsx` no calcula `langs` sobre el meal CRUDO. [{_MARKER}]"
    )
    for rel in ("components/recipes/RecipesView.jsx", "components/recipes/MobileRecipes.jsx"):
        assert "langDeCampo(" not in _leer(rel), (
            f"{rel} llama a `langDeCampo` por su cuenta, y ahí el meal ya viene "
            f"traducido: la respuesta sería siempre «no hace falta marcar». [{_MARKER}]"
        )


def test_el_hueco_del_pdf_esta_declarado() -> None:
    """LO QUE NO ENTRA, escrito. Un alcance que se decide y no se anota vuelve como
    hallazgo en la siguiente auditoría — y esta vez se sabe POR QUÉ: html2pdf rasteriza."""
    src = _leer("utils/displayMeal.js")
    assert "rasteriza" in src, (
        f"no está declarado por qué el PDF queda fuera. Sin esa razón escrita, el "
        f"siguiente auditor lo cuenta como olvido. [{_MARKER}]"
    )
