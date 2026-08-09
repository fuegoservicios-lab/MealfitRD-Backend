"""[P1-MEAL-CARD-ROWS · 2026-08-09] La tarjeta de comida del Dashboard ponía el
texto y los botones a competir por la MISMA fila, y el texto siempre perdía.

MEDIDO antes del arreglo:

  · El cluster de acciones ocupa 310 px y el coste fijo de la fila (cluster +
    padding de la tarjeta + gap de la rejilla) 446 px.
  · La rejilla era `1fr auto`: la columna de botones se dimensiona por su
    contenido, así que la columna elástica —el texto— absorbía el 100 % de
    cualquier recorte. Hacían falta 746 px de tarjeta para que la descripción
    alcanzara 40 caracteres por línea.
  · A ~600 px de tarjeta el párrafo caía a 155 px y se leía en una columna de
    DIEZ líneas cortas.

El owner lo reportó como «los botones ocupan mucho espacio y por eso el texto
se ve encogido» — que es exactamente el mecanismo.

Tras el arreglo (mismas mediciones, tarjeta de 600 px): 488 px y 4 líneas. Y
entre 46 y 68 caracteres por línea a CUALQUIER ancho, porque el tope de medida
cierra el problema contrario (sin él, a 1000 px la línea llegaba a ~123).

Tooltip-anchor: P1-MEAL-CARD-ROWS
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def _src() -> str:
    return _DASH.read_text(encoding="utf-8")


def _rule(css: str, selector: str) -> str | None:
    m = re.search(rf"^\s*\.{selector}\s*\{{([^}}]*)\}}", css, re.MULTILINE)
    return m.group(1) if m else None


def test_the_card_stacks_instead_of_competing_for_one_row():
    """`1fr auto` es la causa raíz: con la columna de botones dimensionada por
    su contenido, la del texto absorbe todo el recorte."""
    css = _src()
    block = _rule(css, "meal-card")
    assert block, "P1-MEAL-CARD-ROWS: no encuentro la regla .meal-card"
    assert "grid-template-columns: 1fr;" in block, (
        "P1-MEAL-CARD-ROWS: `.meal-card` debe ser de UNA columna. Si vuelve a "
        "`1fr auto`, el bloque de acciones (310 px) vuelve a robarle el ancho al "
        "párrafo y hacen falta 746 px de tarjeta para leerlo."
    )
    assert "1fr auto" not in block, (
        "P1-MEAL-CARD-ROWS: `.meal-card` volvió a dos columnas."
    )


def test_the_kcal_lives_in_the_header_not_in_the_actions():
    """Las kcal son metadato del plato, no una acción. Mientras vivían en la
    columna de acciones OBLIGABAN a que esa columna existiera al lado del
    texto — eran la mitad del problema, aunque no ocupen casi ancho."""
    css = _src()
    assert _rule(css, "meal-head") is not None, (
        "P1-MEAL-CARD-ROWS: falta `.meal-head` (rótulo + título | kcal)."
    )
    assert _rule(css, "meal-kcal") is not None, (
        "P1-MEAL-CARD-ROWS: falta `.meal-kcal`."
    )
    jsx = _src()
    head = jsx.index('className="meal-head"')
    kcal = jsx.index('className="meal-kcal"')
    actions = jsx.index('className="meal-actions"')
    assert head < kcal < actions, (
        "P1-MEAL-CARD-ROWS: la cifra de kcal debe renderizarse DENTRO de la "
        "cabecera y ANTES de la fila de acciones. Si vuelve a bajar con los "
        "botones, vuelve la columna lateral que estrujaba el texto."
    )


def test_the_actions_are_a_row_of_their_own():
    css = _src()
    block = _rule(css, "meal-actions")
    assert block, "P1-MEAL-CARD-ROWS: no encuentro la regla .meal-actions"
    assert "border-top" in block, (
        "P1-MEAL-CARD-ROWS: `.meal-actions` pierde su hairline superior — es lo "
        "que separa la lectura de la acción."
    )
    assert _rule(css, "meal-actions-row") is not None, (
        "P1-MEAL-CARD-ROWS: falta `.meal-actions-row`."
    )


def test_the_paragraph_has_a_reading_measure_cap():
    """LA REGLA QUE MÁS SE VA A EROSIONAR: `max-width` en un párrafo que acaba
    de ganar ancho parece contradictorio y es justo lo que lo hace legible.
    Sin tope, a 1000 px de tarjeta la línea llega a ~123 caracteres — un
    renglón demasiado largo cansa igual que uno demasiado corto."""
    css = _src()
    block = _rule(css, "meal-desc")
    assert block, (
        "P1-MEAL-CARD-ROWS: no encuentro `.meal-desc`. El párrafo necesita tope "
        "de medida o cambiamos un problema por el contrario."
    )
    assert "max-width" in block, (
        "P1-MEAL-CARD-ROWS: `.meal-desc` perdió su `max-width`. Medido: sin él, "
        "a 1000 px de tarjeta la línea llega a ~123 caracteres (el rango legible "
        "es 45-75)."
    )
    assert 'className="meal-desc"' in _src(), (
        "P1-MEAL-CARD-ROWS: el <p> de la descripción dejó de consumir `.meal-desc`."
    )


def test_the_old_side_column_is_fully_gone():
    """Cuatro reglas de `<=768px` reconstruían A MANO esta misma fila solo en
    móvil. Se fueron con la clase: dejar la vieja invita a recablear la columna
    lateral que este P-fix eliminó.

    ESCANEA EL FICHERO ENTERO, comentarios incluidos. El primer intento traía
    un filtro que descartaba líneas de comentario y falló por lo mismo de
    siempre: las continuaciones de un bloque no empiezan por el carácter que
    el filtro busca. La convención del repo es la contraria y es más simple —
    al documentar lo que se borró, DESCRÍBELO, no lo cites. Los comentarios de
    este P-fix hablan de «la vieja columna lateral» justamente por esto."""
    css = _src()
    assert "meal-right-side" not in css, (
        "P1-MEAL-CARD-ROWS: reapareció el nombre de la columna lateral que este "
        "P-fix eliminó. Si es una regla o markup, la columna volvió y con ella el "
        "estrujado del texto. Si es solo un comentario, descríbela en vez de "
        "citarla: este test escanea el fichero entero a propósito."
    )


def test_no_backticks_inside_the_meal_card_style_literal():
    """El CSS de las tarjetas vive dentro de un template literal
    `<style>{` … `}</style>`. Un backtick en un comentario CSS lo TERMINA y el
    fichero deja de parsear — me pasó escribiendo este mismo P-fix, y el error
    (`Expected "}" but found "grid"`) no señala al backtick sino al CSS que
    queda huérfano detrás, así que cuesta leerlo."""
    css = _src()
    ini = css.index(".meal-card {")
    fin = css.index(".plan-tier-badge {")
    assert "`" not in css[ini:fin], (
        "P1-MEAL-CARD-ROWS: hay un backtick dentro del template literal del "
        "<style> de las meal-cards. Cierra el literal y rompe el parseo del "
        "fichero entero. Usa comillas angulares para citar selectores ahí."
    )
