"""[P1-HOWITWORKS-ALIGN · 2026-08-09] La hoja de proceso «02 / CÓMO SE CALCULA»
tenía dos desalineaciones, y el dueño solo veía la pequeña.

MEDIDO en navegador, antes del arreglo:

  · `.figureBox` de la celda 01 = 165 px; el de sus tres hermanas = 64 px. La
    diferencia eran 4 rótulos que `ProfileFigure` devolvía DENTRO del mismo
    contenedor que el SVG. Efecto: su título caía 101 px por debajo de los
    otros tres. Era la única de las cuatro figuras con rótulos — de ahí que
    fuese «la única que se ve diferente».

  · Las 4 mini-cotas caían a 4 alturas distintas: 124 px entre la más alta y
    la más baja, porque cada una flotaba detrás de una descripción de largo
    distinto. Con las 4 celdas midiendo lo mismo (428 px) y sitio de sobra.
    Esta era la PEOR: la del título afectaba a una celda, esta a las cuatro, y
    es la que dejaba el bloque sin rematar por abajo.

En una hoja de proceso la alineación no es cosmética, es la lectura: cuatro
cotas a cuatro alturas se leen como cuatro fichas sueltas, no como una hoja.

Tooltip-anchor: P1-HOWITWORKS-ALIGN
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HOME = _REPO_ROOT / "frontend" / "src" / "components" / "home"
_JSX = _HOME / "HowItWorks.jsx"
_CSS = _HOME / "HowItWorks.module.css"


def test_the_four_figures_are_peers():
    """`ProfileFigure` debe devolver UN solo <svg>, no un fragmento con el SVG
    más una lista. Cualquier hijo extra vuelve a inflar `.figureBox` y a
    desalinear el título de esa celda respecto a las otras tres."""
    jsx = _JSX.read_text(encoding="utf-8")
    body = re.search(r"function ProfileFigure\(\)\s*\{(.*?)\n\}", jsx, re.DOTALL)
    assert body, "P1-HOWITWORKS-ALIGN: no encuentro ProfileFigure en HowItWorks.jsx"
    src = body.group(1)
    assert "<ul" not in src, (
        "P1-HOWITWORKS-ALIGN: ProfileFigure volvió a renderizar una lista. Los 4 "
        "nombres de entrada son CONTENIDO (`STEPS[0].inputs`), no rotulación del "
        "dibujo: dentro de la figura miden 101 px que las otras tres celdas no "
        "tienen, y le hunden el título por esa misma cantidad."
    )
    assert "<>" not in src and "</>" not in src, (
        "P1-HOWITWORKS-ALIGN: ProfileFigure volvió a devolver un fragmento. Debe "
        "devolver el <svg> a secas — es la única forma de que las 4 figuras midan "
        "lo mismo y los 4 títulos caigan en la misma línea."
    )


def test_the_inputs_are_content_and_only_the_first_cell_has_them():
    jsx = _JSX.read_text(encoding="utf-8")
    assert "inputs:" in jsx, (
        "P1-HOWITWORKS-ALIGN: `STEPS[0].inputs` desapareció. Los 4 nombres de "
        "entrada son lo más concreto de la sección; no se borran al mudarlos."
    )
    assert jsx.count("inputs:") == 1, (
        "P1-HOWITWORKS-ALIGN: más de un paso declara `inputs`. La lista solo la "
        "tiene la celda 01; si otra la necesita, revisa antes que no vuelva a "
        "desalinear la rejilla."
    )
    assert "s.inputs &&" in jsx, (
        "P1-HOWITWORKS-ALIGN: la lista debe renderizarse condicionalmente, o las "
        "otras tres celdas emitirían un <ul> vacío que sí ocupa caja."
    )
    # Los 4 literales sobreviven a la mudanza.
    for label in ("PRESUPUESTO RD$", "ALERGIAS", "CONDICIÓN CLÍNICA", "LO QUE HAY EN LA NEVERA"):
        assert label in jsx, f"P1-HOWITWORKS-ALIGN: se perdió el literal `{label}`."


def test_the_cota_is_anchored_to_the_foot_of_the_cell():
    """LA REGLA QUE MÁS SE VA A EROSIONAR: `margin-top: auto` parece un valor
    arbitrario y es lo que alinea las cuatro cotas. Quien lo cambie por un rem
    concreto «para dar aire» devuelve las 4 alturas distintas."""
    css = _CSS.read_text(encoding="utf-8")
    cota = re.search(r"^\.miniCota\s*\{([^}]*)\}", css, re.MULTILINE)
    assert cota, "P1-HOWITWORKS-ALIGN: no encuentro la regla .miniCota"
    assert "margin-top: auto" in cota.group(1), (
        "P1-HOWITWORKS-ALIGN: `.miniCota` perdió `margin-top: auto`. Sin él la "
        "cota vuelve a flotar detrás de su descripción y las 4 caen a 4 alturas "
        "distintas (124 px de rango, medido)."
    )
    text = re.search(r"^\.cellText\s*\{([^}]*)\}", css, re.MULTILINE)
    assert text, "P1-HOWITWORKS-ALIGN: no encuentro la regla .cellText"
    block = text.group(1)
    assert "display: flex" in block and "flex-direction: column" in block, (
        "P1-HOWITWORKS-ALIGN: `.cellText` debe ser flex column — es lo que le da "
        "a `margin-top: auto` de la cota un eje contra el que resolverse."
    )
    desc = re.search(r"^\.cellDesc\s*\{([^}]*)\}", css, re.MULTILINE)
    assert desc and re.search(r"margin:[^;]*\s\S+\s+\S+;", desc.group(1)), (
        "P1-HOWITWORKS-ALIGN: `.cellDesc` necesita margen inferior — es el SUELO "
        "del hueco: cuando la descripción llega hasta abajo, el `auto` de la cota "
        "se resuelve a 0 y sin él quedarían pegadas."
    )


def test_the_old_legend_class_is_fully_gone():
    """Una clase muerta en la hoja invita a recablearla al sitio del que se la
    sacó."""
    css = _CSS.read_text(encoding="utf-8")
    assert not re.search(r"^\s*\.figLegend\s*[,{]", css, re.MULTILINE), (
        "P1-HOWITWORKS-ALIGN: `.figLegend` sigue declarada en el .module.css. Se "
        "renombró a `.inputList` al mudarse; dejar la vieja invita a devolverla "
        "dentro de `.figureBox`."
    )
    jsx = _JSX.read_text(encoding="utf-8")
    assert "styles.figLegend" not in jsx, (
        "P1-HOWITWORKS-ALIGN: HowItWorks.jsx sigue consumiendo `styles.figLegend`."
    )
