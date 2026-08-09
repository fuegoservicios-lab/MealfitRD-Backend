"""[P1-BRANDS-UNMATCHED-VISIBLE · 2026-08-09] «Marcas del súper · 60/61 con
opciones» — el owner preguntó por qué el número no cuadraba.

FORENSE (2026-08-09, contra el endpoint real de producción con su lista real):
el número era CORRECTO. Sus 61 nombres únicos contra un catálogo de 1.739 filas
activas dan 60 con opciones y 1 sin: **Edamame**, que existe en
`master_ingredients` (204 alimentos, el catálogo del que el generador arma
platos) pero NO en `supermarket_products` (251 alimentos). El plan puede
recetarlo; el súper no tiene presentación que ofrecer.

EL DEFECTO NO ERA EL NÚMERO, ERA QUE NO SE PODÍA INSPECCIONAR. El panel solo
renderizaba `matchedNames`, así que el ítem que falta no aparecía en NINGÚN
sitio de la UI. El contador anunciaba un hueco y no daba forma de ver cuál —
por eso hubo que preguntar en vez de mirar.

Un número que declara una diferencia tiene que dejar ver de qué está hecha esa
diferencia; si no, solo genera dudas.

Tooltip-anchor: P1-BRANDS-UNMATCHED-VISIBLE
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BRANDS = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "SupermarketBrands.jsx"


def _src() -> str:
    return _BRANDS.read_text(encoding="utf-8")


def test_the_unmatched_items_are_derived():
    """El complemento de `matchedNames` tiene que existir como valor, no como
    resta mental del usuario."""
    src = _src()
    assert "unmatchedNames" in src, (
        "P1-BRANDS-UNMATCHED-VISIBLE: falta `unmatchedNames`. El contador dice "
        "N/M; sin el complemento, la diferencia no es inspeccionable."
    )
    m = re.search(r"const unmatchedNames\s*=\s*([^;]+);", src, re.DOTALL)
    assert m, "P1-BRANDS-UNMATCHED-VISIBLE: no encuentro la derivación de unmatchedNames"
    body = m.group(1)
    assert "matches" in body, (
        "P1-BRANDS-UNMATCHED-VISIBLE: `unmatchedNames` debe derivarse de los "
        "`matches` reales del endpoint, no de una lista aparte que pueda drift-ear "
        "respecto al numerador del contador."
    )


def test_the_unmatched_items_are_rendered():
    """Derivarlos y no pintarlos deja el defecto igual."""
    src = _src()
    usos = src.count("unmatchedNames")
    assert usos >= 2, (
        f"P1-BRANDS-UNMATCHED-VISIBLE: `unmatchedNames` aparece {usos} vez/veces. "
        "Se espera al menos la derivación y su render — calcularlo sin mostrarlo "
        "no cierra nada."
    )
    assert re.search(r"unmatchedNames\.(join|map)\(", src), (
        "P1-BRANDS-UNMATCHED-VISIBLE: los nombres sin opciones deben pintarse "
        "(join o map). El usuario tiene que poder LEER cuál falta."
    )


def test_the_panel_no_longer_hides_the_gap_when_nothing_matches():
    """El caso extremo (0 con opciones) ya tenía copy propio; debe seguir
    existiendo y no quedar tapado por el bloque nuevo."""
    src = _src()
    assert "matchedNames.length === 0" in src, (
        "P1-BRANDS-UNMATCHED-VISIBLE: desapareció la rama de «ninguna variante "
        "cargada». Es el caso en que el panel no tiene nada que ofrecer y hay que "
        "decirlo."
    )
