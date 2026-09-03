"""[P1-TRACKING-MACROS-CONTAINER · 2026-08-14] La rejilla de macros se decide por
el ancho del PANEL, no por el del viewport.

EL SÍNTOMA. Con la generación de planes en pausa (modo contador), la celda de
Carbohidratos del panel «Progreso en Tiempo Real» se partía: la etiqueta arriba,
el «0 / 269 g» debajo, la barra más abajo — mientras Proteína y Grasas quedaban
enteras. El dueño lo reportó como «la barra de carbohidratos se ve diferente».

LA CADENA. `.macroGrid` es `repeat(3, 1fr)` en escritorio y `.barHeader` lleva
`flex-wrap: wrap`. En modo PLAN el panel vive en una columna ancha y cada celda
tiene sitio para «Carbohidratos» + su valor. En modo CONTADOR el layout
(`DashboardTracking.module.css`) es `minmax(0,1fr) 320px`: el panel pierde 320px
de barra lateral, cada celda queda en ~200px, y la única etiqueta larga de las
tres es la que envuelve. Proteína y Grasas caben — por eso solo se rompía una.

POR QUÉ NO BASTABA EL BREAKPOINT EXISTENTE. P1-MACRO-BARS-UNIFORM ya apila la
rejilla… bajo `@media (max-width: 768px)`, que mide el VIEWPORT. Aquí el viewport
es un monitor de sobra de ancho; lo estrecho es el PANEL. Es la lección que este
repo ya pagó una vez: «un banco sin la cadena de contenedores mide otra
pantalla». La medida que corresponde es la del propio contenedor — y el repo ya
usa container queries para exactamente esto (WaterTracker, la Nevera).

POR QUÉ NO SE REINTRODUCE la etiqueta corta («Carbos»): P1-MACRO-BARS-UNIFORM la
retiró a propósito junto con la disposición que la pedía. Resucitarla contradiría
esa decisión; apilar cuando no cabe es la MISMA salida que aquel fix eligió para
el teléfono, extendida al caso que su premisa no cubría.

Tooltip-anchor: P1-TRACKING-MACROS-CONTAINER
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CSS = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.module.css"
)


def _read() -> str:
    if not _CSS.exists():
        pytest.fail("[P1-TRACKING-MACROS-CONTAINER] No existe TrackingProgress.module.css")
    return _CSS.read_text(encoding="utf-8")


def _sin_comentarios(t: str) -> str:
    return re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)


def test_el_panel_es_un_contenedor_medible():
    css = _sin_comentarios(_read())
    m = re.search(r"\.card\s*\{[^}]*container-type:\s*inline-size", css)
    assert m, (
        "[P1-TRACKING-MACROS-CONTAINER] `.card` no declara `container-type: inline-size`.\n"
        "Sin él, la rejilla de macros solo puede reaccionar al VIEWPORT — y el bug "
        "ocurre con un monitor ancho y un panel estrecho (el modo contador le quita "
        "320px de barra lateral). Mismo patrón que WaterTracker y la Nevera."
    )


def test_la_rejilla_apila_cuando_el_PANEL_es_estrecho():
    css = _sin_comentarios(_read())
    contenedor = re.search(
        r"@container\s*\(max-width:\s*(\d+)px\)\s*\{[^@]*?\.macroGrid\s*\{[^}]*grid-template-columns:\s*1fr\b",
        css, re.DOTALL,
    )
    assert contenedor, (
        "[P1-TRACKING-MACROS-CONTAINER] No hay `@container` que apile `.macroGrid` "
        "cuando el panel es estrecho.\n"
        "En modo contador cada celda queda en ~200px y «Carbohidratos» + su valor "
        "no caben: el `flex-wrap` parte la única celda con etiqueta larga y las "
        "tres barras dejan de verse iguales."
    )
    umbral = int(contenedor.group(1))
    assert 700 <= umbral <= 860, (
        f"[P1-TRACKING-MACROS-CONTAINER] Umbral {umbral}px fuera de rango.\n"
        "Medido: cada celda necesita ~240px para la etiqueta más larga + valor, y "
        "con dos gaps de 2rem son ~780px de panel. Por debajo de 700 el bug del "
        "modo contador (~700px de panel) seguiría vivo; por encima de 860 apilaría "
        "también el modo plan, que sí cabe en tres columnas."
    )


def test_el_breakpoint_de_viewport_sigue_para_el_telefono():
    """El @media de 768px cubre lo que el contenedor no puede (header, paddings)."""
    css = _sin_comentarios(_read())
    assert "@media (max-width: 768px)" in css, (
        "[P1-TRACKING-MACROS-CONTAINER] Desapareció el breakpoint móvil de "
        "P1-MACRO-BARS-UNIFORM. El @container lo complementa, no lo sustituye: el "
        "resto de ajustes móviles (header, paddings) siguen midiendo viewport."
    )


def test_no_vuelve_el_par_de_etiquetas_larga_corta():
    """La salida es apilar, no resucitar «Carbos»."""
    jsx = (_CSS.parent / "TrackingProgress.jsx").read_text(encoding="utf-8")
    codigo = re.sub(r"/\*.*?\*/", "", jsx, flags=re.DOTALL)
    codigo = re.sub(r"^\s*//.*$", "", codigo, flags=re.MULTILINE)
    assert '"Carbos"' not in codigo and "'Carbos'" not in codigo, (
        "[P1-TRACKING-MACROS-CONTAINER] Volvió la etiqueta corta «Carbos».\n"
        "P1-MACRO-BARS-UNIFORM retiró ese par a propósito («la uniformidad ES la "
        "decisión»). Cuando la celda no da para la etiqueta larga, la salida es "
        "APILAR la rejilla — la misma que aquel fix eligió para el teléfono."
    )
