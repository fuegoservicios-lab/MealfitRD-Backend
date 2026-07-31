"""[P2-CARDS-EQUAL-COLUMNS + P2-CTA-PRIMARY-DARK · 2026-07-31] Dos trampas de
CSS que se vieron en producción en las tarjetas de planes.

1. COLUMNAS DESIGUALES. `grid-template-columns: repeat(4, 1fr)` parece repartir
   a partes iguales, pero `1fr` es `minmax(AUTO, 1fr)`: el suelo de cada
   columna es el ancho de su contenido más indivisible. El pill de lanzamiento
   llevaba `white-space: nowrap` con el texto "PRECIO DE LANZAMIENTO · SUBE EL
   15 DE SEPTIEMBRE", así que las 3 columnas de pago impusieron un suelo enorme
   y dejaron la tarjeta Gratis a media anchura (medido: 188px contra 340px), con
   su texto partido en 3 líneas. `minmax(0, 1fr)` fija el suelo en 0 → columnas
   idénticas pase lo que pase dentro.

2. CTA PRIMARIO APAGADO EN OSCURO. `:global(html[data-theme="dark"]) .planButton`
   le gana por especificidad a `.planButtonPrimary`, así que el botón de la
   tarjeta destacada —al que apunta la página entera— perdía su gradiente y se
   veía igual que los secundarios. Medido con getComputedStyle sobre el MISMO
   elemento: claro = `linear-gradient(...)`, oscuro = `rgb(30, 41, 59)` plano.
   El arreglo repite el estilo con la especificidad del tema (la misma cura que
   este CSS ya documentaba para `.planCardPopular`).

Parser-based sobre el CSS: no hay navegador en CI. Ancla las dos REGLAS, no el
píxel.
"""
from __future__ import annotations

import re
from pathlib import Path

_CSS = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend" / "src" / "pages" / "Upgrade.module.css"
).read_text(encoding="utf-8")


def _block(selector: str) -> str:
    """Cuerpo de la primera regla cuyo selector coincide literalmente, SIN
    comentarios. Sin el filtro, este test fallaba contra su propio arreglo: el
    comentario que explica por qué se quitó `white-space: nowrap` contiene esa
    misma cadena, y el `not in` la encontraba. Lo que se audita es lo que el
    navegador aplica, no la prosa de al lado."""
    i = _CSS.index(selector + " {")
    body = _CSS[i:_CSS.index("}", i)]
    return re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)


def test_grid_columns_have_zero_floor():
    for selector in (".plansGrid",):
        body = _block(selector)
        assert "minmax(0, 1fr)" in body, (
            "la grid de planes debe usar `minmax(0, 1fr)`: con `1fr` el suelo "
            "es el contenido y una tarjeta con texto indivisible aplasta a las "
            "demás (pasó en prod con el pill de lanzamiento)"
        )
    # Los breakpoints también: el mismo desbalance aparece a 2 columnas.
    for m in re.finditer(r"grid-template-columns:\s*repeat\(\s*\d+\s*,\s*([^)]+)\)", _CSS):
        arg = m.group(1).strip()
        assert arg.startswith("minmax(0"), (
            f"`repeat(n, {arg})` deja el suelo en el contenido — usar minmax(0, 1fr)"
        )


def test_launch_pill_is_breakable():
    body = _block(".launchOfferPill")
    assert "white-space: nowrap" not in body, (
        "el pill de lanzamiento NO puede ser indivisible: es lo que rompió el "
        "reparto de columnas. Si el copy crece, acortarlo (deadlineShort)"
    )


def test_primary_cta_keeps_fill_in_dark():
    body = _block(':global(html[data-theme="dark"]) .planButtonPrimary')
    assert "linear-gradient" in body, (
        "el CTA de la tarjeta destacada debe conservar su relleno en oscuro; "
        "sin este override la regla genérica `.planButton` del tema lo apaga"
    )
    # Y debe ir DESPUÉS de la regla genérica: misma especificidad en el :hover,
    # así que ahí gana el orden de aparición.
    generico = _CSS.index(':global(html[data-theme="dark"]) .planButton {')
    primario = _CSS.index(':global(html[data-theme="dark"]) .planButtonPrimary {')
    assert primario > generico, (
        "el override del CTA primario debe declararse DESPUÉS del genérico"
    )
