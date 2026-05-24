"""[P3-TRACKING-PERC-NARROW-FIX · 2026-05-22 → REVERTIDO P3-TRACKING-PERC-INSIDE-ALWAYS · 2026-05-22]
Tests del comportamiento "% siempre dentro del fill".

Historia de la decisión (3 iteraciones del mismo día):

  1. Pre-fix: el `.fillPerc` vivía dentro del `.fill` con `justify-content:
     flex-end + padding-right`. Cuando el fill era estrecho (proteína 7%),
     el texto se renderizaba al borde derecho del fill y desbordaba hacia
     la izquierda — pero el contraste blanco/gris-claro era insuficiente,
     el user vio solo "%" sin el "7".

  2. P3-TRACKING-PERC-NARROW-FIX: el badge se movió de child del `.fill`
     a child del `.track` con `position: absolute + left: ${fillWidth}%`.
     Cuando `fillWidth <= 15`, una clase `.fillPercOutside` aplicaba
     `translateX(0)` que ALEJABA el badge del fill (sobre el track gris,
     color del macro). Cerró el caso "solo % visible" pero introdujo un
     nuevo problema: el badge se ve SEPARADO del fill — el user explícita-
     mente lo rechazó ("no me gusta que esté separado, me gustaba más
     cuando estaba dentro").

  3. P3-TRACKING-PERC-INSIDE-ALWAYS (actual): mantiene la arquitectura
     position absolute del fix narrow (badge en `.track`) pero remueve la
     variante outside. El badge SIEMPRE se posiciona con `translateX(-100%)`
     — alineado al borde DERECHO del fill. Cuando el fill es estrecho, el
     texto desborda HACIA LA IZQUIERDA, ahora SÍ legible gracias al
     text-shadow doble layer (capa cercana 0.65 opacity + capa difusa
     0.35 opacity). Battery style consistente sin importar fillWidth.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_tracking_perc_narrow_fix`
preservado por compatibilidad con el marker del día anterior.

Tooltip-anchor: P3-TRACKING-PERC-INSIDE-ALWAYS (anchor actualizado;
P3-TRACKING-PERC-NARROW-FIX queda como referencia histórica en comments).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"
_TRACKING_CSS = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.module.css"


@pytest.fixture(scope="module")
def jsx_src() -> str:
    return _TRACKING_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css_src() -> str:
    return _TRACKING_CSS.read_text(encoding="utf-8")


def _strip_css_comments(src: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", src)


# ===========================================================================
# Sección 1 — JSX: sin branch outside, badge siempre dentro
# ===========================================================================

def test_jsx_no_outside_class_branch(jsx_src: str):
    """El branch `fillWidth <= N ? styles.fillPercOutside : ''` debe
    haberse eliminado. El user rechazó el badge "separado" del fill."""
    assert "fillPercOutside" not in jsx_src, (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: clase `fillPercOutside` "
        "reapareció en el JSX. El user rechazó esa variante visual — "
        "el badge SIEMPRE debe estar dentro del fill (battery style)."
    )


def test_jsx_color_is_always_white(jsx_src: str):
    """El color del badge NO debe alternar entre `effectiveGlowColor` y
    `white` — siempre debe ser white (battery style consistente)."""
    # Buscar la asignación de color en el span del fillPerc.
    span_block = re.search(
        r"<span\s+className=\{styles\.fillPerc\}[^>]*>",
        jsx_src,
    )
    assert span_block is not None, "<span styles.fillPerc> no encontrado."
    body = span_block.group(0)
    assert "effectiveGlowColor" not in body, (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: el color del badge aún "
        "depende de `effectiveGlowColor` (caso outside). Debe ser blanco "
        "constante."
    )


def test_jsx_passes_left_inline_style(jsx_src: str):
    """`left: ${fillWidth}%` debe estar en inline style — el CSS no puede
    leer fillWidth, viene del state JS."""
    m = re.search(r"left:\s*`\$\{fillWidth\}%`", jsx_src)
    assert m is not None, (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: `left: ${fillWidth}%` "
        "removido del inline style. Sin esto el badge no se posiciona al "
        "borde del fill."
    )


def test_jsx_fill_div_remains_self_closing(jsx_src: str):
    """El `.fill` div sigue siendo self-closing — el badge vive como
    sibling en el `.track` (no child del fill)."""
    fill_block = re.search(
        r"className=\{styles\.fill\}[\s\S]*?(/>|</div>)",
        jsx_src,
    )
    assert fill_block is not None
    body = fill_block.group(0)
    assert "styles.fillPerc" not in body, (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: span `.fillPerc` "
        "reapareció dentro del `.fill` div. Eso reintroduce el bug "
        "original donde el overflow del fill recorta el texto."
    )


# ===========================================================================
# Sección 2 — CSS: variante outside eliminada, text-shadow reforzado
# ===========================================================================

def test_css_no_fill_perc_outside_class(css_src: str):
    """`.fillPercOutside` debe eliminarse del CSS — el user rechazó la
    variante visual."""
    css = _strip_css_comments(css_src)
    assert not re.search(r"\.fillPercOutside\s*\{", css), (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: clase "
        "`.fillPercOutside` reapareció en CSS. El user rechazó esa "
        "variante — debe quedar fuera."
    )


def test_css_fill_perc_has_default_translate_negative(css_src: str):
    """`.fillPerc` default mantiene `translateX(-100%)` — badge alineado
    al borde DERECHO del fill (dentro)."""
    css = _strip_css_comments(css_src)
    m = re.search(r"\.fillPerc\s*\{([^}]+)\}", css)
    assert m is not None
    body = m.group(1)
    assert re.search(r"transform:\s*translateX\(-100%\)", body), (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: `.fillPerc` perdió el "
        "`transform: translateX(-100%)`. Sin esto el badge se renderea "
        "afuera del fill — viola el battery style."
    )


def test_css_fill_perc_color_white(css_src: str):
    """El color del `.fillPerc` debe ser `white` declarado en base
    (battery consistency)."""
    css = _strip_css_comments(css_src)
    m = re.search(r"\.fillPerc\s*\{([^}]+)\}", css)
    assert m is not None
    body = m.group(1)
    assert re.search(r"color:\s*white", body), (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: `.fillPerc` perdió el "
        "`color: white` base. Sin esto el texto puede heredar otro color."
    )


def test_css_text_shadow_has_double_layer(css_src: str):
    """El text-shadow DEBE ser doble layer (capa cercana + capa difusa)
    para garantizar legibilidad cuando el texto desborda el fill estrecho
    y queda sobre el track gris claro. Una sola capa débil (0.35 opacity
    pre-fix) NO da contraste suficiente sobre #E2E8F0."""
    css = _strip_css_comments(css_src)
    m = re.search(r"\.fillPerc\s*\{([^}]+)\}", css)
    assert m is not None
    body = m.group(1)
    # text-shadow con dos capas separadas por coma.
    shadow_match = re.search(
        r"text-shadow:\s*([^;]+);",
        body,
    )
    assert shadow_match is not None, (
        "P3-TRACKING-PERC-INSIDE-ALWAYS regresión: text-shadow ausente. "
        "Sin shadow, el texto blanco sobre track gris claro es invisible."
    )
    shadow_value = shadow_match.group(1)
    # Una capa: `0 1px 1.5px rgba(...)`. Dos capas: dos clausulas separadas por coma.
    layer_count = shadow_value.count("rgba(") + shadow_value.count("rgb(")
    assert layer_count >= 2, (
        f"P3-TRACKING-PERC-INSIDE-ALWAYS regresión: text-shadow tiene solo "
        f"{layer_count} layer(s); esperado ≥2 (capa cercana + capa difusa). "
        f"Sin doble layer el contraste sobre track gris claro es marginal."
    )


def test_marker_present_as_tooltip_anchor(css_src: str, jsx_src: str):
    """Marker `P3-TRACKING-PERC-INSIDE-ALWAYS` presente en CSS y JSX."""
    assert "P3-TRACKING-PERC-INSIDE-ALWAYS" in css_src, (
        "Marker `P3-TRACKING-PERC-INSIDE-ALWAYS` ausente del CSS."
    )
    assert "P3-TRACKING-PERC-INSIDE-ALWAYS" in jsx_src, (
        "Marker `P3-TRACKING-PERC-INSIDE-ALWAYS` ausente del JSX."
    )
