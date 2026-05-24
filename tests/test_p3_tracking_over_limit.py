"""[P3-TRACKING-OVER-LIMIT · 2026-05-20] Test anti-regresión del signaling
visual cuando el usuario excede una meta en el card "Progreso en Tiempo Real".

Pre-fix:
    `calcPerc(val, max) = Math.min(Math.round((val/max)*100) || 0, 100)`
    capeaba al 100%. Un usuario con 2240 kcal sobre meta 2100 veía la barra
    al 100% en el mismo color amber que el caso "llegó exacto a la meta" —
    sin diferenciación visual entre "completaste" y "te pasaste".

Fix:
    1. `calcPerc` retorna el % real (sin cap).
    2. `ProgressBar` deriva `isOver = perc > 100` y `fillWidth = min(perc, 100)`.
    3. Cuando `isOver`:
        - Bar gradient → rojo (#FCA5A5 → #DC2626).
        - Número consumido → rojo.
        - Glow → rojo (rgba(220, 38, 38, 0.45)).
        - Track border → rojo tenue.
        - Badge "+exceso unit" inline (e.g., "+140 kcal").
        - `.fillPerc` muestra el % real uncapped (e.g., "107%").
    4. `isComplete = perc >= 100` se mantiene como concepto separado para
       el glow celebración cuando llega exacto al 100%.

Tests parser-based: anchor literal en JSX para que un refactor que remueva
el signaling visual de exceso falle CI explícitamente.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_tracking_over_limit`
matchea este archivo `test_p3_tracking_over_limit.py`.

Tooltip-anchor: P3-TRACKING-OVER-LIMIT.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"


@pytest.fixture(scope="module")
def jsx_src() -> str:
    return _TRACKING_JSX.read_text(encoding="utf-8")


def test_calc_perc_uncapped(jsx_src: str):
    """`calcPerc` NO debe capear al 100%. Pre-fix usaba `Math.min(..., 100)`
    que ocultaba el exceso. Post-fix retorna el % real."""
    # Buscar la asignación de calcPerc.
    calc_perc_match = re.search(
        r"calcPerc\s*=\s*\([^)]*\)\s*=>\s*([^;\n]+)",
        jsx_src,
    )
    assert calc_perc_match, "Definición de `calcPerc` no encontrada."
    body = calc_perc_match.group(1)
    assert "Math.min" not in body or ", 100)" not in body, (
        f"P3-TRACKING-OVER-LIMIT regresión: `calcPerc` parece capear al 100% "
        f"de nuevo. Body actual: {body!r}. Si necesitas cap, hazlo en "
        f"`fillWidth` dentro de ProgressBar, NO en `calcPerc` global."
    )


def test_is_over_flag_present(jsx_src: str):
    """`isOver = perc > 100` debe existir en ProgressBar — gate del
    signaling rojo."""
    assert re.search(r"isOver\s*=\s*perc\s*>\s*100", jsx_src), (
        "P3-TRACKING-OVER-LIMIT regresión: `isOver = perc > 100` ausente "
        "en TrackingProgress.jsx. Sin él, el signaling rojo NO se gatea."
    )


def test_fill_width_capped_inside_bar(jsx_src: str):
    """El cap superior al 100% debe existir. Pre-fix: `fillWidth = Math.min(perc, 100)`.
    [P3-TRACKING-FILL-MIN-VISUAL · 2026-05-22] Ahora el cap vive en
    `_percCapped = Math.min(perc, 100)` antes de aplicar el piso del
    `_FILL_VISUAL_MIN`. Sin el cap, una meta excedida al 150% haría el
    div fill desbordarse del track."""
    assert re.search(
        r"(_percCapped\s*=\s*Math\.min\(perc,\s*100\))|"
        r"(fillWidth\s*=\s*Math\.min\s*\(\s*perc\s*,\s*100\s*\))",
        jsx_src,
    ), (
        "P3-TRACKING-OVER-LIMIT regresión: cap superior al 100% ausente. "
        "Buscado `_percCapped = Math.min(perc, 100)` (post-FILL-MIN-VISUAL) "
        "o el legacy `fillWidth = Math.min(perc, 100)`. Sin ninguno, fill "
        "se desborda cuando perc > 100."
    )


def test_red_gradient_constant_present(jsx_src: str):
    """El gradient rojo (`#FCA5A5 → #DC2626`) debe estar en la fuente
    — anchor de la paleta del estado over."""
    assert "#DC2626" in jsx_src, (
        "P3-TRACKING-OVER-LIMIT regresión: color rojo `#DC2626` (Tailwind "
        "red-600) ausente. Es el ancla de la paleta del estado over. "
        "Si cambias a otro rojo, actualiza el test."
    )
    assert "#FCA5A5" in jsx_src, (
        "P3-TRACKING-OVER-LIMIT regresión: color rojo claro `#FCA5A5` "
        "(Tailwind red-300) ausente. Es el inicio del gradient red→darker "
        "del fill cuando over."
    )


# Nota: el test del badge "+exceso unit" se movió a
# `test_p3_tracking_over_no_badge.py` tras el follow-up del mismo día.
# El user pidió remover el badge — el signaling vía color + % uncapped
# dentro del fill ya comunica el exceso. Este archivo sigue anclando el
# RESTO del signaling visual (gradient rojo, número rojo, glow rojo,
# % uncapped, etc.) que se mantiene intacto.


def test_consumed_text_color_switches_when_over(jsx_src: str):
    """El color del número consumido (`barConsumed`) debe switchear a
    rojo cuando `isOver`. Variable `consumedTextColor` con condicional."""
    assert re.search(
        r"consumedTextColor\s*=\s*isOver",
        jsx_src,
    ), (
        "P3-TRACKING-OVER-LIMIT regresión: `consumedTextColor` no usa "
        "`isOver` como gate. Sin esto, el número grande se queda en slate "
        "aunque la meta esté excedida."
    )


def test_effective_gradient_uses_over_gradient(jsx_src: str):
    """`effectiveGradient` debe switchear entre `OVER_GRADIENT` y el
    gradient original según `isOver`. Esta es la mutación más visible."""
    assert re.search(
        r"effectiveGradient\s*=\s*isOver\s*\?\s*OVER_GRADIENT",
        jsx_src,
    ), (
        "P3-TRACKING-OVER-LIMIT regresión: `effectiveGradient = isOver ? "
        "OVER_GRADIENT : gradient` ausente. Sin esto la barra mantiene el "
        "gradient amber/blue/green/pink aunque haya exceso → el signaling "
        "rojo principal se pierde."
    )


def test_glow_uses_red_when_over(jsx_src: str):
    """El `boxShadow` debe usar rgba red cuando `isOver` (no el color
    original del macro). El glow refuerza el signaling rojo."""
    # Buscar el shadow `rgba(220, 38, 38, 0.45)` (red glow over).
    assert re.search(
        r"rgba\(\s*220\s*,\s*38\s*,\s*38\s*,\s*0\.45\s*\)",
        jsx_src,
    ), (
        "P3-TRACKING-OVER-LIMIT regresión: red glow `rgba(220, 38, 38, 0.45)` "
        "ausente. Sin él, el glow se queda en el color del macro aunque "
        "la barra esté roja — inconsistencia visual."
    )


def test_fill_perc_uses_uncapped_value(jsx_src: str):
    """`.fillPerc` debe mostrar `{perc}%` (no `{fillWidth}%`). El número
    dentro de la barra es el % real — cuando over, debe leer 107% no 100%.
    [P3-TRACKING-PERC-NARROW-FIX · 2026-05-22] Tolerante al template literal
    `${styles.fillPerc} ${...}` introducido por el fix narrow — el regex ahora
    busca `styles.fillPerc` en cualquier expression del className."""
    fill_perc_match = re.search(
        r"styles\.fillPerc[\s\S]*?>\s*\n?\s*\{(\w+)\}%",
        jsx_src,
    )
    assert fill_perc_match, "Span `.fillPerc` no encontrado."
    var_used = fill_perc_match.group(1)
    assert var_used == "perc", (
        f"P3-TRACKING-OVER-LIMIT regresión: `.fillPerc` muestra "
        f"`{{{var_used}}}%` en lugar de `{{perc}}%`. Si el span lee "
        f"`fillWidth`, siempre dirá 100% cuando over — perdiendo el "
        f"signaling numérico del exceso."
    )


def test_tooltip_anchor_count(jsx_src: str):
    """Marker `P3-TRACKING-OVER-LIMIT` aparece ≥2× en TrackingProgress.jsx
    (al menos: calcPerc + ProgressBar)."""
    count = jsx_src.count("P3-TRACKING-OVER-LIMIT")
    assert count >= 2, (
        f"P3-TRACKING-OVER-LIMIT: tooltip-anchor aparece {count}× "
        f"en TrackingProgress.jsx, esperado ≥2 (bloque de calcPerc + "
        f"bloque de ProgressBar con isOver)."
    )
