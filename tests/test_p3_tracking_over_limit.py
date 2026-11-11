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


def test_fill_perc_uses_uncapped_value(jsx_src: str):
    """`.fillPerc` debe mostrar `{perc}%` (no `{fillWidth}%`). El número
    dentro de la barra es el % real — cuando over, debe leer 107% no 100%.
    [P3-TRACKING-PERC-NARROW-FIX · 2026-05-22] Tolerante al template literal
    `${styles.fillPerc} ${...}` introducido por el fix narrow — el regex ahora
    busca `styles.fillPerc` en cualquier expression del className."""
    fill_perc_match = re.search(
        # [reapuntado 2026-08-23, lote 4 i18n] `{perc}%` paso a `{formatPercent(perc)}`
        # (el % se formatea por locale). La propiedad medida no cambia: el argumento
        # es `perc` (uncapped), no `fillWidth` (capped a 100).
        r"styles\.fillPerc[\s\S]*?>\s*\{formatPercent\((\w+)\)\}",
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


# [P1-PLAN-LOTE-980] Retirados is_over/rojo #DC2626/glow/consumedTextColor/anchors: 9ffa8b92 (2026-10-08) cambió el exceso a la paleta por meta (naranja ámbar), cubierta por vitest TrackingProgress.goal_colors.
