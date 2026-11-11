"""[P3-TRACKING-OVER-NO-BADGE · 2026-05-20] Follow-up del cierre original
P3-TRACKING-OVER-LIMIT del mismo día.

User feedback inmediato post-implementación: "lo de +54 kcal y +140 kcal
no quiero que aparezca visualmente ya que con el porcentaje y el color se
entiende todo".

Decisión:
    Remover el badge inline `+{excessAmount} {unit}` que se renderizaba
    al lado de los valores cuando `isOver`. Mantener el resto del
    signaling visual del over (gradient rojo, número rojo, glow rojo,
    track border rojo tenue, % uncapped dentro del fill "107%").

    El % uncapped dentro del fill (e.g., "107%") + el color rojo del
    número grande + la barra completamente roja son suficientes
    cognitivos del estado "excedido" sin texto adicional.

Este test enforza la NO-presencia del badge. Anchor literal contra
`excessAmount` y el patrón `+{excessAmount}` en la fuente JSX.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_tracking_over_no_badge`
matchea este archivo. El test `test_p3_tracking_over_limit.py` (separado)
sigue anclando el resto del signaling visual.

Tooltip-anchor: P3-TRACKING-OVER-NO-BADGE.
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


def test_excess_amount_variable_removed(jsx_src: str):
    """La variable `excessAmount` debe NO estar declarada como variable
    activa. Si reaparece como `const excessAmount = ...`, el badge se
    está re-introduciendo."""
    # Permitimos menciones en comments (documentar la decisión) — solo
    # bloqueamos la declaración activa.
    assert not re.search(
        r"const\s+excessAmount\s*=",
        jsx_src,
    ), (
        "P3-TRACKING-OVER-NO-BADGE regresión: `const excessAmount = ...` "
        "reapareció en TrackingProgress.jsx. El user explícitamente pidió "
        "remover el badge '+exceso unit'. Si necesitas reintroducir, "
        "primero actualizar este test + memoria + bump marker."
    )


def test_excess_amount_template_removed(jsx_src: str):
    """El literal `+{excessAmount}` (interpolación JSX del valor numérico
    del exceso) NO debe estar en la fuente. Es el render del badge."""
    assert "+{excessAmount}" not in jsx_src, (
        "P3-TRACKING-OVER-NO-BADGE regresión: el literal `+{excessAmount}` "
        "del badge reapareció. Remover el bloque `{isOver && (...)}` "
        "completo dentro de `.barValues` que contenía el `<span>` del badge."
    )


def test_isover_block_not_inside_barvalues(jsx_src: str):
    """No debe haber un `{isOver && (` dentro del bloque `.barValues`
    (que es donde vivía el badge). Otros usos de `isOver` (style
    conditionals) son OK porque no son render condicional dentro de
    barValues."""
    # Extraer el bloque `<div className={styles.barValues}> ... </div>`.
    bar_values_match = re.search(
        r"<div\s+className=\{styles\.barValues\}>(.*?)</div>",
        jsx_src,
        re.DOTALL,
    )
    assert bar_values_match, (
        "Bloque `.barValues` no encontrado en JSX — refactor inesperado."
    )
    body = bar_values_match.group(1)
    assert "{isOver" not in body, (
        "P3-TRACKING-OVER-NO-BADGE regresión: el patrón `{isOver` "
        "reapareció dentro del bloque `.barValues`. Es el indicador "
        "del badge condicional. Removerlo — el signaling del over vive "
        "en el bar fill (rojo) y el número consumido (rojo)."
    )


# [P1-PLAN-LOTE-980] Retirados over_signaling_other_layers_preserved y tooltip_anchor_present: el rojo de exceso salió en 9ffa8b92 (paleta por meta, vitest goal_colors).
