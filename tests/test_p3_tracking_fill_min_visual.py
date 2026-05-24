"""[P3-TRACKING-FILL-MIN-VISUAL · 2026-05-22] Test del piso visual del
fillWidth en ProgressBar del card "Progreso en Tiempo Real".

Bug recurrente del día 2026-05-22:
  1. Pre-fix original: el badge "7%" del fill estrecho de proteína se
     cortaba.
  2. P3-TRACKING-PERC-NARROW-FIX: movió el badge afuera del fill (user
     rechazó).
  3. P3-TRACKING-PERC-INSIDE-ALWAYS: text-shadow doble layer para que el
     badge dentro del fill sea legible cuando desborda (user reportó que
     aún se veía mal en proteína 7%).
  4. P3-TRACKING-FILL-MIN-VISUAL (actual): solución sugerida por el user
     literal: "subir la barra de los números mínimos para que se pueda
     visualizar". Lógica: si `perc > 0` pero el ancho proporcional al
     real es muy pequeño, renderizar fillWidth visual mínimo de 18%.

Trade-off explícito (justificado en código + memoria):
  - Visual deja de ser 1:1 entre `%` y ancho del fill.
  - Numérico real ("11 / 158 g" + badge "7%") sigue siendo preciso.
  - User explícitamente pidió esta solución.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_tracking_fill_min_visual`
matchea este archivo.

Tooltip-anchor: P3-TRACKING-FILL-MIN-VISUAL.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _TRACKING_JSX.read_text(encoding="utf-8")


def test_fill_visual_min_constant_present(src: str):
    """`_FILL_VISUAL_MIN` constante declarada en el ProgressBar."""
    assert re.search(r"_FILL_VISUAL_MIN\s*=\s*\d+", src), (
        "P3-TRACKING-FILL-MIN-VISUAL regresión: constante `_FILL_VISUAL_MIN` "
        "removida. Sin ella el badge `{perc}%` vuelve a no caber dentro "
        "del fill cuando el % real es bajo."
    )


def test_fill_min_value_is_at_least_15(src: str):
    """El piso debe ser ≥15 para que "100%" (max width 26px) entre cómodo
    dentro del fill (~18% × 380px ≈ 68px)."""
    m = re.search(r"_FILL_VISUAL_MIN\s*=\s*(\d+)", src)
    assert m is not None
    val = int(m.group(1))
    assert val >= 15, (
        f"P3-TRACKING-FILL-MIN-VISUAL regresión: piso={val} (<15). "
        f"Insuficiente para que el badge `{{perc}}%` quepa dentro del fill."
    )
    assert val <= 30, (
        f"P3-TRACKING-FILL-MIN-VISUAL: piso={val} (>30) está exageradamente "
        f"alto. Distorsiona demasiado la magnitud visual. Recomendado 15-22."
    )


def test_fill_width_uses_max_with_min(src: str):
    """`fillWidth = perc <= 0 ? 0 : Math.max(_percCapped, _FILL_VISUAL_MIN)`
    pattern. Si quitan el Math.max el piso no aplica."""
    assert "Math.max(_percCapped, _FILL_VISUAL_MIN)" in src or re.search(
        r"Math\.max\(\s*_percCapped\s*,\s*_FILL_VISUAL_MIN\s*\)", src
    ), (
        "P3-TRACKING-FILL-MIN-VISUAL regresión: `fillWidth` no usa "
        "`Math.max(_percCapped, _FILL_VISUAL_MIN)`. Sin esto, el piso no "
        "aplica en runtime y el bug original recurre."
    )


def test_fill_width_zero_when_perc_zero(src: str):
    """Cuando `perc === 0`, fillWidth debe seguir siendo 0 (no aplicar piso).
    Pre-fix tenía guard `isEmpty = perc === 0` que oculta el badge. El piso
    NO debe sobre-pintar cuando no hay consumo."""
    # Buscar el ternary o branch del fillWidth que gate por perc <= 0.
    assert re.search(
        r"fillWidth\s*=\s*perc\s*<=?\s*0\s*\?\s*0\s*:",
        src,
    ), (
        "P3-TRACKING-FILL-MIN-VISUAL regresión: `fillWidth` no tiene guard "
        "`perc <= 0 ? 0 : ...`. Sin esto, una card recién creada con "
        "perc=0 renderea un fill de 18% sin consumo — UX engañosa."
    )


def test_perc_capped_at_100_present(src: str):
    """`_percCapped = Math.min(perc, 100)` para cap del isOver case."""
    assert re.search(r"_percCapped\s*=\s*Math\.min\(perc,\s*100\)", src), (
        "P3-TRACKING-FILL-MIN-VISUAL regresión: cap al 100% removido. "
        "Casos `isOver` (perc > 100) dispararían fillWidth > 100% — "
        "desborda el track."
    )


def test_marker_present_as_tooltip_anchor(src: str):
    """Marker P3-TRACKING-FILL-MIN-VISUAL presente en JSX."""
    assert "P3-TRACKING-FILL-MIN-VISUAL" in src
