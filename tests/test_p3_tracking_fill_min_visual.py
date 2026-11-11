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


# [P1-PLAN-LOTE-980] 9ffa8b92 (2026-10-08, marcas de meta y referencias) reescribió la barra: la
# constante `_FILL_VISUAL_MIN` y el `_percCapped` desaparecieron, pero las tres garantías siguen —
# piso visual con consumo, 0 sin consumo y nunca más del 100 % del carril. Se anclan sobre la
# fórmula vigente.

def _fill_line(src: str) -> str:
    for ln in src.splitlines():
        if "const fillWidth =" in ln:
            return ln
    raise AssertionError("`const fillWidth =` ausente en TrackingProgress.jsx")


def test_fill_width_zero_when_perc_zero(src: str):
    assert "perc <= 0 ? 0 :" in _fill_line(src), "sin consumo, la barra no puede pintar el piso visual"


def test_fill_min_value_is_at_least_15(src: str):
    m = re.search(r"Math\.max\(proportionalWidth,\s*(\d+)\)", _fill_line(src))
    assert m and int(m.group(1)) >= 15, "el piso visual debe dejar caber el badge del % dentro del fill"


def test_perc_capped_at_100_present(src: str):
    assert re.search(r"const proportionalWidth = Math\.min\(100,", src), "el fill no puede desbordar el carril"


def test_badge_never_left_of_the_floor(src: str):
    assert re.search(r"const badgePosition = Math\.max\(fillWidth,\s*\d+\)", src)
