"""[P2-ICON-CLOSE-UNIFORM · 2026-09-02] Un solo botón de cierre («X») en toda la app.

Había 34 «X» en 24 archivos estilizadas a mano (13-26 px, radios y hovers distintos). Los 21
cierres de paneles/modales/avisos llevan ahora la clase de sistema `ui-close` (index.css):
tinte de fondo al hover, leve escala, foco visible, 44 px en táctil, y NUNCA la sombra de
elevación de los CTAs (la «X» es acción terciaria de salida). Los quitadores de chips/filas y
los iconos de tabla quedan fuera a propósito: son otro patrón.

Tooltip-anchor: P2-ICON-CLOSE-UNIFORM | button.ui-close
"""
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _src(rel: str) -> str:
    p = FRONTEND / rel
    if not p.exists():
        pytest.skip("frontend no visible desde este checkout")
    return p.read_text(encoding="utf-8")


def test_system_class_exists_without_cta_shadow():
    css = _src("src/index.css")
    assert "button.ui-close {" in css
    h = css.index("button.ui-close:hover:not(:disabled) {")
    assert "box-shadow: none !important" in css[h:h + 400]
    assert "@media (pointer: coarse)" in css


def test_close_buttons_carry_the_class():
    n = 0
    for rel in ("src/components/common/Modal.jsx", "src/pages/Dashboard.jsx", "src/pages/History.jsx",
                "src/components/dashboard/NotificationCenter.jsx", "src/components/layout/Header.jsx"):
        n += _src(rel).count("ui-close")
    assert n >= 9, n


def test_vitest_anchor_exists():
    assert (FRONTEND / "src/__tests__/IconClose.uniform.test.jsx").exists()
