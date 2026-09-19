# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-126 · 2026-09-19] Una Nevera vacía no se desplaza: el alto de la cabecera del teléfono se IMPONE.

El dueño: «cuando la nevera está vacía tiene el scroll de manera innecesaria, ¿no crees?».

Nevera, Recetas e Historial van de borde a borde en el teléfono y llenan la pantalla con
`min-height: 100dvh − (safe-area + 48px)`, donde 48 px «era» la cabecera. La cabecera mide 66 px (relleno 6 + 9,6 + 9,6,
contenido 40, borde 1) y nunca midió 48: la página salía 18 px más alta que la pantalla. Medido en el arnés a 393×852:
`scrollHeight` 870 antes, 852 después.

No se cambió 48 por 66: un número que DESCRIBE a otro elemento vuelve a mentir el día que ese elemento cambia (así nació
este). La cabecera TOMA su alto de `--dash-header-h` y las tres páginas restan la misma variable.

Contrato fino: `frontend/src/__tests__/lote126.test.js`. Cero cambios de backend."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_la_cabecera_toma_su_alto_de_la_variable():
    css = _front("src/components/dashboard/DashboardLayout.module.css")
    assert "--dash-header-h: calc(max(env(safe-area-inset-top), 6px) + 1.2rem + 41px);" in css
    assert re.search(r"\.mobileHeader \{\n\s+box-sizing: border-box;\n\s+height: var\(--dash-header-h\);", css)
    # los términos de la variable salen del relleno real de la cabecera: si cambia uno, cambia el otro
    assert "padding-top: calc(max(env(safe-area-inset-top), 6px) + 0.6rem);" in css


@pytest.mark.parametrize("rel", [
    "src/pages/Pantry.mobileFridge.module.css",
    "src/components/recipes/MobileRecipes.module.css",
    "src/pages/History.module.css",
])
def test_las_paginas_de_borde_a_borde_restan_la_misma_variable(rel):
    css = _front(rel)
    assert "min-height: calc(100dvh - var(--dash-header-h, 66.2px));" in css
    assert "env(safe-area-inset-top, 0px) - 48px" not in css, "volvió el «48px» que describía una cabecera de 66"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 126
