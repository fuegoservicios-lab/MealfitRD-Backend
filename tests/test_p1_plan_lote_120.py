# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-120 · 2026-09-19] La pestaña activa se marca con un BRILLO, no con una raya.

El dueño, con la lengüeta del lote 119 ya en el teléfono: «quiero que el brillo de cuando algo está seleccionado no se
parezca a la raya de bajar el menú, quiero que sea un brillo eso de arriba». El indicador de la pestaña activa era una
barrita sólida de 32×2,5 px en el borde superior de la barra — el mismo borde donde vive el asa de plegar desde el lote
118: dos rayitas en el mismo sitio diciendo cosas distintas. Ahora es un degradado radial que cae desde el borde sobre
el icono y se apaga sin contorno (luz, no control), pintado DETRÁS del icono y del rótulo (`z-index: -1` dentro del
contexto de apilamiento de la barra), y el icono activo emite un halo suave. Tema claro y oscuro.
Contrato fino: `frontend/src/__tests__/lote120.test.js`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _css() -> str:
    p = _FRONT / "src/components/dashboard/BottomTabBar.module.css"
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _regla(css: str, selector: str) -> str:
    i = css.index(selector + " {")
    return css[i:css.index("}", i)]


def test_el_indicador_es_luz_y_no_una_raya():
    css = _css()
    r = _regla(css, "    .activeIndicator")
    assert "radial-gradient(ellipse 50% 100% at 50% 0%" in r
    assert "linear-gradient" not in r and "height: 2.5px" not in r, "volvió la raya: se confunde con el asa de plegar"
    assert "z-index: -1;" in r, "el brillo va DETRÁS del icono y del rótulo: encima los tiñe"
    assert "pointer-events: none;" in r
    assert "radial-gradient" in _regla(css, ':global(html[data-theme="dark"]) .activeIndicator')


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 120
