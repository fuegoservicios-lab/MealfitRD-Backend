# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-118 · 2026-09-19] La barra de pestañas del teléfono se puede plegar.

El dueño: «crea algo desplegable para poder cerrar el menú para abajo y así tener más espacio en el chat… no puede
estorbar y tiene que cerrarse como el menú de actualizar platos». La barra (64 px fijos) se pliega deslizándola hacia
abajo —sigue al dedo— o tocando un asa en su borde superior. Plegada NO desaparece: queda una franja de 22 px con el
asa, ENCIMA de la zona del indicador de inicio de iOS (ahí un gesto hacia arriba es del sistema). Se recuerda en el
dispositivo y el espacio se DEVUELVE: `html[data-tabbar-plegada]` sube `--tabbar-recupera` (42 px) y todo lo que
reservaba sitio para la barra lo resta. Con el teclado abierto sigue mandando `data-kb-open` (la barra se va entera).
Medido en el arnés a 375×812: abierta 65 px, plegada 22 px a la vista, relleno del contenido 107 → 65 px.
Contrato fino: `frontend/src/__tests__/lote118.test.jsx`."""
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
    return p.read_text(encoding="utf-8")


def test_plegada_queda_una_puerta_visible_y_el_teclado_sigue_mandando():
    css = _front("src/components/dashboard/BottomTabBar.module.css")
    assert "transform: translateY(calc(100% - var(--tabbar-asa) - env(safe-area-inset-bottom, 0px)));" in css
    assert "--tabbar-asa: 22px;" in css
    assert "transform: translateY(110%) !important;" in css, "con el teclado abierto la barra se va ENTERA, plegada o no"
    barra = _front("src/components/dashboard/BottomTabBar.jsx")
    assert "aria-expanded={!plegada}" in barra
    assert "tabIndex={plegada ? -1 : undefined}" in barra, "plegadas, las pestañas no reciben foco"


def test_el_espacio_se_devuelve_en_todos_los_que_lo_reservaban():
    assert "--tabbar-recupera: 42px;" in _front("src/index.css")
    assert _front("src/components/dashboard/DashboardLayout.module.css").count("- var(--tabbar-recupera, 0px)") == 2
    chat = _front("src/pages/AgentPage.jsx")
    assert "calc(1.4rem + 64px - var(--tabbar-recupera, 0px) + env(safe-area-inset-bottom, 0px))" in chat
    for rel in ("src/components/recipes/MobileRecipes.module.css", "src/pages/Pantry.mobileFridge.module.css",
                "src/pages/History.module.css"):
        assert "calc(80px - var(--tabbar-recupera, 0px) + env(safe-area-inset-bottom, 0px))" in _front(rel), rel


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 118
