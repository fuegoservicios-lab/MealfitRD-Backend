# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-119 · 2026-09-19] El asa de la barra se NOTA y la barra del teléfono lleva 5 pestañas.

El dueño, con la barra plegable del lote 118 ya en el teléfono: «se ve bien pero si se pudiera notar más que eso se
puede presionar para bajarse fuera mejor» y «con el generador encendido esto se ve con demasiados apartados, ¿no
recomiendas fusionar alguno? como por ejemplo el de historial».

  · El asa era una píldora gris de 30×4 DENTRO de la barra: se veía, no decía «tócame». Ahora es una lengüeta con
    flecha que SOBRESALE 16 px del borde superior, en el color de acento; plegada, la franja lleva la flecha hacia
    arriba. Las primeras veces la flecha se mece y CALLA sola (al usarse, o a las 5 apariciones: la barra se monta en
    cada página y una pista sin tope se mecería para siempre).
  · Con el generador encendido la nav tiene 6 entradas; una barra inferior aguanta 5. `repartoTelefono`
    (config/dashboardNav.js, el SSOT) saca el Historial —el destino menos diario— al menú ☰ de la cabecera. En modo
    contador son 4 y nadie se mueve. El lateral de escritorio y el menú del Agente siguen pintando la nav ENTERA.
    `navItemsFor` NO cambia: lo que cambia es quién pinta qué.

Medido en el arnés a 375×812: 5 pestañas de 75 px (eran 6 de 62), lengüeta 52×17 sobre una barra de 65 px, el menú ☰
abre con «Historial» primero. Contrato fino: `frontend/src/__tests__/lote119.test.jsx`."""
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


def test_el_reparto_es_del_ssot_y_lo_consumen_los_dos():
    nav = _front("src/config/dashboardNav.js")
    assert "export const TAB_BAR_MAX = 5;" in nav
    assert "export const repartoTelefono = (items) => {" in nav
    assert "const SALEN_PRIMERO = ['history'];" in nav, "el Historial es el primero en salir de la barra"
    barra = _front("src/components/dashboard/BottomTabBar.jsx")
    # [P1-NEVERA-OPCIONAL] la nav también depende de la Nevera
    assert "repartoTelefono(navItemsFor({ trackingMode: isTrackingMode(userProfile, planData), nevera: neveraActiva(userProfile) })).barra" in barra
    layout = _front("src/components/dashboard/DashboardLayout.jsx")
    assert "const menuTelefono = repartoTelefono(menuItems).menu;" in layout
    assert "{menuTelefono.map((item) => {" in layout, "lo que sale de la barra ENTRA en el menú ☰: nada queda sin puerta"


def test_el_asa_sobresale_con_flecha_y_la_pista_calla():
    css = _front("src/components/dashboard/BottomTabBar.module.css")
    assert "asaPildora" not in css
    asa = css[css.index("    .asa {"):]
    asa = asa[:asa.index("}")]
    assert "top: -17px;" in asa, "la lengüeta sobresale de la barra: eso es lo que la hace evidente"
    assert ".tabBar.plegada .asaFlecha {" in css
    assert "prefers-reduced-motion: reduce" in css
    hook = _front("src/hooks/useTabBarPlegable.js")
    assert "export const PISTAS_MAX = 5;" in hook
    assert "safeLocalStorageSet(CLAVE_ASA_USADA, '1');" in hook, "quien ya usó el asa no necesita que se la presenten"
    # la geometría del lote 118 no se toca: plegada sigue dejando 22 px y devolviendo 42
    assert "--tabbar-asa: 22px;" in css
    assert "--tabbar-recupera: 42px;" in _front("src/index.css")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 119
