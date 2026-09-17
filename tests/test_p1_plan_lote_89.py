# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-89 · 2026-09-17] La campana de notificaciones ATRACA en la cabecera en teléfono y tableta. Pedido del dueño
con captura a 392px (el tirador del borde derecho quedaba encima de la tarjeta de la invitación): «en vez de la derecha, ¿por
qué no mejor arriba? un diseño único y abstracto, llamativo y que no estorbe ni choque con nada».

Lo que se ancla aquí es el CONTRATO entre ficheros que no se conocen entre sí: la cabecera móvil ofrece el hueco con la MISMA
condición con la que se monta el centro, el centro se portaliza a él, y los dos usan el MISMO corte de 1024px (si uno cambia,
la campana atraca en una cabecera invisible o se queda en el borde con la cabecera a la vista)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_DASH = "src/components/dashboard/"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _plano(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def test_la_cabecera_ofrece_el_hueco_con_la_condicion_del_centro():
    layout = _front(_DASH + "DashboardLayout.jsx")
    acciones = layout[layout.index("className={styles.mobileHeaderActions}"):layout.index("</header>")]
    assert "{showNotifCenter && <NotificationSlot />}" in acciones
    assert acciones.index("<NotificationSlot />") < acciones.index("className={styles.menuBtn}"), "la campana va a la IZQUIERDA del menú"
    # una sola condición para el centro y para su hueco: si divergen, la campana atraca sin centro o el centro sin muelle
    assert "const showNotifCenter = location.pathname.replace(" in layout
    assert "{showNotifCenter && ( <NotificationCenter hidden=" in _plano(layout)


def test_el_corte_de_1024_es_el_mismo_en_la_cabecera_y_en_el_centro():
    assert "useMediaQuery('(max-width: 1024px)')" in _front(_DASH + "NotificationCenter.jsx")
    css = _front(_DASH + "DashboardLayout.module.css")
    corte = css[css.index("@media (max-width: 1024px) {"):]
    assert ".mobileHeader { display: flex; }" in _plano(corte[:corte.index(".sidebar {")])


def test_el_centro_se_portaliza_al_hueco_y_atracado_no_se_esconde():
    nc = _front(_DASH + "NotificationCenter.jsx")
    assert "useSyncExternalStore(subscribeNotifSlot, getNotifSlot, () => null)" in nc
    assert "createPortal(trigger, docked ? slot : document.body)" in nc
    assert "createPortal(drawer, document.body)" in nc, "el cajón sigue yendo al body: dentro de la cabecera quedaría bajo todo"
    assert "if (hidden && !docked) return null;" in nc
    store = _front("src/utils/notifSlot.js")
    # solo suelta el hueco quien lo tiene: el desmontaje tardío de la cabecera vieja no borra a la nueva
    assert "if (_node !== node) return;" in store
    slot = _front(_DASH + "NotificationSlot.jsx")
    assert "useLayoutEffect" in slot and "width: 40" in slot and "height: 40" in slot


def test_el_orbe_mide_lo_que_el_menu_y_solo_se_mueve_con_algo_sin_leer():
    css = _front(_DASH + "NotificationCenter.module.css")
    i = css.index(".handle.handleDocked {")
    regla = css[i:css.index("}", i)]
    for decl in ("position: relative;", "width: 40px;", "height: 40px;", "border-radius: 50%;", "transform: none;"):
        assert decl in regla, decl
    plano = _plano(css)
    assert ".handleDocked.handleAlert::before { opacity: 1; animation: notifRingSpin" in plano
    assert ".handleAlert .handleOrbit { opacity: 1; animation: notifOrbit" in plano
    assert ':global(html:not([data-theme="dark"])) .handle.handleDocked {' in plano
    reducido = plano[plano.rindex("@media (prefers-reduced-motion: reduce)"):]
    assert ".handleDocked.handleAlert::before, .handleAlert .handleOrbit { animation: none; }" in reducido
    # el tirador de escritorio no se tocó
    assert "border-radius: 16px 0 0 16px;" in css and "top: 38%;" in css


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 89
    assert "P1-PLAN-LOTE-89" in (_BACKEND / "docs" / "notificaciones_campana.md").read_text(encoding="utf-8")
