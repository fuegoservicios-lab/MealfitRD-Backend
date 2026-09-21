# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-144 · 2026-09-20] Los avisos del Dashboard con el generador encendido hablan un solo lenguaje.

El dueño, sobre su pantalla: «quiero que mejoremos visualmente como se ve bioboros con el generador de planes encendido,
en especial lo de micronutrientes, plan congelado y lo de comprobar ahora». Eran tres lenguajes apilados:

    sondeo en descanso   nota subrayada con «Comprobar ahora» en el rosa de ERROR, para decir «todo va bien»
    plan congelado       recuadro con emoji y colores clavados para el tema oscuro, DEBAJO de los micronutrientes
    micros en espera     un panel entero (insignia llena, halo, sombra) para una sola frase

y dos se contradecían a la vista: «Todo va bien» justo encima de «Plan congelado». Ahora: una pieza (`StatusNotice`) con
dos pesos —acción / calma—, el congelado primero (es lo único que pide algo), «todo va bien» calla mientras dure el
congelado, y los micronutrientes en espera son una fila de borde discontinuo. Solo frontend; cero cambios de backend."""
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


def test_el_congelado_va_primero_y_con_peso_de_accion():
    dash = _front("src/pages/Dashboard.jsx")
    i_cong = dash.index("{planData?._frozen_at && (")
    i_sondeo = dash.index("{planPollGaveUp && !isPlanCorrupted")
    i_micros = dash.index("<MicronutrientMeter")
    assert i_cong < i_sondeo < i_micros
    bloque = dash[i_cong:i_cong + 700]
    assert 'peso="accion"' in bloque and "navigate('/dashboard/pantry')" in bloque


def test_todo_va_bien_calla_con_el_plan_congelado():
    dash = _front("src/pages/Dashboard.jsx")
    i = dash.index("{planPollGaveUp && !isPlanCorrupted")
    assert "=== 'partial' && !(planData?._frozen_at) && (" in dash[i:i + 140]
    assert 'peso="calma"' in dash[i:i + 2200]


def test_micros_en_espera_es_una_fila():
    assert "styles.panelQuiet" in _front("src/components/dashboard/MicronutrientMeter.jsx")
    css = _front("src/components/dashboard/MicronutrientMeter.module.css")
    k = css.index(".panelQuiet {")
    assert "border-style: dashed;" in css[k:css.index("}", k)]


def test_los_controles_con_estilo_en_linea_responden_al_raton():
    """«hay muchos botones que deben tener su sombreado o algo cuando le pasan el mouse por encima» — un `style={{}}`
    no admite :hover; el control declara `data-hover` y la respuesta vive en index.css, solo con puntero fino."""
    css = _front("src/index.css")
    k = css.index("[P1-PLAN-LOTE-144 · 2026-09-20] Respuesta al ratón")
    regla = css[k:]
    assert "@media (hover: hover) and (pointer: fine) {" in regla
    for tipo in ("boton", "fila", "icono"):
        assert f'[data-hover="{tipo}"]:not(:disabled):hover {{' in regla, tipo
    assert re.search(r'data-hover="fila"\s+onClick=\{\(\) => setShowDespensaDropdown', _front("src/pages/Dashboard.jsx"))
    assert re.search(r'disabled=\{isExportingData\}\s+data-hover="boton"', _front("src/pages/Settings.jsx"))


def test_la_demo_del_login_y_la_ilustracion_del_movil():
    """«mejora el diseño de esto radicalmente, y anima el de móviles»."""
    demo = _front("src/components/auth/PlanShowcase.jsx")
    for ident in ("mfRingP", "mfRingC", "mfRingG"):
        assert f'id="{ident}"' in demo, ident
    assert "reduced.current" not in demo, "reduce-motion volvió a leerse de un ref que ya no existe"
    illu = _front("src/components/auth/HeroIllustration.jsx")
    assert "{!reduced && GOTAS.map(" in illu, "las gotas (SMIL) no se apagan con un media query: no se montan"
    css = _front("src/pages/Login.css")
    assert ".mf-illu-linea, .mf-illu-bol, .mf-illu-tallo { stroke-dasharray: 1; stroke-dashoffset: 0; }" in css, (
        "el estado BASE de la ilustración debe ser el final: sin animación tiene que verse entera")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 144
