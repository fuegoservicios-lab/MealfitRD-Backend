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


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 144
