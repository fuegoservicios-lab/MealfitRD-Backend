# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-91 · 2026-09-17] Descartada, la invitación al plan no encabeza el contador en el teléfono.

Reporte del dueño con captura, tras pulsar «Ahora no»: «se puso raro lo del progreso en tiempo real». Medido en el arnés a
392px: la tarjeta colapsa a un enlace tenue de 27px que se quedaba en la PRIMERA posición y, desde que las secciones van sin
marco (`P1-PLAN-LOTE-88`), a 24px del título se leía como una línea del propio «Progreso en Tiempo Real». Ahora cierra la
pantalla —la puerta sigue ahí— con la misma línea fina que separa las demás secciones.

[P1-PLAN-LOTE-98 · 2026-09-18] SUPERSEDED en lo del enlace: el dueño lo quitó del todo («ya está el interruptor en
configuración»). Quedan vigentes las reglas que no dependían de él: la tarjeta sin descartar sigue primera, se reordenan
ÁREAS y no filas, y la columna lateral vacía sale del reparto. El contrato nuevo vive en `test_p1_plan_lote_98.py`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_CSS = "src/components/dashboard/DashboardTracking.module.css"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _sin_comentarios(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _bloque(css: str, cabecera: str) -> str:
    i = css.index(cabecera)
    return re.sub(r"\s+", " ", css[i:css.index("\n}\n", css.index("{", i))])


def test_la_tarjeta_sin_descartar_sigue_primera_y_se_reordenan_areas():
    css = _sin_comentarios(_front(_CSS))
    movil = _bloque(css, "@media (max-width: 900px) {")
    assert 'grid-template-areas: "plan" "main" "side";' in movil, "la tarjeta SIN descartar sigue primera (lote 87)"
    # se reordenan las ÁREAS y no la fila del bloque: una fila vacía sigue cobrando sus dos huecos
    assert "grid-row:" not in movil
    escritorio = re.sub(r"\s+", " ", css[:css.index("@media")])
    assert 'grid-template-areas: "main side" "main plan";' in escritorio


def test_la_columna_lateral_vacia_sale_del_reparto():
    css = _sin_comentarios(_front(_CSS))
    assert ".sideCol:empty { display: none; }" in re.sub(r"\s+", " ", css)
    assert ".sideCol:not(:empty) {" in css


def test_el_descarte_persiste():
    """[P1-PLAN-LOTE-98] Ya no hay enlace; lo que sigue es que el descarte se recuerda."""
    jsx = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "_DISMISS_KEY = 'mealfit_turnon_card_dismissed'" in jsx
    assert "safeLocalStorageSet(_DISMISS_KEY, '1')" in jsx


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 91
    assert "P1-PLAN-LOTE-91" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
