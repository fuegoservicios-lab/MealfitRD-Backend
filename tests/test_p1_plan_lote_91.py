# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-91 · 2026-09-17] Descartada, la invitación al plan no encabeza el contador en el teléfono.

Reporte del dueño con captura, tras pulsar «Ahora no»: «se puso raro lo del progreso en tiempo real». Medido en el arnés a
392px: la tarjeta colapsa a un enlace tenue de 27px que se quedaba en la PRIMERA posición y, desde que las secciones van sin
marco (`P1-PLAN-LOTE-88`), a 24px del título se leía como una línea del propio «Progreso en Tiempo Real». Ahora cierra la
pantalla —la puerta sigue ahí— con la misma línea fina que separa las demás secciones."""
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


def test_solo_el_enlace_se_va_al_final():
    css = _sin_comentarios(_front(_CSS))
    movil = _bloque(css, "@media (max-width: 900px) {")
    assert 'grid-template-areas: "plan" "main" "side";' in movil, "la tarjeta SIN descartar sigue primera (lote 87)"
    assert '.page:has(.turnOnSlot > .turnOnLink) { grid-template-areas: "main" "side" "plan"; }' in movil
    # se reordenan las ÁREAS y no la fila del bloque: una fila vacía sigue cobrando sus dos huecos
    assert "grid-row:" not in movil
    escritorio = re.sub(r"\s+", " ", css[:css.index("@media")])
    assert 'grid-template-areas: "main side" "main plan";' in escritorio


def test_el_enlace_cierra_con_su_linea_y_la_tarjeta_no():
    # [P1-PLAN-LOTE-92] el bloque del teléfono es el de 768, no el de 480
    plano = _bloque(_sin_comentarios(_front(_CSS)), "@media (max-width: 768px) {")
    assert ".turnOnSlot:has(> .turnOnLink) { border-top: 1px solid var(--border," in plano
    assert "padding-top: 1.5rem;" in plano
    assert ".turnOnCard { border-top" not in plano


def test_la_columna_lateral_vacia_sale_del_reparto():
    css = _sin_comentarios(_front(_CSS))
    assert ".sideCol:empty { display: none; }" in re.sub(r"\s+", " ", css)
    assert ".sideCol:not(:empty) {" in css


def test_el_descarte_sigue_siendo_una_puerta_y_persiste():
    """Lo que cambia es DÓNDE se ve, no que exista: el enlace sigue llevando al formulario y el descarte se recuerda."""
    jsx = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "styles.turnOnLink" in jsx and "_DISMISS_KEY = 'mealfit_turnon_card_dismissed'" in jsx
    assert "safeLocalStorageSet(_DISMISS_KEY, '1')" in jsx


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 91
    assert "P1-PLAN-LOTE-91" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
