# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-88 · 2026-09-17] Sin «tarjeticas» en el teléfono (contador de seguimiento). Pedido del dueño con captura a
392px. Las dos secciones grandes (progreso, hidratación) pierden borde, fondo, sombra y relleno en ≤480px y usan el ancho
entero; la invitación al plan sigue siendo tarjeta. El aplanado es opt-in (`flatOnMobile`): el dashboard de plan comparte
esas tarjetas y no cambia."""
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


def _bloque_plano(rel: str) -> str:
    """El ÚLTIMO `@media (max-width: 768px)` del módulo, sin comentarios (que nombran las clases) y con el espacio
    colapsado. Nada del aplanado puede vivir fuera de él: en tableta y escritorio las tarjetas siguen siéndolo.

    [P1-PLAN-LOTE-92 · 2026-09-17] Era el bloque de 480px y no casaba en el teléfono del dueño (basta con que baje
    el zoom del sitio para que su viewport CSS pase de 480). El aplanado se movió al corte de 768, que es donde el
    armazón ya decide «esto es un teléfono»."""
    css = re.sub(r"/\*.*?\*/", "", _front(rel), flags=re.S)
    i = css.rindex("@media (max-width: 768px) {")
    assert "flatMobile" not in css[:i], f"{rel}: el aplanado se salió del bloque de 480px"
    return re.sub(r"\s+", " ", css[i:])


def test_solo_el_contador_pide_las_secciones_planas():
    tracking = _front(_DASH + "DashboardTracking.jsx")
    # [P1-PLAN-LOTE-103] `metasMacros` (era `targets`): las del plan en modo plan, las del perfil en modo contador
    assert "<TrackingProgress planData={metasMacros} userId={userProfile?.id} flatOnMobile />" in tracking
    assert "|| 'guest'} flatOnMobile />" in tracking
    dash = _front("src/pages/Dashboard.jsx")
    assert "flatOnMobile" not in dash, "el dashboard de plan no se aplana en este lote"
    # [P1-PLAN-LOTE-103] y ya no monta el contador: vive en la pestaña «Progreso» (ProgressPage → modo="plan")
    assert "<TrackingProgress" not in dash and "<WaterTracker" not in dash
    assert "flatOnMobile = false" in _front(_DASH + "TrackingProgress.jsx")
    assert "flatOnMobile = false" in _front(_DASH + "WaterTracker.jsx")


def test_plano_es_sin_marco_y_tambien_en_oscuro():
    tp = _bloque_plano(_DASH + "TrackingProgress.module.css")
    # `html[data-theme="dark"] .card` (0,2,1) le ganaría el fondo y el borde a un `.card.flatMobile` (0,2,0) suelto
    assert '.card.flatMobile, :global(html[data-theme="dark"]) .card.flatMobile {' in tp
    for decl in ("padding: 0;", "border: 0;", "border-radius: 0;", "background: none;", "box-shadow: none;", "overflow: visible;"):
        assert decl in tp, decl
    agua = _bloque_plano(_DASH + "WaterTracker.module.css")
    for decl in ("background: none;", "border: 0;", "box-shadow: none;", ".card.flatMobile .inner { padding: 0; gap: 0; }",
                 ".card.flatMobile .body { display: contents; }", ".card.flatMobile .head { order: -1; }"):
        assert decl in agua, decl


def test_la_pagina_canal_linea_y_la_invitacion_sigue_siendo_tarjeta():
    pagina = _bloque_plano(_DASH + "DashboardTracking.module.css")
    assert ".page { padding: 0.35rem 0.15rem 0.5rem; gap: 1.5rem; }" in pagina
    assert ".sideCol:not(:empty) { border-top: 1px solid var(--border," in pagina
    assert "border: 0" not in pagina
    base = re.sub(r"/\*.*?\*/", "", _front(_DASH + "DashboardTracking.module.css"), flags=re.S)
    i = base.index(".turnOnCard {")
    assert "border: 1px solid" in base[i:base.index("}", i)]


def test_el_lote_87_sigue_en_pie():
    """El primer bloque de 480px (aire de la tarjeta) no se tocó: el aplanado va en un bloque nuevo al final."""
    css = _front(_DASH + "TrackingProgress.module.css")
    i = css.index("@media (max-width: 480px) {")
    primero = css[i:css.index("\n}\n", i)]
    assert "padding: 1.15rem;" in primero and "flatMobile" not in primero


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 88
    assert "P1-PLAN-LOTE-88" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
