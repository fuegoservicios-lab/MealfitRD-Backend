# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-87 · 2026-09-17] El contador en el teléfono: la invitación a encender el plan va PRIMERA (hija directa de
la rejilla, orden por `grid-template-areas`) y la tarjeta de macros recupera aire en ≤480px sin tocar letras ni alturas de
barra. Pedido del dueño con tres capturas a 392px."""
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


def test_la_invitacion_al_plan_es_hija_directa_y_va_primera_en_movil():
    jsx = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "<div className={styles.turnOnSlot}>" in jsx
    lateral = jsx[jsx.index("className={styles.sideCol}"):jsx.index("className={styles.turnOnSlot}")]
    assert "<TurnOnPlanCard" not in lateral
    css = re.sub(r"\s+", " ", _front("src/components/dashboard/DashboardTracking.module.css"))
    assert 'grid-template-areas: "main side" "main plan";' in css
    assert 'grid-template-areas: "plan" "main" "side";' in css
    assert ".turnOnSlot { grid-area: plan;" in css


def test_la_tarjeta_de_macros_recupera_aire_en_480():
    css = _front("src/components/dashboard/TrackingProgress.module.css")
    i = css.index("@media (max-width: 480px) {")
    block = css[i:css.index("\n}\n", i)]
    assert "padding: 1.15rem;" in block and "gap: 1.3rem;" in block and "P1-PLAN-LOTE-87" in block
    # las letras y la altura de la barra NO se tocan (las dos correcciones del dueño de agosto)
    assert "font-size: 1rem;" in block and "height: 22px !important;" in css


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 87
    assert "P1-PLAN-LOTE-87" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
