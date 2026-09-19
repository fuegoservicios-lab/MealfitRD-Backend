# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-101 · 2026-09-18] «Registrar comida» se cierra deslizando hacia abajo y el fondo no se mueve.

El gesto de la hoja de actualizar platos (P2-SWAP-SHEET-SCROLL v4/v5), sin framer, y el `touchmove` NO pasivo que
cancela el pan que el cuerpo no puede consumir: era lo que pasaba a la PÁGINA en iOS y dejaba el dashboard
desplazado al cerrar. Ancla cross-repo; el contrato fino vive en `LogMealModal.lote99.test.jsx`."""
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


def test_el_pan_que_el_cuerpo_no_consume_se_cancela_y_la_hoja_sigue_al_dedo():
    # [P1-PLAN-LOTE-106] el gesto vive en el hook compartido con el escáner (useBottomSheet)
    hook = _front("src/hooks/useBottomSheet.js")
    assert "el.addEventListener('touchmove', block, { passive: false });" in hook
    assert "if (!scrollable) { e.preventDefault(); return; }" in hook
    assert "if ((dir > 0 && atTop) || (dir < 0 && atBottom)) e.preventDefault();" in hook
    # mismos umbrales que la hoja de actualizar platos (v4)
    assert "if (y > 70 || vy > 0.35 || y + vy * 150 > 100) {" in hook
    jsx = _front("src/components/dashboard/LogMealModal.jsx")
    for h in ("onTouchStart={hoja.onTouchStart}", "onTouchMove={hoja.onTouchMove}", "onTouchEnd={hoja.onTouchEnd}", "onTouchCancel={hoja.onTouchEnd}"):
        assert h in jsx, h
    css = _front("src/components/dashboard/LogMealModal.module.css")
    i = css.index("\n.body {")
    assert "touch-action: pan-y;" in css[i:css.index("}", i)]


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 101
    assert "P1-PLAN-LOTE-101" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
