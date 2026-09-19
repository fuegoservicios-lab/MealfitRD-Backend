# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-106 · 2026-09-18] El escáner de fotos, rehecho para el teléfono y con «¿Cuándo?» (hoy/ayer/antier).

El dueño, con captura: «quiero que sea más cómodo, fácil de entender, de interactuar y también agrega algo para poder
seleccionar el día, ayer y antier, ya que esa es mi cena del día de ayer que no pude agregar». La misma hoja que el
componedor (gesto extraído a `useBottomSheet`, chips extraídos a `Chips.jsx`), cuatro preguntas en vez de una columna
de campos, y `days_ago` —que el backend aceptaba desde P1-DIARY-EDITABLE y el escáner nunca mandaba—. Ancla cross-repo;
el contrato fino vive en `lote106.test.jsx`."""
from __future__ import annotations

import json
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


def test_la_hoja_y_el_gesto_son_los_del_componedor():
    hook = _front("src/hooks/useBottomSheet.js")
    assert "export function useBottomSheet({ containerRef, bodyRef, onClose, disabled = false })" in hook
    assert "if (y > 70 || vy > 0.35 || y + vy * 150 > 100) {" in hook
    assert "const scrollY0 = window.scrollY;" in hook
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    lm = _front("src/components/dashboard/LogMealModal.jsx")
    assert "const hoja = useBottomSheet({ containerRef, bodyRef, onClose, disabled: isBusy });" in sm
    assert "const hoja = useBottomSheet({ containerRef, bodyRef, onClose, disabled: saving });" in lm
    for src in (sm, lm):
        assert "import Chips from './Chips';" in src
        assert "onTouchStart={hoja.onTouchStart}" in src and "onTouchCancel={hoja.onTouchEnd}" in src
    # el gesto no queda duplicado en ningún modal
    assert "gestureRef" not in sm and "gestureRef" not in lm
    css = _front("src/components/dashboard/ScanMealModal.module.css")
    for sel in ("\n.body {", "\n.footer {", "\n.grip {", "\n.section {", "\n.sectionTitle {"):
        assert sel in css, sel
    i = css.index("\n.card {")
    assert "flex-direction: column;" in css[i:css.index("}", i)]
    i = css.index("\n.overlay {")
    assert "align-items: flex-end;" in css[i:css.index("}", i)]


def test_la_revision_son_cuatro_preguntas_y_manda_days_ago():
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    for q in ("t('¿Qué es?')", "t('¿Cuánto comiste?')", "t('¿Qué comida es?')", "t('¿Cuándo?')", "t('Revisa y registra')"):
        assert q in sm, q
    assert "<select" not in sm
    assert "days_ago: daysAgo," in sm
    assert "options={_getDayOptions(t)} value={daysAgo} onChange={setDaysAgo}" in sm
    assert "t('Quedó en el diario de {dia}; la ves en «Ver días anteriores».', { dia })" in sm
    # el backend acepta 0..7 en el POST del escáner (ConsumedMealRequest)
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "days_ago: int = Field(default=0, ge=0, le=7)" in diary
    css = _front("src/components/dashboard/ScanMealModal.module.css")
    assert ".selectInput" not in css
    i = css.index("\n.textInput {")
    assert "font-size: 1rem;" in css[i:css.index("}", i)] and "background-image" not in css[i:css.index("}", i)]


def test_i18n_y_marcador():
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        d = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        for k in ("Revisa y registra", "¿Qué es?", "¿Cuánto comiste?", "antier",
                  "Quedó en el diario de {dia}; la ves en «Ver días anteriores».",
                  "La IA estimó esto por la foto. Corrige lo que no cuadre."):
            assert d.get(k), f"{loc}: {k}"
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 106
    assert "P1-PLAN-LOTE-106" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
