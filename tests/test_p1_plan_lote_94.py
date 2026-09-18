# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-94 · 2026-09-17] Dos defectos encontrados en la revisión a fondo del móvil que pidió el dueño.

1. El desplegable del tipo de comida (componedor) cortaba su propio valor por defecto: «Extra (fuera del» a 392px y
   «Extra (fuel» a 320px. Los dos selects se repartían la fila a partes iguales (`flex: 1`) aunque, medido con la
   tipografía real, uno pide 185px y el otro 85; un `<select>` nativo corta sin puntos suspensivos.
2. En tema CLARO, desde que el contador va sin tarjeta (lote 88), su texto queda sobre la imagen decorativa del dashboard:
   el subtítulo `#64748B` ronda 4,1:1 (sobre la tarjeta blanca daba 4,9:1), por debajo del 4,5:1 de AA."""
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


def test_el_componedor_ya_no_tiene_desplegables_que_cortar():
    """[P1-PLAN-LOTE-99 · 2026-09-18] SUPERSEDED: el tipo de comida y el día son chips con todas las opciones a la
    vista; no queda `<select>` que corte su valor. El contrato nuevo vive en `test_p1_plan_lote_99.py`."""
    css = _front(_DASH + "LogMealModal.module.css")
    assert ".selectors {" not in css and "\n.select {" not in css


def test_en_claro_el_contador_se_apoya_en_el_color_de_pagina():
    css = _front(_DASH + "DashboardTracking.module.css")
    bloque = _plano(css[css.rindex("@media (max-width: 768px) {"):])
    assert ':global(html:not([data-theme="dark"])) .page { background: var(--bg-page' in bloque
    assert ':global(html[data-theme="dark"]) .page { background' not in bloque, "en oscuro el degradado es del dueño"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 94
    assert "P1-PLAN-LOTE-94" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
