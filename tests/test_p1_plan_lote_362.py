# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-362 · 2026-09-26] La respuesta de una duda también cambia el INGREDIENTE.

El dueño: «✓ 4 huevos» arriba y «Huevo revuelto · 2 unidades» abajo. El frontend ya lleva la respuesta al ingrediente
de la duda (cantidad si tiene número, nombre si no; `frontend/src/__tests__/lote362.test.js`); para casar SIN
adivinar, la visión pone en `sobre` el `name` EXACTO del item al que se refiere. Tooltip-anchor: P1-PLAN-LOTE-362
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_la_vision_nombra_el_item_exacto_de_la_duda():
    import vision_agent as va
    assert "'sobre' (el 'name' EXACTO del item de 'items'" in va._MEAL_VISION_PROMPT
    va._MEAL_VISION_PROMPT.encode("ascii")
    assert "EXACTO" in va._MealVisionDuda.model_fields["sobre"].description


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 362
