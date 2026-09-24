# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-202 · 2026-09-24] El delta de la lista de compras (frontend) no da por caducados los secos.

Hermano del 201 en el Dashboard: `buildDeltaShoppingList` descartaba del inventario lo «caducado» con el
`shelf_life_days` de relleno (14) y la lista volvía a pedir pasta, avena, habichuelas secas y especias a los 14 días.
El contrato vive en `frontend/src/__tests__/lote202.test.js`; aquí, el marker y el cableado leído desde el backend.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_el_dashboard_usa_el_mayor_de_los_dos_plazos():
    dash = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    assert "const shelfLife = Math.max(Number(item.master_ingredients?.shelf_life_days) || 0, inferShelfLifeDays(name, category));" in dash
    assert "item.master_ingredients?.shelf_life_days || inferShelfLifeDays(name, category)" not in dash


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 202 and m.group(2) >= "2026-09-24"
