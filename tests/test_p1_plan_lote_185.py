# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-185 · 2026-09-23] Batería rd14 (DM2 + insulina): rechazo CRÍTICO del revisor por «batidos con piña/guineo
o lechosa/guineo» (fruta licuada, carga glucémica). La regla DM2 del prompt pedía «fruta entera con cáscara» pero no
prohibía los batidos de fruta; el tope determinista (lote 182, ≤120 g de fruta dulce por línea) no cubre la forma."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_la_regla_dm2_prohibe_la_fruta_licuada():
    from condition_rules import CONDITION_RULES
    dm2 = next(r for r in CONDITION_RULES if r.id == "dm2")
    assert "NADA de batidos ni licuados con fruta" in dm2.prompt_block
    assert "nada de majarete ni dulces de maíz" in dm2.prompt_block, "lo anterior sigue"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 185 and m.group(2) >= "2026-09-23"
