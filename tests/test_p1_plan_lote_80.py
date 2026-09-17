# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-80 · 2026-09-17] El coach no pregunta la FRANJA de una comida que el usuario ya contó en pasado: en la 3.ª
corrida de la batería, «ayer me comí un chimi en la calle» recibió «¿fue el almuerzo o la cena?» en vez del registro (en las
otras dos corridas registró como cena de ayer). El dueño pidió proactividad: asume la franja más probable por el plato y la
hora, registra y lo dice; una corrección posterior va por `correct_consumed_meal`. La cláusula vive en las DOS variantes de
las instrucciones de herramientas (inline y stream)."""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_la_franja_no_se_pregunta_en_las_dos_variantes():
    p = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert p.count("[P1-PLAN-LOTE-80] Tampoco le preguntes la FRANJA si no la dice") == 2
    assert p.count("«ayer me comí un chimi en la calle» → cena de AYER") == 2
    i1 = p.index("[P1-PLAN-LOTE-80] Tampoco le preguntes la FRANJA")
    assert "no le preguntes si se lo comió ni si lo registras" in p[i1 - 200:i1]   # va justo tras la regla de actuar


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 80
    assert "P1-PLAN-LOTE-80" in (_BACKEND / "docs" / "coach_bateria_2026_09_15.md").read_text(encoding="utf-8")
