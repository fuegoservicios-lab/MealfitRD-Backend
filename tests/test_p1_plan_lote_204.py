# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-204 · 2026-09-24] El fin del bloque previo cuenta también los días archivados.

Hueco del 200: con todos los días vivos ya archivados por el shift, `plan_data.days` llega vacío al gate y la fórmula
inflada volvía justo para el usuario sin días. El test funcional vive en `test_p1_plan_lote_200.py`.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import fin_bloque_previo as fbp  # noqa: E402


def test_vivos_y_archivados_cuentan():
    pd = {"days": [{"date": "2026-09-24"}], "_archived_days": [{"date": "2026-09-22"}, {"date": "2026-09-23"}]}
    assert fbp.ultimo_dia_planificado(pd) == date(2026, 9, 24)
    assert fbp.ultimo_dia_planificado({"days": [], "_archived_days": [{"date": "2026-09-23"}]}) == date(2026, 9, 23)
    assert fbp.ultimo_dia_planificado({"days": [], "_archived_days": "x"}) is None


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 204 and m.group(2) >= "2026-09-24"
