# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-200 · 2026-09-24] El gate temporal mide el fin del bloque anterior con las fechas del plan.

Producción: el gate sumaba `_shift_days_accumulated` a un ancla que el shift ya había movido a hoy; el 23-sep dijo que el
bloque previo terminaba el 27 cuando terminó el 22. Ver `fin_bloque_previo.py`.
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import fin_bloque_previo as fbp  # noqa: E402


def test_ultimo_dia_planificado():
    pd = {"days": [{"date": "2026-09-22"}, {"date": "2026-09-24T00:00:00"}, {"date": ""}, {"day": 4}, "x"]}
    assert fbp.ultimo_dia_planificado(pd) == date(2026, 9, 24)
    assert fbp.ultimo_dia_planificado({"days": [{"day": 1}]}) is None
    assert fbp.ultimo_dia_planificado(None) is None


def test_acota_pero_nunca_retrasa():
    pd = {"days": [{"date": "2026-09-22"}]}
    assert fbp.acotar(date(2026, 9, 27), pd) == date(2026, 9, 22), "el caso real: 27 por fórmula, 22 planificado"
    assert fbp.acotar(date(2026, 9, 20), pd) == date(2026, 9, 20), "si la fórmula da antes, no se toca"
    assert fbp.acotar(date(2026, 9, 27), {"days": [{"day": 1}]}) == date(2026, 9, 27), "sin fechas, la fórmula"
    assert fbp.acotar(None, pd) is None


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_GATE_PREV_END_FROM_DAYS", "false")
    assert fbp.acotar(date(2026, 9, 27), {"days": [{"date": "2026-09-22"}]}) == date(2026, 9, 27)


def _gate(plan_data, now):
    """El caso del 23-sep: ancla ya movida a hoy por el shift + acumulado 3 + bloque previo de 3 días en offset 0."""
    import cron_tasks
    hoy = now.replace(hour=0, minute=0, second=0, microsecond=0)
    snapshot = {"form_data": {"_plan_start_date": hoy.isoformat(), "tz_offset_minutes": 0}, "totalDays": 4}

    def _sql(q, *a, **k):
        return {"days_offset": 0, "days_count": 3} if "days_offset, days_count" in q else None

    with patch("cron_tasks._dt_p0b_now", return_value=now), \
            patch("cron_tasks.execute_sql_query", side_effect=_sql), \
            patch("cron_tasks.execute_sql_write", MagicMock()), \
            patch("cron_tasks._record_chunk_deferral", MagicMock()), \
            patch("cron_tasks._dispatch_push_notification", MagicMock()), \
            patch("cron_tasks.get_consumed_meals_since", return_value=[]), \
            patch("cron_tasks.get_inventory_activity_since", return_value={}):
        return cron_tasks._check_chunk_learning_ready(user_id="u", meal_plan_id="p", week_number=8, days_offset=1,
                                                      plan_data=plan_data, snapshot=snapshot)


def test_el_gate_ya_no_aplaza_por_el_shift_contado_dos_veces():
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    ayer = (now - timedelta(days=1)).date().isoformat()
    base = {"_shift_days_accumulated": 3, "days": [{"day": 1, "date": ayer, "meals": []}]}
    r = _gate(dict(base), now)
    assert r.get("reason") != "prev_chunk_day_not_yet_elapsed", r


def test_con_el_knob_apagado_vuelve_la_formula(monkeypatch):
    monkeypatch.setenv("MEALFIT_GATE_PREV_END_FROM_DAYS", "false")
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    ayer = (now - timedelta(days=1)).date().isoformat()
    r = _gate({"_shift_days_accumulated": 3, "days": [{"day": 1, "date": ayer, "meals": []}]}, now)
    assert r.get("reason") == "prev_chunk_day_not_yet_elapsed" and r.get("days_until_prev_end") == 5, r


def test_con_todos_los_dias_archivados_manda_el_archivo():
    """[P1-PLAN-LOTE-204] El shift archivó el bloque entero: sin días vivos, la fecha sale de `_archived_days`."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    ayer = (now - timedelta(days=1)).date().isoformat()
    antier = (now - timedelta(days=2)).date().isoformat()
    pd = {"_shift_days_accumulated": 3, "days": [], "_archived_days": [{"date": antier}, {"date": ayer}]}
    assert fbp.ultimo_dia_planificado(pd).isoformat() == ayer
    r = _gate(pd, now)
    assert r.get("reason") != "prev_chunk_day_not_yet_elapsed", r


def test_si_el_bloque_previo_no_termino_sigue_aplazando():
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    manana = (now + timedelta(days=1)).date().isoformat()
    r = _gate({"_shift_days_accumulated": 3, "days": [{"day": 1, "date": manana, "meals": []}]}, now)
    assert r.get("reason") == "prev_chunk_day_not_yet_elapsed" and r.get("days_until_prev_end") == 1, r


def test_el_gate_consulta_las_fechas_del_plan():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index('_prev_end_date = __import__("fin_bloque_previo").acotar(_prev_end_date, plan_data')
    j = src.index("_days_until_prev_end = (_prev_end_date - _today_user).days")
    assert i < j, "el acotado va ANTES de medir los días que faltan"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 200 and m.group(2) >= "2026-09-24"
