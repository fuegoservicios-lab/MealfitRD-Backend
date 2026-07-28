"""[P1-CHAT-PAST-DAYS · 2026-07-27] Test ancla del P-fix.

Spec: backend/docs/chat_past_days_memory.md
"""
import os
import re
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chat_history_context import resolve_day_dates, find_plan_day_for_date  # noqa: E402

HOY = date(2026, 7, 27)  # lunes


def _day(name, n, meals=None, fecha=None):
    d = {"day": n, "day_name": name, "meals": meals if meals is not None else []}
    if fecha:
        d["date"] = fecha
    return d


def test_ancla_1_fecha_estampada_gana():
    """Si el día trae `date`, es autoritativa y `inferred` es False."""
    plan = {"days": [_day("Lunes", 1, fecha="2026-07-27"), _day("Martes", 2, fecha="2026-07-28")]}
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["inferred"] for r in rows] == [False, False]


def test_ancla_2_weekday_match_para_plan_shifteado():
    """Sin `date`, el ancla es el day_name que coincide con HOY.

    Regresión: `cycle_start_date + i` desplazaría el plan entero, porque tras
    un shift days[0] es HOY y NO el inicio del ciclo.
    """
    plan = {
        "days": [_day("Lunes", 1), _day("Martes", 2), _day("Miércoles", 3)],
        "cycle_start_date": "2026-07-20T10:00:00+00:00",  # una semana antes: trampa
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    assert all(r["inferred"] for r in rows)


def test_ancla_3_cycle_start_cuando_ningun_day_name_es_hoy():
    """Plan que aún no shifteó y cuyo primer día ya pasó."""
    plan = {
        "days": [_day("Sábado", 1), _day("Domingo", 2)],
        "cycle_start_date": "2026-07-25T10:00:00+00:00",  # sábado
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26)]


def test_archivados_van_antes_del_primer_dia_vivo():
    plan = {
        "days": [_day("Lunes", 1)],
        "_archived_days": [_day("Sábado", 1), _day("Domingo", 2)],
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26), date(2026, 7, 27)]
    assert [r["archived"] for r in rows] == [True, True, False]


def test_find_plan_day_for_date_encuentra_en_el_archivo():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 2)]}
    got = find_plan_day_for_date(plan, date(2026, 7, 26), HOY)
    assert got is not None and got["day"]["day_name"] == "Domingo"
    assert find_plan_day_for_date(plan, date(2026, 1, 1), HOY) is None


@pytest.mark.parametrize("basura", [None, "no soy un dict", 42, [], {"days": "tampoco"}])
def test_fail_open_ante_shapes_raras(basura):
    assert resolve_day_dates(basura, HOY) == []
