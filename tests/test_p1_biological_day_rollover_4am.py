"""[P1-DIARY-ROLLOVER-4AM · 2026-11-02] Test suite para el rollover del día biológico a las 4:00 AM.
Verifica que las comidas registradas en la madrugada (00:00 - 03:59:59 AM local) pertenezcan
al día biológico/despierto activo (fecha calendario anterior) y que a las 04:00 AM local
se reinicie el contador.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import constants
import db_facts
from tools import _diary_date_str_for_user


def test_diary_local_date_str_biological_cutoff():
    """Prueba que antes de las 4:00 AM local la fecha pertenezca al día biológico anterior."""
    tz_rd = 240  # UTC-4

    # 1:00 AM RD del 3 de nov = 05:00 UTC del 3 de nov
    t_1am = datetime(2026, 11, 3, 5, 0, 0, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_1am) == "2026-11-02"

    # 3:59:59 AM RD del 3 de nov = 07:59:59 UTC del 3 de nov
    t_359am = datetime(2026, 11, 3, 7, 59, 59, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_359am) == "2026-11-02"

    # 4:00:00 AM RD del 3 de nov = 08:00:00 UTC del 3 de nov -> Día nuevo
    t_4am = datetime(2026, 11, 3, 8, 0, 0, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_4am) == "2026-11-03"

    # 11:00 PM RD del 2 de nov = 03:00 UTC del 3 de nov
    t_11pm = datetime(2026, 11, 3, 3, 0, 0, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_11pm) == "2026-11-02"


def test_tools_diary_date_str_for_user_matches():
    """`tools._diary_date_str_for_user` debe delegar consistentemente en el cálculo biológico."""
    date_tools = _diary_date_str_for_user()
    date_db = db_facts.diary_local_date_str()
    assert date_tools == date_db


def test_get_consumed_meals_today_window_encloses_late_night(monkeypatch):
    """La ventana construida por `get_consumed_meals_today` para una fecha dada (ej. 2026-11-02)
    debe abarcar desde 04:00 AM local hasta 03:59:59 AM local del día siguiente."""
    capturado = {}

    def _fake_exec(query, params, **kwargs):
        capturado["start"] = params[1]
        capturado["end"] = params[2]
        return []

    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", _fake_exec, raising=True)

    db_facts.get_consumed_meals_today("u-test-rollover", date_str="2026-11-02", tz_offset_mins=240)
    assert capturado["start"] == "2026-11-02T08:00:00Z"
    assert capturado["end"] == "2026-11-03T07:59:59Z"

    # Comida a las 2:15 AM RD del 3 de nov (= 06:15:00Z) cae dentro de la ventana del 2 de nov
    meal_consumed_utc = "2026-11-03T06:15:00Z"
    assert capturado["start"] <= meal_consumed_utc < capturado["end"]


def test_custom_rollover_hour_knob(monkeypatch):
    """Verifica que el rollover respeta el knob si se configura a otra hora (ej. 5 AM)."""
    monkeypatch.setattr(constants, "DIARY_ROLLOVER_HOUR", 5, raising=True)

    tz_rd = 240  # UTC-4
    # A las 4:30 AM con rollover a las 5 AM sigue perteneciendo al día anterior
    t_430am = datetime(2026, 11, 3, 8, 30, 0, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_430am) == "2026-11-02"

    # A las 5:00 AM ya es el día nuevo
    t_5am = datetime(2026, 11, 3, 9, 0, 0, tzinfo=timezone.utc)
    assert db_facts.diary_local_date_str(tz_offset_mins=tz_rd, now_utc=t_5am) == "2026-11-03"
