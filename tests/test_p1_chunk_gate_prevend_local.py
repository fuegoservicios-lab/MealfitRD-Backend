"""[P1-CHUNK-GATE-PREVEND-LOCAL · 2026-08-23]

El gate temporal debe esperar el mismo último día LOCAL del bloque previo en
los siete offsets del producto. Prueba el helper real del worker, no una copia
de la fórmula.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]
_OFFSETS = {
    "DO": 240,
    "PR": 240,
    "CO": 300,
    "MX": 360,
    "US-Pac": 420,
    "ES-verano": -120,
    "ES-invierno": -60,
}


def _local_midnight_utc(year: int, month: int, day: int, tz_min: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc) + timedelta(minutes=tz_min)


@pytest.mark.parametrize("label,tz_min", sorted(_OFFSETS.items()))
def test_gate_espera_el_mismo_fin_local_en_los_siete_offsets(monkeypatch, label, tz_min):
    import cron_tasks

    plan_start = _local_midnight_utc(2026, 8, 20, tz_min)
    local_noon_utc = datetime(2026, 8, 22, 12, tzinfo=timezone.utc) + timedelta(
        minutes=tz_min
    )
    monkeypatch.setattr(cron_tasks, "_dt_p0b_now", lambda _tz=None: local_noon_utc)
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *args, **kwargs: None)
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(cron_tasks, "_record_chunk_deferral", lambda *args, **kwargs: None)
    monkeypatch.setattr(cron_tasks, "_dispatch_push_notification", lambda *args, **kwargs: None)
    monkeypatch.setattr(cron_tasks, "get_consumed_meals_since", lambda *args, **kwargs: [])
    monkeypatch.setattr(cron_tasks, "get_inventory_activity_since", lambda *args, **kwargs: {})

    result = cron_tasks._check_chunk_learning_ready(
        user_id=f"user-{label}",
        meal_plan_id=f"plan-{label}",
        week_number=2,
        days_offset=3,
        plan_data={"days": [{"day": day, "meals": []} for day in (1, 2, 3)]},
        snapshot={
            "totalDays": 30,
            "form_data": {
                "_plan_start_date": plan_start.isoformat(),
                "tzOffset": tz_min,
                "tz_offset_minutes": tz_min,
            },
        },
    )

    assert result.get("reason") == "prev_chunk_day_not_yet_elapsed", (
        f"{label}: el gate abrió antes de concluir el día local previo: {result}"
    )
    assert result.get("days_until_prev_end") == 0, (
        f"{label}: el mismo 22-ago local debe producir delta cero"
    )


def test_marker_movil_y_ssot_del_gate_existen():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    assert "P1-CHUNK-GATE-PREVEND-LOCAL" in cron
    assert "_calmu_gate" in cron
