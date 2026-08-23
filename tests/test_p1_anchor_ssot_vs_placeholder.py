"""[P1-ANCHOR-SSOT-VS-PLACEHOLDER · 2026-08-23]

Las tres fuentes resolubles deben devolver la MISMA semántica: el instante UTC
de la medianoche local. ``profile_today``/``last_plan`` no pueden devolver un
marcador de fecha UTC y confiar en que cada consumidor adivine qué significa.
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
_NOW_UTC = datetime(2026, 8, 23, 10, tzinfo=timezone.utc)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW_UTC if tz is not None else _NOW_UTC.replace(tzinfo=None)


def _real_local_midnight(tz_min: int) -> datetime:
    local_date = (_NOW_UTC - timedelta(minutes=tz_min)).date()
    return datetime.combine(local_date, datetime.min.time()).replace(
        tzinfo=timezone.utc
    ) + timedelta(minutes=tz_min)


def _assert_targets_share_semantics(anchor: datetime, tz_min: int, label: str):
    from constants import chunk_anchor_local_midnight_utc

    expected = _real_local_midnight(tz_min)
    assert anchor == expected, f"{label}: el resolver devolvió {anchor}, esperado {expected}"

    fresh_target = chunk_anchor_local_midnight_utc(anchor, tz_min) + timedelta(
        days=3, minutes=30
    )
    real_target_midnight = expected + timedelta(days=3)
    delta_hours = (fresh_target - real_target_midnight).total_seconds() / 3600
    assert 0 <= delta_hours <= 1, f"{label}: execute_after cae {delta_hours:+.1f} h"

    # El retry usa el ancla cruda. Sólo es coherente si todas las fuentes devuelven
    # el mismo tipo de instante; no debe cambiar de significado según la fuente.
    retry_target = anchor + timedelta(days=3, hours=-3)
    assert retry_target == expected + timedelta(days=3, hours=-3)


@pytest.mark.parametrize("label,tz_min", sorted(_OFFSETS.items()))
@pytest.mark.parametrize("anchor_form", ["snapshot", "fallbacks"])
def test_catorce_combinaciones_comparten_instante_local(
    monkeypatch, anchor_form, label, tz_min
):
    import cron_tasks

    monkeypatch.setattr(cron_tasks, "datetime", _FrozenDateTime)
    monkeypatch.setattr(cron_tasks, "_record_tz_fallback", lambda *args, **kwargs: False)
    expected = _real_local_midnight(tz_min)

    if anchor_form == "snapshot":
        monkeypatch.setattr(cron_tasks, "_get_user_tz_live", lambda *args, **kwargs: tz_min)
        anchor, got_tz, source = cron_tasks._resolve_chunk_start_anchor(
            "user",
            {"form_data": {"_plan_start_date": expected.isoformat(), "tzOffset": tz_min}},
            "plan",
            2,
        )
        assert source == "snapshot" and got_tz == tz_min
        _assert_targets_share_semantics(anchor, got_tz, f"{label}/snapshot")
        return

    # La segunda forma cubre LAS DOS fuentes que antes fabricaban el mismo marcador UTC.
    monkeypatch.setattr(cron_tasks, "_get_user_tz_minutes_optional", lambda *_: tz_min)
    profile_anchor, profile_tz, profile_source = cron_tasks._resolve_chunk_start_anchor(
        "user", {"form_data": {}}, "plan", 2
    )
    assert profile_source == "profile_today" and profile_tz == tz_min
    _assert_targets_share_semantics(profile_anchor, profile_tz, f"{label}/profile_today")

    monkeypatch.setattr(cron_tasks, "_get_user_tz_minutes_optional", lambda *_: None)
    monkeypatch.setattr(
        cron_tasks,
        "execute_sql_query",
        lambda *args, **kwargs: {
            "id": "prior",
            "plan_data": {"health_profile": {"tz_offset_minutes": tz_min}},
        },
    )
    last_anchor, last_tz, last_source = cron_tasks._resolve_chunk_start_anchor(
        "user", {"form_data": {}}, "plan", 2
    )
    assert last_source == "last_plan" and last_tz == tz_min
    _assert_targets_share_semantics(last_anchor, last_tz, f"{label}/last_plan")


def test_marker_movil_y_fuentes_normalizadas_existen():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    assert "P1-ANCHOR-SSOT-VS-PLACEHOLDER" in cron
    assert "_local_midnight_for_fallback" in cron
