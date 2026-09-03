"""[P1-COACH-CYCLE-DAY-RD · 2026-08-23]

El coach recibe la fecha local correcta, pero también necesita el offset para
convertir el instante UTC de inicio del ciclo. El día k debe ser idéntico en
los siete offsets del producto.
"""
from __future__ import annotations

import re
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


def _plan_started_local_2026_08_20(tz_min: int) -> dict:
    start_utc = datetime(2026, 8, 20, tzinfo=timezone.utc) + timedelta(minutes=tz_min)
    return {
        "days": [{"day": 1, "day_name": "Lunes", "meals": []}],
        "cycle_start_date": start_utc.isoformat(),
        "calc_grocery_duration": "monthly",
    }


@pytest.mark.parametrize("label,tz_min", sorted(_OFFSETS.items()))
def test_coach_dice_dia_cuatro_en_los_siete_offsets(label, tz_min):
    from agent import _build_plan_today_context

    text = _build_plan_today_context(
        _plan_started_local_2026_08_20(tz_min),
        local_date_str="2026-08-23",
        tz_offset=tz_min,
    )
    match = re.search(r"Ciclo del plan: día (\d+) de 30", text)
    assert match, f"{label}: no apareció posición del ciclo: {text!r}"
    assert int(match.group(1)) == 4, f"{label}: el coach dijo día {match.group(1)}"


def test_fallback_legacy_usa_el_ssot_dominicano():
    from agent import _build_plan_today_context

    plan = _plan_started_local_2026_08_20(240)
    explicit = _build_plan_today_context(
        plan, local_date_str="2026-08-23", tz_offset=240
    )
    legacy = _build_plan_today_context(plan, local_date_str="2026-08-23")
    assert legacy == explicit


def test_stream_nonstream_y_router_enhebran_el_offset():
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    router = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")

    call = "_build_plan_today_context(plan_vigente, local_date_str=local_date, tz_offset=tz_offset)"
    assert agent.count(call) >= 2, "stream y non-stream deben pasar el mismo par fecha/offset"

    nonstream_start = router.index("def api_chat(")
    nonstream = router[nonstream_start:]
    assert 'local_date = data.get("local_date", None)' in nonstream
    assert 'tz_offset = data.get("tz_offset", None)' in nonstream
    assert "chat_with_agent(" in nonstream
    assert "local_date=local_date" in nonstream
    assert "tz_offset=tz_offset" in nonstream


def test_marker_movil_y_ancla_del_gap_existen():
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")

    assert "P1-COACH-CYCLE-DAY-RD" in agent
    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
