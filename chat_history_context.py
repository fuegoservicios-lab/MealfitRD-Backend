"""[P1-CHAT-PAST-DAYS · 2026-07-27] Memoria de días pasados para el chat-agent.

Lógica PURA (sin DB, sin I/O): resolver la fecha calendario de cada día del
plan, y renderizar los bloques compactos que se inyectan al system prompt.

Doc canónica: backend/docs/chat_past_days_memory.md
tooltip-anchor: P1-CHAT-PAST-DAYS
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WEEKDAYS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS_ES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def rd_today() -> date:
    """Fecha local RD (UTC-4) — convención del repo."""
    return (datetime.now(timezone.utc) - timedelta(hours=4)).date()


def _parse_date(value: Any) -> Optional[date]:
    """ISO date/datetime → `date`. None ante cualquier shape rara."""
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _live_anchor(live: list, plan_data: dict, today: date):
    """(idx, fecha) del día vivo que sirve de ancla. Ver spec §3 Pieza 1.

    Prioridad: (1) primera fecha estampada; (2) day_name == weekday(hoy) — el
    MISMO criterio que `agent._build_plan_today_context`; (3) cycle_start_date.
    NUNCA `cycle_start_date + i` como ancla primaria: tras un shift days[0] es
    HOY, no el inicio del ciclo, y esa fórmula desplaza el plan entero.
    """
    for i, d in enumerate(live):
        stamped = _parse_date(d.get("date"))
        if stamped:
            return i, stamped
    wd_today = _WEEKDAYS_ES[today.weekday()]
    for i, d in enumerate(live):
        if str(d.get("day_name") or "").strip().lower() == wd_today:
            return i, today
    cs = _parse_date(plan_data.get("cycle_start_date")) or _parse_date(plan_data.get("grocery_start_date"))
    if cs:
        return 0, cs
    return None, None


def resolve_day_dates(plan_data: Any, today: date) -> list[dict]:
    """Fecha calendario de cada día del plan (archivados + vivos).

    Devuelve `[{"date", "day", "inferred", "archived"}]` en orden cronológico
    ascendente. Fail-open a `[]` ante cualquier shape rara.
    """
    try:
        if not isinstance(plan_data, dict):
            return []
        live = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        archived = [d for d in (plan_data.get("_archived_days") or []) if isinstance(d, dict)]
        if not live and not archived:
            return []

        anchor_idx, anchor_date = _live_anchor(live, plan_data, today)
        if anchor_date is None:
            return []

        rows: list[dict] = []
        for i, d in enumerate(live):
            stamped = _parse_date(d.get("date"))
            rows.append({
                "date": stamped or (anchor_date + timedelta(days=i - anchor_idx)),
                "day": d,
                "inferred": stamped is None,
                "archived": False,
            })

        # Los archivados son estrictamente anteriores al primer día vivo.
        first_live = rows[0]["date"] if rows else anchor_date
        n_arch = len(archived)
        for k, d in enumerate(archived):
            stamped = _parse_date(d.get("date"))
            rows.append({
                "date": stamped or (first_live - timedelta(days=(n_arch - k))),
                "day": d,
                "inferred": stamped is None,
                "archived": True,
            })

        rows.sort(key=lambda r: r["date"])
        return rows
    except Exception as e:
        logger.warning(f"[P1-CHAT-PAST-DAYS] resolve_day_dates fail-open: {e}")
        return []


def find_plan_day_for_date(plan_data: Any, target: date, today: date) -> Optional[dict]:
    """El registro (`resolve_day_dates`) cuya fecha sea exactamente `target`."""
    for row in resolve_day_dates(plan_data, today):
        if row["date"] == target:
            return row
    return None
