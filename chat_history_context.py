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


# ---------------------------------------------------------------------------
# Knobs — auto-registran en `_KNOBS_REGISTRY` (P3-NEW-D).
# tooltip-anchor: P1-CHAT-PAST-DAYS-KNOBS
# ---------------------------------------------------------------------------

def chat_history_days() -> int:
    """Ventana de días pasados de los bloques 2 y 3. `0` apaga ambos."""
    from knobs import _env_int
    return _env_int("MEALFIT_CHAT_HISTORY_DAYS", 7, validator=lambda v: 0 <= v <= 30)


def chat_history_max_chars() -> int:
    """Cap duro por bloque. Al excederse se recortan los días más antiguos."""
    from knobs import _env_int
    return _env_int("MEALFIT_CHAT_HISTORY_MAX_CHARS", 3000, validator=lambda v: 500 <= v <= 20000)


def _fmt_date_es(d: date, inferred: bool = False) -> str:
    """`Domingo 26 jul`; con `~` delante si la fecha es inferida, no estampada."""
    tag = "~" if inferred else ""
    return f"{tag}{_WEEKDAYS_ES[d.weekday()].capitalize()} {d.day} {_MONTHS_ES[d.month - 1]}"


def _assemble(header: str, lines: list, footer: str, max_chars: int, label: str) -> str:
    """Junta líneas (ya en orden newest-first) respetando el cap.

    Los días más antiguos se caen primero y el recorte se DECLARA en el texto —
    un cap silencioso se lee como "esto es todo lo que hay", que es justo la
    confusión que este P-fix existe para evitar.
    """
    budget = max(0, max_chars - len(header) - len(footer))
    kept, used, dropped = [], 0, 0
    for line in lines:
        if used + len(line) + 1 > budget:
            dropped += 1
            continue
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return ""
    if dropped:
        logger.info(f"[P1-CHAT-PAST-DAYS] {label}: {dropped} día(s) recortados por cap={max_chars}")
        footer = f"\n(+{dropped} día(s) más antiguos omitidos por espacio.)" + footer
    return header + "\n".join(kept) + footer


def build_past_plan_days_block(plan_data: Any, today: date,
                               days_back: Optional[int] = None,
                               max_chars: Optional[int] = None) -> str:
    """Pieza 2 del spec: índice compacto de lo que el plan MANDABA en los días
    que ya pasaron. Solo nombre + slot + kcal — las cantidades y las recetas
    las sirve la tool `consultar_dia_del_plan` bajo demanda."""
    days_back = chat_history_days() if days_back is None else days_back
    max_chars = chat_history_max_chars() if max_chars is None else max_chars
    if days_back <= 0:
        return ""
    floor = today - timedelta(days=days_back)
    past = [r for r in resolve_day_dates(plan_data, today) if floor <= r["date"] < today]
    if not past:
        return ""

    lines = []
    for r in sorted(past, key=lambda x: x["date"], reverse=True):
        parts = []
        for m in (r["day"].get("meals") or []):
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name:
                continue
            slot = str(m.get("meal") or m.get("meal_type") or "Comida").strip()
            cals = m.get("cals") if m.get("cals") is not None else m.get("calories")
            seg = f'{slot} "{name}"'
            if isinstance(cals, (int, float)) and cals:
                seg += f" {int(cals)} kcal"
            parts.append(seg)
        if parts:
            lines.append(f"- {_fmt_date_es(r['date'], r['inferred'])}: " + " · ".join(parts))
    if not lines:
        return ""

    header = ("\n\n📖 DÍAS QUE YA PASARON — esto es lo que el plan MANDABA esos días "
              "(NO es prueba de que el usuario se lo comiera):\n")
    footer = ("\nLas fechas con '~' son estimadas, no exactas. Para las cantidades, los gramos "
              "o los pasos de la receta de uno de estos días, usa la herramienta "
              "`consultar_dia_del_plan`.")
    return _assemble(header, lines, footer, max_chars, "plan_days")


def build_past_diary_block(consumed_rows: Any, today: date,
                           days_back: Optional[int] = None,
                           max_chars: Optional[int] = None) -> str:
    """Pieza 3 del spec: lo que el usuario REGISTRÓ haber comido en los días
    anteriores a hoy. Declara explícitamente los días sin registro — esa es la
    guarda que impide que el modelo rellene el hueco con el plan."""
    days_back = chat_history_days() if days_back is None else days_back
    max_chars = chat_history_max_chars() if max_chars is None else max_chars
    if days_back <= 0:
        return ""
    floor = today - timedelta(days=days_back)

    by_date: dict = {}
    for row in (consumed_rows or []):
        if not isinstance(row, dict):
            continue
        d = _parse_date(row.get("consumed_at"))
        if d is None or not (floor <= d < today):
            continue
        by_date.setdefault(d, []).append(row)

    lines = []
    cursor = today - timedelta(days=1)
    while cursor >= floor:
        rows = by_date.get(cursor)
        if rows:
            parts = []
            for r in rows:
                nm = str(r.get("meal_name") or "").strip() or "(sin nombre)"
                slot = str(r.get("meal_type") or "").strip()
                cal = r.get("calories")
                seg = f"{slot}: {nm}" if slot else nm
                if isinstance(cal, (int, float)) and cal:
                    seg += f" ({int(cal)} kcal)"
                parts.append(seg)
            lines.append(f"- {_fmt_date_es(cursor)}: " + " · ".join(parts))
        else:
            lines.append(f"- {_fmt_date_es(cursor)}: SIN REGISTRO")
        cursor -= timedelta(days=1)
    if not lines:
        return ""

    header = ("\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES — lo ÚNICO que sabes que el usuario "
              "comió esos días:\n")
    footer = ("\n⚠️ 'SIN REGISTRO' significa que NO tienes ningún dato de lo que comió ese día. "
              "Si te pregunta qué comió un día SIN REGISTRO, dile con claridad que no lo tienes "
              "registrado y NUNCA respondas con lo que el plan mandaba como si se lo hubiera "
              "comido. El bloque del PLAN es lo prescrito; este bloque es lo real.")
    return _assemble(header, lines, footer, max_chars, "diary")
