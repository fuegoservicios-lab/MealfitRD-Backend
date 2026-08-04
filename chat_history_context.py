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


def _to_local_date(value: Any, tz_offset_mins: int = 240) -> Optional[date]:
    """Fecha LOCAL DEL USUARIO de un instante persistido en UTC.

    Dos consumidores, el mismo modo de fallo:

    - `consumed_at` (timestamptz UTC). Tomar `.date()` directo atribuye al día
      SIGUIENTE todo lo comido después de las 20:00 hora RD, y de paso declara
      'SIN REGISTRO' el día real — justo la mentira que este bloque existe para
      impedir.
    - `grocery_start_date` / `cycle_start_date` del plan, que la flota persiste
      en DOS shapes: date-only (`"2026-07-28"`, ver marker P3-SHIFT-DATEONLY-LOCAL)
      y timestamp completo (`"2026-07-28T02:00:00+00:00"`). Un plan shifteado a
      las 22:00 RD se guarda como 02:00 UTC del día siguiente: sin convertir, el
      ancla del plan entero nace un día tarde.

    Los strings de 10 chars ya SON una fecha local: se devuelven tal cual.
    Convención del repo: RD = UTC-4 (tz_offset_mins=240).
    tooltip-anchor: P1-CHAT-PAST-DAYS-TZ-CONSUMED
    """
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 10:  # 'YYYY-MM-DD': ya es una fecha, no hay hora que convertir
        return _parse_date(text)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return _parse_date(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return (dt - timedelta(minutes=tz_offset_mins)).date()


def _live_anchor(live: list, plan_data: dict, today: date):
    """(idx, fecha) del día vivo que sirve de ancla. Ver spec §3 Pieza 1.

    Cuatro tiers, en este orden y por estas razones (las 2ª y 3ª nacieron de dos
    fallos reproducidos contra planes de PRODUCCIÓN el 2026-07-27):

    1. Primera fecha ESTAMPADA en un día vivo. La única autoritativa
       (`inferred=False`); las otras tres marcan `inferred=True`.
    2. `grocery_start_date` → la fecha de `days[0]`. Es el campo que el shift
       REESCRIBE a HOY en cada rotación (`routers/plans.py` ~2575 y su gemelo de
       `cron_tasks.py`), así que es el que de verdad sigue a `days[0]`.
       Se pasa por `_to_local_date` porque la flota lo persiste en dos shapes
       (date-only y timestamp UTC); sin convertir, un plan shifteado a las 22:00
       RD ancla un día tarde.
    3. `day_name == weekday(hoy)` — el MISMO criterio que
       `agent._build_plan_today_context`. Degradado a FALLBACK: en una ventana
       viva de más de 7 días el `day_name` se REPITE y este `for` coge la
       PRIMERA coincidencia, así que un desfase de 1 día se convierte en uno de
       7. Plan real `69f9e03d` (26 días vivos): emitía 7 líneas de "días que ya
       pasaron" de las cuales 6 eran días FUTUROS (28 jul – 2 ago).
    4. `cycle_start_date` — último recurso. Es el ancla INMUTABLE de creación
       (`routers/plans.py:7754`), NO la de `days[0]`: tras un shift desplaza el
       plan entero hacia atrás. Plan real `4d2c1111` (17 archivados,
       grocery=28-jul, cycle=11-jul): fechaba el plan 24-jun…12-jul, fuera de la
       ventana de 7 días, y ese usuario se quedaba SIN bloque.

    NUNCA `cycle_start_date + i` como ancla primaria: tras un shift `days[0]` es
    HOY, no el inicio del ciclo, y esa fórmula desplaza el plan entero.
    tooltip-anchor: P1-CHAT-PAST-DAYS-ANCHOR
    """
    for i, d in enumerate(live):
        stamped = _parse_date(d.get("date"))
        if stamped:
            return i, stamped
    gsd = _to_local_date(plan_data.get("grocery_start_date"))
    if gsd:
        return 0, gsd
    wd_today = _WEEKDAYS_ES[today.weekday()]
    for i, d in enumerate(live):
        if str(d.get("day_name") or "").strip().lower() == wd_today:
            return i, today
    cs = _to_local_date(plan_data.get("cycle_start_date"))
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


def compute_chunk_overdue(plan_data, in_flight_count, today=None):
    """[P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] ¿Hoy debería existir un día del plan que no
    existe, sin nada en la cola que lo vaya a resolver solo?

    SSOT del predicado — 3 consumidores (/chunk-status, cron horario, índice del coach).
    Fail-open doble: plan legacy sin `date` estampada ⇒ (False, None) (no alerta en falso);
    cualquier excepción ⇒ (False, None). `today` inyectable para tests deterministas; en
    producción usa rd_today() (date-only TZ RD — NUNCA datetime.utcnow()).

    [Ronda 2 · fix revisor] `days` es una VENTANA ROLLING: cada shift poda los
    días ya vividos hacia `_archived_days` y renumera `days` 1..N, mientras
    `total_days_requested` nunca se toca. El guard "¿el plan ya entregó todo?"
    NO puede comparar `total_days_requested` contra `len(days)` — eso
    subcuenta cualquier plan que ya rotó al menos una vez. Reproducido
    ejecutando la función: plan de 20 días YA COMPLETO (15 archivados + 5
    vivos), hoy 4 días después del último día vivo → `(True, '2026-08-13')`,
    un falso positivo PERMANENTE (todo plan terminado cuyo usuario pasó el
    último día queda ATRASADO para siempre — el cron horario emitiría
    `chunk_overdue:<plan_id>` indefinidamente sobre un plan ya completo, el
    modo de fallo invertido que la alerta existe para evitar). Fix: mismo
    patrón que `resolve_day_dates` (arriba) y que
    `build_pending_plan_days_lines` ya aplica — `n_generated = len(archived)
    + len(days)`. La fecha ancla (`last`) sigue viniendo SOLO de `days` (los
    archivados son estrictamente anteriores, ver `_live_anchor`)."""
    try:
        if not isinstance(plan_data, dict):
            return (False, None)
        days = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        if not days:
            return (False, None)
        archived = [d for d in (plan_data.get("_archived_days") or []) if isinstance(d, dict)]
        n_generated = len(archived) + len(days)
        total = int(plan_data.get("total_days_requested") or 0)
        if total <= n_generated or int(in_flight_count or 0) > 0:
            return (False, None)
        last = None
        for d in days:
            pd = _parse_date(d.get("date"))
            if pd is not None and (last is None or pd > last):
                last = pd
        if last is None:
            return (False, None)  # legacy sin dates: fail-open
        _today = today or rd_today()
        if _today > last:
            return (True, (last + timedelta(days=1)).isoformat())
        return (False, None)
    except Exception:
        return (False, None)


def build_pending_plan_days_lines(plan_data: dict, today: date, in_flight_count: int) -> list[str]:
    """[P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] Índice de los días que el plan
    TODAVÍA NO tiene generados — para que el coach declare PENDIENTE/ATRASADO
    en vez de inventar un menú que aún no existe (mismo modo de fallo que
    `DÍAS QUE YA PASARON` cierra hacia atrás, este lo cierra hacia adelante).

    [Ronda 1 · fix ALTO 1] `days` es una VENTANA ROLLING: cada shift poda los
    días vividos hacia `_archived_days` y renumera `days` 1..N, mientras
    `total_days_requested` nunca se toca. Contar/numerar solo contra `days`
    (como hacía la v1 de esta función) sobrecuenta pendientes y desnumera en
    CUALQUIER plan que ya rotó al menos una vez — reproducido con 10
    archivados + 5 vivos + total 20 (15 generados, 5 pendientes reales):
    la v1 declaraba 15 "pendientes" numerados desde el 4. El fix reusa la
    MISMA técnica que `resolve_day_dates` arriba: la unión `_archived_days +
    days` es el conteo real de días ya generados (`n_generated`), y
    `total_days_requested - n_generated` es el pendiente real. La fecha ancla
    (`last`) sigue viniendo SOLO de `days` (los archivados son estrictamente
    anteriores — ver comentario de `_live_anchor`), pero la NUMERACIÓN del
    día (`día N`) arranca en `n_generated + 1`, no en `len(days) + 1`.

    Deriva las fechas esperadas como `última date estampada + i` (i=1..N días
    pendientes) y usa `compute_chunk_overdue` (SSOT ya usado por
    `/chunk-status` y el cron horario) para decidir si el plan está ATRASADO;
    un día individual solo se marca ATRASADO si el flag global es True Y su
    fecha esperada ya llegó (`<= today`) — el resto son PENDIENTE. Cap DURO
    de 3 líneas de día + 1 línea resumen ("… y N día(s) más pendientes") si
    sobran más.

    [Ronda 1 · MEDIO 4] Este cap NO pasa por `_assemble()`/`chat_history_max_chars()`
    (el presupuesto de chars compartido de los otros dos bloques del módulo).
    Es deliberado, no un descuido: el volumen de ESTE bloque está acotado
    ESTRUCTURALMENTE por `total_days_requested` (máximo 4 líneas, ~400 chars
    en el peor caso), mientras que los bloques hermanos (`build_past_plan_days_block`,
    `build_past_diary_block`) llevan texto libre sin cota (nombres de recetas,
    N días de historial) y por eso SÍ necesitan recorte dinámico. Compartir el
    pool de `_assemble` aquí sería presupuesto para un problema que este
    bloque no tiene.

    [Ronda 1 · decisión (b)] El "ATRASADO desde el <fecha>" de cada línea es
    la fecha esperada de ESE día individual — NO el `overdue_since` global
    que devuelven `/chunk-status` y la alerta del cron (que siempre es
    `último día vivo + 1`, fijo, sin importar cuántos días lleve atrasado el
    plan). Es una segunda fuente de verdad DELIBERADA (más útil para el
    coach, que habla de UN día a la vez) — un operador que cruce el chat
    contra `/chunk-status` durante un incidente puede ver fechas "desde"
    distintas para días distintos sin que eso sea drift entre sistemas.

    Fail-open a `[]`: plan legacy sin ninguna `date` estampada en `days`
    (mismo criterio que `compute_chunk_overdue` — los archivados nunca son la
    única fuente de fecha, ver `_live_anchor`), plan ya completo (`total <=
    n_generated`), o cualquier excepción — sin este índice el coach
    simplemente no menciona los días futuros, que es el comportamiento previo
    (no regresión).
    tooltip-anchor: P2-CHUNK-OVERDUE-SIGNAL-COACH
    """
    try:
        if not isinstance(plan_data, dict):
            return []
        days = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        if not days:
            return []
        archived = [d for d in (plan_data.get("_archived_days") or []) if isinstance(d, dict)]
        n_generated = len(archived) + len(days)
        total = int(plan_data.get("total_days_requested") or 0)
        n_pending = total - n_generated
        if n_pending <= 0:
            return []

        last = None
        for d in days:
            pd = _parse_date(d.get("date"))
            if pd is not None and (last is None or pd > last):
                last = pd
        if last is None:
            return []  # legacy sin dates: fail-open, mismo criterio que compute_chunk_overdue

        overdue, _since = compute_chunk_overdue(plan_data, in_flight_count, today=today)

        cap = 3
        shown = min(n_pending, cap)
        lines: list[str] = []
        for k in range(1, shown + 1):
            day_num = n_generated + k
            expected = last + timedelta(days=k)
            fecha = _fmt_date_es(expected)
            if overdue and expected <= today:
                lines.append(f"- día {day_num} ({fecha}): ATRASADO desde el {fecha}")
            else:
                lines.append(f"- día {day_num} ({fecha}): PENDIENTE — se genera por etapas")

        remaining = n_pending - shown
        if remaining > 0:
            lines.append(f"… y {remaining} día(s) más pendientes")
        return lines
    except Exception as e:
        logger.warning(f"[P2-CHUNK-OVERDUE-SIGNAL] build_pending_plan_days_lines fail-open: {e}")
        return []


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


def _plan_day_tool_footer_sentence() -> str:
    """[P1-CHAT-PAST-DAYS · 2026-07-28] Sombra de `tools._chat_plan_day_tool_enabled`
    sobre el footer de `build_past_plan_days_block`: con el kill switch OFF la tool
    `consultar_dia_del_plan` sale de `agent_tools`, así que este footer no puede
    seguir remitiendo a ella. Se lee el knob DIRECTO (no se importa `tools`, que
    es pesado) — mismo patrón que `chat_history_days`/`chat_history_max_chars` de
    arriba: import inline, módulo import-leaf. Default seguro True, igual al del
    knob real, para que un import fallido no borre copy que sí funciona.
    tooltip-anchor: P1-CHAT-PAST-DAYS-TOOL-KNOB
    """
    from knobs import _env_bool
    if not _env_bool("MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED", True):
        return ""
    return (" Para las cantidades, los gramos o los pasos de la receta de uno de estos "
            "días, usa la herramienta `consultar_dia_del_plan`.")


def _fmt_date_es(d: date, inferred: bool = False) -> str:
    """`Domingo 26 jul`; con `~` delante si la fecha es inferida, no estampada."""
    tag = "~" if inferred else ""
    return f"{tag}{_WEEKDAYS_ES[d.weekday()].capitalize()} {d.day} {_MONTHS_ES[d.month - 1]}"


def _assemble(header: str, lines: list, footer: str, max_chars: int, label: str) -> str:
    """Junta líneas (ya en orden newest-first) respetando el cap DURO.

    Los días más antiguos se caen primero y el recorte se DECLARA en el texto —
    un cap silencioso se lee como "esto es todo lo que hay", que es justo la
    confusión que este P-fix existe para evitar. La nota de recorte ocupa espacio,
    así que se aplica punto fijo: se descuenta del presupuesto ANTES de decidir
    qué cae. Converge en 2-3 vueltas.
    """
    def _pack(budget: int):
        kept, used, dropped = [], 0, 0
        for line in lines:
            if used + len(line) + 1 > budget:
                dropped += 1
            else:
                kept.append(line)
                used += len(line) + 1
        return kept, dropped

    # Punto fijo: la nota de recorte OCUPA espacio, así que hay que descontarla
    # del presupuesto ANTES de decidir qué cae. Converge en 2-3 vueltas (el
    # largo de la nota solo depende de los dígitos de `dropped`).
    notice = ""
    kept, dropped = [], 0
    for _ in range(4):
        kept, dropped = _pack(max_chars - len(header) - len(footer) - len(notice))
        nueva = f"\n(+{dropped} día(s) más antiguos omitidos por espacio.)" if dropped else ""
        if nueva == notice:
            break
        notice = nueva
    if not kept:
        logger.info(
            f"[P1-CHAT-PAST-DAYS] {label}: cap={max_chars} no da ni para una línea; bloque omitido"
        )
        return ""
    if dropped:
        logger.info(f"[P1-CHAT-PAST-DAYS] {label}: {dropped} día(s) recortados por cap={max_chars}")
    return header + "\n".join(kept) + notice + footer


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
    footer = ("\nLas fechas con '~' son estimadas, no exactas." + _plan_day_tool_footer_sentence())
    return _assemble(header, lines, footer, max_chars, "plan_days")


def build_past_diary_block(consumed_rows: Any, today: date,
                           days_back: Optional[int] = None,
                           max_chars: Optional[int] = None,
                           tz_offset_mins: int = 240) -> str:
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
        d = _to_local_date(row.get("consumed_at"), tz_offset_mins)
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
    # [P1-CHAT-DAY-DEFAULT-TODAY · 2026-07-30] La segunda frase del pie cierra
    # el hueco día→día. La primera solo prohibía la confusión PLAN→diario
    # ("no des lo prescrito por comido"), y el incidente del 2026-07-30 fue
    # OTRO eje: el coach citó TEXTUALMENTE la línea fechada "Miércoles 29 jul"
    # de este mismo bloque y la presentó como "el desayuno de hoy ya lo tienes
    # registrado" — con `DIARIO DE HOY` diciendo lo contrario tres bloques más
    # arriba. Entre una negación corta y lejana ("hoy no hay nada") y una lista
    # rica y cercana, gana la lista: por eso el cruce va DECLARADO aquí, pegado
    # a las líneas que se citan mal, y no solo como regla general del prompt.
    # tooltip-anchor: P1-CHAT-DAY-DEFAULT-TODAY-FOOTER
    footer = ("\n⚠️ 'SIN REGISTRO' significa que NO tienes ningún dato de lo que comió ese día. "
              "Si te pregunta qué comió un día SIN REGISTRO, dile con claridad que no lo tienes "
              "registrado y NUNCA respondas con lo que el plan mandaba como si se lo hubiera "
              "comido. El bloque del PLAN es lo prescrito; este bloque es lo real. "
              "Cada línea de aquí lleva SU fecha y esa fecha manda: NUNCA presentes una de "
              "estas comidas como si fuera de hoy, ni la cites para decirle que hoy ya tiene "
              "algo registrado. Lo registrado hoy es SOLO lo que diga el bloque DIARIO DE HOY, "
              "incluido cuando dice que hoy todavía no hay nada.")
    return _assemble(header, lines, footer, max_chars, "diary")
