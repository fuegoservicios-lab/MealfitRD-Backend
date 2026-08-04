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


_CYCLE_ANCHOR_KEY = "_cycle_started_at"


def upcoming_days_signal_enabled() -> bool:
    """[Ronda 4 · B4] Knob ÚNICO de la señal de días futuros, para las TRES
    superficies (`/chunk-status`, cron horario, índice del coach).

    Antes solo gateaba el payload del endpoint: el cron horario y el `COUNT`
    por turno del coach seguían vivos sin palanca. El cron es justo la pieza
    que hubo que corregir de urgencia (19 de 23 planes alertando), o sea la
    que MÁS necesita rollback sin redeploy. Se lee por-llamada (no constante
    de import-time) para que flipear la env var no requiera reiniciar el
    worker."""
    from knobs import _env_bool
    return _env_bool("MEALFIT_UPCOMING_DAYS_UI", True)


def plan_cycle_window(plan_data):
    """[Ronda 4 · B1+B3] `(inicio, fin_EXCLUSIVO)` de la ventana de fechas que el
    ciclo VIGENTE del plan pretende cubrir. `(None, None)` si no se puede anclar.

    SSOT del término "¿el plan aún debe días?" para las tres superficies. Antes
    era un CONTEO (`len(_archived_days) + len(days) >= total_days_requested`) y
    ese conteo tiene dos agujeros que esta ventana cierra a la vez:

    **B1 — la RENOVACIÓN.** `_archived_days` NUNCA se vacía, ni cuando el plan
    renueva (`P0-1 RENEWAL`, `routers/plans.py`): mismo `plan_id`,
    `total_days_requested` intacto, `days=[]` y offsets desde 0. Dos ciclos
    comparten el array, así que el conteo alcanza `total` y apaga el predicado
    PARA SIEMPRE. Reproducido ejecutando la función: plan de 30 días renovado
    (30 archivados + 3 vivos, 27 sin generar, cola vacía) devolvía
    `(False, None)` — las tres superficies mudas justo en el bug que la feature
    existe para cerrar.

    **B3 — la caducidad.** El conteo no tiene término de vencimiento: un plan
    abandonado sigue "debiendo días" para siempre, así que declaraba ATRASADO
    también a +30 días, cuando `api_shift_plan` ya no puede encolar nada (no
    existe camino de resolución y la alerta del cron quedaba abierta sin
    remedio). Medido contra `76a6836d` en producción.

    **Por qué el ancla es un campo NUEVO y no uno de los que ya existen.**
    Medido contra los 24 planes de producción (2026-08-04,
    `scratchpad/forense_ciclo.py`), NO por el nombre del campo:

    - `grocery_start_date == days[0].date` en **23 de 23** planes con dato. Es
      la fecha de la VENTANA ROLLING (el shift la reescribe a HOY en cada
      rotación, `routers/plans.py`), no la del ciclo. Anclar ahí daría una
      ventana que se desplaza con el consumo y nunca vence.
    - `cycle_start_date == created_at` en **19 de 23** (los otros 4 difieren un
      día por TZ). Es el ancla INMUTABLE de creación y la renovación NO la
      toca, así que para un plan renovado apunta al ciclo ANTERIOR.

    Ninguno marca el inicio del ciclo vigente, y por fechas el ciclo tampoco es
    recuperable: la renovación arranca el día siguiente al último archivado, de
    modo que la línea temporal entregada es CONTIGUA y un ciclo nuevo es
    indistinguible de uno viejo completo. Por eso la renovación estampa
    `_cycle_started_at` (ver `test_b1_la_renovacion_estampa_el_ancla_del_ciclo`).

    Cascada del ancla:
      1. `_cycle_started_at` — escrito por la renovación. Autoritativo.
      2. La primera fecha ENTREGADA (archivados + vivos). Para un plan que
         nunca renovó ES el inicio del ciclo — los 24 planes de la flota hoy.
      3. `days[0].date - len(_archived_days)` cuando los archivados no llevan
         fecha (planes anteriores a P1-CHAT-PAST-DAYS). Se toma la estimación
         MÁS TEMPRANA de 2 y 3: adelantar el inicio adelanta el fin, y el modo
         de fallo que más caro sale es el falso ATRASADO.

    Nota honesta: sin el ancla (planes legacy ya renovados) el fallback NO
    recupera la renovación y el plan sigue mudo — el comportamiento de hoy, no
    una regresión. Anclado en
    `test_b1_sin_ancla_un_plan_renovado_legacy_queda_mudo_a_sabiendas`.
    tooltip-anchor: P2-CHUNK-OVERDUE-SIGNAL-CYCLE-WINDOW
    """
    try:
        if not isinstance(plan_data, dict):
            return (None, None)
        total = int(plan_data.get("total_days_requested") or 0)
        if total <= 0:
            return (None, None)
        live = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        archived = [d for d in (plan_data.get("_archived_days") or []) if isinstance(d, dict)]

        start = _to_local_date(plan_data.get(_CYCLE_ANCHOR_KEY))
        if start is None:
            dated = [p for p in (_parse_date(d.get("date")) for d in archived + live)
                     if p is not None]
            start = min(dated) if dated else None
            first_live = None
            for d in live:
                p = _parse_date(d.get("date"))
                if p is not None and (first_live is None or p < first_live):
                    first_live = p
            if first_live is not None and archived:
                by_count = first_live - timedelta(days=len(archived))
                start = by_count if start is None else min(start, by_count)
        if start is None:
            return (None, None)
        return (start, start + timedelta(days=total))
    except Exception as e:
        logger.warning(f"[P2-CHUNK-OVERDUE-SIGNAL] plan_cycle_window fail-open: {e}")
        return (None, None)


def plan_cycle_pending_days(plan_data) -> int:
    """[Ronda 4 · B1] Días que el ciclo VIGENTE todavía NO ha entregado. `0` si
    ya terminó, si no hay ancla o ante cualquier shape rara (fail-open).

    Segundo consumidor de `plan_cycle_window`, y la razón de que exista como
    función y no como tres copias: este cálculo vivía DUPLICADO en
    `build_pending_plan_days_lines` (numeración del índice del coach) y en
    `agent._build_pending_days_lines_block` (short-circuit para no pagar el
    COUNT), cada uno con su propio `len(archived) + len(days)`. Duplicar el
    criterio es exactamente cómo el defecto de ventana rolling llegó a tener
    cuatro instancias.

    Se mide por FECHAS (`último_día_vivo - inicio_ciclo + 1`), no contando
    arrays, para que los archivados del ciclo anterior no cuenten como
    entregados tras una renovación."""
    try:
        if not isinstance(plan_data, dict):
            return 0
        days = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        if not days:
            return 0
        total = int(plan_data.get("total_days_requested") or 0)
        if total <= 0:
            return 0
        last = None
        for d in days:
            pd = _parse_date(d.get("date"))
            if pd is not None and (last is None or pd > last):
                last = pd
        if last is None:
            return 0
        cycle_start, _cycle_end = plan_cycle_window(plan_data)
        if cycle_start is None:
            return 0
        delivered = max(0, min((last - cycle_start).days + 1, total))
        return max(0, total - delivered)
    except Exception as e:
        logger.warning(f"[P2-CHUNK-OVERDUE-SIGNAL] plan_cycle_pending_days fail-open: {e}")
        return 0


def compute_chunk_overdue(plan_data, in_flight_count, today=None):
    """[P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] ¿Hoy debería existir un día del plan que no
    existe, sin nada en la cola que lo vaya a resolver solo?

    SSOT del predicado — 3 consumidores (/chunk-status, cron horario, índice del coach).
    Fail-open doble: plan legacy sin `date` estampada ⇒ (False, None) (no alerta en falso);
    cualquier excepción ⇒ (False, None). `today` inyectable para tests deterministas; en
    producción usa rd_today() (date-only en TZ RD — nunca la hora UTC del servidor;
    escribir aquí el nombre de esa función deprecada dispara el escáner P3-DEPRECATED-UTCNOW,
    que no distingue docstring de código).

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
    modo de fallo invertido que la alerta existe para evitar).

    [Ronda 4 · B1+B3] Ese guard pasó de CONTAR (`len(archived) + len(days)`)
    a la VENTANA DE FECHAS de `plan_cycle_window` (arriba está el porqué con
    los números medidos contra producción). El conteo seguía roto en dos
    formas: se contamina entre ciclos cuando el plan RENUEVA
    (`_archived_days` nunca se vacía ⇒ predicado mudo PARA SIEMPRE, con 27
    días sin generar y la cola vacía) y no tenía término de caducidad (⇒
    alerta ATRASADO sin ningún camino de resolución en un plan cuyo ciclo ya
    venció). La ventana cubre ambas y además SUBSUME al conteo: si el ciclo
    entregó sus `total` días contiguos, el último día vivo es `fin - 1`, así
    que `hoy > último_vivo` implica `hoy >= fin` y el plan sale por la
    ventana — el caso de la Ronda 2 sigue dando `(False, None)`, ahora por la
    razón correcta (su ciclo terminó) y no por un conteo de arrays.

    La fecha ancla (`last`) sigue viniendo SOLO de `days` (los archivados son
    estrictamente anteriores, ver `_live_anchor`)."""
    try:
        if not isinstance(plan_data, dict):
            return (False, None)
        days = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
        if not days:
            return (False, None)
        if int(in_flight_count or 0) > 0:
            return (False, None)
        last = None
        for d in days:
            pd = _parse_date(d.get("date"))
            if pd is not None and (last is None or pd > last):
                last = pd
        if last is None:
            return (False, None)  # legacy sin dates: fail-open
        cycle_start, cycle_end = plan_cycle_window(plan_data)
        if cycle_start is None:
            return (False, None)  # sin ancla de ciclo: fail-open
        _today = today or rd_today()
        if _today <= last:
            return (False, None)  # ya existe un día para hoy
        if not (cycle_start <= _today < cycle_end):
            return (False, None)  # el ciclo venció: el plan ya no promete hoy
        return (True, (last + timedelta(days=1)).isoformat())
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
        total = int(plan_data.get("total_days_requested") or 0)

        last = None
        for d in days:
            pd = _parse_date(d.get("date"))
            if pd is not None and (last is None or pd > last):
                last = pd
        if last is None:
            return []  # legacy sin dates: fail-open, mismo criterio que compute_chunk_overdue

        # [Ronda 4 · B1] El conteo `len(archived) + len(days)` era la CUARTA
        # instancia del defecto de ventana rolling — y aquí vivía DUPLICADO
        # (esta función y `compute_chunk_overdue` llevaban cada una su copia,
        # así que arreglar solo el predicado habría dejado mudo al coach).
        # Ahora ambas pasan por `plan_cycle_pending_days`, que mide por fechas
        # y no se contamina con los archivados del ciclo anterior tras renovar.
        n_pending = plan_cycle_pending_days(plan_data)
        if n_pending <= 0:
            return []
        n_generated = total - n_pending

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
