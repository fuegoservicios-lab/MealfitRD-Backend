# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-816 · 2026-09-29] El día del CICLO de compra que ven los filtros de compra única.

Qué estaba roto (medido el 28-sep, sólo SELECT):
  `constants.rebase_pending_chunk_offsets` (P1-CHUNK-OFFSET-REBASE) reescribe el `days_offset` de la cola contra el ancla
  MÓVIL del plan (el shift la lleva a hoy): el primer bloque pendiente queda en `len(días vivos)`. Esa columna es la que
  el worker pone en `form_data["_days_offset"]`, y los consumidores de durabilidad la leían como el día del ciclo de la
  compra — `ai_helpers._age_pantry_for_block` (Nevera envejecida, P1-STEP14-SHOPPING-COOKING),
  `ai_helpers._single_trip_durable_filter` (sembrador, P1-SINGLE-TRIP-ROTATION), `compra_unica.nevera_virtual` (lo que
  LLEGA al bloque, P1-PLAN-LOTE-221), `compra_unica.candidatos_del_dia` (cerrador de proteína, P1-PLAN-LOTE-521, que
  además usaba el `day` de la ventana renumerada) y la sustitución de frescos del merge del bloque
  (`graph_orchestrator._single_trip_fresh_substitute` vía el `_days_offset` del view del chain).
  Plan vivo 6594aae1 (30 días, sin congelador): el bloque 4 pendiente lleva columna 5 y rebanada 11; los bloques 2 y 3
  corrieron con columna 1 (rebanadas 3 y 7 INFERIDAS: su snapshot está nulo en DB). Con columna 1
  `single_trip_requirements(…, 1)` no exige nada (los 3 días libres sin congelador): el plan lleva pescado fresco los
  días 7 y 8 del ciclo (09-29 y 09-30, contando desde la compra del 09-23).

Qué NO cambia: el significado de `days_offset` para la ventana y las fechas (P1-CHUNK-OFFSET-REBASE,
P1-CHUNK-EXECUTE-CEILING, la numeración `day = days_offset + i + 1`, `_plan_start_date`). Sólo el ÍNDICE que reciben los
filtros de durabilidad.

El día del ciclo del PRIMER día del bloque (0-based) es el más exigente de lo que se sabe, nunca menos que la columna:
  1. `_blueprint_slice.days_offset` — la rebanada inmutable del blueprint (índice del ciclo del run, H2).
  2. El calendario: (fecha del primer día del bloque = `_plan_start_date` local + columna) − inicio del ciclo
     (`chat_history_context.plan_cycle_window`, SSOT: `_cycle_started_at` → primera fecha entregada → días[0] −
     archivados). Sólo lo calcula el worker, que tiene el plan (`sellar`); se descarta si cae fuera del ciclo (un plan
     renovado sin `_cycle_started_at` mide desde el ciclo anterior).
  3. La columna (conducta previa): sin rebanada ni plan, o con el knob apagado.
Por qué el máximo y no «la rebanada si existe»: las dos pueden desviarse del reloj en sentidos opuestos. En 6594aae1 el
reloj dice 10 (09-23 → 10-03) y la rebanada 11: la línea de tiempo perdió un día (09-26 archivado dos veces). Un bloque
pausado días y reanclado cubre fechas POSTERIORES a su rebanada, y ahí la rebanada se queda corta. Para «¿aguanta hasta
ese día lo comprado el día 1?» el error caro es quedarse corto.

La Nevera virtual (`compra_unica.nevera_virtual`) conserva su puerta `_days_offset > 0` y se evalúa DOS veces en la rama
LLM del worker: en el 1.er refresco de la Nevera (cron_tasks `_refresh_chunk_pantry` antes del desvío degradado/LLM)
`_days_offset` aún no está y la puerta queda cerrada; en el 2.º (tras `form_data["_days_offset"] = days_offset` y
`sellar`, ya con el perfil vivo fusionado — `_merge_chunk_live_profile` conserva las claves `_`) la puerta está ABIERTA
y filtra con el día sellado. Así que este lote cambia en vivo lo que recibe cualquier usuario de compra única con la
Nevera vacía o apagada. Hoy no aparece en prod porque el único plan vivo de compra única (6594aae1, usuario 4da5c079)
tiene Nevera real (36 alimentos, SELECT del 29-sep) y `real and not _nevera_apagada` devuelve antes del filtro. (Corregido en la ronda del revisor: la
versión anterior de este párrafo la daba por dormida.)

Knob `MEALFIT_SINGLE_TRIP_CYCLE_DAY_TRUE` (True). False ⇒ la columna, byte a byte como antes.
Test: tests/test_p1_plan_lote_816.py. tooltip-anchor: P1-PLAN-LOTE-816
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

CLAVE = "_single_trip_cycle_day"     # el sello del worker en `form_data` (whitelist `_TRUSTED_INTERNAL_FORM_KEYS`)


def activo() -> bool:
    """tooltip-anchor: MEALFIT_SINGLE_TRIP_CYCLE_DAY_TRUE"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SINGLE_TRIP_CYCLE_DAY_TRUE", True)
    except Exception:
        return True


def _entero(v) -> Optional[int]:
    try:
        if v is None or v == "" or isinstance(v, bool):
            return None
        n = int(v)
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def columna(form_data) -> int:
    """El `_days_offset` del bloque: índice en la ventana viva (lo que el worker copia de `plan_chunk_queue`)."""
    fd = form_data if isinstance(form_data, dict) else {}
    return _entero(fd.get("_days_offset")) or 0


def _de_la_rebanada(fd: dict) -> Optional[int]:
    sl = fd.get("_blueprint_slice")
    return _entero(sl.get("days_offset")) if isinstance(sl, dict) else None


def _dias_del_ciclo(fd: dict, plan_data) -> int:
    eff = fd.get("_plan_policy_effective")
    if not isinstance(eff, dict):
        pp = plan_data.get("_plan_policy") if isinstance(plan_data, dict) else None
        eff = pp.get("effective") if isinstance(pp, dict) else None
    try:
        return int(((eff or {}).get("shopping") or {}).get("main_cycle_days") or 0)
    except (TypeError, ValueError):
        return 0


def por_calendario(form_data, plan_data, col: Optional[int] = None) -> Optional[int]:
    """(fecha del primer día del bloque − inicio del ciclo) en días, o None si no se puede anclar o sale del ciclo."""
    try:
        fd = form_data if isinstance(form_data, dict) else {}
        if not isinstance(plan_data, dict):
            return None
        from chat_history_context import plan_cycle_window, _parse_date, _to_local_date
        inicio = plan_cycle_window(plan_data)[0]
        if inicio is None:
            return None
        from constants import tz_offset_min_for_form_data
        tz = tz_offset_min_for_form_data(fd)
        ancla = _to_local_date(fd.get("_plan_start_date"), tz)
        if ancla is None:
            vivos = [d for d in (plan_data.get("days") or []) if isinstance(d, dict)]
            ancla = _parse_date(vivos[0].get("date")) if vivos else None
        if ancla is None:
            ancla = _to_local_date(plan_data.get("grocery_start_date"), tz)
        if ancla is None:
            return None
        dia = (ancla - inicio).days + (columna(fd) if col is None else int(col))
        ciclo = _dias_del_ciclo(fd, plan_data)
        if dia < 0 or (ciclo and dia >= ciclo):
            return None
        return dia
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-816] día por calendario no disponible: {type(e).__name__}: {e}")
        return None


def calcular(form_data, plan_data=None, col: Optional[int] = None) -> int:
    """El día del ciclo del primer día del bloque: el más exigente de rebanada, calendario y columna."""
    fd = form_data if isinstance(form_data, dict) else {}
    c = columna(fd) if col is None else max(0, int(col))
    if not activo():
        return c
    fuentes = [c, _de_la_rebanada(fd), por_calendario(fd, plan_data, c)]
    return max(x for x in fuentes if x is not None)


def sellar(form_data, plan_data, col) -> None:
    """El worker, con el plan en la mano: deja el día del ciclo en `form_data[CLAVE]`. Nunca lanza."""
    try:
        if not isinstance(form_data, dict) or not activo():
            return
        dia = calcular(form_data, plan_data, col)
        form_data[CLAVE] = dia
        if dia != int(col or 0):
            logger.info(f"🧳 [P1-PLAN-LOTE-816] día del ciclo {dia + 1} (columna {int(col or 0) + 1}, rebanada "
                        f"{_de_la_rebanada(form_data)}): los filtros de compra única miden desde la compra.")
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-816] sello no-op: {type(e).__name__}: {e}")


def dia_real(form_data) -> Optional[int]:
    """El día del ciclo si hay una fuente de verdad (sello del worker o rebanada); None si sólo queda la columna."""
    try:
        fd = form_data if isinstance(form_data, dict) else {}
        if not activo():
            return None
        sellado = _entero(fd.get(CLAVE))
        if sellado is not None:
            return max(sellado, columna(fd))
        sl = _de_la_rebanada(fd)
        return max(sl, columna(fd)) if sl is not None else None
    except Exception:
        return None


def dia(form_data) -> int:
    """Lo que leen los filtros: el día del ciclo (0-based) del primer día del bloque. Sin fuente, la columna."""
    d = dia_real(form_data)
    return columna(form_data) if d is None else d


def desplazamiento(form_data) -> int:
    """Cuánto hay que sumar a un índice de la ventana del bloque (`day − 1`) para llevarlo al ciclo. 0 sin fuente."""
    d = dia_real(form_data)
    return 0 if d is None else max(0, d - columna(form_data))


def indice_del_chain(form_data, por_defecto: int) -> int:
    """El `_days_offset` del view del chain en el merge del bloque: el día del ciclo de su primer día nuevo."""
    d = dia_real(form_data)
    return int(por_defecto or 0) if d is None else d


__all__ = ["CLAVE", "activo", "columna", "por_calendario", "calcular", "sellar", "dia_real", "dia", "desplazamiento",
           "indice_del_chain"]
