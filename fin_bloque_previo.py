# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-200 · 2026-09-24] El gate temporal mide el fin del bloque anterior con las FECHAS del plan.

`_check_chunk_learning_ready` (cron_tasks) difiere el bloque N hasta que haya pasado el último día del bloque N-1, y lo
calculaba con aritmética: `ancla + (offset_previo + días_previos − 1) + _shift_days_accumulated`. Dos piezas de esa suma
ya no significan lo que la fórmula supone:
  · el ancla (`_plan_start_date`) la re-escribe CADA shift a hoy (P1-CHUNK-OFFSET-REBASE, `[P0-5 FIX]` del shift en
    segundo plano), así que sumarle además el acumulado de shifts cuenta el desplazamiento DOS veces;
  · el offset del bloque previo, ya completado, NO se re-ancla: es relativo al ancla del día en que se generó.
Producción (el único usuario con relleno semanal, plan 3957a669): el 23-sep el gate dijo que el bloque 7 terminaba el
27 cuando terminó el 22 (acumulado = 3). Resultado en cada relleno desde el 16-sep: aplazamiento cada ~1,4 min (2.707
filas en `chunk_deferrals` en 4 días), pausa `prev_chunk_not_concluded`, push de «revisa tu zona horaria», 24 h de TTL,
escalada a modo flexible con 12-14 «🚨 Compra Urgente»… y un día SIN PLAN (el 23-sep; el bloque llegó el 24 con su
primer día ya pasado, y el siguiente shift lo archivó).

El plan ya sabe cuándo termina lo que tiene: cada día lleva su `date` (la ponen el shift y el merge). Cuando se evalúa
el bloque N, el último día planificado ES el fin del bloque N-1. Aquí sólo se ACOTA la fórmula por ese dato: si la
fórmula da una fecha posterior al último día planificado, manda el día planificado; nunca se retrasa nada. Sin fechas,
la fórmula de siempre. Knob `MEALFIT_GATE_PREV_END_FROM_DAYS` (True).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def activo() -> bool:
    """tooltip-anchor: MEALFIT_GATE_PREV_END_FROM_DAYS"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_GATE_PREV_END_FROM_DAYS", True)
    except Exception:
        return True


def ultimo_dia_planificado(plan_data) -> Optional[date]:
    """La fecha más tardía entre los días del plan —vivos (`days`) y ARCHIVADOS (`_archived_days`, que los dos shifts
    fechan antes de archivar)—, «YYYY-MM-DD…», o None si no hay ninguna.

    [P1-PLAN-LOTE-204 · 2026-09-24] Con sólo los vivos, un plan cuyo último bloque ya pasó entero (el usuario abre la app
    tras la medianoche, el shift archiva sus días) llegaba al gate SIN fechas y volvía la fórmula inflada: justo en el
    usuario que se quedó sin días. Pasa, por ejemplo, con el bloque 2 de un plan de 30 días."""
    if not isinstance(plan_data, dict):
        return None
    fechas = []
    for clave in ("days", "_archived_days"):
        dias = plan_data.get(clave)
        for d in dias if isinstance(dias, list) else []:
            texto = str((d or {}).get("date") or "")[:10] if isinstance(d, dict) else ""
            try:
                fechas.append(date.fromisoformat(texto))
            except ValueError:
                continue
    return max(fechas) if fechas else None


def acotar(prev_end: Optional[date], plan_data, meal_plan_id=None, week_number=None) -> Optional[date]:
    """`prev_end` de la fórmula, acotado por el último día planificado (nunca lo mueve hacia adelante)."""
    if prev_end is None or not activo():
        return prev_end
    try:
        real = ultimo_dia_planificado(plan_data)
    except Exception:
        return prev_end
    if real is None or real >= prev_end:
        return prev_end
    logger.info(f"📅 [P1-PLAN-LOTE-200] plan {str(meal_plan_id)[:8]} bloque {week_number}: fin del bloque previo "
                f"{prev_end.isoformat()} (fórmula) → {real.isoformat()} (último día planificado)")
    return real


# ─────────────── [P1-PLAN-LOTE-207 · 2026-09-24] esperar al bloque previo sin quemar intentos ───────────────
# Cuando el worker recoge un bloque ANTES de que termine el anterior (un `execute_after` viejo que el re-anclaje dejó en
# el pasado y el suelo de NOW() soltó de inmediato — dea00a2f el 23-sep), el gate lo difería con un backoff de minutos
# pensado para desfases de huso… y el worker salía dejando el bloque en `processing`. Nadie lo devolvía a `pending`: lo
# hacía el rescate de zombies a los 10 min, SUMANDO un intento cada vez (tope 5 ⇒ `failed`). En producción los bloques
# 5, 6 y 8 de 3957a669 terminaron con attempts = 5 — a un rescate de morir —, re-evaluados cada ~16 min. Y un bloque sin
# la marca proactiva de zero-log caía antes en el aplazamiento GENÉRICO de aprendizaje: +12 h y el push «Tu próximo
# bloque espera más feedback… loguea tus comidas» (4 de 4 medidos eran `temporal_gate`): nada que el usuario pudiera
# hacer, y 12 h que pueden pasarse de la frontera y dejarle un día sin menú.
# Aquí: el gate programa el bloque para el primer instante en que va a pasar (la frontera), el worker lo devuelve a
# `pending` en vez de dejarlo huérfano, y la espera de calendario no pasa por el aplazamiento de aprendizaje.
# Knob `MEALFIT_TEMPORAL_GATE_WAIT_BOUNDARY` (True).
RAZON_CALENDARIO = "prev_chunk_day_not_yet_elapsed"


def espera_frontera() -> bool:
    """tooltip-anchor: MEALFIT_TEMPORAL_GATE_WAIT_BOUNDARY"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_TEMPORAL_GATE_WAIT_BOUNDARY", True)
    except Exception:
        return True


def frontera_utc(prev_end: Optional[date], tz_min, margen_dias=0) -> Optional[datetime]:
    """Primer instante en que el gate deja pasar al bloque siguiente: medianoche LOCAL del día `prev_end + 1 − margen`
    (el gate difiere mientras `prev_end − hoy ≥ margen`), +30 min como el encolado, en UTC. None sin fecha o knob off."""
    if prev_end is None or not espera_frontera():
        return None
    try:
        dia = prev_end + timedelta(days=1 - max(0, int(margen_dias or 0)))
        return datetime.combine(dia, time(0), tzinfo=timezone.utc) + timedelta(minutes=int(tz_min or 0) + 30)
    except Exception:
        return None


def es_espera_de_calendario(learning_ready) -> bool:
    """El gate difirió porque el bloque previo aún no termina: no es falta de registros y no se aplaza 12 h."""
    return espera_frontera() and isinstance(learning_ready, dict) and learning_ready.get("reason") == RAZON_CALENDARIO


def soltar_a_pendiente(task_id, escribir) -> None:
    """El worker devuelve el bloque que difirió a `pending` (con el `execute_after` que eligió el gate) en vez de dejarlo
    en `processing` para que el rescate de zombies lo recoja sumando un intento."""
    if not espera_frontera():
        return
    try:
        escribir("UPDATE plan_chunk_queue SET status = 'pending', updated_at = NOW() "
                 "WHERE id = %s AND status = 'processing'", (task_id,))
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-207] no se pudo devolver el bloque {task_id} a pending: {e}")
