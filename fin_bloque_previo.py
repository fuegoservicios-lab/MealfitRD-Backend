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
from datetime import date
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
    """La fecha más tardía entre los días vivos del plan (`days[*].date`, «YYYY-MM-DD…»), o None si no hay ninguna."""
    dias = plan_data.get("days") if isinstance(plan_data, dict) else None
    fechas = []
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
