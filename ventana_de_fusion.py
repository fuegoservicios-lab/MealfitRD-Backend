# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] La fusión por franja (lote 982) cubre también la merienda y el picoteo.

El dueño (21-nov, 00:42-01:02) anotó papitas Lay's, galletas Koko y 3 papitas con queso en 20 minutos y quedaron TRES
«Merienda»: la fusión «misma franja dentro del TTL» solo existía para desayuno/almuerzo/cena. Ahora una merienda (o
snack) que llega dentro de 60 min de la anterior de la MISMA franja es la misma merienda — la propuesta del dueño era
4 h; 60 min es el punto medio que no junta la merienda de la tarde con el picoteo de la noche (knob
`MEALFIT_PICOTEO_TTL_FUSION_MIN`). Las principales siguen en 30 min.

De madrugada (antes de las 4:00) lo anterior vive en el día de AYER (`dia_biologico`): la búsqueda mira también ayer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

PRINCIPALES = ("desayuno", "almuerzo", "cena")
PICOTEO = ("merienda", "snack")
CON_TTL = PRINCIPALES + PICOTEO
TTL_PRINCIPAL_MIN = 30.0


def ttl_de(slot: str) -> float:
    if str(slot or "").lower() in PICOTEO:
        try:
            from knobs import _env_float
            return max(0.0, float(_env_float("MEALFIT_PICOTEO_TTL_FUSION_MIN", 60.0)))
        except Exception:  # noqa: BLE001
            return 60.0
    return TTL_PRINCIPAL_MIN


def es_madrugada(tz_offset_min, ahora_utc=None) -> bool:
    if tz_offset_min is None:
        return False
    ahora = ahora_utc or datetime.now(timezone.utc)
    local = ahora.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(minutes=int(tz_offset_min))
    from hora_de_la_franja import CORTE_BIOLOGICO_H
    return local.hour < CORTE_BIOLOGICO_H


def comidas_recientes(user_id: str) -> list:
    """Las comidas de hoy y, de madrugada, también las de ayer (donde `dia_biologico` las guardó)."""
    from db_facts import get_consumed_meals_today, user_tz_offset_min
    filas = list(get_consumed_meals_today(user_id) or [])
    try:
        from dia_biologico import activo as _madrugada_activa
        off = user_tz_offset_min(user_id) if _madrugada_activa() else None
        if es_madrugada(off):
            local = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=int(off))
            ayer = (local.date() - timedelta(days=1)).isoformat()
            filas += list(get_consumed_meals_today(user_id, date_str=ayer, tz_offset_mins=off) or [])
    except Exception:  # noqa: BLE001
        pass
    return filas
