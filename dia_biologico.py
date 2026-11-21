# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] De madrugada (antes de las 4:00) todavía es el día de AYER.

Decisión del dueño (21-nov). Él anotó a las 00:40-00:44 el almuerzo, la cena, la merienda y los suplementos del 20: el
almuerzo se fue al 20 (lote 1075) pero la cena, la merienda y las tomas quedaron en el 21 — el 20 sin cena en la racha y
el mapa, el 21 con 1.500 kcal antes de amanecer, y el coach preguntando a las 00:40 «¿qué comiste en la cena de hoy?».

Regla: lo que se anota entre las 00:00 y las 4:00 hora local, sin `days_ago` explícito, va al día anterior, con una
hora representativa al final de ese día (es una ESTIMACIÓN: `consumed_at` ≠ `created_at`, la app no la enseña como hora).
El desayuno NO se mueve (de madrugada, «desayuné» es el de hoy que empieza). El corte es el mismo de `hora_de_la_franja`.
Knob `MEALFIT_MADRUGADA_ES_AYER` (True).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from hora_de_la_franja import CORTE_BIOLOGICO_H

# franja → hora local representativa en el día de ayer (en orden: cena, picoteo, tomas)
_HORA_DE_AYER = {
    "almuerzo": (13, 0),
    "cena": (23, 0),
    "merienda": (23, 30),
    "snack": (23, 30),
    "complemento": (23, 30),
    "suplemento": (23, 45),
}


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_MADRUGADA_ES_AYER", True)
    except Exception:  # noqa: BLE001
        return True


def hora_en_el_dia_de_ayer(meal_type: Optional[str], ahora_utc: datetime,
                           tz_offset_min: Optional[int]) -> Optional[str]:
    """ISO UTC de la hora representativa en el día de AYER si ahora es de madrugada; si no, None.

    `tz_offset_min` = «UTC − local» en minutos (RD = 240), como `db_facts.user_tz_offset_min`."""
    tipo = str(meal_type or "").strip().lower()
    hm = _HORA_DE_AYER.get(tipo)
    if hm is None or tz_offset_min is None or not activo():
        return None
    if ahora_utc.tzinfo is None:
        ahora_utc = ahora_utc.replace(tzinfo=timezone.utc)
    local = ahora_utc.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(minutes=int(tz_offset_min))
    if local.hour >= CORTE_BIOLOGICO_H:
        return None
    ayer = local.date() - timedelta(days=1)
    local_ayer = datetime(ayer.year, ayer.month, ayer.day, hm[0], hm[1])
    return (local_ayer + timedelta(minutes=int(tz_offset_min))).replace(tzinfo=timezone.utc).isoformat()
