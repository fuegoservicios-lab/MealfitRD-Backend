# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1075 · 2026-11-17] Una comida de una franja YA PASADA, anotada tarde, se guarda en su franja.

El dueño anotó el día entero a las 22:23 («No desayuné, almorcé esto», «Merendé esto», «Cené esto»): el almuerzo y la
merienda quedaron con `consumed_at` 22:23 y 22:25. El diario los ordenaba como cenas tardías y el informe para el médico
habría dicho «primera comida a las 22:23» y «100 % de cenas después de las 22:00» — falso, y justo lo que un clínico
lee primero.

Regla (determinista, solo hoy —`days_ago == 0`—, nunca para cena/snack/suplemento): si la franja nombrada ya pasó con
margen, la hora guardada es la representativa de esa franja en el día biológico (corte 4:00):

    desayuno  pasada desde las 12:00  →  08:00
    almuerzo  pasada desde las 17:00  →  13:00
    merienda  pasada desde las 20:00  →  16:30

Esa hora es una ESTIMACIÓN: `consumed_at` ≠ `created_at`, y por eso la app no la enseña como hora (`horaFiable`) y el
informe no la usa para los horarios (`informe_medico`, `hora_fiable`). Lo que sí arregla: el día, el orden del diario y
la franja.

[18-nov] La MERIENDA no se mueve si la cena ya está anotada: «Merendé esto» a las 00:25 tras cenar es un picoteo de
noche, no la merienda de la tarde (el primer día lo movió a las 16:30: dos filas del dueño, corregidas a mano).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# franja → (hora local desde la que se considera pasada, hora representativa)
_FRANJAS = {
    "desayuno": (12.0, (8, 0)),
    "almuerzo": (17.0, (13, 0)),
    "merienda": (20.0, (16, 30)),
}
CORTE_BIOLOGICO_H = 4


def consumed_at_de_franja_pasada(meal_type: Optional[str], ahora_utc: datetime, tz_offset_min: Optional[int],
                                 ya_ceno: bool = False) -> Optional[str]:
    """ISO UTC de la hora representativa de la franja, o None si no aplica.

    `tz_offset_min` es «UTC − local» en minutos (RD = 240), como `db_facts.user_tz_offset_min`. `ya_ceno`: la cena de
    hoy ya está anotada (una merienda después de cenar es de noche y se queda con su hora real)."""
    tipo = str(meal_type or "").strip().lower()
    franja = _FRANJAS.get(tipo)
    if franja is None or tz_offset_min is None:
        return None
    if tipo == "merienda" and ya_ceno:
        return None
    if ahora_utc.tzinfo is None:
        ahora_utc = ahora_utc.replace(tzinfo=timezone.utc)
    local = ahora_utc.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(minutes=int(tz_offset_min))
    hora = local.hour + local.minute / 60.0
    dia = local.date()
    if hora < CORTE_BIOLOGICO_H:   # de madrugada sigue siendo el día de ayer: todas sus franjas ya pasaron
        dia = dia - timedelta(days=1)
        hora += 24.0
    pasada_desde, (h, m) = franja
    if hora < pasada_desde:
        return None
    local_franja = datetime(dia.year, dia.month, dia.day, h, m)
    return (local_franja + timedelta(minutes=int(tz_offset_min))).replace(tzinfo=timezone.utc).isoformat()
