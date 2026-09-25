# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-241 · 2026-09-25] El horario del formulario («Turno nocturno: duermo de día, trabajo de noche»;
«Rotativo / variable») llega al plan.

Auditoría del 25-sep: el generador de planes no leía `scheduleType` en ningún sitio (solo el coach, los recordatorios de
agua y el agente proactivo). Batería real, perfil nocturno: «Desayuno 06:30 · Almuerzo 13:00 · Merienda 16:30 · Cena
19:30» — un almuerzo a la hora en que la persona duerme — y el día 2 «Flexible».

Los nombres de las franjas no cambian (la app los usa como SSOT: «Desayuno» es la primera comida al despertar). Lo que
cambia es CUÁNDO y CÓMO: al despertar, la comida principal antes del turno, algo ligero y portátil durante el turno y
una comida ligera y sin cafeína al salir, antes de dormir. Las horas se fijan aquí, deterministas; la composición se la
dice el prompt (`horizon.explain_form_codes_for_prompt` / `horizon.schedule_rule`). Rotativo ⇒ «Flexible».
tooltip-anchor: P1-PLAN-LOTE-241-HORARIO
"""
from __future__ import annotations

import unicodedata

HORAS_NOCTURNO = {"desayuno": "15:30", "almuerzo": "19:30", "merienda": "01:00", "cena": "07:00",
                  "merienda am": "01:00", "merienda pm": "17:30", "merienda nocturna": "03:30"}


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower().strip()


def horario(form_data) -> str:
    return _sa((form_data or {}).get("scheduleType") if isinstance(form_data, dict) else "")


def asignar_horas(plan: dict, form_data) -> int:
    """Fija `time` por franja según el horario declarado. Estándar ⇒ no toca nada. Devuelve cuántas comidas cambió."""
    h = horario(form_data)
    if h not in ("night_shift", "variable") or not isinstance(plan, dict):
        return 0
    n = 0
    for day in plan.get("days") or []:
        for meal in (day.get("meals") or []) if isinstance(day, dict) else []:
            if not isinstance(meal, dict):
                continue
            nueva = HORAS_NOCTURNO.get(_sa(meal.get("meal")), "Flexible") if h == "night_shift" else "Flexible"
            if meal.get("time") != nueva:
                meal["time"] = nueva
                n += 1
    return n
