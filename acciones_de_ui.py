# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1040 · A-27 · 2026-11-16] Las acciones de interfaz del turno, como dato del evento `done`.

Hasta hoy la pantalla se refrescaba (diario, agua, plan) solo si el MODELO escribía en su respuesta una marca
`[UI_ACTION: REFRESH_INVENTORY]`; si la olvidaba, el diario no se movía aunque la herramienta sí hubiera registrado (el
WaterTracker ya tenía un parche propio para eso). Aquí las acciones salen de lo que EJECUTÓ el turno —las herramientas
que el tracker de fases registró, sin las que bloqueó una guarda— más las marcas que el texto traiga. Viajan en
`done.ui_actions`; las marcas en el texto se quedan mientras haya paquetes viejos que solo las entiendan (A-33).
"""
from __future__ import annotations

import re
from typing import Iterable

ACCIONES = ("REFRESH_INVENTORY", "REFRESH_HYDRATION", "REFRESH_PLAN")

POR_HERRAMIENTA = {
    "log_consumed_meal": "REFRESH_INVENTORY",
    "correct_consumed_meal": "REFRESH_INVENTORY",
    "delete_consumed_meal": "REFRESH_INVENTORY",
    "modify_pantry_inventory": "REFRESH_INVENTORY",
    "mark_shopping_list_purchased": "REFRESH_INVENTORY",
    "log_water_glass": "REFRESH_HYDRATION",
    "generate_new_plan_from_chat": "REFRESH_PLAN",
    "modify_single_meal": "REFRESH_PLAN",
    "regenerate_full_day": "REFRESH_PLAN",
    "update_form_field": "REFRESH_PLAN",
}

_RE_MARCA = re.compile(r"\[UI_ACTION:\s*([A-Z_]+)\s*\]", re.IGNORECASE)


def acciones_del_turno(herramientas: Iterable, texto: str = "") -> list:
    """Acciones únicas, en el orden de `ACCIONES`. `herramientas`: entradas del tracker ({name, guard_blocked?}) o
    nombres sueltos."""
    vistas = set()
    for h in herramientas or ():
        if isinstance(h, dict):
            if h.get("guard_blocked"):
                continue
            nombre = h.get("name")
        else:
            nombre = h
        accion = POR_HERRAMIENTA.get(str(nombre or ""))
        if accion:
            vistas.add(accion)
    for m in _RE_MARCA.finditer(str(texto or "")):
        if m.group(1).upper() in ACCIONES:
            vistas.add(m.group(1).upper())
    return [a for a in ACCIONES if a in vistas]
