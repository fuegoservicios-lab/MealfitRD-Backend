"""[P3-DAY-LABEL-FROM-PLAN · 2026-05-17] Fix del mismatch entre tabs de día
en Dashboard y el indicador "Hoy" (dot azul).

Síntoma reportado por usuario:
> "hoy es domingo, por que esta marcando el lunes?"

Screenshot mostraba tabs `[Domingo, Lunes, Martes]` con dot "Hoy" en Lunes,
pese a que el día actual era Domingo.

Causa raíz: dos cálculos divergentes en `Dashboard.jsx`:
  - Tab labels: calculadas desde calendario (`new Date() + visibleIdx`)
  - Dot "Hoy": calculada desde índice del plan (`todayPlanDayIndex`)

Cuando `grocery_start_date != hoy` (e.g., localStorage con plan de ayer
y user vuelve hoy), los 2 cálculos divergen:
  - Labels muestran "Domingo/Lunes/Martes" (días desde hoy)
  - Pero los MEALS de cada tab corresponden a plan_days desde grocery_start_date
  - El dot cae en el slot donde está el contenido de hoy en el plan (plan_day 1
    si el plan empezó ayer) → "Lunes" según el label, pero realmente es la
    posición de domingo en el plan.

Fix: usar el `day.day_name` que el backend ya inyecta en
`graph_orchestrator.py:7278` (computed desde grocery_start_date + day_index,
TZ-aware). Ahora label = day.day_name → tabs siempre alineados con los meals
del plan. Fallback al cálculo viejo solo para planes legacy sin day_name.

Backend evidence del inject (log 2026-05-17 00:08:20):
> 📅 [DAY NAMES] Inyectados: ['Domingo', 'Lunes', 'Martes']
> (start=2026-05-17T00:00:00+00:00, tzOffset=240)

Cada day en `plan_data.days` ya tiene `day_name` poblado.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-DAY-LABEL-FROM-PLAN" in _DASHBOARD, (
        "Marker P3-DAY-LABEL-FROM-PLAN ausente — un refactor podría borrar "
        "el fix y reintroducir el mismatch label↔dot."
    )


def test_day_name_read_from_plan_day():
    """El render del tab DEBE leer `day?.day_name` antes de fallback al
    cálculo desde `new Date()`."""
    # El render relevante es el INTERIOR del button de los tabs (NO el
    # skeleton placeholder de días no generados que está más abajo).
    # Anchor: el bloque del primer map con weekDays.
    idx = _DASHBOARD.find("if (day?.day_name) return day.day_name")
    assert idx > 0, (
        "Frontend NO está leyendo `day.day_name` para el label del tab. "
        "Esto revierte P3-DAY-LABEL-FROM-PLAN — labels divergerán del dot "
        "'Hoy' cuando plan empieza en día distinto a hoy."
    )


def test_fallback_to_old_compute_preserved():
    """Para planes legacy en localStorage sin `day_name` (pre-P3-DAY-LABEL-FROM-PLAN
    deployment), el fallback al cálculo viejo debe preservarse."""
    idx = _DASHBOARD.find("if (day?.day_name) return day.day_name")
    assert idx > 0
    # Próximo bloque debe ser el fallback con diasSemana[d.getDay()]
    block = _DASHBOARD[idx:idx + 1500]
    assert "diasSemana[d.getDay()]" in block, (
        "Fallback al cálculo viejo desde new Date() removido — planes legacy "
        "en localStorage sin day_name renderizarían `undefined` en el tab."
    )


def test_no_silent_break_of_existing_pattern():
    """Defensa: el cálculo viejo `new Date() + visibleIdx` sigue presente
    para el path de skeleton placeholders (días no generados aún). Esos
    no tienen day_name del backend porque el chunk no se generó."""
    # Anchor: el skeleton placeholder más abajo
    skel_idx = _DASHBOARD.find("_d.setDate(_d.getDate() + _slotVisibleIdx)")
    assert skel_idx > 0, (
        "Skeleton placeholder path fue removido — los tabs de días futuros "
        "no generados (chunk-2 in flight) perderían su label estimado."
    )
