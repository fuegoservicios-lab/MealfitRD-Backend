"""[P1-HIST-COMPLETE-PROGRESS · 2026-05-31] El Historial debe mostrar el
progreso COMPLETO del plan (todos los días generados, incluidos los que ya
pasaron), NO la ventana rolling podada que ven el Dashboard "Tu Menú" y
Recetas.

Bug reportado por el usuario (con screenshot del modal del Historial):
    "los Dias tampoco deben eliminarse visualmente en el historial, eso solo
     debe pasar en el apartado de dashboard Tu menu y en recetas, el
     historial debe tener el progreso completo obvio."

Raíz (NO era un filtro visual — era pérdida de datos):
    El shift rolling del backend (`api_shift_plan` en routers/plans.py y
    `_background_shift_plan_for_user` en cron_tasks.py) hace
    `shifted_days = shifted_days[shift_amount:]` — PODA físicamente del array
    `plan_data.days` los días ya pasados y persiste el resultado. El chunk
    worker REQUIERE ese renumerado 1..N desde hoy, así que la poda es
    load-bearing del sistema rolling (no se puede simplemente dejar de podar).
    El modal del Historial lee `plan_data.days` → mostraba solo la ventana
    vigente, perdiendo el progreso pasado.

Fix (aditivo, no altera el rolling):
    Backend: ANTES de podar, los días removidos se preservan en
    `plan_data._archived_days` (con cap defensivo). El Dashboard/Recetas
    siguen leyendo `plan_data.days` (ventana rolling intacta); SOLO el
    Historial lee `_archived_days`.
    Frontend: el modal del Historial muestra `[..._archived_days, ...days]`
    (helper `_fullHistoryDays`) y etiqueta los tabs desde `cycle_start_date`
    (inicio original inmutable) cuando hay archive — porque el shift rebasa
    `grocery_start_date` a "hoy" (helper `_historyLabelStart`).

Tests parser-based: anclan que el archivo existe en AMBOS sitios de shift,
que es ADITIVO (la poda sigue ocurriendo después), y que el frontend usa el
timeline completo. Si un refactor reintroduce la poda sin archivar, falla.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_CRON_PY = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
_HISTORY_JSX = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.jsx"
).read_text(encoding="utf-8")

_MARKER = "P1-HIST-COMPLETE-PROGRESS"


def _shift_block(src: str) -> str:
    """Aísla el bloque del shift: desde `shift_amount = min(` hasta el slice
    `shifted_days = shifted_days[shift_amount:]` inclusive."""
    i = src.find("shift_amount = min(days_since_creation, len(shifted_days))")
    assert i > 0, "No se encontró el cómputo de `shift_amount` del shift."
    j = src.find("shifted_days = shifted_days[shift_amount:]", i)
    assert j > 0, "No se encontró el slice `shifted_days[shift_amount:]`."
    return src[i:j + len("shifted_days = shifted_days[shift_amount:]")]


# ---------------------------------------------------------------------------
# 1. Backend: ambos sitios de shift archivan ANTES de podar (aditivo)
# ---------------------------------------------------------------------------
def test_api_shift_plan_archives_before_pruning():
    block = _shift_block(_PLANS_PY)
    assert _MARKER in block, (
        f"Marker {_MARKER} ausente en el bloque de shift de routers/plans.py."
    )
    assert 'shifted_data["_archived_days"]' in block, (
        "api_shift_plan NO preserva `_archived_days` antes de podar — el "
        "Historial perdería el progreso pasado."
    )
    assert "copy.deepcopy(shifted_days[:shift_amount])" in block, (
        "El archivo debe ser un deepcopy de los días REMOVIDOS "
        "(`shifted_days[:shift_amount]`), no una referencia."
    )
    # Aditivo: la poda sigue ocurriendo (no se reemplazó por el archivo).
    assert block.rstrip().endswith("shifted_days = shifted_days[shift_amount:]"), (
        "El slice de la ventana rolling DEBE seguir ocurriendo tras archivar "
        "— el archivo es aditivo, el chunk worker requiere el renumerado 1..N."
    )


def test_background_shift_plan_archives_before_pruning():
    block = _shift_block(_CRON_PY)
    assert _MARKER in block, (
        f"Marker {_MARKER} ausente en el bloque de shift de cron_tasks.py "
        "(_background_shift_plan_for_user)."
    )
    assert 'shifted_data["_archived_days"]' in block, (
        "_background_shift_plan_for_user NO preserva `_archived_days`."
    )
    assert "copy.deepcopy(shifted_days[:shift_amount])" in block
    assert block.rstrip().endswith("shifted_days = shifted_days[shift_amount:]")


def test_archive_has_defensive_cap():
    """El archivo no debe crecer sin límite — cap por `_arch_cap`."""
    for name, src in (("plans.py", _PLANS_PY), ("cron_tasks.py", _CRON_PY)):
        block = _shift_block(src)
        assert "_arch_cap" in block and "_archived[-_arch_cap:]" in block, (
            f"{name}: falta el cap defensivo `_archived[-_arch_cap:]` — "
            "el array `_archived_days` podría crecer sin límite."
        )


# ---------------------------------------------------------------------------
# 2. Frontend: el modal del Historial usa el timeline COMPLETO
# ---------------------------------------------------------------------------
def test_frontend_exports_full_history_helpers():
    assert "export const _fullHistoryDays = (planData) =>" in _HISTORY_JSX, (
        "Helper `_fullHistoryDays` ausente — el modal no podría mergear "
        "`_archived_days` + `days`."
    )
    assert "export const _historyLabelStart = (planData, createdAt) =>" in _HISTORY_JSX, (
        "Helper `_historyLabelStart` ausente — los tabs no etiquetarían el "
        "timeline completo desde `cycle_start_date`."
    )
    # El merge concatena archivados + vigentes.
    assert "[...archived, ...live]" in _HISTORY_JSX, (
        "`_fullHistoryDays` no concatena `[...archived, ...live]`."
    )
    # La base de fecha prefiere cycle_start_date cuando hay archive.
    assert "planData?.cycle_start_date" in _HISTORY_JSX, (
        "`_historyLabelStart` no usa `cycle_start_date` (inicio original) "
        "cuando hay archive — los tabs quedarían mal etiquetados."
    )


def test_frontend_modal_uses_full_days_not_raw_days():
    """El render del modal (selector + meals + título) debe leer del
    timeline completo via `_fullHistoryDays(selectedPlan.plan_data)`, NO de
    `selectedPlan.plan_data.days` directo (que es la ventana podada)."""
    # Al menos 3 callsites de `_fullHistoryDays(selectedPlan.plan_data)`
    # (selector de días, título del menú, lista de meals).
    n = _HISTORY_JSX.count("_fullHistoryDays(selectedPlan.plan_data)")
    assert n >= 3, (
        f"Esperaba ≥3 usos de `_fullHistoryDays(selectedPlan.plan_data)` en "
        f"el modal (selector + título + meals). Encontrados: {n}."
    )
    # El meals lookup ya no usa `plan_data?.days?.[_safeIdx]?.meals` (podado).
    assert "selectedPlan.plan_data?.days?.[_safeIdx]?.meals" not in _HISTORY_JSX, (
        "El lookup de meals sigue leyendo `plan_data.days[_safeIdx]` (ventana "
        "podada) en vez del timeline completo (`_fullDays[_safeIdx]`)."
    )
