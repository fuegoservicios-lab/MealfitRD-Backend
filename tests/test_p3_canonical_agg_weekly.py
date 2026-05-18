"""[P3-CANONICAL-AGG-WEEKLY · 2026-05-18] Invariante semántico de
`aggregated_shopping_list_weekly` (+ biweekly/monthly).

Bug raíz que cierra:
    El usuario hacía Compré-todo → agotar/reponer en Pantry → Borrar Todos
    → descargaba el PDF de la lista de compras → veía SOLO ~10 items en vez
    de los 35 esperados.

    Cadena de eventos pre-fix:
      1. Compré-todo: pantry tiene 35 items. plan_data.is_restocked=true,
         restocked_items={35}. agg_weekly sin tocar (35 items, canonical).
      2. Agotar item X → DELETE row. _scheduleRecalcShoppingList →
         /recalculate-shopping-list con `is_new_plan_flag=False` (default).
         `get_shopping_list_delta(is_new_plan=False)` deduce inventario →
         delta = 1 item (X). agg_weekly = [X].  ← mutación stale en DB.
      3. Reponer item X → INSERT row + recalc. inventory=35, delta=0.
         agg_weekly = []. ← peor.
      4. Borrar Todos → recalc con inventory=0. delta=35. agg_weekly=[35].
      5. PERO el frontend tiene localStorage con `_plan_modified_at` que
         matchea el del intermedio (paso 2 o 3) → drift detection no fira →
         PDF se genera con la lista corta del intermedio.

Fix aplicado (5 callsites + 1 helper):
    a) `routers/plans.py::_apply_recalc`: scaled_7/15/30 con is_new_plan=True.
    b) `tools.py::execute_modify_single_meal`: aggr_7/15/30 con is_new_plan=True.
    c) `cron_tasks.py::_persist_pantry_supplement_to_plan_data`: idem.
    d) `cron_tasks.py::_clear_pantry_supplement_from_plan_data`: idem.
    e) `cron_tasks.py::_process_pending_shopping_lists` (chunk worker T1): idem.
    f) `cron_tasks.py::process_plan_chunk_queue` (chunk worker T2): idem.

Plus el FIX semántico crítico:
    `shopping_calculator.py::get_shopping_list_delta` ahora hace `is_new_plan`
    PRECEDER sobre el override de inventario. Pre-fix, pasar
    `is_new_plan=True, inventory_override=[35 items]` deducía contra el
    override igualmente — el flag era cosmético. Post-fix, is_new_plan=True
    fuerza listas vacías ANTES del check del override.

Invariante:
    `aggregated_shopping_list_weekly` SIEMPRE representa la lista CANÓNICA
    (full needs del plan a household_size + duration), sin deducir
    inventario. La deducción se hace at-render-time en el frontend vía
    Dashboard.buildDeltaShoppingList(canonical, freshInventory).

Tests (4 asserts parser-based):
    1. Función `get_shopping_list_delta` tiene la precedencia is_new_plan.
    2. routers/plans.py::_apply_recalc usa is_new_plan=True para scaled_*.
    3. tools.py callsite usa is_new_plan=True.
    4. cron_tasks.py callsites usan is_new_plan=True (al menos en los
       4 sitios identificados — drift detection prosa).

Tooltip-anchor: P3-CANONICAL-AGG-WEEKLY-START | test_p3_canonical_agg_weekly
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_CALC = _BACKEND_ROOT / "shopping_calculator.py"
_ROUTERS_PLANS = _BACKEND_ROOT / "routers" / "plans.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"


def _read(p: Path) -> str:
    assert p.exists(), f"Archivo requerido no encontrado: {p}"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Precedencia is_new_plan en shopping_calculator.get_shopping_list_delta
# ---------------------------------------------------------------------------
def test_is_new_plan_precedes_inventory_override():
    """`get_shopping_list_delta(is_new_plan=True, inventory_override=...)`
    DEBE forzar listas vacías ANTES del check del override. Sin esto, el
    refactor canónico era cosmético: callers que pasaban ambos params
    seguían produciendo delta (porque el override ganaba al flag).

    Patrón esperado:
        if is_new_plan:
            physical_inventory = []
            consumed_ingredients = []
        elif inventory_override is not None or consumed_override is not None:
            ...
    """
    src = _read(_SHOPPING_CALC)
    # Match: `if is_new_plan:` seguido de las dos listas vacías dentro de
    # los próximos 200 chars (cuerpo del if).
    pattern = re.compile(
        r"if\s+is_new_plan\s*:\s*\n"
        r"\s+physical_inventory\s*=\s*\[\s*\]\s*\n"
        r"\s+consumed_ingredients\s*=\s*\[\s*\]",
    )
    assert pattern.search(src) is not None, (
        "`get_shopping_list_delta` debe tener `if is_new_plan:` con asignación "
        "explícita de `physical_inventory = []` y `consumed_ingredients = []` "
        "ANTES de evaluar el override. Sin esta precedencia, callers que pasan "
        "`is_new_plan=True, inventory_override=_inv_snap` seguirían produciendo "
        "delta — el flag is_new_plan sería cosmético."
    )


# ---------------------------------------------------------------------------
# 2. /recalculate-shopping-list usa is_new_plan=True para storage
# ---------------------------------------------------------------------------
def test_recalc_endpoint_stores_canonical():
    """`_apply_recalc` en routers/plans.py debe invocar
    `get_shopping_list_delta` con `is_new_plan=True` para los 3 scaled_*
    (weekly/biweekly/monthly).
    """
    src = _read(_ROUTERS_PLANS)
    # Buscar las 3 invocaciones a get_shopping_list_delta dentro de _apply_recalc
    # con is_new_plan=True.
    matches = re.findall(
        r"scaled_(?:7|15|30)\s*=\s*get_shopping_list_delta\s*\([^)]*?is_new_plan=True",
        src,
        re.DOTALL,
    )
    assert len(matches) >= 3, (
        f"`_apply_recalc` (routers/plans.py) debe tener 3 invocaciones a "
        f"`get_shopping_list_delta(... is_new_plan=True ...)` para scaled_7, "
        f"scaled_15, scaled_30. Encontradas: {len(matches)}. "
        f"Si las firmas cambiaron, actualizar este test + bump marker."
    )


# ---------------------------------------------------------------------------
# 3. tools.py callsite (modify_single_meal) usa is_new_plan=True
# ---------------------------------------------------------------------------
def test_tools_modify_single_meal_stores_canonical():
    """`execute_modify_single_meal` debe usar is_new_plan=True para aggr_*."""
    src = _read(_TOOLS_PY)
    matches = re.findall(
        r"aggr_(?:7|15|30)\s*=\s*get_shopping_list_delta\s*\([^)]*?is_new_plan=True",
        src,
        re.DOTALL,
    )
    assert len(matches) >= 3, (
        f"tools.py::execute_modify_single_meal debe invocar "
        f"`get_shopping_list_delta(... is_new_plan=True ...)` para los 3 "
        f"aggr_*. Encontradas: {len(matches)}."
    )


# ---------------------------------------------------------------------------
# 4. cron_tasks.py callsites usan is_new_plan=True
# ---------------------------------------------------------------------------
def test_cron_tasks_callsites_store_canonical():
    """Los 4 callsites de cron_tasks.py que computan aggr_* y los persisten
    en aggregated_shopping_list_weekly DEBEN usar is_new_plan=True.

    Sitios identificados:
      - _persist_pantry_supplement_to_plan_data (~línea 8546)
      - _clear_pantry_supplement_from_plan_data (~línea 8688)
      - _process_pending_shopping_lists chunk worker (~línea 14512)
      - process_plan_chunk_queue chunk worker T2 (~línea 25316)

    Este test cuenta TODAS las invocaciones aggr_X = ... con is_new_plan=True
    en cron_tasks.py. Debe haber al menos 12 (3 por sitio × 4 sitios).
    """
    src = _read(_CRON_TASKS)
    # Buscar tanto invocaciones get_shopping_list_delta como _gsld (alias local).
    matches = re.findall(
        r"aggr_(?:7|15|30)\s*=\s*(?:get_shopping_list_delta|_gsld)\s*\([^)]*?is_new_plan=True",
        src,
        re.DOTALL,
    )
    assert len(matches) >= 12, (
        f"cron_tasks.py debe tener ≥12 invocaciones aggr_X = (get_shopping_list_delta|_gsld)"
        f"(... is_new_plan=True ...) — 3 por cada uno de los 4 sitios identificados. "
        f"Encontradas: {len(matches)}. Si la cuenta bajó, alguien revertió is_new_plan a "
        f"False en algún callsite → bug 'agotar+reponer rompe PDF' puede regresar."
    )


# ---------------------------------------------------------------------------
# 5. Sanity check: el marker está presente en los archivos modificados
# ---------------------------------------------------------------------------
def test_marker_present_in_modified_files():
    """Cada archivo modificado por el refactor debe contener el marker
    `P3-CANONICAL-AGG-WEEKLY` en al menos un comment — ancla para grep
    y para que un futuro mantenedor entienda el invariante al renombrar.
    """
    for p in (_SHOPPING_CALC, _ROUTERS_PLANS, _TOOLS_PY, _CRON_TASKS):
        src = _read(p)
        assert "P3-CANONICAL-AGG-WEEKLY" in src, (
            f"Archivo {p.name} debe contener el marker P3-CANONICAL-AGG-WEEKLY "
            f"en un comment — sin esto, el invariante queda sin ancla."
        )
