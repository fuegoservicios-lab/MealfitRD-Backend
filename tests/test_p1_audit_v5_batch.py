"""[P1-AUDIT-V5-BATCH · 2026-07-02] Los 2 P1 del audit objetivo v5.

1. P1-BUDGET-REF-RESCALE — la referencia de presupuesto persistida no se
   re-escalaba al cambiar duración de ciclo u hogar post-generación:
   `reconcile_budget_with_cost` comparaba el `cycle_total_rd` de la duración
   ACTIVA contra la referencia nacida con la duración de GENERACIÓN → banner
   dentro/cerca/excedido falso ×2.5-4 en Dashboard y PDF al usar el dropdown
   "Duración del Plan" (weekly→monthly: piso 7d=4000 vs 30d=13000, ratio real
   ~3.25×). Fix: re-escalado tier-basis por ratio NO lineal del piso por ciclo
   + hogar; custom lineal per-día con flag `rescaled_from_days`.

2. P1-CHATMODIFY-LISTS-SSOT — execute_modify_single_meal recomputaba y
   persistía las 4 `aggregated_shopping_list*` con `householdSize` entero
   crudo del form (ignorando `householdComposition` y el SSOT
   `calc_household_multiplier`) y elegía la lista activa con `groceryDuration`
   del form (ignorando `calc_grocery_duration`). Fix: precedencia SSOT, espejo
   de `_rebuild_plan_shopping_lists_inline`.

Origen: memoria project_objective_audit_v5_2026_07_02.md (GAP-01 y GAP-02).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_NC_SRC = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
_PL_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TO_SRC = (_BACKEND / "tools.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


def _summary(cycle_total, duration="weekly", priced=10, total=12):
    return {"active_duration": duration,
            "by_duration": {duration: {"cycle_total_rd": cycle_total,
                                       "items_priced": priced, "items_total": total}}}


# ════════════════════════════════════════════════════════════════════════════
# 1. P1-BUDGET-REF-RESCALE — reconcile re-escala la referencia
# ════════════════════════════════════════════════════════════════════════════
def test_tier_reference_rescales_nonlinear_on_duration_change(nc):
    """weekly→monthly con basis tier: referencia × ratio del piso por ciclo
    (13000/4000 = 3.25), NO lineal (30/7 ≈ 4.29). Un costo mensual proporcional
    al semanal debe seguir `dentro` — pre-fix marcaba `excedido` falso."""
    ref = {"tier": "medium", "basis": "medium", "reference_rd": 8000,
           "floor_rd": 5000, "days": 7, "household": 1}
    rec = nc.reconcile_budget_with_cost(ref, _summary(24000, duration="monthly"))
    assert rec is not None
    assert rec["reference_rd"] == 26000  # 8000 × (13000/4000)
    assert rec["floor_rd"] == 16250      # 5000 × 3.25
    assert rec["status"] == "dentro", f"24000 ≤ 26000 debía ser 'dentro' (got {rec['status']})"
    assert rec["days"] == 30 and rec["rescaled_from_days"] == 7


def test_custom_reference_rescales_linear_with_provenance_flag(nc):
    """basis custom: monto declarado re-escala lineal per-día + flag de caveat."""
    ref = {"tier": "custom", "basis": "custom", "reference_rd": 7000,
           "floor_rd": 4000, "days": 7, "household": 2}
    rec = nc.reconcile_budget_with_cost(ref, _summary(29000, duration="monthly"))
    assert rec is not None
    assert rec["reference_rd"] == 30000  # 7000 × 30/7
    assert rec["status"] == "dentro"
    assert rec["rescaled_from_days"] == 7 and rec["days"] == 30


def test_household_rescales_tier_basis_only(nc):
    """Hogar 1→3 re-escala tier-basis ×3; un custom NO (monto declarado = total)."""
    tier_ref = {"tier": "low", "basis": "low", "reference_rd": 6000,
                "floor_rd": 4000, "days": 7, "household": 1}
    rec = nc.reconcile_budget_with_cost(tier_ref, _summary(15000), active_household=3)
    assert rec["reference_rd"] == 18000 and rec["household"] == 3
    assert rec["rescaled_from_household"] == 1
    assert rec["status"] == "dentro"
    custom_ref = {"tier": "custom", "basis": "custom", "reference_rd": 6000,
                  "floor_rd": 4000, "days": 7, "household": 1}
    rec_c = nc.reconcile_budget_with_cost(custom_ref, _summary(5000), active_household=3)
    assert rec_c["reference_rd"] == 6000, "custom no re-escala por hogar"
    assert "rescaled_from_household" not in rec_c


def test_same_duration_and_household_is_noop(nc):
    """Sin cambio de duración/hogar: cero re-escalado, cero flags (no regresión)."""
    ref = {"tier": "medium", "basis": "medium", "reference_rd": 8000,
           "floor_rd": 5000, "days": 7, "household": 2}
    rec = nc.reconcile_budget_with_cost(ref, _summary(7000), active_household=2)
    assert rec["reference_rd"] == 8000 and rec["floor_rd"] == 5000
    assert "rescaled_from_days" not in rec and "rescaled_from_household" not in rec
    assert rec["status"] == "dentro"


def test_legacy_reference_without_days_fails_open(nc):
    """Referencia legacy sin `days`: no re-escala (comportamiento previo intacto)."""
    ref = {"tier": "custom", "basis": "custom", "reference_rd": 10000, "floor_rd": 4000}
    rec = nc.reconcile_budget_with_cost(ref, _summary(13000, duration="monthly"))
    assert rec["reference_rd"] == 10000 and rec["status"] == "excedido"


def test_refresh_migrates_reference_and_next_refresh_is_stable(nc):
    """El refresh persiste la referencia MIGRADA (days activos) → un segundo
    refresh con la misma duración no vuelve a escalar; provenance se preserva."""
    plan = {
        "budget_reconciliation": {"tier": "medium", "basis": "medium",
                                  "reference_rd": 8000, "floor_rd": 5000,
                                  "days": 7, "household": 1},
        "shopping_cost_summary": _summary(24000, duration="monthly"),
    }
    nc.refresh_budget_reconciliation(plan)
    first = plan["budget_reconciliation"]
    assert first["reference_rd"] == 26000 and first["days"] == 30
    nc.refresh_budget_reconciliation(plan)  # misma duración → estable
    second = plan["budget_reconciliation"]
    assert second["reference_rd"] == 26000 and second["status"] == first["status"]
    assert second["rescaled_from_days"] == 7, "provenance preservada entre refreshes"


def test_recalc_endpoint_passes_active_household():
    """Anchor: el recalc (única superficie que cambia hogar) pasa active_household."""
    assert "P1-BUDGET-REF-RESCALE" in _PL_SRC, "falta el marker en plans.py"
    # [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-H) el callsite ahora pasa además user_id
    # (sugerencias brand-aware) — el anchor acepta kwargs extra tras active_household.
    assert re.search(
        r"_p1b_rbr\(\s*plan_data_fresh\s*,\s*active_household\s*=\s*household_size\s*[,)]",
        _PL_SRC,
    ), "el recalc debe pasar active_household=household_size a refresh_budget_reconciliation"
    assert "P1-BUDGET-REF-RESCALE" in _NC_SRC, "falta el marker en nutrition_calculator.py"


# ════════════════════════════════════════════════════════════════════════════
# 2. P1-CHATMODIFY-LISTS-SSOT — listas del chat-modify con multiplier/duración SSOT
# ════════════════════════════════════════════════════════════════════════════
def _modify_lists_block() -> str:
    """Región de execute_modify_single_meal que computa las listas (marker → guard)."""
    start = _TO_SRC.index("P1-CHATMODIFY-LISTS-SSOT")
    end = _TO_SRC.index("warn_only_agent_tool")
    assert start < end
    return _TO_SRC[start:end]


def test_chatmodify_household_uses_ssot_multiplier():
    block = _modify_lists_block()
    assert 'plan_data.get("calc_household_multiplier")' in block, (
        "el multiplier de las listas debe salir del SSOT calc_household_multiplier"
    )
    assert "compute_household_multiplier" in block, (
        "fallback sin SSOT debe ser compute_household_multiplier (householdComposition-aware)"
    )
    # El patrón viejo (householdSize crudo como fuente primaria) no debe volver:
    assert not re.search(
        r"household\s*=\s*max\(1,\s*int\(form_data\.get\(\"householdSize\"", block
    ), "regresión: householdSize entero crudo como fuente primaria del multiplier"


def test_chatmodify_duration_uses_ssot():
    block = _modify_lists_block()
    assert re.search(
        r"plan_data\.get\(\"calc_grocery_duration\"\)", block
    ), "la duración de la lista activa debe preferir calc_grocery_duration"


def test_chatmodify_deltas_scale_by_multiplier():
    """Las 3 llamadas get_shopping_list_delta escalan 1×/2×/4× el multiplier SSOT."""
    block = _modify_lists_block()
    for factor in ("1.0 * household", "2.0 * household", "4.0 * household"):
        assert factor in block, f"falta multiplier={factor} en el recompute de listas"
