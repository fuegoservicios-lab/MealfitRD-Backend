"""[P1-BUDGET-T2-CONVERGENCE · 2026-07-29] En el flujo deep-search el costeo de assemble
sale VACÍO (la lista aún no trae precios) → `budget_reconciliation` y la convergencia de
assemble mueren de hambre; el único momento con costo REAL es el seam T2 del chunk worker,
que MEDÍA `excedido` y no actuaba. Plan vivo 73db1e79: RD$21,673 vs RD$15,324 (ratio 1.414),
banner rojo, CERO líneas de CHEAPEN/DRIVER-AWARE en toda la generación (verificado en logs
corr=8469264b). Cierre: helper síncrono `apply_budget_convergence_for_days` (misma secuencia
del bloque de assemble) + gate en T2: excedido → sustituir → rebuild con los MISMOS
snapshots → re-costear → re-reconciliar. Una pasada, jamás loop, fail-open total.

tooltip-anchor: P1-BUDGET-T2-CONVERGENCE
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _plan():
    return {"days": [{"meals": [{"name": "Mero al Horno", "ingredients": ["200 g de mero"],
                                 "recipe": ["Hornea."], "protein": 40, "carbs": 10,
                                 "fats": 8, "cals": 280}]}],
            "aggregated_shopping_list_weekly": [
                {"name": "Mero", "price_estimate_rd": 671, "total_grams": 975}]}


def test_helper_orchestrates_and_marks(monkeypatch):
    calls = {}
    monkeypatch.setattr(go, "_apply_budget_driver_aware_pass",
                        lambda days, fd, weekly: calls.setdefault("driver", (len(days), len(weekly))) and 2 or 2)
    monkeypatch.setattr(go, "_apply_budget_cheapen_pass",
                        lambda days, fd, force=False: calls.setdefault("cheapen_force", force) or 1)
    monkeypatch.setattr(go, "_protein_repeat_autofix", lambda days, fd: 0)
    monkeypatch.setattr(go, "apply_update_macro_engine",
                        lambda pd, surface=None, db=None: calls.setdefault("engine_surface", surface))
    monkeypatch.setattr(go, "recompute_micronutrient_report_for_plan",
                        lambda pd, fd, db=None: calls.setdefault("micros", True))
    plan = _plan()
    n = go.apply_budget_convergence_for_days(plan, {"budget": "low"})
    assert n == 3
    assert plan.get("_budget_adjusted") is True
    assert calls.get("cheapen_force") is True, "el cheapen corre con force=True (salta el gate de economía)"
    assert calls.get("engine_surface") == "budget_convergence_t2"


def test_helper_zero_subs_is_pure_noop(monkeypatch):
    monkeypatch.setattr(go, "_apply_budget_driver_aware_pass", lambda *a: 0)
    monkeypatch.setattr(go, "_apply_budget_cheapen_pass", lambda *a, **k: 0)
    plan = _plan()
    assert go.apply_budget_convergence_for_days(plan, {}) == 0
    assert "_budget_adjusted" not in plan


def test_t2_seam_wired_with_rebuild_and_rereconcile():
    i = _CRON.index("P1-BUDGET-T2-CONVERGENCE")
    blk = _CRON[i:i + 4200]
    assert 'str(_bc_rec_t2.get("status") or "") == "excedido"' in blk, "gate por status excedido"
    assert "apply_budget_convergence_for_days" in blk
    assert blk.count("get_shopping_list_delta(") == 3, "rebuild de las 3 multiplicidades"
    assert "inventory_override=_inv_s" in blk, "MISMOS snapshots de la 1ª pasada (no re-fetch)"
    assert "_rbr_t2(full_plan_data)" in blk, "re-reconciliación honesta post-rebuild"
    assert "_build_hybrid_shopping_list" in blk, "las listas re-construidas pasan por el híbrido"


def test_helper_fail_open():
    assert go.apply_budget_convergence_for_days(None, None) == 0
    assert go.apply_budget_convergence_for_days({}, {}) == 0
