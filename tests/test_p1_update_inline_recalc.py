"""[P1-UPDATE-LIST-INLINE-RECALC · 2026-07-02] (batch P1-OBJECTIVE-V4-BATCH)

Cierra la ventana "recetas nuevas + listas vacías": swap-persist y regen-day strippeaban las
4 `aggregated_shopping_list*` y delegaban el recompute al frontend (/recalculate-shopping-list);
si la pestaña moría post-persist, el plan quedaba SIN lista de compras indefinidamente (violación
de la invariante recetas↔lista). Ahora `_rebuild_plan_shopping_lists_inline` recompone las listas
DENTRO del mutator atómico como ÚLTIMO paso (post closer/requantize/qty-sync), con la misma
matemática canónica del recalc + coherence guard warn (action_taken reusado "warn_only_recalc" —
set canónico cerrado) + refresh del cost summary y la reconciliación de presupuesto. El strip
queda como estado de FALLBACK si el rebuild falla (contrato legacy).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PL_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _body_between(src: str, start_marker: str, end_marker: str) -> str:
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[i:j]


# ════════════════════════════════════════════════════════════════════════════
# Anchors estructurales
# ════════════════════════════════════════════════════════════════════════════
def test_helper_exists_with_guard_and_budget_refresh():
    assert "def _rebuild_plan_shopping_lists_inline(" in _PL_SRC
    body = _body_between(_PL_SRC, "def _rebuild_plan_shopping_lists_inline(",
                         '@router.post("/{plan_id}/swap-meal/persist")')
    # Guard SSOT dentro del helper (contrato P1-NEXT-2: write + guard en la misma ventana).
    assert "run_shopping_coherence_guard_and_append_history" in body
    # Reusa el action_taken del recalc (ES la matemática del recalc inline) — set canónico cerrado.
    assert 'action_taken="warn_only_recalc"' in body
    # Refresh de presupuesto (P1-BUDGET-COST-SSOT / P1-BUDGET-RECONCILE).
    assert "compute_shopping_cost_summary" in body
    assert "refresh_budget_reconciliation" in body
    # Matemática canónica del recalc: is_new_plan=True + híbrido con restock flags.
    assert "is_new_plan=True" in body
    assert "_build_hybrid_shopping_list" in body
    assert "restocked_at_iso" in body
    # Knob de rollback.
    assert "MEALFIT_UPDATE_INLINE_LIST_RECALC" in body


def test_both_mutators_call_rebuild_as_last_step():
    """El rebuild corre al FINAL del mutator (post band-parity) en swap-persist Y regen-day."""
    swap_body = _body_between(_PL_SRC, "def _swap_mutator(", "result = update_plan_data_atomic(")
    day_body = _body_between(_PL_SRC, "def _day_mutator(", "result = update_plan_data_atomic(plan_id, _day_mutator")
    for body, label in ((swap_body, "swap"), (day_body, "regen-day")):
        assert "_rebuild_plan_shopping_lists_inline(" in body, f"{label}: falta el rebuild inline"
        # Orden: el rebuild va DESPUÉS del band-parity (última mutación de ingredientes ya pasó).
        assert body.rindex("_rebuild_plan_shopping_lists_inline(") > body.rindex("apply_update_band_parity"), \
            f"{label}: el rebuild debe ser el ÚLTIMO paso del mutator"
        # El strip legacy sigue presente como fallback.
        assert '"aggregated_shopping_list_weekly"' in body, f"{label}: el strip-fallback desapareció"


def test_surface_labels_passed():
    assert re.search(r'_rebuild_plan_shopping_lists_inline\(\s*plan_data,\s*verified_user_id,\s*surface="swap_persist"', _PL_SRC)
    assert re.search(r'_rebuild_plan_shopping_lists_inline\(\s*pd,\s*verified_user_id,\s*surface="regen_day"', _PL_SRC)


# ════════════════════════════════════════════════════════════════════════════
# Funcional (sin DB): knob OFF → False rápido; fallo interno → False (fail-open)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def plans_mod():
    import importlib
    return importlib.import_module("routers.plans")


def test_knob_off_returns_false(plans_mod, monkeypatch):
    monkeypatch.setenv("MEALFIT_UPDATE_INLINE_LIST_RECALC", "false")
    assert plans_mod._rebuild_plan_shopping_lists_inline({}, "u1", surface="swap_persist") is False


def test_internal_failure_is_fail_open(plans_mod, monkeypatch):
    """Si el builder de listas revienta, el helper retorna False (el caller conserva el strip)."""
    monkeypatch.setenv("MEALFIT_UPDATE_INLINE_LIST_RECALC", "true")
    import shopping_calculator as _sc

    def _boom(*a, **k):
        raise RuntimeError("builder caído")

    monkeypatch.setattr(_sc, "get_shopping_list_delta", _boom)
    plan = {"days": [], "calc_household_multiplier": 1.0}
    assert plans_mod._rebuild_plan_shopping_lists_inline(plan, "u1", surface="regen_day") is False
    assert "aggregated_shopping_list" not in plan, "ante fallo NO debe escribir listas parciales"
