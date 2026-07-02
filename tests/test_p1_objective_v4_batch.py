"""[P1-OBJECTIVE-V4-BATCH · 2026-07-02] Test ancla del batch de 8 P1 del audit objetivo v4.

Ejes: presupuesto↔lista (COST-SSOT / RECONCILE / TIER-LEVERS), updates (LIST-INLINE-RECALC),
micros (PERDAY-FLOOR / SODIUM-SUGAR-EXCESS-ON), recetas (RECIPE-CONTRACT-GATE) y blindaje de
knobs load-bearing (VERIFIED-ONLY-DEFAULT-ON). Cada fix tiene su test dedicado
(test_p1_budget_intelligence / test_p1_update_inline_recalc / test_p1_micro_perday_sodium /
test_p1_recipe_contract_gate); este archivo ancla el marker (cross-link P2-HIST-AUDIT-14) y
el snapshot de defaults del batch completo.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_marker_bumped_in_app():
    assert 'P1-OBJECTIVE-V4-BATCH · 2026-07-02' in _read("app.py")


def test_all_eight_submarkers_present():
    hay = "\n".join(_read(rel) for rel in (
        "graph_orchestrator.py", "shopping_calculator.py", "nutrition_calculator.py",
        "ai_helpers.py", "tools.py", "micronutrients.py", "agent.py", "routers/plans.py",
    ))
    for marker in (
        "P1-BUDGET-COST-SSOT",
        "P1-BUDGET-RECONCILE",
        "P1-BUDGET-TIER-LEVERS",
        "P1-UPDATE-LIST-INLINE-RECALC",
        "P1-MICRO-PERDAY-FLOOR",
        "P1-SODIUM-SUGAR-EXCESS-ON",
        "P1-RECIPE-CONTRACT-GATE",
        "P1-VERIFIED-ONLY-DEFAULT-ON",
    ):
        assert marker in hay, f"falta el sub-marker {marker} en el árbol"


def test_batch_knob_defaults_snapshot():
    """Snapshot de los defaults de CÓDIGO del batch — un flip accidental falla aquí primero."""
    sc = _read("shopping_calculator.py")
    go = _read("graph_orchestrator.py")
    ag = _read("agent.py")
    pl = _read("routers/plans.py")
    # ON por diseño (levers deterministas + honestidad):
    assert '"MEALFIT_SHOPPING_COST_SUMMARY", True' in sc
    assert '"MEALFIT_VERIFIED_INGREDIENTS_ONLY", True' in sc
    assert '"MEALFIT_BUDGET_CHEAPEN_PASS", True' in go
    assert '"MEALFIT_SODIUM_SUGAR_DEGRADE", True' in go
    assert '"MEALFIT_MICRO_PERDAY_DEGRADE", True' in go
    assert '"MEALFIT_RECIPE_TIMETEMP_BACKSTOP", True' in go
    assert '"MEALFIT_SWAP_TARGET_FROM_SLOT", "true"' in ag
    assert '"MEALFIT_UPDATE_INLINE_LIST_RECALC", "true"' in pl
    # OFF por diseño (gates nuevos — flip con datos de flota, playbook dish-quality):
    assert '"MEALFIT_SODIUM_EXCESS_GATE", False' in go
    assert '"MEALFIT_RECIPE_CONTRACT_GATE", False' in go


def test_frontend_consumes_reconciliation():
    dash = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    assert "budget_reconciliation" in dash, "el Dashboard debe consumir la reconciliación"
    assert "shopping_cost_summary" in dash, "el Dashboard debe preferir el summary backend (SSOT)"
    assert "P1-BUDGET-RECONCILE" in dash
