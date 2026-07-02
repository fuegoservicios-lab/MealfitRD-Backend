"""[P1-BUDGET-COST-SSOT + P1-BUDGET-RECONCILE + P1-BUDGET-TIER-LEVERS · 2026-07-02]

Bloque A del batch P1-OBJECTIVE-V4-BATCH (audit v4 presupuesto↔lista):
1. COST-SSOT — el backend ya no descarta `total_estimated_cost`: computa y persiste
   `plan_data.shopping_cost_summary` (trip/cycle por duración) en los mismos persist-sites
   que escriben `aggregated_shopping_list*` (assemble / recalc / chat-modify / inline-recalc).
2. RECONCILE — el costo real del ciclo se compara contra el presupuesto del formulario
   (custom → monto declarado; low/medium/high → banda × piso de metas; unlimited → sin techo)
   y se persiste `plan_data.budget_reconciliation` con status {dentro|cerca|excedido|sin_limite}.
3. TIER-LEVERS — el tier deja de ser solo-prompt: boost del sorteo hacia el tercio más barato
   (economía), cheapen-pass determinista pre-engine (sustituciones curadas premium→económico,
   acotadas y allergen/dislike-safe) y sugerencias de ahorro con el Supermercado RD al excederse.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_SC_SRC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_NC_SRC = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
_AI_SRC = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
_PL_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TO_SRC = (_BACKEND / "tools.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# ════════════════════════════════════════════════════════════════════════════
# 1. P1-BUDGET-COST-SSOT — compute_shopping_cost_summary
# ════════════════════════════════════════════════════════════════════════════
def _items(*pairs):
    return [{"name": f"item{i}", "estimated_cost_rd": c, "is_perishable": p}
            for i, (c, p) in enumerate(pairs)]


def test_cost_summary_math_and_cycle_semantics(sc):
    weekly = _items((100.0, True), (200.0, False), (0.0, False))  # 1 sin precio
    biweekly = _items((100.0, True), (400.0, False))   # híbrida: estables ×2 ya dentro
    monthly = _items((100.0, True), (800.0, False))
    s = sc.compute_shopping_cost_summary(weekly, biweekly, monthly, "biweekly")
    assert s and s["active_duration"] == "biweekly"
    w = s["by_duration"]["weekly"]
    assert w["trip_total_rd"] == 300.0 and w["cycle_total_rd"] == 300.0  # weekly: ciclo == compra
    assert w["stable_rd"] == 200.0 and w["perishable_rd"] == 100.0
    assert w["items_priced"] == 2 and w["items_total"] == 3
    b = s["by_duration"]["biweekly"]
    assert b["cycle_weeks"] == 2 and b["cycle_total_rd"] == 400.0 + 100.0 * 2
    m = s["by_duration"]["monthly"]
    assert m["cycle_weeks"] == 4 and m["cycle_total_rd"] == 800.0 + 100.0 * 4


def test_cost_summary_knob_off_returns_none(sc, monkeypatch):
    monkeypatch.setenv("MEALFIT_SHOPPING_COST_SUMMARY", "false")
    assert sc.compute_shopping_cost_summary([], [], [], "weekly") is None


def test_cost_summary_persist_sites_wired():
    """Los 4 persist-sites de listas también persisten el summary (anchors)."""
    for src, label in ((_GO_SRC, "graph"), (_PL_SRC, "plans"), (_TO_SRC, "tools")):
        assert "shopping_cost_summary" in src, f"{label}: falta persist del summary"
        assert "P1-BUDGET-COST-SSOT" in src, f"{label}: falta el marker"
    # inline-recalc de updates refresca summary + reconciliación
    assert "refresh_budget_reconciliation" in _PL_SRC and "refresh_budget_reconciliation" in _TO_SRC


# ════════════════════════════════════════════════════════════════════════════
# 2. P1-BUDGET-RECONCILE — referencia + reconciliación
# ════════════════════════════════════════════════════════════════════════════
def _summary(cycle_total, duration="weekly", priced=10, total=12):
    return {"active_duration": duration,
            "by_duration": {duration: {"cycle_total_rd": cycle_total,
                                       "items_priced": priced, "items_total": total}}}


def test_reference_custom_dop_and_usd(nc, monkeypatch):
    monkeypatch.setenv("MEALFIT_BUDGET_USD_TO_DOP", "60")
    ref = nc.build_budget_reference({"budget": "custom", "budgetAmount": "8000",
                                     "budgetCurrency": "DOP", "groceryDuration": "weekly"})
    assert ref and ref["tier"] == "custom" and ref["reference_rd"] == 8000
    ref_usd = nc.build_budget_reference({"budget": "custom", "budgetAmount": "100",
                                         "budgetCurrency": "USD", "groceryDuration": "weekly"})
    assert ref_usd and ref_usd["reference_rd"] == 6000


def test_reference_tiers_and_unlimited(nc):
    low = nc.build_budget_reference({"budget": "low", "groceryDuration": "weekly"})
    high = nc.build_budget_reference({"budget": "high", "groceryDuration": "weekly"})
    unl = nc.build_budget_reference({"budget": "unlimited", "groceryDuration": "weekly"})
    assert low and high and low["reference_rd"] < high["reference_rd"]  # banda low < high
    assert low["floor_rd"] > 0
    assert unl and unl["reference_rd"] is None
    assert nc.build_budget_reference({"budget": ""}) is None  # sin presupuesto → None


def test_reconcile_statuses(nc):
    ref = {"tier": "custom", "basis": "custom", "reference_rd": 10000, "floor_rd": 4000}
    assert nc.reconcile_budget_with_cost(ref, _summary(9000))["status"] == "dentro"
    assert nc.reconcile_budget_with_cost(ref, _summary(10500))["status"] == "cerca"     # ≤ +10%
    exced = nc.reconcile_budget_with_cost(ref, _summary(13000))
    assert exced["status"] == "excedido" and exced["delta_rd"] == 3000
    sin_lim = nc.reconcile_budget_with_cost({"tier": "unlimited", "reference_rd": None}, _summary(9000))
    assert sin_lim["status"] == "sin_limite"
    # sin precios (cycle_total<=0) → None: mejor callar que inventar
    assert nc.reconcile_budget_with_cost(ref, _summary(0)) is None


def test_refresh_uses_persisted_reference_and_keeps_suggestions(nc):
    plan = {
        "budget_reconciliation": {"tier": "custom", "basis": "custom", "reference_rd": 10000,
                                  "floor_rd": 4000, "status": "dentro",
                                  "suggestions": [{"type": "marca", "text": "x"}]},
        "shopping_cost_summary": _summary(13000),
    }
    nc.refresh_budget_reconciliation(plan)
    rec = plan["budget_reconciliation"]
    assert rec["status"] == "excedido" and rec["estimated_cycle_rd"] == 13000
    assert rec["suggestions"], "las suggestions previas se preservan en el refresh"


# ════════════════════════════════════════════════════════════════════════════
# 3. P1-BUDGET-TIER-LEVERS — economía determinista
# ════════════════════════════════════════════════════════════════════════════
def test_budget_prefers_economy(nc, monkeypatch):
    assert nc.budget_prefers_economy({"budget": "low"}) is True
    assert nc.budget_prefers_economy({"budget": "medium"}) is False
    assert nc.budget_prefers_economy({"budget": "unlimited"}) is False
    assert nc.budget_prefers_economy({}) is False
    # custom AJUSTADO (< piso×1.3) → True; holgado → False. Piso semanal base = 4000.
    tight = {"budget": "custom", "budgetAmount": "4500", "groceryDuration": "weekly"}
    loose = {"budget": "custom", "budgetAmount": "50000", "groceryDuration": "weekly"}
    assert nc.budget_prefers_economy(tight) is True
    assert nc.budget_prefers_economy(loose) is False


def _cheapen_days():
    return [{"meals": [{
        "name": "Salmón a la Plancha con Quinoa",
        "ingredients": ["150g de salmón", "80g de quinoa", "100g de brócoli"],
        "recipe": ["Mise en place: pesa todo.", "El Toque de Fuego: cocina 8 min.", "Montaje: sirve."],
    }]}]


@pytest.fixture()
def cheapen_env(go, monkeypatch):
    """Cheapen-pass determinista sin DB: precio map sintético + economía forzada."""
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 3)
    prices = {"salmon": 600.0, "filete de pescado blanco": 127.0,
              "quinoa": 380.0, "arroz integral": 72.0,
              "almendras": 900.0, "mani": 280.0}
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: prices)
    import nutrition_calculator as _nc
    monkeypatch.setattr(_nc, "budget_prefers_economy", lambda fd: True)
    return go


def test_cheapen_pass_substitutes_and_marks(cheapen_env):
    go = cheapen_env
    days = _cheapen_days()
    n = go._apply_budget_cheapen_pass(days, {"budget": "low"})
    assert n == 2, f"salmón y quinoa debían sustituirse (n={n})"
    meal = days[0]["meals"][0]
    ings = " | ".join(meal["ingredients"])
    assert "Filete de pescado blanco" in ings and "Arroz integral" in ings
    assert "salmón" not in ings.lower().replace("salmon", "salmón") or "salmón" not in ings.lower()
    assert "Filete de pescado blanco" in meal["name"], "el nombre del plato debe quedar honesto"
    assert meal["_budget_substitutions"], "debe marcar las sustituciones (honestidad)"


def test_cheapen_pass_respects_allergies_and_dislikes(cheapen_env):
    go = cheapen_env
    # Alergia a pescado → el candidato 'Filete de pescado blanco' viola el scan → skip.
    days = _cheapen_days()
    n = go._apply_budget_cheapen_pass(days, {"budget": "low", "allergies": ["pescado"]})
    assert "Filete de pescado blanco" not in " ".join(days[0]["meals"][0]["ingredients"])
    # Dislike del candidato → skip (almendras → Maní bloqueado).
    days2 = [{"meals": [{"name": "Avena con Almendras",
                         "ingredients": ["30g de almendras", "40g de avena"], "recipe": []}]}]
    go._apply_budget_cheapen_pass(days2, {"budget": "low", "dislikes": ["maní"]})
    assert "Maní" not in " ".join(days2[0]["meals"][0]["ingredients"])
    assert n >= 1  # la quinoa (no-pescado) sí pudo sustituirse en el primer plan


def test_cheapen_pass_noop_without_economy_signal(go, monkeypatch):
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    days = _cheapen_days()
    assert go._apply_budget_cheapen_pass(days, {"budget": "unlimited"}) == 0
    assert "salmón" in days[0]["meals"][0]["ingredients"][0]


def test_tier_lever_anchors_and_defaults(go):
    # Knobs del cheapen-pass (defaults de código).
    assert go.BUDGET_CHEAPEN_PASS_ENABLED is True
    assert go.BUDGET_CHEAPEN_MAX_SUBS == 3
    # Pool weighting cableado en el sorteo (ai_helpers) tras el transform-boost.
    assert "P1-BUDGET-TIER-LEVERS (pool weighting)" in _AI_SRC
    assert 'MEALFIT_BUDGET_POOL_WEIGHT", "2.0"' in _AI_SRC
    assert "budget_prefers_economy" in _AI_SRC
    # Sugerencias de ahorro con el Supermercado RD (assemble) + helper con cache.
    assert "cheapest_supermarket_variant" in _GO_SRC
    assert "def cheapest_supermarket_variant" in _SC_SRC
    assert "_fetch_supermarket_price_floor_map" in _SC_SRC
    # SSOT compartido del trigger de economía.
    assert "def budget_prefers_economy" in _NC_SRC
