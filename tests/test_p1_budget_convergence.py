"""[P1-BUDGET-CONVERGENCE · 2026-07-03] (audit v6 · P1-3) Loop de convergencia presupuesto→plan.

Antes: el cheapen-pass corría A CIEGAS (pre-costeo, solo tier económico, 1 vez) y nunca se
verificaba si el plan seguía excedido; las sugerencias de ahorro solo nacían en generación
(refresh reusaba stale o nada); `adjusted`/`substitutions` se perdían en cada refresh. Cierra:
  1. Pasada de convergencia post-costeo en assemble: `excedido` (CUALQUIER tier con referencia)
     → cheapen force → truth-up + re-banda (motor updates) → rebuild listas → re-costeo →
     re-reconciliación. Acotado a 1 pasada. Knob MEALFIT_BUDGET_CONVERGENCE (ON).
  2. `_apply_budget_cheapen_pass(..., force=True)` salta el gate de economía (guards intactos).
  3. `build_budget_suggestions` (shopping_calculator, SSOT) + refresh recomputa sugerencias del
     estado ACTUAL cuando cae en excedido y preserva adjusted/substitutions entre refreshes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_NC = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


@pytest.fixture()
def go():
    import graph_orchestrator
    return graph_orchestrator


@pytest.fixture()
def nc():
    import nutrition_calculator
    return nutrition_calculator


@pytest.fixture()
def sc():
    import shopping_calculator
    return shopping_calculator


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-BUDGET-CONVERGENCE" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_knob_default_on():
    assert re.search(
        r'BUDGET_CONVERGENCE_ENABLED\s*=\s*_env_bool\("MEALFIT_BUDGET_CONVERGENCE",\s*True\)', _GO
    ), "el knob MEALFIT_BUDGET_CONVERGENCE debe nacer ON"


# ════════════════════════════════════════════════════════════════════════════
# 1. cheapen force=True — salta el gate de economía, guards intactos
# ════════════════════════════════════════════════════════════════════════════
def _cheapen_days():
    return [{"meals": [{
        "name": "Salmón a la Plancha con Quinoa",
        "ingredients": ["150g de salmón", "80g de quinoa", "100g de brócoli"],
        "recipe": ["Mise en place: pesa todo.", "El Toque de Fuego: cocina 8 min.", "Montaje: sirve."],
    }]}]


@pytest.fixture()
def cheapen_env(go, monkeypatch):
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 3)
    prices = {"salmon": 600.0, "filete de pescado blanco": 127.0,
              "quinoa": 380.0, "arroz integral": 72.0}
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: prices)
    return go


def test_force_skips_economy_gate(cheapen_env, monkeypatch):
    go = cheapen_env
    import nutrition_calculator as _nc
    monkeypatch.setattr(_nc, "budget_prefers_economy", lambda fd: False)  # tier NO económico
    days_no_force = _cheapen_days()
    assert go._apply_budget_cheapen_pass(days_no_force, {"budget": "medium"}) == 0, \
        "sin force, tier no-económico → gate cerrado (comportamiento previo intacto)"
    days_force = _cheapen_days()
    n = go._apply_budget_cheapen_pass(days_force, {"budget": "medium"}, force=True)
    assert n >= 1, "force=True (post-costeo excedido) debe sustituir aunque el tier no sea económico"
    assert "Filete de pescado blanco" in " ".join(days_force[0]["meals"][0]["ingredients"])


def test_force_keeps_allergy_guard(cheapen_env, monkeypatch):
    go = cheapen_env
    import nutrition_calculator as _nc
    monkeypatch.setattr(_nc, "budget_prefers_economy", lambda fd: False)
    days = _cheapen_days()
    go._apply_budget_cheapen_pass(days, {"budget": "high", "allergies": ["pescado"]}, force=True)
    assert "Filete de pescado blanco" not in " ".join(days[0]["meals"][0]["ingredients"]), \
        "force JAMÁS debilita el guard de alergias"


# ════════════════════════════════════════════════════════════════════════════
# 2. Pasada de convergencia cableada en assemble (parser-based)
# ════════════════════════════════════════════════════════════════════════════
def test_convergence_wired_in_assemble():
    assert "P1-BUDGET-CONVERGENCE" in _GO
    blk_start = _GO.index('str(_bc_rec0.get("status") or "") == "excedido"')
    # [P0-BAND-PRE-REVIEW · 2026-07-10] 6000→7500: el re-fire del chain post-convergencia
    # (+~1k chars) empujaba el re-costeo fuera de la ventana fija.
    blk = _GO[blk_start:blk_start + 7500]
    assert "force=True" in blk, "la pasada post-costeo usa force (todos los tiers con referencia)"
    assert "apply_update_macro_engine(result, surface=\"budget_convergence\"" in blk, \
        "tras sustituir, el motor de updates re-apunta la banda (P1-UPDATE-MACRO-PARITY)"
    assert "compute_shopping_cost_summary as _bc_ccs" in blk, "re-costeo tras la sustitución"
    assert "compute_budget_reconciliation as _bc_cbr" in blk, "re-reconciliación honesta"
    assert '"converged_pass"' in blk, "la reconciliación declara que hubo pasada de convergencia"
    # orden: la convergencia corre DESPUÉS de la primera reconciliación
    assert _GO.index('result["budget_reconciliation"] = _p1b_rec') < blk_start


def test_convergence_failopen_never_wipes_lists():
    # el bloque vive dentro del try de SHOPPING MATH (que resetea listas a [] al fallar):
    # DEBE tener su propio try/except que trague todo (fail-open al estado de la 1ª pasada).
    blk_start = _GO.index("tooltip-anchor: P1-BUDGET-CONVERGENCE")
    # [P0-BAND-PRE-REVIEW · 2026-07-10] 7000→10000: mismo motivo que arriba (re-fire del chain;
    # el except del fail-open quedó a ~8.6k del anchor).
    blk = _GO[blk_start:blk_start + 10000]
    assert "except Exception as _bc_e:" in blk
    assert "pasada de convergencia no-op" in blk


# ════════════════════════════════════════════════════════════════════════════
# 3. build_budget_suggestions (SSOT) + refresh recomputa/preserva
# ════════════════════════════════════════════════════════════════════════════
def test_build_budget_suggestions(sc, monkeypatch):
    monkeypatch.setattr(sc, "cheapest_supermarket_variant", lambda name: {
        "brand": "MarcaX", "presentation": "1 lb", "price_rd": 99.0,
    })
    weekly = [
        {"name": "Salmón", "estimated_cost_rd": 900},
        {"name": "Pollo", "estimated_cost_rd": 300},
        {"name": "sin precio"},
        {"name": "Arroz", "estimated_cost_rd": 120},
    ]
    sugs = sc.build_budget_suggestions(weekly, limit=2)
    assert len(sugs) == 2
    assert sugs[0]["item"] == "Salmón", "ordenadas por costo descendente (más caro primero)"
    assert "MarcaX" in sugs[0]["text"]


def test_build_budget_suggestions_failopen(sc, monkeypatch):
    def _boom(name):
        raise RuntimeError("db down")
    monkeypatch.setattr(sc, "cheapest_supermarket_variant", _boom)
    assert sc.build_budget_suggestions([{"name": "x", "estimated_cost_rd": 10}]) == []
    assert sc.build_budget_suggestions(None) == []


def _summary(cycle_total, duration="weekly"):
    return {"active_duration": duration,
            "by_duration": {duration: {"cycle_total_rd": cycle_total,
                                       "items_priced": 10, "items_total": 12}}}


def test_refresh_preserves_adjusted_and_substitutions(nc):
    plan = {
        "budget_reconciliation": {"tier": "custom", "basis": "custom", "reference_rd": 10000,
                                  "floor_rd": 4000, "status": "excedido",
                                  "adjusted": True, "substitutions": ["salmón → pescado blanco"]},
        "shopping_cost_summary": _summary(9000),
    }
    nc.refresh_budget_reconciliation(plan)
    rec = plan["budget_reconciliation"]
    assert rec["status"] == "dentro"
    assert rec["adjusted"] is True and rec["substitutions"], \
        "adjusted/substitutions son hechos del plan — deben sobrevivir el refresh"


def test_refresh_recomputes_suggestions_on_flip_to_excedido(nc, monkeypatch):
    import shopping_calculator as _sc
    monkeypatch.setattr(_sc, "build_budget_suggestions",
                        lambda weekly, limit=5, user_id=None: [{"type": "marca", "item": "Salmón", "text": "fresco"}])
    plan = {
        "budget_reconciliation": {"tier": "custom", "basis": "custom", "reference_rd": 10000,
                                  "floor_rd": 4000, "status": "dentro"},
        "shopping_cost_summary": _summary(13000),
        "aggregated_shopping_list_weekly": [{"name": "Salmón", "estimated_cost_rd": 900}],
    }
    nc.refresh_budget_reconciliation(plan)
    rec = plan["budget_reconciliation"]
    assert rec["status"] == "excedido"
    assert rec.get("suggestions") and rec["suggestions"][0]["text"] == "fresco", \
        "el flip a excedido EN el refresh debe generar sugerencias frescas (antes: ninguna)"


def test_refresh_fallback_to_previous_suggestions(nc, monkeypatch):
    import shopping_calculator as _sc
    monkeypatch.setattr(_sc, "build_budget_suggestions", lambda weekly, limit=5, user_id=None: [])
    plan = {
        "budget_reconciliation": {"tier": "custom", "basis": "custom", "reference_rd": 10000,
                                  "floor_rd": 4000, "status": "dentro",
                                  "suggestions": [{"type": "marca", "text": "previa"}]},
        "shopping_cost_summary": _summary(13000),
    }
    nc.refresh_budget_reconciliation(plan)
    rec = plan["budget_reconciliation"]
    assert rec["status"] == "excedido"
    assert rec["suggestions"][0]["text"] == "previa", \
        "sin lista fresca disponible, las sugerencias previas se preservan (contrato previo)"
