"""[P1-AUDIT-V3-BATCH · 2026-07-02] Los 8 P1 del audit objetivo v3 (plan 1 P0 / 8 P1 / ~22 P2):

1. P1-STEPS-STALE-POSTCLOSER — qty-sync de pasos re-corre TRAS el micro-closer/retrim/requantize en
   swap-persist y chat-modify (antes los pasos quedaban stale plan-wide).
2. P1-EXPAND-QTY-SYNC — count/qty-sync sobre los pasos del Chef en expand, EN ORIGEN (persist + guests),
   ANTES del cobro.
3. P1-INGREDIENTS-NONEMPTY — invariante de comida mínima viable: ingredients=[] se materializa desde el
   nombre en ambos finalizers, ANTES del qty-presence (que inyecta porciones default).
4. P1-BAND-PARITY-UPDATES — `apply_update_band_parity`: contrato `_quality_degraded` de S1 (marca Y
   limpieza) en regen-day/swap-persist/chat-modify.
5. P1-PANTRY-DEGRADED-SIGNAL — reverts pantry de agent.py marcan `_out["_pantry_limited"]=True` (señal
   estructurada que el parity atribuye como `_quality_degraded_pantry_limited`).
6. P1-CONDITION-CEILINGS-UPDATES — `apply_update_condition_ceilings`: re-verifica los techos/pisos por
   condición (sodio/HTA, satfat/dislipidemia) sobre el panel recomputado en las 3 superficies de update.
7. P1-BAKING-STAPLES — despensa básica de horneado: polvo de hornear/levadura/bicarbonato/vainilla NO se
   amputan de la lista (se listan ~1 empaque sin precio) + excepción explícita en el prompt verified-only.
8. P1-DISH-QUALITY-GATE-ON — soft gate de calidad de plato flipped ON (datos flota: low_quality_ratio=0.0
   → tasa de disparo ~0; red determinista, nunca cero-plan).

tooltip-anchors: P1-STEPS-STALE-POSTCLOSER, P1-EXPAND-QTY-SYNC, P1-INGREDIENTS-NONEMPTY,
P1-BAND-PARITY-UPDATES, P1-PANTRY-DEGRADED-SIGNAL, P1-CONDITION-CEILINGS-UPDATES, P1-BAKING-STAPLES,
P1-DISH-QUALITY-GATE-ON
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_SHOP = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch: los 8 markers viven en el código
# ---------------------------------------------------------------------------

def test_all_batch_markers_present():
    src_all = _GO + _PLANS + _TOOLS + _AGENT + _SHOP
    for marker in ("P1-STEPS-STALE-POSTCLOSER", "P1-EXPAND-QTY-SYNC", "P1-INGREDIENTS-NONEMPTY",
                   "P1-BAND-PARITY-UPDATES", "P1-PANTRY-DEGRADED-SIGNAL",
                   "P1-CONDITION-CEILINGS-UPDATES", "P1-BAKING-STAPLES", "P1-DISH-QUALITY-GATE-ON"):
        assert marker in src_all, f"marker {marker} ausente del código"


# ---------------------------------------------------------------------------
# 1) P1-STEPS-STALE-POSTCLOSER — qty-sync DESPUÉS del requantize del closer
# ---------------------------------------------------------------------------

def test_steps_stale_postcloser_wired_after_requantize():
    # swap-persist: el qty-sync del fix aparece tras el _qz_swap del requantize
    i_qz = _PLANS.index("_qz_swap(plan_data, _qdb_swap)")
    i_sq = _PLANS.index("_sync_recipe_step_quantities as _sq_swap")
    assert i_qz < i_sq, "swap-persist: qty-sync debe correr DESPUÉS del re-quantize del closer"
    # chat-modify: espejo
    i_qz_cm = _TOOLS.index("_qz_cm(plan_data_fresh, _qdb_cm)")
    i_sq_cm = _TOOLS.index("_sync_recipe_step_quantities as _sq_cm")
    assert i_qz_cm < i_sq_cm, "chat-modify: qty-sync debe correr DESPUÉS del re-quantize del closer"


# ---------------------------------------------------------------------------
# 2) P1-EXPAND-QTY-SYNC — en origen, antes del cobro
# ---------------------------------------------------------------------------

def test_expand_qty_sync_before_charge():
    i_sq = _PLANS.index("_sync_recipe_step_quantities as _sq_exp")
    i_charge = _PLANS.index('log_api_usage(user_id, "llm_recipe_expand")')
    assert i_sq < i_charge, "el qty-sync de expand debe correr ANTES de cobrar/persistir"


# ---------------------------------------------------------------------------
# 3) P1-INGREDIENTS-NONEMPTY — funcional + wiring antes del qty-presence
# ---------------------------------------------------------------------------

def test_ingredients_nonempty_fills_from_name():
    meal = {"name": "Pollo Guisado con Yuca y Tayota", "ingredients": [], "recipe": ["paso"]}
    assert g._ensure_nonempty_ingredients(meal) is True
    assert meal["ingredients"] == ["Pollo Guisado", "Yuca", "Tayota"]
    assert meal["_dish_quality_degraded"] is True
    assert meal["_ingredients_backfilled"] is True


def test_ingredients_nonempty_noop_when_present():
    meal = {"name": "Pollo", "ingredients": ["120 g de Pollo"]}
    assert g._ensure_nonempty_ingredients(meal) is False
    assert meal["ingredients"] == ["120 g de Pollo"]
    assert "_ingredients_backfilled" not in meal


def test_ingredients_nonempty_normalizes_string():
    meal = {"name": "Pollo", "ingredients": "120 g de Pollo"}
    assert g._ensure_nonempty_ingredients(meal) is False   # normaliza, no cuenta como fill
    assert meal["ingredients"] == ["120 g de Pollo"]


def test_ingredients_nonempty_no_name_marks_degraded():
    meal = {"name": "", "ingredients": []}
    assert g._ensure_nonempty_ingredients(meal) is False
    assert meal.get("_dish_quality_degraded") is True


def test_ingredients_nonempty_knob_off(monkeypatch):
    monkeypatch.setattr(g, "INGREDIENTS_NONEMPTY_BACKSTOP_ENABLED", False)
    meal = {"name": "Pollo Guisado", "ingredients": []}
    assert g._ensure_nonempty_ingredients(meal) is False
    assert meal["ingredients"] == []


def test_ingredients_nonempty_runs_before_qty_presence():
    # persist boundary (finalize_plan_data_coherence)
    i_pb = _GO.index("def finalize_plan_data_coherence")
    body_pb = _GO[i_pb:_GO.index("def finalize_single_meal_recipe_coherence", i_pb)]
    assert body_pb.index("_ensure_nonempty_ingredients(") < body_pb.index("_ensure_ingredient_quantities(")
    # finalizador per-meal de updates
    i_sm = _GO.index("def finalize_single_meal_recipe_coherence")
    body_sm = _GO[i_sm:i_sm + 8000]
    assert body_sm.index("_ensure_nonempty_ingredients(") < body_sm.index("_ensure_ingredient_quantities(")


# ---------------------------------------------------------------------------
# 4) P1-BAND-PARITY-UPDATES — marca / limpia / respeta razones ajenas
# ---------------------------------------------------------------------------

def _plan_with_ratio(ratio: float) -> dict:
    """Plan de 1 día cuyos macros entregados son `ratio` × target (banda = [90,112]%)."""
    tp, tc, tf = 150.0, 200.0, 60.0
    return {
        "calories": 2000,
        "macros": {"protein": f"{tp:.0f}g", "carbs": f"{tc:.0f}g", "fats": f"{tf:.0f}g"},
        "days": [{"day": 1, "meals": [{
            "meal": "Almuerzo", "name": "Test",
            "cals": 2000 * ratio, "protein": tp * ratio, "carbs": tc * ratio, "fats": tf * ratio,
        }]}],
    }


def test_band_parity_marks_out_of_band():
    plan = _plan_with_ratio(0.40)   # 40% del target → todas las celdas fuera de banda
    bs = g.apply_update_band_parity(plan, surface="swap_persist", pantry_limited=True)
    assert isinstance(bs, dict)
    assert plan.get("_quality_degraded") is True
    assert str(plan.get("_quality_degraded_reason", "")).startswith("low_band")
    assert plan.get("_quality_degraded_surface") == "swap_persist"
    assert plan.get("_quality_degraded_pantry_limited") is True


def test_band_parity_clears_when_back_in_band():
    plan = _plan_with_ratio(1.0)    # 100% del target → en banda
    plan["_quality_degraded"] = True
    plan["_quality_degraded_reason"] = "low_band_score"
    plan["_quality_degraded_surface"] = "regen_day"
    g.apply_update_band_parity(plan, surface="chat_modify")
    assert "_quality_degraded" not in plan
    assert "_quality_degraded_reason" not in plan
    assert "_quality_degraded_surface" not in plan


def test_band_parity_respects_foreign_reasons():
    plan = _plan_with_ratio(1.0)
    plan["_quality_degraded"] = True
    plan["_quality_degraded_reason"] = "max_attempts"
    g.apply_update_band_parity(plan, surface="swap_persist")
    assert plan.get("_quality_degraded") is True        # razón ajena → intacta
    assert plan.get("_quality_degraded_reason") == "max_attempts"


def test_band_parity_knob_off(monkeypatch):
    monkeypatch.setattr(g, "UPDATE_BAND_PARITY_ENABLED", False)
    assert g.apply_update_band_parity(_plan_with_ratio(0.4), surface="swap_persist") is None


def test_band_parity_wired_on_three_surfaces():
    assert 'surface="swap_persist"' in _PLANS
    assert 'surface="regen_day"' in _PLANS
    assert 'surface="chat_modify"' in _TOOLS


# ---------------------------------------------------------------------------
# 5) P1-PANTRY-DEGRADED-SIGNAL — reverts marcan el meal
# ---------------------------------------------------------------------------

def test_pantry_reverts_set_structured_flag():
    assert _AGENT.count('_out["_pantry_limited"] = True') >= 2, \
        "ambos reverts pantry (rebalance + protein-closer) deben marcar _pantry_limited"


# ---------------------------------------------------------------------------
# 6) P1-CONDITION-CEILINGS-UPDATES — marca con surface / limpia / respeta ajenas
# ---------------------------------------------------------------------------

def test_condition_ceilings_marks_with_surface(monkeypatch):
    def _fake_mark(plan, fd, was_fb, attempt):
        plan["_quality_degraded"] = True
        plan["_quality_degraded_reason"] = "condition_panel_gap"
        return True
    monkeypatch.setattr(g, "_maybe_mark_panel_degraded", _fake_mark)
    plan = {"micronutrient_report": {"gaps": []}}
    assert g.apply_update_condition_ceilings(plan, {}, surface="swap_persist") is True
    assert plan.get("_quality_degraded_surface") == "swap_persist"


def test_condition_ceilings_clears_stale_panel_reason(monkeypatch):
    monkeypatch.setattr(g, "_maybe_mark_panel_degraded", lambda p, f, w, a: False)
    plan = {"_quality_degraded": True, "_quality_degraded_reason": "condition_panel_gap",
            "_quality_degraded_severity": "minor"}
    assert g.apply_update_condition_ceilings(plan, {}, surface="regen_day") is False
    assert "_quality_degraded" not in plan     # el update reparó el techo → banner limpiado


def test_condition_ceilings_respects_foreign_reasons(monkeypatch):
    monkeypatch.setattr(g, "_maybe_mark_panel_degraded", lambda p, f, w, a: False)
    plan = {"_quality_degraded": True, "_quality_degraded_reason": "medical_critical"}
    g.apply_update_condition_ceilings(plan, {}, surface="swap_persist")
    assert plan.get("_quality_degraded") is True   # razón ajena → intacta


def test_condition_ceilings_wired_on_three_surfaces():
    assert _PLANS.count("apply_update_condition_ceilings") >= 2   # swap-persist + regen-day
    assert "apply_update_condition_ceilings" in _TOOLS            # chat-modify


# ---------------------------------------------------------------------------
# 7) P1-BAKING-STAPLES — despensa básica de horneado
# ---------------------------------------------------------------------------

def test_baking_staple_detector():
    import shopping_calculator as sc
    for nm in ("polvo de hornear", "Polvo para Hornear", "levadura seca", "bicarbonato de sodio",
               "extracto de vainilla", "Vainilla"):
        assert sc.is_baking_pantry_staple(nm) is True, nm
    for nm in ("Pollo", "arroz blanco", "harina de trigo", "salsa de soya"):
        assert sc.is_baking_pantry_staple(nm) is False, nm


def test_baking_staples_keep_wired_in_drop_path():
    # callsites (no las defs): el branch keep del aggregator y el drop indentado bajo su else.
    i_keep = _SHOP.index("if _baking_staples_keep_enabled() and is_baking_pantry_staple(name):")
    i_drop = _SHOP.index("                record_verified_only_drop(name)")
    assert i_keep < i_drop, "el keep de staples debe evaluarse ANTES del drop VERIFIED-ONLY"
    assert 'display_cat = "DESPENSA BÁSICA"' in _SHOP


def test_baking_staples_prompt_exception():
    assert "EXCEPCIÓN (despensa básica de horneado)" in _GO
    assert "polvo de hornear" in _GO


def test_baking_staples_knob_default_on():
    import shopping_calculator as sc
    assert sc._baking_staples_keep_enabled() is True


# ---------------------------------------------------------------------------
# 8) P1-DISH-QUALITY-GATE-ON
# ---------------------------------------------------------------------------

def test_dish_quality_soft_gate_default_on():
    assert g.DISH_QUALITY_SOFT_GATE_ENABLED is True
    assert "P1-DISH-QUALITY-GATE-ON" in _GO
