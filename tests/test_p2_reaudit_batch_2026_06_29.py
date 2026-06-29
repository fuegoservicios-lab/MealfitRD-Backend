"""[P2-REAUDIT-BATCH · 2026-06-29] Lote de 18 P2 del re-audit del objetivo plan-gen (post 3 P1).

Cubre (anclas estructurales parser-based + funcionales Neon-free):
  Slot:    P2-SLOT-CENA-HEAVY-SOUP, P2-SLOT-ALMUERZO, P2-SLOT-MERIENDA-JUNK (funcional en test_p2_slot_audit_objective)
  Finalizer: P2-STEPVEG-PANTRY-GUARD (pantry_strict), P2-APPET-UPD-SANITY, P2-RECIPE-HUMANIZE-UPDATES, P2-APPET-UPD-CRITIQUE
  Micro:   P2-MICROCLOSER-BAND-RECHECK, P2-MICRO-ESTIMADO-BAJO, P2-MICRO-MULTISOURCE (funcional en test_p1_micro_closer_coverage)
  Macro:   P2-CARB-FLOOR (funcional), P2-BAND-GATE-PER-MACRO (knob OFF)
  Updates: P2-MACRO-UPD-1, P2-MACRO-UPD-3, P2-DISHQUAL-SURFACE-UPDATES
  Expand:  P2-EXPAND-FINALIZE
  Creatividad: P2-CREATIVITY-TRANSFORM, P2-OFF-CATALOG-SNAP-RESOLVED (F4 ya-mitigado, documentado)
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_SHOP = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_DAYGEN = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
_PLANGEN = (_BACKEND / "prompts" / "plan_generator.py").read_text(encoding="utf-8")


# ───────────────────────── knobs default-safe ─────────────────────────
def test_new_knobs_default_off():
    assert g.CARB_FLOOR_ENABLED is False, "MEALFIT_CARB_FLOOR debe nacer OFF (A/B-pending)"
    assert g.BAND_GATE_PER_MACRO is False, "MEALFIT_BAND_GATE_PER_MACRO debe nacer OFF (A/B-pending)"
    assert 0.0 <= g.BAND_GATE_PER_MACRO_THRESHOLD <= 1.0
    assert g.CARB_FLOOR_MAX_SCALE >= 1.8


# ───────────────────────── markers presentes en su archivo ─────────────────────────
def test_markers_present():
    for mk in ("P2-SLOT-CENA-HEAVY-SOUP", "P2-SLOT-ALMUERZO", "P2-SLOT-MERIENDA-JUNK"):
        assert mk in (_BACKEND / "constants.py").read_text(encoding="utf-8"), mk
    for mk in ("P2-APPET-UPD-SANITY", "P2-RECIPE-HUMANIZE-UPDATES", "P2-APPET-UPD-CRITIQUE",
               "P2-STEPVEG-PANTRY-GUARD", "P2-MICROCLOSER-BAND-RECHECK", "P2-MICRO-ESTIMADO-BAJO",
               "P2-MICRO-MULTISOURCE", "P2-CARB-FLOOR", "P2-BAND-GATE-PER-MACRO"):
        assert mk in _GRAPH, mk
    for mk in ("P2-MACRO-UPD-1", "P2-MACRO-UPD-3", "P2-DISHQUAL-SURFACE-UPDATES"):
        assert mk in _TOOLS, mk
    assert "P2-MACRO-UPD-3" in _AGENT
    assert "P2-EXPAND-FINALIZE" in _PLANS
    assert "P2-DISHQUAL-SURFACE-UPDATES" in _PLANS
    assert "P2-OFF-CATALOG-SNAP-RESOLVED" in _SHOP


# ───────────────────────── Group B: finalizer wiring ─────────────────────────
def test_finalizer_accepts_pantry_strict():
    import inspect
    sig = inspect.signature(g.finalize_single_meal_recipe_coherence)
    assert "pantry_strict" in sig.parameters
    assert sig.parameters["pantry_strict"].default is False


def test_finalizer_skips_stepveg_when_pantry_strict(monkeypatch):
    """En pantry-strict el veg-guard (que AÑADE veg de catálogo) NO corre."""
    calls = {"veg": 0}
    monkeypatch.setattr(g, "UPDATE_RECIPE_FINALIZE_ENABLED", True)
    monkeypatch.setattr(g, "RECIPE_STEP_VEG_GUARD_ENABLED", True)
    monkeypatch.setattr(g, "_add_missing_recipe_step_vegetables",
                        lambda *a, **k: calls.__setitem__("veg", calls["veg"] + 1) or 0)
    # apaga el resto para aislar
    for kb in ("RECIPE_SLICE_GRAMS_ENABLED", "LEAF_VOLUME_CAP_ENABLED", "NIGHT_RICE_AUTOFIX_ENABLED",
               "RECIPE_NONEMPTY_BACKSTOP_ENABLED", "ASSEMBLE_FINAL_QUANTIZE", "GEN_SANITY_AUTOFIX_ENABLED"):
        if hasattr(g, kb):
            monkeypatch.setattr(g, kb, False)
    meal = {"name": "Pollo", "ingredients": ["100g Pollo"], "recipe": ["Saltea con brócoli."]}
    g.finalize_single_meal_recipe_coherence(meal, db=object(), pantry_strict=True)
    assert calls["veg"] == 0, "pantry-strict NO debe correr el veg-guard"
    g.finalize_single_meal_recipe_coherence(meal, db=object(), pantry_strict=False)
    assert calls["veg"] == 1, "no-pantry-strict SÍ corre el veg-guard"


def test_finalizer_wires_sanity_and_humanize_and_combo():
    body = _GRAPH[_GRAPH.find("def finalize_single_meal_recipe_coherence("):
                 _GRAPH.find("\ndef _recipe_slice_units_to_grams(")]
    assert "_generation_sanity_autofix({\"days\": _wrap}" in body
    assert "humanize_plan_ingredients" in body
    assert "_appetibility_combo_warning" in body


# ───────────────────────── Group D: carb closer functional ─────────────────────────
class _CarbDB:
    """arroz = carbo-dominante; micros escalan con la cantidad líder."""
    def _f(self, s):
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        return (float(m.group(1)) / 100.0) if m else 1.0

    def macros_from_ingredient_string(self, s):
        if "arroz" in s.lower():
            f = self._f(s)
            return {"protein": 2.0 * f, "carbs": 28.0 * f, "fats": 0.3 * f, "kcal": 130.0 * f}
        return None


def test_carb_floor_noop_when_disabled():
    meals = [{"name": "Arroz", "ingredients": ["100g Arroz"], "carbs": 28, "cals": 130, "protein": 2, "fats": 0}]
    assert g._close_carb_gap_for_day(meals, 60.0, 500.0, _CarbDB()) is False  # knob OFF


def test_carb_floor_scales_up_when_enabled(monkeypatch):
    monkeypatch.setattr(g, "CARB_FLOOR_ENABLED", True)
    monkeypatch.setattr(g, "_ingredient_macro_group", lambda s, db: "carbs" if "arroz" in str(s).lower() else "other")
    meals = [{"name": "Arroz", "ingredients": ["100g Arroz"], "carbs": 28, "cals": 130, "protein": 2, "fats": 0}]
    changed = g._close_carb_gap_for_day(meals, 60.0, 500.0, _CarbDB(), tol=0.10, max_scale=2.5)
    assert changed is True, "debió escalar el arroz hacia el target de carbos"
    assert meals[0]["ingredients"][0] != "100g Arroz", "el ingrediente carbo-dominante debió crecer"
    assert meals[0]["carbs"] > 28, "los carbos del día debieron subir hacia el target"


# ───────────────────────── Group E/F/G wiring ─────────────────────────
def test_chat_modify_wires_macro_closers_and_band_telemetry():
    assert "_rebalance_day_macros_to_target as _rb_m" in _TOOLS
    assert "_close_protein_gap_for_meal as _cpg_m" in _TOOLS
    assert "_macro_band_low" in _TOOLS
    assert "_macro_band_low" in _AGENT  # swap band telemetry


def test_expand_finalize_wired():
    assert "_add_missing_recipe_step_vegetables as _veg_exp" in _PLANS
    assert "_append_expand_veg" in _PLANS


def test_dishqual_recompute_on_all_update_surfaces():
    assert _PLANS.count("compute_dish_quality_report") >= 2  # swap + regen
    assert "compute_dish_quality_report" in _TOOLS            # chat-modify


# ───────────────────────── Group H: creativity directive ─────────────────────────
def test_creativity_transform_directive_in_prompts():
    for src, label in ((_DAYGEN, "day_generator"), (_PLANGEN, "plan_generator")):
        assert "P2-CREATIVITY-TRANSFORM" in src, label
        # menciona transformaciones por nombre que el owner pidió
        assert "panqueque" in src.lower() and "bollos de yuca" in src.lower(), label
