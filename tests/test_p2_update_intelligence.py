"""[P2-UPDATE-INTELLIGENCE · 2026-06-23] Regresión de los 6 P2 del audit de inteligencia del motor.

  P2-8  MEALFIT_SWAP_TARGET_FROM_SLOT (OFF) — swap valida contra el slot OBJETIVO, no el plato drifteado.
  P2-9  MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN (ON) — gain_muscle: proteína-main de alta densidad en swap.
  P2-10 MEALFIT_CROSS_DAY_PROTEIN_GATE (advisory) — detección always-on de monotonía cross-day en review.
  P2-11 (bug fix, sin knob) — fallback de proteína para el gate scope='day' (espejo de scope='meal').
  P2-12 MEALFIT_UPDATE_HYDRATE_BIOMETRICS (ON) — hidrata biométricos server-side para el gate del swap.
  P2-13 MEALFIT_SWAP_CALS_TOLERANCE_MULT (1.5 default) — multiplicador de tolerancia kcal hecho knob.
"""
import ast
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
PLANS = _read("routers/plans.py")
ORCH = _read("graph_orchestrator.py")
INVSUF = _read("inventory_sufficiency.py")
AIHELP = _read("ai_helpers.py")
NUTRI = _read("nutrition_calculator.py")


# ── P2-8: swap valida contra el slot objetivo ─────────────────────────────────
def test_p2_8_swap_slot_target_wired_and_default_off():
    src = _func_src(AGENT, "swap_meal")
    assert "MEALFIT_SWAP_TARGET_FROM_SLOT" in src
    assert "allocate_macros_per_slot" in src and "get_nutrition_targets" in src
    # default OFF (riesgo de más fallos pantry-strict): el knob se lee con default 'false'
    assert '"MEALFIT_SWAP_TARGET_FROM_SLOT", "false"' in src


# ── P2-9: gain_muscle proteína de alta densidad en updates ────────────────────
def test_p2_9_low_density_set_is_module_level():
    from ai_helpers import _LOW_DENSITY_AS_MAIN
    assert isinstance(_LOW_DENSITY_AS_MAIN, set)
    for m in ("habichuelas rojas", "queso ricotta", "yogurt", "gandules"):
        assert m in _LOW_DENSITY_AS_MAIN
    # NO debe contener "yogurt griego" (alto en proteína)
    assert "yogurt griego" not in _LOW_DENSITY_AS_MAIN


def test_p2_9_swap_filters_low_density_for_gain_muscle():
    src = _func_src(AGENT, "swap_meal")
    assert "_LOW_DENSITY_AS_MAIN" in src, "swap debe reusar el set módulo-level"
    assert "gain_muscle" in src and "MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN" in src


# ── P2-10: monotonía cross-day always-on ──────────────────────────────────────
def test_p2_10_variety_report_detects_cross_day_protein():
    from graph_orchestrator import build_variety_report
    plan = {"days": [
        {"day": 1, "meals": [{"name": "Pollo guisado", "ingredients": ["Pechuga de pollo", "Arroz"]}]},
        {"day": 2, "meals": [{"name": "Pollo al horno", "ingredients": ["Pollo", "Yuca"]}]},
        {"day": 3, "meals": [{"name": "Pollo a la plancha", "ingredients": ["Pollo", "Ensalada"]}]},
    ]}
    rep = build_variety_report(plan)
    assert "cross_day_proteins" in rep
    assert any("pollo" in k.lower() for k in rep["cross_day_proteins"]), "debe detectar pollo en 3 días"


def test_p2_10_advisory_by_default_does_not_add_to_issues(monkeypatch):
    """Default (gate OFF): cross-day NO entra a `issues` (advisory puro, sin loops de retry)."""
    monkeypatch.delenv("MEALFIT_CROSS_DAY_PROTEIN_GATE", raising=False)
    from graph_orchestrator import build_variety_report
    plan = {"days": [
        {"day": 1, "meals": [{"name": "Pollo a", "ingredients": ["Pollo"]}]},
        {"day": 2, "meals": [{"name": "Pollo b", "ingredients": ["Pollo"]}]},
        {"day": 3, "meals": [{"name": "Pollo c", "ingredients": ["Pollo"]}]},
    ]}
    rep = build_variety_report(plan)
    assert not any("cross-day" in str(i).lower() or "monoton" in str(i).lower() for i in rep["issues"])


def test_p2_10_gate_on_adds_to_issues(monkeypatch):
    monkeypatch.setenv("MEALFIT_CROSS_DAY_PROTEIN_GATE", "true")
    from graph_orchestrator import build_variety_report
    plan = {"days": [
        {"day": 1, "meals": [{"name": "Pollo a", "ingredients": ["Pollo"]}]},
        {"day": 2, "meals": [{"name": "Pollo b", "ingredients": ["Pollo"]}]},
        {"day": 3, "meals": [{"name": "Pollo c", "ingredients": ["Pollo"]}]},
    ]}
    rep = build_variety_report(plan)
    assert any("monoton" in str(i).lower() for i in rep["issues"])


# ── P2-11: fallback de proteína scope='day' ───────────────────────────────────
def test_p2_11_day_scope_protein_fallback_present():
    assert "P5-DAY-PROTEIN-FALLBACK" in INVSUF
    # el elif espeja la condición de scope='meal' (kcal>0 pero protein==0 → daily)
    assert 'required["protein_g"] <= 0 and required["kcal"] > 0' in INVSUF


# ── P2-12: biométricos hidratados server-side ─────────────────────────────────
def test_p2_12_enrich_hydrates_biometrics():
    src = _func_src(PLANS, "_enrich_clinical_from_profile")
    assert "MEALFIT_UPDATE_HYDRATE_BIOMETRICS" in src
    for _bk in ("weight", "height", "age", "gender", "activityLevel", "weightUnit", "goal"):
        assert f'"{_bk}"' in src, f"el enrich debe hidratar {_bk}"


# ── P2-13: tolerancia kcal como knob ──────────────────────────────────────────
def test_p2_13_cals_tolerance_mult_default_and_clamp(monkeypatch):
    from nutrition_calculator import _swap_cals_tolerance_mult
    monkeypatch.delenv("MEALFIT_SWAP_CALS_TOLERANCE_MULT", raising=False)
    assert _swap_cals_tolerance_mult() == 1.5, "default 1.5 = comportamiento previo (sin cambio)"
    monkeypatch.setenv("MEALFIT_SWAP_CALS_TOLERANCE_MULT", "1.2")
    assert abs(_swap_cals_tolerance_mult() - 1.2) < 1e-9
    monkeypatch.setenv("MEALFIT_SWAP_CALS_TOLERANCE_MULT", "9.0")
    assert _swap_cals_tolerance_mult() == 2.0  # clamp alto
    monkeypatch.setenv("MEALFIT_SWAP_CALS_TOLERANCE_MULT", "0.1")
    assert _swap_cals_tolerance_mult() == 1.0  # clamp bajo


def test_p2_13_validator_uses_knob():
    src = _read("nutrition_calculator.py")
    assert "_swap_cals_tolerance_mult()" in src
    assert "tolerance_pct * _swap_cals_tolerance_mult()" in src
