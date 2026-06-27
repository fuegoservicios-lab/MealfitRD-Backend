"""[P2-REGEN-DAY-MACRO-REBALANCE + P2-REGEN-DAY-BAND-SCORE · 2026-06-27] (audit Fase 2)

Lleva a /regenerate-day (S2) el rebalanceador de macros a nivel-DÍA que en S1 lleva all-4-en-banda
de ~53% a ~87% (re-apunta carbos/grasas al target diario re-cuantizando), con revalidación pantry
(revert never-worse-than-current), + telemetría band_score del día (antes los updates no tenían
visibilidad de banda).

NO se tocan los 3 closers per-comida del swap (siguen OFF): requieren A/B con macro_sizing_replay.

Cubre:
  - routers.plans._day_exceeds_pantry (revert guard: consumo acumulado vs pantry original)
  - compute_clinical_band_score con targets vía plan.macros/calories (forma de la telemetría)
  - parser-anchored: rebalance cableado en api_regenerate_day (rebalance→revalida→revert) + band_score
    en la respuesta + knob default ON + closers per-comida del swap siguen OFF
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as go
import routers.plans as _rp
from routers.plans import _day_exceeds_pantry

_PLANS = Path(_rp.__file__).resolve()
_BACKEND = Path(go.__file__).resolve().parent


class _FakeDB:
    """macros_from_ingredient_string: '<grams>g <Nombre>' -> {name, grams}."""
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+)\s*g\s+(.*)", str(s))
        if not m:
            return None
        return {"name": m.group(2).strip().title(), "grams": float(m.group(1))}


# ──────────────────────────── revert guard (_day_exceeds_pantry) ────────────────────────────

def test_exceeds_when_day_over_consumes_pantry_item():
    meals = [{"ingredients": ["200g Arroz", "100g Arroz"]}]          # 300g
    ex, why = _day_exceeds_pantry(meals, {"Arroz": 250.0}, _FakeDB())
    assert ex is True and "Arroz" in why


def test_within_tolerance_does_not_exceed():
    meals = [{"ingredients": ["260g Arroz"]}]                        # 260 < 250*1.05+15=277.5
    ex, _ = _day_exceeds_pantry(meals, {"Arroz": 250.0}, _FakeDB())
    assert ex is False


def test_external_ingredient_allowed():
    """Un ingrediente NO presente en el ledger (externo, permitido por _external_tolerance) no cuenta."""
    ex, _ = _day_exceeds_pantry([{"ingredients": ["999g Quinoa"]}], {"Arroz": 250.0}, _FakeDB())
    assert ex is False


def test_empty_or_garbage_failsafe():
    assert _day_exceeds_pantry([], {"Arroz": 100.0}, _FakeDB()) == (False, "")
    assert _day_exceeds_pantry([{"ingredients": ["sin gramaje"]}], {"Arroz": 100.0}, _FakeDB())[0] is False


# ──────────────────────────── band_score (telemetría) ────────────────────────────

def test_band_score_shape_with_day_targets():
    plan = {"days": [{"meals": [{"protein": 40, "carbs": 50, "fats": 15, "cals": 500}]}],
            "macros": {"protein": 40, "carbs": 50, "fats": 15}, "calories": 500}
    bs = go.compute_clinical_band_score(plan, {})
    assert bs.get("score") == 1.0
    assert bs.get("score_macros_only") == 1.0


def test_band_score_detects_out_of_band():
    # entrega proteína al 50% del target → fuera de banda
    plan = {"days": [{"meals": [{"protein": 20, "carbs": 50, "fats": 15, "cals": 500}]}],
            "macros": {"protein": 40, "carbs": 50, "fats": 15}, "calories": 500}
    bs = go.compute_clinical_band_score(plan, {})
    assert bs["per_macro"]["protein"] == 0.0


# ──────────────────────────── parser-anchored (no-regression) ────────────────────────────

def _src(p):
    return Path(p).read_text(encoding="utf-8")


def test_rebalance_wired_in_regen_day():
    src = _src(_PLANS)
    assert "P2-REGEN-DAY-MACRO-REBALANCE" in src
    assert "MEALFIT_REGEN_DAY_MACRO_REBALANCE" in src
    # secuencia rebalance → revalida pantry → revert
    assert "_rebalance_day_macros_to_target(" in src
    assert "_day_exceeds_pantry(" in src
    assert "new_meals[:] = _pre_rb" in src


def test_band_score_in_response():
    src = _src(_PLANS)
    assert "P2-REGEN-DAY-BAND-SCORE" in src
    assert '"band_score": _band_score' in src


def test_rebalance_knob_default_on():
    """El knob del rebalance default ON (revert lo hace seguro)."""
    src = _src(_PLANS)
    assert 'os.environ.get("MEALFIT_REGEN_DAY_MACRO_REBALANCE", "true")' in src


def test_swap_per_meal_closers_still_off():
    """Los 3 closers per-comida del swap NO se flipearon (requieren A/B): siguen default 'false'."""
    agent_src = _src(_BACKEND / "agent.py")
    for knob in ("MEALFIT_UPDATE_MACRO_REBALANCE", "MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER", "MEALFIT_SWAP_TARGET_FROM_SLOT"):
        assert f'os.environ.get("{knob}", "false")' in agent_src, f"{knob} ya no es default false"
