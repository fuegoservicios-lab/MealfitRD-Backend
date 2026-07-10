"""[P1-BAND-PER-MACRO-ON + P1-ALL4-KPI · 2026-07-01] (audit macros GAP-1/GAP-7)

Con el band gate solo-AGREGADO (umbral 0.5), un macro entero podía fallar 7/7 días (carbs al 120%
todos los días → score 21/28 = 0.75) y el plan se entregaba SIN retry NI banner — el hueco
dominante contra el objetivo "100% all-4".

Fix: (1) flip `MEALFIT_BAND_GATE_PER_MACRO` ON (el retry-gate per-macro ya existía dark);
(2) `_maybe_mark_low_band_degraded` acepta `band_payload` → banner reason=low_band_macro:<k>
cuando el agregado pasa pero un macro quedó bajo el umbral (honestidad en agotamiento);
(3) KPI conjunto `all4_days/days_total/all4_ratio` en `compute_clinical_band_score`.
"""
from __future__ import annotations

import graph_orchestrator as go


def test_per_macro_gate_default_on():
    assert go.BAND_GATE_PER_MACRO is True, "P1-BAND-PER-MACRO-ON: el gate per-macro debe nacer ON"
    assert 0.0 <= go.BAND_GATE_PER_MACRO_THRESHOLD <= 1.0


# ---------------------------------------------------------------------------
# banner per-macro en agotamiento
# ---------------------------------------------------------------------------
def _gate_on(monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO", True)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO_THRESHOLD", 0.34)


def test_aggregate_pass_but_macro_dead_marks_banner(monkeypatch):
    """carbs 0/7 con agregado 0.75 → antes: silencio; ahora: banner low_band_macro:carbs."""
    _gate_on(monkeypatch)
    plan: dict = {}
    payload = {"per_macro": {"protein": 1.0, "carbs": 0.0, "fats": 1.0, "kcal": 1.0}}
    marked = go._maybe_mark_low_band_degraded(plan, 0.75, False, attempt=3, band_payload=payload)
    assert marked is True
    assert plan["_quality_degraded"] is True
    assert plan["_quality_degraded_reason"] == "low_band_macro:carbs"
    assert plan["_quality_degraded_band_per_macro_low"] == ["carbs"]


def test_aggregate_low_keeps_classic_reason(monkeypatch):
    _gate_on(monkeypatch)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.1, "carbs": 0.1, "fats": 0.1, "kcal": 1.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.30, False, attempt=2, band_payload=payload) is True
    assert plan["_quality_degraded_reason"] == "low_band_score"


def test_all_macros_healthy_not_marked(monkeypatch):
    _gate_on(monkeypatch)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.9, "carbs": 0.8, "fats": 0.7, "kcal": 1.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.85, False, attempt=1, band_payload=payload) is False
    assert plan == {}


def test_kcal_triggers_per_macro_via_backstop(monkeypatch):
    # [P2-BAND-GATE-KCAL-SEMANTICS · 2026-07-10] Semántica actualizada deliberadamente: kcal ya NO
    # está excluido incondicionalmente — usa el backstop ya establecido en el retry-gate
    # (BAND_GATE_KCAL_BACKSTOP/THRESHOLD). Ver test_p2_2_band_gate_kcal_semantics.py para la cobertura
    # completa (incluye el rollback MEALFIT_BAND_GATE_KCAL_BACKSTOP=false).
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_BACKSTOP", True)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_THRESHOLD", 0.5)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.9, "carbs": 0.8, "fats": 0.7, "kcal": 0.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.7, False, attempt=1, band_payload=payload) is True
    assert plan["_quality_degraded_reason"] == "low_band_macro:kcal"


def test_backwards_compatible_without_payload(monkeypatch):
    _gate_on(monkeypatch)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.30, False, 1) is True
    assert plan["_quality_degraded_reason"] == "low_band_score"


def test_fallback_never_marked(monkeypatch):
    _gate_on(monkeypatch)
    plan: dict = {}
    payload = {"per_macro": {"carbs": 0.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.9, True, 1, band_payload=payload) is False


def test_per_macro_knob_off_disables_banner_branch(monkeypatch):
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO", False)
    plan: dict = {}
    payload = {"per_macro": {"carbs": 0.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.75, False, 1, band_payload=payload) is False


# ---------------------------------------------------------------------------
# KPI all-4 por día
# ---------------------------------------------------------------------------
def _plan(day_deliveries):
    days = []
    for i, d in enumerate(day_deliveries):
        days.append({"day": i + 1, "meals": [{"protein": d[0], "carbs": d[1], "fats": d[2], "cals": d[3]}]})
    return {"macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}, "calories": 2000, "days": days}


def test_all4_kpi_counts_joint_days():
    # día 1: las 4 en banda; día 2: carbs al 60% (fuera) → all4 = 1/2.
    plan = _plan([(150, 200, 60, 2000), (150, 120, 60, 2000)])
    out = go.compute_clinical_band_score(plan, {})
    assert out["days_total"] == 2
    assert out["all4_days"] == 1
    assert out["all4_ratio"] == 0.5


def test_all4_kpi_perfect_plan():
    plan = _plan([(150, 200, 60, 2000)] * 3)
    out = go.compute_clinical_band_score(plan, {})
    assert out["all4_days"] == 3 and out["all4_ratio"] == 1.0
