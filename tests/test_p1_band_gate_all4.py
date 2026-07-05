"""[P1-BAND-GATE-ALL4 · 2026-07-01] (audit objetivo v2 · P1-1 · gate de banda hacia "100% all-4")

Gap que cierra: los gates de banda (retry + banner) puntuaban sobre el score AGREGADO kcal-inflado
(la celda kcal ≈1.0 por el reconcile infla ~25%) con umbral 0.5, y el per-macro estaba en 0.34 →
un plan con carbohidratos en banda solo 3/7 días (per-macro 0.43, agregado combinado ~0.86) se
entregaba SIN retry NI banner. El KPI `all4_ratio` se medía pero jamás gateaba.

Fix (reversible por env, patrón P1-OBJECTIVE-LEVERS-ON — vigilar retry-rate post-deploy vía
pipeline_metrics node='clinical_band'):
  - `MEALFIT_BAND_GATE_USE_MACROS_ONLY` flip ON: ambos gates puntúan la precisión REAL de macros.
  - Umbrales RE-TUNEADOS dedicados `*_MACROS_ONLY` (0.45): combined 0.5 ≈ macros-only 0.33 (kcal≈1.0),
    así que 0.45 macros-only ≈ ligeramente más duro que el 0.5 combinado previo, sin mass-retry.
    Rollback exacto: flag=false → score+umbral combinados originales intactos.
  - `MEALFIT_BAND_GATE_PER_MACRO_THRESHOLD` 0.34 → 0.5: cualquier macro con menos de la mitad de sus
    días en banda fuerza retry + banner honesto en agotamiento.
  - `all4_ratio` sigue KPI-only A PROPÓSITO (techo físico ~66.7% por granularidad de porción).
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as go

_GRAPH_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Defaults de knobs
# ---------------------------------------------------------------------------
def test_macros_only_gate_default_on():
    assert go.BAND_GATE_USE_MACROS_ONLY is True, \
        "P1-BAND-GATE-ALL4: los gates deben puntuar macros-only por default"


def test_retuned_macros_only_thresholds():
    # [P2-BAND-THRESHOLDS · 2026-07-02] 0.45 → 0.60 con datos de flota (todo plan entregado
    # no-fallback puntúa macros_only ≥ 0.667 → 0.60 endurece el piso con CERO retries extra).
    # El re-tuneo no actualizó esta expectativa → rojo preexistente en HEAD, re-anclado
    # 2026-07-05 durante el batch P2 del audit solver+seeder.
    assert go.BAND_RETRY_THRESHOLD_MACROS_ONLY == 0.60
    assert go.BAND_SCORE_GATE_THRESHOLD_MACROS_ONLY == 0.60


def test_combined_thresholds_untouched_for_rollback():
    """Rollback MEALFIT_BAND_GATE_USE_MACROS_ONLY=false debe restaurar EXACTAMENTE el gate previo."""
    assert go.BAND_RETRY_THRESHOLD == 0.5
    assert go.BAND_SCORE_GATE_THRESHOLD == 0.5


def test_per_macro_threshold_raised_to_half():
    assert go.BAND_GATE_PER_MACRO_THRESHOLD == 0.5, \
        "per-macro 0.34→0.5: menos de la mitad de los días en banda para un macro = retry"
    assert go.BAND_GATE_PER_MACRO is True


# ---------------------------------------------------------------------------
# 2. Funcional: el escenario literal del audit (carbs 3/7 en banda)
# ---------------------------------------------------------------------------
def _plan_carbs_3_of_7():
    days = []
    for i in range(7):
        carbs = 200 if i < 3 else 120  # 3 días en banda, 4 días al 60% (fuera)
        days.append({"day": i + 1,
                     "meals": [{"protein": 150, "carbs": carbs, "fats": 60, "cals": 2000}]})
    return {"macros": {"protein": "150g", "carbs": "200g", "fats": "60g"},
            "calories": 2000, "days": days}


def test_audit_scenario_now_flagged():
    """Pre-fix: carbs 0.43 > 0.34 y agregado combinado 0.86 > 0.5 → silencio total.
    Post-fix: carbs 0.43 < 0.5 (per-macro) → banner low_band_macro:carbs en agotamiento."""
    out = go.compute_clinical_band_score(_plan_carbs_3_of_7(), {})
    pm_carbs = out["per_macro"]["carbs"]
    assert 0.34 < pm_carbs < 0.5, f"fixture debe caer entre el umbral viejo y el nuevo: {pm_carbs}"
    assert out["score"] > 0.5, "el agregado combinado debe pasar (documenta el hueco pre-fix)"
    plan: dict = {}
    marked = go._maybe_mark_low_band_degraded(
        plan, out["score_macros_only"], False, attempt=3, band_payload=out)
    assert marked is True
    assert plan["_quality_degraded_reason"] == "low_band_macro:carbs"


def test_all4_kpi_still_computed_not_gated():
    out = go.compute_clinical_band_score(_plan_carbs_3_of_7(), {})
    assert out["all4_ratio"] is not None and out["days_total"] == 7
    # KPI-only por techo físico (~66.7%): ningún gate debe comparar all4_ratio contra un umbral.
    gate_region = _GRAPH_SRC[_GRAPH_SRC.find("BAND_RETRY_GATE_ENABLED:"):][:3000]
    assert "all4_ratio" not in gate_region, \
        "all4_ratio debe seguir KPI-only (techo físico de porción cocinable) — no gatear"


# ---------------------------------------------------------------------------
# 3. Funcional: el umbral acompaña al score (param score_threshold del banner)
# ---------------------------------------------------------------------------
def test_banner_honors_explicit_threshold(monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO", False)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.44, False, 1, score_threshold=0.45) is True
    assert plan["_quality_degraded_reason"] == "low_band_score"
    plan2: dict = {}
    assert go._maybe_mark_low_band_degraded(plan2, 0.46, False, 1, score_threshold=0.45) is False
    assert plan2 == {}


def test_banner_falls_back_to_global_threshold(monkeypatch):
    """Compat: sin score_threshold, usa BAND_SCORE_GATE_THRESHOLD (callers/tests previos)."""
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO", False)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.49, False, 1) is True


# ---------------------------------------------------------------------------
# 4. Parser-based: el umbral macros-only está cableado en ambos gates
# ---------------------------------------------------------------------------
def test_retry_gate_pairs_score_with_threshold():
    assert re.search(r"_bsr_thr\s*=\s*BAND_RETRY_THRESHOLD_MACROS_ONLY\s+if\s+_bsr_used_mo", _GRAPH_SRC), \
        "el retry-gate no selecciona el umbral re-tuneado cuando puntúa macros-only"


def test_banner_callsite_pairs_score_with_threshold():
    assert re.search(r"BAND_SCORE_GATE_THRESHOLD_MACROS_ONLY\s+if\s+_band_used_mo", _GRAPH_SRC), \
        "el call site del banner no selecciona el umbral re-tuneado cuando puntúa macros-only"
    assert "score_threshold=_band_gate_thr" in _GRAPH_SRC, \
        "el call site del banner no pasa el umbral efectivo a _maybe_mark_low_band_degraded"


def test_marker_anchor_present():
    assert "P1-BAND-GATE-ALL4" in _GRAPH_SRC
