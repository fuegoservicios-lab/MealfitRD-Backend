"""[P1-MICRO-DEGRADED-STALE-CLEAR · 2026-07-12] El banner `micro_worst_day_ceiling` se limpia
cuando los swaps del usuario YA resolvieron el día que se pasaba del techo.

Caso vivo (plan 1bfda745, madrugada 2026-07-12): banner "Un día se pasa del techo de sodio o
azúcar añadida" marcado al GENERAR — pero tras los swaps del owner el reporte fresco decía
`per_day_ceilings: {flagged: false, days_above: 0}` y el banner seguía ahí (stale). Espejo
CLEAR-ONLY de P1-BAND-DEGRADED-STALE-CLEAR, aplicado dentro de
`recompute_micronutrient_report_for_plan` (cubre swap-persist / chat-modify / regen-day, las
3 superficies que recomputan el panel).

Contrato:
1. reason `micro_worst_day_ceiling` + reporte recomputado con flagged=false/days_above=0 →
   flags `_quality_degraded*` limpiados.
2. Razones NO-micro (clínicas/banda/fallback) jamás se tocan aquí.
3. Techo aún violado (days_above>0) → banner intacto (el clear jamás miente).
4. CLEAR-ONLY: la función jamás MARCA degradación.

tooltip-anchor: P1-MICRO-DEGRADED-STALE-CLEAR
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def _run_recompute(plan, monkeypatch_report):
    """Ejecuta recompute_micronutrient_report_for_plan con el builder de reporte mockeado
    (el clear opera sobre el reporte RECIÉN recomputado)."""
    import graph_orchestrator as go
    import micronutrients as mn

    _orig = mn.build_micronutrient_report
    mn.build_micronutrient_report = lambda *a, **k: monkeypatch_report
    try:
        return go.recompute_micronutrient_report_for_plan(
            plan, {"age": 30, "gender": "male", "weight": 70})
    finally:
        mn.build_micronutrient_report = _orig


def _plan(reason="micro_worst_day_ceiling:sodium"):
    return {
        "days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Moro",
                                       "ingredients": ["100 g de arroz"],
                                       "protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}],
        "_quality_degraded": True,
        "_quality_degraded_reason": reason,
        "_quality_degraded_severity": "minor",
    }


def test_cleared_when_ceiling_resolved():
    plan = _plan()
    ok = _run_recompute(plan, {"coverage": 0.9, "gaps": [],
                               "per_day_ceilings": {"flagged": False, "days_above": 0,
                                                    "days_evaluated": 3}})
    assert ok is True
    assert "_quality_degraded" not in plan and "_quality_degraded_reason" not in plan, (
        "el usuario ya resolvió el día del techo vía swaps → el banner stale debe limpiarse "
        "(vivo: plan 1bfda745 con days_above=0 y banner de sodio pegado)"
    )


def test_kept_when_ceiling_still_violated():
    plan = _plan()
    _run_recompute(plan, {"coverage": 0.9, "gaps": [],
                          "per_day_ceilings": {"flagged": True, "days_above": 1,
                                               "days_evaluated": 3}})
    assert plan.get("_quality_degraded") is True, "techo aún violado → el banner NO miente"


def test_non_micro_reasons_untouched():
    plan = _plan(reason="clinical_layer_incomplete")
    _run_recompute(plan, {"coverage": 0.9, "gaps": [],
                          "per_day_ceilings": {"flagged": False, "days_above": 0}})
    assert plan.get("_quality_degraded") is True, (
        "razones clínicas/banda/fallback jamás se limpian desde el recompute de micros"
    )


def test_clear_only_never_marks():
    plan = _plan()
    plan.pop("_quality_degraded")
    plan.pop("_quality_degraded_reason")
    plan.pop("_quality_degraded_severity")
    _run_recompute(plan, {"coverage": 0.9, "gaps": [],
                          "per_day_ceilings": {"flagged": True, "days_above": 2}})
    assert "_quality_degraded" not in plan, "CLEAR-ONLY: jamás marca degradación nueva"


def test_knob_and_marker():
    import graph_orchestrator as go
    assert go.MICRO_DEGRADED_STALE_CLEAR_ENABLED is True
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P1-MICRO-DEGRADED-STALE-CLEAR") >= 3
