"""[P1-MICRO-PERDAY-FLOOR + P1-SODIUM-SUGAR-EXCESS-ON · 2026-07-02] (batch P1-OBJECTIVE-V4-BATCH)

Bloque B del audit v4 (micros):
- PERDAY: el panel evaluaba SOLO el promedio del plan (v/num_days) — un plan podía "cumplir"
  en promedio con días individuales deficitarios (asimetría vs macros per-día×celda). Ahora
  `compute_plan_micronutrient_totals` acumula per-día y el reporte trae `per_day_floors`
  (resumen worst-day compacto, cobertura-cierta, exclusiones clínicas) que
  `_maybe_mark_panel_degraded` consume como banner `micro_worst_day`.
- SODIUM/SUGAR: el chequeo de EXCESO (techos WHO) estaba OFF por default — para el usuario sin
  condición el exceso ni marcaba banner. Flip ON con umbral anti-ruido MIN_RATIO=1.25 (solo
  excesos materiales) + gate de retry para exceso flagrante (default OFF, flip con datos).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_MN_SRC = (_BACKEND / "micronutrients.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mn():
    import micronutrients as _mn
    return _mn


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _FakeDB:
    """micros_from_ingredient_string determinista por nombre (sin catálogo/Neon)."""
    RICH = {"fiber": 12.0, "sodium_mg": 300.0, "vit_d_mcg": 2.0, "calcium_mg": 500.0,
            "iron_mg": 8.0, "b12_mcg": 1.5, "potassium_mg": 1800.0, "magnesium_mg": 180.0,
            "saturated_fat_g": 2.0, "zinc_mg": 5.0, "folate_mcg": 220.0, "vit_a_mcg": 450.0,
            "vit_c_mg": 60.0, "vit_e_mg": 7.0, "vit_k_mcg": 60.0, "selenium_mcg": 30.0,
            "omega3_g": 0.9, "sugars_g": 0.0}
    POOR = {k: 0.1 for k in RICH}

    def micros_from_ingredient_string(self, s):
        return dict(self.POOR) if "pobre" in str(s).lower() else dict(self.RICH)


def _plan(day1_ings, day2_ings):
    return {"days": [
        {"day": 1, "meals": [{"name": "A", "ingredients_raw": day1_ings}]},
        {"day": 2, "meals": [{"name": "B", "ingredients_raw": day2_ings}]},
    ]}


# ════════════════════════════════════════════════════════════════════════════
# P1-MICRO-PERDAY-FLOOR
# ════════════════════════════════════════════════════════════════════════════
def test_totals_include_per_day_snapshots(mn):
    plan = _plan(["200g de pollo rico", "100g de espinaca rica"], ["100g de arroz pobre"])
    totals = mn.compute_plan_micronutrient_totals(plan, _FakeDB())
    assert len(totals["per_day"]) == 2
    d1, d2 = totals["per_day"]
    assert d1["calcium_mg"] > d2["calcium_mg"], "el snapshot per-día debe distinguir días"
    # El promedio sigue siendo la media de los días (contrato previo intacto).
    assert totals["daily"]["calcium_mg"] == pytest.approx((d1["calcium_mg"] + d2["calcium_mg"]) / 2, abs=0.2)


def test_report_flags_worst_day(mn, monkeypatch):
    monkeypatch.setenv("MEALFIT_MICRO_PERDAY_RATIO", "0.6")
    monkeypatch.setenv("MEALFIT_MICRO_PERDAY_MIN_MICROS", "2")
    plan = _plan(["300g de pollo rico", "200g de espinaca rica", "100g de yogurt rico"],
                 ["100g de arroz pobre"])
    report = mn.build_micronutrient_report(plan, _FakeDB(), sex="M", age=30)
    pdf = report.get("per_day_floors")
    assert isinstance(pdf, dict), "el reporte debe traer per_day_floors"
    assert pdf["days_evaluated"] == 2
    assert pdf["worst_day"]["day_index"] == 1, "el día pobre debe ser el peor"
    assert pdf["flagged"] is True and len(pdf["worst_day"]["low"]) >= 2
    # Exclusiones de la regla de oro: vit D / B12 / vit K jamás acusan el día.
    for k in ("vit_d_mcg", "b12_mcg", "vit_k_mcg"):
        assert k not in pdf["worst_day"]["low"]


def test_report_no_flag_when_days_uniform(mn):
    plan = _plan(["300g de pollo rico", "200g de espinaca rica"],
                 ["300g de res rica", "200g de batata rica"])
    report = mn.build_micronutrient_report(plan, _FakeDB(), sex="M", age=30)
    pdf = report.get("per_day_floors")
    assert pdf is not None and pdf["flagged"] is False


# ════════════════════════════════════════════════════════════════════════════
# P1-SODIUM-SUGAR-EXCESS-ON — knobs + gate de review
# ════════════════════════════════════════════════════════════════════════════
def test_knob_defaults(go):
    assert go.SODIUM_SUGAR_DEGRADE_ENABLED is True, "banner de exceso ON para todos"
    assert go.SODIUM_SUGAR_DEGRADE_MIN_RATIO == pytest.approx(1.25)
    assert go.SODIUM_EXCESS_GATE_ENABLED is False, "gate de retry nace OFF (flip con datos)"
    assert go.SODIUM_EXCESS_GATE_RATIO == pytest.approx(1.5)
    assert go.MICRO_PERDAY_DEGRADE_ENABLED is True


def test_review_gate_wired_with_final_advisory():
    """El gate de sodio vive en review con el molde slot/dish-quality: rechazo + advisory final."""
    i = _GO_SRC.index("if SODIUM_EXCESS_GATE_ENABLED:")
    window = _GO_SRC[i:i + 3000]
    assert "_sodium_excess_advisory_final" in window
    assert "SODIO EXCESIVO" in window
    assert 'severity = _severity_max(severity, "high")' in window


def test_perday_anchor_in_micronutrients():
    assert "P1-MICRO-PERDAY-FLOOR" in _MN_SRC
    assert '"per_day_floors": per_day_floors' in _MN_SRC
    assert "MEALFIT_MICRO_PERDAY_RATIO" in _MN_SRC


def test_perday_subcheck_in_panel_degraded():
    i = _GO_SRC.index("def _maybe_mark_panel_degraded(")
    body = _GO_SRC[i:_GO_SRC.index("def _maybe_mark_clinical_layer_incomplete_degraded", i)]
    assert "MICRO_PERDAY_DEGRADE_ENABLED" in body
    assert '"micro_worst_day"' in body
    assert "SODIUM_SUGAR_DEGRADE_MIN_RATIO" in body, "el sub-check de sodio debe usar el umbral anti-ruido"
