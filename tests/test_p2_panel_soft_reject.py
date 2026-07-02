"""[P2-PANEL-SOFT-REJECT · 2026-06-15] Soft-reject observable sobre el panel de micros (cubre 4 P2).

`_maybe_mark_panel_degraded` marca _quality_degraded (banner existente, NO hard-gate → cero loop) cuando:
  - P2-4/P2-13: una CONDICIÓN declarada tiene su target cuantitativo fuera de banda (satfat/K/Mg/fibra)
  - P2-5: un micro ALCANZABLE (fibra/K/Mg/Ca) bajo el piso DRI (vit D/hierro/B12 EXCLUIDOS → inalcanzables)
  - P2-8: sodio/azúcar sobre el techo WHO + cobertura del catálogo alta (anti falso-positivo sal 'al gusto')
Cada sub-check tras SU knob (default OFF). Validación determinista (sin LLM/créditos).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _plan(gaps, coverage=0.85):
    return {"micronutrient_report": {"gaps": list(gaps), "coverage": coverage}}


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Defaults OFF + no-op
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_knob_defaults(go):
    # [P2-SATFAT-CEILING-OBSERVABLE] el sub-check de condición fue flippeado ON (honestidad accionable del
    # build todo-terreno); micros-soft-reject sigue OFF (A/B-pending).
    # [P1-SODIUM-SUGAR-EXCESS-ON · 2026-07-02] sodio/azúcar flippeado ON para TODOS (con umbral
    # anti-ruido MIN_RATIO=1.25 — solo excesos materiales marcan). El gate de retry sigue OFF.
    assert go.CONDITION_PANEL_DEGRADE_ENABLED is True
    assert go.MICRONUTRIENT_SOFT_REJECT_ENABLED is False
    assert go.SODIUM_SUGAR_DEGRADE_ENABLED is True
    assert go.SODIUM_EXCESS_GATE_ENABLED is False
    assert go.MICRO_PERDAY_DEGRADE_ENABLED is True


def test_off_does_not_mark(go, monkeypatch):
    # comportamiento con el sub-check apagado (rollback path MEALFIT_CONDITION_PANEL_DEGRADE=false)
    monkeypatch.setattr(go, "CONDITION_PANEL_DEGRADE_ENABLED", False)
    plan = _plan([{"key": "saturated_fat_g", "status": "alto", "valor": 40, "techo": 15}])
    assert go._maybe_mark_panel_degraded(plan, {"medicalConditions": ["Colesterol alto"]}, False, 1) is False
    assert "_quality_degraded" not in plan


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-4/P2-13: condición declarada + target fuera de banda
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_dyslipidemia_satfat_over_ceiling_marks(go, monkeypatch):
    monkeypatch.setattr(go, "CONDITION_PANEL_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "saturated_fat_g", "status": "alto", "valor": 40, "techo": 15}])
    assert go._maybe_mark_panel_degraded(plan, {"medicalConditions": ["Colesterol alto"]}, False, 1) is True
    assert plan["_quality_degraded_reason"] == "condition_panel_gap"


def test_satfat_high_but_no_dyslipidemia_not_marked(go, monkeypatch):
    """Anti-falso-positivo: satfat alto pero SIN dislipidemia declarada → no degrada."""
    monkeypatch.setattr(go, "CONDITION_PANEL_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "saturated_fat_g", "status": "alto", "valor": 40, "techo": 15}])
    assert go._maybe_mark_panel_degraded(plan, {"medicalConditions": ["Ninguna"]}, False, 1) is False


def test_margin_avoids_marginal_overage(go, monkeypatch):
    """Un satfat apenas sobre el techo (dentro del margen 15%) NO degrada (anti-ruido de redondeo)."""
    monkeypatch.setattr(go, "CONDITION_PANEL_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "saturated_fat_g", "status": "alto", "valor": 21, "techo": 20}])  # +5% < margen
    assert go._maybe_mark_panel_degraded(plan, {"medicalConditions": ["Colesterol alto"]}, False, 1) is False


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-5: micros alcanzables vs inalcanzables
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_low_reachable_micro_marks(go, monkeypatch):
    monkeypatch.setattr(go, "MICRONUTRIENT_SOFT_REJECT_ENABLED", True)
    plan = _plan([{"key": "fiber_g", "status": "bajo", "valor": 10, "piso": 28}])
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is True
    assert plan["_quality_degraded_reason"] == "low_micros"


def test_unreachable_micro_iron_not_marked(go, monkeypatch):
    """Hierro/vit D son inalcanzables con dieta → NUNCA degradan (regla de oro, fuera del whitelist)."""
    monkeypatch.setattr(go, "MICRONUTRIENT_SOFT_REJECT_ENABLED", True)
    plan = _plan([{"key": "iron_mg", "status": "bajo", "valor": 8, "piso": 18}])
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False


def test_estimado_bajo_not_marked(go, monkeypatch):
    """status 'estimado_bajo' (cobertura parcial, incierto) NO degrada."""
    monkeypatch.setattr(go, "MICRONUTRIENT_SOFT_REJECT_ENABLED", True)
    plan = _plan([{"key": "potassium_mg", "status": "estimado_bajo", "valor": 2000, "piso": 4700}])
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-8: sodio/azúcar + gate de cobertura
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_high_sodium_high_coverage_marks(go, monkeypatch):
    monkeypatch.setattr(go, "SODIUM_SUGAR_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "sodium_mg", "status": "alto", "valor": 3000, "techo": 2000}], coverage=0.85)
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is True
    assert plan["_quality_degraded_reason"] == "high_sodium_sugar"


def test_high_sodium_low_coverage_not_marked(go, monkeypatch):
    """Cobertura baja → sal 'al gusto' no medible → NO degrada (anti-falso-positivo)."""
    monkeypatch.setattr(go, "SODIUM_SUGAR_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "sodium_mg", "status": "alto", "valor": 3000, "techo": 2000}], coverage=0.5)
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False


def test_marginal_sodium_excess_not_marked(go, monkeypatch):
    """[P1-SODIUM-SUGAR-EXCESS-ON] Exceso MARGINAL (techo×1.05 < MIN_RATIO 1.25) NO marca —
    lección P1-COHERENCE-BANNER-NOISE: surface solo lo accionable."""
    monkeypatch.setattr(go, "SODIUM_SUGAR_DEGRADE_ENABLED", True)
    plan = _plan([{"key": "sodium_mg", "status": "alto", "valor": 2100, "techo": 2000}], coverage=0.85)
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False


def test_micro_worst_day_marks(go, monkeypatch):
    """[P1-MICRO-PERDAY-FLOOR] per_day_floors.flagged=True (peor día con ≥2 micros bajo el ratio)
    → banner micro_worst_day con el detalle del día."""
    monkeypatch.setattr(go, "MICRO_PERDAY_DEGRADE_ENABLED", True)
    plan = {"micronutrient_report": {
        "gaps": [], "coverage": 0.9,
        "per_day_floors": {"flagged": True, "days_below": 1,
                           "worst_day": {"day_index": 2, "low": ["iron_mg", "calcium_mg"]}},
    }}
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is True
    assert plan["_quality_degraded_reason"] == "micro_worst_day"
    assert "día 3" in plan["_quality_degraded_panel_detail"]


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Precedencia + fallback + anchor
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_does_not_override_existing_reason(go, monkeypatch):
    monkeypatch.setattr(go, "MICRONUTRIENT_SOFT_REJECT_ENABLED", True)
    plan = _plan([{"key": "fiber_g", "status": "bajo", "valor": 10, "piso": 28}])
    plan["_quality_degraded"] = True
    plan["_quality_degraded_reason"] = "max_attempts"
    assert go._maybe_mark_panel_degraded(plan, {}, False, 1) is False
    assert plan["_quality_degraded_reason"] == "max_attempts"


def test_fallback_not_marked(go, monkeypatch):
    monkeypatch.setattr(go, "MICRONUTRIENT_SOFT_REJECT_ENABLED", True)
    plan = _plan([{"key": "fiber_g", "status": "bajo", "valor": 10, "piso": 28}])
    assert go._maybe_mark_panel_degraded(plan, {}, True, 1) is False


def test_marker_present(go):
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-PANEL-SOFT-REJECT" in src
    assert "def _maybe_mark_panel_degraded(" in src
