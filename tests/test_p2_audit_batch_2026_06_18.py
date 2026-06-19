"""[P2 audit fresco · 2026-06-18] Batch backend de P2 (clínico + precisión/knobs + seguridad-DiD + obs).

Cubre:
  - P2-DRI-AGE-AWARE-VITD: piso DRI de vit D age-aware (15 → 20 mcg a 71+).
  - P2-MIN-KCAL-KNOB-CLAMP: validator de rango en los knobs del piso de kcal.
  - P2-GENDER-ENUM-WARN: warning cuando gender no es reconocido (cae a la ecuación femenina).
  - P2-PROTEIN-CEILING-ADJ-WEIGHT: techo de proteína sobre peso ajustado en obesidad (>30% grasa).
  - P2-SOLVER-KNOBS-REGISTRY: solver/resolver/protein-ceiling knobs auto-registrados en _KNOBS_REGISTRY.
  - P2-CARB-TRIM-RAW-LOCKSTEP: el carb-trim reescala ingredients_raw por el factor efectivo (factor*_f).
  - P2-REGEN-CHUNK-USER-SCOPE: UPDATE plan_chunk_queue de /regen-degraded scoped por user_id.
  - P2-DIET-SCAN-FAIL-SECURE: el except del diet-scan escala a crítico (no fail-open).
  - P2-P1-5-BAND-METRIC: re-emit de la métrica band sobre el fallback de emergencia P1-5.

Determinista (sin LLM/DB/créditos): funcional donde es limpio + parser-anchors para invariantes estructurales.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


_BE = Path(__file__).resolve().parent.parent
_NC = _BE / "nutrition_calculator.py"
_GO = _BE / "graph_orchestrator.py"
_PS = _BE / "portion_solver.py"
_NDB = _BE / "nutrition_db.py"
_PLANS = _BE / "routers" / "plans.py"


# --------------------------------------------------------------------------- vit D age-aware
def test_vit_d_floor_age_aware():
    from micronutrients import dri_targets
    assert dri_targets("F", 40)["vit_d_mcg"]["floor"] == 15.0
    assert dri_targets("M", 70)["vit_d_mcg"]["floor"] == 15.0
    assert dri_targets("F", 75)["vit_d_mcg"]["floor"] == 20.0   # 71+ → 800 UI
    assert dri_targets("M", 80)["vit_d_mcg"]["floor"] == 20.0
    assert dri_targets("F", None)["vit_d_mcg"]["floor"] == 15.0  # edad desconocida → adulto joven


# --------------------------------------------------------------------------- min-kcal knob clamp
def test_min_kcal_validator_rejects_out_of_range():
    import nutrition_calculator as nc
    assert nc._MIN_KCAL_RANGE(1200) is True
    assert nc._MIN_KCAL_RANGE(1500) is True
    assert nc._MIN_KCAL_RANGE(0) is False        # un knob=0 desactivaría el piso → rechazado
    assert nc._MIN_KCAL_RANGE(120) is False       # typo
    assert nc._MIN_KCAL_RANGE(5000) is False


def test_min_kcal_anchor():
    src = _NC.read_text(encoding="utf-8")
    assert "P2-MIN-KCAL-KNOB-CLAMP" in src
    assert "validator=_MIN_KCAL_RANGE" in src


# --------------------------------------------------------------------------- gender enum warning
def test_gender_unknown_warns(caplog):
    import nutrition_calculator as nc
    with caplog.at_level(logging.WARNING):
        nc.calculate_bmr(70, 170, 30, "nonbinary")
    assert "P2-GENDER-ENUM-WARN" in caplog.text


def test_gender_female_does_not_warn(caplog):
    import nutrition_calculator as nc
    with caplog.at_level(logging.WARNING):
        nc.calculate_bmr(70, 170, 30, "female")
    assert "P2-GENDER-ENUM-WARN" not in caplog.text


def test_gender_none_is_safe(caplog):
    import nutrition_calculator as nc
    # No debe lanzar AttributeError (None-safe) y debe warnear (desconocido).
    with caplog.at_level(logging.WARNING):
        bmr = nc.calculate_bmr(70, 170, 30, None)
    assert isinstance(bmr, int)
    assert "P2-GENDER-ENUM-WARN" in caplog.text


# --------------------------------------------------------------------------- protein ceiling adjusted weight
def test_protein_ceiling_uses_adjusted_weight_in_obesity():
    import nutrition_calculator as nc
    # Target alto para que el split de proteína supere ambos techos.
    full = nc.calculate_macros(5000, "gain_muscle", weight_kg=150, body_fat_pct=None)
    adj = nc.calculate_macros(5000, "gain_muscle", weight_kg=150, body_fat_pct=45)
    assert adj["protein_g"] < full["protein_g"], "obesidad debe capear sobre peso ajustado (menor)"
    # Peso ajustado ≈ LBM + 0.25*(peso-LBM) = 82.5 + 16.875 = 99.375 → techo 2.2*99.4 ≈ 219 g
    assert 200 <= adj["protein_g"] <= 235


def test_protein_ceiling_no_bodyfat_uses_total_weight():
    import nutrition_calculator as nc
    # Sin bodyFat (o <=30%) → comportamiento previo (peso total). Techo 2.2*150 = 330.
    low_bf = nc.calculate_macros(5000, "gain_muscle", weight_kg=150, body_fat_pct=20)
    full = nc.calculate_macros(5000, "gain_muscle", weight_kg=150, body_fat_pct=None)
    assert low_bf["protein_g"] == full["protein_g"]


# --------------------------------------------------------------------------- knobs registry (#19)
def test_solver_knobs_in_registry():
    import portion_solver  # noqa: F401 (import puebla el registry)
    from knobs import _KNOBS_REGISTRY
    for k in ("MEALFIT_SOLVER_LSQ", "MEALFIT_SOLVER_W_KCAL", "MEALFIT_SOLVER_W_PROTEIN",
              "MEALFIT_SOLVER_W_CARBS", "MEALFIT_SOLVER_W_FATS", "MEALFIT_SOLVER_LSQ_REG"):
        assert k in _KNOBS_REGISTRY, f"{k} no auto-registrado (invisible en /health/version)"


def test_resolver_and_ceiling_knobs_in_registry():
    import nutrition_db  # noqa: F401
    import nutrition_calculator  # noqa: F401
    from knobs import _KNOBS_REGISTRY
    assert "MEALFIT_NUTRITION_UNIFIED_RESOLVER" in _KNOBS_REGISTRY
    assert "MEALFIT_PROTEIN_CEILING_G_PER_KG" in _KNOBS_REGISTRY


def test_protein_ceiling_clamp_preserved():
    import nutrition_calculator as nc
    # El clamp [1.6, 3.0] se preserva tras migrar a registry.
    assert 1.6 <= nc._protein_ceiling_g_per_kg() <= 3.0


# --------------------------------------------------------------------------- carb-trim raw lockstep (#9)
def test_carb_trim_raw_lockstep_anchor():
    src = _GO.read_text(encoding="utf-8")
    assert "P2-CARB-TRIM-RAW-LOCKSTEP" in src
    # El raw debe reescalarse por el factor efectivo (factor * _f), no solo por factor.
    assert "_resc(str(raw[idx]), factor * _f)" in src


# --------------------------------------------------------------------------- regen chunk user scope (#16)
def _func_body(src: str, name: str) -> str:
    import re
    m = re.search(rf"\ndef {re.escape(name)}\(", src)
    assert m is not None, f"{name} no encontrado"
    start = m.start()
    nxt = re.search(r"\n(?:@router\.|def |class )", src[start + 1:])
    return src[start: (start + 1 + nxt.start()) if nxt else len(src)]


def test_regen_degraded_chunk_update_scoped_by_user():
    src = _PLANS.read_text(encoding="utf-8")
    body = _func_body(src, "api_regen_degraded_chunks")
    assert "P2-REGEN-CHUNK-USER-SCOPE" in body
    # El UPDATE plan_chunk_queue debe atar user_id (defense-in-depth).
    assert "meal_plan_id IN (SELECT id FROM meal_plans WHERE user_id = %s)" in body


# --------------------------------------------------------------------------- diet-scan fail-secure (#1)
def test_diet_scan_except_is_fail_secure():
    src = _GO.read_text(encoding="utf-8")
    assert "P2-DIET-SCAN-FAIL-SECURE" in src
    # El bloque except debe escalar a crítico (no solo loggear).
    i = src.find("P2-DIET-SCAN-FAIL-SECURE")
    block = src[i - 600:i + 600]
    assert "_had_diet_critical = True" in block
    assert "approved = False" in block
    assert 'severity = _severity_max(severity, "critical")' in block


# --------------------------------------------------------------------------- P1-5 band metric (#15)
def test_p1_5_reemits_band_metric():
    src = _GO.read_text(encoding="utf-8")
    assert "P2-P1-5-BAND-METRIC" in src
    i = src.find("P2-P1-5-BAND-METRIC")
    block = src[i:i + 700]
    assert 'final_state["plan_result"] = plan_to_return' in block
    assert "_compute_pipeline_holistic_score_and_emit(" in block
