"""[P1-REGEN-DAY-CLINICAL-PARITY · 2026-06-27] Cierra la brecha de paridad S1↔S2: actualizar un día COMPLETO de un
plan bariátrico (endpoint /regenerate-day) NO corría la maquinaria clínica que sí corre en la generación completa
(assemble_plan_node) → el día regenerado salía con target proteico NO-bariátrico (100g vs 80g) y SIN los caps de
porción (queso/yogurt/fruta/aguacate) ni el re-cierre del piso de proteína (FASE A). Detectado en vivo (corr=1381e458,
band_score=0.25, proteína target 100g, sin logs de BARIATRIC-PORTION/PROTEIN-FLOOR).

Fix en routers/plans.py::api_regenerate_day:
 1. El retarget (P1-REGEN-DAY-RETARGET) pasa medicalConditions/otherConditions/medications a get_nutrition_targets
    → aplica el cap clínico (proteína bariátrica ≤80g).
 2. Tras el rebalance, antes del persist: cap_dm2_high_gi_portions + cap_bariatric_portions (trim, pantry-safe) +
    _repair_protein_floor_post_caps (FASE A) revalidado contra la Nevera (revierte si rompe pantry).

Test parser-based (el endpoint necesita DB+auth; anclamos el contrato en el source de prod). tooltip-anchor:
P1-REGEN-DAY-CLINICAL-PARITY
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _regen_day_body() -> str:
    """El cuerpo de api_regenerate_day (hasta el siguiente `def ` de nivel módulo)."""
    i = _PLANS.index("def api_regenerate_day")
    nxt = re.search(r"\n(def |@router\.)", _PLANS[i + 10:])
    return _PLANS[i: i + 10 + (nxt.start() if nxt else len(_PLANS))]


def test_marker_and_knob_present():
    body = _regen_day_body()
    assert "P1-REGEN-DAY-CLINICAL-PARITY" in body
    assert "MEALFIT_REGEN_DAY_CLINICAL_PARITY" in body


def test_retarget_passes_conditions_to_nutrition_targets():
    """El retarget debe pasar medicalConditions a _bio → get_nutrition_targets aplica el cap bariátrico (80g)."""
    body = _regen_day_body()
    # el loop que vuelca condiciones/alergias/fármacos en _bio
    assert '"medicalConditions"' in body and "_bio[_ck]" in body
    assert "otherConditions" in body


def test_applies_dm2_and_bariatric_caps():
    body = _regen_day_body()
    assert "cap_dm2_high_gi_portions" in body
    assert "cap_bariatric_portions" in body


def test_applies_fase_a_protein_floor():
    body = _regen_day_body()
    assert "_repair_protein_floor_post_caps" in body


def test_fase_a_is_pantry_guarded():
    """FASE A AÑADE proteína → DEBE revalidarse contra la Nevera y revertir si rompe (never-worse)."""
    body = _regen_day_body()
    # el patrón: deepcopy pre-FASE-A, _day_exceeds_pantry, y revert new_meals[:] = _pre_pf
    assert "_pre_pf" in body and "deepcopy" in body
    assert "_day_exceeds_pantry" in body
    assert "new_meals[:] = _pre_pf" in body


def test_parity_runs_after_rebalance_before_persist():
    """Orden correcto (espejo de S1: motor → caps → FASE A → persist): el bloque de paridad va DESPUÉS del
    rebalance y ANTES del _day_mutator/persist."""
    body = _regen_day_body()
    i_reb = body.index("P2-REGEN-DAY-MACRO-REBALANCE")
    i_parity = body.index("P1-REGEN-DAY-CLINICAL-PARITY", body.index("if (\n") if "if (\n" in body else 0)
    i_parity = body.index("cap_bariatric_portions")
    i_persist = body.index("def _day_mutator")
    assert i_reb < i_parity < i_persist, "paridad clínica debe ir tras rebalance y antes del persist"
