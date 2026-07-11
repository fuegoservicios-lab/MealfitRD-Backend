"""[P1-CRITIQUE-SLOT-PARITY · 2026-07-11] El self-critique usa el MISMO SSOT de horario que
el gate del reviewer — y la banda se re-cierra tras el review-patch.

Evidencia viva:
- Renovación 03:26 (plan 5424440f): intento 1 RECHAZADO por «Tortitas de Queso blanco y
  Avena al Horno» de CENA — el critique usaba `_detect_slot_incoherence` (overlap/merienda)
  pero NO `_detect_slot_appropriateness` (cena-desayuno / arroz-desayuno). Detectores
  asimétricos critique↔reviewer = replan completo evitable (~3 min + $).
- Firma banda post-patch: post-finalize 1.00 → entregado 0.83 (renovación 19:32) / 0.50
  (chunk T2 01:04) — el auto-patch de huérfanos removía ingredientes DESPUÉS del cierre
  de banda y nada re-cerraba.

tooltip-anchor: P1-CRITIQUE-SLOT-PARITY
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Paridad pre-corrección: el SSOT del reviewer alimenta slot_issues
# ---------------------------------------------------------------------------

def test_reviewer_ssot_feeds_critique_slot_issues():
    i = _GO.find("mismo SSOT del gate del reviewer: las violaciones")
    assert i > 0, "la paridad de horario desapareció del critique"
    blk = _GO[i - 400: i + 1200]
    assert "slot_issues = _detect_slot_incoherence(days)" in blk, "vive junto al detector legacy"
    assert "_detect_slot_appropriateness(days)" in blk
    assert "CRITIQUE_SLOT_PARITY_ENABLED" in blk
    assert 'slot_issues.append(_t)' in blk, "las violaciones entran al MISMO canal (slot_block + skip-gate)"


def test_knob_default_on():
    assert 'CRITIQUE_SLOT_PARITY_ENABLED = _env_bool("MEALFIT_CRITIQUE_SLOT_PARITY", True)' in _GO


def test_skip_when_clean_gate_still_guarded_by_slot_issues():
    # el early-exit del critique usa `not slot_issues` — con la paridad dentro de slot_issues,
    # un plan con cena-desayuno YA NO puede saltarse el evaluador+correcciones.
    i = _GO.find("SELF_CRITIQUE_SKIP_WHEN_CLEAN and not staple_repetitions")
    assert i > 0
    assert "not slot_issues" in _GO[i: i + 200]


# ---------------------------------------------------------------------------
# 2. Residual post-corrección → retry quirúrgico (espejo del SAMEDAY parity)
# ---------------------------------------------------------------------------

def test_residual_slot_marks_critique_unresolved():
    i = _GO.find("residual de HORARIO tras corrección")
    assert i > 0, "el residual de horario desapareció"
    blk = _GO[i: i + 1800]
    assert "_detect_slot_appropriateness(days)" in blk
    assert '"slot_appropriateness_unresolved"' in blk
    assert "_mark_critique_unresolved(" in blk
    # corre en el mismo seam que el SAMEDAY (post-corrección)
    i_same = _GO.find("SELF_CRITIQUE_VERIFY_SAME_DAY_PROTEIN:", i)
    assert i_same > 0, "el espejo SAMEDAY debe seguir vivo aguas abajo"


# ---------------------------------------------------------------------------
# 3. Banda re-cerrada tras review-patch, ANTES de re-agregar listas
# ---------------------------------------------------------------------------

def test_band_reclose_after_review_patch_before_reaggregation():
    i = _GO.find("P1-POST-PATCH-BAND-RECLOSE] banda re-cerrada")
    assert i > 0, (
        "el re-cierre de banda post-review-patch desapareció (firma: post-finalize 1.00 → "
        "entregado 0.83/0.50 con fats/kcal caídos)"
    )
    blk_up = _GO[i - 2200: i]
    assert "_review_patch_removed_ingredients" in blk_up, "gateado al caso patch-removió-ingredientes"
    assert "apply_plan_quality_finalize_chain" in blk_up, "chain SSOT idempotente"
    # orden: re-cierre ANTES de la re-agregación de listas
    i_agg = _GO.find("_recompute_aggregates_after_swap(final_state)", i)
    assert i_agg > 0, "la re-agregación GAP-04 debe seguir DESPUÉS del re-cierre"


def test_marker_anchored_in_source():
    assert _GO.count("P1-CRITIQUE-SLOT-PARITY") >= 3
