"""[P1-RENAL-RECHECK-POST-SUBS · 2026-06-18] (audit fresco P1-C) Re-enforza el cap renal tras las subs.

Bug: Guard 1 (cap renal) verificaba `meals_enforced` ANTES de las sustituciones por condición (Guard 3).
Para un perfil renal+dislipidemia, Guard 3 hace swaps que SUBEN proteína ('yogur entero'→'Yogurt griego',
'queso amarillo'→cottage) → el flag `meals_enforced` quedaba stale-True y el fail-hard gate de should_retry
(que lo lee sin re-sumar) entregaba proteína sobre el techo KDIGO presentada como enforced (iatrogénico en ERC).

Fix en DOS capas:
  - Guard 3.6 (PRE-cuantización): re-trima la proteína al cap absoluto sobre los totales POST-subs (las
    porciones trimadas las limpia la cuantización de Guard 4).
  - Guard 4d (POST-cuantización): verifica HONESTAMENTE `meals_enforced` sobre los totales FINALES, porque
    la cuantización (Guard 4/4b/4c) puede re-inflar proteína por redondeo después del trim de Guard 3.6.
    Si un día excede el cap (tol 5%) → meals_enforced=False → el exit-net re-trima y el fail-hard gate escala.

Parser-anchors (posición/gate/delegación) + funcional determinista de Guard 4d (aislado con _db=None).
"""
from __future__ import annotations

from pathlib import Path

import pytest


_GO_PATH = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def src() -> str:
    return _GO_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser-anchors
# ---------------------------------------------------------------------------
def test_marker_present(src):
    assert "P1-RENAL-RECHECK-POST-SUBS" in src
    assert "Guard 3.6" in src
    assert "Guard 4d" in src


def test_guards_ordered(src):
    """Guard 3.6 (trim) entre Guard 3 (subs) y Guard 4 (quantize); Guard 4d (verify) entre Guard 4c y Guard 5."""
    i_subs = src.find("── Guard 3: sustitución de ingredientes por condición")
    i_36 = src.find("── Guard 3.6 (FS6/ERC re-check)")
    i_q = src.find("── Guard 4 (FS2): cuantización de porciones")
    i_4c = src.find("── Guard 4c (P1-MACRO-POSTQUANT-RECONCILE")
    i_4d = src.find("── Guard 4d (FS6/ERC verificación FINAL)")
    i_5 = src.find("── Guard 5 (FS4/FS8)")
    assert -1 not in (i_subs, i_36, i_q, i_4c, i_4d, i_5), "headers de guards no encontrados"
    assert i_subs < i_36 < i_q, "Guard 3.6 (trim) debe ir entre Guard 3 y Guard 4"
    assert i_4c < i_4d < i_5, "Guard 4d (verify) debe ir entre Guard 4c y Guard 5"


def _block(src: str, start_hdr: str, end_hdr: str) -> str:
    s = src.find(start_hdr)
    e = src.find(end_hdr, s)
    assert s != -1 and e != -1 and e > s
    return src[s:e]


def test_guard36_trims_via_validated_enforcement(src):
    """Guard 3.6 re-usa el enforcement validado (no reimplementa): engine o _enforce_renal_per_meal."""
    block = _block(src, "── Guard 3.6 (FS6/ERC re-check)", "── Guard 4 (FS2): cuantización de porciones")
    assert "RENAL_CAP_ENABLED" in block and 'plan["renal_protein_cap"].get("applied")' in block and "_pg > 0" in block
    assert 'enforce_one("renal"' in block
    assert "_enforce_renal_per_meal(" in block


def test_guard4d_is_verify_only(src):
    """Guard 4d re-suma y setea meals_enforced PERO no re-trima (no llama _trim/_enforce — eso de-cuantizaría)."""
    block = _block(src, "── Guard 4d (FS6/ERC verificación FINAL)", "── Guard 5 (FS4/FS8)")
    assert 'plan["renal_protein_cap"]["meals_enforced"]' in block
    assert "_pg * 1.05" in block
    assert "_trim_day_protein_to_ceiling" not in block, "Guard 4d debe ser verify-only (no re-trima)"
    assert "_enforce_renal_per_meal" not in block, "Guard 4d debe ser verify-only (no re-enforza)"


# ---------------------------------------------------------------------------
# Funcional determinista de Guard 4d (aislado: _db=None → Guards 1/3.6/4 se saltan)
# ---------------------------------------------------------------------------
def _force_db_off(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("db off for test")
    # El layer hace `from nutrition_db import IngredientNutritionDB` → parcheamos el origen.
    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", _raise)


def _renal_nutrition():
    return {
        "total_daily_macros": {"protein_str": "100g", "protein_g": 100, "carbs_g": 100, "fats_g": 40},
        "target_calories": 800,
        "macros": {"protein_g": 100, "carbs_g": 100, "fats_g": 40},
    }


def test_guard4d_flags_false_when_final_protein_over_cap(go, monkeypatch):
    """Día con proteína final 135g sobre un cap de 100g (>105 = cap*1.05) → meals_enforced pasa de
    stale-True a False (lo que el fail-hard gate de should_retry necesita ver)."""
    _force_db_off(monkeypatch)
    plan = {
        "renal_protein_cap": {"applied": True, "protein_g": 100, "meals_enforced": True},  # stale True
        "days": [{"meals": [{"name": "A", "protein": "75g", "calories": "400 kcal"},
                            {"name": "B", "protein": "60g", "calories": "400 kcal"}]}],
    }
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male"}, _renal_nutrition())
    assert out["renal_protein_cap"]["meals_enforced"] is False


def test_guard4d_keeps_true_when_within_cap(go, monkeypatch):
    """Día con proteína final 98g bajo el cap (<=105) → meals_enforced queda True."""
    _force_db_off(monkeypatch)
    plan = {
        "renal_protein_cap": {"applied": True, "protein_g": 100, "meals_enforced": True},
        "days": [{"meals": [{"name": "A", "protein": "50g", "calories": "400 kcal"},
                            {"name": "B", "protein": "48g", "calories": "400 kcal"}]}],
    }
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male"}, _renal_nutrition())
    assert out["renal_protein_cap"]["meals_enforced"] is True


def test_guard4d_noop_when_cap_not_applied(go, monkeypatch):
    """Sin cap renal aplicado, Guard 4d no toca nada (no hay renal_protein_cap que verificar)."""
    _force_db_off(monkeypatch)
    plan = {"days": [{"meals": [{"name": "A", "protein": "200g", "calories": "800 kcal"}]}]}
    out = go._apply_deterministic_clinical_layer(plan, {"gender": "male"}, _renal_nutrition())
    assert "renal_protein_cap" not in out or not out.get("renal_protein_cap", {}).get("applied")
