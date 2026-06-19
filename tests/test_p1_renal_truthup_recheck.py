"""[P1-RENAL-TRUTHUP-RECHECK · 2026-06-19] (audit fresco P1-3) Re-verifica el cap renal tras Guard 8z.

Bug (audit 2026-06-19): Guard 8z (MACRO-TRUTHUP) reescribe meal['protein'] re-sumando los strings FINALES
de ingredientes — corre DESPUÉS de la verificación renal de Guard 4d. La proteína en ingredientes NO
proteína-dominantes (yogur/leche/leguminosas: el trim renal escala el NÚMERO pero solo reescribe los strings
de los protein-dominant) puede restaurarse sobre el cap KDIGO, dejando `meals_enforced` STALE-True. AMBAS
defensas downstream confían en ese flag: el exit-net `_renal_exit_safety_net` (re-trima solo si NOT
meals_enforced) y el fail-hard gate de should_retry (escala solo si meals_enforced is False) → fail-open de
un fail-hard de seguridad clínica.

Fix: Guard 8z.1 re-verifica HONESTAMENTE tras el ÚLTIMO pase que reescribe números (8z) y antes del return.
Parser-anchors (posición tras 8z + verify-only) + funcional aislado (truth-up monkeypatcheado para inflar).
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


# ── Parser-anchors ──
def test_marker_and_ordering(src):
    assert "P1-RENAL-TRUTHUP-RECHECK" in src
    i_8z = src.find("Guard 8z (P2-MACRO-TRUTHUP")
    i_recheck = src.find("Guard 8z.1 (ERC re-verificación POST-truthup)")
    i_applied = src.find('plan["_clinical_layer_applied"] = True')
    assert -1 not in (i_8z, i_recheck, i_applied)
    # La re-verificación DEBE ir DESPUÉS del truth-up (8z) y ANTES del return/marker final.
    assert i_8z < i_recheck < i_applied, "Guard 8z.1 debe correr tras 8z y antes del return"


def test_recheck_is_verify_only(src):
    s = src.find("Guard 8z.1 (ERC re-verificación POST-truthup)")
    e = src.find('plan["_clinical_layer_applied"] = True', s)
    block = src[s:e]
    assert 'plan["renal_protein_cap"]["meals_enforced"]' in block
    assert "_pg * 1.05" in block
    assert "_trim_day_protein_to_ceiling" not in block, "8z.1 es verify-only (no re-trima)"
    assert "_enforce_renal_per_meal" not in block, "8z.1 es verify-only (no re-enforza)"


# ── Funcional aislado: el truth-up (8z) infla proteína POST-4d → 8z.1 cae meals_enforced a False ──
class _StubDB:
    def __getattr__(self, name):
        def _f(*a, **k):
            return None
        return _f


def _renal_nutrition():
    return {"total_daily_macros": {"protein_str": "100g", "protein_g": 100, "carbs_g": 100, "fats_g": 40},
            "target_calories": 800, "macros": {"protein_g": 100, "carbs_g": 100, "fats_g": 40}}


def _plan_under_cap():
    # 40 + 40 = 80g ≤ 105 (cap*1.05) → Guard 4d setea meals_enforced=True. Sin 'ingredients' → quantize/
    # reconcile/closer/enforce reales no-op; el ÚNICO pase que mueve la proteína es el truth-up monkeypatcheado.
    return {"renal_protein_cap": {"applied": True, "protein_g": 100, "meals_enforced": True},
            "days": [{"meals": [{"name": "A", "protein": "40g", "calories": "400 kcal"},
                                {"name": "B", "protein": "40g", "calories": "400 kcal"}]}]}


def test_recheck_catches_truthup_reinflation(go, monkeypatch):
    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", lambda *a, **k: _StubDB())
    monkeypatch.setattr(go, "MACRO_TRUTHUP_ENABLED", True)

    def _inflate(meal, db):
        meal["protein"] = "90g"   # 90 + 90 = 180g >> 105 → debe caer a False
        return True
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", _inflate)

    out = go._apply_deterministic_clinical_layer(_plan_under_cap(), {"gender": "male"}, _renal_nutrition())
    assert out["renal_protein_cap"]["meals_enforced"] is False, \
        "tras la re-inflación del truth-up, 8z.1 debe marcar meals_enforced=False (no stale-True)"


def test_recheck_keeps_true_when_truthup_within_cap(go, monkeypatch):
    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", lambda *a, **k: _StubDB())
    monkeypatch.setattr(go, "MACRO_TRUTHUP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: False)  # no cambia nada
    out = go._apply_deterministic_clinical_layer(_plan_under_cap(), {"gender": "male"}, _renal_nutrition())
    assert out["renal_protein_cap"]["meals_enforced"] is True, "80g ≤ cap → no debe falsear meals_enforced"
