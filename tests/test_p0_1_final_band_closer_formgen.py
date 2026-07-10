"""[P0-1-FINAL-BAND-CLOSER · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10 03:06-03:12 UTC): el banner
"Calidad por debajo del óptimo" venía del band-gate (P2-BAND-SCORE-GATE), NO del reviewer — el plan quedó
APROBADO pero carbs/kcal cayeron a 0.333 per-macro. `reconcile_protein_band_post_finalize` (shield
pre-INSERT) SOLO re-encuadra proteína; cuando falla carbs/fats/kcal nada lo corregía (12/33 planes
marcados en 72h, banda media de flota 0.761). `apply_update_macro_engine` ya es el motor SSOT all-4
(rebalance→refine 5g→truth-up) usado en updates — este batch solo lo conecta al shield universal
pre-INSERT (mismo punto que finalize_plan_data_coherence, cubre form-gen + partial + SSE-fallback).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


# ───────────────────────── estructural ─────────────────────────

def test_marker_and_knob_present():
    assert "P0-1-FINAL-BAND-CLOSER" in _GO
    assert "MEALFIT_FORMGEN_FINAL_BAND_CLOSER" in _GO


def test_function_defined():
    assert "def reconcile_all_macros_band_post_finalize(plan_data" in _GO


def test_called_in_dbplans_between_protein_reconcile_and_recheck():
    """Corre DESPUÉS del pase de proteína (proteína ya estable) y ANTES de clear_stale_low_band_degraded
    (que re-mide sobre el estado final) / refresh_clinical_band_score_post_finalize."""
    assert "reconcile_all_macros_band_post_finalize" in _DBP
    i_fpc = _DBP.index("_n, _summ = _fpc(")
    i_rpb = _DBP.index("reconcile_protein_band_post_finalize")
    i_all4 = _DBP.index("reconcile_all_macros_band_post_finalize")
    i_csd = _DBP.index("clear_stale_low_band_degraded")
    i_rbs = _DBP.index("refresh_clinical_band_score_post_finalize")
    assert i_fpc < i_rpb < i_all4 < i_csd, \
        "debe correr tras el finalize + pase de proteína y antes del clear-stale"
    assert i_all4 < i_rbs, "debe correr antes del re-check de banda"


# ───────────────────────── funcional ─────────────────────────

def _meal(name, protein, carbs, fats, ing):
    return {
        "meal": name, "name": name,
        "protein": protein, "carbs": carbs, "fats": fats,
        "cals": round(4 * protein + 4 * carbs + 9 * fats),
        "ingredients": ing, "ingredients_raw": list(ing),
        "macros": [f"P:{protein}g", f"C:{carbs}g", f"G:{fats}g"],
    }


def _plan_carbs_undersupplied():
    # target carbs=270g; el día SOLO entrega ~90g (0.33 ratio) — protein/fats en banda.
    meals = [
        _meal("Desayuno", 30, 20, 12, ["150 g de pechuga de pollo", "60 g de arroz blanco", "10 g de aceite de oliva"]),
        _meal("Almuerzo", 33, 25, 12, ["165 g de pechuga de pollo", "70 g de arroz blanco", "10 g de aceite de oliva"]),
        _meal("Cena", 32, 25, 12, ["160 g de pechuga de pollo", "70 g de arroz blanco", "10 g de aceite de oliva"]),
    ]
    return {"macros": {"protein": "95g", "carbs": "270g", "fats": "36g"},
            "calories": 1934,
            "days": [{"day": 1, "day_name": "Día 1", "meals": meals}]}


def _day_macro(pd, key):
    return sum(float(m[key]) for m in pd["days"][0]["meals"])


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\s+de\s+(.+)", str(s).strip(), re.IGNORECASE)
        if not m:
            return {"protein": 0, "carbs": 0, "fats": 0}
        grams, name = float(m.group(1)), m.group(2).lower()
        if "pollo" in name:
            return {"protein": round(grams * 0.20, 1), "carbs": 0, "fats": round(grams * 0.03, 1)}
        if "arroz" in name:
            return {"protein": round(grams * 0.03, 1), "carbs": round(grams * 0.28, 1), "fats": 0}
        if "aceite" in name:
            return {"protein": 0, "carbs": 0, "fats": round(grams * 1.0, 1)}
        return {"protein": 0, "carbs": 0, "fats": 0}


def test_carbs_undersupplied_day_rebalanced_toward_target(go, monkeypatch):
    monkeypatch.setattr(go, "FORMGEN_FINAL_BAND_CLOSER_ENABLED", True)
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    pd = _plan_carbs_undersupplied()
    before = _day_macro(pd, "carbs")
    assert before < 270 * 0.90, "fixture debe arrancar fuera de banda en carbs"
    go.reconcile_all_macros_band_post_finalize(pd, db=_FakeDB())
    after = _day_macro(pd, "carbs")
    assert after > before, "debió escalar carbos hacia el target"
    assert abs(after - 270) < abs(before - 270), "debió acercar carbos al target"


def test_in_band_day_untouched(go, monkeypatch):
    monkeypatch.setattr(go, "FORMGEN_FINAL_BAND_CLOSER_ENABLED", True)
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    meals = [_meal("Desayuno", 40, 90, 20, ["200 g de pechuga de pollo", "320 g de arroz blanco", "20 g de aceite de oliva"])]
    pd = {"macros": {"protein": "40g", "carbs": "90g", "fats": "20g"}, "calories": 1000,
          "days": [{"day": 1, "meals": meals}]}
    before_c, before_f, before_p = _day_macro(pd, "carbs"), _day_macro(pd, "fats"), _day_macro(pd, "protein")
    changed = go.reconcile_all_macros_band_post_finalize(pd, db=_FakeDB())
    assert changed == 0
    assert _day_macro(pd, "carbs") == before_c
    assert _day_macro(pd, "fats") == before_f
    assert _day_macro(pd, "protein") == before_p


def test_knob_off_noop(go, monkeypatch):
    monkeypatch.setattr(go, "FORMGEN_FINAL_BAND_CLOSER_ENABLED", False)
    pd = _plan_carbs_undersupplied()
    before = _day_macro(pd, "carbs")
    assert go.reconcile_all_macros_band_post_finalize(pd, db=_FakeDB()) == 0
    assert _day_macro(pd, "carbs") == before


def test_none_and_malformed_input_safe(go, monkeypatch):
    monkeypatch.setattr(go, "FORMGEN_FINAL_BAND_CLOSER_ENABLED", True)
    assert go.reconcile_all_macros_band_post_finalize(None) == 0
    assert go.reconcile_all_macros_band_post_finalize({"days": "corrupted"}) == 0
    assert go.reconcile_all_macros_band_post_finalize({}) == 0
