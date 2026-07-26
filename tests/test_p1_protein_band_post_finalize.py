"""[P1-PROTEIN-BAND-POST-FINALIZE · 2026-07-09] El truth-up pre-INSERT (finalize_plan_data_coherence)
recomputa la proteína HONESTA desde los strings DESPUÉS de que los closers de proteína ya corrieron en
assemble → re-expone drift que nada re-encuadra. Forense plan vivo 42310dba (gain_muscle): banda
pre-finalize 1.0 → post-finalize 0.75 (protein 0.333; día 1 114g < piso 117, día 2 146g > techo 145.6).

`reconcile_protein_band_post_finalize` corre en el shield pre-INSERT DESPUÉS del finalize y ANTES del
re-check de banda: por día re-escala porciones proteína-dominantes EXISTENTES (sin ingredientes nuevos →
sin riesgo de alérgeno) — trim si sobre el techo, bump si bajo el piso.
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
    assert "P1-PROTEIN-BAND-POST-FINALIZE" in _GO
    assert "MEALFIT_PROTEIN_BAND_POST_FINALIZE" in _GO


def test_functions_defined():
    assert "def reconcile_protein_band_post_finalize(plan_data" in _GO
    assert "def _bump_day_protein_to_floor(" in _GO


def test_called_in_dbplans_between_finalize_and_recheck():
    """Corre DESPUÉS de _fpc (finalize/truth-up) y ANTES de _csd/_rbs (re-medición de banda)."""
    assert "reconcile_protein_band_post_finalize" in _DBP
    i_fpc = _DBP.index("_n, _summ = _fpc(")
    i_rpb = _DBP.index("reconcile_protein_band_post_finalize")
    i_csd = _DBP.index("clear_stale_low_band_degraded")
    i_rbs = _DBP.index("refresh_clinical_band_score_post_finalize")
    assert i_fpc < i_rpb < i_csd, "debe correr tras el finalize y antes del clear-stale"
    assert i_rpb < i_rbs, "debe correr antes del re-check de banda"


# ───────────────────────── funcional ─────────────────────────

def _meal(prot_label, pollo_g):
    return {
        "meal": "Comida", "name": "Pollo Guisado",
        "protein": prot_label, "carbs": 20, "fats": 8,
        "cals": round(4 * prot_label + 4 * 20 + 9 * 8),
        "ingredients": [f"{pollo_g} g de pollo", "20 g de arroz"],
        "ingredients_raw": [f"{pollo_g} g de pollo", "20 g de arroz"],
        "macros": [f"P:{prot_label}g", "C:20g", "G:8g"],
    }


def _plan(meals):
    return {"macros": {"protein": "130g", "carbs": "273g", "fats": "60g"},
            "calories": 2150,
            "days": [{"day": 1, "day_name": "Día 1", "meals": meals}]}


def _day_protein(pd):
    return sum(float(m["protein"]) for m in pd["days"][0]["meals"])


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def test_over_ceiling_day_trimmed_toward_target(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", True)
    pd = _plan([_meal(43, 140) for _ in range(4)])  # ~172g > techo 145.6
    before = _day_protein(pd)
    assert before > 130 * 1.12
    go.reconcile_protein_band_post_finalize(pd)
    after = _day_protein(pd)
    assert after < before, "debió trimar la proteína sobre-techo"
    assert abs(after - 130) < abs(before - 130), "debió acercarla al target"


def test_under_floor_day_bumped_toward_target(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", True)
    pd = _plan([_meal(25, 80) for _ in range(4)])  # ~100g < piso 117
    before = _day_protein(pd)
    assert before < 130 * 0.90
    go.reconcile_protein_band_post_finalize(pd)
    after = _day_protein(pd)
    assert after > before, "debió bumpear la proteína bajo-piso"
    assert abs(after - 130) < abs(before - 130), "debió acercarla al target"


def test_in_band_day_untouched(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", True)
    pd = _plan([_meal(32, 105) for _ in range(4)])  # ~128g en banda [117, 145.6]
    before = _day_protein(pd)
    changed = go.reconcile_protein_band_post_finalize(pd)
    assert _day_protein(pd) == before, "un día en banda no debe tocarse"
    assert changed is False


def test_idempotent(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", True)
    pd = _plan([_meal(43, 140) for _ in range(4)])
    go.reconcile_protein_band_post_finalize(pd)
    mid = _day_protein(pd)
    go.reconcile_protein_band_post_finalize(pd)  # 2ª pasada
    assert _day_protein(pd) == mid, "2ª pasada sobre un día ya en banda = no-op"


def test_knob_off_noop(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", False)
    pd = _plan([_meal(43, 140) for _ in range(4)])
    before = _day_protein(pd)
    assert go.reconcile_protein_band_post_finalize(pd) is False
    assert _day_protein(pd) == before


def test_no_kcal_push_when_at_ceiling(go, monkeypatch):
    """Un día bajo el piso de proteína PERO ya en el techo de kcal no se bumpea (evita cambiar un fallo
    de proteína por uno de kcal)."""
    monkeypatch.setattr(go, "PROTEIN_BAND_POST_FINALIZE_ENABLED", True)
    # 4 comidas: proteína baja (~100g) pero kcal del día ya en el techo (2150×1.10 = 2365).
    meals = [_meal(25, 80) for _ in range(4)]
    for m in meals:
        m["cals"] = 600  # 4×600 = 2400 > techo 2365
    pd = _plan(meals)
    before = _day_protein(pd)
    go.reconcile_protein_band_post_finalize(pd)
    assert _day_protein(pd) == before, "no debe empujar proteína si el día ya rebasó el techo de kcal"
