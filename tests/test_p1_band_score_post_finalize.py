"""[P1-BAND-SCORE-POST-FINALIZE · 2026-07-08] Refresca `plan_data['clinical_band_score']` sobre el
estado ENTREGADO (post-finalize), no solo el banner de degradado.

Forense en vivo (plan 830d9aaa): `🎯 [CLINICAL BAND SCORE] Precisión medida: 0.58 (kcal=0.333)` se
logueó a las 08:20:07 — 11s ANTES de `[P1-COHERENCE-FINALIZE] pre-INSERT aplicó coherencia` (08:20:18,
fats_relevel=2, final_truthup=5). `clear_stale_low_band_degraded` (mismo día) ya recomputa este score
post-finalize, pero SOLO cuando el plan estaba flagged `_quality_degraded=low_band_*`, y descarta el
resultado sin persistirlo. Para el caso común (plan NO flagged), la métrica en `plan_data` seguía
siendo la lectura stale — sin SQL forense no hay forma de saber si el ENTREGADO quedó mejor.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_knob_and_function():
    assert "P1-BAND-SCORE-POST-FINALIZE" in _GO
    assert 'BAND_SCORE_POST_FINALIZE_REFRESH_ENABLED = _env_bool("MEALFIT_BAND_SCORE_POST_FINALIZE_REFRESH", True)' in _GO
    assert "def refresh_clinical_band_score_post_finalize(plan_data" in _GO


def test_callsite_after_finalize_and_alongside_clear_in_db_plans():
    """Corre en el mismo shield pre-INSERT, DESPUÉS de finalize y junto a clear_stale_low_band_degraded
    (no depende de él — ambos son llamadas hermanas independientes)."""
    assert "refresh_clinical_band_score_post_finalize" in _DBP
    i_fin = _DBP.index("finalize_plan_data_coherence as _fpc")
    i_clr = _DBP.index("clear_stale_low_band_degraded")
    i_rbs = _DBP.index("refresh_clinical_band_score_post_finalize")
    assert i_clr > i_fin and i_rbs > i_clr, "orden: finalize → clear (banner) → refresh (métrica)"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_SCORE_POST_FINALIZE_REFRESH_ENABLED", True)
    return g


def _meal(P, C, F):
    return {"protein": P, "carbs": C, "fats": F, "cals": P * 4 + C * 4 + F * 9}


def _plan(day_macros, *, stale_score=None, degraded=False):
    """day_macros: lista de (P,C,F) por día. Target P=120 C=270 F=58 kcal=2100."""
    pd = {
        "macros": {"protein": "120g", "carbs": "270g", "fats": "58g"},
        "calories": 2100,
        "main_goal": "maintain",
        "days": [{"day": i + 1, "meals": [_meal(*dm)]} for i, dm in enumerate(day_macros)],
    }
    if stale_score is not None:
        pd["clinical_band_score"] = stale_score
    if degraded:
        pd["_quality_degraded"] = True
        pd["_quality_degraded_reason"] = "low_band_macro:fats"
    return pd


def test_refreshes_score_for_non_degraded_plan(go):
    """El caso COMÚN: el plan nunca fue flagged degradado (clear_stale_low_band_degraded no haría
    nada), pero la métrica sigue siendo stale — este helper la refresca de todas formas."""
    pd = _plan([(120, 270, 58)] * 3,
               stale_score={"score": 0.333, "cells_in_band": 4, "cells_total": 12, "per_macro": {}})
    refreshed = go.refresh_clinical_band_score_post_finalize(pd)
    assert refreshed is True
    assert pd["clinical_band_score"]["score"] != 0.333, "la lectura stale debe reemplazarse"
    assert pd["clinical_band_score"]["score"] > 0.9, "3 días todos en target → score alto"


def test_overwrites_stale_prefinalize_value(go):
    """Réplica del forense 830d9aaa: score pre-finalize bajo, estado entregado real está en banda."""
    pd = _plan([(120, 270, 58)] * 3,
               stale_score={"score": 0.58, "cells_in_band": 7, "cells_total": 12,
                            "per_macro": {"kcal": 0.333}})
    go.refresh_clinical_band_score_post_finalize(pd)
    assert pd["clinical_band_score"]["score"] == 1.0
    assert pd["clinical_band_score"]["per_macro"]["kcal"] != 0.333


def test_does_not_touch_quality_degraded_flag(go):
    """CLEAR-ONLY del banner es responsabilidad exclusiva de clear_stale_low_band_degraded — este
    helper nunca marca ni limpia `_quality_degraded`, solo refresca la métrica."""
    pd = _plan([(120, 270, 30)] * 3, degraded=True)  # fuera de banda + flagged
    go.refresh_clinical_band_score_post_finalize(pd)
    assert pd.get("_quality_degraded") is True, "el flag es intocado por este helper"
    assert "clinical_band_score" in pd, "pero la métrica sí se refresca"


def test_knob_off_noop(go, monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_POST_FINALIZE_REFRESH_ENABLED", False)
    pd = _plan([(120, 270, 58)] * 3, stale_score={"score": 0.333})
    refreshed = go.refresh_clinical_band_score_post_finalize(pd)
    assert refreshed is False
    assert pd["clinical_band_score"]["score"] == 0.333, "knob OFF → no toca la métrica"


def test_no_op_on_non_dict():
    import graph_orchestrator as g
    assert g.refresh_clinical_band_score_post_finalize(None) is False
    assert g.refresh_clinical_band_score_post_finalize("not a dict") is False
