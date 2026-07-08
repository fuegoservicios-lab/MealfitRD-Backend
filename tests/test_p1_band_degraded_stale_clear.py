"""[P1-BAND-DEGRADED-STALE-CLEAR · 2026-07-08] Limpieza del banner _quality_degraded stale.

Forense del plan vivo 618ecd21 (gain_muscle): el gate de banda del pipeline corrió a las 07:38:16 y vio
fats per_macro=0.333 (un día ALTO + uno bajo fuera de banda) → marcó `_quality_degraded=low_band_macro:fats`.
DESPUÉS, a las 07:38:20-21, FATS-RELEVEL-UNIVERSAL recortó el día alto y COHERENCE-FINALIZE re-trutheó → el
estado ENTREGADO (persistido) tiene fats=0.667 (≥ umbral 0.5). El banner "Calidad por debajo del óptimo" era
un FALSO POSITIVO de timing: el gate midió un estado intermedio peor que el entregado.

`clear_stale_low_band_degraded` corre post-finalize (db_plans.py, pre-INSERT) y LIMPIA el flag low_band_* si el
estado entregado ya NO califica (band_val>=thr Y ningún macro<per-macro-thr — complemento exacto del gate).
CLEAR-ONLY: nunca marca; razones no-banda (clínicas/fallback/max_attempts) jamás se tocan.
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
    assert "P1-BAND-DEGRADED-STALE-CLEAR" in _GO
    assert 'BAND_DEGRADED_STALE_CLEAR_ENABLED = _env_bool("MEALFIT_BAND_DEGRADED_STALE_CLEAR", True)' in _GO
    assert "def clear_stale_low_band_degraded(plan_data" in _GO


def test_callsite_after_finalize_in_db_plans():
    """El re-check corre en el shield pre-INSERT, DESPUÉS de finalize_plan_data_coherence."""
    assert "clear_stale_low_band_degraded" in _DBP
    i_fin = _DBP.index("finalize_plan_data_coherence as _fpc")
    i_clr = _DBP.index("clear_stale_low_band_degraded")
    assert i_clr > i_fin, "el clear debe correr DESPUÉS de importar/llamar finalize"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_DEGRADED_STALE_CLEAR_ENABLED", True)
    monkeypatch.setattr(g, "BAND_SCORE_GATE_ENABLED", True)
    return g


def _meal(P, C, F):
    return {"protein": P, "carbs": C, "fats": F, "cals": P * 4 + C * 4 + F * 9}


def _plan(day_macros, *, reason="low_band_macro:fats", degraded=True):
    """day_macros: lista de (P,C,F) por día. Target P=120 C=270 F=58 kcal=2100."""
    pd = {
        "macros": {"protein": "120g", "carbs": "270g", "fats": "58g"},
        "calories": 2100,
        "main_goal": "maintain",
        "days": [{"day": i + 1, "meals": [_meal(*dm)]} for i, dm in enumerate(day_macros)],
    }
    if degraded:
        pd["_quality_degraded"] = True
        pd["_quality_degraded_reason"] = reason
        pd["_quality_degraded_severity"] = "minor"
        pd["_quality_degraded_band_score"] = 0.333
    return pd


def test_clears_when_delivered_in_band(go):
    """3 días TODOS en banda → ningún macro bajo umbral → el flag stale low_band se limpia."""
    pd = _plan([(120, 270, 58)] * 3)  # cada macro 1.0× target
    assert pd["_quality_degraded"] is True
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is True
    assert not pd.get("_quality_degraded"), "banner limpiado"
    assert "_quality_degraded_reason" not in pd
    assert "_quality_degraded_band_score" not in pd


def test_leaves_flag_when_still_out_of_band(go):
    """Grasa fuera de banda en los 3 días (fats per_macro=0.0<0.5) → NO se limpia (honesto)."""
    pd = _plan([(120, 270, 30)] * 3)  # fats 30/58=0.52× → fuera [0.9,1.12]
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is False
    assert pd.get("_quality_degraded") is True, "sigue degradado — el estado entregado sí está fuera"
    assert pd["_quality_degraded_reason"] == "low_band_macro:fats"


def test_ignores_non_band_reasons(go):
    """Una razón clínica/no-banda jamás se toca aunque el plan esté en banda."""
    pd = _plan([(120, 270, 58)] * 3, reason="condition_panel_gap")
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is False
    assert pd.get("_quality_degraded") is True
    assert pd["_quality_degraded_reason"] == "condition_panel_gap"


def test_clear_only_never_marks(go):
    """CLEAR-ONLY: un plan SIN flag y fuera de banda NO recibe un banner nuevo."""
    pd = _plan([(120, 270, 30)] * 3, degraded=False)
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is False
    assert not pd.get("_quality_degraded"), "nunca marca en la dirección de degradar"


def test_knob_off_leaves_flag(go, monkeypatch):
    monkeypatch.setattr(go, "BAND_DEGRADED_STALE_CLEAR_ENABLED", False)
    pd = _plan([(120, 270, 58)] * 3)
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is False
    assert pd.get("_quality_degraded") is True, "knob OFF → no limpia"


def test_matches_live_618ecd21_shape(go):
    """Réplica del forense real: día ALTO en grasa + día BAJO + día ok. Tras el trim del día alto por
    finalize, la celda fats sube a 2/3 (0.667) → sobre umbral → se limpia. (Aquí simulamos el estado
    ENTREGADO post-finalize: Día1 61g ok, Día2 51g bajo, Día3 62g ok.)"""
    pd = _plan([(116, 269, 61), (120, 264, 51), (126, 243, 62)])
    # fats: 61(ok) 51(fuera) 62(ok) = 2/3=0.667 ; carbs/protein mayormente en banda → sin macro<0.5
    cleared = go.clear_stale_low_band_degraded(pd)
    assert cleared is True, "el estado entregado (fats 0.667) no califica para degradar → limpio"
    assert not pd.get("_quality_degraded")
