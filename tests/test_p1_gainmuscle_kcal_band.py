"""[P1-GAINMUSCLE-KCAL-BAND · 2026-07-06] Banda kcal DIRECCIONAL por objetivo en compute_clinical_band_score.

Forense verificado (plan 4339544f, ganancia muscular, target 2100): D1=2243 (107%), D2=2214 (105%),
D3=2133 (102%). La banda kcal por defecto [0.95,1.05] dejaba la celda kcal en 1/3 → el backstop de retry
(BAND_GATE_KCAL_THRESHOLD 0.5) forzaba rechazo aun con macros 8/9 en banda → 3 intentos → max_attempts +
banner de degradación. El techo laxo [0.95,1.10] SOLO para gain pone los 3 días en banda (kcal 3/3) → el
plan aprueba en intento 1. Déficit/mantenimiento/salud conservan la banda estricta (pasarse ES el defecto).
"""
import graph_orchestrator as go


def _day(k, dk):
    # una comida con P/C/F == target (trivialmente en banda) y kcal = dk → aísla la celda kcal.
    return {"day": k, "meals": [{"protein": 100, "carbs": 100, "fats": 100, "cals": dk}]}


# réplica exacta de los kcal/día del plan real 4339544f contra target 2100.
_BASE = {"calories": 2100, "macros": {"protein": 100, "carbs": 100, "fats": 100},
         "days": [_day(1, 2243), _day(2, 2214), _day(3, 2133)]}


def test_gain_muscle_loosens_kcal_upper_all_days_in_band():
    gain = dict(_BASE, main_goal="Ganancia Muscular (Superávit 15% — ritmo decidido)")
    r = go.compute_clinical_band_score(gain, {})
    assert r["band_kcal"] == [0.95, go.GAINMUSCLE_KCAL_BAND_UPPER]
    assert r["per_macro"]["kcal"] == 1.0, "107%/105%/102% caen en [0.95,1.10] → kcal 3/3"


def test_deficit_keeps_strict_band():
    cut = dict(_BASE, main_goal="Pérdida de Peso (Déficit)")
    r = go.compute_clinical_band_score(cut, {})
    assert r["band_kcal"] == [0.95, 1.05], "en déficit la banda kcal NO se afloja"
    # 107% y 105% fuera; 102% dentro → 1/3
    assert abs(r["per_macro"]["kcal"] - 0.333) < 0.01, "pasarse de kcal en un cut SIGUE siendo defecto"


def test_maintenance_and_health_keep_strict_band():
    for lbl in ("Mantenimiento", "Salud General", "Recomposición Corporal"):
        r = go.compute_clinical_band_score(dict(_BASE, main_goal=lbl), {})
        assert r["band_kcal"] == [0.95, 1.05], f"{lbl} → banda estricta"
        assert not go._plan_goal_is_gainmuscle(dict(_BASE, main_goal=lbl))


def test_persisted_plan_shape_derives_from_main_goal_not_empty_form_data():
    # OJO: en el plan PERSISTIDO form_data se vacía → derivar de form_data.mainGoal daría no-op silencioso.
    # La fuente correcta es el top-level main_goal (verificado en DB para 4339544f).
    persisted = dict(_BASE, main_goal="Ganancia Muscular (Superávit 15%)", form_data={})
    r = go.compute_clinical_band_score(persisted, {})
    assert r["per_macro"]["kcal"] == 1.0
    assert go._plan_goal_is_gainmuscle(persisted) is True


def test_explicit_goal_param_overrides():
    # un caller con form_data en runtime puede pasar el goal explícito.
    plan_no_goal = dict(_BASE)  # sin main_goal
    assert go.compute_clinical_band_score(plan_no_goal, {})["band_kcal"] == [0.95, 1.05]
    r = go.compute_clinical_band_score(plan_no_goal, {}, goal="gain_muscle")
    assert r["band_kcal"] == [0.95, go.GAINMUSCLE_KCAL_BAND_UPPER]
    assert r["per_macro"]["kcal"] == 1.0


def test_detection_tokens_and_accents():
    assert go._plan_goal_is_gainmuscle({}, "gain_muscle")
    assert go._plan_goal_is_gainmuscle({}, "Ganancia Muscular")
    assert go._plan_goal_is_gainmuscle({}, "bulk agresivo")
    assert go._plan_goal_is_gainmuscle({}, "Superávit calórico")  # acento + token superavit
    assert not go._plan_goal_is_gainmuscle({}, "Pérdida de Peso (Déficit)")
    assert not go._plan_goal_is_gainmuscle({}, "")
    assert not go._plan_goal_is_gainmuscle({}, None)


def test_knob_clamp_and_default():
    # default 1.10, clamp [1.05, 1.20].
    assert 1.05 <= go.GAINMUSCLE_KCAL_BAND_UPPER <= 1.20


def test_fail_safe_on_bad_plan():
    # detección fail-safe → False (banda estricta) ante entradas raras; el score no explota.
    assert go._plan_goal_is_gainmuscle(None) is False
    assert go._plan_goal_is_gainmuscle({"main_goal": 12345}) is False  # no-str → _norm_text lo castea


def test_marker_anchored():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-GAINMUSCLE-KCAL-BAND" in src
    assert "GAINMUSCLE_KCAL_BAND_UPPER" in src
