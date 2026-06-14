"""[P4-SCOREBOARD · 2026-06-14] La precisión de macros deja de ser AUTOAFIRMADA: se mide en cada plan
vía `clinical_band_score` (fracción de celdas día×macro del plan ENTREGADO en la banda clínica vs el
target). Se inyecta a result['clinical_band_score'], se emite la métrica `clinical_band` a
pipeline_metrics, y un cron de drift (`_clinical_band_drift_alert_job`) agrega la flota y alerta.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go


def _nutrition(p=150, c=200, f=60, kcal=2000):
    return {"total_daily_calories": kcal,
            "total_daily_macros": {"protein_str": f"{p}g", "carbs_str": f"{c}g", "fats_str": f"{f}g"}}


def _day(protein, carbs, fats, cals):
    return {"meals": [{"protein": protein, "carbs": carbs, "fats": fats, "cals": cals}]}


# ── El scorer determinista ──
def test_perfect_plan_scores_one():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(150, 200, 60, 2000), _day(150, 200, 60, 2000)]}
    r = go.compute_clinical_band_score(plan, nut)
    assert r["score"] == 1.0
    assert r["cells_total"] == 8   # 2 días × 4 macros
    assert all(v == 1.0 for v in r["per_macro"].values())


def test_protein_below_band_drops_score():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(75, 200, 60, 2000)]}   # proteína 50% del target
    r = go.compute_clinical_band_score(plan, nut)
    assert r["per_macro"]["protein"] == 0.0      # 75 < 0.9×150
    assert r["per_macro"]["carbs"] == 1.0
    assert r["score"] == 0.75                    # 3/4 celdas en banda


def test_kcal_uses_stricter_band():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(150, 200, 60, 2160)]}  # kcal 1.08× — dentro de [0.9,1.12] macro pero fuera de [0.95,1.05] kcal
    r = go.compute_clinical_band_score(plan, nut)
    assert r["per_macro"]["kcal"] == 0.0
    assert r["band_kcal"] == [0.95, 1.05]


def test_band_edges_protein():
    nut = _nutrition(100, 100, 100, 1000)
    assert go.compute_clinical_band_score({"days": [_day(90, 100, 100, 1000)]}, nut)["per_macro"]["protein"] == 1.0   # 0.90 borde inclusivo
    assert go.compute_clinical_band_score({"days": [_day(89, 100, 100, 1000)]}, nut)["per_macro"]["protein"] == 0.0   # 0.89 fuera
    assert go.compute_clinical_band_score({"days": [_day(112, 100, 100, 1000)]}, nut)["per_macro"]["protein"] == 1.0  # 1.12 borde inclusivo
    assert go.compute_clinical_band_score({"days": [_day(113, 100, 100, 1000)]}, nut)["per_macro"]["protein"] == 0.0  # 1.13 fuera


def test_missing_targets_is_graceful():
    r = go.compute_clinical_band_score({"days": []}, {})
    assert r["cells_total"] == 0 and r["score"] is None


def test_falls_back_to_plan_header_when_no_nutrition_macros():
    # sin nutrition.total_daily_macros, usa plan["macros"] + plan["calories"]
    plan = {"macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}, "calories": 2000,
            "days": [_day(150, 200, 60, 2000)]}
    r = go.compute_clinical_band_score(plan, {})
    assert r["score"] == 1.0


def test_knob_band_widens(monkeypatch):
    nut = _nutrition(100, 100, 100, 1000)
    # con banda default [0.9,1.12], 80 está fuera; con lower=0.75 (override) entra
    assert go.compute_clinical_band_score({"days": [_day(80, 100, 100, 1000)]}, nut)["per_macro"]["protein"] == 0.0
    r = go.compute_clinical_band_score({"days": [_day(80, 100, 100, 1000)]}, nut, lower=0.75, upper=1.25)
    assert r["per_macro"]["protein"] == 1.0


# ── Cableado: score emitido por-plan + cron de drift registrado ──
def test_band_score_wired_into_holistic_emit():
    src = inspect.getsource(go._compute_pipeline_holistic_score_and_emit)
    assert 'compute_clinical_band_score(plan, nutrition)' in src
    assert 'plan["clinical_band_score"]' in src
    assert '"node": "clinical_band"' in src


def test_drift_cron_exists_and_registered():
    import cron_tasks as ct
    assert hasattr(ct, "_clinical_band_drift_alert_job")
    reg_src = inspect.getsource(ct.register_plan_chunk_scheduler)
    assert "clinical_band_drift_alert" in reg_src
    cron_src = inspect.getsource(ct._clinical_band_drift_alert_job)
    assert 'alert_key = "clinical_band_drift"' in cron_src
    assert "node = 'clinical_band'" in cron_src


def test_drift_alert_key_documented():
    import pathlib
    doc = (pathlib.Path(ct_dir()) / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "clinical_band_drift" in doc


def ct_dir():
    import cron_tasks as ct
    return os.path.dirname(os.path.abspath(ct.__file__))
