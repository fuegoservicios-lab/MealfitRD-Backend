"""[P1-BAND-TELEMETRY-PER-DAY · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): reconstruir CUÁL día y
CUÁL macro cayó fuera de banda requirió reprocesar logs línea por línea desde 0 (el KV del guest plan ya
había sido borrado por el ack) — `clinical_band_score` solo persistía el score AGREGADO por-macro, no la
matriz día×macro. `compute_clinical_band_score` ahora expone `per_day`: por cada día, el ratio
entregado/target de cada macro (redondeado, compacto para jsonb) — permite auditar sin reprocesar logs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go


def _nutrition(p=150, c=200, f=60, kcal=2000):
    return {"total_daily_calories": kcal,
            "total_daily_macros": {"protein_str": f"{p}g", "carbs_str": f"{c}g", "fats_str": f"{f}g"}}


def _day(protein, carbs, fats, cals, day_num=None):
    d = {"meals": [{"protein": protein, "carbs": carbs, "fats": fats, "cals": cals}]}
    if day_num is not None:
        d["day"] = day_num
    return d


def test_per_day_matrix_present_with_correct_length():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(150, 200, 60, 2000, 1), _day(75, 200, 60, 2000, 2)]}
    r = go.compute_clinical_band_score(plan, nut)
    assert "per_day" in r
    assert len(r["per_day"]) == 2


def test_per_day_ratios_match_delivered_over_target():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(75, 200, 60, 2000, 1)]}   # proteína 50% del target
    r = go.compute_clinical_band_score(plan, nut)
    row = r["per_day"][0]
    assert row["day"] == 1
    assert row["ratios"]["protein"] == 0.5
    assert row["ratios"]["carbs"] == 1.0
    assert row["ratios"]["fats"] == 1.0
    assert row["ratios"]["kcal"] == 1.0


def test_per_day_all4_in_band_flag():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(150, 200, 60, 2000, 1), _day(75, 200, 60, 2000, 2)]}
    r = go.compute_clinical_band_score(plan, nut)
    assert r["per_day"][0]["all4_in_band"] is True
    assert r["per_day"][1]["all4_in_band"] is False


def test_per_day_falls_back_to_index_when_no_day_key():
    nut = _nutrition(150, 200, 60, 2000)
    plan = {"days": [_day(150, 200, 60, 2000), _day(150, 200, 60, 2000)]}  # sin 'day' key
    r = go.compute_clinical_band_score(plan, nut)
    assert [row["day"] for row in r["per_day"]] == [1, 2]


def test_per_day_empty_when_no_targets():
    r = go.compute_clinical_band_score({"days": []}, {})
    assert r["per_day"] == []
