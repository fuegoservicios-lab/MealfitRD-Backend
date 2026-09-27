# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-464 · 2026-09-27] La sustitución de la compra única corre ANTES del band-closer del escudo.

La sustitución cambia el CONTENIDO del plato (pechuga → sardinas, filete → garbanzos) y el band-closer decide las
PORCIONES. Sólo corría dentro del bucle de caps, DESPUÉS del closer; en el merge de un bloque (días 4+, los únicos que
sustituye) el chain corre una vez, así que el día quedaba con el contenido nuevo y las porciones del viejo. Aquí:
contenido → porciones → caps (última palabra). La llamada del bucle de caps se queda como red.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import db_plans  # noqa: E402
import graph_orchestrator as go  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced"}}


def _plan():
    return {"days": [{"day": 4, "meals": [{"meal": "Cena", "name": "Pollo con arroz",
                                           "ingredients": ["150 g de pechuga de pollo", "1 taza de arroz"],
                                           "ingredients_raw": ["150 g de pechuga de pollo", "1 taza de arroz"],
                                           "recipe": ["Montaje: sirve el pollo con el arroz."]}]}],
            "_plan_policy": {"effective": SINGLE}, "_days_offset": 3, "calories": 2000,
            "macros": {"protein": "120g", "carbs": "220g", "fats": "60g"}}


def test_la_sustitucion_va_antes_del_band_closer(monkeypatch):
    orden = []
    real_stfs = go._single_trip_fresh_substitute

    def stfs(*a, **k):
        orden.append("sustitucion")
        return real_stfs(*a, **k)

    monkeypatch.setattr(go, "_single_trip_fresh_substitute", stfs)
    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", lambda pd, *a, **k: orden.append("closer_proteina"))
    monkeypatch.setattr(go, "reconcile_all_macros_band_post_finalize",
                        lambda pd, *a, **k: orden.append("closer_all4"))
    monkeypatch.setattr(db_plans, "_build_clinical_form", lambda uid: {})
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    pd = _plan()
    db_plans.apply_plan_quality_finalize_chain(pd, surface="test-464", form_data={})
    assert "sustitucion" in orden and "closer_proteina" in orden, orden
    assert orden.index("sustitucion") < orden.index("closer_proteina") < orden.index("closer_all4"), orden
    linea = pd["days"][0]["meals"][0]["ingredients"][0]
    assert "pechuga" not in linea, "el closer ya trabajó con el duradero"


def test_sin_compra_unica_no_se_llama_antes_del_closer(monkeypatch):
    orden = []
    monkeypatch.setattr(go, "_single_trip_fresh_substitute", lambda *a, **k: orden.append("sustitucion") or 0)
    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", lambda pd, *a, **k: orden.append("closer_proteina"))
    monkeypatch.setattr(db_plans, "_build_clinical_form", lambda uid: {})
    pd = _plan()
    pd["_plan_policy"] = {"effective": {"shopping": {}, "diet": {"type": "balanced"}}}
    db_plans.apply_plan_quality_finalize_chain(pd, surface="test-464", form_data={})
    assert "closer_proteina" in orden
    assert "sustitucion" not in orden[:orden.index("closer_proteina")], orden


def test_ancla_en_el_codigo():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("P1-PLAN-LOTE-464-SUSTITUIR-ANTES-DEL-CLOSER")
    j = src.index("from graph_orchestrator import reconcile_protein_band_post_finalize as _rpb")
    assert i < j, "la sustitución va antes del closer de proteína"
