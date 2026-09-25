# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-235 · 2026-09-25] Colesterol alto: las yemas del plan caben en la nota que el propio plan escribe.

Batería real del 25-sep (colesterol + ganar músculo): «6½ yemas en el día 2» → dos críticos → plan «Pollo y Arroz».
La nota de dislipidemia dice «limita las yemas a 3-4 por semana; las claras puedes usarlas libremente».
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yemas_colesterol as yc  # noqa: E402

FORM = {"medicalConditions": ["Colesterol Alto"]}


def _plan(dias):
    return {"days": [{"day": i + 1, "meals": [
        {"meal": "Desayuno", "name": f"Revoltillo {i}", "ingredients": list(ings),
         "recipe": ["Mise en place: bate 3 huevos con sal.", "El Toque de Fuego: cocina 3 min.", "Montaje: sirve."]}
        for ings in dia]} for i, dia in enumerate(dias)]}


def _yemas_por_dia(plan):
    return [sum(yc._yemas_de_linea(l)[0] for m in d["meals"] for l in m["ingredients"]) for d in plan["days"]]


def test_una_al_dia_y_cuatro_a_la_semana():
    plan = _plan([[["3 huevos", "1 taza de espinaca"], ["2 huevos duros"]]] + [[["62 g de huevo"]]] * 6)
    n = yc.topar_yemas(plan, FORM)
    assert n >= 1
    por_dia = _yemas_por_dia(plan)
    assert all(y <= 1.0001 for y in por_dia), por_dia
    assert sum(por_dia[:7]) <= 4.0001, por_dia
    d1 = plan["days"][0]["meals"][0]
    assert "1 huevo" in d1["ingredients"] and "4 claras de huevo" in d1["ingredients"], d1["ingredients"]
    assert "bate 1 huevo y 4 claras con sal" in d1["recipe"][0], d1["recipe"]
    assert plan["days"][0]["meals"][1]["ingredients"][-1] == "4 claras de huevo", "el 2º plato del día ya no tiene yema"


def test_sin_colesterol_no_toca_nada():
    plan = _plan([[["3 huevos"]]])
    antes = copy.deepcopy(plan)
    assert yc.topar_yemas(plan, {"medicalConditions": ["Hipertensión"]}) == 0
    assert plan == antes


def test_las_claras_no_cuentan_y_es_idempotente():
    plan = _plan([[["1 huevo", "4 claras de huevo"]], [["1 huevo"]]])
    assert yc.topar_yemas(plan, FORM) == 0
    plan2 = _plan([[["3 huevos"]]])
    yc.topar_yemas(plan2, FORM)
    otra = copy.deepcopy(plan2)
    assert yc.topar_yemas(plan2, FORM) == 0 and plan2 == otra


def test_el_raw_se_ajusta_por_alimento():
    plan = _plan([[["3 huevos", "1 taza de espinaca"]]])
    plan["days"][0]["meals"][0]["ingredients_raw"] = ["30g de espinaca", "150g de huevo"]
    yc.topar_yemas(plan, FORM)
    raw = plan["days"][0]["meals"][0]["ingredients_raw"]
    assert "30g de espinaca" in raw and "50g de huevo" in raw and "132g de clara de huevo" in raw, raw


def test_cableado():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = go.index('__import__("yemas_colesterol").topar_yemas(plan, form_data, db=_db)')
    assert i < go.index("# ── Guard 3.5 (FS-embarazo): anotaciones de seguridad alimentaria embarazo/lactancia ──")
    dp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert dp.index("_ycol.topar_yemas(_pd, _clin_ctx, db=_db_ins)") < dp.index("_etq.etiquetar(_pd, _clin_ctx)")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 235
