# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-267 · 2026-09-25] Tras una sustitución determinista, la descripción tampoco nombra lo que se quitó.

Verificación con IA real del lote 264 (DM2 + HTA, metformina): lista con tayota, título y pasos reescritos, y la
descripción seguía diciendo «con cundeamor salteado».
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_DESC = ("Cena ligera pero con identidad propia: filete de tilapia al horno con limón, ajo y cilantro sobre quinoa "
         "esponjosa, con cundeamor salteado y cebolla roja.")


def _plan():
    return {"days": [{"day": 2, "meals": [{
        "meal": "Cena", "name": "Tilapia al horno con quinoa y cundeamor", "desc": _DESC,
        "ingredients": ["120 g de filete de tilapia", "40 g de quinoa", "100 g de cundeamor"],
        "ingredients_raw": ["120 g de filete de tilapia", "40 g de quinoa", "100 g de cundeamor"],
        "recipe": ["Mise en place: corta el cundeamor en rodajas.", "El Toque de Fuego: saltea el cundeamor 4 min.",
                   "Montaje: sirve la tilapia sobre la quinoa con el cundeamor."]}]}]}


def test_la_descripcion_deja_de_nombrar_el_cundeamor():
    plan = _plan()
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes T2"],
                                                    "medications": ["Metformina"]}) >= 1
    m = plan["days"][0]["meals"][0]
    assert not any("cundeamor" in i.lower() for i in m["ingredients"])
    assert "cundeamor" not in m["desc"].lower() and "tayota" in m["desc"].lower()
    assert "cundeamor" not in m["name"].lower()


def test_idempotente_y_sin_duplicar():
    plan = _plan()
    form = {"medicalConditions": ["Diabetes T2"], "medications": ["Metformina"]}
    go._apply_condition_substitutions(plan, form)
    primero = plan["days"][0]["meals"][0]["desc"]
    go._apply_condition_substitutions(plan, form)
    assert plan["days"][0]["meals"][0]["desc"] == primero
    assert primero.lower().count("tayota") == 1


def test_sin_sustitucion_la_descripcion_no_se_toca():
    plan = _plan()
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Ninguna"]})
    assert plan["days"][0]["meals"][0]["desc"] == _DESC


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("desc_tras_sustitucion").titulo_y_desc(meal, recipe_token_subs)' in src
    assert "P1-PLAN-LOTE-267-DESC" in (_BACKEND / "desc_tras_sustitucion.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 267
