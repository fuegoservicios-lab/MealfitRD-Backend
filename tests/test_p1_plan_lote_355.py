# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-355 · 2026-09-26] DM2/bariátrica: la tortilla INTEGRAL no se cambia por pan.

Corpus de baterías: 18 comidas con «se sustituyó tortilla de trigo refinada (IG alto)» en platos «Wrap Integral de
Lentejas…» / «Tortilla Integral Humedecida…»: la fila casaba «tortilla de trigo» dentro de «tortilla de trigo integral» y
el plato salía «Wrap de Pan Integral»."""
from __future__ import annotations

import pathlib

import condition_rules as cr
import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
BARI = {"medicalConditions": ["Cirugía Bariátrica"]}
DM2 = {"medicalConditions": ["Diabetes tipo 2"]}


def _plan(lineas, nombre="Wrap Integral de Lentejas Guisadas"):
    return {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": nombre, "ingredients": list(lineas), "ingredients_raw": list(lineas),
        "recipe": ["Montaje: rellena la tortilla con las lentejas."]}]}]}


def test_las_dos_reglas_traen_la_fila():
    for perfil in (BARI, DM2):
        fila = [s for s in cr.collect_substitutions(perfil) if "tortilla de trigo" in s["tokens"]]
        assert fila and fila[0]["replacement"] == "Tortilla integral", fila
        assert "integral" in fila[0]["negatives"] and "maiz" in fila[0]["negatives"], fila


def test_la_tortilla_integral_no_se_cambia_por_pan():
    for perfil in (BARI, DM2):
        plan = _plan(["1 tortilla de trigo integral", "120 g de lentejas cocidas"])
        go._apply_condition_substitutions(plan, perfil)
        m = plan["days"][0]["meals"][0]
        assert m["ingredients"][0] == "1 tortilla de trigo integral", (perfil, m["ingredients"])
        assert not any("⚕" in p for p in m["recipe"]), m["recipe"]


def test_la_refinada_pasa_a_tortilla_integral_y_no_a_pan():
    plan = _plan(["1 tortilla de trigo", "120 g de lentejas cocidas"], nombre="Wrap de lentejas")
    go._apply_condition_substitutions(plan, BARI)
    ing = plan["days"][0]["meals"][0]["ingredients"][0].lower()
    assert "tortilla integral" in ing and "pan" not in ing, ing


def test_tortilla_de_maiz_y_harina_de_verdad():
    plan = _plan(["1 tortilla de harina de maíz", "30 g de harina de trigo"], nombre="Panqueque con tortilla")
    go._apply_condition_substitutions(plan, DM2)
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert ings[0] == "1 tortilla de harina de maíz", ings
    assert "avena" in ings[1].lower(), ings                  # la harina de trigo sigue pasando a avena (lote 178)


def test_ancla():
    assert "_NEGATIVAS_DE_FILA.get(label, ())" in (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
