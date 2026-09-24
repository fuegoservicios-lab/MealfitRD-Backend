# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-196 · 2026-09-24] Sin lácteos, el huevo es un básico: se repite el mismo día sólo con otra técnica.

4 de 4 corridas de «alergia a lácteos y mariscos» perdieron un intento por «MISMA PROTEÍNA REPETIDA EL MISMO DÍA»: el
huevo del desayuno (revoltillo) y el de la merienda (huevo cocido, lo único denso que el cerrador puede poner sin
lácteos). La exención «básico + técnica distinta» (P1-STAPLE-FOODS, Decisión B del dueño) ya existía; faltaba que el
huevo contara como básico para quien no puede tomar lácteos, y que «cocido» fuera una técnica reconocible."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_LACTEOS = {"allergies": ["Lacteos", "Mariscos"], "dietType": "balanced"}


def _dia(merienda_nombre="Manzana con crema de maní y huevo cocido", merienda_pasos=None):
    return {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Revoltillo criollo con casabe y aguacate",
         "ingredients": ["3 huevos", "1 torta de casabe", "½ aguacate"], "recipe": ["Bate los huevos y revuélvelos."]},
        {"meal": "Merienda", "name": merienda_nombre, "ingredients": ["1 manzana", "60 g de huevo cocido"],
         "recipe": merienda_pasos or ["Corta la manzana y sirve con el huevo."]},
        {"meal": "Almuerzo", "name": "Pollo guisado con arroz", "ingredients": ["150 g de pechuga de pollo"],
         "recipe": ["Guisa el pollo."]},
    ]}]}


@pytest.mark.parametrize("form, esperado", [
    (_LACTEOS, {"huevo"}),
    ({"allergies": ["Intolerancia a la lactosa"]}, {"huevo"}),
    ({"allergies": ["Lacteos", "Huevo"]}, set()),
    ({"allergies": ["Lacteos"], "dietType": "vegan"}, set()),
    ({"allergies": ["Ninguna"]}, set()),
])
def test_quien_no_toma_lacteos_tiene_el_huevo_como_basico(form, esperado):
    import graph_orchestrator as go
    assert go._user_staple_labels(form) == esperado


def test_revoltillo_y_huevo_cocido_no_son_repeticion_sin_lacteos():
    import graph_orchestrator as go
    plan = _dia()
    assert go._days_with_same_day_protein_repeat(plan, user_staples=go._user_staple_labels(_LACTEOS)) == []
    assert go._days_with_same_day_protein_repeat(plan, user_staples=go._user_staple_labels({})) == [1], \
        "sin alergia a lácteos la regla de siempre"


def test_la_misma_tecnica_sigue_siendo_repeticion():
    import graph_orchestrator as go
    plan = _dia("Revoltillo de huevo con tomate", ["Bate el huevo y revuélvelo con el tomate."])
    assert go._days_with_same_day_protein_repeat(plan, user_staples=go._user_staple_labels(_LACTEOS)) == [1]


def test_knob_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_EGG_STAPLE_WITHOUT_DAIRY", "false")
    import graph_orchestrator as go
    assert go._user_staple_labels(_LACTEOS) == set()


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 196 and m.group(2) >= "2026-09-24"
