"""[P2-1-CONDIMENT-PORTION-SANITY · 2026-07-10] (recipe plausibility roadmap, item P2-1) Evidencia
visual (plan 564d6e4e): "7 dientes de ajo" para UNA porción. Los caps P6 (`P6-SPICE-CAP` etc.) protegen
la lista de compras AGREGADA (semanal/mensual) — el plato INDIVIDUAL no tiene tope. Cap conservador
por-porción: ajo ≤3 dientes, cebolla ≤1 unidad — espejo del patrón `CITRUS_MEAL_CAP_UNITS` (P1-FINALIZE-
COUNTABLE-POLISH), display+raw, clamp del lead numérico preservando el resto de la línea.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_present():
    assert "P2-1-CONDIMENT-PORTION-SANITY" in _GO


def test_function_defined():
    assert "def cap_condiments_per_portion(plan_data" in _GO


def _plan(ingredients):
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Pollo guisado",
                                            "ingredients": list(ingredients),
                                            "ingredients_raw": list(ingredients)}]}]}


def test_caps_excessive_garlic():
    import graph_orchestrator as g
    pd = _plan(["7 dientes de ajo", "200 g de pollo"])
    n = g.cap_condiments_per_portion(pd)
    assert n >= 1
    line = pd["days"][0]["meals"][0]["ingredients"][0]
    assert "7 dientes" not in line
    assert "3 dientes de ajo" in line


def test_caps_excessive_onion():
    import graph_orchestrator as g
    pd = _plan(["2 cebollas medianas", "200 g de pollo"])
    n = g.cap_condiments_per_portion(pd)
    assert n >= 1
    line = pd["days"][0]["meals"][0]["ingredients"][0]
    assert line.startswith("1 cebolla")


def test_noop_within_cap():
    import graph_orchestrator as g
    pd = _plan(["2 dientes de ajo", "1 cebolla mediana", "200 g de pollo"])
    before = list(pd["days"][0]["meals"][0]["ingredients"])
    n = g.cap_condiments_per_portion(pd)
    assert n == 0
    assert pd["days"][0]["meals"][0]["ingredients"] == before


def test_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.cap_condiments_per_portion({}) == 0
    assert g.cap_condiments_per_portion(None) == 0
