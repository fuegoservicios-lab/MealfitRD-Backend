"""[P3-2-INGREDIENT-COUNT-AGREEMENT · 2026-07-10] (recipe plausibility roadmap, item P3-2) Evidencia
visual (plan 564d6e4e): línea de ingrediente "1 naranjas" (conteo=1 con sustantivo en PLURAL). El pase
existente `_QTYSYNC_COUNT_NOUNS` (P2-QTYSYNC-COUNT-NOUNS) ya resuelve esta clase de error pero SOLO
sobre los PASOS de la receta ("Bate los huevos"→"Bate el huevo") — nunca sobre la línea de `ingredients`
misma. Fix: extiende la tabla curada con "naranja" + aplica un pase espejo sobre las líneas de
ingredientes (display+raw).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_present():
    assert "P3-2-INGREDIENT-COUNT-AGREEMENT" in _GO


def test_naranja_added_to_shared_count_noun_table():
    import graph_orchestrator as g
    assert "naranja" in g._QTYSYNC_COUNT_NOUNS


def test_function_defined():
    assert "def fix_ingredient_count_agreement(plan_data" in _GO


def _plan(ing_line):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "X",
                                            "ingredients": [ing_line], "ingredients_raw": [ing_line]}]}]}


def test_fixes_singular_count_with_plural_noun():
    import graph_orchestrator as g
    pd = _plan("1 naranjas")
    n = g.fix_ingredient_count_agreement(pd)
    assert n >= 1
    assert pd["days"][0]["meals"][0]["ingredients"][0] == "1 naranja"
    assert pd["days"][0]["meals"][0]["ingredients_raw"][0] == "1 naranja"


def test_noop_when_plural_count_matches_plural_noun():
    import graph_orchestrator as g
    pd = _plan("3 naranjas medianas")
    before = pd["days"][0]["meals"][0]["ingredients"][0]
    g.fix_ingredient_count_agreement(pd)
    assert pd["days"][0]["meals"][0]["ingredients"][0] == before


def test_noop_on_unrelated_line():
    import graph_orchestrator as g
    pd = _plan("200 g de pollo")
    assert g.fix_ingredient_count_agreement(pd) == 0


def test_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.fix_ingredient_count_agreement({}) == 0
    assert g.fix_ingredient_count_agreement(None) == 0
