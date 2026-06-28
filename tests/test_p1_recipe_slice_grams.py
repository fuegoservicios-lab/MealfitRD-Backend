"""[P1-RECIPE-SLICE-GRAMS · 2026-06-27] El usuario señaló "0.5 lonja/pedazo de queso ¿qué significa eso?" — una
unidad VAGA (lonja/pedazo) en fracción no comunica una cantidad medible. Fix: convertir queso/embutido en unidades
de tajada/loncha/pedazo a GRAMOS ('15 g de queso'). Default 25g/unidad, floor 15g, redondeo a múltiplo de 5.
"""
from __future__ import annotations

import graph_orchestrator as g


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)


def _day(ings):
    return [{"day": 0, "meals": [{"meal": "Desayuno", "name": "X", "ingredients": list(ings)}]}]


def test_cheese_vague_unit_to_grams(monkeypatch):
    _wire(monkeypatch)
    days = _day(["0.5 lonja/pedazo de queso"])
    n = g._recipe_slice_units_to_grams(days, db=object())
    assert n == 1
    assert days[0]["meals"][0]["ingredients"][0] == "15 g de queso"  # floor 15g


def test_preserves_cheese_descriptor(monkeypatch):
    _wire(monkeypatch)
    days = _day(["1 lonja de queso cottage (25g)"])
    g._recipe_slice_units_to_grams(days, db=object())
    assert days[0]["meals"][0]["ingredients"][0] == "25 g de queso cottage"


def test_deli_meat_slices_to_grams(monkeypatch):
    _wire(monkeypatch)
    days = _day(["2 pedazos de jamon de pavo"])
    g._recipe_slice_units_to_grams(days, db=object())
    assert days[0]["meals"][0]["ingredients"][0] == "50 g de jamon de pavo"


def test_leaves_halvable_fruit_with_grams_alone(monkeypatch):
    """Medio guineo CON gramos es coherente — no se toca (no es unidad vaga de queso/embutido)."""
    _wire(monkeypatch)
    days = _day(["0.5 guineo maduro mediano (41g)"])
    assert g._recipe_slice_units_to_grams(days, db=object()) == 0
    assert "guineo" in days[0]["meals"][0]["ingredients"][0]


def test_leaves_non_slice_food_alone(monkeypatch):
    """'pedazo de pollo' NO se convierte (pollo no es alimento lonjeable de la lista)."""
    _wire(monkeypatch)
    days = _day(["1 pedazo de pollo a la plancha (120g)"])
    assert g._recipe_slice_units_to_grams(days, db=object()) == 0


def test_knob_off(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "RECIPE_SLICE_GRAMS_ENABLED", False)
    days = _day(["0.5 lonja/pedazo de queso"])
    assert g._recipe_slice_units_to_grams(days, db=object()) == 0


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-RECIPE-SLICE-GRAMS" in src and "def _recipe_slice_units_to_grams" in src
    assert "RECIPE_SLICE_GRAMS_ENABLED" in src
