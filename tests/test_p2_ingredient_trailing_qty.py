"""[P2-INGREDIENT-TRAILING-QTY · 2026-09-04] «Nombre: cantidad» es una cantidad, no un nombre.

Swap del dueño (plan 05fb9d22, desayuno del día 1): el LLM devolvió «Cebada: 50 g», «Leche
descremada: 300 ml», «lechosa: 100 g»… Ningún lector lo entendía: el solver de porciones se
abstuvo (0/5 líneas), la guarda de presencia (P2-QTY-PRESENCE-GUARD) antepuso «90 g de» y el
plato se persistió como «90 g de Cebada: 50 g» (633 kcal contra 500 de objetivo; el motor de
paridad tuvo que reapuntar el día). Forma canónica = cantidad líder; se aplica en el validador
de `MealModel` (embudo de generación, swap y autocrítica) y como respaldo en el parser SSOT.
"""
from __future__ import annotations

from pathlib import Path

import nutrition_db as ndb
from schemas import MealModel

_BACKEND = Path(__file__).resolve().parents[1]


def test_canonicalizer_moves_trailing_qty_to_the_front():
    c = ndb.canonicalize_trailing_qty_line
    assert c("Cebada: 50 g") == "50 g de Cebada"
    assert c("Leche descremada: 300 ml") == "300 ml de Leche descremada"
    assert c("lechosa: 100 g") == "100 g de lechosa"
    assert c("Mantequilla de maní: 1 cda") == "1 cda de Mantequilla de maní"
    assert c("huevos: 2") == "2 huevos"
    assert c("Cebada: 50 g (cruda)") == "50 g de Cebada (cruda)"


def test_canonicalizer_leaves_everything_else_alone():
    c = ndb.canonicalize_trailing_qty_line
    for s in ("50 g de Cebada", "2 huevos", "Sal: al gusto", "Cdta de miel", "Mise en place: picar la cebolla", "Pollo"):
        assert c(s) == s, s
    assert c(None) is None


def test_parser_reads_the_trailing_form_as_a_quantity():
    assert ndb._split_qty_unit_name("Cebada: 50 g") == (50.0, "g", "Cebada")
    assert ndb._split_qty_unit_name("Leche descremada: 300 ml") == (300.0, "ml", "Leche descremada")
    qty, _unit, name = ndb._split_qty_unit_name("huevos: 2")
    assert qty == 2.0 and name == "huevos"
    # sin número tras los dos puntos sigue siendo «sin cantidad»
    qty, _unit, _name = ndb._split_qty_unit_name("Sal: al gusto")
    assert qty == 0.0


def test_meal_model_canonicalizes_on_parse():
    m = MealModel(meal="Desayuno", name="x", desc="d", prep_time="5 min", cals=500,
                  ingredients=["Cebada: 50 g", "2 huevos", "Sal: al gusto", "Leche descremada: 300 ml"],
                  recipe=["Mise en place: a"])
    assert m.ingredients == ["50 g de Cebada", "2 huevos", "Sal: al gusto", "300 ml de Leche descremada"]


def test_presence_guard_no_longer_prepends_a_default_to_a_trailing_qty():
    import graph_orchestrator as go

    class _Info:
        protein, carbs, fats = 10.0, 60.0, 2.0

    class _DB:
        def lookup(self, name):
            return _Info()

    meal = {"ingredients": ["Cebada: 50 g", "Pollo"], "ingredients_raw": ["Cebada: 50 g", "Pollo"]}
    if not getattr(go, "QTY_PRESENCE_GUARD_ENABLED", True):
        return
    go._ensure_ingredient_quantities(meal, _DB())
    assert meal["ingredients"][0] in ("Cebada: 50 g", "50 g de Cebada")
    assert not meal["ingredients"][0].startswith("90 g de Cebada")


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P2-INGREDIENT-TRAILING-QTY · 2026-09-04" in app
