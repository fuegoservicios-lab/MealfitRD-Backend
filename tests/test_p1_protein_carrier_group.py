# -*- coding: utf-8 -*-
"""[P1-PROTEIN-CARRIER-GROUP · 2026-09-05] Prueba B, día 10: merienda «Casabe con queso blanco, fresas y Yogurt Griego
Entero» entregada con 10 g de queso, sin el yogurt que el cerrador puso (nombre y pasos lo nombran) y 3 g de proteína.
Lácteos y huevo eran «grasas» para el rebalance de banda (por kcal), que los encogía hasta el piso que borra la línea.
Y en el bloque 2 «Tortilla integral» de la Nevera se reconocía como huevo por el alias «tortilla»."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import ai_helpers as ah  # noqa: E402
from constants import strip_accents, DOMINICAN_PROTEINS, DOMINICAN_CARBS  # noqa: E402


class _DB:
    TABLE = {
        "yogurt griego entero": {"protein": 9.0, "carbs": 4.0, "fats": 5.0},
        "queso blanco": {"protein": 20.0, "carbs": 2.0, "fats": 20.0},
        "huevo": {"protein": 13.0, "carbs": 1.0, "fats": 11.0},
        "queso crema": {"protein": 6.0, "carbs": 4.0, "fats": 34.0},
        "mantequilla de mani": {"protein": 25.0, "carbs": 20.0, "fats": 50.0},
        "almendras": {"protein": 21.0, "carbs": 22.0, "fats": 50.0},
        "aceite": {"protein": 0.0, "carbs": 0.0, "fats": 100.0},
        "arroz": {"protein": 7.0, "carbs": 28.0, "fats": 0.3},
    }

    def macros_from_ingredient_string(self, s):
        low = strip_accents(str(s).lower())
        for k, v in self.TABLE.items():
            if k in low:
                return dict(v)
        return None


def test_dairy_and_egg_are_protein_carriers_not_fat_levers(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_CARRIER_GROUP_ENABLED", True)
    db = _DB()
    assert go._ingredient_macro_group("¼ taza de yogurt griego entero", db) == "protein"
    assert go._ingredient_macro_group("40 g de queso blanco", db) == "protein"
    assert go._ingredient_macro_group("2 huevos", db) == "protein"
    # grasas de verdad siguen siendo grasas; carbos siguen siendo carbos
    assert go._ingredient_macro_group("30 g de queso crema", db) == "fats"
    assert go._ingredient_macro_group("1 cda de mantequilla de maní", db) == "fats"
    assert go._ingredient_macro_group("15 g de almendras", db) == "fats"
    assert go._ingredient_macro_group("20g de aceite", db) == "fats"
    assert go._ingredient_macro_group("100g de arroz", db) == "carbs"


def test_knob_off_restores_kcal_dominance(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_CARRIER_GROUP_ENABLED", False)
    assert go._ingredient_macro_group("¼ taza de yogurt griego entero", _DB()) == "fats"


def test_tortilla_integral_in_the_pantry_is_bread_not_egg():
    pool_p = set(DOMINICAN_PROTEINS)
    assert ah._pantry_pick_in_pool(strip_accents("tortilla integral"), DOMINICAN_PROTEINS, ah.protein_synonyms, pool_p) is None
    assert ah._pantry_pick_in_pool(strip_accents("tortillas de maíz"), DOMINICAN_PROTEINS, ah.protein_synonyms, pool_p) is None
    # una tortilla de huevo sí es huevo
    assert ah._pantry_pick_in_pool(strip_accents("tortilla de huevo"), DOMINICAN_PROTEINS, ah.protein_synonyms, pool_p) is not None
    # y como carbo sigue resolviendo
    assert ah._pantry_pick_in_pool(strip_accents("tortilla integral"), DOMINICAN_CARBS, ah.carb_synonyms, set(DOMINICAN_CARBS)) is not None


def test_anchor():
    assert "tooltip-anchor: P1-PROTEIN-CARRIER-GROUP" in (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
