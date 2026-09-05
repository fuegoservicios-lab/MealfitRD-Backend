# -*- coding: utf-8 -*-
"""[P1-YOGURT-MEAL-CAP · 2026-09-05] Plan vivo 0b0250ab (vegetariano, ganancia muscular, mercado ES): yogurt griego como
relleno de proteína en casi todo — 1½ tazas en un almuerzo, 2 tazas (≈500 g) en una merienda — mientras dos comidas
fuertes quedaban en 22-24 g. El queso tenía tope por comida; el yogurt no. Y en comida fuerte NO caliente (ensalada
tibia) el cerrador no relegaba el lácteo dulce."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _days(*ings):
    return [{"day": 1, "meals": [{"meal": "Merienda", "name": "Plato", "ingredients": list(ings), "ingredients_raw": list(ings),
                                  "recipe": ["Mise en place: prepara.", "Montaje: sirve."]}]}]


def _lead(days, tok):
    line = next(s for s in days[0]["meals"][0]["ingredients"] if tok in s.lower())
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?|[½¼¾⅓⅔])", line)
    return line


def test_yogurt_capped_in_cups_and_grams(monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    d = _days("2 tazas de yogurt griego entero")
    assert go._cap_unrealistic_portions(d, db=_NoopDB()) >= 1
    line = _lead(d, "yogur")
    assert "2 tazas" not in line and line.lstrip().startswith("1 taza"), line
    d2 = _days("450 g de yogurt griego sin azúcar")
    assert go._cap_unrealistic_portions(d2, db=_NoopDB()) >= 1
    g = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", _lead(d2, "yogur")).group(1).replace(",", "."))
    assert g <= go.YOGURT_MEAL_CAP_G, _lead(d2, "yogur")
    # dentro del tope, intacto
    d3 = _days("½ taza de yogurt griego entero", "250 g de yogurt griego")
    assert go._cap_unrealistic_portions(d3, db=_NoopDB()) == 0


class _Info:
    def __init__(self, name, protein, kcal, carbs=4.0, fats=3.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


def test_main_meal_salad_prefers_legume_over_yogurt(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [(2.0, "Yogurt griego", _Info("Yogurt griego", 10.0, 97.0)),
                  (1.0, "Garbanzos cocidos", _Info("Garbanzos cocidos", 9.0, 164.0, carbs=27.0, fats=2.6))]
    meal = {"meal": "Almuerzo", "name": "Ensalada tibia de queso blanco, piña y lechosa con tostadas",
            "protein": 12, "carbs": 40, "fats": 10, "cals": 300,
            "ingredients": ["75 g de lechuga", "30 g de queso blanco", "2 tostadas"],
            "ingredients_raw": ["75 g de lechuga", "30 g de queso blanco", "2 tostadas"],
            "recipe": ["Mise en place: corta.", "Montaje: mezcla y sirve."]}
    g = go._close_protein_gap_for_meal(meal, 40.0, None, candidates, enforce_min_threshold=False, diet="vegetarian")
    assert g > 0
    blob = " ".join(meal["ingredients"]).lower()
    assert "garbanzos" in blob and "yogur" not in blob, meal["ingredients"]


def test_anchor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-YOGURT-MEAL-CAP" in src and "_REALISM_CUP_CAPS = ((_REALISM_YOGURT_TOKENS, YOGURT_MEAL_CAP_CUPS)," in src
