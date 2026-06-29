"""[P1-RECIPE-STEP-VEG-VARIETY-DEDUP · 2026-06-29] El guard _add_missing_recipe_step_vegetables añadía el genérico del
catálogo ("Lechuga") cuando los pasos mencionaban una VARIEDAD ("lechuga romana picada") ya presente en ingredients[],
porque "lechuga romana" no normalizaba igual que "Lechuga" → ingrediente DUPLICADO (visto en vivo). Fix: dedup por TOKEN
base ("lechuga") en el texto crudo de ingredients antes de añadir. Test PURO con catálogo mockeado.
"""
from __future__ import annotations

import re
import unicodedata

import graph_orchestrator as g
import shopping_calculator


def _norm(s):
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"^[\d\s/.,]+\s*(g|gr|kg|mg|ml|l|lb|oz)?\b\s*(de\s+)?", "", s).strip()


_MOCK = [
    {"name": "Lechuga", "category": "Vegetales", "kcal_per_100g": 15.0},
    {"name": "Calabacín", "category": "Vegetales", "kcal_per_100g": 20.0},
]


def _patch(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: _MOCK)
    monkeypatch.setattr(shopping_calculator, "normalize_name", _norm)


def test_lechuga_variety_not_duplicated(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "Wrap", "ingredients": ["1.5 taza de lechuga romana picada"],
                        "recipe": ["Agrega la lechuga romana picada y enrolla."]}]}]
    n = g._add_missing_recipe_step_vegetables(days)
    assert n == 0  # NO duplica la lechuga
    ings = days[0]["meals"][0]["ingredients"]
    assert sum(1 for i in ings if "lechuga" in i.lower()) == 1  # sigue habiendo UNA sola lechuga


def test_genuinely_missing_veg_still_added(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "Revoltillo", "ingredients": ["2 huevos"],
                        "recipe": ["Ralla el calabacín y saltéalo."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 1  # calabacín ausente SÍ se añade


def test_knob_off_reverts(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(g, "RECIPE_STEP_VEG_VARIETY_DEDUP", False)
    days = [{"meals": [{"name": "Wrap", "ingredients": ["1.5 taza de lechuga romana picada"],
                        "recipe": ["Agrega la lechuga romana picada."]}]}]
    # con el dedup off, el bug reaparece (añade el genérico) — confirma que el knob controla el fix
    assert g._add_missing_recipe_step_vegetables(days) == 1


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-RECIPE-STEP-VEG-VARIETY-DEDUP" in src
    assert "RECIPE_STEP_VEG_VARIETY_DEDUP" in src
    assert g.RECIPE_STEP_VEG_VARIETY_DEDUP is True
