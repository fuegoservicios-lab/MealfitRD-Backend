"""[P3-LEAF-VOLUME-CAP · 2026-06-28] El solver inflaba hojas crudas para clavar carbs ('5.5 taza de rúcula (165g)' en
un revoltillo — el usuario lo señaló como incoherente). Cap post-assemble: recorta hojas a ≤LEAF_VOLUME_CAP_G por
comida usando rescale_ingredient_string (escala qty líder + hint '(Ng)' juntos). Hojas ~5 kcal/taza → recorte sin
impacto material en macros.
"""
from __future__ import annotations

import re

import graph_orchestrator as g


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"\((\d+(?:\.\d+)?)\s*g\)", s) or re.search(r"^\s*(\d+(?:\.\d+)?)\s*g\b", s)
        return float(m.group(1)) if m else None


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)


def test_rucula_5cups_capped(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Revoltillo",
             "ingredients": ["5.5 taza de rucula fresca (165g)", "2 huevos enteros"]}]}]
    n = g._cap_leaf_volume_in_meals(days, _DB())
    assert n == 1
    ing0 = days[0]["meals"][0]["ingredients"][0]
    grams = float(re.search(r"\((\d+(?:\.\d+)?)\s*g\)", ing0).group(1))
    assert grams <= g.LEAF_VOLUME_CAP_G, f"rúcula debe capearse a ≤{g.LEAF_VOLUME_CAP_G}g: {ing0}"
    # el huevo (no-hoja) intacto
    assert days[0]["meals"][0]["ingredients"][1] == "2 huevos enteros"


def test_leaf_within_cap_untouched(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Ensalada",
             "ingredients": ["1.5 taza de espinaca (45g)"]}]}]
    n = g._cap_leaf_volume_in_meals(days, _DB())
    assert n == 0  # 45g ≤ cap → no se toca
    assert days[0]["meals"][0]["ingredients"][0] == "1.5 taza de espinaca (45g)"


def test_idempotent(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "X",
             "ingredients": ["8 taza de lechuga (240g)"]}]}]
    g._cap_leaf_volume_in_meals(days, _DB())
    first = days[0]["meals"][0]["ingredients"][0]
    n2 = g._cap_leaf_volume_in_meals(days, _DB())  # segunda pasada
    assert n2 == 0, "ya está al cap → idempotente"
    assert days[0]["meals"][0]["ingredients"][0] == first


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P3-LEAF-VOLUME-CAP" in src
    assert "def _cap_leaf_volume_in_meals" in src
