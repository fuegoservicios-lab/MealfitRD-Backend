"""[P3-CAP-GRAMS-FALLBACK · 2026-06-28] Los caps de porción (DM2 + bariátrico) usaban SOLO
`macros_from_ingredient_string` (requiere que el nombre resuelva al catálogo). Un nombre fuera-de-catálogo
("queso de freír" no resuelto) devolvía macros=None → el cap se SALTABA → "750 g de queso de freír" (¡riesgo de
dumping/sobrecarga, rechazo CRÍTICO en corr=4887e8c1!) pasaba sin recortar. Fix: si el catálogo no resuelve, usar
los GRAMOS LÍDERES del string (`grams_from_ingredient_string`) → atrapa cantidades peligrosas explícitas.
"""
from __future__ import annotations

import re

import graph_orchestrator as g


class _DBNoResolve:
    """macros NO resuelve (nombre fuera de catálogo) pero los gramos líderes sí se parsean."""
    def macros_from_ingredient_string(self, s):
        return {}

    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", str(s))
        return float(m.group(1)) if m else None


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)


def test_bariatric_cheese_cap_catches_unresolved_750g(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Plato",
             "ingredients": ["750 g de queso de freir", "100g de pollo"]}]}]
    n = g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, _DBNoResolve())
    assert n >= 1
    ing0 = days[0]["meals"][0]["ingredients"][0]
    grams = float(re.search(r"(\d+(?:\.\d+)?)\s*g", ing0).group(1))
    assert grams <= g.BARIATRIC_CHEESE_CAP_G, f"el queso 750g debe capearse a ≤{g.BARIATRIC_CHEESE_CAP_G}g: {ing0}"


def test_dm2_starch_cap_catches_unresolved_grams(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "Plato",
             "ingredients": ["600 g de batata exotica no catalogada"]}]}]
    n = g.cap_dm2_high_gi_portions(days, {"medicalConditions": ["Diabetes tipo 2"]}, _DBNoResolve())
    assert n >= 1
    grams = float(re.search(r"(\d+(?:\.\d+)?)\s*g", days[0]["meals"][0]["ingredients"][0]).group(1))
    assert grams <= g.DM2_HIGH_GI_CAP_G


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert src.count("P3-CAP-GRAMS-FALLBACK") >= 2  # en ambos caps (DM2 + bariátrico)
    assert "grams = db.grams_from_ingredient_string(ing)" in src
