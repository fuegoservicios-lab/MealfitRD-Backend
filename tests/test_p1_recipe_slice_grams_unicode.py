"""[P1-RECIPE-SLICE-GRAMS-UNICODE · 2026-06-28] El usuario vio "1¼ lonjas/pedazos de queso" SOBREVIVIR pese al fix
P1-RECIPE-SLICE-GRAMS. Causa: `_recipe_slice_units_to_grams` llamaba `_LEAD_QTY_RE.match(ing.strip())` SIN normalizar
la fracción unicode "¼" primero → capturaba solo "1" (qty=1.0, no 1.25). Fix: `_normalize_unicode_fractions` antes del
match (mismo patrón que nutrition_db). "1¼ lonjas de queso" → 1.25 → 30g (no 25g).
"""
from __future__ import annotations

import graph_orchestrator as g


class _DB:
    def macros_from_ingredient_string(self, s):
        return {}

    def grams_from_ingredient_string(self, s):
        return None


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)


def test_unicode_mixed_fraction_parsed(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "X",
             "ingredients": ["1¼ lonjas/pedazos de queso"]}]}]
    n = g._recipe_slice_units_to_grams(days, _DB())
    assert n == 1
    out = days[0]["meals"][0]["ingredients"][0]
    # 1.25 × 25g/unidad = 31.25 → redondeo a múltiplo de 5 = 30g (con qty=1 daría 25g)
    assert out == "30 g de queso", out
    assert "¼" not in out and "lonja" not in out


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-RECIPE-SLICE-GRAMS-UNICODE" in src
    assert "_normalize_unicode_fractions as _nuf" in src
    assert "_LQ.match(_nuf(ing.strip()))" in src
