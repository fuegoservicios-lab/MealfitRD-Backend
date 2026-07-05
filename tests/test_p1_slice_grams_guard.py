"""[P1-SLICE-GRAMS-GUARD · 2026-07-05] Forensics del "1250 g de queso blanco" (plan 3aa6e58a).

Causa raíz encontrada por SQL + journal: `_recipe_slice_units_to_grams` multiplicaba el número
LÍDER × 25 g/lonja sin verificar que el líder no fuera YA métrico — "50 g de queso blanco en
lonjas finas" → 50×25 = 1250 g (los macros del meal quedaron ciegos a la línea: P41/C85/F21 con
1.25 kg de queso en el texto). Además el conversor solo reescribía `ingredients` — jamás
`ingredients_raw` → divergencia display/raw medida en el mismo plan ("75 g de queso" vs raw
"75g de queso cottage cocido"; la lista de compras lee raw).

Fix: (a) skip de líneas con lead métrico (g/gr/gramos/kg/ml); (b) techo de sanidad 250 g por
conversión (nadie sirve >10 lonjas); (c) raw en lockstep (elemento textualmente idéntico).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_anchored():
    assert "P1-SLICE-GRAMS-GUARD" in _GO
    i = _GO.index("def _recipe_slice_units_to_grams")
    body = _GO[i:i + 5500]
    assert "kg|ml" in body, "skip de lead métrico"
    assert "min(grams, 250)" in body, "techo de sanidad de la conversión"
    assert "ingredients_raw" in body, "raw en lockstep"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "RECIPE_SLICE_GRAMS_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _mk_days(ing, raw=None):
    return [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Wrap de Tostones con Queso Blanco",
        "ingredients": [ing, "150 g de yuca"],
        "ingredients_raw": [raw if raw is not None else ing, "150 g de yuca"],
        "recipe": ["Mise en place: prepara.", "Montaje: sirve."],
    }]}]


def test_metric_lead_never_reconverted(go):
    """La repro del 1250g: lead en GRAMOS que solo MENCIONA lonjas → intocable."""
    days = _mk_days("50 g de queso blanco en lonjas finas")
    assert go._recipe_slice_units_to_grams(days, db=object()) == 0
    assert days[0]["meals"][0]["ingredients"][0] == "50 g de queso blanco en lonjas finas"


def test_legit_slice_conversion_with_raw_lockstep(go):
    days = _mk_days("3 lonjas de queso blanco")
    assert go._recipe_slice_units_to_grams(days, db=object()) == 1
    m = days[0]["meals"][0]
    assert m["ingredients"][0].startswith("75 g de queso"), m["ingredients"][0]
    assert m["ingredients_raw"][0] == m["ingredients"][0], \
        "raw en lockstep (la lista de compras lee raw — antes divergía)"


def test_sanity_cap_at_250(go):
    days = _mk_days("20 lonjas de queso blanco")
    assert go._recipe_slice_units_to_grams(days, db=object()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert line.startswith("250 g de"), f"20×25=500 → techo 250: {line}"


def test_fraction_slice_still_floors(go):
    """Regresión del comportamiento original: '0.5 lonja' → floor 15 g."""
    days = _mk_days("½ lonja de queso blanco")
    assert go._recipe_slice_units_to_grams(days, db=object()) == 1
    assert days[0]["meals"][0]["ingredients"][0].startswith("15 g de queso")
