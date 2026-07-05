"""[P2-OIL-MICRO-BUMP · 2026-07-05] "¼ cdta de aceite de oliva (1ml)" — medido ×2 en el plan
vivo 23c958bb. Nadie mide 1 ml de aceite; la exención de condimentos del shrink-floor (aceite en
`_SHRINK_FLOOR_EXEMPT_TOKENS` — "5g de aceite = 1 cdta REAL") dejaba pasar intactos los leads
POR DEBAJO de esa cucharadita mínima. Bump determinista a "1 cdta ... (5ml)" dentro de
`_floor_subservible_portions` (todas las superficies que ya llaman el shrink-floor lo heredan).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_and_regex():
    assert "P2-OIL-MICRO-BUMP" in _GO
    i = _GO.index("def _floor_subservible_portions")
    assert "_MICRO_OIL_LEAD_RE" in _GO[i:i + 4500], "el bump vive dentro del shrink-floor (todas las superficies)"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "PORTION_SHRINK_FLOOR_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _mk_days(*ings):
    return [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Plato",
        "ingredients": list(ings), "ingredients_raw": list(ings),
        "recipe": ["Mise en place: prepara.", "Montaje: sirve."],
    }]}]


def test_quarter_tsp_oil_bumped_to_one_tsp(go):
    days = _mk_days("¼ cdta de aceite de oliva (1ml)", "Sal al gusto")
    assert go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB()) >= 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert line.startswith("1 cdta"), f"lead al mínimo servible: {line}"
    assert "(5ml)" in line
    assert days[0]["meals"][0]["ingredients_raw"][0] == line, "raw en lockstep"


def test_half_tsp_and_decimal_leads_also_bumped(go):
    days = _mk_days("½ cdta de aceite de oliva", "0.25 cdta de aceite de oliva (1ml)")
    assert go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB()) >= 2
    for line in days[0]["meals"][0]["ingredients"]:
        assert line.startswith("1 cdta"), line


def test_full_spoon_oil_untouched(go):
    days = _mk_days("1 cda de aceite de oliva (15ml)", "1 cdta de aceite de oliva (5ml)")
    assert go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB()) == 0


def test_non_oil_fraction_untouched(go):
    days = _mk_days("¼ cdta de miel (2g)")
    go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB())
    assert "¼ cdta de miel (2g)" in days[0]["meals"][0]["ingredients"], \
        "el bump es SOLO de aceite (miel ¼ cdta es medible y legítima)"
