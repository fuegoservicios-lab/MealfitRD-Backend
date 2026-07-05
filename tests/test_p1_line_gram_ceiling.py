"""[P1-LINE-GRAM-CEILING · 2026-07-05] Techo DURO genérico por-línea en gramos.

El "1250 g de queso blanco" (plan 3aa6e58a) evadió el cap de proteína (≤300g, existente desde
2026-07-01) porque la clasificación por grupo no aplicó a esa línea — todos los caps previos
dependían de resolver el grupo/token del alimento. Este techo NO depende de nada: ninguna línea
individual de comida supera LINE_GRAM_HARD_CAP (600 g — nadie sirve más de eso de un solo
alimento a 1 persona). Exentos: agua/caldo. La clase 47.5-tallos/1250-g muere por construcción.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_default_600():
    assert '_env_int("MEALFIT_LINE_GRAM_HARD_CAP", 600' in _GO


def test_hard_cap_checked_before_group_caps():
    i = _GO.index("0) techo DURO genérico")
    win = _GO[i:i + 700]
    assert "LINE_GRAM_HARD_CAP" in win
    i_prot = _GO.index("cur_g > PORTION_CAP_PROTEIN_G", i)
    assert i < i_prot, "el techo duro corre ANTES de los caps por clase (no depende del grupo)"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _NoopDB:
    """Simula el lookup fallido que dejó pasar el 1250g (grupo irresoluble)."""

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


def _lead_g(days, idx=0):
    line = days[0]["meals"][0]["ingredients"][idx]
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)", line)
    return float(m.group(1).replace(",", ".")), line


def test_monster_line_capped_even_without_group(go):
    """La repro exacta: 1250 g de queso con DB que no resuelve → ≤600 g."""
    days = _mk_days("1250 g de queso blanco")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    g, line = _lead_g(days)
    assert g <= 600.0 + 0.5, f"el techo duro no depende del grupo: {line}"


def test_water_and_broth_exempt(go):
    days = _mk_days("800 g de caldo de pollo", "1000 g de agua")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 0


def test_normal_lines_untouched(go):
    days = _mk_days("300 g de arroz blanco", "150 g de yuca")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 0
