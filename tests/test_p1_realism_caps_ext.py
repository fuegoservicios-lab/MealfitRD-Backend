"""[P1-REALISM-CAPS-EXT · 2026-07-05] Extensión de `_cap_unrealistic_portions` (screenshots del
plan vivo 23c958bb):

- "36.5 tomates cherry (365g)" en un casabe — el sustantivo del conteo captura solo "tomate"
  (cap 3 sería muy poco para cherry) → compuestos ANTES del genérico (cherry ≤10 unidades).
- "3¾ taza de melón en cubos (562g)" en UN batido — frutas ACUOSAS de volumen sin techo; además
  el lead con fracción unicode ("3¾") era invisible para el regex decimal de tazas.
- Conteos nuevos: tomate ≤3, uva ≤20, aceituna ≤12, fresa ≤10, limón ≤3.

Misma filosofía del cap original ("505g de calamar"): la banda perfecta no justifica un plato
no-servible; el déficit lo redistribuyen los closers aguas abajo.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_and_knob():
    assert "P1-REALISM-CAPS-EXT" in _GO
    assert '_env_int("MEALFIT_REALISM_FRUIT_VOLUME_CAP_G", 300' in _GO


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _mk_days(*ings):
    return [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Plato",
        "ingredients": list(ings), "ingredients_raw": list(ings),
        "recipe": ["Mise en place: prepara.", "Montaje: sirve."],
    }]}]


def _lead(days, token):
    line = next(s for s in days[0]["meals"][0]["ingredients"] if token in s.lower())
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)", line)
    return float(m.group(1).replace(",", ".")) if m else None, line


def test_cherry_count_capped_via_compound(go):
    days = _mk_days("36.5 tomates cherry (365g)")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    n, line = _lead(days, "cherry")
    assert n is not None and n <= 10.0 + 0.01, f"36.5 cherries → ≤10: {line}"


def test_whole_tomato_generic_cap_untouched_at_three(go):
    days = _mk_days("3 tomate picado (299g)")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 0
    assert "3 tomate picado (299g)" in days[0]["meals"][0]["ingredients"]


def test_melon_cup_lead_with_unicode_fraction_capped(go):
    days = _mk_days("3¾ taza de melón en cubos (562g)")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert "562" not in line, f"el hint de gramos debe re-escalarse con el lead: {line}"
    m = re.search(r"\((\d+(?:[.,]\d+)?)\s*g\)", line)
    assert m and float(m.group(1).replace(",", ".")) <= 310.0, f"melón ≤~300g por comida: {line}"


def test_melon_gram_lead_capped(go):
    days = _mk_days("562g de melón en cubos")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    n, line = _lead(days, "melon" if "melon" in days[0]["meals"][0]["ingredients"][0].lower() else "melón")
    assert n is not None and n <= 310.0, f"562g de melón → ≤300g: {line}"


def test_grapes_within_cap_untouched(go):
    days = _mk_days("18 uvas")
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 0


def test_lockstep_raw_synced(go):
    days = _mk_days("36.5 tomates cherry (365g)")
    go._cap_unrealistic_portions(days, db=_NoopDB())
    meal = days[0]["meals"][0]
    assert meal["ingredients"] == meal["ingredients_raw"], "raw en lockstep (lista de compras coherente)"
