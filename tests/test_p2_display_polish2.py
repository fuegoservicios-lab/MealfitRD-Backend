"""[P2-CDAS-TO-CUPS + P2-INNER-ASCII-FRAC + P2-AROMATIC-MICRO-BUMP · 2026-07-05] Pulido de
display del review visual del plan 7e4e5570:
- "11 cdas de harina de trigo (77g)" → "⅔ taza de harina de trigo (77g)" (16 cdas = 1 taza,
  conversión de volumen exacta sin densidad; ≥6 cdas se promueve).
- "Jugo de 1/2 limón" / "de 1/4 limón" → "Jugo de ½ limón" (el prettify interno solo cubría
  decimales, no fracciones ASCII).
- "1 cdta de cebolla picada (2g)" → "2 cdas de cebolla picada (20g)" (micro-aromático que el
  piso no veía — solo cubre leads en gramos).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


def test_eleven_spoons_to_cups():
    out = _pretty("11 cdas de harina de trigo (77g)")
    assert out.startswith("⅔ taza"), f"11/16 cdas ≈ ⅔ taza: {out}"
    assert "(77g)" in out, "el hint de gramos se preserva"


def test_eight_spoons_to_half_cup():
    assert _pretty("8 cdas de avena (56g)").startswith("½ taza")


def test_few_spoons_untouched():
    assert _pretty("3½ cdas de leche descremada (52ml)") == "3½ cdas de leche descremada (52ml)"
    assert _pretty("2 cdas de aceite de oliva") == "2 cdas de aceite de oliva"


def test_inner_ascii_fractions():
    assert _pretty("Jugo de 1/2 limón") == "Jugo de ½ limón"
    assert _pretty("Jugo de 1/4 limón") == "Jugo de ¼ limón"


# ───────────────────── aromatic micro bump ─────────────────────

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


def test_micro_onion_bumped(go):
    days = _mk_days("1 cdta de cebolla picada (2g)")
    assert go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB()) >= 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert line.startswith("2 cdas") and "(20g)" in line, line
    assert days[0]["meals"][0]["ingredients_raw"][0] == line, "raw en lockstep"


def test_onion_without_tiny_hint_untouched(go):
    days = _mk_days("1 cdta de cebolla en polvo", "1 cebolla picada (110g)")
    go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB())
    assert "1 cdta de cebolla en polvo" in days[0]["meals"][0]["ingredients"], \
        "sin hint ≤5g no hay evidencia de micro-línea → intacta"
    assert "1 cebolla picada (110g)" in days[0]["meals"][0]["ingredients"]
