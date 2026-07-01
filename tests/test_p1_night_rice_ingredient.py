"""[P1-NIGHT-RICE-INGREDIENT + P1-NIGHT-RICE-COMPOUND-FINAL · 2026-07-01] (audit slots GAP-1/GAP-2)

Dos rutas restantes del "arroz de noche":
  (a) Los 3 detectores de slot son name-only → cena «Bowl criollo de pollo» con "180 g de arroz
      blanco" en ingredients[] pasaba las 4 superficies. Fix: pase INGREDIENT-DRIVEN del autofix
      (≥ MIN_G, sin renombrar; hereda a updates vía el finalizer).
  (b) Moro/locrio/chofán en cena degradaba a advisory en el intento final y SE ENTREGABA. Fix:
      modo `compound=True` (solo desde la rama advisory-final del gate) → «Guiso de X» + tubérculo.
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+)\s*g", s)
        return float(m.group(1)) if m else 150.0


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    monkeypatch.setattr(g, "NIGHT_RICE_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(g, "NIGHT_RICE_INGREDIENT_PASS", True)
    monkeypatch.setattr(g, "NIGHT_RICE_COMPOUND_FINAL", True)


def _cena(name, ings, day=0):
    return [{"day": day, "meals": [{"meal": "Cena", "name": name, "ingredients": ings,
                                    "recipe": ["El Toque de Fuego: cocina el arroz 15 min a fuego medio."]}]}]


# ---------------------------------------------------------------------------
# (a) pase ingredient-driven
# ---------------------------------------------------------------------------
def test_hidden_rice_in_ingredients_is_replaced(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Bowl criollo de pollo y vegetales", ["120 g de Pollo", "180 g de arroz blanco"])
    n = g._night_rice_autofix(days, db=_DB())
    meal = days[0]["meals"][0]
    assert n == 1
    assert not any(re.search(r"\barroz\b", i.lower()) for i in meal["ingredients"]), \
        "el arroz escondido en ingredients[] debe sustituirse por el tubérculo"
    assert any("batata" in i.lower() for i in meal["ingredients"])  # día 0 → Batata
    assert meal["name"] == "Bowl criollo de pollo y vegetales", "el nombre NO se toca (no delata arroz)"
    assert not any("arroz" in str(s).lower() for s in meal["recipe"]), "los pasos también se reescriben"


def test_small_garnish_below_min_g_untouched(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "NIGHT_RICE_INGREDIENT_MIN_G", 60)
    days = _cena("Ensalada de pollo", ["120 g de Pollo", "30 g de arroz blanco"])
    assert g._night_rice_autofix(days, db=_DB()) == 0, \
        "una guarnición tangencial (< MIN_G) no debe tocarse en el pase ingredient-driven"


def test_ingredient_pass_knob_off_restores_old_behavior(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "NIGHT_RICE_INGREDIENT_PASS", False)
    days = _cena("Bowl criollo de pollo", ["180 g de arroz blanco"])
    assert g._night_rice_autofix(days, db=_DB()) == 0, "rollback: sin el knob, name-only como antes"


def test_flour_exclusion_still_respected(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Cena ligera", ["80 g de harina de arroz"])
    assert g._night_rice_autofix(days, db=_DB()) == 0


# ---------------------------------------------------------------------------
# (b) modo compound (solo intento final)
# ---------------------------------------------------------------------------
def test_compound_untouched_in_normal_flow(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Moro de gandules con pollo", ["200 g de arroz blanco", "100 g de gandules"])
    assert g._night_rice_autofix(days, db=_DB()) == 0, "sin compound=True, moro/locrio se dejan al gate"
    assert "moro" in days[0]["meals"][0]["name"].lower()


def test_compound_final_converts_to_guiso(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Moro de gandules con pollo", ["200 g de arroz blanco", "100 g de gandules"])
    n = g._night_rice_autofix(days, db=_DB(), compound=True)
    meal = days[0]["meals"][0]
    assert n == 1
    assert "moro" not in meal["name"].lower(), f"nombre sigue diciendo moro: {meal['name']}"
    assert "guiso" in meal["name"].lower(), f"esperado 'Guiso de …': {meal['name']}"
    assert not any(re.search(r"\barroz\b", i.lower()) for i in meal["ingredients"])
    assert any("batata" in i.lower() for i in meal["ingredients"])


def test_compound_knob_off_no_conversion(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "NIGHT_RICE_COMPOUND_FINAL", False)
    days = _cena("Locrio de pollo", ["200 g de arroz blanco"])
    assert g._night_rice_autofix(days, db=_DB(), compound=True) == 0


# ---------------------------------------------------------------------------
# wiring estructural
# ---------------------------------------------------------------------------
def test_review_gate_wires_compound_in_advisory_final():
    i = _GRAPH.find("(audit slots GAP-2) Antes de degradar")
    assert i != -1, "la rama advisory-final del gate de slot no intenta el autofix compuesto"
    seg = _GRAPH[i:i + 1200]
    assert "compound=True" in seg and "_detect_slot_appropriateness" in seg, \
        "el autofix compuesto debe correr y RE-DETECTAR antes de degradar a advisory"


def test_knob_defaults():
    assert g.NIGHT_RICE_INGREDIENT_PASS is True
    assert g.NIGHT_RICE_COMPOUND_FINAL is True
    assert 20 <= g.NIGHT_RICE_INGREDIENT_MIN_G <= 200
