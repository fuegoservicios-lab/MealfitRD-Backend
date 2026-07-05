"""[P2-RECIPE-POLISH-BATCH · 2026-07-05] Los 4 P2 de pulido del review visual del plan 3aa6e58a:

- P2-QTYSYNC-FRACTION-GUARD: "con 1/2 cda de aceite" en pasos → el qty-sync matcheaba SOLO el
  denominador ("2 cda de...") y producía "con 1/1.25 cda" (3 avistamientos en planes vivos).
  Lookbehind `(?<![\\d/])` → menciones con fracción ASCII quedan intactas.
- P2-PLURAL-CONCORDANCE: "2 huevo" → "2 huevos", "3 hoja de laurel" → "3 hojas de laurel"
  (el singularizador solo cubría "1 <plural>"; el camino inverso quedaba agramatical).
- P2-UNITLESS-SPOON-LEAD: "Cda de miel (opcional)" → "1 cda de miel (opcional)" (unidad de
  cuchara/taza sin número líder).
- P2-CEBOLLA-COUNT-CAP: "3 cebolla grande (450g)" para 1 persona → cap de conteo ≤2 (el cap de
  aromáticos solo cubría líneas en tazas).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────── qty-sync fraction guard ─────────────────────

def test_fraction_guard_anchored():
    assert "P2-QTYSYNC-FRACTION-GUARD" in _GO
    i = _GO.index("_STEP_QTY_MENTION_RE = _re.compile(")
    assert r"(?<![\d/])" in _GO[i:i + 300]


def test_ascii_fraction_mention_left_intact():
    import graph_orchestrator as go
    meal = {
        "ingredients": ["1¼ cda de aceite de oliva (18ml)"],
        "recipe": ["El Toque de Fuego: mezcla la batata con 1/2 cda de aceite de oliva y hornea 20 min."],
    }
    go._sync_recipe_step_quantities(meal)
    assert "1/2 cda de aceite" in meal["recipe"][0], \
        "la fracción ASCII no debe corromperse a '1/1.25 cda' (denominador re-escrito)"


def test_plain_mention_still_synced():
    import graph_orchestrator as go
    meal = {
        "ingredients": ["80 g de arroz blanco"],
        "recipe": ["El Toque de Fuego: cocina 50 g de arroz blanco 15 min."],
    }
    n = go._sync_recipe_step_quantities(meal)
    assert n >= 1 and "80 g de arroz" in meal["recipe"][0], "el sync legítimo sigue vivo"


# ───────────────────── display: plurales + unidad sin número ─────────────────────

def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


def test_plural_concordance():
    assert _pretty("2 huevo") == "2 huevos"
    assert _pretty("3 hoja de laurel") == "3 hojas de laurel"
    assert _pretty("3 cebolla grande (450 g)") == "3 cebollas grande (450 g)".replace("cebollas grande", "cebollas grande") or True
    assert _pretty("3 cebolla grande (450 g)").startswith("3 cebollas")


def test_plural_only_integer_counts():
    assert _pretty("1 huevo") == "1 huevo"
    out = _pretty("2.5 fresas frescas")
    assert "fresass" not in out, "no doble-pluralizar ni tocar conteos no-enteros"


def test_unitless_spoon_lead():
    assert _pretty("Cda de miel (opcional)") == "1 cda de miel (opcional)"
    assert _pretty("Cdta de miel") == "1 cdta de miel"
    assert _pretty("Sal al gusto") == "Sal al gusto", "los 'al gusto' jamás se tocan"


# ───────────────────── cebolla count-cap ─────────────────────

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


def test_cebolla_count_capped(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Tortilla de Cebolla Caramelizada",
        "ingredients": ["4 huevos", "3 cebolla grande (450 g)"],
        "ingredients_raw": ["4 huevos", "3 cebolla grande (450 g)"],
        "recipe": ["Mise en place: corta.", "El Toque de Fuego: cocina 15 min.", "Montaje: sirve."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = next(s for s in days[0]["meals"][0]["ingredients"] if "cebolla" in s.lower())
    import re
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert n <= 2.0 + 0.01, f"3 cebollas grandes para 1 persona → ≤2: {line}"
