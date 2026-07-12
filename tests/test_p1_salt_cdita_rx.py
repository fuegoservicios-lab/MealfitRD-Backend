"""[P1-SALT-CDITA-RX · 2026-07-12] El converter de sal cuantificada no conocía "cdita".

Forense del plan vivo df263d1b (banner "1 de 3 días se pasa del techo de sodio", día-1
medido 3,274mg): el día tenía DOS líneas "0.5 cdita de Sal" / "0.5 cdita Sal" (~1,179mg
c/u — 'Sal' resuelve vía density_g_per_cup=292) y `_day_sodium_autofix` devolvía 0
acciones. El peldaño (1.5) P1-SALT-LINE-AUTOFIX existía desde 2026-07-05 pero su regex
solo cubría `cdta` — el day-generator y el finalizador de swaps escriben `cdita`, y el
finalizador además omite el "de" ("½ cdita Sal"). Dos gaps de spelling = corrector muerto.

Fix: `_SALT_QTY_RX` (+ regex espejo de pasos) gana `cd(?:i)?tas?`, mixto "1½", y "de"
opcional. Este test replica las DOS formas vivas exactas + guarda el no-match de 'salsa'.
tooltip-anchor: P1-SALT-CDITA-RX
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _SaltDB:
    """Toda línea de sal CUANTIFICADA = 2600mg (>techo del día por sí sola — el gate del
    autofix exige día>2000mg antes de convertir); al gusto = 0; resto 20."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "sal" in low and "salsa" not in low:
            if "al gusto" in low or "pizca" in low:
                return {"sodium_mg": 0.0}
            return {"sodium_mg": 2600.0}
        return {"sodium_mg": 20.0}


def _mk_live_day():
    """Réplica del día-1 vivo: dos meals, cada una con su forma de línea de sal."""
    return [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Panqueques de Harina de Trigo",
         "ingredients": ["0.5 cdita de Sal", "31 g de Harina de trigo"],
         "ingredients_raw": ["0.5 cdita de Sal", "31 g de Harina de trigo"],
         "recipe": ["Mezcla la harina con 0.5 cdita de Sal.", "Cocina a la plancha."]},
        {"meal": "Cena", "name": "Pechuga de Pollo Rellena",
         "ingredients": ["0.5 cdita Sal", "125 g Pechuga de pollo"],
         "ingredients_raw": ["0.5 cdita Sal", "125 g Pechuga de pollo"],
         "recipe": ["Sazona con ½ cdita Sal y hornea."]},
    ]}]


# ---------------------------------------------------------------------------

def test_cdita_de_sal_and_cdita_sal_both_convert(go):
    days = _mk_live_day()
    n = go._day_sodium_autofix(days, {}, db=_SaltDB())
    assert n >= 2, "las DOS formas vivas ('cdita de Sal' y 'cdita Sal' sin de) convierten"
    for meal in days[0]["meals"]:
        assert "Sal al gusto" in meal["ingredients"], meal["name"]
        assert not any("cdita" in s.lower() and "sal" in s.lower()
                       for s in meal["ingredients"]), meal["name"]
        assert meal["ingredients"] == meal["ingredients_raw"]
        assert meal["_sodium_autofix_applied"] == "salt_to_taste"


def test_recipe_steps_also_rewritten(go):
    days = _mk_live_day()
    go._day_sodium_autofix(days, {}, db=_SaltDB())
    joined = " ".join(s for m in days[0]["meals"] for s in m["recipe"]).lower()
    assert "cdita de sal" not in joined and "cdita sal" not in joined, \
        "los pasos dejan de dictar la cucharadita medida"
    assert "sal al gusto" in joined


def test_mixed_number_and_fraction_forms(go):
    for form in ("1½ cdita de sal", "½ cdita de sal", "2 cditas de sal", "3 g de sal"):
        days = [{"day": 1, "meals": [
            {"meal": "Almuerzo", "name": "Arroz",
             "ingredients": [form, "150 g de arroz"],
             "ingredients_raw": [form, "150 g de arroz"],
             "recipe": ["Cocina."]},
        ]}]
        assert go._day_sodium_autofix(days, {}, db=_SaltDB()) >= 1, form
        assert "Sal al gusto" in days[0]["meals"][0]["ingredients"], form


def test_salsa_and_al_gusto_still_untouched(go):
    days = [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Pescado",
         "ingredients": ["Sal al gusto", "2 cditas de salsa de soya",
                         "1 cdita de sal", "150 g de pescado"],
         "ingredients_raw": ["Sal al gusto", "2 cditas de salsa de soya",
                             "1 cdita de sal", "150 g de pescado"],
         "recipe": ["Cocina."]},
    ]}]
    go._day_sodium_autofix(days, {}, db=_SaltDB())
    ings = days[0]["meals"][0]["ingredients"]
    assert "Sal al gusto" in ings, "la forma honesta no se toca"
    assert "2 cditas de salsa de soya" in ings, "'salsa' no es 'sal' (word-boundary)"
    assert not any(s == "1 cdita de sal" for s in ings), "la cuantificada sí convierte"


def test_rx_anchored_in_source():
    assert "P1-SALT-CDITA-RX" in _GO
    assert r"cd(?:i)?tas?" in _GO, "el RX con cdita vive en el source (ingredientes + pasos)"
