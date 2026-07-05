"""[P1-SALT-LINE-AUTOFIX · 2026-07-05] El "sodio fantasma" era sal explícita cuantificada.

Forense de los rechazos SODIO EXCESIVO recurrentes (10,866mg/día en 214635d9; 6,517 en
0b04c3c9; banner de techo en el plan d6bd9d04): el LLM escribe "1 cdta de sal" como
INGREDIENTE (= 2,358 mg de sodio — química correcta: 1 cdta ≈ 5.9 g de NaCl) y una sola
línea supera el techo OMS del día entero. La escalera del autofix solo cubría cubitos y
enlatados — la sal cuantificada pasaba de largo.

Fix doble:
  1. Peldaño (1.5) en `_day_sodium_autofix`: cuando el día supera el techo, toda línea de
     sal CUANTIFICADA (cdta/cda/gramos) pasa a "Sal al gusto" (culinariamente honesto en
     es-DO; el panel deja de asumir la cucharada completa) + pasos re-escritos.
     "Sal al gusto"/"pizca de sal" existentes NO se tocan.
  2. §17 del prompt day-gen: 'Si listas sal, escribe SIEMPRE Sal al gusto — JAMÁS cantidades'.
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


class _FakeDB:
    """Sal cuantificada = 2358 mg (química real); 'al gusto'/'pizca' = 0; resto 20."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "de sal" in low and "salsa" not in low:
            if "al gusto" in low or "pizca" in low:
                return {"sodium_mg": 0.0}
            return {"sodium_mg": 2358.0}
        return {"sodium_mg": 20.0}


def _mk_day(extra_ing=None):
    ings = ["1 cdta de sal", "150 g de arroz blanco"]
    if extra_ing:
        ings.append(extra_ing)
    return [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Arroz con Pollo",
         "ingredients": list(ings), "ingredients_raw": list(ings),
         "recipe": ["Añade 1 cdta de sal al arroz.", "Cocina."]},
    ]}]


# ---------------------------------------------------------------------------

def test_quantified_salt_becomes_al_gusto(go):
    days = _mk_day()
    n = go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert n >= 1
    meal = days[0]["meals"][0]
    assert "Sal al gusto" in meal["ingredients"]
    assert not any("cdta de sal" in s.lower() for s in meal["ingredients"])
    assert meal["ingredients"] == meal["ingredients_raw"]
    assert meal["_sodium_autofix_applied"] == "salt_to_taste"
    assert "sal al gusto" in " ".join(meal["recipe"]).lower(), \
        "los pasos también dejan de decir '1 cdta de sal'"


def test_al_gusto_and_pizca_untouched(go):
    days = [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Pescado",
         "ingredients": ["Sal al gusto", "1 pizca de sal", "1 cubito de pollo",
                         "150 g de pescado", "1 cdta de sal"],
         "ingredients_raw": ["Sal al gusto", "1 pizca de sal", "1 cubito de pollo",
                             "150 g de pescado", "1 cdta de sal"],
         "recipe": ["Cocina."]},
    ]}]

    class _DB(_FakeDB):
        def micros_from_ingredient_string(self, s):
            low = str(s).lower()
            if "cubito" in low:
                return {"sodium_mg": 1000.0}
            return super().micros_from_ingredient_string(s)

    go._day_sodium_autofix(days, {}, db=_DB())
    ings = days[0]["meals"][0]["ingredients"]
    assert "Sal al gusto" in ings and "1 pizca de sal" in ings, \
        "las formas ya-honestas de sal no se tocan"
    assert not any("cdta de sal" in s.lower() for s in ings)


def test_salsa_not_matched(go):
    days = _mk_day(extra_ing="2 cdas de salsa de tomate")
    go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert "2 cdas de salsa de tomate" in days[0]["meals"][0]["ingredients"], \
        "'salsa' no es 'sal' (word-boundary)"


def test_day_under_ceiling_untouched(go):
    days = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Arroz",
         "ingredients": ["1 cdta de sal baja en sodio (test)", "150 g de arroz"],
         "ingredients_raw": ["1 cdta de sal baja en sodio (test)", "150 g de arroz"],
         "recipe": ["Cocina."]},
    ]}]

    class _DBLow(_FakeDB):
        def micros_from_ingredient_string(self, s):
            return {"sodium_mg": 100.0}

    assert go._day_sodium_autofix(days, {}, db=_DBLow()) == 0, \
        "día bajo el techo → mínima intervención (ni la sal se toca)"


def test_prompt_rule_added():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as _P
    assert "Sal al gusto" in _P and "JAMÁS cantidades" in _P


def test_marker_anchored_in_source():
    assert "P1-SALT-LINE-AUTOFIX" in _GO
