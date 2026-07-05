"""[P1-SODIUM-DAY-AUTOFIX · 2026-07-04] Corrector determinista del techo de sodio POR DÍA.

Caso vivo (renovación 2026-07-04, plan bb595697): aprobado con sodio PROMEDIO 1,818mg
"bajo control", pero `per_day_ceilings` flaggeó 2 de 3 días > 2,000mg. El motor tenía
closers que SUBEN micros al piso pero nada que BAJARA un día sobre el techo — el usuario
come días, no promedios.

Escalera determinista (pre-motor, el solver re-dimensiona después):
  1. STRIP de cubitos/sazón completo (≈1000mg c/u, cero rol de macros) + pasos → sazón natural.
  2. Si el día SIGUE sobre el techo: SWAP del enlatado más rico (sardinas/atún) → pescado
     fresco del catálogo (sodio 10-50× menor, misma familia culinaria).
Respeta alergias (pescado → solo strip) y dislikes. Gate/banner per-día como backstop.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


class _FakeDB:
    """Medidor de sodio determinista por substring (sin catálogo real).
    Accent-insensitive como la DB real ('atún' debe medir igual que 'atun')."""
    _MAP = (("sardina", 900.0), ("atun", 700.0), ("cubito", 1000.0),
            ("queso", 400.0), ("pescado blanco", 60.0))

    @staticmethod
    def _norm(s):
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                       if unicodedata.category(c) != "Mn")

    def micros_from_ingredient_string(self, s):
        low = self._norm(s)
        for tok, na in self._MAP:
            if tok in low:
                return {"sodium_mg": na}
        return {"sodium_mg": 20.0}


def _mk_salty_day():
    return [{
        "day": 1,
        "meals": [
            {"meal": "Almuerzo",
             "name": "Sardinas en Lata con Arroz",
             "ingredients": ["1 lata de sardinas en lata", "150 g de arroz blanco", "1 cubito de pollo"],
             "ingredients_raw": ["1 lata de sardinas en lata", "150 g de arroz blanco", "1 cubito de pollo"],
             "recipe": ["Desmenuza las sardinas.", "Disuelve el cubito en el arroz."]},
            {"meal": "Cena",
             "name": "Queso con Casabe",
             "ingredients": ["100 g de queso blanco", "60 g de casabe"],
             "ingredients_raw": ["100 g de queso blanco", "60 g de casabe"],
             "recipe": ["Sirve."]},
        ],
    }]
    # sodio: 900 (sardinas) + 20 (arroz) + 1000 (cubito) + 400 (queso) + 20 (casabe) = 2340 > 2000


# ---------------------------------------------------------------------------
# knobs + wiring
# ---------------------------------------------------------------------------

def test_knobs_defaults():
    assert '_env_bool("MEALFIT_SODIUM_DAY_AUTOFIX", True)' in _GO
    assert '_env_int("MEALFIT_SODIUM_DAY_CEILING_MG", 2000' in _GO
    assert '_env_int("MEALFIT_SODIUM_DAY_AUTOFIX_MAX_SWAPS", 1' in _GO


def test_wired_in_assemble_pre_engine():
    i = _GO.index("pareo(s) fruta-dulce+salado reescrito(s)")
    win = _GO[i:i + 1400]
    assert "_day_sodium_autofix(days, form_data)" in win, \
        "el autofix de sodio corre en el mismo seam pre-motor que night-rice/appetit"


# ---------------------------------------------------------------------------
# funcional
# ---------------------------------------------------------------------------

def test_strip_cubito_first_then_swap_if_needed(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_salty_day()
    n = go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert n >= 1
    lunch = days[0]["meals"][0]
    # el cubito se fue de ingredients y raw…
    assert not any("cubito" in s for s in lunch["ingredients"])
    assert not any("cubito" in s for s in lunch["ingredients_raw"])
    # …y el paso lo reemplaza por sazón natural.
    assert "cubito" not in " ".join(lunch["recipe"]).lower()
    # tras el strip: 900+20+400+20 = 1340 ≤ 2000 → NO hace falta swap (escalera se detiene).
    assert any("sardina" in s for s in lunch["ingredients"]), \
        "si el strip basta, el enlatado se respeta (mínima intervención)"


def test_swap_canned_when_strip_not_enough(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_salty_day()
    # sin cubito pero con DOS enlatados → strip no aplica y el día queda 900+700+400+40 = 2040 > 2000.
    lunch = days[0]["meals"][0]
    lunch["ingredients"] = ["1 lata de sardinas en lata", "1 lata de atún en agua", "150 g de arroz blanco"]
    lunch["ingredients_raw"] = list(lunch["ingredients"])
    n = go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert n == 1
    joined = " ".join(lunch["ingredients"]).lower()
    # swapea el MÁS rico (sardinas 900) y conserva el otro (mínima intervención, MAX_SWAPS=1).
    assert "sardina" not in joined and "pescado blanco" in joined
    assert "atún" in joined or "atun" in joined
    assert lunch["_sodium_autofix_applied"] == "swap_canned"
    assert "sardina" not in lunch["name"].lower()


def test_day_under_ceiling_untouched(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Pollo con Arroz",
        "ingredients": ["150 g de pollo", "150 g de arroz blanco"],
        "ingredients_raw": ["150 g de pollo", "150 g de arroz blanco"],
        "recipe": ["Cocina."],
    }]}]
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) == 0
    assert days[0]["meals"][0]["ingredients"] == ["150 g de pollo", "150 g de arroz blanco"]


def test_fish_allergy_blocks_swap_but_strip_still_works(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    monkeypatch.setattr(go, "_scan_allergen_violations", lambda plan, allergies: ["pescado"])
    days = _mk_salty_day()
    n = go._day_sodium_autofix(days, {"allergies": ["pescado"]}, db=_FakeDB())
    assert n >= 1
    lunch = days[0]["meals"][0]
    assert not any("cubito" in s for s in lunch["ingredients"]), "el strip no depende del swap"
    assert any("sardina" in s for s in lunch["ingredients"]), \
        "con alergia a pescado el swap a pescado fresco NO ocurre (conservador)"


def test_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "SODIUM_DAY_AUTOFIX_ENABLED", False)
    assert go._day_sodium_autofix(_mk_salty_day(), {}, db=_FakeDB()) == 0


# ---------------------------------------------------------------------------
# §17 refuerzo per-día + marker
# ---------------------------------------------------------------------------

def test_prompt_says_per_day():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as _P
    assert "CADA DÍA por separado" in _P, \
        "el §17 debe explicitar que el techo es per-día (no promedio)"


def test_marker_anchored_in_source():
    assert "P1-SODIUM-DAY-AUTOFIX" in _GO
