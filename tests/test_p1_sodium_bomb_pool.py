"""[P1-SODIUM-BOMB-POOL · 2026-07-05] Proteínas curadas en sal — dos palancas (medido en vivo:
plan 3aa6e58a con pools "Salami Dominicano" + "Bacalao" → sodio 4,576mg/2,000, banner
micro_worst_day_ceiling y 3 intentos):

1. SORTEO (ai_helpers): penalty UNIVERSAL ×0.1 (knob MEALFIT_SODIUM_BOMB_POOL_PENALTY) a
   bacalao/arenque/salami/tocino/longaniza... en el pool de proteínas — el presupuesto OMS de
   2000mg no depende del goal. Se apila con el penalty de embutidos por goal (salami en
   gain_muscle queda ×0.01). Graceful: pueden salir si no hay alternativa.
2. AUTOFIX (`_day_sodium_autofix`): rungs nuevas curado→fresco (bacalao/arenque→pescado blanco;
   salami/salchichón/pepperoni/mortadela/tocino→pechuga de pollo; longaniza/chorizo→pollo molido)
   con el mismo mecanismo grams-preserving del swap de enlatados, check de alergia/dislike
   PER-ENTRY (alergia a pescado bloquea bacalao→pescado pero salami→pollo sigue), marker
   `swap_saltcured` reconocido por el fidelity-discount.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "ai_helpers.py"), encoding="utf-8") as f:
    _AI = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_pool_penalty_universal_with_knob():
    assert "P1-SODIUM-BOMB-POOL" in _AI
    assert '_sb_envf("MEALFIT_SODIUM_BOMB_POOL_PENALTY", 0.1)' in _AI
    i = _AI.index("[P1-SODIUM-BOMB-POOL")
    win = _AI[i:i + 2600]
    for t in ("bacalao", "salami", "longaniza", "tocino"):
        assert t in win


def test_ladder_extended_and_fidelity_marker():
    i = _GO.index("_SODIUM_SWAP_LADDER = (")
    win = _GO[i:i + 1800]
    assert '"bacalao"' in win and "salami" in win and "longaniza|chorizo" in win
    assert '"swap_saltcured"' in _GO, "marker propio del swap curado"
    i_fid = _GO.index('_sodium_autofix_applied") == "swap_saltcured"')
    assert "bacalao arenque salami" in _GO[i_fid:i_fid + 400], \
        "el fidelity-discount debe reconocer la proteína fuente del swap curado"


def test_per_entry_replacement_check():
    i = _GO.index("def _day_sodium_autofix")
    body = _GO[i:i + 9000]
    assert "_repl_allowed(_repl)" in body, "alergia/dislike se chequea POR ENTRY de la escalera"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "SODIUM_DAY_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(g, "SODIUM_DAY_CEILING_MG", 2000)
    monkeypatch.setattr(g, "SODIUM_DAY_AUTOFIX_MAX_SWAPS", 3)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """sodio por gramo: curados altísimos, frescos despreciables."""

    _RATES = (("bacalao", 20.0), ("salami", 18.0), ("sardina", 6.0),
              ("pescado", 0.5), ("pollo", 0.4))

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        rate = next((r for t, r in self._RATES if t in low), 0.1)
        return {"sodium_mg": rate * g}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None


def _meal(name, ings, recipe=None):
    return {"meal": "Almuerzo", "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings),
            "recipe": recipe or ["Mise en place: prepara.", "Montaje: sirve."]}


def test_bacalao_swapped_grams_preserved(go):
    days = [{"day": 1, "meals": [
        _meal("Bacalao Guisado con Yuca", ["200 g de bacalao", "150 g de yuca"],
              ["Mise en place: desala el bacalao.",
               "El Toque de Fuego: guisa el bacalao 15 min a fuego medio.",
               "Montaje: sirve."]),
    ]}]
    n = go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert n >= 1
    m = days[0]["meals"][0]
    joined = " ".join(m["ingredients"]).lower()
    assert "bacalao" not in joined and "200 g de filete de pescado blanco" in joined, \
        "gramos PRESERVADOS + curado→fresco"
    assert m["ingredients"] == m["ingredients_raw"], "raw en lockstep"
    assert m["_sodium_autofix_applied"] == "swap_saltcured"
    assert "bacalao" not in str(m["name"]).lower(), f"nombre coherente: {m['name']}"
    assert "bacalao" not in " ".join(m["recipe"]).lower(), "pasos reescritos"


def test_salami_swapped_to_pollo(go):
    days = [{"day": 1, "meals": [
        _meal("Sándwich de Salami Dominicano", ["120 g de salami dominicano", "2 lonjas de pan"]),
    ]}]
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) >= 1
    m = days[0]["meals"][0]
    assert "120 g de pechuga de pollo" in " ".join(m["ingredients"]).lower()
    assert m["_sodium_autofix_applied"] == "swap_saltcured"


def test_fish_allergy_blocks_bacalao_but_salami_still_swaps(go, monkeypatch):
    """El check per-entry: alergia a pescado bloquea el reemplazo pescado-blanco (bacalao queda)
    pero salami→pollo sigue disponible en el MISMO día."""
    monkeypatch.setattr(go, "_scan_allergen_violations",
                        lambda plan, allergies: ([("m", "i", "c")]
                                                 if allergies and "pescado" in str(plan).lower()
                                                 else []))
    days = [{"day": 1, "meals": [
        _meal("Bacalao con Yuca", ["150 g de bacalao", "150 g de yuca"]),
        _meal("Sándwich de Salami", ["120 g de salami dominicano", "2 lonjas de pan"]),
    ]}]
    go._day_sodium_autofix(days, {"allergies": ["pescado"]}, db=_FakeDB())
    joined_all = " ".join(s for m in days[0]["meals"] for s in m["ingredients"]).lower()
    assert "bacalao" in joined_all, "alergia a pescado → bacalao NO se swapea a pescado"
    assert "pechuga de pollo" in joined_all, "…pero salami→pollo sigue disponible (per-entry)"


def test_canned_regression_marker_unchanged(go):
    # 350 g × 6 mg/g = 2,100 mg > techo 2,000 → el swap de enlatado dispara.
    days = [{"day": 1, "meals": [
        _meal("Ensalada con Sardinas", ["350 g de sardinas en lata", "100 g de lechuga"]),
    ]}]
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) >= 1
    assert days[0]["meals"][0]["_sodium_autofix_applied"] == "swap_canned", \
        "el swap de enlatados conserva su marker histórico"


def test_day_under_ceiling_untouched(go):
    days = [{"day": 1, "meals": [_meal("Bacalao Chiquito", ["50 g de bacalao", "150 g de yuca"])]}]
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) == 0
    assert "50 g de bacalao" in days[0]["meals"][0]["ingredients"], \
        "1000mg < techo → el día no se toca (el swap es correctivo, no censura)"
