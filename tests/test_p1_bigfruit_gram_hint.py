"""[P1-BIGFRUIT-GRAM-HINT · 2026-07-06] Las frutas grandes (lechosa/papaya/melón/sandía/piña) tienen
peso POR-UNIDAD = fruta ENTERA en el catálogo → `macros_from_ingredient_string("1 lechosa") = 711 kcal`
(papaya ~700g). Un conteo PELADO en un meal es un FANTASMA que corrompe el census del refinador global,
el coherence guard receta↔lista y la lista de compras. Fix: al conteo pelado se le fija una porción
servible con gramaje ("1 lechosa (200g)") — el hint gana sobre el conteo en el parser.

Validado en plan real 4339544f d2: "1 lechosa" → "1 lechosa (200g)", macros del día STORED sin cambio
(96/110/103/105) → stored-neutral (cero regresión de banda), string ahora consistente con la porción real.
"""
from pathlib import Path
import graph_orchestrator as go
from constants import strip_accents as _sa

_GO = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _h(s):
    return go._bigfruit_bare_count_serving(s, _sa(s.lower()))


# --- unit del helper --------------------------------------------------------

def test_bare_count_bigfruit_gets_serving_hint():
    assert _h("1 lechosa") == f"1 lechosa ({go.BIGFRUIT_SERVING_G}g)"
    assert _h("1 papaya madura") == f"1 papaya madura ({go.BIGFRUIT_SERVING_G}g)"


def test_count_capped_to_one_and_rest_preserved():
    # "2 lechosa en cubos" → conteo a 1, resto intacto, hint añadido.
    assert _h("2 lechosa en cubos") == f"1 lechosa en cubos ({go.BIGFRUIT_SERVING_G}g)"


def test_accents_preserved():
    out = _h("1 melón en cubos")
    assert out == f"1 melón en cubos ({go.BIGFRUIT_SERVING_G}g)"


def test_skips_lines_with_existing_gram_hint():
    assert _h("1 lechosa (203g)") is None
    assert _h("1 lechosa mediana (203g)") is None


def test_skips_gram_lead_and_cup_lead():
    assert _h("200g de lechosa") is None        # gram-lead → rama existente
    assert _h("2 tazas de lechosa") is None      # cup-lead → rama existente


def test_skips_non_phantom_fruit():
    # guineo/mango/chinola/uva tienen peso por-unidad correcto → no se tocan.
    assert _h("1 guineo") is None
    assert _h("2 guineos") is None
    assert _h("1 chinola") is None
    assert _h("20 uvas") is None


def test_skips_fraction_lead():
    assert _h("½ lechosa") is None


# --- correctness de macros (el hint gana sobre el conteo fantasma) ----------

def test_hint_yields_serving_macros_not_whole_fruit():
    from nutrition_db import IngredientNutritionDB
    import db_core
    db_core.connection_pool.open()
    db = IngredientNutritionDB()
    phantom = db.macros_from_ingredient_string("1 lechosa")
    fixed = db.macros_from_ingredient_string(_h("1 lechosa"))
    assert phantom and fixed
    assert phantom["kcal"] > 400, "sanity: el conteo pelado ES fantasma (fruta entera)"
    assert fixed["kcal"] < phantom["kcal"] / 2, "el hint reduce a una porción servible"
    # ~200g de lechosa ≈ 70-90 kcal
    assert 40 <= fixed["kcal"] <= 160, f"porción servible: {fixed['kcal']}"


# --- integración con _cap_unrealistic_portions ------------------------------

class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None
    def lookup(self, s):
        return None


def test_cap_rewrites_and_locksteps_raw():
    days = [{"meals": [{"meal": "Merienda", "name": "X",
                        "ingredients": ["1 lechosa", "60g de pollo"],
                        "ingredients_raw": ["1 lechosa", "60g de pollo"]}]}]
    n = go._cap_unrealistic_portions(days, db=_NoopDB())
    assert n >= 1
    ing = days[0]["meals"][0]["ingredients"][0]
    assert ing == f"1 lechosa ({go.BIGFRUIT_SERVING_G}g)"
    # lockstep raw (lista de compras coherente)
    assert days[0]["meals"][0]["ingredients"] == days[0]["meals"][0]["ingredients_raw"]


def test_knob_off_skips(monkeypatch):
    monkeypatch.setattr(go, "BIGFRUIT_GRAM_HINT_ENABLED", False)
    days = [{"meals": [{"meal": "M", "name": "X",
                        "ingredients": ["1 lechosa"], "ingredients_raw": ["1 lechosa"]}]}]
    go._cap_unrealistic_portions(days, db=_NoopDB())
    assert days[0]["meals"][0]["ingredients"][0] == "1 lechosa"  # sin tocar


def test_marker_anchored():
    assert "P1-BIGFRUIT-GRAM-HINT" in _GO
    assert "BIGFRUIT_SERVING_G" in _GO
