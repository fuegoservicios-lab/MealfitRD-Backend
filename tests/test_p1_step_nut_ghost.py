"""[P1-STEP-NUT-GHOST · 2026-07-05] Frutos secos fantasma en pasos (screenshots plan 3aa6e58a).

Los panqueques decían "Pica los MEREYES toscamente" y "espolvorea los mereyes picados" — y los
mereyes NO estaban en ingredients[] → jamás se compran, receta rota al cocinar. El veg-guard solo
materializa vegetales y el carb-ghost solo carbs curados (avena/casabe/batata/yuca) — frutos
secos/semillas se escapaban. Espejo exacto del mecanismo carb-ghost (detección nombre+pasos,
excludes, scan de alérgenos fail-secure) con escalera propia: merey/nueces/almendra/pistacho/maní.
Knob MEALFIT_RECIPE_STEP_NUT_GUARD (ON).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_and_ladder_anchored():
    m = re.search(r'RECIPE_STEP_NUT_GUARD_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_RECIPE_STEP_NUT_GUARD"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"
    i = _GO.index("_STEP_NUT_GHOSTS = (")
    win = _GO[i:i + 500]
    for t in ("merey", "nueces", "almendra", "pistacho", "mani"):
        assert t in win


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "RECIPE_STEP_CARB_GUARD_ENABLED", True)
    monkeypatch.setattr(g, "RECIPE_STEP_NUT_GUARD_ENABLED", True)
    return g


def _pancakes(ings):
    return [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Panqueques de Harina con Mermelada de Kiwi",
        "ingredients": list(ings), "ingredients_raw": list(ings),
        "recipe": [
            "Mise en place: mezcla la harina. Pica los mereyes toscamente.",
            "El Toque de Fuego: cocina los panqueques 2-3 min por lado.",
            "Montaje: apila y espolvorea los mereyes picados.",
        ],
    }]}]


def test_ghost_nut_materialized_in_ings_and_raw(go):
    days = _pancakes(["⅓ taza de harina de trigo (41 g)", "2 huevos", "1 kiwi"])
    n = go._add_missing_recipe_step_carbs(days, db=None, allergies=None)
    assert n >= 1
    m = days[0]["meals"][0]
    assert any("merey" in s.lower() for s in m["ingredients"]), \
        "los mereyes de los pasos deben existir en la lista (si no, no se compran)"
    assert any("merey" in s.lower() for s in m["ingredients_raw"]), "raw en lockstep"


def test_nut_allergy_blocks_materialization(go, monkeypatch):
    monkeypatch.setattr(go, "_scan_allergen_violations",
                        lambda plan, allergies: [("m", "i", "c")] if allergies else [])
    days = _pancakes(["⅓ taza de harina de trigo (41 g)", "2 huevos"])
    go._add_missing_recipe_step_carbs(days, db=None, allergies=["frutos secos"])
    assert not any("merey" in s.lower() for s in days[0]["meals"][0]["ingredients"]), \
        "fail-secure: jamás materializar un alérgeno declarado"


def test_peanut_butter_exclude(go):
    """'mantequilla de maní' presente → el token 'mani' de los pasos NO materializa maní suelto."""
    days = [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Avena con Mantequilla de Maní",
        "ingredients": ["45 g de avena", "1½ cda de mantequilla de maní (24g)"],
        "ingredients_raw": ["45 g de avena", "1½ cda de mantequilla de maní (24g)"],
        "recipe": ["Mise en place: mide la avena.",
                   "El Toque de Fuego: cocina la avena 5-7 min a fuego medio.",
                   "Montaje: mezcla la mantequilla de maní y sirve."],
    }]}]
    assert go._add_missing_recipe_step_carbs(days, db=None, allergies=None) == 0


def test_knob_off_restores_carbs_only(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_STEP_NUT_GUARD_ENABLED", False)
    days = _pancakes(["⅓ taza de harina de trigo (41 g)", "2 huevos"])
    go._add_missing_recipe_step_carbs(days, db=None, allergies=None)
    assert not any("merey" in s.lower() for s in days[0]["meals"][0]["ingredients"])
