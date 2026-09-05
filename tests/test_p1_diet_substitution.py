# -*- coding: utf-8 -*-
"""[P1-DIET-SUBSTITUTION · 2026-09-05] En 5 de 7 planes vegetarianos generados hoy el intento 1 murió por «pechuga de
pollo» que el modelo inventó FUERA del pool — el pool ya era vegetariano y el prompt lleva la prohibición desde
P1-DAYGEN-VEG-HARD-LINE. Cada rechazo costaba una replanificación completa. Ahora la carne se SUSTITUYE por su
equivalente vegetal antes del review (mismo motor que los alérgenos); el guard duro sigue siendo el backstop."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Pastelón de yuca con pechuga de pollo", "protein": 40, "carbs": 50, "fats": 15,
        "cals": 600, "ingredients": list(ings), "ingredients_raw": list(ings),
        "recipe": ["Mise en place: prepara.", "El Toque de Fuego: cocina.", "Montaje: sirve."],
    }]}]}


def _ings(plan):
    return plan["days"][0]["meals"][0]["ingredients"]


def test_meat_is_replaced_in_a_vegetarian_plan(monkeypatch):
    monkeypatch.setattr(go, "DIET_SUBSTITUTION_ENABLED", True)
    p = _plan("100 g de pechuga de pollo", "1 yuca mediana")
    assert go._apply_diet_substitutions(p, {"dietType": "vegetariana"}) == 1
    blob = " ".join(_ings(p)).lower()
    assert "pollo" not in blob and "soya" in blob, _ings(p)
    assert "100 g" in " ".join(_ings(p)), "la cantidad se conserva"
    assert p["days"][0]["meals"][0].get("_diet_subs_fixed")


def test_fish_is_replaced_for_vegetarian_but_kept_for_pescatarian(monkeypatch):
    monkeypatch.setattr(go, "DIET_SUBSTITUTION_ENABLED", True)
    veg = _plan("150 g de filete de pescado blanco")
    assert go._apply_diet_substitutions(veg, {"dietType": "vegetariana"}) == 1
    assert "garbanzos" in " ".join(_ings(veg)).lower(), _ings(veg)
    pes = _plan("150 g de filete de pescado blanco")
    assert go._apply_diet_substitutions(pes, {"dietType": "pescetariana"}) == 0
    assert "pescado" in " ".join(_ings(pes)).lower()


def test_plant_analogues_and_other_diets_are_untouched(monkeypatch):
    monkeypatch.setattr(go, "DIET_SUBSTITUTION_ENABLED", True)
    ana = _plan("80 g de carne de soya", "1 taza de leche de almendra")
    assert go._apply_diet_substitutions(ana, {"dietType": "vegana"}) == 0
    assert go._apply_diet_substitutions(_plan("100 g de pollo"), {"dietType": "balanceada"}) == 0
    assert go._apply_diet_substitutions(_plan("100 g de pollo"), {}) == 0
    monkeypatch.setattr(go, "DIET_SUBSTITUTION_ENABLED", False)
    off = _plan("100 g de pollo")
    assert go._apply_diet_substitutions(off, {"dietType": "vegana"}) == 0
    assert "pollo" in " ".join(_ings(off)).lower()


def test_runs_before_the_review_and_twice(monkeypatch):
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_apply_diet_substitutions(plan, form_data)") == 2, "pasada inicial + re-pasada tras condición"
    i = src.index("Guard 2.6 (dieta declarada)")
    j = src.index("_apply_allergen_substitutions(plan, form_data)")
    assert j < i, "el alérgeno tiene precedencia y corre antes"
    assert "tooltip-anchor: P1-DIET-SUBSTITUTION" in src
