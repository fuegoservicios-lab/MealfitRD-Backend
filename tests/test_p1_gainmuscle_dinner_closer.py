# -*- coding: utf-8 -*-
"""[P1-GAINMUSCLE-DINNER-CLOSER · 2026-09-05] Plan vivo 9b73656d (prueba A v9, entregado en el intento 3, donde la
autocrítica ya no corre): cena «Plátano verde majado con queso fresco» + 40 g de soya texturizada del cerrador, y
«600 g de papa en cubos» en otra cena. Tres cierres: el queso cede en cena de gain_muscle, la carne vegetal solo para
vegetarianos/veganos, y tope de realismo para tubérculos."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents as _sa  # noqa: E402


class _Info:
    def __init__(self, name, protein, kcal, carbs=2.0, fats=3.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _cheese_dinner():
    return {"meal": "Cena", "name": "Plátano verde majado con aguacate, espinaca y queso fresco a la parrilla",
            "protein": 18, "carbs": 60, "fats": 20, "cals": 500,
            "ingredients": ["½ plátano verde grande", "40 g de queso blanco", "½ aguacate", "2 tazas de espinacas"],
            "ingredients_raw": ["½ plátano verde grande", "40 g de queso blanco", "½ aguacate", "2 tazas de espinacas"],
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: asa el queso.", "Montaje: sirve."]}


def test_cheese_dish_cedes_to_lean_meat_only_in_gainmuscle_dinner(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "CLOSER_NO_DOUBLE_MAIN_ENABLED", True)
    m = _cheese_dinner()
    assert not go._dish_coherence_filter(m, _sa)("pechuga de pollo"), "sin objetivo: el queso como plato veta la carne (legado)"
    assert go._dish_coherence_filter(m, _sa, goal="gain_muscle")("pechuga de pollo")
    assert go._dish_coherence_filter(m, _sa, goal="gain_muscle")("filete de pescado blanco")
    # almuerzo de queso en gain_muscle: la cesión es SOLO para la cena
    m2 = dict(_cheese_dinner(), meal="Almuerzo")
    assert not go._dish_coherence_filter(m2, _sa, goal="gain_muscle")("pechuga de pollo")
    # la cena ya tiene carne principal: nunca una segunda
    m3 = dict(_cheese_dinner(), ingredients=["150 g de pollo", "40 g de queso blanco"])
    assert not go._dish_coherence_filter(m3, _sa, goal="gain_muscle")("filete de pescado blanco")


def test_closer_adds_chicken_not_soy_to_the_live_dinner(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "CLOSER_NO_DOUBLE_MAIN_ENABLED", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [
        (3.0, "Soya texturizada", _Info("Soya texturizada", 50.0, 330.0)),
        (2.0, "Pechuga de pollo", _Info("Pechuga de pollo", 31.0, 165.0)),
        (1.0, "Queso cottage", _Info("Queso cottage", 11.0, 98.0)),
    ]
    m = _cheese_dinner()
    g = go._close_protein_gap_for_meal(m, 45.0, _NoopDB(), candidates, enforce_min_threshold=False,
                                       diet="balanced", goal="gain_muscle")
    assert g > 0
    blob = " ".join(m["ingredients"]).lower()
    assert "pollo" in blob and "soya" not in blob, m["ingredients"]
    assert "Pollo" in m["name"]


def test_plant_meat_only_for_vegetarians(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [(3.0, "Soya texturizada", _Info("Soya texturizada", 50.0, 330.0)),
                  (1.0, "Lentejas", _Info("Lentejas", 9.0, 116.0))]
    base = {"meal": "Almuerzo", "name": "Guiso de vegetales con arroz", "protein": 8, "carbs": 70, "fats": 8, "cals": 380,
            "ingredients": ["1 taza de arroz", "vegetales"], "ingredients_raw": ["1 taza de arroz", "vegetales"],
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: guisa.", "Montaje: sirve."]}
    m_omni = dict(base, ingredients=list(base["ingredients"]), ingredients_raw=list(base["ingredients_raw"]))
    go._close_protein_gap_for_meal(m_omni, 30.0, _NoopDB(), candidates, enforce_min_threshold=False, diet="balanced")
    assert not any("soya" in i.lower() for i in m_omni["ingredients"]), m_omni["ingredients"]
    m_veg = dict(base, ingredients=list(base["ingredients"]), ingredients_raw=list(base["ingredients_raw"]))
    go._close_protein_gap_for_meal(m_veg, 30.0, _NoopDB(), candidates, enforce_min_threshold=False, diet="vegana")
    assert any("soya" in i.lower() for i in m_veg["ingredients"]), m_veg["ingredients"]


def test_tuber_portion_cap(monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "Pescado al vapor con papa",
                                  "ingredients": ["½ filete de pescado", "600 g de papa en cubos"],
                                  "ingredients_raw": ["½ filete de pescado", "600 g de papa en cubos"],
                                  "recipe": ["Mise en place: prepara.", "Montaje: sirve."]}]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) >= 1
    line = next(s for s in days[0]["meals"][0]["ingredients"] if "papa" in s.lower())
    grams = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert grams <= go.PORTION_CAP_TUBER_G, line
    # bajo el tope no se toca
    days2 = [{"day": 1, "meals": [{"meal": "Cena", "name": "Pollo con batata",
                                   "ingredients": ["150 g de pollo", "250 g de batata"],
                                   "ingredients_raw": ["150 g de pollo", "250 g de batata"], "recipe": []}]}]
    assert go._cap_unrealistic_portions(days2, db=_NoopDB()) == 0


def test_wiring():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'goal=form_data.get("mainGoal") or form_data.get("goal"))' in src, "FASE A pasa el objetivo al cerrador"
    assert "_coh_ok = _dish_coherence_filter(meal, _sa, goal=goal)" in src
    assert "tooltip-anchor: P1-GAINMUSCLE-DINNER-CLOSER" in src
