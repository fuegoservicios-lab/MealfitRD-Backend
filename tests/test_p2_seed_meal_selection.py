"""[P2-SEED-MEAL-SELECTION · 2026-07-05] (audit solver+seeder P2-3)

El seed del micro-closer caía SIEMPRE en la PRIMERA merienda del día → concentración de añadidos
en un solo plato (merienda de 765 kcal observada en vivo: mango + maní + cottage + semillas).
Con `MEALFIT_MICRO_SEED_SPREAD` (ON): preferencia meriendas SIN seed previo → cualquier meal sin
seed → merienda → cualquiera; dentro del pool gana la de MENOS kcal actuales (headroom físico).
Knob OFF → comportamiento previo (primera merienda).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_default_on():
    m = re.search(r'MICRO_SEED_SPREAD_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_MICRO_SEED_SPREAD"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_SEED_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """Solo la semilla de girasol porta vit E; nada más resuelve micros."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "girasol" in low:
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
            g = float(m.group(1).replace(",", ".")) if m else 10.0
            return {"vit_e_mg": 0.352 * g}
        return {}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 58.0} if "girasol" in str(s).lower() else {"kcal": 0.0}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None

    def rescale_ingredient_string(self, s, factor):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return s
        g = float(m.group(1).replace(",", ".")) * factor
        return re.sub(r"^\s*\d+(?:[.,]\d+)?\s*g", f"{g:.0f} g", s, count=1)


def _meal(slot, name, kcal, seeded=None):
    m = {"meal": slot, "name": name, "cals": kcal,
         "ingredients": ["100 g de yogurt"], "ingredients_raw": ["100 g de yogurt"],
         "recipe": ["Mise en place: mide el yogurt.", "Montaje: sirve."]}
    if seeded:
        m["_micro_seed_applied"] = seeded
    return m


def _run(go, monkeypatch, plan):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "vit_e_mg", "piso": 15.0, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "vit_e_mg", "piso": 15.0, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    return go._close_micro_gaps_for_plan(plan, {}, _FakeDB())


def test_seed_lands_in_lowest_kcal_unseeded_merienda(go, monkeypatch):
    plan = {"days": [{"day": 1, "meals": [
        _meal("Merienda", "Merienda Grande", 700),
        _meal("Almuerzo", "Almuerzo", 900),
        _meal("Merienda", "Merienda Chica", 200),
    ]}]}
    assert _run(go, monkeypatch, plan) >= 1
    meals = plan["days"][0]["meals"]
    assert meals[2].get("_micro_seed_applied") == "vit_e_mg", \
        "la merienda de MENOS kcal (más headroom) recibe la semilla"
    assert not any("girasol" in " ".join(meals[0]["ingredients"]).lower() for _ in [0])


def test_already_seeded_merienda_skipped(go, monkeypatch):
    plan = {"days": [{"day": 1, "meals": [
        _meal("Merienda", "Merienda Chica", 200, seeded="omega3_g"),
        _meal("Merienda", "Merienda Grande", 700),
    ]}]}
    _run(go, monkeypatch, plan)
    meals = plan["days"][0]["meals"]
    assert any("girasol" in s.lower() for s in meals[1]["ingredients"]), \
        "la merienda YA sembrada (otro micro) se salta → reparte entre platos"


def test_knob_off_restores_first_merienda(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_SEED_SPREAD_ENABLED", False)
    plan = {"days": [{"day": 1, "meals": [
        _meal("Merienda", "Merienda Grande", 700),
        _meal("Merienda", "Merienda Chica", 200),
    ]}]}
    _run(go, monkeypatch, plan)
    meals = plan["days"][0]["meals"]
    assert any("girasol" in s.lower() for s in meals[0]["ingredients"]), \
        "con el knob OFF vuelve el comportamiento previo (primera merienda)"
