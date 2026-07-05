"""[P2-SEED-STEP-NOTE · 2026-07-05] (audit solver+seeder P2-2)

La línea sembrada por el micro-closer no se mencionaba en los pasos: "10 g de semillas de
linaza" aparecía en `ingredients` (y en la lista de compras) pero la receta jamás decía qué
hacer con ella — incoherencia receta↔pasos que el propio repo persigue en toda otra superficie.
Fix: nota 🌱 insertada ANTES del Montaje (espejo de la nota 💪 del closer de proteína, vía
`_insert_step_before_montaje`). Las notas 🌱 están excluidas del detector no-cook
(P1-RECIPE-CONTRACT-NOCOOK) y no llevan prefijo de pilar → cero interacción con el contrato
de pasos. Knob `MEALFIT_MICRO_SEED_STEP_NOTE` (ON).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_default_on():
    m = re.search(r'MICRO_SEED_STEP_NOTE_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_MICRO_SEED_STEP_NOTE"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_SEED_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
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


def _mk_plan():
    return {"days": [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Mango con Yogurt", "cals": 250,
         "ingredients": ["100 g de yogurt"], "ingredients_raw": ["100 g de yogurt"],
         "recipe": ["Mise en place: corta el mango y mide el yogurt.",
                    "Montaje: sirve el mango sobre el yogurt."]},
    ]}]}


def _run(go, monkeypatch, plan):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "vit_e_mg", "piso": 15.0, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "vit_e_mg", "piso": 15.0, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    return go._close_micro_gaps_for_plan(plan, {}, _FakeDB())


def test_seed_adds_note_before_montaje(go, monkeypatch):
    plan = _mk_plan()
    assert _run(go, monkeypatch, plan) >= 1
    rec = plan["days"][0]["meals"][0]["recipe"]
    _notes = [i for i, s in enumerate(rec) if isinstance(s, str) and s.startswith("🌱")]
    assert _notes, "la semilla debe vivir también en los pasos (nota 🌱)"
    _mo = next(i for i, s in enumerate(rec) if str(s).lower().startswith("montaje"))
    assert _notes[0] < _mo, "la nota se inserta ANTES del Montaje (patrón 💪)"
    assert "girasol" in rec[_notes[0]].lower(), "la nota nombra la semilla sembrada"
    assert "vitamina e" in rec[_notes[0]].lower(), "la nota explica QUÉ cierra (honestidad)"


def test_note_does_not_break_nocook_contract(go, monkeypatch):
    """La nota 🌱 no re-clasifica el plato frío como cocinado ni rompe el lint de 2 pilares."""
    import graph_orchestrator as g
    plan = _mk_plan()
    _run(go, monkeypatch, plan)
    assert g._recipe_step_contract_issues(plan["days"][0]["meals"][0]) == []


def test_knob_off_no_note(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_SEED_STEP_NOTE_ENABLED", False)
    plan = _mk_plan()
    _run(go, monkeypatch, plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert not any(str(s).startswith("🌱") for s in rec)
