"""[P0-1-PAIRING-PLAUSIBILITY-GATE · 2026-07-10] (recipe plausibility roadmap, item P0-1) Evidencia
visual (plan 564d6e4e): "3 papas hervidas cubiertas de mantequilla de maní + gajos de naranja + queso
cottage mezclado" — pasó TODOS los gates existentes (DISH-COHERENCE valida estructura, SLOT-
APPROPRIATENESS el slot, cookable-min el realismo de cocción) porque ninguno valida la COMBINACIÓN de
categorías de alimentos en el MISMO plato. El solver usa mantequilla de maní como filler de grasa sin
preguntar "¿esto se come junto?".

Scope de ESTA fase (fase 1 del roadmap, warn-first): DETECCIÓN + telemetría determinista de pares
incompatibles conocidos sobre el plan ENTREGADO. NO re-siembra el filler todavía (acción "fix" diferida
a cuando haya evidencia de 48h sin falsos positivos, per roadmap) — ese es el patrón "evidence-first" ya
usado en el repo (P1-SOLVER-SATURATION-RELIEF). Knob `MEALFIT_PAIRING_GATE` (off/warn), default warn.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


def test_marker_and_knob_present():
    assert "P0-1-PAIRING-PLAUSIBILITY-GATE" in _GO
    assert "MEALFIT_PAIRING_GATE" in _GO


def test_function_defined():
    assert "def detect_pairing_plausibility_violations(plan_data" in _GO


def test_called_in_dbplans_shield():
    assert "detect_pairing_plausibility_violations" in _DBP


def _meal(name, ingredients):
    return {"meal": "Merienda", "name": name, "ingredients": ingredients}


def _plan(meals):
    return {"days": [{"day": 1, "meals": meals}]}


def test_flags_nut_butter_plus_boiled_tuber_plus_citrus_plus_cheese():
    import graph_orchestrator as g
    plan = _plan([_meal("Papas con maní", [
        "3 papas medianas hervidas", "2 cdas de mantequilla de maní",
        "1 naranja en gajos", "15 g de queso cottage",
    ])])
    violations = g.detect_pairing_plausibility_violations(plan)
    assert len(violations) >= 1
    v = violations[0]
    assert v["day"] == 1
    assert "nut_butter" in v["categories"]


def test_passes_legitimate_nut_butter_pairings():
    """avena+maní, batido+maní, tostada+maní: PB con carbo dulce/desayuno — NO debe flaggear."""
    import graph_orchestrator as g
    plan = _plan([
        _meal("Avena con maní", ["¾ taza de avena", "1¼ cdas de mantequilla de maní", "½ manzana"]),
        _meal("Tostadas con maní", ["2 tortillas de trigo", "1.25 cdas de mantequilla de maní", "0.5 toronja"]),
    ])
    assert g.detect_pairing_plausibility_violations(plan) == []


def test_off_knob_short_circuits(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "PAIRING_GATE_MODE", "off")
    plan = _plan([_meal("Papas con maní", [
        "3 papas medianas hervidas", "2 cdas de mantequilla de maní", "1 naranja en gajos",
    ])])
    assert g.detect_pairing_plausibility_violations(plan) == []


def test_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.detect_pairing_plausibility_violations({}) == []
    assert g.detect_pairing_plausibility_violations(None) == []
    assert g.detect_pairing_plausibility_violations({"days": "nope"}) == []
