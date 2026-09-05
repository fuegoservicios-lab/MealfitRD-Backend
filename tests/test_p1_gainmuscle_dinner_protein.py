# -*- coding: utf-8 -*-
"""[P1-GAINMUSCLE-DINNER-PROTEIN · 2026-09-05] Plan vivo b4316db6 (gain_muscle): dos cenas «batata rellena de
queso» (fresco / mozzarella) con 23 y 28 g de proteína, con pollo y pescado en el pool del día; Diversidad 4/10.
Dos capas: aviso en el prompt de la cena (solo gain_muscle) + señal determinista en la autocrítica."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import prompts.day_generator as dg  # noqa: E402

FD = {"mainGoal": "gain_muscle"}


def _cena(name, protein, ingredients):
    return {"meal": "Cena", "name": name, "protein": protein, "ingredients": ingredients}


def _live_days():
    return [
        {"day": 1, "meals": [{"meal": "Almuerzo", "name": "Arroz criollo de pollo", "protein": 44, "ingredients": ["pollo"]},
                             _cena("Batata Rellena de Queso Fresco y Espinaca al Airfryer", 23, ["1 batata mediana", "80 g de queso blanco fresco"])]},
        {"day": 2, "meals": [_cena("Filete de pescado blanco con arepita de maíz", 44, ["1 filete de pescado"])]},
        {"day": 3, "meals": [_cena("Batata horneada rellena de mozzarella fresca y auyama", 28, ["½ batata", "80 g de queso mozzarella"])]},
    ]


def test_detector_flags_cheese_dinners_and_the_repeated_concept(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DINNER_PROTEIN_ENABLED", True)
    issues = go._detect_gainmuscle_dinner_issues(_live_days(), FD)
    assert len(issues) == 3, issues
    assert issues[0].startswith("Día 1, cena") and "QUESO" in issues[0] and "23 g" in issues[0]
    assert issues[1].startswith("Día 3, cena")
    assert "Días 1, 3" in issues[2] and "MISMO concepto" in issues[2]


def test_detector_is_quiet_for_lean_dinners_other_goals_and_knob_off(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DINNER_PROTEIN_ENABLED", True)
    ok = [{"day": 1, "meals": [_cena("Pollo guisado con batata y queso rallado", 40, ["150 g de pollo", "20 g de queso"])]}]
    assert go._detect_gainmuscle_dinner_issues(ok, FD) == []            # el queso como extensor está bien
    assert go._detect_gainmuscle_dinner_issues(_live_days(), {"mainGoal": "lose_weight"}) == []
    monkeypatch.setattr(go, "GAINMUSCLE_DINNER_PROTEIN_ENABLED", False)
    assert go._detect_gainmuscle_dinner_issues(_live_days(), FD) == []


def test_prompt_adds_the_dinner_rule_only_for_gain_muscle(monkeypatch):
    monkeypatch.setenv("MEALFIT_DAYGEN_DINNER_IDENTITY", "true")
    skel = {"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}
    assert "CENA EN GANANCIA MUSCULAR" in dg.build_day_assignment_context(skel, 1, goal="gain_muscle")
    assert "CENA EN GANANCIA MUSCULAR" not in dg.build_day_assignment_context(skel, 1, goal="lose_weight")
    assert "CENA EN GANANCIA MUSCULAR" not in dg.build_day_assignment_context(skel, 1)


def test_wiring_in_orchestrator():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '{crossday_block}{gm_dinner_block}{user_context}' in src, "el bloque entra en el prompt de la autocrítica"
    assert 'goal=(form_data or {}).get("mainGoal") or (form_data or {}).get("goal"),' in src, "el day-gen recibe el objetivo"
    assert "tooltip-anchor: P1-GAINMUSCLE-DINNER-PROTEIN" in src
