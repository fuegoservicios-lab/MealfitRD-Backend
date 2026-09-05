# -*- coding: utf-8 -*-
"""[P2-LIGHT-BASE-NO-REPEAT · 2026-09-05] Plan vivo 82d6f2a5, día 2: «Bowl batido de piña, avena y yogur» (80 g de avena)
en el desayuno y «Vasito frío de higo, avena y mantequilla de maní» (65 g) en la merienda. La regla «no repitas la base»
solo miraba almuerzo↔cena."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import prompts.day_generator as dg  # noqa: E402


def _day(meals):
    return [{"day": 1, "meals": [{"meal": s, "name": n, "ingredients": list(i)} for s, n, i in meals]}]


def test_detects_the_live_case(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_BASE_REPEAT_ENABLED", True)
    days = _day([("Desayuno", "Bowl batido de piña, avena y yogur", ["80 g de avena", "1 taza de yogurt"]),
                 ("Merienda", "Vasito frío de higo, avena y mantequilla de maní", ["65 g de avena", "2 higos"]),
                 ("Almuerzo", "Ropa vieja de lentejas sobre pan integral", ["100 g de lentejas", "pan integral"])])
    issues = go._detect_light_base_repeats(days)
    assert len(issues) == 1 and issues[0].startswith("Día 1") and "avena" in issues[0], issues


def test_quiet_when_bases_differ_or_knob_off(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_BASE_REPEAT_ENABLED", True)
    ok = _day([("Desayuno", "Avena con fruta", ["60 g de avena"]),
               ("Merienda", "Yogur con almendras", ["1 taza de yogurt", "20 g de almendras"])])
    assert go._detect_light_base_repeats(ok) == []
    # el almuerzo con pan NO cuenta (solo franjas ligeras)
    ok2 = _day([("Desayuno", "Casabe con huevo", ["1 casabe"]),
                ("Almuerzo", "Sándwich de pan integral", ["pan integral"])])
    assert go._detect_light_base_repeats(ok2) == []
    monkeypatch.setattr(go, "LIGHT_BASE_REPEAT_ENABLED", False)
    bad = _day([("Desayuno", "Avena", ["60 g de avena"]), ("Merienda", "Avena", ["40 g de avena"])])
    assert go._detect_light_base_repeats(bad) == []


def test_prompt_has_the_rule():
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Huevo"], "carb_pool": ["Avena", "Papa"]}, 1)
    assert "TAMPOCO REPITAS LA BASE ENTRE DESAYUNO Y MERIENDA" in ctx


def test_wired_in_the_critique():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "{crossday_block}{light_base_block}{gm_dinner_block}" in src
    assert "and not _light_base_issues):" in src
    assert "tooltip-anchor: P2-LIGHT-BASE-NO-REPEAT" in src
