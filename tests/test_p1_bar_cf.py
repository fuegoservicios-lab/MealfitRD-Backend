"""[P1-BAR-CF · 2026-06-28] Los caps clínicos de porción (DM2/bariátrico) vivían SOLO inline en assemble_plan_node. Los
paths degradados/partial (fallback matemático, partial reparado, post-swap) BYPASSAN assemble y entregaban porciones sin
capear ("920 g de queso fresco" visto en vivo). Esos 3 paths SÍ pasan por la SSOT clínica _apply_deterministic_clinical_
layer (con form_data) → añadir los caps ahí los cubre. Idempotente (no-op donde assemble ya capeó), no-op no-bariátrico.
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


class _DB:
    def macros_from_ingredient_string(self, s):
        return {}

    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+(?:\.\d+)?)\s*g", str(s))
        return float(m.group(1)) if m else None


def test_caps_wired_in_clinical_layer():
    # los caps aparecen DENTRO de _apply_deterministic_clinical_layer, antes de marcarla aplicada
    i = _SRC.index("def _apply_deterministic_clinical_layer")
    j = _SRC.index('plan["_clinical_layer_applied"] = True', i)
    body = _SRC[i:j]
    assert "cap_bariatric_portions(" in body
    assert "cap_dm2_high_gi_portions(" in body
    assert "P1-BAR-CF" in body
    assert "BARIATRIC_CLINICAL_FINALIZE_ENABLED" in body


def test_cap_bariatric_caps_920g_cheese():
    # el cap en sí atrapa "920 g de queso fresco" (catálogo no resuelve → grams líderes)
    days = [{"day": 1, "meals": [{"meal": "Merienda", "name": "Q", "ingredients": ["920 g de queso fresco"]}]}]
    n = g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, _DB())
    assert n >= 1
    grams = float(re.search(r"(\d+(?:\.\d+)?)\s*g", days[0]["meals"][0]["ingredients"][0]).group(1))
    assert grams <= g.BARIATRIC_CHEESE_CAP_G


def test_clinical_layer_calls_caps_when_bariatric(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    called = {"n": 0}
    _orig = g.cap_bariatric_portions

    def _spy(days, fd, db=None):
        called["n"] += 1
        return _orig(days, fd, _DB())
    monkeypatch.setattr(g, "cap_bariatric_portions", _spy)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Q", "ingredients": ["920 g de queso fresco"]}]}]}
    g._apply_deterministic_clinical_layer(plan, {"medicalConditions": ["Cirugía Bariátrica"], "allergies": []})
    assert called["n"] >= 1
    grams = float(re.search(r"(\d+(?:\.\d+)?)\s*g", plan["days"][0]["meals"][0]["ingredients"][0]).group(1))
    assert grams <= g.BARIATRIC_CHEESE_CAP_G  # capeado en path degradado


def test_non_bariatric_noop(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Q", "ingredients": ["920 g de queso fresco"]}]}]}
    g._apply_deterministic_clinical_layer(plan, {"medicalConditions": ["Ninguna"], "allergies": []})
    assert "920" in plan["days"][0]["meals"][0]["ingredients"][0]  # no-bariátrico → intacto


def test_knob_default_on():
    assert g.BARIATRIC_CLINICAL_FINALIZE_ENABLED is True
