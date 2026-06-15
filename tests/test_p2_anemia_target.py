"""[P2-ANEMIA-TARGET · 2026-06-15] Anemia ferropénica: condition_target advisory en el panel.

Decisión (P2-3): anemia es un DÉFICIT, no un ofensor → NO se añaden `substitutions` (un swap magro→res
perdería proteína y el catálogo es-DO no tiene hígado). La mejora segura es surfacear un condition_target
de hierro en el panel (paralelo a DM2/HTA/dislipidemia), advisory, sin elevar el piso (RDA 18F/8M ya
vigente) → imposible de loopear. Default OFF (user-facing).
"""
from __future__ import annotations

import pytest

import condition_rules as cr
import micronutrients as mn
from nutrition_db import IngredientNutritionDB

_ROWS = [
    {"name": "Espinaca", "aliases": ["espinaca", "espinacas"], "kcal_per_100g": 23,
     "protein_g_per_100g": 2.9, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4,
     "fiber_g_per_100g": 2.2, "iron_mg_per_100g": 2.7, "potassium_mg_per_100g": 558},
]


def _db():
    return IngredientNutritionDB(rows=_ROWS)


def _plan():
    return {"days": [{"meals": [{"ingredients": ["100g de espinaca (100g)"]}]}]}


# ── _has_anemia ──
@pytest.mark.parametrize("conds,expected", [
    (["anemia"], True), (["Anemia ferropénica"], True), (["ferropenica"], True),
    (["ferritina baja"], True), (["diabetes"], False), ([], False),
])
def test_has_anemia(conds, expected):
    assert mn._has_anemia(conds) is expected


# ── La fila de anemia NO tiene substitutions (decisión P2-3) ──
def test_anemia_rule_has_no_substitutions():
    assert cr._RULES_BY_ID["anemia"].substitutions == (), "anemia es déficit, no ofensor — sin swaps (P2-3)"


# ── Knob default OFF ──
def test_knob_default_off():
    assert mn._ANEMIA_CONDITION_TARGET_ENABLED is False


# ── condition_target gated ──
def test_anemia_condition_target_when_enabled(monkeypatch):
    monkeypatch.setattr(mn, "_ANEMIA_CONDITION_TARGET_ENABLED", True)
    rep = mn.build_micronutrient_report(_plan(), _db(), sex="female", conditions=["anemia"], daily_kcal=1800)
    conds = [c.get("condicion") for c in rep.get("condition_targets", [])]
    assert "Anemia ferropénica" in conds


def test_no_anemia_target_when_disabled(monkeypatch):
    monkeypatch.setattr(mn, "_ANEMIA_CONDITION_TARGET_ENABLED", False)
    rep = mn.build_micronutrient_report(_plan(), _db(), sex="female", conditions=["anemia"], daily_kcal=1800)
    conds = [c.get("condicion") for c in rep.get("condition_targets", [])]
    assert "Anemia ferropénica" not in conds


def test_marker_present():
    from pathlib import Path
    src = Path(mn.__file__).read_text(encoding="utf-8")
    assert "P2-ANEMIA-TARGET" in src
    assert "def _has_anemia(" in src
