"""[P1-COHERENCE-BANNER-NOISE · 2026-06-22] El banner "Lista revisada — N items
pueden necesitar ajuste manual" (recálculo + agent tool) debe surfacear SOLO
divergencias ACCIONABLES, no artefactos de magnitud sobre alimentos que SÍ están
en la lista.

Caso real (plan del owner, recalc a 15/30 días): aparecían 5 ítems "Causa
indeterminada" — ajo (receta "10 dientes" / lista "3 cabezas"), cebolla, cilantro,
perejil, ají morrón; y al filtrarlos afloraban cerdo/camarón (yield cocido↔crudo) y
plátano/guineo/aguacate (compra por unidad entera). Todos están CORRECTOS en la
lista — falso positivo.

`summarize_divergences_for_ui` ahora surface solo `cap_swallowed_modifier` (ausente
de la lista → "se te olvida comprarlo") y `pantry_overdeduct` (sub-suministro severo
→ "te quedas corto"). Omite `unknown`/`unit_mismatch`/`yield_uncovered` (artefactos
no accionables). El historial conserva TODAS las divergencias.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shopping_calculator import summarize_divergences_for_ui

_BACKEND = Path(__file__).resolve().parent.parent


def _foods(items):
    return {i["food"] for i in items}


def test_marker_present():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-COHERENCE-BANNER-NOISE" in src


def test_aromatic_unit_mismatch_dropped():
    """Ajo presente en otra unidad (par fantasma 'unknown' + 'unit_mismatch') → fuera del banner."""
    divs = [
        {"food": "Ajo", "magnitude": True, "hypothesis": "unknown",
         "expected_qty": 0.0, "actual_qty": 3.0, "delta_pct": float("inf")},
        {"food": "Ajo", "magnitude": True, "hypothesis": "unit_mismatch",
         "expected_qty": 10.0, "actual_qty": 0.0, "delta_pct": 1.0},
    ]
    assert summarize_divergences_for_ui(divs) == []


def test_yield_and_wholeunit_magnitude_dropped():
    """Cerdo (yield cocido/crudo) y guineo (unidad entera) = magnitud benigna → fuera."""
    divs = [
        {"food": "Cerdo", "magnitude": True, "hypothesis": "unknown", "expected_qty": 800, "actual_qty": 1150},
        {"food": "Guineo", "magnitude": True, "hypothesis": "unknown", "expected_qty": 120, "actual_qty": 350},
        {"food": "Limón", "magnitude": True, "hypothesis": "unit_mismatch", "expected_qty": 60, "actual_qty": 0},
        {"food": "Pollo", "magnitude": True, "hypothesis": "yield_uncovered", "expected_qty": 1000, "actual_qty": 1350},
    ]
    assert summarize_divergences_for_ui(divs) == []


def test_truly_missing_food_is_kept():
    """Alimento REALMENTE ausente de la lista (cap_swallowed_modifier) → accionable, se surface."""
    divs = [
        {"food": "Ajonjolí", "side": "presence", "magnitude": False,
         "hypothesis": "cap_swallowed_modifier", "expected_qty": 30.0, "actual_qty": 0.0},
    ]
    out = summarize_divergences_for_ui(divs)
    assert _foods(out) == {"Ajonjolí"}


def test_severe_undersupply_is_kept():
    """Sub-suministro SEVERO (pantry_overdeduct) → accionable ('te quedas corto'), se surface."""
    divs = [
        {"food": "Pechuga de pollo", "magnitude": True, "hypothesis": "pantry_overdeduct",
         "expected_qty": 1000.0, "actual_qty": 300.0, "delta_pct": -0.7},
    ]
    out = summarize_divergences_for_ui(divs)
    assert _foods(out) == {"Pechuga de pollo"}


def test_mixed_keeps_only_actionable():
    """Mezcla real: aromáticos/yield/unidad benignos fuera; ausente + sub-suministro severo dentro."""
    divs = [
        {"food": "Ajo", "magnitude": True, "hypothesis": "unknown", "expected_qty": 0.0, "actual_qty": 3.0},
        {"food": "Cebolla", "magnitude": True, "hypothesis": "unit_mismatch", "expected_qty": 60.0, "actual_qty": 0.0},
        {"food": "Camarón", "magnitude": True, "hypothesis": "unknown", "expected_qty": 200, "actual_qty": 280},
        {"food": "Aguacate", "magnitude": True, "hypothesis": "unknown", "expected_qty": 100, "actual_qty": 400},
        {"food": "Ajonjolí", "magnitude": False, "hypothesis": "cap_swallowed_modifier", "expected_qty": 30.0, "actual_qty": 0.0},
        {"food": "Habichuelas", "magnitude": True, "hypothesis": "pantry_overdeduct", "expected_qty": 900, "actual_qty": 200},
    ]
    foods = _foods(summarize_divergences_for_ui(divs))
    assert foods == {"Ajonjolí", "Habichuelas"}  # solo lo accionable


def test_max_items_respected():
    divs = [
        {"food": f"Item{i}", "magnitude": False, "hypothesis": "cap_swallowed_modifier",
         "expected_qty": 10.0, "actual_qty": 0.0}
        for i in range(10)
    ]
    assert len(summarize_divergences_for_ui(divs, max_items=3)) == 3


def test_empty_and_nondict_safe():
    assert summarize_divergences_for_ui([]) == []
    assert summarize_divergences_for_ui(None) == []
    out = summarize_divergences_for_ui(
        [None, "x", {"food": "Sal", "hypothesis": "cap_swallowed_modifier", "expected_qty": 5, "actual_qty": 0}]
    )
    assert _foods(out) == {"Sal"}
