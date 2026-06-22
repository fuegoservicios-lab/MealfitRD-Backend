"""[P1-COHERENCE-BANNER-NOISE · 2026-06-22] El banner "Lista revisada — N items
pueden necesitar ajuste manual" del recálculo NO debe marcar aromáticos/sazón que
SÍ están en la lista (en una unidad de compra distinta a la de la receta).

Caso real (plan del owner, recalc a 15 días): ajo (receta "10 dientes" / lista "3
cabezas"), cebolla (receta "60 g" / lista por peso), cilantro/perejil (receta "g" /
lista "mazo"), ají morrón (receta "g" / lista "unidad") aparecían como "Causa
indeterminada". El food ESTÁ en la lista → falso positivo.

`summarize_divergences_for_ui` omite del banner los foods con "fantasma"
(expected≈0, actual>0 = presente en lista). NO oculta sub-suministros reales
(expected>0 Y actual>0 en la MISMA unidad, sin fantasma). El historial conserva todo.
"""
from __future__ import annotations

from pathlib import Path

from shopping_calculator import summarize_divergences_for_ui

_BACKEND = Path(__file__).resolve().parent.parent


def _foods(items):
    return {i["food"] for i in items}


def test_marker_present():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-COHERENCE-BANNER-NOISE" in src


def test_aromatic_present_in_list_is_dropped():
    """Ajo con par fantasma+unit_mismatch (presente en lista) → omitido del banner."""
    divs = [
        # fantasma: la lista tiene 3 cabezas, receta esperaba 0 en esa unidad
        {"food": "Ajo", "side": "magnitude", "magnitude": True, "hypothesis": "unknown",
         "expected_qty": 0.0, "actual_qty": 3.0, "delta_pct": float("inf")},
        # faltante en la unidad de receta: receta pide 10 dientes, lista 0 dientes
        {"food": "Ajo", "side": "magnitude", "magnitude": True, "hypothesis": "unit_mismatch",
         "expected_qty": 10.0, "actual_qty": 0.0, "delta_pct": 1.0},
    ]
    out = summarize_divergences_for_ui(divs)
    assert out == [], "Ajo está en la lista (fantasma) → no debe salir en el banner"


def test_truly_missing_food_is_kept():
    """Un food REALMENTE ausente de la lista (sin fantasma) sí se surface."""
    divs = [
        {"food": "Ajonjolí", "side": "presence", "magnitude": False, "hypothesis": "cap_swallowed_modifier",
         "expected_qty": 30.0, "actual_qty": 0.0},
    ]
    out = summarize_divergences_for_ui(divs)
    assert _foods(out) == {"Ajonjolí"}


def test_same_unit_undersupply_is_kept():
    """Sub-suministro real (misma unidad, exp>0 Y act>0, sin fantasma) NO se oculta."""
    divs = [
        {"food": "Pechuga de pollo", "side": "magnitude", "magnitude": True, "hypothesis": "yield_uncovered",
         "expected_qty": 1000.0, "actual_qty": 500.0, "delta_pct": -0.5},
    ]
    out = summarize_divergences_for_ui(divs)
    assert _foods(out) == {"Pechuga de pollo"}


def test_mixed_drops_only_present():
    """Mezcla: aromáticos presentes se omiten, faltante real + magnitud real se conservan."""
    divs = [
        {"food": "Ajo", "magnitude": True, "hypothesis": "unknown", "expected_qty": 0.0, "actual_qty": 3.0},
        {"food": "Ajo", "magnitude": True, "hypothesis": "unit_mismatch", "expected_qty": 10.0, "actual_qty": 0.0},
        {"food": "Cebolla", "magnitude": True, "hypothesis": "unknown", "expected_qty": 0.0, "actual_qty": 453.0},
        {"food": "Cilantro", "magnitude": True, "hypothesis": "unknown", "expected_qty": 0.0, "actual_qty": 1.0},
        {"food": "Ajonjolí", "magnitude": False, "hypothesis": "cap_swallowed_modifier", "expected_qty": 30.0, "actual_qty": 0.0},
        {"food": "Ají cubanela", "magnitude": True, "hypothesis": "unknown", "expected_qty": 0.5, "actual_qty": 2.0, "delta_pct": 3.0},
    ]
    out = summarize_divergences_for_ui(divs)
    foods = _foods(out)
    assert "Ajo" not in foods and "Cebolla" not in foods and "Cilantro" not in foods
    assert "Ajonjolí" in foods       # realmente ausente → accionable
    assert "Ají cubanela" in foods   # exp>0 Y act>0 sin fantasma → magnitud real, se conserva


def test_max_items_respected():
    divs = [
        {"food": f"Item{i}", "magnitude": False, "hypothesis": "cap_swallowed_modifier",
         "expected_qty": 10.0, "actual_qty": 0.0}
        for i in range(10)
    ]
    out = summarize_divergences_for_ui(divs, max_items=3)
    assert len(out) == 3


def test_empty_and_nondict_safe():
    assert summarize_divergences_for_ui([]) == []
    assert summarize_divergences_for_ui(None) == []
    out = summarize_divergences_for_ui([None, "x", {"food": "Sal", "hypothesis": "cap_swallowed_modifier", "expected_qty": 5, "actual_qty": 0}])
    assert _foods(out) == {"Sal"}
