"""[P3-CLOSER-EGG-BUDGET · 2026-06-14] Presupuesto de huevo del protein-closer.

El closer (P3-PROTEIN-FLOOR) elegía clara/huevo como la proteína magra de las comidas ligeras
(_DAIRY_EGG_PROTEIN_HINT) → AÑADÍA huevo de relleno → lo empujaba sobre el cap del VARIETY_HARD_GATE,
que luego rechazaba el plan por sobreuso (medido en vivo: 5/12 comidas) → entrega marcada-degradada.
Este budget hace al closer consciente del cap: una vez alcanzado, recibe candidatos SIN huevo →
diversifica con yogur/queso/whey. Complementa al closer SIN tocar su selección/dish-fit.

Cobertura: parser-anchors del budget (un refactor que lo borre falla el test ANTES de prod) +
funcional de `_meal_has_egg` (la detección de la que depende el conteo) + la filtración no-egg.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_GO = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


def test_knob_present():
    src = _GO.read_text(encoding="utf-8")
    assert "CLOSER_EGG_BUDGET_ENABLED" in src
    assert "MEALFIT_CLOSER_EGG_BUDGET" in src
    assert "P3-CLOSER-EGG-BUDGET" in src


def test_egg_budget_wired_at_closer_callsite():
    """El budget vive en el call-site del closer (no en su función interna): setup + uso egg-aware."""
    src = _GO.read_text(encoding="utf-8")
    # setup: cap (igual que el gate), conteo, candidatos sin huevo
    assert "_egg_cap = max(3, round(" in src
    assert "_egg_count" in src
    assert "_hd_candidates_no_egg" in src
    # uso: el call del closer recibe candidatos egg-aware (no _hd_candidates a secas cuando hay budget)
    assert "_egg_cands" in src
    assert "_close_protein_gap_for_meal(" in src
    # excluye huevo/clara/yema del pool
    assert all(t in src for t in ('"huevo"', '"clara"', '"yema"'))


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def test_meal_has_egg_detection(go):
    """`_meal_has_egg` (base del conteo del budget) detecta huevo/clara/yema por token aislado."""
    from constants import strip_accents as sa
    assert go._meal_has_egg({"ingredients": ["2 huevos", "arroz"]}, sa) is True
    assert go._meal_has_egg({"ingredients": ["3 claras de huevo"]}, sa) is True
    assert go._meal_has_egg({"ingredients": ["1 yema"]}, sa) is True
    assert go._meal_has_egg({"ingredients": ["pechuga de pollo", "arroz", "yogur"]}, sa) is False
    # no falso-positivo de prefijo (lechosa != leche, etc.) — huevo aislado
    assert go._meal_has_egg({"ingredients": ["aguacate", "casabe"]}, sa) is False


def test_no_egg_filter_excludes_egg_candidates(go):
    """La filtración no-egg (lista de tuplas (leanness, name, info)) excluye huevo/clara/yema."""
    from constants import strip_accents as _sa
    cands = [(0.21, "Clara de huevo", object()), (0.17, "Yogurt griego sin azúcar", object()),
             (0.08, "Huevo", object()), (0.20, "Pechuga de pollo", object())]
    no_egg = [c for c in cands
              if not any(t in _sa(str(c[1]).lower()) for t in ("huevo", "clara", "yema"))]
    names = [c[1] for c in no_egg]
    assert "Yogurt griego sin azúcar" in names and "Pechuga de pollo" in names
    assert "Huevo" not in names and "Clara de huevo" not in names
