"""[P1-MEAL-CHEESE-CAP · 2026-07-07] Techo de queso en COMIDAS PRINCIPALES.

Review visual (plan 5610de53, ganancia muscular): un batido con "245 g de queso" ≈ 68g
grasa en UNA comida → la grasa del día quedó 148% del target (86g vs 58g), en banda solo
1/3 días → banner "precisión de macros por debajo de la banda". El fat-trimmer NO toca el
queso (proteína-dominante) por diseño, y FAT-LEAN-SWAP está apagado. El cap de queso solo
cubría meriendas (SNACK_CHEESE_CAP_G=120), no comidas principales.

Fix: cap de queso en comidas principales (MEAL_CHEESE_CAP_G=180, generoso → solo caza
bombas extremas como 245g, no el uso normal ≤150g). Yogurt exento. SENSIBLE AL BENCHMARK
de macros — knob para tunear.
tooltip-anchor: P1-MEAL-CHEESE-CAP
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


_GRAPH = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


# --- Parser-based ---
def test_marker_and_knob_present():
    assert "P1-MEAL-CHEESE-CAP" in _GRAPH
    assert 'MEALFIT_MEAL_CHEESE_CAP_G' in _GRAPH
    assert 'MEAL_CHEESE_CAP_G' in _GRAPH


def test_meal_cap_branch_excludes_merienda():
    """El branch nuevo cappea queso en NO-merienda (la merienda usa su propio cap)."""
    assert re.search(
        r'cur_g > float\(MEAL_CHEESE_CAP_G\)\s*\n\s*and "merienda" not in',
        _GRAPH,
    ), "el branch del meal cheese cap debe excluir merienda"


# --- Funcional ---
def test_main_meal_cheese_capped(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Batido Proteico de Guanábana con Queso",
        "ingredients": ["245 g de queso", "1½ tazas de guanábana"],
        "ingredients_raw": ["245 g de queso", "1½ tazas de guanábana"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    q = days[0]["meals"][0]["ingredients"][0]
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", q).group(1).replace(",", "."))
    assert n <= float(go.MEAL_CHEESE_CAP_G) + 0.01, f"queso de comida principal al techo: {q}"


def test_normal_cheese_not_capped(go, monkeypatch):
    """Uso normal de queso (150g < 180g) en comida principal NO se toca."""
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Ensalada con Queso Blanco",
        "ingredients": ["150 g de queso blanco", "2 tazas de lechuga"],
        "ingredients_raw": ["150 g de queso blanco", "2 tazas de lechuga"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    go._cap_unrealistic_portions(days, db=_NoopDB())
    assert days[0]["meals"][0]["ingredients"][0].startswith("150 g de queso"), \
        "150g de queso está bajo el techo (180g) → intacto"


def test_yogurt_in_main_meal_exempt(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Bowl de Yogurt",
        "ingredients": ["250 g de yogurt griego"],
        "ingredients_raw": ["250 g de yogurt griego"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    go._cap_unrealistic_portions(days, db=_NoopDB())
    assert days[0]["meals"][0]["ingredients"][0].startswith("250 g de yogurt"), \
        "yogurt exento del cap de queso"
