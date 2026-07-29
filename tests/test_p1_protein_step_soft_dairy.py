"""[P1-PROTEIN-STEP-SOFT-DAIRY · 2026-06-29] El closer de proteína inyectaba "💪 Cocina X a la plancha o hervido y sírvelo
como proteína del plato" SIN validar que X fuera cocinable así → "Cocina queso cottage a la plancha" (el cottage se desarma)
+ "como proteína del plato" miente cuando es añadido. Fix: para lácteos blandos (_NO_COOK_SAFE_PROTEIN_HINT) → "Incorpora X
a la preparación y mézclalo" (honesto sea principal o añadido). Proteínas cocinables (pollo/res/pescado) → siguen "a la
plancha". Tests PUROS (db=None; el candidato lleva los macros).
"""
from __future__ import annotations

import graph_orchestrator as g


class _Info:
    def __init__(self, name, protein, carbs=3.0, fats=2.0, kcal=98.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


def _meal():
    return {"name": "Revoltillo de Huevo", "protein": 5, "carbs": 10, "fats": 5, "cals": 150,
            "ingredients": ["2 huevos"], "recipe": ["Bate los huevos y cocínalos."]}


def _step_text(meal):
    return " ".join(str(s) for s in meal.get("recipe", [])).lower()


def _cold_meal():
    """[P3-SOFT-DAIRY-HOT-COLD · 2026-07-29] Plato FRÍO: es la rama donde el lácteo blando sí se
    INCORPORA. El fixture original era un 'Revoltillo', que `P1-HOT-DAIRY-ASIDE · 2026-07-28`
    reclasificó como plato caliente a propósito (el lácteo frío en un revoltillo caliente lo corta),
    así que este archivo llevaba dos tests rojos anclando el verbo que ese fix cambió."""
    return {"name": "Bowl de Avena Fría", "protein": 5, "carbs": 10, "fats": 5, "cals": 150,
            "ingredients": ["50 g de avena"], "recipe": ["Mezcla la avena con la leche fría."]}


def test_soft_dairy_never_says_plancha_nor_claims_main_protein():
    """Lo que P1-PROTEIN-STEP-SOFT-DAIRY protege: el cottage NO se cocina a la plancha (se desarma)
    y el paso no puede afirmar que es 'la proteína del plato' cuando es un añadido. El VERBO exacto
    lo decide la temperatura del plato (ver los dos tests de abajo); esto vale para ambas ramas."""
    for mk in (_meal, _cold_meal):
        m = mk()
        g._close_protein_gap_for_meal(m, 25, None,
                                      [(0.25, "Queso cottage", _Info("Queso cottage", 11))],
                                      slot_cal_target=480)
        t = _step_text(m)
        assert "💪" in " ".join(m["recipe"])
        assert "a la plancha" not in t and "hervido" not in t
        assert "proteína del plato" not in t and "proteina del plato" not in t


def test_soft_dairy_in_a_hot_dish_goes_aside():
    """[P1-HOT-DAIRY-ASIDE · 2026-07-28] 'Revoltillo' es plato CALIENTE: el lácteo frío que añade el
    closer va AL LADO, no mezclado (mezclarlo lo corta)."""
    m = _meal()   # Revoltillo de Huevo
    g._close_protein_gap_for_meal(m, 25, None, [(0.25, "Queso cottage", _Info("Queso cottage", 11))],
                                  slot_cal_target=480)
    t = _step_text(m)
    assert "al lado" in t, f"plato caliente → el lácteo va al lado: {t!r}"


def test_ricotta_and_yogur_incorporate_in_a_cold_dish():
    for name in ("Queso ricotta", "Yogurt griego"):
        m = _cold_meal()
        g._close_protein_gap_for_meal(m, 25, None, [(0.25, name, _Info(name, 10))], slot_cal_target=480)
        t = _step_text(m)
        assert "incorpora" in t and "a la plancha" not in t, f"{name}: {t!r}"


def test_cookable_protein_keeps_plancha():
    m = _meal()
    g._close_protein_gap_for_meal(m, 25, None, [(0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))], slot_cal_target=480)
    t = _step_text(m)
    assert "a la plancha" in t or "hervido" in t


def test_knob_off_reverts(monkeypatch):
    monkeypatch.setattr(g, "PROTEIN_STEP_SOFT_DAIRY_WORDING", False)
    m = _meal()
    g._close_protein_gap_for_meal(m, 25, None, [(0.25, "Queso cottage", _Info("Queso cottage", 11))], slot_cal_target=480)
    t = _step_text(m)
    assert "a la plancha" in t or "hervido" in t  # wording previo


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-PROTEIN-STEP-SOFT-DAIRY" in src
    assert "PROTEIN_STEP_SOFT_DAIRY_WORDING" in src
    assert g.PROTEIN_STEP_SOFT_DAIRY_WORDING is True
    # cottage/ricotta/yogur en la constante reusada
    assert "cottage" in g._NO_COOK_SAFE_PROTEIN_HINT and "ricotta" in g._NO_COOK_SAFE_PROTEIN_HINT
