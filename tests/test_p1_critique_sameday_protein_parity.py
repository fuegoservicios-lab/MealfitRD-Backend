"""[P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY · 2026-07-08] Verificación post-corrección del self_critique.

Forense del plan vivo 70f802ec (guest, gain_muscle, 2026-07-08 23:22-23:27): el self_critique detectó
proteína repetida almuerzo↔cena (Día 2 pescado, Día 3 cerdo) y corrigió los días con deepseek-v4-pro
(3 llamadas, 63s), pero el gate DETERMINISTA del revisor (`build_variety_report.same_day_protein_repeats`
→ P1-VARIETY-SAME-DAY-PROTEIN) VOLVIÓ a rechazar por la MISMA causa. La corrección se logueaba "corregido"
con que el LLM devolviera algo no-nulo — NUNCA se re-verificaba con el detector del gate. Como el gate
determinista hace *bypass* del LLM reviewer (guest sin restricciones) no puebla `affected_days` → el retry
regeneró el PLAN COMPLETO (esqueleto nuevo + 3 días), no quirúrgico.

Este es el MISMO bug-class ya cerrado para FRUTA el 2026-07-05 (`P1-FRUIT-DEDUP-GATE-PARITY`,
`_plan_has_same_day_fruit_repeat`) — faltaba la paridad para PROTEÍNA.

Fix: tras el loop de corrección, `_days_with_same_day_protein_repeat(plan)` (MISMO SSOT que el gate) marca
los días residuales `_critique_unresolved` → el retry se vuelve QUIRÚRGICO (P1-SURGICAL-1 regenera solo esos
días). Knob `MEALFIT_SELF_CRITIQUE_VERIFY_SAME_DAY_PROTEIN` (default True).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY" in _GO


def test_knob_defined_default_on():
    assert (
        'SELF_CRITIQUE_VERIFY_SAME_DAY_PROTEIN = _env_bool("MEALFIT_SELF_CRITIQUE_VERIFY_SAME_DAY_PROTEIN", True)'
        in _GO
    )


def test_helper_defined():
    assert "def _days_with_same_day_protein_repeat(plan" in _GO


def test_hook_marks_unresolved_with_reason():
    """El hook post-corrección marca los días residuales con la razón canónica."""
    assert "same_day_protein_repeat_unresolved" in _GO
    # El hook debe usar el SSOT (llamar al helper) dentro del nodo self_critique.
    i_helper_call = _GO.index("_days_with_same_day_protein_repeat(partial)")
    i_mark = _GO.index("same_day_protein_repeat_unresolved")
    assert abs(i_mark - i_helper_call) < 1500, "el marcado debe seguir a la verificación"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def g():
    import graph_orchestrator as _g
    return _g


def _meal(meal_type, name, ingredients):
    return {"meal": meal_type, "name": name, "ingredients": ingredients}


def test_detects_same_day_heavy_protein_lunch_dinner(g):
    """Día 2: pescado en almuerzo Y cena → detectado (caso vivo 70f802ec)."""
    plan = {"days": [
        {"day": 1, "meals": [
            _meal("desayuno", "Avena con guineo", ["avena", "guineo"]),
            _meal("almuerzo", "Pollo guisado con arroz", ["pechuga de pollo", "arroz"]),
            _meal("cena", "Ensalada de queso fresco", ["queso fresco", "lechuga"]),
        ]},
        {"day": 2, "meals": [
            _meal("desayuno", "Mangú con huevo", ["platano verde", "huevo"]),
            _meal("almuerzo", "Pescado guisado", ["filete de pescado blanco", "cebolla"]),
            _meal("cena", "Sopa de pescado", ["pescado", "yuca"]),
        ]},
    ]}
    assert g._days_with_same_day_protein_repeat(plan) == [2]


def test_detects_same_day_egg(g):
    """Huevo en desayuno + cena el mismo día → detectado (egg está en los labels del gate)."""
    plan = {"days": [
        {"day": 1, "meals": [
            _meal("desayuno", "Revoltillo de huevo", ["huevo", "cebolla"]),
            _meal("almuerzo", "Res guisada con arroz", ["carne molida", "arroz"]),
            _meal("cena", "Tortilla de huevo", ["huevo", "tomate"]),
        ]},
    ]}
    assert g._days_with_same_day_protein_repeat(plan) == [1]


def test_no_repeat_distinct_proteins(g):
    """Proteína distinta por comida → vacío."""
    plan = {"days": [
        {"day": 1, "meals": [
            _meal("desayuno", "Avena con maní", ["avena", "mani"]),
            _meal("almuerzo", "Pollo al horno", ["pechuga de pollo"]),
            _meal("cena", "Pescado a la plancha", ["pescado"]),
        ]},
    ]}
    assert g._days_with_same_day_protein_repeat(plan) == []


def test_word_boundary_no_false_positive_res_in_fresas(g):
    """'res' NO debe matchear 'fresas'/'queso fresco' (word-boundary, P1-SLOT-PROTEIN-WORDBOUNDARY)."""
    plan = {"days": [
        {"day": 1, "meals": [
            _meal("desayuno", "Ensalada de fresas", ["fresas", "yogur"]),
            _meal("almuerzo", "Bowl con queso fresco", ["queso fresco", "lechuga"]),
            _meal("cena", "Batida de fresas", ["fresas", "leche"]),
        ]},
    ]}
    assert g._days_with_same_day_protein_repeat(plan) == []


def test_parity_with_gate_ssot(g):
    """SSOT: si el helper marca un día, build_variety_report cuenta ≥1 same_day_protein_repeat."""
    plan = {"days": [
        {"day": 1, "meals": [
            _meal("almuerzo", "Cerdo guisado", ["lomo de cerdo", "arroz"]),
            _meal("cena", "Cerdo desmenuzado", ["cerdo", "yuca"]),
        ]},
    ]}
    residual = g._days_with_same_day_protein_repeat(plan)
    report = g.build_variety_report(plan)
    assert residual == [1]
    assert int(report.get("same_day_protein_repeats", 0)) >= 1


def test_fail_safe_on_garbage(g):
    """Fail-safe: input basura → [] sin excepción."""
    assert g._days_with_same_day_protein_repeat({}) == []
    assert g._days_with_same_day_protein_repeat({"days": [None, 42, "x"]}) == []
