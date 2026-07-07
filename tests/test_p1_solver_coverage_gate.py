"""[P1-SOLVER-COVERAGE-GATE · 2026-07-07] El solver per-meal se abstiene / capea cuando el catálogo
NO ve suficiente masa de la comida.

Bug de raíz (audit solver+seeder 2026-07-07): `solve_meal_macros` arma su LSQ SOLO con las líneas
resueltas por el catálogo. Una comida con masa NO-resuelta (plato compuesto/criollo invisible al
catálogo) + guarniciones resolubles → el LSQ infla las guarniciones (×hasta 3.5) para clavar el target
del slot ELLAS SOLAS → el plato SOBRE-entrega (guarniciones infladas + masa no-resuelta intacta),
mientras `_apply_macro_solver_to_meal` sobrescribe el display con la suma resuelta-sola = "on-target"
engañoso. Nadie downstream lo caza (rebalance/truth-up leen esos números o tampoco ven la masa).

Fix (espejo de la abstención de `_truth_up_meal_macros_from_catalog`, MISMO min_coverage 0.6 y criterio
de líneas-con-cantidad): cobertura < piso → abstiene (return False, comida intacta; el closer/rebalance
downstream dimensionan); cobertura parcial (<1.0) → cap de escala más estricto (SOLVER_PARTIAL_MAX_SCALE)
para que las resueltas no se inflen cubriendo masa invisible.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _FakeDB:
    """pollo/arroz/aguacate resueltos (macros absolutos por línea); sancocho/moro NO-resueltos (bulk)."""

    _MAC = {
        "pollo": {"protein": 30.0, "carbs": 0.0, "fats": 3.0, "kcal": 150.0},
        "arroz": {"protein": 3.0, "carbs": 40.0, "fats": 0.5, "kcal": 180.0},
        "aguacate": {"protein": 2.0, "carbs": 9.0, "fats": 15.0, "kcal": 160.0},
    }

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        for tok, mac in self._MAC.items():
            if tok in low:
                return dict(mac)
        return None  # sancocho/moro/etc → masa no-resuelta (bulk invisible)

    def macros_for_line(self, qty, unit, name):
        return self.macros_from_ingredient_string(name)

    def grams_from_ingredient_string(self, s):
        return 100.0


# ─────────────────────────── Parser anchors ───────────────────────────

def test_marker_anchored_in_source():
    assert "P1-SOLVER-COVERAGE-GATE" in _GO


def test_gate_uses_same_criterion_as_catalog_truthup():
    """El gate reusa el criterio de cobertura del catalog truth-up (líneas resueltas / líneas-con-
    cantidad, excluyendo condimentos vía _TRUTHUP_CONDIMENT_HINTS) y se abstiene bajo el piso."""
    i = _GO.index("def _apply_macro_solver_to_meal")
    win = _GO[i:i + 3800]
    assert "SOLVER_MIN_COVERAGE" in win
    assert "_TRUTHUP_CONDIMENT_HINTS" in win
    assert "SOLVER_PARTIAL_MAX_SCALE" in win
    # abstención bajo el piso (return False) + cap parcial en el else de <0.999.
    assert "_cov < SOLVER_MIN_COVERAGE" in win
    assert "return False" in win


def test_knobs_defined_with_validators():
    assert 'SOLVER_MIN_COVERAGE = _env_float("MEALFIT_SOLVER_MIN_COVERAGE", 0.6' in _GO
    assert 'SOLVER_PARTIAL_MAX_SCALE = _env_float("MEALFIT_SOLVER_PARTIAL_MAX_SCALE", 2.0' in _GO


# ─────────────────────────── Knob defaults ───────────────────────────

def test_knob_defaults():
    import graph_orchestrator as g
    assert g.SOLVER_MIN_COVERAGE == pytest.approx(0.6)
    assert g.SOLVER_PARTIAL_MAX_SCALE == pytest.approx(2.0)


# ─────────────────────────── Funcional ───────────────────────────

def _slot(protein):
    return {"kcal": 600.0, "protein": float(protein), "carbs": 60.0, "fats": 20.0}


def test_low_coverage_meal_abstains_intact():
    """1/3 líneas resueltas (sancocho + moro = bulk invisible, solo aguacate resuelve) → 0.33 < 0.6 →
    el solver NO toca la comida (evita inflar el aguacate para cubrir el sancocho invisible)."""
    import graph_orchestrator as g
    meal = {
        "name": "Sancocho dominicano",
        "ingredients": ["1 plato de sancocho", "1 plato de moro de guandules", "80 g de aguacate"],
        "ingredients_raw": ["1 plato de sancocho", "1 plato de moro de guandules", "80 g de aguacate"],
        "protein": 12, "carbs": 40, "fats": 20, "cals": 400,
    }
    before = list(meal["ingredients"])
    changed = g._apply_macro_solver_to_meal(meal, _slot(40), _FakeDB())
    assert changed is False
    assert meal["ingredients"] == before  # comida intacta (abstención)


def test_full_coverage_meal_scales():
    """2/2 resueltas → cobertura 1.0 → el solver escala la proteína sub-entregada hacia el target."""
    import graph_orchestrator as g
    meal = {
        "name": "Pollo con arroz",
        "ingredients": ["100 g de pollo", "100 g de arroz"],
        "ingredients_raw": ["100 g de pollo", "100 g de arroz"],
        "protein": 33, "carbs": 40, "fats": 4, "cals": 330,
    }
    changed = g._apply_macro_solver_to_meal(meal, _slot(55), _FakeDB())
    assert changed is True
    assert meal["protein"] > 33  # escaló el pollo hacia el target (base 30 → ~55)


def test_partial_coverage_caps_scale_at_partial_max():
    """2/3 resueltas (0.67 ≥ 0.6 pero < 1.0) con target de proteína EXTREMO → el pollo se capea a
    SOLVER_PARTIAL_MAX_SCALE (2.0×) en vez de 3.5×, acotando la sobre-entrega sobre masa invisible."""
    import graph_orchestrator as g
    meal = {
        "name": "Pollo con sancocho",
        "ingredients": ["50 g de pollo", "1 plato de sancocho", "60 g de arroz"],
        "ingredients_raw": ["50 g de pollo", "1 plato de sancocho", "60 g de arroz"],
        "protein": 18, "carbs": 24, "fats": 2, "cals": 210,
    }
    changed = g._apply_macro_solver_to_meal(meal, _slot(100), _FakeDB())  # target imposible → maxea el clamp
    assert changed is True
    # base proteína resoluble = pollo 30 + arroz 3; cap parcial 2.0× → ≲ 66 (con 3.5× habría llegado a
    # ~100+). El rango prueba que se escaló hacia el target PERO acotado por SOLVER_PARTIAL_MAX_SCALE.
    assert 40 < meal["protein"] <= 70, f"cap parcial no aplicado: proteína={meal['protein']}"
