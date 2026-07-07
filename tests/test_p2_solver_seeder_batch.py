"""[P2-SOLVER-SEEDER-BATCH · 2026-07-07] Los 5 P2 del audit solver+seeder v2:

- S-P2-a (P2-SOLVER-SCALE-KNOBS): clamp del solver → knobs + max_scale mayor para proteína-dominantes
  (hi per-coordenada en _box_lsq) + saturación exacta desde el solver.
- S-P2-b (P2-SOLVER-METHOD-OBS): converged/method dejan de ser telemetría muerta → flag + log.
- S-P2-c (P2-REFINE-HOUSEHOLD): el refinador alcanza líneas en unidad casera (knob OFF, A/B pendiente).
- Sd-P2-b (P2-FATSWAP-BAND-GUARD): el fatswap no revienta la banda de grasa + estados de log distintos.
- Sd-P2-c (P2-MICRO-RESIDUAL-NOCARRIER): residual-log del caso día-sin-portador-no-sembrable.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "portion_solver.py"), encoding="utf-8") as f:
    _PS = f.read()
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ═══════════════════════ S-P2-a: clamp knobs + protein max_scale ═══════════════════════

def test_sp2a_knobs_defined():
    assert 'SOLVER_MIN_SCALE = _envf("MEALFIT_SOLVER_MIN_SCALE", 0.3)' in _PS
    assert 'SOLVER_MAX_SCALE = _envf("MEALFIT_SOLVER_MAX_SCALE", 3.5)' in _PS
    assert 'SOLVER_MAX_SCALE_PROTEIN = _envf("MEALFIT_SOLVER_MAX_SCALE_PROTEIN", 5.0)' in _PS
    import portion_solver as ps
    assert (ps.SOLVER_MIN_SCALE, ps.SOLVER_MAX_SCALE, ps.SOLVER_MAX_SCALE_PROTEIN) == (0.3, 3.5, 5.0)


def test_sp2a_protein_gets_higher_max_scale():
    """La línea PROTEÍNA-dominante puede escalar por encima del clamp general (3.5) hasta 5.0; la
    carbo-dominante queda capeada en 3.5 (hi per-coordenada)."""
    import portion_solver as ps
    ents = [
        {"macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 4}, "group": "protein"},
        {"macros": {"kcal": 130, "protein": 3, "carbs": 28, "fats": 0.5}, "group": "carbs"},
    ]
    tgt = {"kcal": 99999, "protein": 99999, "carbs": 99999, "fats": 0}  # imposible → maxea ambos
    f, method, sat_hi, sat_lo = ps._compute_scale_factors(ents, tgt, 0.3, 3.5, 5.0)
    assert method == "lsq"
    assert f[0] > 3.5, f"proteína debería pasar de 3.5: {f[0]}"
    assert f[1] <= 3.501, f"carbo debería capear en 3.5: {f[1]}"
    assert sat_hi >= 1  # al menos uno clavado en su bound superior


def test_sp2a_protein_rollback_equals_general():
    """Rollback (MEALFIT_SOLVER_MAX_SCALE_PROTEIN=3.5) → la proteína se capea como el general."""
    import portion_solver as ps
    ents = [{"macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 4}, "group": "protein"}]
    f, _, _, _ = ps._compute_scale_factors(ents, {"kcal": 99999, "protein": 99999, "carbs": 0, "fats": 0},
                                           0.3, 3.5, 3.5)
    assert f[0] <= 3.501


def test_sp2a_box_lsq_per_coordinate_hi():
    """_box_lsq acepta hi por-coordenada (lista) además de escalar."""
    import portion_solver as ps
    xs = ps._box_lsq([[10.0, 10.0]], [999.0], [1.0], 0.3, [5.0, 3.5], 0.1)
    assert xs[0] <= 5.001 and xs[1] <= 3.501


def test_sp2a_solve_meal_returns_saturation():
    import portion_solver as ps
    res = ps.solve_meal_macros(["100 g de pollo"], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 6},
                               db=_FakeDB())
    assert "saturated_hi" in res and "saturated_lo" in res


# ═══════════════════════ S-P2-b: converged/method observability ═══════════════════════

def test_sp2b_method_obs_anchored():
    assert "P2-SOLVER-METHOD-OBS" in _GO
    i = _GO.index("P2-SOLVER-METHOD-OBS")
    win = _GO[i:i + 900]
    assert "_solver_greedy_fallback" in win
    assert "_solver_not_converged" in win
    assert 'res.get("method") == "greedy"' in win
    # y vive dentro del wiring del solver.
    assert _GO.index("def _apply_macro_solver_to_meal") < i < _GO.index("def _protein_topup_meal")


def test_sp2b_greedy_fallback_flagged(monkeypatch):
    """Con el LSQ forzado OFF, el solver cae al greedy → el wiring lo flaguea (antes: silencioso)."""
    import graph_orchestrator as g
    import portion_solver as ps
    monkeypatch.setattr(ps, "SOLVER_LSQ", False)
    meal = {"name": "Pollo con arroz", "ingredients": ["100 g de pollo", "100 g de arroz"],
            "ingredients_raw": ["100 g de pollo", "100 g de arroz"],
            "protein": 33, "carbs": 40, "fats": 4, "cals": 330}
    g._apply_macro_solver_to_meal(meal, {"kcal": 600, "protein": 55, "carbs": 60, "fats": 15}, _FakeDB())
    assert meal.get("_solver_greedy_fallback") is True


# ═══════════════════════ S-P2-c: refiner household lines ═══════════════════════

def test_sp2c_household_knob_default_off():
    import portion_solver as ps
    assert ps.REFINE_HOUSEHOLD_LINES is False
    assert 'REFINE_HOUSEHOLD_LINES = _envb("MEALFIT_REFINE_HOUSEHOLD_LINES", False)' in _PS


def test_sp2c_household_line_moves_only_when_enabled(monkeypatch):
    """Día con SOLO una línea en unidad casera fuera de banda: OFF → 0 movimientos (ignorada);
    ON → el refinador la mueve (grams desde el hint + re-render quantize)."""
    import portion_solver as ps
    targets = {"kcal": 100, "protein": 2, "carbs": 25, "fats": 1}  # el día entrega ~2× esto → mover abajo

    def _mk():
        return [{"ingredients": ["2 tazas de arroz (300g)"], "ingredients_raw": ["2 tazas de arroz (300g)"]}]

    monkeypatch.setattr(ps, "REFINE_HOUSEHOLD_LINES", False)
    assert ps.refine_day_portions_integer(_mk(), targets, _FakeDB(), floor_g=15.0, cap_g=400.0) == 0

    monkeypatch.setattr(ps, "REFINE_HOUSEHOLD_LINES", True)
    moves = ps.refine_day_portions_integer(_mk(), targets, _FakeDB(), floor_g=15.0, cap_g=400.0)
    assert moves > 0, "con el knob ON la línea casera debe volverse movible"


# ═══════════════════════ Sd-P2-b: fatswap band guard ═══════════════════════

def test_sd_p2b_band_guard_anchored():
    assert "P2-FATSWAP-BAND-GUARD" in _GO
    i = _GO.index("P2-FATSWAP-BAND-GUARD")
    win = _GO[i:i + 1600]
    assert "_fs_fat_budget" in win  # budget de grasa (headroom + movible)
    assert "_fs_movable" in win     # grasa no-protegida que el trim SÍ puede recortar
    # cap por línea + estado de log distinto
    assert "_fat_room" in _GO
    assert "NO-PUDO" in _GO  # blowout ya no se confunde con 'no-necesario'


# ═══════════════════════ Sd-P2-c: no-carrier residual log ═══════════════════════

def test_sd_p2c_no_carrier_residual_anchored():
    assert "P2-MICRO-RESIDUAL-NOCARRIER" in _GO
    assert "no_carrier_no_seed" in _GO
    # el log va ANTES del continue del caso sin-contribuyentes (no silencioso).
    i = _GO.index("if not contributors:")
    win = _GO[i:i + 1000]
    assert "no_carrier_no_seed" in win
    assert "continue" in win


# ═══════════════════════ Fixtures ═══════════════════════

class _FakeDB:
    """pollo (proteína), arroz (carbo). macros por gramo escaladas por el líder/hint del string."""

    import re as _re
    _PER100 = {
        "pollo": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 4},
        "arroz": {"kcal": 130, "protein": 3, "carbs": 28, "fats": 0.5},
    }

    def _grams(self, s):
        low = str(s).lower()
        m = self._re.search(r"\((\d+(?:[.,]\d+)?)\s*g\)", low)  # hint "(300g)"
        if m:
            return float(m.group(1).replace(",", "."))
        m = self._re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)  # gram-led "100 g"
        if m:
            return float(m.group(1).replace(",", "."))
        return 0.0

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        g = self._grams(s)
        for tok, per in self._PER100.items():
            if tok in low and g > 0:
                return {k: v * g / 100.0 for k, v in per.items()}
        return None

    def grams_from_ingredient_string(self, s):
        return self._grams(s) or None
