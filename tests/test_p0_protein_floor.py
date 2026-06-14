"""[P3-PROTEIN-FLOOR · 2026-06-14] Piso de proteína post-solve en el portion_solver.

Cierra (la parte de SOLVER del) gap P0-3 del audit: déficit sistémico de proteína. El LSQ es
simétrico + regulariza hacia el porcionado bajo del LLM → retiene la fuente de proteína para no
pasarse de kcal → sub-entrega. El piso escala SOLO las FUENTES de proteína escalables (proteína
>=25% de sus kcal Y no carbo-dominantes) para cerrar el gap si quedó bajo target; clamp max_scale;
nunca recorta. Medido (micro-benchmark, 10 comidas DR): proteína MAPE 11.6%→7.7%, ±10% 50%→70%,
4-macros sin regresión, sin porciones absurdas (leguminosas/granos excluidos).

Tests PUROS (sin DB) sobre las funciones del solver.
"""
from __future__ import annotations

import portion_solver as ps


# ── _is_protein_source: qué cuenta como fuente de proteína ESCALABLE ──
def test_is_protein_source_includes_lean_and_fatty_protein():
    # pollo (proteína-dominante)
    assert ps._is_protein_source({"protein": 30, "carbs": 0, "fats": 3, "kcal": 147})
    # huevo (grasa-dominante por kcal PERO proteína-rico, bajo carbo) → SÍ es fuente
    assert ps._is_protein_source({"protein": 6, "carbs": 0.5, "fats": 5, "kcal": 69})
    # queso blanco (grasa-dom, bajo carbo)
    assert ps._is_protein_source({"protein": 24, "carbs": 3, "fats": 22, "kcal": 306})


def test_is_protein_source_excludes_carbs_and_fat_and_legumes():
    assert not ps._is_protein_source({"protein": 4, "carbs": 45, "fats": 0.5, "kcal": 200})   # arroz
    assert not ps._is_protein_source({"protein": 18, "carbs": 40, "fats": 1, "kcal": 241})    # lentejas (carbo-dom)
    assert not ps._is_protein_source({"protein": 0, "carbs": 0, "fats": 14, "kcal": 124})     # aceite
    assert not ps._is_protein_source({"protein": 1, "carbs": 27, "fats": 0, "kcal": 110})     # fruta


def _entries(*macros):
    out = []
    for m in macros:
        g = None
        contrib = {"protein": m["protein"] * 4, "carbs": m["carbs"] * 4, "fats": m["fats"] * 9}
        if any(contrib.values()):
            g = max(contrib, key=contrib.get)
        out.append({"macros": m, "group": g})
    return out


# ── _apply_protein_floor: cierra el gap escalando solo fuentes ──
def test_floor_scales_protein_source_on_undershoot():
    e = _entries({"protein": 30, "carbs": 0, "fats": 3, "kcal": 147},   # pollo
                 {"protein": 4, "carbs": 45, "fats": 0.5, "kcal": 200})  # arroz
    factors = [1.0, 1.0]                                                 # achieved P = 34
    applied = ps._apply_protein_floor(e, [0, 1], factors, {"protein": 50}, 3.5)
    assert applied is True
    assert factors[0] > 1.0 and factors[1] == 1.0                        # solo el pollo escaló
    achieved_p = 30 * factors[0] + 4 * factors[1]
    assert abs(achieved_p - 50) < 0.6                                    # clava el target


def test_floor_noop_when_no_scalable_protein_source():
    # lentejas + arroz (ambos carbo-dom) → no hay fuente escalable → no inflar carbos
    e = _entries({"protein": 18, "carbs": 40, "fats": 1, "kcal": 241},
                 {"protein": 4, "carbs": 45, "fats": 0.5, "kcal": 200})
    factors = [1.0, 1.0]
    assert ps._apply_protein_floor(e, [0, 1], factors, {"protein": 50}, 3.5) is False
    assert factors == [1.0, 1.0]


def test_floor_noop_on_overshoot():
    e = _entries({"protein": 30, "carbs": 0, "fats": 3, "kcal": 147})
    factors = [2.0]                                                      # achieved P = 60
    assert ps._apply_protein_floor(e, [0], factors, {"protein": 50}, 3.5) is False
    assert factors == [2.0]


def test_floor_never_trims():
    """Jamás recorta: si la proteína está sobre el target, no baja el factor."""
    e = _entries({"protein": 40, "carbs": 0, "fats": 3, "kcal": 187})
    factors = [1.5]                                                      # achieved P = 60 (sobre 45)
    ps._apply_protein_floor(e, [0], factors, {"protein": 45}, 3.5)
    assert factors[0] >= 1.5


def test_floor_respects_max_scale_clamp():
    # gap enorme → el factor se clampa a max_scale, no lo excede
    e = _entries({"protein": 10, "carbs": 0, "fats": 1, "kcal": 49})     # poca proteína
    factors = [1.0]
    ps._apply_protein_floor(e, [0], factors, {"protein": 200}, 3.5)      # target imposible
    assert factors[0] == 3.5                                             # clamp exacto


def test_floor_gated_by_knob(monkeypatch):
    """El piso solo corre si SOLVER_PROTEIN_FLOOR; _compute_scale_factors lo respeta."""
    e = _entries({"protein": 30, "carbs": 0, "fats": 3, "kcal": 147},
                 {"protein": 4, "carbs": 45, "fats": 0.5, "kcal": 200})
    monkeypatch.setattr(ps, "SOLVER_PROTEIN_FLOOR", False)
    f_off, m_off = ps._compute_scale_factors(e, {"kcal": 350, "protein": 50, "carbs": 49, "fats": 4}, 0.3, 3.5)
    assert "pfloor" not in m_off
    monkeypatch.setattr(ps, "SOLVER_PROTEIN_FLOOR", True)
    f_on, m_on = ps._compute_scale_factors(e, {"kcal": 350, "protein": 50, "carbs": 49, "fats": 4}, 0.3, 3.5)
    assert "pfloor" in m_on  # con el knob ON el piso disparó (la proteína quedaba bajo target)


def test_anchor_protein_floor_in_source():
    import inspect
    src = inspect.getsource(ps)
    assert "P3-PROTEIN-FLOOR" in src
    assert "MEALFIT_SOLVER_PROTEIN_FLOOR" in src
    assert "_apply_protein_floor" in src
