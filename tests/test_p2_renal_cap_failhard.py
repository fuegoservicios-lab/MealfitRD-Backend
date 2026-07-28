"""[P2-RENAL-CAP-FAILHARD · 2026-06-15] El cap renal de proteína pasa de best-effort a fail-hard.

Audit P2-11: (A) el Guard 1 del cap renal estaba gateado por PROTEIN_FLOOR_ENABLED (knob de hipertrofia)
— el cap renal es SEGURIDAD iatrogénica, no debe compartir kill-switch; ahora knob dedicado RENAL_CAP_
ENABLED. (B) meals_enforced se marcaba True incondicional (cosmético); ahora se VERIFICA (ningún día
sobre el cap) tras el trim. (C) el exit-net seteaba meals_enforced=True sin trimar; ahora trima de verdad
reusando las funciones validadas.

Validación determinista con catálogo stub (sin LLM/DB real/créditos).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _StubDB:
    # [reapuntado 2026-07-28] El recompute honesto (P3-8, 07-07) trima RE-ESCALANDO el
    # string y midiendo el DELTA de macros — un stub que devuelve macros fijas sin mirar
    # los gramos deja delta=0 y el trim se vuelve no-op (el test moría con el día intacto
    # en 105 g). El stub ahora escala por la cantidad líder, como la DB real (per-100g).
    _PER100 = {
        "pollo": {"kcal": 165.0, "protein": 31.0, "carbs": 0.0, "fats": 3.6},
        "res": {"kcal": 165.0, "protein": 31.0, "carbs": 0.0, "fats": 3.6},
        "pescado": {"kcal": 165.0, "protein": 31.0, "carbs": 0.0, "fats": 3.6},
        "arroz": {"kcal": 130.0, "protein": 2.7, "carbs": 28.0, "fats": 0.3},
    }

    def macros_from_ingredient_string(self, s):
        import re as _re
        t = str(s).lower()
        base = next((v for k, v in self._PER100.items() if k in t), None)
        if base is None:
            return None
        mq = _re.match(r"\s*(\d+(?:[.,]\d+)?)\s*g\b", t)
        grams = float(mq.group(1).replace(",", ".")) if mq else 100.0
        f = grams / 100.0
        return {k: round(v * f, 2) for k, v in base.items()}


def test_renal_cap_knob_exists_default_on(go):
    assert go.RENAL_CAP_ENABLED is True


def test_gate_decoupled_from_protein_floor(go):
    """El Guard 1 del cap renal se gatea por RENAL_CAP_ENABLED, no por PROTEIN_FLOOR_ENABLED."""
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "RENAL_CAP_ENABLED = _env_bool" in src
    # El gate del Guard 1 usa RENAL_CAP_ENABLED (no el knob de hipertrofia).
    # [P2-SAFETY-KNOB-DECOUPLE · 2026-06-15, gap-audit G10] Además se DESACOPLÓ de CONDITION_RULES_ENABLED:
    # apagar las reglas-por-condición (calidad) NO debe apagar el cap renal (seguridad) → el gate del Guard 1
    # es ahora SOLO `RENAL_CAP_ENABLED and _db is not None` (el `applied` upstream ya restringe a quién aplica).
    assert "RENAL_CAP_ENABLED and _db is not None" in src
    assert "CONDITION_RULES_ENABLED and RENAL_CAP_ENABLED and _db is not None" not in src


def test_enforce_trims_and_verifies(go):
    """Día con proteína sobre el cap → trim al cap + meals_enforced=True (verificado)."""
    plan = {
        "renal_protein_cap": {"applied": True, "protein_g": 64, "meals_enforced": False},
        "days": [{"meals": [
            {"name": "Almuerzo", "ingredients": ["200g de pollo", "1 taza de arroz"],
             "protein": 60, "carbs": 40, "fats": 12, "cals": 520},
            {"name": "Cena", "ingredients": ["150g de res"],
             "protein": 45, "carbs": 0, "fats": 15, "cals": 315},
        ]}],
    }
    go._enforce_renal_per_meal(plan, 64, 1800, _StubDB())  # cap 64 g/día, día parte en 105 g
    day_p = sum(m["protein"] for m in plan["days"][0]["meals"])
    assert day_p <= 64 * 1.05, ("el día debe quedar ≤ cap", day_p)
    assert plan["renal_protein_cap"]["meals_enforced"] is True


def test_exit_net_actually_trims_not_cosmetic(go, monkeypatch):
    """El exit-net (path que bypasó la capa clínica) ahora TRIMA de verdad, no solo setea el flag."""
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    plan = {
        "days": [{"meals": [
            {"name": "Almuerzo", "ingredients": ["250g de pollo"],
             "protein": 75, "carbs": 0, "fats": 12, "cals": 410},
        ]}],
        "calories": 1700,
    }
    nutrition = {"renal_protein_cap": {"applied": True, "protein_g": 40, "comorbid_diabetes": False},
                 "target_calories": 1700}
    form_data = {"medicalConditions": ["Enfermedad renal crónica"]}
    go._renal_exit_safety_net(plan, nutrition, form_data)
    # Backstop real: el día se trimó al cap + meals_enforced verificado True (no cosmético).
    day_p = sum(m["protein"] for m in plan["days"][0]["meals"])
    assert day_p <= 40 * 1.05, ("el exit-net debe trimar de verdad", day_p)
    assert plan["renal_protein_cap"]["meals_enforced"] is True
    # Y la nota/gate de derivación renal sigue presente (seguridad + lo surfacea el banner del Dashboard).
    assert plan["requires_professional_review"]["renal_gate"] is True
