"""[P1-MICRO-SEED-DENSITY · 2026-07-05] Escaleras por densidad + presupuesto 2-semillas + refuerzo.

Forense del plan 176ca62c (banner micro_worst_day pese al seeder v2): el Día 3 necesitaba
omega-3 Y vit E pero (a) el presupuesto kcal del closer (80/día) solo dejaba entrar UNA semilla
(linaza 53 kcal → quedaban 27 < guard 30 → la de vit E nunca entró), y (b) la escalera de vit E
(maní 0.9mg/10g, linaza 0.03mg/10g) era demasiado débil para cerrar un gap de 6mg.

Fix triple (datos del catálogo vivo, /100g):
  1. Escaleras por DENSIDAD: vit E → girasol 35.2 ≫ almendras 25.6 ≫ maní 9.1;
     omega-3 → linaza 22.8 ≫ chía 17.8 ≫ nueces (chía = 2ª opción seed-safe).
  2. Presupuesto 80→120 kcal/día + guard 30→25: dos semillas del mismo día caben.
  3. REFUERZO post-erosión: día ya sembrado que sigue corto en el re-check post-motor →
     UNA re-escalada de la semilla (cap 30 g, marker _micro_seed_reinforced anti-loop).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "linaza" in low:
            return {"omega3_g": 0.3}
        return {"omega3_g": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 55.0}


def _run(go, monkeypatch, plan, piso=2.0):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "omega3_g", "piso": piso, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "omega3_g", "piso": piso, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    return plan


def _mk_seeded_plan(reinforced=None):
    meal = {"meal": "Merienda", "name": "Merienda con Linaza",
            "ingredients": ["1 manzana", "10 g de semillas de linaza"],
            "ingredients_raw": ["1 manzana", "10 g de semillas de linaza"],
            "recipe": ["Sirve."], "_micro_seed_applied": "omega3_g"}
    if reinforced:
        meal["_micro_seed_reinforced"] = reinforced
    return {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Avena", "ingredients": ["40 g de avena"],
         "ingredients_raw": ["40 g de avena"], "recipe": ["Cocina."]},
        meal,
    ]}]}


# ---------------------------------------------------------------------------
# escaleras + presupuesto
# ---------------------------------------------------------------------------

def test_density_first_ladders():
    assert '"vit_e_mg": ("10 g de semillas de girasol", "10 g de almendras fileteadas", "10 g de maní")' in _GO
    assert '"omega3_g": ("10 g de semillas de linaza", "10 g de semillas de chía", "10 g de nueces")' in _GO


def test_budget_fits_two_seeds():
    assert '_env_int("MEALFIT_MICRONUTRIENT_CLOSER_MAX_KCAL_PER_DAY", 120)' in _GO
    # [Sd-P3-a · 2026-07-07] el guard 30→25 vive ahora en el knob MICRO_SEED_MIN_BUDGET_KCAL (default 25).
    assert "kcal_budget_left > MICRO_SEED_MIN_BUDGET_KCAL" in _GO, \
        "guard de colocación de semilla (default 25) para que la 2ª semilla del día quepa"
    assert '_env_int("MEALFIT_MICRO_SEED_MIN_BUDGET_KCAL", 25' in _GO


# ---------------------------------------------------------------------------
# refuerzo post-erosión
# ---------------------------------------------------------------------------

def _seed_grams(meal):
    import re
    line = next(s for s in meal["ingredients"] if "linaza" in s.lower())
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", ".")), line


def test_reinforces_eroded_seed(go, monkeypatch):
    plan = _run(go, monkeypatch, _mk_seeded_plan())
    meal = plan["days"][0]["meals"][1]
    g, line = _seed_grams(meal)
    # 10 → 18 por el refuerzo; el loop de escala del closer puede subirlo más (nunca bajarlo:
    # el contributor se actualiza al string reforzado — sin eso el loop lo pisaba a 16).
    assert g >= 18, f"día sembrado y aún corto → semilla reforzada (≥18 g), quedó: {line!r}"
    assert meal["_micro_seed_reinforced"] == "omega3_g"
    assert meal["ingredients"] == meal["ingredients_raw"]
    assert sum("linaza" in s.lower() for s in meal["ingredients"]) == 1, \
        "refuerza — NO añade una segunda línea"


def test_reinforce_only_once(go, monkeypatch):
    plan = _run(go, monkeypatch, _mk_seeded_plan(reinforced="omega3_g"))
    g, line = _seed_grams(plan["days"][0]["meals"][1])
    # ya reforzado → sin nuevo refuerzo; el loop de escala normal puede llevarlo a lo sumo a
    # MAX_SCALE (10→~16), jamás al nivel de refuerzo (18+).
    assert g < 18, f"día ya reforzado → la semilla no debe crecer al nivel de refuerzo: {line!r}"


def test_reinforce_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_SEED_REINFORCE_ENABLED", False)
    plan = _run(go, monkeypatch, _mk_seeded_plan())
    meal = plan["days"][0]["meals"][1]
    g, line = _seed_grams(meal)
    assert g < 18 and not meal.get("_micro_seed_reinforced"), \
        f"knob off → solo escala normal, sin refuerzo: {line!r}"


def test_marker_anchored_in_source():
    assert "P1-MICRO-SEED-DENSITY" in _GO
