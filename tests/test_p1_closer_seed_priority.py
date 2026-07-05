"""[P1-CLOSER-SEED-PRIORITY · 2026-07-05] Micros sembrables PRIMERO en la cola del closer.

Forense definitiva del banner micro_worst_day recurrente (plan a5c9983d, "tras 1 intento"):
el Día 1 estaba en 7.2/15 de vit E y 0.41/1.6 de omega-3 (ambos bajo el umbral 0.6× del
banner), el panel promedio los tenía 'bajo' (el set de floors los INCLUÍA), el seeder tenía
girasol/linaza en el arsenal — y aún así CERO 🌱 en toda la corrida. La causa: el presupuesto
kcal del día (120) es COMPARTIDO y se consume en el ORDEN de iteración de `floors` (= orden
del panel), con vit E/omega-3 al final. Los micros tempranos (fibra/calcio/magnesio) agotaron
el presupuesto escalando ingredientes → el guard de semilla (>25 kcal) llegó sin fondos.

Fix: `_floors_ordered` — los micros con fuente sembrable (`_MICRO_SEED_SOURCES`) van PRIMERO
(son el cierre más kcal-eficiente: girasol 3.5mg de vit E por 58 kcal); el resto conserva su
orden relativo (sorted estable).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """fibra abundante y escalable (consume presupuesto); omega3 sin portador (necesita seed)."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "avena" in low:
            return {"fiber_g": 4.0, "omega3_g": 0.0}
        if "linaza" in low or "chía" in low or "chia" in low:
            return {"omega3_g": 0.6, "fiber_g": 1.0}
        return {"fiber_g": 0.0, "omega3_g": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 110.0}  # cada escala/semilla consume MUCHO presupuesto (120/día)

    def grams_from_ingredient_string(self, s):
        return 40.0

    def rescale_ingredient_string(self, s, factor):
        return s + " (x)"  # marcador de que se escaló


def _mk_report():
    # fibra 'bajo' ANTES que omega3 en el orden del panel (reproduce el orden real).
    return {"panel": [
        {"key": "fiber_g", "piso": 30.0, "valor": 10.0, "status": "bajo"},
        {"key": "omega3_g", "piso": 1.6, "valor": 0.2, "status": "bajo"},
    ], "gaps": [
        {"key": "fiber_g", "piso": 30.0, "status": "bajo"},
        {"key": "omega3_g", "piso": 1.6, "status": "bajo"},
    ], "coverage": 1.0, "per_day_floors": {"flagged": False}}


def _mk_plan():
    return {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Avena", "ingredients": ["40 g de avena"],
         "ingredients_raw": ["40 g de avena"], "recipe": ["Cocina."]},
        {"meal": "Merienda", "name": "Fruta", "ingredients": ["1 manzana"],
         "ingredients_raw": ["1 manzana"], "recipe": ["Sirve."]},
    ]}]}


def test_ordered_iteration_anchored():
    i = _GO.index("[P1-CLOSER-SEED-PRIORITY · 2026-07-05]")
    win = _GO[i:i + 1600]
    assert "_floors_ordered = sorted(floors.items(), key=lambda kv: kv[0] not in _MICRO_SEED_SOURCES)" in win
    assert "for k, floor in _floors_ordered:" in win


def test_seed_gets_budget_before_scaling_exhausts_it(go, monkeypatch):
    """Sin la prioridad, la fibra (primera en el panel) escalaba la avena (110 kcal) y dejaba
    el presupuesto en 10 < guard 25 → la semilla de omega-3 jamás entraba. Con la prioridad,
    omega-3 (sembrable) va primero → linaza sembrada."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: _mk_report())
    plan = _mk_plan()
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    _all = " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"])
    assert "linaza" in _all, \
        "el micro sembrable debe recibir presupuesto ANTES de que el escalado lo agote"


def test_marker_anchored_in_source():
    assert "P1-CLOSER-SEED-PRIORITY" in _GO
