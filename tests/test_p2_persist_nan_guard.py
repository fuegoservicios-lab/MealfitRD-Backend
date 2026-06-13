"""[P2-PERSIST-NAN-GUARD · 2026-06-13] El guard que evita que un NaN/Inf en plan_data
pierda un plan al persistir (Postgres jsonb rechaza NaN/Infinity → INSERT falla →
'invalid input syntax for type json' → plan generado+aprobado PERDIDO).

Bug live 2026-06-13: el plan se generó, pasó review y se perdió al guardar porque un
macro era no-finito. El guard sanea NaN/Inf → 0.0 antes del INSERT.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import _sanitize_nonfinite_for_json, _scrub_plan_data_floats


def test_replaces_nan_and_inf_with_zero():
    obj = {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 3.5, "e": 10}
    out, found = _sanitize_nonfinite_for_json(obj)
    assert out["a"] == 0.0 and out["b"] == 0.0 and out["c"] == 0.0
    assert out["d"] == 3.5 and out["e"] == 10
    assert len(found) == 3


def test_nested_plan_data_structure():
    plan = {"calories": 2050, "days": [
        {"meals": [
            {"name": "Desayuno", "protein": float("nan"), "cals": 350,
             "macros": ["P:0g"], "ingredients": ["2 huevos"]},
            {"name": "Almuerzo", "protein": 40, "fats": float("inf")},
        ]},
    ]}
    out, found = _sanitize_nonfinite_for_json(plan)
    assert out["days"][0]["meals"][0]["protein"] == 0.0
    assert out["days"][0]["meals"][1]["fats"] == 0.0
    assert out["days"][0]["meals"][1]["protein"] == 40  # intacto
    assert len(found) == 2


def test_output_is_valid_postgres_json():
    # json.dumps con allow_nan=False emula la validación estricta de Postgres jsonb.
    plan = {"x": float("nan"), "days": [{"meals": [{"p": float("inf")}]}]}
    out = _scrub_plan_data_floats(plan, "test-user")
    # Sin el scrub esto lanzaría ValueError: Out of range float values are not JSON compliant
    json.dumps(out, allow_nan=False)  # no debe lanzar


def test_finite_plan_unchanged():
    plan = {"calories": 2050, "days": [{"meals": [{"protein": 30, "cals": 400.5}]}]}
    out = _scrub_plan_data_floats(plan, "test-user")
    assert out == plan


def test_scrub_is_failsafe_on_bad_input():
    # Si algo raro pasa, no debe lanzar (mejor intentar persistir que perder el plan).
    assert _scrub_plan_data_floats({"ok": 1}, "u") == {"ok": 1}
