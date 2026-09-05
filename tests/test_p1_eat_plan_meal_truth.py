"""[P1-EAT-PLAN-MEAL-TRUTH · 2026-09-04] «Me lo comí» contrasta lo declarado con lo que la app sabe.

Dos observaciones del dueño (04-09, 9:04 a. m.): (1) «Me lo comí» sobre el ALMUERZO a las 9 entra
sin preguntar; (2) con la Nevera VACÍA el desayuno se registra igual, «¿cómo lo cocinaste?». Las dos
son el mismo defecto: el registro se tragaba la declaración. Ahora el cliente decide si pregunta
(hoja de un toque) con dos datos: la hora local (ventanas por slot, `config/mealWindows.js`) y la
cobertura de la Nevera, que sale de UNA vista previa server-side que usa la MISMA resolución de
nombres que la resta real (`deduct_consumed_meal_from_inventory(dry_run=True)`), sin escribir.
«Comí otra cosa» / «todavía no» quedan como desvío en `pipeline_metrics` (node='plan_meal_deviation').

Invariantes: coordenadas, nunca contenido (I2: `AND user_id = %s`); quota-exempt (limitadores
propios); el dry-run NO escribe (ni restas, ni failed_inventory_deductions, ni ledger).
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_DIARY = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
_INV = (_BACKEND / "db_inventory.py").read_text(encoding="utf-8")


def _handler(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    end = src.find("\n@router.", start)
    return src[start:end if end != -1 else len(src)]


def test_preview_and_deviation_are_coordinates_only_and_user_scoped():
    for name in ("api_preview_consumed_meal_from_plan", "api_plan_meal_deviation"):
        body = _handler(_DIARY, name)
        assert "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s" in body, name
        assert "verify_api_quota" not in body, name
    from routers.diary import PlanMealDeviationRequest, ConsumedFromPlanRequest
    assert set(PlanMealDeviationRequest.model_fields) == {"plan_id", "day_index", "meal_index", "reason", "local_hour"}
    assert "ingredients" not in ConsumedFromPlanRequest.model_fields
    with pytest.raises(Exception):
        PlanMealDeviationRequest(plan_id="x" * 36, day_index=0, meal_index=0, reason="invented")


def test_preview_uses_the_same_resolution_as_the_real_deduction_in_dry_run():
    body = _handler(_DIARY, "api_preview_consumed_meal_from_plan")
    assert 'source="plan_meal_preview", dry_run=True' in body
    assert "_PLAN_MEAL_PREVIEW_LIMITER" in body
    assert re.search(r'"coverage": \(len\(present\) / total\) if total else 1\.0', body)
    # [v2] la cobertura por conteo engaña: el ingrediente PRINCIPAL ausente también dispara la pregunta
    assert '"main_missing": main_missing' in body and "main_line = next(" in body


def test_dry_run_never_writes(monkeypatch):
    import db_inventory as inv
    monkeypatch.setattr(inv, "_db_available", lambda: True)
    monkeypatch.setattr(inv, "execute_sql_query", lambda *a, **k: [{"id": 1, "ingredient_name": "Huevo", "quantity": 12.0, "unit": "unidad", "reserved_quantity": 0.0, "reservation_details": None}])
    monkeypatch.setattr(inv, "find_pantry_rows_for_name", lambda user_id, name, prefetched_rows=None: (([{"id": 1}], "exact") if "huevo" in name.lower() else ([], None)))
    writes = []
    monkeypatch.setattr(inv, "add_or_update_inventory_item", lambda *a, **k: writes.append(("upd", a)) or True)
    monkeypatch.setattr(inv, "_consume_reserved_inventory", lambda *a, **k: writes.append(("res", a)))
    monkeypatch.setattr(inv, "_persist_failed_inventory_deductions", lambda *a, **k: writes.append(("fail", a)))
    monkeypatch.setattr(inv, "_persist_consumption_events", lambda *a, **k: writes.append(("ledger", a)))
    out = inv.deduct_consumed_meal_from_inventory("u", ["2 huevos", "50 g de Cebada"], source="plan_meal_preview", dry_run=True)
    assert out["succeeded"] == ["2 huevos"]
    assert out["not_in_pantry"] == ["50 g de Cebada"]
    assert writes == [], writes
    # y el camino real sigue escribiendo
    out2 = inv.deduct_consumed_meal_from_inventory("u", ["2 huevos"], source="plan_meal")
    assert out2["succeeded"] == ["2 huevos"] and any(w[0] == "upd" for w in writes)


def test_deviation_persists_a_pipeline_metric_with_the_plans_own_content():
    body = _handler(_DIARY, "api_plan_meal_deviation")
    assert "'plan_meal_deviation'" in body
    assert '"meal_name": str(meal.get("name") or "")[:120]' in body
    assert "_PLAN_MEAL_DEVIATION_LIMITER" in body


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-EAT-PLAN-MEAL-TRUTH · 2026-09-04" in app
