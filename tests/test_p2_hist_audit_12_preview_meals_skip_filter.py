"""[P2-HIST-AUDIT-12 · 2026-05-09] Tests del filtro `isSkipped` en
``preview_meals`` del endpoint ``/api/plans/history-list``.

Bug original (audit Historial 2026-05-09):
    El backend extraía hasta 4 meals del primer día con `m.get("name")`
    válido, pero NO filtraba `isSkipped`. El frontend
    (`renderMealPreview` en History.jsx) hacía el filter post-fetch:
        meals.filter(m => m.name && !m.isSkipped).slice(0, 3)
    Resultado: si los primeros 4 meals del SQL incluían 3 con
    `isSkipped=true`, el preview mostraba SOLO 1 chip (el filter
    elimina los 3 skipped y luego slice 3 al array de 1). El
    backend ya pagó el bandwidth de los 4 meals; el chip se ve
    incompleto.

Fix:
    Filter `if m.get("isSkipped"): continue` ANTES del slice cap
    en el loop Python del endpoint. El cap 4 ahora cuenta meals
    VÁLIDOS, no posiciones del array. El frontend mantiene su
    filter como defense-in-depth.

Cobertura:
    1. Anchor del marker.
    2. Cuando todos los meals son válidos (no skipped), shape
       legacy preservado (cap 4 max).
    3. Cuando hay meals skipped intercalados, el response solo
       devuelve los no-skipped, hasta cap 4.
    4. Cuando los primeros N son skipped, el endpoint avanza por
       los siguientes para llenar el cap.
    5. Bug original reproducible: 3 skipped + 3 válidos → 3 chips
       (NO 1 que era el bug pre-fix).
    6. `isSkipped=false` explícito NO filtra.
    7. `isSkipped` no booleano (string "true", número 1) — defensivo.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _row_with_preview(meals_list):
    """Fixture: row del SELECT con `preview_meals_raw=meals_list` y
    todos los demás campos en defaults safe."""
    return {
        "id": _PLAN_A,
        "name": "Plan Test",
        "created_at": None,
        "calories": 2000,
        "macros": {},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 4,
        "days_generated": 4,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "coherence_history": [],
        "preview_meals_raw": meals_list,
        "goal_root": None,
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 4,
        "chunk_tier_breakdown": None,
    }


def _client():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A
    return client


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P2-HIST-AUDIT-12" in src


# ---------------------------------------------------------------------------
# 2. Comportamiento legacy: meals todos válidos → shape preservado
# ---------------------------------------------------------------------------
def test_no_skipped_meals_returns_first_4():
    """Sin meals skipped, el endpoint devuelve los primeros 4 (cap).
    Comportamiento legacy preservado para no romper consumidores."""
    meals = [
        {"name": "Avena", "meal": "Desayuno"},
        {"name": "Pollo", "meal": "Almuerzo"},
        {"name": "Snack 1", "meal": "Snack"},
        {"name": "Cena", "meal": "Cena"},
        {"name": "Extra", "meal": "Extra"},  # 5th, descartado por cap.
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    assert len(plan["preview_meals"]) == 4
    names = [m["name"] for m in plan["preview_meals"]]
    assert names == ["Avena", "Pollo", "Snack 1", "Cena"]


# ---------------------------------------------------------------------------
# 3. Skip filter: skipped meals excluidos pre-slice
# ---------------------------------------------------------------------------
def test_skipped_meals_filtered_out():
    """Meals con `isSkipped=true` son excluidos antes del cap 4."""
    meals = [
        {"name": "Avena", "meal": "Desayuno", "isSkipped": True},
        {"name": "Pollo", "meal": "Almuerzo"},
        {"name": "Snack 1", "meal": "Snack", "isSkipped": True},
        {"name": "Cena", "meal": "Cena"},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    names = [m["name"] for m in plan["preview_meals"]]
    assert names == ["Pollo", "Cena"]


def test_first_n_skipped_endpoint_advances_to_fill_cap():
    """Cuando los primeros N son skipped, el loop avanza por los
    siguientes para llenar el cap 4 con meals válidos."""
    meals = [
        {"name": "Skip 1", "meal": "Desayuno", "isSkipped": True},
        {"name": "Skip 2", "meal": "Almuerzo", "isSkipped": True},
        {"name": "Skip 3", "meal": "Snack", "isSkipped": True},
        {"name": "Valid 1", "meal": "Cena"},
        {"name": "Valid 2", "meal": "Extra"},
        {"name": "Valid 3", "meal": "Tarde"},
        {"name": "Valid 4", "meal": "Noche"},
        {"name": "Valid 5", "meal": "Snack 2"},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    names = [m["name"] for m in plan["preview_meals"]]
    assert len(names) == 4
    assert names == ["Valid 1", "Valid 2", "Valid 3", "Valid 4"]


def test_bug_original_3_skipped_3_valid_returns_3():
    """Reproducción exacta del bug del audit: 3 skipped intercalados
    con 3 válidos. Pre-fix devolvía 4 meals (3 skipped + 1 válido)
    → frontend filtraba a 1 chip. Post-fix devuelve 3 válidos."""
    meals = [
        {"name": "Skip 1", "meal": "Desayuno", "isSkipped": True},
        {"name": "Valid 1", "meal": "Almuerzo"},
        {"name": "Skip 2", "meal": "Snack", "isSkipped": True},
        {"name": "Valid 2", "meal": "Cena"},
        {"name": "Skip 3", "meal": "Extra", "isSkipped": True},
        {"name": "Valid 3", "meal": "Tarde"},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    names = [m["name"] for m in plan["preview_meals"]]
    assert names == ["Valid 1", "Valid 2", "Valid 3"]


# ---------------------------------------------------------------------------
# 4. isSkipped=false explícito NO filtra
# ---------------------------------------------------------------------------
def test_is_skipped_false_does_not_filter():
    """`isSkipped=false` explícito (no truthy) NO filtra el meal —
    indica que NO está skipped."""
    meals = [
        {"name": "Avena", "meal": "Desayuno", "isSkipped": False},
        {"name": "Pollo", "meal": "Almuerzo", "isSkipped": False},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    assert len(plan["preview_meals"]) == 2


# ---------------------------------------------------------------------------
# 5. Defensa contra valores no-booleanos en isSkipped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truthy_value", [1, "true", "yes", [1]])
def test_truthy_non_boolean_is_skipped_filters(truthy_value):
    """Cualquier valor truthy de `isSkipped` filtra (Python `if x:`).
    Esto es coherente con la convención del frontend
    (`!m.isSkipped`). Si el jsonb persistiera string "true" por bug
    legacy, la lógica truthy lo trata correctamente."""
    meals = [
        {"name": "Skipped Truthy", "meal": "X", "isSkipped": truthy_value},
        {"name": "Valid", "meal": "Y"},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    names = [m["name"] for m in plan["preview_meals"]]
    assert names == ["Valid"]


@pytest.mark.parametrize("falsy_value", [0, "", None])
def test_falsy_is_skipped_does_not_filter(falsy_value):
    """`isSkipped` falsy (0, '', None) NO filtra — Python `if x:`
    devuelve False, así que el meal pasa el guard."""
    meals = [
        {"name": "Falsy isSkipped", "meal": "X", "isSkipped": falsy_value},
    ]
    rows = [_row_with_preview(meals)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = _client().get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    names = [m["name"] for m in plan["preview_meals"]]
    assert names == ["Falsy isSkipped"]
