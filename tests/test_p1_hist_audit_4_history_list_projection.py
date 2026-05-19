"""[P1-HIST-AUDIT-4 · 2026-05-09] Tests del endpoint
``GET /api/plans/history-list`` con projection mínima.

Bug original (audit historial 2026-05-08):
    ``History.jsx::fetchHistory`` hacía
    ``supabase.from('meal_plans').select('*')``. Para tier ultra
    con 50+ planes, eso descargaba el ``plan_data`` jsonb completo
    (~30-80KB por plan) → MBs por apertura del Historial.

Fix:
    Endpoint backend con projection vía operadores jsonb que extrae
    solo los keys que la card consume. El response es ~50× más
    liviano por plan. El modal carga lazy ``plan_data.days`` solo
    del plan abierto.

Cobertura:
    - Anchor del marker.
    - Auth: 401 sin verified_user_id.
    - SQL contract:
      - Selecciona ``id``, ``name``, ``created_at``, ``calories``,
        ``macros`` top-level.
      - Extrae ``_plan_modified_at``, ``generation_status``,
        ``total_days_requested``, ``user_action_required``,
        ``recovery_exhausted_count``, ``user_forced_simplified_weeks``,
        ``coherence_history``, ``preview_meals_raw``, ``goal``,
        ``diet_preference``, ``allergies`` desde plan_data via ``->``/``->>``.
      - NO selecciona ``plan_data`` completo (negativo: regression
        guard contra revertir al ``select('*')``).
      - WHERE filter ``name IS NOT NULL``.
      - ORDER BY ``GREATEST(created_at, _plan_modified_at)`` SSOT
        con P1-HIST-AUDIT-1.
      - LIMIT 200 cap defensivo.
    - Behavior:
      - ``coherence_adjusts_count`` cuenta solo anomalous (degrade,
        reject_minor, reject_high, hydration_error). Excluye
        not_applicable y post_swap_revalidation.
      - ``preview_meals`` cap 4 + solo {name, meal}.
      - Goal/diet con fallback root → assessment.
      - Allergies normaliza a [] cuando no es lista.
      - ``created_at`` se serializa como ISO string.
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "11111111-1111-1111-1111-111111111111"


def _client():
    # [P1-HIST-AUDIT-NEW-1 · 2026-05-10] El endpoint pasó de
    # `verify_api_quota` a `get_verified_user_id` (12 endpoints
    # read-only/housekeeping del Historial). Mantenemos OVERRIDE
    # de AMBAS deps para tolerar deploy lag y para que el test
    # funcione si el endpoint vuelve a swap.
    from auth import verify_api_quota, get_verified_user_id
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER
    return client


def _make_row(**overrides) -> dict:
    """Row simulada como la devolvería psycopg con dict_row."""
    base = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Plan A",
        "created_at": datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        "calories": 2200,
        "macros": {"protein": 140, "carbs": 250, "fats": 60},
        "plan_modified_at": "2026-05-08T18:00:00+00:00",
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "coherence_history": [],
        "preview_meals_raw": [
            {"name": "Mangú", "meal": "Desayuno", "cals": 450},
            {"name": "Pollo guisado", "meal": "Almuerzo", "cals": 600},
        ],
        "goal_root": None,
        "goal_assessment": "build_muscle",
        "diet_root": None,
        "diet_assessment_snake": "omnivorous",
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": ["lactose", "gluten"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P1-HIST-AUDIT-4" in src


# ---------------------------------------------------------------------------
# 2. Auth
# ---------------------------------------------------------------------------
def test_history_list_requires_auth():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    client.app.dependency_overrides[get_verified_user_id] = lambda: None

    r = client.get("/api/plans/history-list")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 3. SQL contract (positivo + negativo)
# ---------------------------------------------------------------------------
def _captured_sql_for_history_list() -> str:
    from routers.plans import api_plans_history_list
    return inspect.getsource(api_plans_history_list)


def test_sql_extracts_top_level_columns():
    src = _captured_sql_for_history_list()
    for col in ("id::text", "name", "created_at", "calories", "macros"):
        assert col in src, f"Falta `{col}` en el SELECT de history-list"


def test_sql_extracts_jsonb_keys_via_operators():
    src = _captured_sql_for_history_list()
    expected_paths = [
        "plan_data->>'_plan_modified_at'",
        "plan_data->>'generation_status'",
        "plan_data->>'total_days_requested'",
        "plan_data->'_user_action_required'",
        "plan_data->'_recovery_exhausted_chunks'",
        "plan_data->'_user_forced_simplified_weeks'",
        "plan_data->'_shopping_coherence_block_history'",
        "plan_data->'days'",
        "plan_data->'assessment'",
    ]
    for path in expected_paths:
        assert path in src, (
            f"Falta extracción jsonb `{path}`. Sin esto, la card pierde "
            f"info y el frontend cae al fallback legacy (que requiere "
            f"plan_data completo)."
        )


def test_sql_does_not_select_plan_data_blob():
    """Regression guard: el SELECT NO debe traer la columna plan_data
    completa. Si alguien revierte el endpoint a `SELECT plan_data, ...`
    se pierde el ahorro de bandwidth.
    """
    src = _captured_sql_for_history_list()
    # Aislar el bloque del SELECT de history-list (entre la docstring y
    # el cierre del execute_sql_query).
    select_match = re.search(
        r"SELECT[\s\S]*?FROM\s+meal_plans",
        src,
        re.IGNORECASE,
    )
    assert select_match is not None
    select_block = select_match.group(0)
    # Permitir `plan_data->...` (los operadores) pero NO `plan_data,`
    # ni `plan_data\n` ni `plan_data AS` ni `plan_data FROM` que
    # indicarían que se está seleccionando la columna entera.
    bad_patterns = [
        r"\bplan_data\s*,",
        r"\bplan_data\s+AS\b",
        r"\bSELECT\s+plan_data\s",
    ]
    for pat in bad_patterns:
        assert not re.search(pat, select_block, re.IGNORECASE), (
            f"El SELECT incluye `plan_data` como columna completa "
            f"(patrón `{pat}`). Eso revierte P1-HIST-AUDIT-4 y trae "
            f"bandwidth otra vez."
        )


def test_sql_filters_name_not_null():
    src = _captured_sql_for_history_list()
    assert "name IS NOT NULL" in src, (
        "Falta filter `name IS NOT NULL` — convención post-P2-HIST-1: "
        "filas sin name son garbage no-actionable."
    )


def test_sql_orders_by_greatest_modified_at():
    """Sort SSOT con api_restore_plan (P1-HIST-AUDIT-1). Sin esto el
    listado y el target del restore divergen."""
    src = _captured_sql_for_history_list()
    assert "GREATEST" in src
    assert "_plan_modified_at" in src
    assert "DESC" in src


def test_sql_caps_with_limit():
    """Cap defensivo evita que un usuario con 10K planes haga
    bandwidth explosion. 200 es 2× margen del tier ultra actual."""
    src = _captured_sql_for_history_list()
    assert re.search(r"LIMIT\s+\d+", src), (
        "Falta cláusula LIMIT en el SQL — sin cap defensivo, un user "
        "malicioso o bug podría descargar miles de filas."
    )


# ---------------------------------------------------------------------------
# 4. Behavior — agregaciones server-side
# ---------------------------------------------------------------------------
def test_coherence_adjusts_count_only_anomalous():
    """`coherence_adjusts_count` cuenta solo anomalous actions.
    Whitelist explícita: degrade, reject_minor, reject_high,
    hydration_error. Excluye not_applicable y post_swap_revalidation.
    """
    client = _client()
    history = [
        {"action_taken": "degrade"},
        {"action_taken": "reject_minor"},
        {"action_taken": "reject_high"},
        {"action_taken": "hydration_error"},
        {"action_taken": "not_applicable"},          # excluido
        {"action_taken": "post_swap_revalidation"},  # excluido
        {"action_taken": "unknown_future_value"},    # excluido
        None,                                         # entry corrupta → skip
        "not-a-dict",                                 # entry corrupta → skip
        {"action_taken": 42},                         # tipo no-string → skip
    ]
    row = _make_row(coherence_history=history)
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200, r.text
    plans = r.json()["plans"]
    assert len(plans) == 1
    assert plans[0]["coherence_adjusts_count"] == 4


def test_preview_meals_capped_at_four_and_normalized():
    """preview_meals: cap 4, solo `{name, meal}` (descarta cals,
    recipe, etc. que son ruido para preview)."""
    client = _client()
    raw = [
        {"name": "M1", "meal": "Desayuno", "cals": 100, "recipe": "..."},
        {"name": "M2", "meal": "Almuerzo", "cals": 200},
        {"name": "M3", "meal": "Cena"},
        {"name": "M4", "meal": "Snack"},
        {"name": "M5", "meal": "Extra"},  # cap → no debe aparecer
    ]
    row = _make_row(preview_meals_raw=raw)
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    plans = r.json()["plans"]
    assert len(plans[0]["preview_meals"]) == 4
    # Solo {name, meal} — no cals/recipe.
    for m in plans[0]["preview_meals"]:
        assert set(m.keys()) == {"name", "meal"}


def test_preview_meals_skips_entries_without_name():
    client = _client()
    raw = [
        {"name": "M1", "meal": "Desayuno"},
        {"meal": "Sin nombre"},     # skip
        None,                        # skip
        "not-a-dict",                # skip
        {"name": "", "meal": "Vacío"},  # skip (falsy name)
        {"name": "M2", "meal": "Cena"},
    ]
    row = _make_row(preview_meals_raw=raw)
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    plans = r.json()["plans"]
    names = [m["name"] for m in plans[0]["preview_meals"]]
    assert names == ["M1", "M2"]


def test_goal_diet_fallback_root_then_assessment():
    """Goal: root → assessment.mainGoal. Diet: root → assessment.{diet_preference,
    dietPreference, dietType}."""
    client = _client()
    # Caso 1: goal en root, diet en assessment.dietPreference.
    row = _make_row(
        goal_root="lose_weight",
        goal_assessment="ignored",
        diet_root=None,
        diet_assessment_snake=None,
        diet_assessment_camel="vegan",
        diet_assessment_type=None,
    )
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    p = r.json()["plans"][0]
    assert p["goal"] == "lose_weight"  # root prioritario
    assert p["diet_preference"] == "vegan"  # fallback al camel


def test_allergies_normalized_to_list():
    """Si allergies viene como objeto/string/null, se normaliza a [] —
    el frontend lo trata como Array.isArray → tags no se rompen."""
    client = _client()
    for bad in [None, "no-soy-array", {"x": 1}, 42]:
        row = _make_row(allergies=bad)
        with patch("db_core.execute_sql_query", return_value=[row]):
            r = client.get("/api/plans/history-list")
        p = r.json()["plans"][0]
        assert p["allergies"] == [], (
            f"Allergies con tipo no-list `{bad!r}` debió normalizar a []; "
            f"got {p['allergies']!r}"
        )


def test_created_at_serialized_as_iso():
    """`created_at` viene como datetime de psycopg. La response JSON
    lo expone como ISO string para que JS lo pueda Date.parse."""
    client = _client()
    row = _make_row(
        created_at=datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc)
    )
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    p = r.json()["plans"][0]
    assert isinstance(p["created_at"], str)
    # ISO con offset.
    assert "2026-04-15" in p["created_at"]
    assert "T" in p["created_at"]


def test_response_shape_is_plans_envelope():
    """Response top-level es `{plans: [...]}` (no array directo). Esto
    permite añadir metadata futura sin romper consumers."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=[]):
        r = client.get("/api/plans/history-list")
    body = r.json()
    assert isinstance(body, dict)
    assert "plans" in body
    assert isinstance(body["plans"], list)


# ---------------------------------------------------------------------------
# 5. Bandwidth assertion: el plan summary NO contiene `plan_data`
# ---------------------------------------------------------------------------
def test_summary_response_omits_plan_data_key():
    """Cada plan en la response NO debe tener key `plan_data`. Si
    alguien añade ese key (e.g., para "compat"), se pierde el ahorro
    de bandwidth — el frontend lazy-loadea el plan_data del modal,
    no lo necesita upfront."""
    client = _client()
    row = _make_row()
    with patch("db_core.execute_sql_query", return_value=[row]):
        r = client.get("/api/plans/history-list")
    plans = r.json()["plans"]
    assert len(plans) == 1
    assert "plan_data" not in plans[0], (
        "El summary NO debe incluir `plan_data`. P1-HIST-AUDIT-4: el "
        "modal carga lazy."
    )
