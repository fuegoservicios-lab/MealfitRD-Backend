"""[P1-AUDIT-HIST-6 · 2026-05-09] Tests del campo `tier_breakdown`
agregado a ``/api/plans/history-list`` y ``/api/plans/history-status-summary``.

Bug original (audit Historial 2026-05-09):
    El Historial mostraba el bucket de status (`complete`/`partial`/
    `failed`/`action_required`) pero NO la "calidad" con la que se
    generaron los chunks completed. El endpoint chunk-status del
    plan ACTIVO ya expone tier_breakdown (`routers/plans.py:3349`),
    pero NO había forma de ver retroactivamente para planes
    archivados — un plan que terminó con todos los chunks en tier
    `emergency` (degraded) se veía igual que uno con todos en tier
    `llm` (mejor calidad).

Fix:
    LATERAL nested aggregation (`jsonb_object_agg(quality_tier, cnt)`)
    en ambos endpoints:
      - /history-list: campo `chunk_tier_breakdown` por plan.
      - /history-status-summary: campo `tier_breakdown` per plan.
    Solo cuenta chunks `completed` con `quality_tier IS NOT NULL`
    (los demás states no tienen tier significativo).

Cobertura:
    1. Anchor del marker.
    2. /history-list SQL contiene LATERAL con jsonb_object_agg.
    3. /history-list response incluye `chunk_tier_breakdown` por plan.
    4. /history-list normaliza dict vacío → None (frontend omite render).
    5. /history-list filtra status='completed' AND quality_tier IS NOT NULL.
    6. /history-status-summary SQL contiene LATERAL.
    7. /history-status-summary response incluye `tier_breakdown` por plan.
    8. /history-status-summary normaliza dict vacío → None.
"""
from __future__ import annotations

import inspect
import re
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


def _base_history_row(pid, **overrides):
    """Fila base para SELECT de /history-list."""
    base = {
        "id": pid,
        "name": f"Plan {pid[:4]}",
        "created_at": None,
        "calories": 2000,
        "macros": {},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "coherence_history": [],
        "preview_meals_raw": [],
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
        "chunk_completed_count": 5,
        "chunk_tier_breakdown": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_history_list_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P1-AUDIT-HIST-6" in src


def test_marker_in_history_status_summary_endpoint():
    from routers.plans import api_plans_history_status_summary
    src = inspect.getsource(api_plans_history_status_summary)
    assert "P1-AUDIT-HIST-6" in src


# ---------------------------------------------------------------------------
# 2. /history-list: SQL contiene LATERAL + jsonb_object_agg
# ---------------------------------------------------------------------------
def test_history_list_sql_uses_lateral_with_jsonb_object_agg():
    """El LATERAL anidado para tier_breakdown debe estar presente.
    Sin el LATERAL, no hay forma de exponer un dict per plan."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LEFT JOIN LATERAL" in norm.upper(), (
        "history-list debe usar LEFT JOIN LATERAL para tier_breakdown."
    )
    assert "jsonb_object_agg" in norm, (
        "tier_breakdown debe agregarse con `jsonb_object_agg(quality_tier, cnt)`."
    )


def test_history_list_tier_subquery_filters_status_completed_and_tier_not_null():
    """La subquery interna del LATERAL debe filtrar `status='completed'`
    Y `quality_tier IS NOT NULL`. Sin esto, contaríamos failed/paused/
    error chunks (que tienen quality_tier sintético) o NULLs."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    # Buscar LATERAL block + verificar ambos filtros dentro.
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(\s*(.+?)\s*\)\s+qtiers",
        norm,
        re.IGNORECASE | re.DOTALL,
    )
    assert m is not None, f"No pude extraer el LATERAL block. Got: {norm[:1500]!r}"
    lateral = m.group(1)
    assert "status = 'completed'" in lateral
    assert "quality_tier IS NOT NULL" in lateral


# ---------------------------------------------------------------------------
# 3. /history-list: response incluye `chunk_tier_breakdown` por plan
# ---------------------------------------------------------------------------
def test_history_list_response_includes_tier_breakdown_when_present():
    """Cuando la fila trae un dict en `chunk_tier_breakdown`,
    el response lo expone tal cual."""
    fake_rows = [
        _base_history_row(_PLAN_A,
                          chunk_tier_breakdown={
                              "llm": 4,
                              "shuffle": 1,
                          }),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200, r.text

    plan = r.json()["plans"][0]
    assert plan["chunk_tier_breakdown"] == {"llm": 4, "shuffle": 1}


def test_history_list_response_normalizes_empty_breakdown_to_none():
    """Dict vacío `{}` (LATERAL no encontró rows con tier no-NULL) →
    el endpoint lo coerce a None para que el frontend omita el
    render del bloque entero."""
    fake_rows = [
        _base_history_row(_PLAN_A, chunk_tier_breakdown={}),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    assert plan["chunk_tier_breakdown"] is None


def test_history_list_response_passes_through_none():
    """Cuando la subquery LATERAL no retorna rows (el plan no tiene
    chunks completed), Postgres devuelve NULL → el endpoint lo
    pasa como None al frontend."""
    fake_rows = [
        _base_history_row(_PLAN_A, chunk_tier_breakdown=None),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    assert plan["chunk_tier_breakdown"] is None


# ---------------------------------------------------------------------------
# 4. /history-status-summary: SQL + response shape
# ---------------------------------------------------------------------------
def test_summary_sql_uses_lateral_with_jsonb_object_agg():
    """history-status-summary también debe exponer tier_breakdown
    para paridad con /history-list (mismo dato, distinto endpoint)."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LEFT JOIN LATERAL" in norm.upper()
    assert "jsonb_object_agg" in norm


def test_summary_response_includes_tier_breakdown():
    """Response per-plan debe incluir `tier_breakdown` (None o dict)."""
    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": 0,
            "failed_count": 0,
            "in_flight_count": 0,
            "completed_count": 7,
            "total": 7,
            "tier_breakdown": {"llm": 6, "emergency": 1},
        },
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200
    body = r.json()
    entry = body["summary"][_PLAN_A]
    assert entry["tier_breakdown"] == {"llm": 6, "emergency": 1}


def test_summary_response_normalizes_empty_breakdown_to_none():
    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": 0,
            "failed_count": 0,
            "in_flight_count": 0,
            "completed_count": 0,
            "total": 0,
            "tier_breakdown": {},
        },
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200
    entry = r.json()["summary"][_PLAN_A]
    assert entry["tier_breakdown"] is None


def test_summary_params_pass_user_id_twice():
    """El SQL del summary tiene 2 ocurrencias del user_id (subquery
    outer + LATERAL inner). Params binding correcto."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["params"] = params
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200
    params = captured.get("params")
    assert isinstance(params, tuple)
    assert all(p == _USER_A for p in params)
    assert len(params) >= 2  # outer subquery + LATERAL


# ---------------------------------------------------------------------------
# 5. Coherencia: ambos endpoints usan la misma fuente
# ---------------------------------------------------------------------------
def test_both_endpoints_filter_same_status_and_tier_constraints():
    """Verificación de paridad: el filtro de tier_breakdown en
    history-list y history-status-summary debe ser idéntico
    (status='completed' AND quality_tier IS NOT NULL). Drift entre
    los dos endpoints causaría que el frontend reportara distinta
    info según qué endpoint consume."""
    captured_list = {}
    captured_summary = {}

    def _fake_list(query, params=None, **kwargs):
        captured_list["query"] = query
        return []

    def _fake_summary(query, params=None, **kwargs):
        captured_summary["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake_list):
        client.get("/api/plans/history-list")
    with patch("db_core.execute_sql_query", side_effect=_fake_summary):
        client.get("/api/plans/history-status-summary")

    norm_list = re.sub(r"\s+", " ", captured_list["query"] or "")
    norm_summary = re.sub(r"\s+", " ", captured_summary["query"] or "")

    # Ambos deben filtrar status='completed' Y quality_tier IS NOT NULL
    # dentro del bloque LATERAL.
    for label, q in (("history-list", norm_list), ("summary", norm_summary)):
        m = re.search(
            r"LEFT\s+JOIN\s+LATERAL\s*\(\s*(.+?)\s*\)\s+qtiers",
            q,
            re.IGNORECASE | re.DOTALL,
        )
        assert m, f"{label}: no pude extraer LATERAL block"
        lateral = m.group(1)
        assert "status = 'completed'" in lateral, f"{label}: filtro status missing"
        assert "quality_tier IS NOT NULL" in lateral, f"{label}: filtro tier missing"
