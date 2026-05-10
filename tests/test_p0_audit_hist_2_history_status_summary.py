"""[P0-AUDIT-HIST-2 · 2026-05-09] Tests del endpoint agregado
``GET /api/plans/history-status-summary``.

Bug original (audit Historial 2026-05-09):
    El Historial deriva su bucket de status (complete/partial/failed/
    action_required) 100% desde `meal_plans.plan_data` jsonb flags
    (`generation_status`, `_user_action_required`,
    `_recovery_exhausted_chunks`). Pero solo
    `_escalate_unrecoverable_chunk` (cron_tasks.py:7928) escribe ese
    flag. Las 6 rutas que setean `status='pending_user_action'`
    (cron_tasks.py:5977/6038/6114/10636/12091/16798 — pausa pantry,
    snapshot stale, TZ unresolved, missing prior lessons
    pre-escalation) NO tocan plan_data. Resultado en producción
    confirmado: chunk en `pending_user_action` con plan_data sin
    `_user_action_required` → Historial muestra "Completo" mientras
    el sistema lo considera bloqueado.

Fix:
    Endpoint single-roundtrip que devuelve un dict por plan con
    contadores de status (`pending_user_action_count`, `failed_count`,
    `in_flight_count`, `completed_count`, `total`). El frontend lo
    consume en `getStatusInfo` para elevar el bucket a
    `action_required` cuando el queue tiene chunks bloqueados que
    plan_data no refleja.

Cobertura:
    1. Anchor del marker en el endpoint.
    2. 401 sin auth.
    3. 200 success: response shape `{summary: {plan_id: {...}}}`.
    4. SQL incluye `WHERE user_id = %s` (defense-in-depth + RLS).
    5. SQL excluye `meal_plan_id IS NULL` (orphans P0-HIST-3).
    6. SQL usa `FILTER (WHERE status = 'X')` para todos los counts
       requeridos (drift detection — si un futuro refactor pierde
       un counter, falla).
    7. Empty user → `{summary: {}}` (no 404).
    8. Counter math correcto (FILTER se respeta; valores no se
       confunden entre status).
    9. Cap defensivo de 200 planes presente.
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
_PLAN_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plans_history_status_summary
    src = inspect.getsource(api_plans_history_status_summary)
    assert "P0-AUDIT-HIST-2" in src, (
        "El endpoint debe citar `P0-AUDIT-HIST-2` para que un grep "
        "+ git blame lleve directo al fix."
    )


# ---------------------------------------------------------------------------
# 2. Auth
# ---------------------------------------------------------------------------
def test_history_status_summary_requires_auth():
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# 3. Success: response shape
# ---------------------------------------------------------------------------
def test_returns_dict_shape_with_per_plan_counters():
    """Response: `{summary: {plan_id: {pending_user_action_count,
    failed_count, in_flight_count, completed_count, total}}}`."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": 1,
            "failed_count": 0,
            "in_flight_count": 0,
            "completed_count": 4,
            "total": 5,
        },
        {
            "pid": _PLAN_B,
            "pending_user_action_count": 0,
            "failed_count": 2,
            "in_flight_count": 1,
            "completed_count": 0,
            "total": 3,
        },
    ]
    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-status-summary")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "summary" in body
    assert isinstance(body["summary"], dict)

    plan_a = body["summary"][_PLAN_A]
    assert plan_a["pending_user_action_count"] == 1
    assert plan_a["failed_count"] == 0
    assert plan_a["in_flight_count"] == 0
    assert plan_a["completed_count"] == 4
    assert plan_a["total"] == 5

    plan_b = body["summary"][_PLAN_B]
    assert plan_b["pending_user_action_count"] == 0
    assert plan_b["failed_count"] == 2
    assert plan_b["in_flight_count"] == 1
    assert plan_b["completed_count"] == 0
    assert plan_b["total"] == 3


# ---------------------------------------------------------------------------
# 4. SQL filter contract
# ---------------------------------------------------------------------------
def test_sql_filters_by_user_id_defense_in_depth():
    """`WHERE user_id = %s` es defense-in-depth además del RLS de
    `plan_chunk_queue`. Sin esto, un eventual bypass del RLS
    (service_role token) podría leer chunks de otros usuarios.
    """
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        captured["params"] = params
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    q = captured.get("query") or ""
    assert "user_id = %s" in q.replace("\n", " ")
    assert _USER_A in (captured.get("params") or ())


def test_sql_excludes_orphan_meal_plan_id():
    """`WHERE meal_plan_id IS NOT NULL` — los orphans de
    `chunk_lesson_telemetry`/`chunk_deferrals` SET NULL post P0-HIST-3
    NO existen aquí (chunk_queue.meal_plan_id es ON DELETE CASCADE)
    pero defensivamente excluimos NULL para prevenir agrupar bajo
    una key vacía.
    """
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    q = (captured.get("query") or "").replace("\n", " ")
    assert "meal_plan_id IS NOT NULL" in q


@pytest.mark.parametrize("required_filter", [
    "status = 'pending_user_action'",
    "status = 'failed'",
    "status = 'completed'",
])
def test_sql_includes_filter_for_each_required_status(required_filter):
    """El SQL debe usar `FILTER (WHERE status = 'X')` para los 3
    counters críticos. Si un refactor pierde uno, el frontend deja
    de detectar el drift respectivo y el bug original re-aparece.
    """
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    q = (captured.get("query") or "").replace("\n", " ")
    norm = re.sub(r"\s+", " ", q)
    assert required_filter in norm, (
        f"SQL debe incluir `FILTER (WHERE {required_filter})`. "
        f"Encontrado: {norm[:500]!r}..."
    )


def test_sql_groups_by_meal_plan_id():
    """`GROUP BY meal_plan_id` es necesario para que cada row del
    response represente un plan único. Sin esto, el endpoint
    devolvería una sola fila agregada de TODOS los chunks del
    usuario.
    """
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    q = (captured.get("query") or "").replace("\n", " ")
    assert "GROUP BY meal_plan_id" in q


def test_sql_caps_at_200_plans():
    """Cap defensivo `LIMIT 200`. Sin esto, un usuario con miles de
    chunks históricos rompería el bandwidth del response. El cap
    aplica al GROUP BY (planes únicos) — los chunks individuales
    siguen agregándose.
    """
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200

    q = (captured.get("query") or "").replace("\n", " ")
    norm = re.sub(r"\s+", " ", q)
    assert "LIMIT 200" in norm


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------
def test_empty_user_returns_empty_summary():
    """Usuario sin chunks → `{summary: {}}` (no 404). El frontend
    lo trata como "todos los planes confían en plan_data legacy"."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=[]):
        r = client.get("/api/plans/history-status-summary")

    assert r.status_code == 200
    body = r.json()
    assert body == {"summary": {}}


def test_row_with_missing_pid_is_skipped():
    """Defense: si la DB devuelve una fila sin `pid` (corrupción
    extrema), el endpoint la ignora en vez de incluir una key vacía
    en el dict."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": 1,
            "failed_count": 0,
            "in_flight_count": 0,
            "completed_count": 0,
            "total": 1,
        },
        {  # Corrupta: pid=None
            "pid": None,
            "pending_user_action_count": 5,
            "failed_count": 0,
            "in_flight_count": 0,
            "completed_count": 0,
            "total": 5,
        },
    ]
    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200
    body = r.json()
    assert _PLAN_A in body["summary"]
    assert None not in body["summary"]
    assert "" not in body["summary"]
    assert len(body["summary"]) == 1


def test_counter_values_are_ints_not_strings():
    """Defense: psycopg podría devolver Decimal/str en agregaciones
    según versión. Verificamos que el response tiene ints."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": "3",  # str (simulando driver edge)
            "failed_count": 1,
            "in_flight_count": 0,
            "completed_count": 0,
            "total": 4,
        },
    ]
    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-status-summary")
    assert r.status_code == 200
    body = r.json()
    entry = body["summary"][_PLAN_A]
    for key in ("pending_user_action_count", "failed_count",
                "in_flight_count", "completed_count", "total"):
        assert isinstance(entry[key], int), (
            f"{key} debe ser int en el response, got {type(entry[key])}"
        )
    assert entry["pending_user_action_count"] == 3
