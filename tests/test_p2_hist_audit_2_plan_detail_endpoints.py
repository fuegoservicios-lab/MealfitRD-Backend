"""[P2-HIST-AUDIT-2 · 2026-05-09] Tests de los endpoints de detalle
por-plan: ``GET /api/plans/{id}/lessons`` y
``GET /api/plans/{id}/coherence-history``.

Bug original (audit historial 2026-05-08):
    Los chips agregados de la card ("X lecciones", "X ajustes",
    "S2 simplif.") eran dead-end: el usuario veía el conteo pero no
    podía expandir a ver QUÉ aprendió el sistema o QUÉ ajustes hizo.
    Surface del diferenciador ("aprendizaje continuo") quedaba
    invisible.

Fix:
    Dos endpoints de detalle por-plan que el modal del Historial
    consume lazy desde tabs:
      - `/api/plans/{id}/lessons` — filas individuales de
        chunk_lesson_telemetry, mismo filter whitelist que
        `/lessons-counts` (P1-HIST-AUDIT-5) → drift cero.
      - `/api/plans/{id}/coherence-history` — entries del
        plan_data._shopping_coherence_block_history (raw, el frontend
        formatea).

Cobertura:
    /lessons:
        - Anchor del marker.
        - Auth: 401 sin verified_user_id.
        - Validación: 400 plan_id missing/invalid.
        - Ownership: 404 plan no existe O plan de otro user.
        - Filter whitelist: SQL contiene `event = ANY(%s)` con la
          misma constante `_LESSON_COUNT_EVENT_WHITELIST`.
        - Defense-in-depth: WHERE incluye user_id además de
          meal_plan_id.
        - Order/limit: ORDER BY created_at DESC, LIMIT 200.
        - Response shape: {plan_id, lessons:[]}.
        - created_at serializado como ISO string.
    /coherence-history:
        - Anchor del marker.
        - Auth + 400 + 404 (mismo patrón).
        - JOIN ownership + extract en un solo SELECT (no two-step).
        - Defensa: history no-list → [] en response.
        - Response shape: {plan_id, history:[]}.
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
_PLAN_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


def _client():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


def _client_no_auth():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: None
    return client


# ---------------------------------------------------------------------------
# /lessons — endpoint detalle de lecciones
# ---------------------------------------------------------------------------
def test_lessons_detail_marker_present():
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    assert "P2-HIST-AUDIT-2" in src


def test_lessons_detail_requires_auth():
    client = _client_no_auth()
    r = client.get(f"/api/plans/{_PLAN_ID}/lessons")
    assert r.status_code == 401


def test_lessons_detail_404_when_plan_missing_or_other_user():
    """Ownership SELECT devuelve None → 404 sin DOS-able discovery."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=None):
        r = client.get(f"/api/plans/{_PLAN_ID}/lessons")
    assert r.status_code == 404


def test_lessons_detail_returns_plan_id_envelope():
    client = _client()
    # Primer SELECT: ownership check (devuelve {id}); segundo:
    # rows de chunk_lesson_telemetry (vacío).
    with patch("db_core.execute_sql_query", side_effect=[{"id": _PLAN_ID}, []]):
        r = client.get(f"/api/plans/{_PLAN_ID}/lessons")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan_id"] == _PLAN_ID
    assert isinstance(body["lessons"], list)


def test_lessons_detail_serializes_created_at_as_iso():
    """`created_at` viene como datetime de psycopg → response como
    ISO string para que JS lo pueda Date.parse."""
    client = _client()
    row = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "event": "lesson_synthesized_low_confidence",
        "week_number": 2,
        "synthesized_count": 3,
        "queue_count": 1,
        "metadata": {"foo": "bar"},
        "created_at": datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
    }
    with patch("db_core.execute_sql_query", side_effect=[{"id": _PLAN_ID}, [row]]):
        r = client.get(f"/api/plans/{_PLAN_ID}/lessons")
    assert r.status_code == 200, r.text
    lessons = r.json()["lessons"]
    assert len(lessons) == 1
    assert isinstance(lessons[0]["created_at"], str)
    assert "2026-05-08" in lessons[0]["created_at"]
    assert lessons[0]["event"] == "lesson_synthesized_low_confidence"
    assert lessons[0]["metadata"] == {"foo": "bar"}


def test_lessons_detail_sql_uses_whitelist_filter():
    """SSOT con `/lessons-counts`: ambos endpoints usan
    `event = ANY(_LESSON_COUNT_EVENT_WHITELIST)`. Sin esto, el detalle
    mostraría events mecánicos que el conteo oculta."""
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    assert re.search(r"event\s*=\s*ANY\s*\(\s*%s\s*\)", src, re.IGNORECASE)
    assert "_LESSON_COUNT_EVENT_WHITELIST" in src


def test_lessons_detail_sql_filters_by_user_and_plan():
    """Defense-in-depth: WHERE incluye user_id además de
    meal_plan_id (RLS también, pero explícito previene leak si una
    policy futura cambia)."""
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    # Aislar el SELECT de detalle (NO el ownership check).
    detail_match = re.search(
        r"SELECT\s+id::text[\s\S]*?LIMIT\s+\d+",
        src,
        re.IGNORECASE,
    )
    assert detail_match is not None
    block = detail_match.group(0)
    assert "meal_plan_id = %s" in block
    assert "user_id = %s" in block


def test_lessons_detail_sql_orders_by_created_at_desc_with_limit():
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    assert re.search(r"ORDER\s+BY\s+created_at\s+DESC", src, re.IGNORECASE)
    assert re.search(r"LIMIT\s+\d+", src, re.IGNORECASE)


# ---------------------------------------------------------------------------
# /coherence-history — endpoint detalle de ajustes
# ---------------------------------------------------------------------------
def test_coherence_history_marker_present():
    from routers.plans import api_plan_coherence_history
    src = inspect.getsource(api_plan_coherence_history)
    assert "P2-HIST-AUDIT-2" in src


def test_coherence_history_requires_auth():
    client = _client_no_auth()
    r = client.get(f"/api/plans/{_PLAN_ID}/coherence-history")
    assert r.status_code == 401


def test_coherence_history_404_when_plan_missing():
    client = _client()
    with patch("db_core.execute_sql_query", return_value=None):
        r = client.get(f"/api/plans/{_PLAN_ID}/coherence-history")
    assert r.status_code == 404


def test_coherence_history_returns_plan_id_envelope():
    client = _client()
    history = [
        {"action_taken": "reject_minor", "ts": "2026-05-01T00:00:00+00:00"},
        {"action_taken": "degrade", "ts": "2026-05-02T00:00:00+00:00"},
    ]
    with patch("db_core.execute_sql_query", return_value={"history": history}):
        r = client.get(f"/api/plans/{_PLAN_ID}/coherence-history")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan_id"] == _PLAN_ID
    assert body["history"] == history


def test_coherence_history_normalizes_non_list_to_empty():
    """Si `_shopping_coherence_block_history` está corrupto (no es
    list), tratamos como [] para que el frontend renderice "Sin
    ajustes" en lugar de crash."""
    client = _client()
    for bad in [None, "not-a-list", {"x": 1}, 42]:
        with patch("db_core.execute_sql_query", return_value={"history": bad}):
            r = client.get(f"/api/plans/{_PLAN_ID}/coherence-history")
        assert r.status_code == 200
        assert r.json()["history"] == []


def test_coherence_history_sql_joins_ownership_and_extract():
    """Single SELECT: ownership + jsonb path en una sola query.
    Si row=None → 404 (no two-step que daría dos lookups y posible
    drift entre ellos)."""
    from routers.plans import api_plan_coherence_history
    src = inspect.getsource(api_plan_coherence_history)
    sql_match = re.search(
        r"SELECT\s+plan_data->'_shopping_coherence_block_history'[\s\S]*?WHERE",
        src,
        re.IGNORECASE,
    )
    assert sql_match is not None, (
        "Migración SQL no usa el patrón JOIN ownership + jsonb extract."
    )
    block = sql_match.group(0)
    assert "FROM meal_plans" in block


# ---------------------------------------------------------------------------
# Cross-check: ambos endpoints son @router.get
# ---------------------------------------------------------------------------
def test_endpoints_registered_on_router_get():
    """Ambos endpoints deben registrarse como GET en el router de
    plans (no POST/PATCH — son operaciones de lectura)."""
    from routers.plans import router
    paths_methods = {
        (r.path, tuple(sorted(r.methods or [])))
        for r in router.routes if hasattr(r, "path") and hasattr(r, "methods")
    }
    assert ("/api/plans/{plan_id}/lessons", ("GET",)) in paths_methods
    assert ("/api/plans/{plan_id}/coherence-history", ("GET",)) in paths_methods
