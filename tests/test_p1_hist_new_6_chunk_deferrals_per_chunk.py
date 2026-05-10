"""[P1-HIST-NEW-6 · 2026-05-09] Tests del surface per-chunk de
`chunk_deferrals` en el endpoint chunk-metrics.

Bug original (audit profundo Historial 2026-05-09):
    `chunk_deferrals` es la tabla de telemetría que registra cada vez
    que un gate del pipeline LangGraph (temporal_gate,
    learning_zero_logs, missing_prior_lessons, etc.) difirió un chunk.
    Solo visible vía endpoint admin `/admin/chunk_deferrals/{user_id}`.
    Para diagnosticar "por qué este plan tardó 3h en arrancar" no
    había surface en el Historial.

Fix:
    LATERAL en el SELECT de chunk-metrics agrega `deferrals_count`
    (COUNT) y `deferral_reasons` (array_agg DISTINCT). Frontend
    renderiza chip "Diferido N×" warn cuando count >= 3.

Cobertura backend:
    1. Anchor del marker en endpoint Y en History.jsx.
    2. SQL incluye LATERAL FROM chunk_deferrals.
    3. Join por (meal_plan_id, week_number, user_id) — sin chunk_id
       (chunk_deferrals no tiene FK a plan_chunk_queue.id).
    4. Response incluye `deferrals_count: int` (COALESCE 0).
    5. Response incluye `deferral_reasons: [str]|null`.
    6. Plan sin deferrals → count=0, reasons=None.
    7. Reasons NULL/vacío sanitizados a None (frontend distingue
       "sin info" de "lista vacía").
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CHUNK_X = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _base_chunk_row(**overrides):
    base = {
        "chunk_id": _CHUNK_X,
        "week_number": 1,
        "days_offset": 0,
        "days_count": 3,
        "status": "completed",
        "quality_tier": "llm",
        "attempts": 1,
        "chunk_kind": "first_chunk",
        "lag_seconds_at_pickup": 60,
        "effective_lag_seconds_at_pickup": 60,
        "expected_preemption_seconds": None,
        "reservation_status": "ok",
        "escalated_at": None,
        "learning_persisted_at": None,
        "dead_letter_reason": None,
        "dead_lettered_at": None,
        "chunk_created_at": None,
        "chunk_updated_at": None,
        "learning_metrics": None,
        "duration_ms": 5000,
        "was_degraded": False,
        "retries": 0,
        "metrics_lag_seconds": 60,
        "learning_repeat_pct": None,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "pantry_snapshot_age_hours": None,
        "error_message": None,
        "is_rolling_refill": False,
        "metrics_created_at": None,
        "blocking_lock_chunk_id": None,
        "blocking_lock_age_seconds": None,
        # P1-HIST-NEW-6 fields.
        "deferrals_count": 0,
        "deferral_reasons": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "P1-HIST-NEW-6" in src, (
        "Endpoint chunk-metrics debe citar `P1-HIST-NEW-6` para que un "
        "grep + git blame lleve directo al fix del LATERAL chunk_deferrals."
    )


def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P1-HIST-NEW-6" in text


# ---------------------------------------------------------------------------
# 2. SQL: LATERAL FROM chunk_deferrals
# ---------------------------------------------------------------------------
def test_sql_includes_lateral_chunk_deferrals():
    """El endpoint debe agregar via LATERAL — no JOIN directo, porque
    queremos count + array como una sola fila por chunk."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\([\s\S]*?FROM\s+chunk_deferrals",
        norm,
        re.IGNORECASE,
    ), (
        "SQL debe agregar chunk_deferrals via LEFT JOIN LATERAL para "
        "preservar chunks sin deferrals (con count=NULL → COALESCE 0)."
    )


def test_sql_join_predicate_uses_meal_plan_and_week():
    """`chunk_deferrals` NO tiene FK a `plan_chunk_queue.id` — el join
    debe ser por `(meal_plan_id, week_number, user_id)`. El unique
    index parcial garantiza ≤1 chunk vivo por (plan, week) así que el
    aggregation es estable."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # Buscar el bloque del LATERAL con chunk_deferrals.
    m = re.search(
        r"FROM\s+chunk_deferrals\s+WHERE([\s\S]+?)\)",
        src,
    )
    assert m is not None, "No pude extraer el WHERE del LATERAL."
    where = m.group(1)
    assert "meal_plan_id = q.meal_plan_id" in where
    assert "week_number = q.week_number" in where
    # user_id como defense-in-depth (RLS también filtra).
    assert "user_id = q.user_id" in where


def test_sql_aggregates_count_and_distinct_reasons():
    """El LATERAL debe devolver dos columnas: COUNT(*) y array_agg
    DISTINCT de reason. Sin ambos, el chip pierde el tooltip
    diagnóstico."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(
        r"COUNT\(\*\)::int\s+AS\s+deferrals_count",
        norm,
    ), "LATERAL debe devolver `deferrals_count` como int."
    assert re.search(
        r"array_agg\(\s*DISTINCT\s+reason",
        norm,
    ), "LATERAL debe agregar reasons como array DISTINCT."


# ---------------------------------------------------------------------------
# 3. Response shape end-to-end
# ---------------------------------------------------------------------------
def test_response_includes_deferrals_count_and_reasons():
    """Row con deferrals → response devuelve count + reasons como
    list of str."""
    fake_row = _base_chunk_row(
        deferrals_count=4,
        deferral_reasons=["learning_zero_logs", "temporal_gate"],
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # SELECT principal (con LEFT JOIN m) — devuelve filas. La
        # query del LATERAL tiene `SELECT COUNT(*)` también pero está
        # embebida — discriminamos por la presencia del JOIN.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        # COUNT total separado (P1-HIST-NEW-4) — query corta sin JOIN.
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200, r.text
    chunk = r.json()["chunks"][0]
    assert chunk["deferrals_count"] == 4
    assert chunk["deferral_reasons"] == ["learning_zero_logs", "temporal_gate"]
    assert isinstance(chunk["deferrals_count"], int)


def test_response_count_zero_when_no_deferrals():
    """Plan sin deferrals → count=0, reasons=None. Sin coercer
    reasons NULL a [], el frontend distingue 'sin info' (None) de
    'lista vacía explícita' (poco probable pero posible)."""
    fake_row = _base_chunk_row(
        deferrals_count=0,
        deferral_reasons=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # SELECT principal (con LEFT JOIN m) — devuelve filas. La
        # query del LATERAL tiene `SELECT COUNT(*)` también pero está
        # embebida — discriminamos por la presencia del JOIN.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        # COUNT total separado (P1-HIST-NEW-4) — query corta sin JOIN.
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["deferrals_count"] == 0
    assert chunk["deferral_reasons"] is None


def test_response_handles_count_none_from_sql():
    """Defensivo: COALESCE en el SELECT cubre count=NULL del LATERAL,
    pero si llega None igualmente, el handler debe coercer a 0."""
    fake_row = _base_chunk_row(
        deferrals_count=None,
        deferral_reasons=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # SELECT principal (con LEFT JOIN m) — devuelve filas. La
        # query del LATERAL tiene `SELECT COUNT(*)` también pero está
        # embebida — discriminamos por la presencia del JOIN.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        # COUNT total separado (P1-HIST-NEW-4) — query corta sin JOIN.
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["deferrals_count"] == 0


def test_response_sanitizes_empty_reasons_list_to_none():
    """Si reasons llega como [] (lista vacía explícita pero count > 0,
    raro pero posible si reasons fueron NULL en DB), normalizamos a
    None para que el frontend no renderice "Razones: " sin contenido."""
    fake_row = _base_chunk_row(
        deferrals_count=2,
        deferral_reasons=[],
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # SELECT principal (con LEFT JOIN m) — devuelve filas. La
        # query del LATERAL tiene `SELECT COUNT(*)` también pero está
        # embebida — discriminamos por la presencia del JOIN.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        # COUNT total separado (P1-HIST-NEW-4) — query corta sin JOIN.
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["deferrals_count"] == 2
    assert chunk["deferral_reasons"] is None
