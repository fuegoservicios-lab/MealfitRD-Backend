"""[P1-HIST-NEW-5 · 2026-05-09] Tests del chip ratio lag/SLA en el
tab Métricas — cross-link del marker + payload contract.

Bug original (audit profundo Historial 2026-05-09):
    El tab Métricas pintaba `Lag: 240s` y `SLA: 60s` como chips
    independientes. Para diagnosticar "este chunk tomó 4× lo esperado"
    el operator hacía math mental — la señal anómala se perdía.

    Fix 100% client-side (chip warn cuando ratio >= 2). Este test
    cierra el cross-link del marker (`test_p2_hist_audit_14_marker_test_link`
    requiere `tests/test_p1_hist_new_5*.py`) Y protege los dos
    campos del payload (`expected_preemption_seconds` + `lag_seconds`)
    contra refactors que los borren.

Cobertura backend:
    1. Anchor del marker en History.jsx.
    2. Endpoint chunk-metrics expone `expected_preemption_seconds`
       (top-level del chunk dict).
    3. Endpoint expone `lag_seconds` dentro de `metrics` o
       `lag_seconds_at_pickup` top-level (frontend usa fallback).
    4. Ambos campos viajan como int o None.
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
        "lag_seconds_at_pickup": 240,
        "effective_lag_seconds_at_pickup": 240,
        "expected_preemption_seconds": 60,
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
        "metrics_lag_seconds": 240,
        "learning_repeat_pct": None,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "pantry_snapshot_age_hours": None,
        "error_message": None,
        "is_rolling_refill": False,
        "metrics_created_at": None,
        "blocking_lock_chunk_id": None,
        "blocking_lock_age_seconds": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker — fix vive en History.jsx (client-side)
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P1-HIST-NEW-5" in text, (
        "Marker `P1-HIST-NEW-5` debe aparecer en History.jsx donde "
        "vive el chip ratio lag/SLA."
    )


# ---------------------------------------------------------------------------
# 2. SQL del SELECT incluye los inputs del ratio
# ---------------------------------------------------------------------------
def test_sql_select_includes_expected_preemption_and_lag():
    """El SELECT debe pedir `q.expected_preemption_seconds` Y los lag
    fields (`q.lag_seconds_at_pickup` o `m.lag_seconds`). Sin estos,
    el frontend no recibe los inputs y el chip ratio nunca se dibuja
    aunque la lógica esté correcta."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(r"q\.expected_preemption_seconds", norm), (
        "SELECT debe incluir `q.expected_preemption_seconds`."
    )
    # Lag puede venir de q.* o m.* — el frontend usa fallback. Aquí
    # exigimos AL MENOS uno (idealmente ambos).
    assert (
        "q.lag_seconds_at_pickup" in norm
        or "m.lag_seconds" in norm
    ), "SELECT debe incluir lag_seconds (de queue o metrics)."


# ---------------------------------------------------------------------------
# 3. Response shape end-to-end
# ---------------------------------------------------------------------------
def test_response_includes_expected_preemption_and_lag():
    """End-to-end: row con SLA + lag → response devuelve ambos como
    int para que el frontend pueda calcular el ratio."""
    fake_row = _base_chunk_row(
        expected_preemption_seconds=60,
        metrics_lag_seconds=240,
        lag_seconds_at_pickup=240,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-6 · 2026-05-09] SELECT principal contiene un
        # `SELECT COUNT(*)` embebido en el LATERAL chunk_deferrals.
        # Discriminamos por el JOIN con plan_chunk_metrics que solo
        # tiene la query principal.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200, r.text
    chunk = r.json()["chunks"][0]
    # Top-level: expected_preemption_seconds + lag_seconds_at_pickup.
    assert chunk["expected_preemption_seconds"] == 60
    assert chunk["lag_seconds_at_pickup"] == 240
    # Inner metrics: lag_seconds (preferido por el frontend cuando hay
    # commit de plan_chunk_metrics).
    assert chunk["metrics"] is not None
    assert chunk["metrics"]["lag_seconds"] == 240


def test_response_handles_zero_sla_without_breaking():
    """Chunk sin reserva (`expected_preemption_seconds = 0`) debe
    viajar como 0 — el frontend tiene guard `_sla > 0` que evita
    división por cero. Este test fija el contrato: NO se coerce a None."""
    fake_row = _base_chunk_row(
        expected_preemption_seconds=0,
        metrics_lag_seconds=240,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-6 · 2026-05-09] SELECT principal contiene un
        # `SELECT COUNT(*)` embebido en el LATERAL chunk_deferrals.
        # Discriminamos por el JOIN con plan_chunk_metrics que solo
        # tiene la query principal.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["expected_preemption_seconds"] == 0


def test_response_handles_null_sla():
    """Chunk legacy sin reserva persistida → SLA viaja como None.
    Frontend skipea el chip ratio."""
    fake_row = _base_chunk_row(
        expected_preemption_seconds=None,
        metrics_lag_seconds=240,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-6 · 2026-05-09] SELECT principal contiene un
        # `SELECT COUNT(*)` embebido en el LATERAL chunk_deferrals.
        # Discriminamos por el JOIN con plan_chunk_metrics que solo
        # tiene la query principal.
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 1}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["expected_preemption_seconds"] is None
