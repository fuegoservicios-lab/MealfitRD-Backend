"""[P2-HIST-NEW-3 · 2026-05-09] Tests del payload `days_offset` y
`days_count` en el endpoint chunk-metrics — cross-link del marker.

Bug original (audit profundo Historial 2026-05-09):
    El badge del tab Métricas mostraba solo "Semana 1 · rolling_refill"
    — el rango de días concretos del chunk ya viajaba en el payload
    (`days_offset` + `days_count`) pero el frontend lo descartaba.
    Operadores no podían correlacionar la card de Métricas con el
    menú renderizado del tab Menú.

    Fix 100% client-side (compute + render del label "Días X–Y").
    Este test cierra el cross-link del marker (P2-HIST-AUDIT-14
    requiere `tests/test_p2_hist_new_3*.py`) Y protege los dos
    campos del payload contra refactors.

Cobertura:
    1. Anchor del marker en History.jsx.
    2. SQL del SELECT incluye `q.days_offset` Y `q.days_count`.
    3. Response shape: ambos campos como int en el chunk dict.
    4. Plan con chunks de days_count=1 y days_count>1 viajan ambos.
    5. days_offset=0 (primer chunk) viaja como 0 sin coerce a None.
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
        "deferrals_count": 0,
        "deferral_reasons": None,
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
    assert "[P2-HIST-NEW-3" in text, (
        "Marker `P2-HIST-NEW-3` debe aparecer en History.jsx donde "
        "vive el compute del label `Días X–Y`."
    )


# ---------------------------------------------------------------------------
# 2. SQL del SELECT incluye los inputs del label
# ---------------------------------------------------------------------------
def test_sql_select_includes_days_offset_and_days_count():
    """SELECT debe pedir `q.days_offset` Y `q.days_count`. Sin estos,
    el frontend no recibe los inputs y el label nunca se dibuja
    aunque la lógica esté correcta."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(r"q\.days_offset", norm), (
        "SELECT debe incluir `q.days_offset`."
    )
    assert re.search(r"q\.days_count", norm), (
        "SELECT debe incluir `q.days_count`."
    )


# ---------------------------------------------------------------------------
# 3. Response shape end-to-end
# ---------------------------------------------------------------------------
def test_response_includes_both_fields_as_int():
    """End-to-end: row con days_offset/days_count → response devuelve
    ambos como int para que el frontend pueda compute el label."""
    fake_row = _base_chunk_row(days_offset=3, days_count=4)

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
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
    assert chunk["days_offset"] == 3
    assert chunk["days_count"] == 4
    assert isinstance(chunk["days_offset"], int)
    assert isinstance(chunk["days_count"], int)


def test_response_handles_zero_offset_first_chunk():
    """First chunk del plan tiene days_offset=0. Edge case: el handler
    NO debe coerce 0 a None (frontend tiene guard >= 0 que cubre esto
    correctamente; el bug clásico sería `int(value or 0)` que pierde
    info cuando el value real es 0)."""
    fake_row = _base_chunk_row(days_offset=0, days_count=3)

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
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
    assert chunk["days_offset"] == 0
    assert chunk["days_count"] == 3


def test_response_single_day_chunk():
    """Chunk de 1 día (days_count=1) — el frontend renderiza singular
    "Día N". Backend solo necesita pasar el campo correcto."""
    fake_row = _base_chunk_row(days_offset=5, days_count=1)

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "LEFT JOIN plan_chunk_metrics" in query:
            return [fake_row]
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
    assert chunk["days_count"] == 1
