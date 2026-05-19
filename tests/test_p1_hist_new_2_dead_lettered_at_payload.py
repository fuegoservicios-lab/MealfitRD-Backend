"""[P1-HIST-NEW-2 · 2026-05-09] Tests del payload `dead_lettered_at`
en el endpoint `/{plan_id}/chunk-metrics`.

Bug original (audit profundo Historial 2026-05-09):
    El endpoint devuelve `dead_lettered_at` Y `escalated_at`. El
    frontend del tab Métricas renderizaba solo `escalated_at` —
    asimetría sin razón. Para chunks terminales `dead_lettered_at`
    es el timestamp canónico (estado final aceptado); `escalated_at`
    es la marca de transición.

    Fix 100% client-side (chip rojo "Dead-letter: <rel>" tras el
    chip amber "Escalado:"). Este test cierra el cross-link del
    marker (`test_p2_hist_audit_14_marker_test_link` requiere
    `tests/test_p1_hist_new_2*.py`) Y protege el campo contra un
    refactor accidental del SELECT que lo borre.

Cobertura backend:
    1. Anchor del marker en History.jsx.
    2. SQL del SELECT incluye `q.dead_lettered_at`.
    3. Handler isoformatea y expone la key en el response.
    4. Response shape: ISO 8601 string o None.
    5. dead_lettered_at puede llegar SIN escalated_at (paths sin
       escalación explícita) — ambos campos son independientes.
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
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
        "status": "failed",
        "quality_tier": None,
        "attempts": 3,
        "chunk_kind": "first_chunk",
        "lag_seconds_at_pickup": 60,
        "effective_lag_seconds_at_pickup": 60,
        "expected_preemption_seconds": None,
        "reservation_status": "ok",
        "escalated_at": None,
        "learning_persisted_at": None,
        "dead_letter_reason": "recovery_exhausted",
        "dead_lettered_at": None,
        "chunk_created_at": None,
        "chunk_updated_at": None,
        "learning_metrics": None,
        "duration_ms": None,
        "was_degraded": None,
        "retries": 3,
        "metrics_lag_seconds": None,
        "learning_repeat_pct": None,
        "rejection_violations": None,
        "allergy_violations": None,
        "pantry_snapshot_age_hours": None,
        "error_message": None,
        "is_rolling_refill": None,
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
    assert "[P1-HIST-NEW-2" in text, (
        "Marker `P1-HIST-NEW-2` debe aparecer en History.jsx donde "
        "vive el render del chip dead-letter."
    )


# ---------------------------------------------------------------------------
# 2. SQL del SELECT incluye q.dead_lettered_at
# ---------------------------------------------------------------------------
def test_sql_select_includes_dead_lettered_at():
    """El SELECT del LEFT JOIN debe pedir `q.dead_lettered_at`. Sin
    esto el handler nunca recibe el campo y el chip del frontend
    queda inerte aunque la lógica esté correcta."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(r"q\.dead_lettered_at", norm), (
        "SELECT debe incluir `q.dead_lettered_at` para que el "
        "frontend renderice el chip terminal."
    )


# ---------------------------------------------------------------------------
# 3. Handler asigna isoformat al response
# ---------------------------------------------------------------------------
def test_response_dict_includes_dead_lettered_at_iso():
    """El handler Python debe popular `dead_lettered_at` en el dict
    de cada chunk con `_iso(...)` (helper local que isoformatea
    datetime). Sin la línea, el field no viaja al frontend."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # `"dead_lettered_at": _iso(r.get("dead_lettered_at"))` (o variante).
    assert re.search(
        r'["\']dead_lettered_at["\']\s*:\s*_iso\(\s*r\.get\(\s*["\']dead_lettered_at["\']',
        src,
    ), (
        "Handler debe asignar `dead_lettered_at: _iso(r.get(...))` "
        "en el dict del chunk. Refactors deben preservar esa línea."
    )


# ---------------------------------------------------------------------------
# 4. Response shape end-to-end
# ---------------------------------------------------------------------------
def test_response_returns_iso_string_when_populated():
    """End-to-end: row con `dead_lettered_at=datetime` → response
    devuelve `chunks[0].dead_lettered_at` como ISO 8601 string."""
    _now = datetime(2026, 5, 9, 14, 30, 0, tzinfo=timezone.utc)
    fake_row = _base_chunk_row(
        status="failed",
        dead_lettered_at=_now,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert len(chunks) == 1
    dl = chunks[0]["dead_lettered_at"]
    assert isinstance(dl, str)
    assert dl.startswith("2026-05-09T14:30:00")  # iso 8601 prefix


def test_response_returns_none_when_absent():
    """Chunk en estado no-terminal → `dead_lettered_at: None`. El
    frontend debe poder distinguir "nunca dead-letter" del valor
    presente."""
    fake_row = _base_chunk_row(
        status="completed",
        dead_lettered_at=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert chunks[0]["dead_lettered_at"] is None


# ---------------------------------------------------------------------------
# 5. Independencia: dead_lettered_at sin escalated_at
# ---------------------------------------------------------------------------
def test_dead_lettered_at_independent_from_escalated_at():
    """Los dos campos son ortogonales en el payload — un chunk puede
    tener `dead_lettered_at` sin `escalated_at` (paths sin escalación
    explícita: timeout cron, mark-dead manual). El frontend renderiza
    los chips con IIFEs separados; este test fija el contrato del
    payload contra una agregación accidental (e.g., COALESCE) que
    los acople en backend."""
    _now = datetime(2026, 5, 9, 14, 30, 0, tzinfo=timezone.utc)
    fake_row = _base_chunk_row(
        status="failed",
        dead_lettered_at=_now,
        escalated_at=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["dead_lettered_at"] is not None
    assert chunk["escalated_at"] is None, (
        "Los dos campos deben viajar independientes — el frontend "
        "decide qué renderizar."
    )
