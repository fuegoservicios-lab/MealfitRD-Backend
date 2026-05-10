"""[P1-HIST-NEW-4 · 2026-05-09] Tests del `total_count` en el endpoint
`/{plan_id}/chunk-metrics`.

Bug original (audit profundo Historial 2026-05-09):
    El endpoint aplica `LIMIT 50` defensivo. Para planes tier ultra
    (90 días) con rolling refills + post-swap re-enqueues que dejan
    completed+failed coexistentes para misma (plan, week) tras
    P0-HIST-NEW-1, la cardinalidad real puede exceder 50. La
    respuesta venía silently truncada — sin contador, sin notice.
    Operadores haciendo post-mortem no sabían si veían la lista
    completa.

Fix:
    Response extendido con `total_count` (COUNT separado, sin LIMIT)
    y `limit` (constante actual). Frontend muestra "Mostrando X de N"
    cuando `total_count > chunks.length`.

Cobertura backend:
    1. Anchor del marker en el endpoint Y en History.jsx.
    2. SQL incluye COUNT(*) separado del SELECT principal con LEFT
       JOIN (window function `COUNT(*) OVER` agregaría I/O por row;
       count separado es más barato).
    3. Response shape: `total_count: int`, `limit: int`, `chunks: []`.
    4. total_count refleja el conteo real (no se afecta por LIMIT).
    5. total_count >= len(chunks) (invariante).
    6. Plan sin chunks → total_count=0, chunks=[].
    7. Ownership check sigue activo (no expone count para planes
       ajenos).
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_USER_B = "22222222-2222-2222-2222-222222222222"
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
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "P1-HIST-NEW-4" in src, (
        "Endpoint chunk-metrics debe citar `P1-HIST-NEW-4` para que "
        "un grep + git blame lleve directo al fix del total_count."
    )


def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P1-HIST-NEW-4" in text, (
        "Marker `P1-HIST-NEW-4` debe aparecer en History.jsx donde "
        "vive el render del notice de truncado."
    )


# ---------------------------------------------------------------------------
# 2. SQL: COUNT separado (no window function)
# ---------------------------------------------------------------------------
def test_sql_uses_separate_count_query():
    """El handler debe usar un SELECT COUNT(*) separado, no una
    window function `COUNT(*) OVER ()` en el SELECT principal. La
    window agregaría el conteo a CADA row del LEFT JOIN, multiplicando
    I/O sin que las filas excluidas por LIMIT se lean. El count
    separado escanea la partición una vez."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    # `SELECT COUNT(*)::int AS total_count FROM plan_chunk_queue
    #  WHERE meal_plan_id = %s AND user_id = %s`
    assert re.search(
        r"SELECT\s+COUNT\(\*\)[^F]+FROM\s+plan_chunk_queue",
        norm,
        re.IGNORECASE,
    ), (
        "Handler debe ejecutar un SELECT COUNT(*) separado (no window). "
        "Got source slice: {!r}".format(norm[:2000])
    )
    # Anti-pattern: si aparece `COUNT(*) OVER` en el SELECT principal,
    # es la window function que queremos evitar.
    main_select_block = re.search(
        r"SELECT[\s\S]+?FROM\s+plan_chunk_queue\s+q",
        src,
    )
    assert main_select_block is not None
    assert "COUNT(*) OVER" not in main_select_block.group(0), (
        "El SELECT principal NO debe usar `COUNT(*) OVER ()` — usa "
        "un SELECT COUNT(*) separado para evitar I/O por row."
    )


def test_sql_count_filters_by_user_id():
    """El COUNT debe filtrar por meal_plan_id Y user_id (defense-in-
    depth + RLS). Sin user_id, agregaría filas de otros usuarios si
    el plan es shared/migrated entre cuentas."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # Buscar el bloque de COUNT y verificar el WHERE.
    m = re.search(
        r"SELECT\s+COUNT\(\*\)[\s\S]+?WHERE([\s\S]+?)\"\"\"",
        src,
    )
    assert m is not None, "No pude extraer el bloque COUNT."
    where = m.group(1)
    assert "meal_plan_id = %s" in where
    assert "user_id = %s" in where


# ---------------------------------------------------------------------------
# 3. Response shape: total_count + limit
# ---------------------------------------------------------------------------
def test_response_includes_total_count_and_limit():
    """Response debe traer ambas keys como int. Frontend depende de
    `total_count > chunks.length` para el notice y de `limit` para
    el tooltip."""
    fake_row = _base_chunk_row()

    def _fake(query, params=None, **kwargs):
        # Ownership check.
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        # COUNT separado.
        if "SELECT COUNT(*)::int AS total_count" in query:
            return {"total_count": 67}
        # SELECT principal.
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "total_count" in body, (
        f"Response debe incluir `total_count`. Got keys: {sorted(body.keys())}"
    )
    assert "limit" in body, (
        f"Response debe incluir `limit` para el tooltip del notice. "
        f"Got keys: {sorted(body.keys())}"
    )
    assert isinstance(body["total_count"], int)
    assert isinstance(body["limit"], int)
    assert body["total_count"] == 67
    assert body["limit"] == 50  # cap actual


def test_response_total_count_zero_when_no_chunks():
    """Plan sin chunks → total_count=0, chunks=[]. Sin esto, un
    plan recién creado disparía el notice falsamente."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "SELECT COUNT(*)::int AS total_count" in query:
            return {"total_count": 0}
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] == 0
    assert body["chunks"] == []


def test_response_total_count_handles_none_from_count_query():
    """Defensivo: si el COUNT devuelve None (cardinalidad cero, fila
    vacía, etc.), el endpoint debe coercer a 0 en lugar de propagar
    None al response (rompería el frontend que espera int)."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "SELECT COUNT(*)::int AS total_count" in query:
            return {"total_count": None}
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] == 0


# ---------------------------------------------------------------------------
# 4. Invariante: total_count >= len(chunks)
# ---------------------------------------------------------------------------
def test_invariant_total_count_at_least_len_chunks():
    """Si el COUNT separado devuelve menos que los rows del SELECT
    principal, hay un bug del COUNT (probable race entre las dos
    queries con un cron purgando rows). El endpoint NO debe coercer
    — preferimos exponer la inconsistencia para que un dev la
    diagnostique. Pero al menos verificamos el caso normal."""
    fake_rows = [_base_chunk_row(), _base_chunk_row(week_number=2)]

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        if "SELECT COUNT(*)::int AS total_count" in query:
            return {"total_count": 50}
        return fake_rows

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] >= len(body["chunks"]), (
        f"Invariante: total_count ({body['total_count']}) debe ser "
        f">= len(chunks) ({len(body['chunks'])})."
    )


# ---------------------------------------------------------------------------
# 5. Ownership check sigue activo
# ---------------------------------------------------------------------------
def test_ownership_check_returns_404_for_other_user_plan():
    """Plan ajeno → 404 sin DOS-able discovery. El total_count NO
    debe filtrarse via timing del COUNT antes de validar ownership."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            # Plan no encontrado para este user.
            return None
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_B

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 404
    # Body NO debe incluir total_count en error response.
    body = r.json()
    assert "total_count" not in body
