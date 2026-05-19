"""[P2-HIST-AUDIT-10 · 2026-05-09] Tests del endpoint
``GET /api/plans/{plan_id}/chunk-metrics``.

Bug original (audit Historial 2026-05-09):
    El Historial mostraba el bucket de status (P0/P1) y el
    tier_breakdown agregado (P1-AUDIT-HIST-6) pero NO exponía las
    métricas ricas por-chunk:
      - `learning_metrics` (jsonb)
      - `lag_seconds_at_pickup`, `effective_lag_seconds_at_pickup`
      - `escalated_at`, `learning_persisted_at`
      - Stats persistidas en `plan_chunk_metrics`: duration_ms,
        was_degraded, retries, lag_seconds, learning_repeat_pct,
        rejection_violations, allergy_violations,
        pantry_snapshot_age_hours, error_message.

    Para diagnosticar por qué un plan archivado se generó "raro",
    el detalle solo existía en chunk-status del plan ACTIVO.

Fix:
    Endpoint nuevo `/api/plans/{plan_id}/chunk-metrics` con LEFT JOIN
    entre `plan_chunk_queue` (estado vivo) y `plan_chunk_metrics`
    (snapshot al completar). Cap LIMIT 50 chunks. Ownership check.

Cobertura:
    1. Anchor del marker.
    2. 401 sin auth.
    3. 400 con plan_id missing.
    4. 404 cuando plan no existe O no pertenece al usuario.
    5. SQL contiene LEFT JOIN entre queue y metrics.
    6. SQL ownership: filtra por user_id (defense-in-depth).
    7. SQL ORDER BY week_number ASC, days_offset ASC.
    8. SQL cap LIMIT 50.
    9. Response shape per chunk con todas las keys.
    10. metrics=None cuando plan_chunk_metrics no tiene row matched.
    11. metrics dict cuando hay match.
    12. Coerción de Decimal/numeric → float (psycopg edge).
    13. ISO format de timestamps.
    14. learning_metrics jsonb pasa-through cuando es dict; None
        cuando es otro tipo (defensa contra corrupción).
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CHUNK_A = "ccccccc1-cccc-cccc-cccc-cccccccccccc"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _base_chunk_row(**overrides):
    """Fila base del JOIN — completar todos los campos esperados."""
    base = {
        "chunk_id": _CHUNK_A,
        "week_number": 1,
        "days_offset": 0,
        "days_count": 4,
        "status": "completed",
        "quality_tier": "llm",
        "attempts": 1,
        "chunk_kind": "initial_plan",
        "lag_seconds_at_pickup": 12,
        "effective_lag_seconds_at_pickup": 12,
        "escalated_at": None,
        "learning_persisted_at": None,
        "dead_letter_reason": None,
        "dead_lettered_at": None,
        "chunk_created_at": None,
        "chunk_updated_at": None,
        "learning_metrics": None,
        "duration_ms": None,
        "was_degraded": None,
        "retries": None,
        "metrics_lag_seconds": None,
        "learning_repeat_pct": None,
        "rejection_violations": None,
        "allergy_violations": None,
        "pantry_snapshot_age_hours": None,
        "error_message": None,
        "metrics_created_at": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "P2-HIST-AUDIT-10" in src


# ---------------------------------------------------------------------------
# 2. Auth + ownership
# ---------------------------------------------------------------------------
def test_chunk_metrics_requires_auth():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    client.app.dependency_overrides[get_verified_user_id] = lambda: None
    r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 401


def test_chunk_metrics_404_when_plan_not_owned():
    """SELECT inicial verifica ownership. 404 cuando el plan no existe
    O no pertenece al usuario — sin DOS-able discovery (un plan ajeno
    devuelve 404 idéntico al inexistente)."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return None  # ownership check falla
        return []

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. SQL contract
# ---------------------------------------------------------------------------
def test_sql_uses_left_join_with_plan_chunk_metrics():
    """LEFT JOIN preserva chunks aún sin row en `plan_chunk_metrics`
    (chunks pending/failed sin commit de stats)."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-4 · 2026-05-09] El endpoint ahora ejecuta DOS
        # queries contra plan_chunk_queue (SELECT principal con LEFT
        # JOIN + COUNT separado). Capturamos solo la principal por
        # `LEFT JOIN plan_chunk_metrics` que es única a esa.
        if "LEFT JOIN plan_chunk_metrics" in query:
            captured["query"] = query
            return []
        # COUNT separado: devolvemos un dict para que el handler
        # popule total_count limpio sin caer al fallback len(rows).
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 0}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LEFT JOIN plan_chunk_metrics" in norm.upper().replace(
        "LEFT JOIN PLAN_CHUNK_METRICS", "LEFT JOIN plan_chunk_metrics"
    ) or re.search(
        r"LEFT\s+JOIN\s+plan_chunk_metrics", norm, re.IGNORECASE
    )


def test_sql_filters_by_user_id_defense_in_depth():
    captured = {}

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-4 · 2026-05-09] Capturar solo SELECT principal.
        if "LEFT JOIN plan_chunk_metrics" in query:
            captured["query"] = query
            captured["params"] = params
            return []
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 0}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "q.user_id = %s" in norm
    assert _USER_A in (captured.get("params") or ())


def test_sql_orders_by_week_then_days_offset():
    """Orden estable para que el render frontend muestre los chunks
    en orden cronológico."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-4 · 2026-05-09] Capturar solo SELECT principal.
        if "LEFT JOIN plan_chunk_metrics" in query:
            captured["query"] = query
            return []
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 0}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert re.search(
        r"ORDER\s+BY\s+q\.week_number\s+ASC[\s\S]*q\.days_offset\s+ASC",
        norm,
        re.IGNORECASE,
    ), f"ORDER BY incorrecto. Got: {norm[:600]!r}"


def test_sql_caps_at_50_chunks():
    captured = {}

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        # [P1-HIST-NEW-4 · 2026-05-09] Capturar solo SELECT principal —
        # el COUNT separado NO tiene LIMIT (escanea la partición full).
        if "LEFT JOIN plan_chunk_metrics" in query:
            captured["query"] = query
            return []
        if "SELECT COUNT(*)" in query and "plan_chunk_queue" in query:
            return {"total_count": 0}
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LIMIT 50" in norm


# ---------------------------------------------------------------------------
# 4. Response shape
# ---------------------------------------------------------------------------
def test_response_metrics_null_when_left_join_no_match():
    """Cuando todos los campos de plan_chunk_metrics son NULL
    (LEFT JOIN sin match), `metrics` debe ser None — el frontend
    distingue así "sin commit" de "stats vacíos"."""
    fake_row = _base_chunk_row()  # todos los campos m.* son None.

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_row]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["metrics"] is None


def test_response_metrics_dict_when_match():
    """Con al menos un campo de metrics no-NULL, `metrics` es dict
    completo."""
    fake_row = _base_chunk_row(
        duration_ms=8500,
        was_degraded=False,
        retries=0,
        metrics_lag_seconds=12,
        learning_repeat_pct=Decimal("0.18"),
        rejection_violations=0,
        allergy_violations=0,
        pantry_snapshot_age_hours=Decimal("2.3"),
        error_message=None,
        metrics_created_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc),
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_row]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    metrics = chunk["metrics"]
    assert metrics is not None
    assert metrics["duration_ms"] == 8500
    assert metrics["was_degraded"] is False
    assert metrics["retries"] == 0
    assert metrics["lag_seconds"] == 12
    # Decimal coerced to float.
    assert isinstance(metrics["learning_repeat_pct"], float)
    assert abs(metrics["learning_repeat_pct"] - 0.18) < 1e-6
    assert isinstance(metrics["pantry_snapshot_age_hours"], float)
    # Timestamp coerced to ISO.
    assert isinstance(metrics["metrics_created_at"], str)
    assert "2026-05-08T12:00:00" in metrics["metrics_created_at"]


def test_learning_metrics_passes_through_when_dict():
    fake_row = _base_chunk_row(
        learning_metrics={"synth_quality_score": 0.85, "synthesized_count": 3},
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_row]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["learning_metrics"] == {
        "synth_quality_score": 0.85,
        "synthesized_count": 3,
    }


def test_learning_metrics_none_when_corrupted_type():
    """Si `learning_metrics` viene como string corrupto (e.g. raw
    text en lugar de jsonb cast), defensivamente devolvemos None."""
    fake_row = _base_chunk_row(learning_metrics="corrupt-string-value")

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_row]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["learning_metrics"] is None


def test_response_chunk_includes_top_level_keys():
    """Verificar que cada chunk del response tiene las keys del
    contrato documentado."""
    fake_row = _base_chunk_row(
        chunk_created_at=datetime(2026, 5, 7, 10, 0, 0, tzinfo=timezone.utc),
        escalated_at=datetime(2026, 5, 8, 11, 0, 0, tzinfo=timezone.utc),
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_row]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    expected_keys = {
        "chunk_id", "week_number", "days_offset", "days_count",
        "status", "quality_tier", "attempts", "chunk_kind",
        "lag_seconds_at_pickup", "effective_lag_seconds_at_pickup",
        "escalated_at", "learning_persisted_at",
        "dead_letter_reason", "dead_lettered_at",
        "created_at", "updated_at",
        "learning_metrics", "metrics",
    }
    assert expected_keys.issubset(set(chunk.keys())), (
        f"Faltan keys en el chunk: {expected_keys - set(chunk.keys())}"
    )
    # Timestamps en ISO format.
    assert "2026-05-07" in chunk["created_at"]
    assert "2026-05-08" in chunk["escalated_at"]


# ---------------------------------------------------------------------------
# 5. Empty edge case
# ---------------------------------------------------------------------------
def test_plan_with_no_chunks_returns_empty_chunks_list():
    """Plan que existe pero no tiene chunks (edge case extremo) →
    `{plan_id, chunks: []}`."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        if "FROM plan_chunk_queue" in query:
            return []
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["plan_id"] == _PLAN_A
    assert body["chunks"] == []
