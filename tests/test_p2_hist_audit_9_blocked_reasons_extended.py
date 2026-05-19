"""[P2-HIST-AUDIT-9 · 2026-05-09] Tests del endpoint extendido
``GET /api/plans/{plan_id}/blocked_reasons``.

Bug original (audit Historial 2026-05-09):
    El endpoint `/blocked_reasons` ya existía pero:
      - Solo conocía 4 reason_codes (zero-log, stale_snapshot,
        stale_snapshot_live_unreachable, empty_pantry). Los demás
        (tz_unresolved, missing_prior_lessons, missing_start_date_no_anchor,
        recovery_exhausted, unrecoverable_*) caían al fallback
        `empty_pantry`, mostrando un título incorrecto al usuario.
      - Solo filtraba `status='pending_user_action'`. Chunks `failed`
        con dead_letter_reason poblado eran invisibles → el modal
        del Historial mostraba un único `_user_action_required`
        agregado de plan_data, sin enumerar cada chunk.

Fix:
    - reason_to_text dict extendido con los 7 reason_codes faltantes
      (3 de pause + 4 dead_letter_reasons + 2 restore_*).
    - Param query `include_failed=true` (default False) que extiende
      el WHERE con `OR (status='failed' AND dead_letter_reason IS NOT NULL)`.
    - Selección de reason_code prioritiza dead_letter_reason cuando
      el chunk es failed; luego `_pause_reason`/`_pantry_pause_reason`.
    - Fallback genérico `_unknown` reemplaza el fallback `empty_pantry`
      mentiroso.
    - Response per-chunk añade `status` + `dead_letter_reason` raw.

Cobertura:
    1. Anchor del marker en el endpoint.
    2. Default (include_failed=False) preserva comportamiento legacy
       (Dashboard plan ACTIVO no se rompe).
    3. include_failed=True extiende el WHERE.
    4. Selección de reason: dead_letter_reason gana sobre snapshot.
    5. Reason_codes nuevos (tz_unresolved, recovery_exhausted, etc.)
       resuelven a templates específicos (no fallback).
    6. Fallback genérico `_unknown` se usa cuando no hay reason
       válido (NO el `empty_pantry` mentiroso pre-fix).
    7. Response per-chunk incluye `status` + `dead_letter_reason`.
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


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert "P2-HIST-AUDIT-9" in src


# ---------------------------------------------------------------------------
# 2. Default (include_failed=False) preserva legacy
# ---------------------------------------------------------------------------
def test_default_filter_only_pending_user_action():
    """Sin `include_failed=true`, el endpoint debe filtrar SOLO por
    `status='pending_user_action'` — el Dashboard del plan ACTIVO
    depende de este comportamiento."""
    captured = {"queries": []}

    def _fake(query, params=None, **kwargs):
        captured["queries"].append(query)
        # Primer query: ownership SELECT. Segundo: rows. Tercer:
        # logging_pref. Devolver shape correcto por orden.
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return []
        if "FROM user_profiles" in query:
            return None
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/blocked_reasons")
    assert r.status_code == 200

    queue_query = next(
        (q for q in captured["queries"] if "FROM plan_chunk_queue" in q),
        None,
    )
    assert queue_query is not None
    norm = re.sub(r"\s+", " ", queue_query)
    # Verificar que SOLO filtra pending_user_action (NO failed).
    assert "status = 'pending_user_action'" in norm
    assert "OR (status = 'failed'" not in norm, (
        "Sin include_failed=true, el endpoint NO debe traer chunks failed."
    )


def test_include_failed_extends_filter():
    """Con `include_failed=true`, el WHERE incluye `OR (status='failed'
    AND dead_letter_reason IS NOT NULL)`."""
    captured = {"queries": []}

    def _fake(query, params=None, **kwargs):
        captured["queries"].append(query)
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return []
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(
            f"/api/plans/{_PLAN_A}/blocked_reasons?include_failed=true"
        )
    assert r.status_code == 200

    queue_query = next(
        (q for q in captured["queries"] if "FROM plan_chunk_queue" in q),
        None,
    )
    assert queue_query is not None
    norm = re.sub(r"\s+", " ", queue_query)
    assert "status = 'pending_user_action'" in norm
    assert re.search(
        r"OR\s*\(\s*status\s*=\s*'failed'\s+AND\s+dead_letter_reason\s+IS\s+NOT\s+NULL",
        norm,
        re.IGNORECASE,
    ), f"WHERE no extendido a failed+dead_letter_reason. Got: {norm[:600]!r}"


# ---------------------------------------------------------------------------
# 3. Selección de reason_code per chunk
# ---------------------------------------------------------------------------
def test_failed_chunk_uses_dead_letter_reason_as_reason_code():
    """Cuando el chunk es status='failed' con dead_letter_reason
    poblado, ese gana sobre cualquier `_pause_reason` del snapshot
    (que pudo quedar stale tras la escalación)."""
    fake_chunk = {
        "id": "ccc",
        "week_number": 2,
        "pipeline_snapshot": {"_pause_reason": "missing_prior_lessons"},
        "status": "failed",
        "dead_letter_reason": "recovery_exhausted",
        "paused_seconds": 3600,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(
            f"/api/plans/{_PLAN_A}/blocked_reasons?include_failed=true"
        )
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is True
    assert len(body["reasons"]) == 1
    reason = body["reasons"][0]
    # `recovery_exhausted` (dead_letter_reason) gana sobre el
    # `missing_prior_lessons` del snapshot.
    assert reason["reason_code"] == "recovery_exhausted"
    assert reason["status"] == "failed"
    assert reason["dead_letter_reason"] == "recovery_exhausted"


def test_pending_chunk_uses_pause_reason_from_snapshot():
    """Para chunks pending_user_action, `_pause_reason` (P1-CHUNKS-3
    missing_prior_lessons) tiene prioridad sobre `_pantry_pause_reason`."""
    fake_chunk = {
        "id": "ccc",
        "week_number": 1,
        "pipeline_snapshot": {
            "_pause_reason": "missing_prior_lessons",
            "_pantry_pause_reason": "empty_pantry",
        },
        "status": "pending_user_action",
        "dead_letter_reason": None,
        "paused_seconds": 1800,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/blocked_reasons")
    assert r.status_code == 200
    reason = r.json()["reasons"][0]
    assert reason["reason_code"] == "missing_prior_lessons"


def test_pantry_pause_reason_fallback_when_no_pause_reason():
    """Cuando solo `_pantry_pause_reason` está presente (path legacy),
    se usa ese."""
    fake_chunk = {
        "id": "ccc",
        "week_number": 1,
        "pipeline_snapshot": {"_pantry_pause_reason": "tz_unresolved"},
        "status": "pending_user_action",
        "dead_letter_reason": None,
        "paused_seconds": 1800,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/blocked_reasons")
    assert r.status_code == 200
    reason = r.json()["reasons"][0]
    assert reason["reason_code"] == "tz_unresolved"
    # Nuevo template, NO fallback empty_pantry.
    assert "zona horaria" in reason["title"].lower()


# ---------------------------------------------------------------------------
# 4. Reason_codes nuevos resuelven a templates específicos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dead_reason,expected_keyword", [
    ("recovery_exhausted", "atención"),
    ("unrecoverable_missing_anchor", "regenerarse"),
    ("unrecoverable_corrupted_date", "regenerarse"),
    ("missing_prior_lessons_unrecoverable", "regenerarse"),
    ("restore_overwrite", "cancelado"),
    ("restore_source_archived", "cancelado"),
])
def test_dead_letter_reason_resolves_to_specific_template(dead_reason, expected_keyword):
    """Cada dead_letter_reason canónico debe tener un template
    específico (NO el fallback genérico)."""
    fake_chunk = {
        "id": "ccc",
        "week_number": 3,
        "pipeline_snapshot": {},
        "status": "failed",
        "dead_letter_reason": dead_reason,
        "paused_seconds": 3600,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(
            f"/api/plans/{_PLAN_A}/blocked_reasons?include_failed=true"
        )
    assert r.status_code == 200
    reason = r.json()["reasons"][0]
    assert reason["reason_code"] == dead_reason
    body = (reason.get("body") or "") + " " + (reason.get("title") or "")
    assert expected_keyword.lower() in body.lower(), (
        f"Reason {dead_reason!r} no tiene template específico — "
        f"cayó al fallback genérico."
    )


# ---------------------------------------------------------------------------
# 5. Fallback genérico (NO empty_pantry mentiroso)
# ---------------------------------------------------------------------------
def test_unknown_reason_uses_generic_fallback_not_empty_pantry():
    """Cuando un chunk no tiene `_pause_reason`, `_pantry_pause_reason`,
    ni `dead_letter_reason`, el fallback debe ser genérico ("Bloqueo
    sin clasificar") — NO el `empty_pantry` mentiroso pre-fix.
    """
    fake_chunk = {
        "id": "ccc",
        "week_number": 1,
        "pipeline_snapshot": {},  # sin reason fields
        "status": "pending_user_action",
        "dead_letter_reason": None,
        "paused_seconds": 1800,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/blocked_reasons")
    assert r.status_code == 200
    reason = r.json()["reasons"][0]
    # NO debe ser el empty_pantry copy mentiroso.
    assert "nevera" not in (reason.get("title") or "").lower()
    # Debe ser el fallback genérico.
    assert "sin clasificar" in (reason.get("title") or "").lower(), (
        f"Fallback genérico esperado. Got title: {reason.get('title')!r}"
    )


# ---------------------------------------------------------------------------
# 6. Response shape: status + dead_letter_reason raw expuestos
# ---------------------------------------------------------------------------
def test_response_includes_status_and_dead_letter_reason_raw():
    """El frontend (modal del Historial) puede inspeccionar
    `status` + `dead_letter_reason` raw para diagnóstico avanzado
    sin tener que parsear el reason_code."""
    fake_chunk = {
        "id": "ccc",
        "week_number": 1,
        "pipeline_snapshot": {"_pause_reason": "tz_unresolved"},
        "status": "pending_user_action",
        "dead_letter_reason": None,
        "paused_seconds": 1800,
    }

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM plan_chunk_queue" in query:
            return [fake_chunk]
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/blocked_reasons")
    assert r.status_code == 200
    reason = r.json()["reasons"][0]
    assert "status" in reason
    assert "dead_letter_reason" in reason
    assert reason["status"] == "pending_user_action"
    assert reason["dead_letter_reason"] is None
