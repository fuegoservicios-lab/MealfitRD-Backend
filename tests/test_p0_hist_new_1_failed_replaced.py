"""[P0-HIST-NEW-1 · 2026-05-09] Tests del counter `failed_unreplaced`
en /history-list y /history-status-summary.

Bug original (audit profundo Historial 2026-05-09):
    El índice parcial `ux_plan_chunk_queue_live_week`
    (`migrations/p2_new_e_consolidate_runtime_ddl.sql:171`)
    permite coexistencia `completed` + `failed` para misma
    `(meal_plan_id, week_number)` — típicamente cuando un chunk completó
    días, fue re-encolado (post-swap revalidation, manual retry) y el
    segundo intento dead-letteró. La fila vieja sigue contribuyendo a
    `chunk_failed_count` aunque los días YA están en plan_data.days vía
    la fila completed hermana.

    Resultado: `getStatusInfo` (frontend) elevaba el bucket a
    `action_required` por la regla `_fc > 0 → action_required` aunque
    el plan tenía 30/30 días generados. Chip rojo "Acción" persistente
    en planes sanos. Banner action_banner del modal también disparaba
    por `_queueFailed > 0` con queue drift inexistente.

Fix:
    Nuevo counter `failed_unreplaced_count` en la subquery de stats
    (`/history-list` + `/history-status-summary`) que cuenta solo
    chunks `failed` SIN sibling completed para misma (plan, week). El
    frontend prefiere esta key sobre `chunk_failed_count` (con cascada
    legacy para deploy lag).

Cobertura:
    1. Anchor del marker en el endpoint.
    2. SQL contiene la subquery `NOT EXISTS` con sibling completed.
    3. Subquery filtra por mismo `meal_plan_id` Y `week_number`.
    4. Subquery excluye el propio row (`sibling.id != plan_chunk_queue.id`).
    5. Solo cuenta sibling con `status = 'completed'`.
    6. COALESCE(qstats.failed_unreplaced_count, 0) en SELECT principal
       de history-list.
    7. Response incluye `chunk_failed_unreplaced_count` por plan.
    8. Response de history-status-summary incluye `failed_unreplaced_count`.
    9. Counter math: failed con sibling completed → unreplaced=0;
       failed sin sibling → unreplaced=count.
    10. Counter siempre presente como int (default 0).
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


def _base_row(pid, **overrides):
    """Fila representativa del SELECT — mismas keys que el test
    test_p1_audit_hist_4 + el nuevo `chunk_failed_unreplaced_count`."""
    base = {
        "id": pid,
        "name": f"Plan {pid[:4]}",
        "created_at": None,
        "calories": 2000,
        "macros": {"protein": "100g"},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "shift_days_accumulated": None,
        "coherence_history": [],
        "preview_meals_raw": [{"name": "Avena", "meal": "Desayuno"}],
        "goal_root": "lose_weight",
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_failed_unreplaced_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 0,
        "chunk_pantry_degraded_count": 0,
        "chunk_pantry_degraded_reasons": None,
        "chunk_tier_breakdown": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_history_list():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P0-HIST-NEW-1" in src, (
        "history-list debe citar `P0-HIST-NEW-1` para que un grep "
        "+ git blame lleve directo al fix del unreplaced counter."
    )


def test_marker_in_status_summary():
    from routers.plans import api_plans_history_status_summary
    src = inspect.getsource(api_plans_history_status_summary)
    assert "P0-HIST-NEW-1" in src, (
        "history-status-summary debe citar `P0-HIST-NEW-1`. Espeja la "
        "subquery del listado para que el fallback legacy del frontend "
        "(deploy lag) tenga la misma SSOT."
    )


# ---------------------------------------------------------------------------
# 2. SQL: subquery NOT EXISTS con sibling completed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", [
    "/api/plans/history-list",
    "/api/plans/history-status-summary",
])
def test_sql_contains_not_exists_sibling(endpoint):
    """Ambos endpoints deben incluir la subquery `NOT EXISTS (SELECT 1
    FROM plan_chunk_queue sibling WHERE ... AND sibling.status =
    'completed')`. Sin esta condición, el counter sería idéntico a
    failed_count y el fix no haría nada."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(endpoint)
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    # NOT EXISTS con la subquery que busca sibling completed.
    assert re.search(
        r"NOT\s+EXISTS\s*\(\s*SELECT\s+1\s+FROM\s+plan_chunk_queue\s+sibling",
        norm,
        re.IGNORECASE,
    ), (
        f"SQL en {endpoint} debe contener `NOT EXISTS (SELECT 1 FROM "
        f"plan_chunk_queue sibling ...)`. Got: {norm[:1000]!r}"
    )
    assert "sibling.status = 'completed'" in norm, (
        f"La subquery sibling debe filtrar por status='completed' en "
        f"{endpoint}. Got: {norm[:1000]!r}"
    )


@pytest.mark.parametrize("endpoint", [
    "/api/plans/history-list",
    "/api/plans/history-status-summary",
])
def test_sql_sibling_matches_same_plan_and_week(endpoint):
    """La subquery debe matchear sibling con MISMO meal_plan_id Y
    MISMO week_number. Si solo hace match por plan_id, contaría
    chunks completed de OTRA semana como "reemplazo" — incorrecto.
    Si solo hace match por week_number sin plan, cruzaría planes."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(endpoint)
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "sibling.meal_plan_id = plan_chunk_queue.meal_plan_id" in norm, (
        f"sibling debe matchear meal_plan_id en {endpoint}. Got: {norm[:1000]!r}"
    )
    assert "sibling.week_number = plan_chunk_queue.week_number" in norm, (
        f"sibling debe matchear week_number en {endpoint}. Got: {norm[:1000]!r}"
    )


@pytest.mark.parametrize("endpoint", [
    "/api/plans/history-list",
    "/api/plans/history-status-summary",
])
def test_sql_sibling_excludes_self(endpoint):
    """`sibling.id != plan_chunk_queue.id` — sin esta exclusión, un
    chunk failed con id=X buscaría sibling con id=X status=completed
    (imposible: una sola fila tiene un solo status). Pero defensivo
    contra un futuro refactor que agregue UNION ALL u otro mecanismo
    que pudiera hacer self-match."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(endpoint)
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "sibling.id != plan_chunk_queue.id" in norm, (
        f"sibling debe excluir self ({endpoint}). Got: {norm[:1000]!r}"
    )


# ---------------------------------------------------------------------------
# 3. SELECT principal: COALESCE en history-list
# ---------------------------------------------------------------------------
def test_history_list_coalesce_unreplaced():
    """`COALESCE(qstats.failed_unreplaced_count, 0)` en SELECT
    principal del history-list. Mismo patrón que los otros counters
    (test_p1_audit_hist_4_history_list_join_chunk_queue.py)."""
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
    assert re.search(
        r"COALESCE\s*\(\s*qstats\.failed_unreplaced_count\s*,\s*0\s*\)",
        norm,
    ), (
        f"COALESCE(qstats.failed_unreplaced_count, 0) faltante. "
        f"Got: {norm[:1000]!r}"
    )


# ---------------------------------------------------------------------------
# 4. Response shape: history-list incluye chunk_failed_unreplaced_count
# ---------------------------------------------------------------------------
def test_response_includes_unreplaced_counter():
    """Cada plan del response de /history-list debe tener
    `chunk_failed_unreplaced_count` como int."""
    fake_rows = [
        _base_row(_PLAN_A,
                  chunk_failed_count=2,
                  chunk_failed_unreplaced_count=1,
                  chunk_completed_count=7),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    body = r.json()
    plans = body.get("plans") or []
    assert len(plans) == 1, body
    p = plans[0]
    assert "chunk_failed_unreplaced_count" in p, (
        f"Response shape debe incluir `chunk_failed_unreplaced_count`. "
        f"Got: {sorted(p.keys())}"
    )
    assert p["chunk_failed_unreplaced_count"] == 1
    assert isinstance(p["chunk_failed_unreplaced_count"], int)
    # `chunk_failed_count` se preserva (compat con frontend pre-fix
    # durante deploy lag inverso: backend nuevo + frontend viejo).
    assert p["chunk_failed_count"] == 2


def test_response_unreplaced_defaults_to_zero():
    """Plan sin chunks fallidos → counter = 0 (no NULL ni missing key)."""
    fake_rows = [
        _base_row(_PLAN_A,
                  chunk_failed_count=0,
                  chunk_failed_unreplaced_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    plans = r.json().get("plans") or []
    assert plans[0]["chunk_failed_unreplaced_count"] == 0


def test_response_unreplaced_handles_none_from_sql():
    """Si la fila SQL devuelve None (sin chunks), el endpoint debe
    coercer a 0. Defensivo contra escenarios donde el COALESCE del
    SQL no se aplica (planes pre-rollout sin entradas en
    plan_chunk_queue)."""
    fake_rows = [
        _base_row(_PLAN_A,
                  chunk_failed_count=None,
                  chunk_failed_unreplaced_count=None),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    plans = r.json().get("plans") or []
    assert plans[0]["chunk_failed_unreplaced_count"] == 0
    assert plans[0]["chunk_failed_count"] == 0


# ---------------------------------------------------------------------------
# 5. Response shape: history-status-summary incluye failed_unreplaced_count
# ---------------------------------------------------------------------------
def test_status_summary_response_includes_unreplaced():
    """Cada entry del summary debe tener `failed_unreplaced_count` como
    int. El frontend lo usa como fallback legacy si los counters
    embebidos del listado no están (deploy lag)."""
    fake_rows = [
        {
            "pid": _PLAN_A,
            "pending_user_action_count": 0,
            "failed_count": 3,
            "failed_unreplaced_count": 0,  # 3 failed con sibling completed
            "in_flight_count": 0,
            "completed_count": 7,
            "total": 10,
            "tier_breakdown": None,
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
    summary = body.get("summary") or {}
    entry = summary.get(_PLAN_A) or {}
    assert "failed_unreplaced_count" in entry, (
        f"Entry shape debe incluir `failed_unreplaced_count`. "
        f"Got: {sorted(entry.keys())}"
    )
    assert entry["failed_unreplaced_count"] == 0
    assert isinstance(entry["failed_unreplaced_count"], int)
    # `failed_count` legacy se preserva.
    assert entry["failed_count"] == 3


# ---------------------------------------------------------------------------
# 6. Semántica del counter: el fix se manifiesta en el counter
# ---------------------------------------------------------------------------
def test_unreplaced_distinct_from_failed_count():
    """Caso del bug: failed_count > 0 pero failed_unreplaced_count = 0.
    El frontend en este caso debe NO elevar a action_required (es la
    razón del fix). El test fija el contrato del payload."""
    fake_rows = [
        _base_row(_PLAN_A,
                  generation_status="complete",
                  days_generated=7,
                  total_days_requested=7,
                  chunk_failed_count=2,         # 2 chunks failed
                  chunk_failed_unreplaced_count=0,  # ambos con sibling completed
                  chunk_completed_count=7,
                  chunk_in_flight_count=0,
                  chunk_pending_user_action_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    p = r.json()["plans"][0]
    # Asimetría intencional — esa es la señal que el frontend usa para
    # NO elevar bucket a `action_required`.
    assert p["chunk_failed_count"] == 2
    assert p["chunk_failed_unreplaced_count"] == 0
    # Sanity: si el frontend lee `chunk_failed_count` legacy elevaría
    # bucket; si lee `chunk_failed_unreplaced_count` (post-fix) no.
    # El test del frontend (History.p0_failed_replaced.test.js) valida
    # el comportamiento client-side. Aquí solo fijamos el payload.


def test_unreplaced_equal_to_failed_when_no_replacement():
    """Caso normal: chunk failed sin sibling completed (los días NO
    se generaron por otra fila). El counter unreplaced == failed_count
    y el frontend debe seguir elevando bucket a `action_required`."""
    fake_rows = [
        _base_row(_PLAN_A,
                  generation_status="partial",
                  days_generated=4,
                  total_days_requested=7,
                  chunk_failed_count=1,
                  chunk_failed_unreplaced_count=1,
                  chunk_completed_count=4,
                  chunk_in_flight_count=0,
                  chunk_pending_user_action_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    p = r.json()["plans"][0]
    assert p["chunk_failed_count"] == p["chunk_failed_unreplaced_count"] == 1
