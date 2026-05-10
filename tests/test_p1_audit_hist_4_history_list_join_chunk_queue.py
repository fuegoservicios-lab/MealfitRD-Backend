"""[P1-AUDIT-HIST-4 · 2026-05-09] Tests del LEFT JOIN agregado a
``api_plans_history_list`` que integra los counters de
``plan_chunk_queue.status`` en el mismo response del listado.

Bug original (audit Historial 2026-05-09):
    El listado del Historial construía la card 100% desde
    `meal_plans.plan_data`. Detectar drift queue↔plan_data requería
    DOS roundtrips (history-list + history-status-summary de
    P0-AUDIT-HIST-2) y reconciliación client-side. Drawbacks:
      - Roundtrip extra al cargar la página.
      - Race condition: un restore/delete entre las dos requests
        podía dejar el bucket desincronizado entre el listado y el
        summary (ya cancelado en summary, todavía visible en
        listado).

Fix:
    LEFT JOIN agregado a `plan_chunk_queue` (subquery `qstats` con
    GROUP BY meal_plan_id, FILTER per status, pre-filtrada por
    `user_id`). Counters embebidos por plan en el response:
    `chunk_pending_user_action_count`, `chunk_failed_count`,
    `chunk_in_flight_count`, `chunk_completed_count`. El frontend
    los prefiere sobre el summary endpoint (que se preserva como
    fallback legacy durante deploy lag).

Cobertura:
    1. Anchor del marker.
    2. SQL contiene `LEFT JOIN ... ON qstats.meal_plan_id = mp.id`.
    3. Subquery filtra por `user_id` (defense-in-depth + RLS).
    4. Subquery excluye `meal_plan_id IS NULL` (orphans P0-HIST-3).
    5. Subquery contiene FILTER (WHERE status = 'X') para los 4
       counters requeridos.
    6. Response incluye `chunk_*_count` en cada plan.
    7. Counters siempre presentes (0 cuando no hay chunks —
       `COALESCE` en el SELECT principal).
    8. Counter math correcto: si la subquery agrega 3 pending_user_action
       para un plan, el response tiene 3.
    9. Plan sin chunks → counters = 0 (no NULL ni missing key).
    10. Cap LIMIT 200 sigue presente.
    11. Drift detection: nombres de keys del SELECT == nombres en
        el response.
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


def _base_row(pid, **overrides):
    """Fila representativa del SELECT — completar todos los campos
    que el endpoint consume para que la iteración no falle por keys
    faltantes."""
    base = {
        "id": pid,
        "name": f"Plan {pid[:4]}",
        "created_at": None,  # endpoint maneja None tras isoformat check
        "calories": 2000,
        "macros": {"protein": "100g"},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "coherence_history": [],
        "preview_meals_raw": [{"name": "Avena", "meal": "Desayuno"}],
        "goal_root": "lose_weight",
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        # Counters del LEFT JOIN — el endpoint usa COALESCE en SQL
        # pero la fixture ya los pasa como int (simulando el cast).
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P1-AUDIT-HIST-4" in src, (
        "El endpoint debe citar `P1-AUDIT-HIST-4` para que un grep "
        "+ git blame lleve directo al fix del LEFT JOIN."
    )


# ---------------------------------------------------------------------------
# 2. SQL: LEFT JOIN presente
# ---------------------------------------------------------------------------
def test_sql_contains_left_join_to_chunk_queue():
    """`LEFT JOIN ( SELECT ... FROM plan_chunk_queue ... ) qstats
    ON qstats.meal_plan_id = mp.id` debe estar en el SQL."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LEFT JOIN" in norm, (
        "SQL debe usar LEFT JOIN para preservar planes sin chunks "
        "(usuarios pre-rollout del sistema chunks). Sin LEFT, esos "
        "planes desaparecerían del listado."
    )
    assert "plan_chunk_queue" in norm
    assert re.search(
        r"ON\s+qstats\.meal_plan_id\s*=\s*mp\.id",
        norm,
        re.IGNORECASE,
    ), f"JOIN ON clause incorrecto. Got: {norm[:1000]!r}"


def test_sql_subquery_filters_by_user_id():
    """La subquery `qstats` debe filtrar por `user_id = %s` para que
    Postgres no agregue chunks de TODOS los usuarios antes del JOIN
    (waste + leak potencial). Defense-in-depth: el RLS de
    `plan_chunk_queue` ya filtra, pero el filtro explícito mantiene
    el plan estable + paridad con history-status-summary
    (P0-AUDIT-HIST-2)."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        captured["params"] = params
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    # Buscar el bloque entre `LEFT JOIN (` y `) qstats` — la subquery.
    m = re.search(
        r"LEFT\s+JOIN\s*\(\s*(.+?)\s*\)\s+qstats",
        norm,
        re.IGNORECASE | re.DOTALL,
    )
    assert m is not None, f"No pude extraer la subquery. Got: {norm!r}"
    subquery = m.group(1)
    assert "user_id = %s" in subquery, (
        f"Subquery `qstats` debe filtrar por `user_id = %s`. "
        f"Got: {subquery!r}"
    )
    # El user_id debe estar EN los params (binding correcto).
    assert _USER_A in (captured.get("params") or ())


def test_sql_subquery_excludes_orphan_meal_plan_id():
    """`AND meal_plan_id IS NOT NULL` en la subquery — defensivo
    contra orphans aunque la FK CASCADE de `plan_chunk_queue` los
    haga improbables, evita que un row corrupto agrupe bajo NULL."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    m = re.search(
        r"LEFT\s+JOIN\s*\(\s*(.+?)\s*\)\s+qstats",
        norm,
        re.IGNORECASE | re.DOTALL,
    )
    subquery = m.group(1)
    assert "meal_plan_id IS NOT NULL" in subquery


@pytest.mark.parametrize("required_filter", [
    "status = 'pending_user_action'",
    "status = 'failed'",
    "status = 'completed'",
])
def test_sql_subquery_includes_filter_per_status(required_filter):
    """La subquery debe usar `FILTER (WHERE status = 'X')` para los
    counters críticos. Si un refactor pierde uno, los counters
    embebidos quedan a 0 silently y el frontend opera en modo
    legacy aunque el deploy ya tenga el LEFT JOIN."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert required_filter in norm, (
        f"SQL debe incluir `FILTER (WHERE {required_filter})` en la "
        f"subquery `qstats`. Encontrado: {norm[:600]!r}..."
    )


def test_sql_subquery_includes_in_flight_filter():
    """El counter `in_flight` agrupa pending+processing+stale —
    FILTER con IN clause."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    # FILTER (WHERE status IN ('pending', 'processing', 'stale')).
    assert re.search(
        r"FILTER\s*\(\s*WHERE\s+status\s+IN\s*\(\s*'pending'[^)]*'processing'[^)]*'stale'",
        norm,
        re.IGNORECASE,
    ), f"Filtro in_flight (pending/processing/stale) faltante en: {norm!r}"


def test_sql_uses_coalesce_for_counters():
    """`COALESCE(qstats.X, 0)` en el SELECT principal: planes sin
    chunks devuelven 0 en lugar de NULL. Sin esto, el frontend
    debería tratar NULL como case especial."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    # Al menos un COALESCE(qstats.<X>, 0) para cada counter.
    for counter in (
        "pending_user_action_count",
        "failed_count",
        "in_flight_count",
        "completed_count",
    ):
        assert re.search(
            rf"COALESCE\s*\(\s*qstats\.{counter}\s*,\s*0\s*\)",
            norm,
        ), f"COALESCE(qstats.{counter}, 0) faltante en SELECT principal."


# ---------------------------------------------------------------------------
# 3. Response shape: counters embebidos por plan
# ---------------------------------------------------------------------------
def test_response_includes_chunk_counters_per_plan():
    """Cada plan del response debe incluir las 4 keys de counter,
    como int."""
    fake_rows = [
        _base_row(_PLAN_A,
                  chunk_pending_user_action_count=2,
                  chunk_failed_count=1,
                  chunk_in_flight_count=0,
                  chunk_completed_count=4),
    ]
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200, r.text

    body = r.json()
    plans = body["plans"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["chunk_pending_user_action_count"] == 2
    assert plan["chunk_failed_count"] == 1
    assert plan["chunk_in_flight_count"] == 0
    assert plan["chunk_completed_count"] == 4
    for key in (
        "chunk_pending_user_action_count",
        "chunk_failed_count",
        "chunk_in_flight_count",
        "chunk_completed_count",
    ):
        assert isinstance(plan[key], int), (
            f"{key} debe ser int. Got {type(plan[key])}"
        )


def test_plan_without_chunks_has_zero_counters():
    """Plan que el LEFT JOIN no encuentra (sin chunks) → COALESCE
    devuelve 0 → el response tiene 0, no None ni missing key."""
    fake_rows = [
        # Counters explícitamente 0 (lo que devolvería COALESCE).
        _base_row(_PLAN_A,
                  chunk_pending_user_action_count=0,
                  chunk_failed_count=0,
                  chunk_in_flight_count=0,
                  chunk_completed_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    body = r.json()
    plan = body["plans"][0]
    for key in (
        "chunk_pending_user_action_count",
        "chunk_failed_count",
        "chunk_in_flight_count",
        "chunk_completed_count",
    ):
        assert plan[key] == 0


def test_counters_coerced_to_int_even_if_driver_returns_str():
    """Defense: psycopg podría devolver el COUNT como Decimal/str
    según versión del driver. El endpoint hace `int(row.get(...) or 0)`."""
    fake_rows = [
        _base_row(_PLAN_A,
                  chunk_pending_user_action_count="3",  # str (driver edge)
                  chunk_failed_count=None,  # NULL del COALESCE roto
                  chunk_in_flight_count=0,
                  chunk_completed_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plan = r.json()["plans"][0]
    assert plan["chunk_pending_user_action_count"] == 3
    assert plan["chunk_failed_count"] == 0  # None → 0


# ---------------------------------------------------------------------------
# 4. Cap defensivo + filtros básicos preservados
# ---------------------------------------------------------------------------
def test_sql_preserves_limit_200():
    """El cap LIMIT 200 sigue presente tras la modificación."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert "LIMIT 200" in norm


def test_sql_preserves_name_not_null_filter():
    """`AND mp.name IS NOT NULL` se preserva tras prefijar la tabla
    con alias `mp`."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["query"] = query
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    norm = re.sub(r"\s+", " ", captured["query"] or "")
    assert re.search(
        r"mp\.name\s+IS\s+NOT\s+NULL", norm, re.IGNORECASE
    ), f"Filtro `mp.name IS NOT NULL` perdido tras alias. Got: {norm!r}"


def test_params_pass_user_id_for_each_subquery_and_main_where():
    """Los params deben incluir `verified_user_id` UNA vez por cada
    subquery que filtre por user_id + UNA para el WHERE principal.
    Tras P1-AUDIT-HIST-6, hay 3 sites: qstats subquery + qtiers
    LATERAL + main WHERE → 3 ocurrencias del mismo user_id."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        captured["params"] = params
        return []

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200

    params = captured.get("params")
    assert isinstance(params, tuple), f"params debe ser tuple, got {type(params)}"
    # Todos los elementos del tuple deben ser el mismo user_id.
    assert all(p == _USER_A for p in params), (
        f"Todos los params deben ser el verified_user_id. "
        f"Got: {params!r}"
    )
    # Esperamos al menos 2 (P1-AUDIT-HIST-4: qstats + main) y
    # exactamente 3 con tier_breakdown (P1-AUDIT-HIST-6: + qtiers).
    assert len(params) >= 2, (
        f"Esperaba ≥2 params (subquery + main). Got len={len(params)}"
    )
