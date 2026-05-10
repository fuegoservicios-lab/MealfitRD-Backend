"""[P1-HIST-3 · 2026-05-09] Tests del endpoint
``GET /api/plans/lessons-counts`` para el chip de lecciones en el
Historial.

Bug original (audit historial 2026-05-08):
    El historial no exponía las lecciones acumuladas del aprendizaje
    continuo (`chunk_lesson_telemetry`). El diferenciador del producto
    (MealfitRD "aprende entre chunks") era invisible al usuario en su
    biblioteca.

Fix:
    Endpoint single-roundtrip que devuelve `{plan_id: count}` para
    todos los planes del usuario. El frontend cachea el resultado en
    state local y renderiza un chip "X lecciones" cuando count > 0.

Cobertura:
    - 401 sin auth.
    - 200 success: response shape `{counts: {...}}` correcto.
    - SQL query usa `WHERE user_id = %s` (defense-in-depth además de RLS).
    - Excluye `meal_plan_id IS NULL` (orphans post-P0-HIST-3 SET NULL).
    - GROUP BY meal_plan_id (no cuenta global).
    - Anchor [P1-HIST-3 · 2026-05-09] presente en endpoint.
"""
import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_USER_A = "11111111-1111-1111-1111-111111111111"


# ---------------------------------------------------------------------------
# 1. Auth
# ---------------------------------------------------------------------------
def test_lessons_counts_requires_auth():
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    r = client.get("/api/plans/lessons-counts")
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# 2. Success path
# ---------------------------------------------------------------------------
def test_lessons_counts_returns_dict_shape():
    """Response shape: `{counts: {plan_id_str: int}}`."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    fake_rows = [
        {"pid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "cnt": 12},
        {"pid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "cnt": 3},
    ]
    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/lessons-counts")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "counts" in body
    assert isinstance(body["counts"], dict)
    assert body["counts"]["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] == 12
    assert body["counts"]["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] == 3


def test_lessons_counts_empty_when_no_telemetry():
    """Usuario sin lecciones registradas → `{counts: {}}` (no 404)."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=[]):
        r = client.get("/api/plans/lessons-counts")
    assert r.status_code == 200
    assert r.json() == {"counts": {}}


def test_lessons_counts_filters_by_user_and_excludes_orphans():
    """SQL debe filtrar por user_id (defense-in-depth) y excluir
    meal_plan_id IS NULL (orphans post-P0-HIST-3 SET NULL)."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    captured = {}
    def _spy(sql, params=None, **kw):
        captured["sql"] = sql
        captured["params"] = params
        return []

    with patch("db_core.execute_sql_query", side_effect=_spy):
        r = client.get("/api/plans/lessons-counts")
    assert r.status_code == 200

    sql = captured["sql"]
    assert "WHERE user_id = %s" in sql
    # Importante: orphans (meal_plan_id NULL) NO deben contar — la
    # FK SET NULL de P0-HIST-3 deja rows con plan eliminado.
    assert "meal_plan_id IS NOT NULL" in sql
    assert "GROUP BY meal_plan_id" in sql
    assert "FROM chunk_lesson_telemetry" in sql
    assert _USER_A in captured["params"]


def test_lessons_counts_skips_rows_without_pid():
    """Defensivo: si una fila viene con pid=None (corner case del cast),
    el handler la ignora en lugar de crashear."""
    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    fake_rows = [
        {"pid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "cnt": 5},
        {"pid": None, "cnt": 99},  # debería ignorarse
        {"pid": "", "cnt": 1},     # idem
    ]
    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/lessons-counts")
    assert r.status_code == 200
    counts = r.json()["counts"]
    assert "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" in counts
    assert None not in counts
    assert "" not in counts


# ---------------------------------------------------------------------------
# 3. Anchor / drift detection
# ---------------------------------------------------------------------------
def test_p1_hist_3_anchor_in_endpoint():
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    assert "P1-HIST-3" in src
    assert "chunk_lesson_telemetry" in src
    assert "GROUP BY meal_plan_id" in src
    # Excluir orphans es contrato — un futuro contributor que lo elimine
    # romperá el conteo (rows con meal_plan_id NULL no pertenecen a
    # ningún plan visible).
    assert "meal_plan_id IS NOT NULL" in src
