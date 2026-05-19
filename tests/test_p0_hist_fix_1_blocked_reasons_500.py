"""[P0-HIST-FIX-1 · 2026-05-09] Test de regresión para el
500 del endpoint `/blocked_reasons` cuando `include_stuck=true`.

Bug original (reportado en producción 2026-05-09 con plan
98d902e3-56f0-4f54-a4f6-cb454b23d4de):
    El frontend del modal del Historial llama a
    `/blocked_reasons?include_failed=true&include_stuck=true` para
    lazy-fetch reasons cuando hay drift. La query construía:

        execute_after < NOW() - make_interval(hours => %s)

    con `_stuck_lag_hours = 3.0` (knob default `MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS`,
    tipo float). PostgreSQL `make_interval(hours => ...)` SOLO acepta
    int — pasarle un numeric/float dispara:

        ERROR: 42883: function make_interval(hours => numeric) does not exist

    Resultado: HTTP 500 al frontend. Modal del Historial mostraba
    error en console pero el banner stuck nunca aparecía → señales
    diagnósticas (chunks atascados >3h) invisibles al usuario.

Fix:
    Reemplazar `make_interval(hours => %s)` por
    `(interval '1 hour' * %s)`. La multiplicación de interval acepta
    numeric sin cast Y preserva precisión sub-hora del knob (3.5h →
    3h 30min, que make_interval habría truncado a 3h).

Cobertura:
    1. Anchor del marker en el endpoint.
    2. SQL del filter stuck NO usa `make_interval` (anti-pattern).
    3. SQL del filter stuck usa `interval '1 hour' * %s`.
    4. Smoke test end-to-end con include_stuck=true (status 200).
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

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
def test_marker_present_in_endpoint():
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert "P0-HIST-FIX-1" in src, (
        "Endpoint /blocked_reasons debe citar `P0-HIST-FIX-1` "
        "para que un grep + git blame lleve directo al fix del cast."
    )


# ---------------------------------------------------------------------------
# 2. SQL: anti-pattern make_interval con numeric eliminado
# ---------------------------------------------------------------------------
def test_sql_stuck_filter_does_not_use_make_interval():
    """El anti-pattern `make_interval(hours => %s)` con numeric debe
    NO aparecer en el SQL ejecutado. Si vuelve, el endpoint regresa al
    500 con 42883.

    Filtramos comentarios (líneas que arrancan con `#`) — el comentario
    explicativo del bug menciona el anti-pattern por contexto pero NO
    debe contar como ocurrencia real."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    # Strip lines that are pure comments (start with optional whitespace
    # then `#`).
    code_only = "\n".join(
        line for line in src.splitlines()
        if not re.match(r"^\s*#", line)
    )
    assert "make_interval(hours =>" not in code_only, (
        "El anti-pattern `make_interval(hours => %s)` con numeric "
        "vuelve a romper el endpoint con `42883: function "
        "make_interval(hours => numeric) does not exist`. Usa "
        "`interval '1 hour' * %s` que sí acepta numeric."
    )


def test_sql_stuck_filter_uses_interval_multiplication():
    """El filter stuck debe usar `interval '1 hour' * %s` —
    multiplicación de interval acepta numeric directamente y preserva
    precisión sub-hora."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(
        r"interval\s+'1\s+hour'\s*\*\s*%s",
        norm,
        re.IGNORECASE,
    ), (
        "Filter stuck debe usar `interval '1 hour' * %s` (acepta "
        "numeric sin cast). Got: {!r}".format(norm[:2000])
    )


# ---------------------------------------------------------------------------
# 3. Smoke test end-to-end
# ---------------------------------------------------------------------------
def test_endpoint_returns_200_with_include_stuck():
    """End-to-end: con include_stuck=true y un knob float (default
    3.0), el endpoint debe responder 200, no 500. Mock retorna lista
    vacía — la validación es que el SQL ejecute sin throw."""
    captured_queries = []

    def _fake(query, params=None, **kwargs):
        captured_queries.append(query)
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM user_profiles" in query:
            return {"logging_preference": "manual"}
        if "FROM plan_chunk_queue" in query:
            return []
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(
            f"/api/plans/{_PLAN_A}/blocked_reasons"
            f"?include_failed=true&include_stuck=true"
        )

    assert r.status_code == 200, (
        f"Esperaba 200 con include_stuck=true; got {r.status_code} "
        f"(body={r.text[:500]!r}). El SQL del filter stuck debe usar "
        f"`interval '1 hour' * %s` (acepta numeric float)."
    )
    body = r.json()
    assert "blocked" in body
    assert body["reasons"] == []


def test_endpoint_query_string_has_interval_multiplication():
    """End-to-end: confirmar que la query construida realmente usa la
    multiplicación de interval — no solo en el código fuente, sino
    en el SQL string que llega al driver."""
    captured = {}

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"user_id": _USER_A}
        if "FROM user_profiles" in query:
            return {"logging_preference": "manual"}
        if "FROM plan_chunk_queue" in query:
            captured["query"] = query
            return []
        return None

    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(
            f"/api/plans/{_PLAN_A}/blocked_reasons"
            f"?include_failed=true&include_stuck=true"
        )

    assert r.status_code == 200
    norm = re.sub(r"\s+", " ", captured.get("query") or "")
    assert "interval '1 hour'" in norm, (
        f"SQL ejecutado debe usar interval multiplication. "
        f"Got: {norm[:1000]!r}"
    )
    assert "make_interval(hours =>" not in norm, (
        "SQL ejecutado NO debe contener el anti-pattern "
        "`make_interval(hours => ...)` con numeric."
    )
