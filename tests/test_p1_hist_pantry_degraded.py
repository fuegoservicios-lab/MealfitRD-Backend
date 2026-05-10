"""[P1-HIST-PANTRY-DEGRADED · 2026-05-09] Tests del surface
retroactivo de `learning_metrics.pantry_degraded_reason` en el
listado del Historial.

Bug original (audit Historial 2026-05-09 · gap P1-5):
    El productor `cron_tasks.py:19631` escribe
    `learning_metrics["pantry_degraded_reason"] =
    form_data["_pantry_degraded_reason"]` cuando el pipeline detectó
    pantry comprometida al pickup del chunk (stale_snapshot,
    empty_pantry_proxy, inventory_unreachable). Esa señal queda en
    el jsonb del chunk pero la card del Historial NO surface al
    usuario — un plan generado con pantry degradada se ve idéntico
    a un plan healthy hasta que el usuario abre el modal y va al
    tab Métricas (post P1-HIST-LM-WHITELIST). Para el listado, el
    diferenciador queda invisible.

Fix:
    Extender `/history-list` con dos campos agregados:
      - `chunk_pantry_degraded_count`: COUNT chunks con la key.
      - `chunk_pantry_degraded_reasons`: array DISTINCT de reasons
        (alimenta el tooltip).
    Frontend renderiza chip "Pantry degradada" ámbar en cardActions
    cuando count > 0, con tooltip que lista las reasons.

Cobertura:
    - Anchor del marker.
    - SQL: subquery `qstats` extiende COUNT FILTER + array_agg
      DISTINCT con guards `learning_metrics ? 'pantry_degraded_reason'`
      (jsonb operator existence) + IS NOT NULL + <> ''.
    - Response shape: `chunk_pantry_degraded_count` (int) y
      `chunk_pantry_degraded_reasons` (list[str] | None).
    - Sanitización defensiva: array_agg puede devolver None o
      `[None]` si todos los chunks son healthy — ambos coercen a None.
    - Docstring del endpoint declara los nuevos campos.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "55555555-5555-5555-5555-555555555555"


def _client():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor + docstring
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P1-HIST-PANTRY-DEGRADED · 2026-05-09" in src


def test_docstring_declares_new_fields():
    """Returns shape del docstring debe declarar los 2 campos
    nuevos para que cualquier consumidor (admin, monitoring) los
    descubra sin leer la SQL."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "chunk_pantry_degraded_count" in src
    assert "chunk_pantry_degraded_reasons" in src


# ---------------------------------------------------------------------------
# 2. SQL: subquery qstats incluye los nuevos agregados
# ---------------------------------------------------------------------------
def test_sql_count_with_jsonb_existence_check():
    """SQL debe usar `learning_metrics ? 'pantry_degraded_reason'`
    (jsonb existence operator) ANTES de `->>'..' IS NOT NULL`. El
    operator `?` es más eficiente que el doble null-check porque
    Postgres usa el GIN index de jsonb si existe."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    # Existence operator + null check + non-empty string check (3
    # guards anidados — pantry_degraded_reason puede llegar como
    # NULL o "" si el productor escribe un default falsy).
    assert "learning_metrics ? 'pantry_degraded_reason'" in src
    assert "learning_metrics->>'pantry_degraded_reason' IS NOT NULL" in src
    assert "learning_metrics->>'pantry_degraded_reason' <> ''" in src


def test_sql_array_agg_distinct():
    """`array_agg(DISTINCT ...)` para que un plan con 5 chunks
    degraded por la misma razón muestre la reason una sola vez en
    el tooltip — no `["stale_snapshot", "stale_snapshot", ...]`.
    Mismo FILTER que el COUNT para no incluir chunks healthy."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert re.search(
        r"array_agg\s*\(\s*DISTINCT\s+learning_metrics->>'pantry_degraded_reason'\s*\)",
        src,
    )


def test_sql_filter_excludes_healthy_chunks():
    """El FILTER del array_agg debe excluir chunks sin la key
    (sino el array tendría NULLs y rompe el tooltip render)."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    # Pattern: array_agg(...) FILTER (WHERE ...). Aceptamos múltiples
    # líneas entre array_agg y FILTER porque el SQL está formateado
    # multi-line.
    assert re.search(
        r"array_agg\([\s\S]{0,300}?\)\s+FILTER\s*\(\s*WHERE",
        src,
    )


def test_sql_response_aliases():
    """Aliases del SELECT outer deben matchear los keys del response
    dict para que el row.get('...') resuelva."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "AS chunk_pantry_degraded_count" in src
    assert "AS chunk_pantry_degraded_reasons" in src


# ---------------------------------------------------------------------------
# 3. Response shape — count int, reasons list[str] o None
# ---------------------------------------------------------------------------
def _row_template(**overrides):
    """Mock row con TODOS los campos esperados por el endpoint.
    Override solo los que el test quiera flex."""
    base = {
        "id": "plan-aaa",
        "name": "Plan test",
        "created_at": None,
        "calories": 2000,
        "macros": {"protein": "120g"},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "coherence_history": [],
        "preview_meals_raw": [],
        "goal_root": "lose_weight",
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 7,
        "chunk_tier_breakdown": None,
        "chunk_pantry_degraded_count": 0,
        "chunk_pantry_degraded_reasons": None,
    }
    base.update(overrides)
    return base


def test_response_includes_pantry_degraded_fields_zero():
    """Plan healthy: count = 0, reasons = None. Frontend NO
    renderiza el chip."""
    client = _client()
    rows = [_row_template()]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    body = r.json()
    plan = body["plans"][0]
    assert plan["chunk_pantry_degraded_count"] == 0
    assert plan["chunk_pantry_degraded_reasons"] is None


def test_response_includes_pantry_degraded_fields_with_data():
    """Plan con 2 chunks degraded: count=2, reasons=lista DISTINCT."""
    client = _client()
    rows = [
        _row_template(
            chunk_pantry_degraded_count=2,
            chunk_pantry_degraded_reasons=["stale_snapshot", "empty_pantry_proxy"],
        ),
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    plan = body["plans"][0]
    assert plan["chunk_pantry_degraded_count"] == 2
    assert plan["chunk_pantry_degraded_reasons"] == [
        "stale_snapshot",
        "empty_pantry_proxy",
    ]


def test_response_sanitizes_reasons_with_nulls_in_array():
    """Edge: array_agg puede devolver None o lista con `None`/strings
    vacíos si el FILTER no atrapa todo. Sanitizamos a lista de
    strings no vacíos; si todo se filtra, devolvemos None."""
    client = _client()
    rows = [
        _row_template(
            chunk_pantry_degraded_count=1,
            chunk_pantry_degraded_reasons=[None, "stale_snapshot", "", None],
        ),
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    plan = body["plans"][0]
    # Solo string truthy sobreviven.
    assert plan["chunk_pantry_degraded_reasons"] == ["stale_snapshot"]


def test_response_reasons_none_when_array_empty():
    """Lista vacía (no None, pero list()) → coercer a None para que
    el frontend distinga "sin info" de "lista con 0 reasons"."""
    client = _client()
    rows = [
        _row_template(
            chunk_pantry_degraded_count=0,
            chunk_pantry_degraded_reasons=[],
        ),
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    plan = body["plans"][0]
    assert plan["chunk_pantry_degraded_reasons"] is None


def test_response_reasons_none_when_count_falsy_in_db():
    """count viene como None del DB (no debería pasar por COALESCE 0,
    pero defense-in-depth) → response devuelve 0."""
    client = _client()
    rows = [_row_template(chunk_pantry_degraded_count=None)]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    plan = body["plans"][0]
    assert plan["chunk_pantry_degraded_count"] == 0
