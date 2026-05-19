"""[P2-HIST-NEW-1 · 2026-05-09] Tests del `primary_action_reason` en
el endpoint `/history-list`.

Bug original (audit profundo Historial 2026-05-09):
    El chip "Acción" de la card del Historial era genérico — el
    reason real estaba en `_blocked_reasons` (lazy-fetched al abrir
    el modal). Inconsistencia con el Dashboard del plan ACTIVO que
    desde P0-DASH-CHIP-HONESTY muestra "Pausado: empty_pantry" en el
    slot. Mismo plan, dos surfaces con diferente nivel de detalle.

Fix:
    LATERAL `qaction` en /history-list extrae el reason del chunk
    bloqueante más temprano (week_number ASC) usando la priority
    chain canónica de /blocked_reasons (dead_letter_reason →
    _pause_reason → _pantry_pause_reason → reason). Frontend usa
    esa key para promover "Acción" a "Acción: empty_pantry".

Cobertura backend:
    1. Anchor del marker en endpoint Y en History.jsx.
    2. SQL incluye LATERAL FROM plan_chunk_queue qa.
    3. Filter: status = 'pending_user_action' OR (failed AND
       dead_letter_reason IS NOT NULL).
    4. Priority chain COALESCE en orden correcto.
    5. ORDER BY week_number ASC NULLS LAST + created_at ASC.
    6. LIMIT 1 (un solo reason por plan).
    7. user_id filter en LATERAL (defense-in-depth).
    8. Response shape: `primary_action_reason: str|None`.
    9. Sanitización: empty/whitespace → None.
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


def _base_row(pid, **overrides):
    base = {
        "id": pid,
        "name": f"Plan {pid[:4]}",
        "created_at": None,
        "calories": 2000,
        "macros": {"protein": "100g"},
        "plan_modified_at": None,
        "generation_status": "partial",
        "total_days_requested": 7,
        "days_generated": 4,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "shift_days_accumulated": None,
        "coherence_history": [],
        "preview_meals_raw": [{"name": "Avena", "meal": "Desayuno"}],
        "goal_root": None,
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 1,
        "chunk_failed_count": 0,
        "chunk_failed_unreplaced_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 4,
        "chunk_pantry_degraded_count": 0,
        "chunk_pantry_degraded_reasons": None,
        "chunk_tier_breakdown": None,
        "primary_action_reason_code": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P2-HIST-NEW-1" in src, (
        "Endpoint history-list debe citar `P2-HIST-NEW-1` para que un "
        "grep + git blame lleve directo al fix del LATERAL qaction."
    )


def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-1" in text


# ---------------------------------------------------------------------------
# 2. SQL: LATERAL qaction
# ---------------------------------------------------------------------------
def test_sql_includes_lateral_qaction():
    """LEFT JOIN LATERAL específico para extraer el reason — no JOIN
    directo con GROUP BY porque queremos UNA sola fila por plan."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\([\s\S]*?\)\s+qaction\s+ON\s+TRUE",
        norm,
        re.IGNORECASE,
    ), (
        "SQL debe agregar el LEFT JOIN LATERAL `qaction ON TRUE` para "
        "que planes sin chunks bloqueados queden con reason_code=NULL."
    )


def test_sql_qaction_filter_includes_pending_action_and_failed():
    """El filter del LATERAL debe incluir AMBOS estados:
       - pending_user_action (chunks pausados por gates)
       - failed con dead_letter_reason (chunks dead-lettered).
    Sin la guard `dead_letter_reason IS NOT NULL`, traeríamos failed
    transitorios (recoverable) que el cron va a reprocesar — esos no
    merecen reason en la card."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    # Buscar el bloque del LATERAL qaction.
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(([\s\S]+?)\)\s+qaction\s+ON\s+TRUE",
        src,
        re.IGNORECASE,
    )
    assert m is not None, "No pude extraer el bloque LATERAL qaction."
    body = m.group(1)
    assert "qa.status = 'pending_user_action'" in body, (
        f"Filter debe incluir status='pending_user_action'. Got: {body[:800]!r}"
    )
    assert re.search(
        r"qa\.status\s*=\s*'failed'\s+AND\s+qa\.dead_letter_reason\s+IS\s+NOT\s+NULL",
        body,
    ), (
        "Filter debe incluir failed con dead_letter_reason IS NOT NULL."
    )


def test_sql_qaction_priority_chain():
    """El COALESCE del LATERAL debe seguir la priority chain canónica:
    dead_letter_reason → _pause_reason → _pantry_pause_reason → reason.
    Mismo orden que /blocked_reasons (plans.py:3823+)."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(([\s\S]+?)\)\s+qaction\s+ON\s+TRUE",
        src,
        re.IGNORECASE,
    )
    body = m.group(1)
    # Encontrar el COALESCE en orden esperado.
    coalesce_match = re.search(
        r"COALESCE\(\s*qa\.dead_letter_reason,\s*qa\.pipeline_snapshot->>'_pause_reason',\s*qa\.pipeline_snapshot->>'_pantry_pause_reason',\s*qa\.pipeline_snapshot->>'reason'\s*\)",
        body,
    )
    assert coalesce_match is not None, (
        f"COALESCE no sigue la priority chain canónica. Got: {body[:1500]!r}"
    )


def test_sql_qaction_orders_by_week_then_created():
    """ORDER BY week_number ASC NULLS LAST + created_at ASC para que
    el reason del chunk MÁS TEMPRANO bloqueante domine (los demás se
    desbloquean en cascada típicamente). LIMIT 1."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(([\s\S]+?)\)\s+qaction\s+ON\s+TRUE",
        src,
        re.IGNORECASE,
    )
    body = m.group(1)
    assert re.search(
        r"ORDER\s+BY\s+qa\.week_number\s+ASC\s+NULLS\s+LAST",
        body,
        re.IGNORECASE,
    )
    assert "LIMIT 1" in body


def test_sql_qaction_filters_by_user_id():
    """Defense-in-depth: el LATERAL debe filtrar también por user_id
    (RLS también aplica en plan_chunk_queue, pero el filtro explícito
    es la convención del repo)."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(([\s\S]+?)\)\s+qaction\s+ON\s+TRUE",
        src,
        re.IGNORECASE,
    )
    body = m.group(1)
    assert "qa.user_id = %s" in body, (
        f"LATERAL debe filtrar por qa.user_id. Got: {body[:800]!r}"
    )


# ---------------------------------------------------------------------------
# 3. Response shape
# ---------------------------------------------------------------------------
def test_response_includes_primary_action_reason():
    """Response debe incluir `primary_action_reason: str|None` por plan."""
    fake_rows = [
        _base_row(_PLAN_A, primary_action_reason_code="empty_pantry"),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200, r.text
    body = r.json()
    plans = body.get("plans") or []
    assert len(plans) == 1
    assert "primary_action_reason" in plans[0], (
        f"Response shape debe incluir `primary_action_reason`. "
        f"Got: {sorted(plans[0].keys())}"
    )
    assert plans[0]["primary_action_reason"] == "empty_pantry"


def test_response_primary_action_reason_none_when_no_blocked_chunks():
    """Plan healthy → primary_action_reason=None. El frontend distingue
    None de string vacío con typeof check."""
    fake_rows = [
        _base_row(_PLAN_A,
                  primary_action_reason_code=None,
                  chunk_pending_user_action_count=0,
                  chunk_failed_count=0),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plans = r.json()["plans"]
    assert plans[0]["primary_action_reason"] is None


def test_response_sanitizes_empty_reason_to_none():
    """Si el SELECT devuelve string vacío o whitespace (caso edge donde
    pipeline_snapshot tiene `_pause_reason: ""`), normalizamos a None
    para no propagar valores vacíos al chip frontend."""
    fake_rows = [
        _base_row(_PLAN_A, primary_action_reason_code="   "),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plans = r.json()["plans"]
    assert plans[0]["primary_action_reason"] is None


def test_response_passes_through_dead_letter_reason():
    """End-to-end con dead_letter_reason — el LATERAL debe priorizar
    sobre _pause_reason cuando ambos están en la priority chain."""
    fake_rows = [
        _base_row(_PLAN_A,
                  primary_action_reason_code="recovery_exhausted",
                  chunk_failed_count=1,
                  chunk_failed_unreplaced_count=1),
    ]
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=fake_rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    plans = r.json()["plans"]
    assert plans[0]["primary_action_reason"] == "recovery_exhausted"
