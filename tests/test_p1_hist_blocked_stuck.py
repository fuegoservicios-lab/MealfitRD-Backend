"""[P1-HIST-BLOCKED-STUCK · 2026-05-09] Tests del parámetro
``include_stuck`` en ``GET /api/plans/{id}/blocked_reasons``.

Bug original (audit Historial 2026-05-09 · gap P1-3):
    El endpoint `/blocked_reasons` solo devolvía chunks
    `pending_user_action` y opcionalmente `failed` (con
    `dead_letter_reason`). Chunks atascados en `processing` o `stale`
    con lag alto NO aparecían — el banner del modal del Historial
    no tenía surface para chunks zombi (worker crash + advisory lock
    no liberado, pipeline colgado tras LLM timeout sin escalar a
    failed, etc.). El cron `_alert_high_chunk_lag` los reportaba al
    ops team pero el usuario solo veía "Generando 2/15" hasta que
    otro cron los pasaba a `failed` (≥1h después).

Fix:
    Nuevo parámetro `include_stuck=true` (default False, retrocompat).
    Cuando se activa, el SQL suma:
      `OR (status IN ('processing','stale') AND
           execute_after < NOW() - make_interval(hours => N))`
    donde N viene del knob `MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS`
    (default 3.0, validator > 0). Reason codes nuevos:
      - `stuck_processing`: chunk con worker activo pero sin avanzar.
      - `stuck_stale`: chunk marcado stale tras crash del worker.

Cobertura:
    - Anchor del marker.
    - Knob `MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS` registrado vía
      `_env_float` con validator > 0.
    - Parámetro `include_stuck` declarado en el endpoint signature.
    - Default False (retrocompat: Dashboard del plan ACTIVO sigue
      recibiendo solo PUAC).
    - SQL contiene la cláusula `make_interval(hours => %s)` cuando
      `include_stuck=True`; ausente cuando False.
    - Reason codes `stuck_processing` y `stuck_stale` en
      `reason_to_text` con copy info (NO scare).
    - Lógica del bucle: status=processing → reason_code='stuck_processing';
      status=stale → reason_code='stuck_stale'.
    - Response shape: cada entry incluye `lag_seconds`.
    - Auth + 404 + 403 (mismo patrón que el endpoint original).
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "33333333-3333-3333-3333-333333333333"
_OTHER = "44444444-4444-4444-4444-444444444444"
_PLAN_ID = "ffffffff-ffff-ffff-ffff-ffffffffffff"


def _client():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


def _client_no_auth():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: None
    return client


# ---------------------------------------------------------------------------
# 1. Anchor + signature del endpoint
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert "P1-HIST-BLOCKED-STUCK · 2026-05-09" in src


def test_include_stuck_param_declared():
    """`include_stuck: bool = False` debe aparecer en la signature
    para que FastAPI lo exponga como query param tipado."""
    from routers.plans import api_blocked_reasons
    sig = inspect.signature(api_blocked_reasons)
    assert "include_stuck" in sig.parameters
    param = sig.parameters["include_stuck"]
    assert param.default is False
    assert param.annotation is bool


def test_include_stuck_default_false_retrocompat():
    """Default False preserva el contrato del Dashboard del plan
    ACTIVO (que solo quiere `pending_user_action`)."""
    from routers.plans import api_blocked_reasons
    sig = inspect.signature(api_blocked_reasons)
    assert sig.parameters["include_stuck"].default is False


# ---------------------------------------------------------------------------
# 2. Knob MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS
# ---------------------------------------------------------------------------
def test_knob_called_with_validator_positive():
    """El endpoint debe leer el knob vía `_env_float` con validator
    `lambda v: v > 0` para rechazar lag <= 0 (DOS-able si se permite
    listar chunks healthy sin filter)."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert "MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS" in src
    # Validator inline: `validator=lambda v: v > 0` o variantes.
    assert re.search(
        r"MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS[\s\S]{0,200}?lambda\s+v\s*:\s*v\s*>\s*0",
        src,
    ), "Falta validator > 0 en la llamada a _env_float"


def test_knob_default_hours_value():
    """Default debe ser razonable: 3h da margen a chunks normales
    (que toman 30-90s) sin enmascarar zombis. El test fija 3.0 como
    default actual; bumpearlo requiere bumpear este test (drift
    detection del SLA visible al usuario)."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert re.search(
        r"MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS[\s\S]{0,80}?,\s*3\.0\s*,",
        src,
    ), "Default del knob debe ser 3.0h"


# ---------------------------------------------------------------------------
# 3. SQL: cláusula make_interval condicional
# ---------------------------------------------------------------------------
def test_sql_includes_interval_multiplication_when_include_stuck():
    """SQL debe contener `interval '1 hour' * %s` para que el
    threshold sea binding (NO injection vía string formatting).

    [P-FIX-BLOCKED-REASONS-500 · 2026-05-09] Cambio del shape:
    `make_interval(hours => %s)` → `interval '1 hour' * %s`.
    PostgreSQL `make_interval(hours => ...)` SOLO acepta int; el knob
    default 3.0 (float) lanzaba 42883. Multiplicación de interval
    acepta numeric directamente."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    import re as _re
    assert _re.search(
        r"interval\s+'1\s+hour'\s*\*\s*%s",
        src,
        _re.IGNORECASE,
    ), (
        "SQL debe usar `interval '1 hour' * %s` para evitar el bug "
        "42883 cuando el knob es float."
    )
    # Anti-pattern: el make_interval con numeric NO debe estar (ignora
    # el comentario explicativo del fix).
    code_only = "\n".join(
        line for line in src.splitlines()
        if not _re.match(r"^\s*#", line)
    )
    assert "make_interval(hours =>" not in code_only, (
        "Anti-pattern `make_interval(hours => ...)` con numeric "
        "regresa al bug 42883. Mantener `interval '1 hour' * %s`."
    )


def test_sql_filters_processing_and_stale_status():
    """El filter stuck cubre ambos status (processing + stale).
    Stale: chunks que el cron `_recover_stale_chunks` marcó tras
    crash del worker; processing: chunks con worker vivo pero sin
    transición. Ambos tienen lag accionable."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert re.search(
        r"status\s+IN\s*\(\s*['\"]processing['\"]\s*,\s*['\"]stale['\"]\s*\)",
        src,
    ), "Filter debe cubrir status IN ('processing', 'stale')"


def test_sql_returns_lag_seconds_extract():
    """Response debe incluir `lag_seconds` (NOW - execute_after)
    además de `paused_seconds`. lag es la métrica diagnóstica para
    chunks stuck (paused puede estar fresco si hubo heartbeat
    reciente sin avanzar)."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert "lag_seconds" in src
    assert re.search(
        r"EXTRACT\(EPOCH\s+FROM\s+\(NOW\(\)\s*-\s*execute_after\)\)",
        src,
    )


# ---------------------------------------------------------------------------
# 4. Reason codes + copy
# ---------------------------------------------------------------------------
def test_reason_to_text_contains_stuck_codes():
    """Catálogo de copy incluye `stuck_processing` y `stuck_stale`."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert '"stuck_processing"' in src
    assert '"stuck_stale"' in src


def test_loop_assigns_stuck_reason_code_by_status():
    """Bucle del endpoint asigna `stuck_processing` cuando
    status='processing' y `stuck_stale` cuando status='stale'.
    Slack de 800 chars cubre el comentario inline que documenta el
    invariante (el comentario anterior a la asignación es largo)."""
    from routers.plans import api_blocked_reasons
    src = inspect.getsource(api_blocked_reasons)
    assert re.search(
        r"row_status\s*==\s*['\"]processing['\"][\s\S]{0,800}?reason_code\s*=\s*['\"]stuck_processing['\"]",
        src,
    )
    assert re.search(
        r"row_status\s*==\s*['\"]stale['\"][\s\S]{0,800}?reason_code\s*=\s*['\"]stuck_stale['\"]",
        src,
    )


# ---------------------------------------------------------------------------
# 5. Smoke E2E con mock — include_stuck=true devuelve stuck reasons
# ---------------------------------------------------------------------------
def _ownership_row():
    return {"user_id": _USER}


def _mock_db_chain(plan_row, queue_rows, pref_row=None):
    """Builder de la secuencia de execute_sql_query del endpoint:
      1) ownership SELECT (returns plan_row)
      2) blocked rows SELECT (returns queue_rows)
      3) logging_preference SELECT (returns pref_row, opcional).
    """
    calls = [plan_row, queue_rows, pref_row]
    def _side_effect(*args, **kwargs):
        return calls.pop(0) if calls else None
    return _side_effect


def test_include_stuck_false_excludes_stuck_chunks():
    """Default include_stuck=False — chunks `processing`/`stale`
    no aparecen en el response (retrocompat)."""
    client = _client()
    plan_row = _ownership_row()
    # Simular que el SQL filter (include_stuck=False) ya NO devuelve
    # processing/stale → queue_rows vacío.
    queue_rows = []
    with patch(
        "db_core.execute_sql_query",
        side_effect=_mock_db_chain(plan_row, queue_rows),
    ):
        r = client.get(f"/api/plans/{_PLAN_ID}/blocked_reasons")
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is False
    assert body["reasons"] == []


def test_include_stuck_true_returns_stuck_processing_reason():
    """include_stuck=true + queue tiene chunk `processing` lagged →
    response trae reason_code='stuck_processing'."""
    client = _client()
    plan_row = _ownership_row()
    queue_rows = [
        {
            "id": "chunk-stuck-1",
            "week_number": 2,
            "pipeline_snapshot": {},
            "status": "processing",
            "dead_letter_reason": None,
            "paused_seconds": 12000,
            "lag_seconds": 14000,  # ~3.9h, > 3h threshold
        },
    ]
    with patch(
        "db_core.execute_sql_query",
        side_effect=_mock_db_chain(plan_row, queue_rows),
    ):
        r = client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons?include_stuck=true"
        )
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is True
    assert len(body["reasons"]) == 1
    reason = body["reasons"][0]
    assert reason["reason_code"] == "stuck_processing"
    assert reason["status"] == "processing"
    assert reason["lag_seconds"] == 14000
    # Title/body del template stuck_processing.
    assert "tardando" in reason["title"].lower() or "demor" in reason["title"].lower()


def test_include_stuck_true_returns_stuck_stale_reason():
    """status='stale' → reason_code='stuck_stale'."""
    client = _client()
    plan_row = _ownership_row()
    queue_rows = [
        {
            "id": "chunk-stuck-2",
            "week_number": 3,
            "pipeline_snapshot": {},
            "status": "stale",
            "dead_letter_reason": None,
            "paused_seconds": 18000,
            "lag_seconds": 20000,  # ~5.5h
        },
    ]
    with patch(
        "db_core.execute_sql_query",
        side_effect=_mock_db_chain(plan_row, queue_rows),
    ):
        r = client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons?include_stuck=true"
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["reasons"]) == 1
    assert body["reasons"][0]["reason_code"] == "stuck_stale"
    assert body["reasons"][0]["status"] == "stale"


def test_include_stuck_combined_with_failed_and_pending():
    """include_stuck=true Y include_failed=true Y status mix:
    cada chunk obtiene su reason_code correcto sin colisión."""
    client = _client()
    plan_row = _ownership_row()
    queue_rows = [
        {
            "id": "c-puac",
            "week_number": 1,
            "pipeline_snapshot": {"_pause_reason": "learning_zero_logs"},
            "status": "pending_user_action",
            "dead_letter_reason": None,
            "paused_seconds": 3600,
            "lag_seconds": 3600,
        },
        {
            "id": "c-failed",
            "week_number": 2,
            "pipeline_snapshot": {},
            "status": "failed",
            "dead_letter_reason": "recovery_exhausted",
            "paused_seconds": 7200,
            "lag_seconds": 7200,
        },
        {
            "id": "c-stuck",
            "week_number": 3,
            "pipeline_snapshot": {},
            "status": "processing",
            "dead_letter_reason": None,
            "paused_seconds": 14000,
            "lag_seconds": 14000,
        },
    ]
    with patch(
        "db_core.execute_sql_query",
        side_effect=_mock_db_chain(plan_row, queue_rows),
    ):
        r = client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons"
            f"?include_failed=true&include_stuck=true"
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["reasons"]) == 3
    codes = {x["reason_code"] for x in body["reasons"]}
    assert codes == {"learning_zero_logs", "recovery_exhausted", "stuck_processing"}


# ---------------------------------------------------------------------------
# 6. Auth + 404 + 403 (regresión del comportamiento existente)
# ---------------------------------------------------------------------------
def test_404_when_plan_missing():
    client = _client()
    with patch("db_core.execute_sql_query", return_value=None):
        r = client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons?include_stuck=true"
        )
    assert r.status_code == 404


def test_403_when_plan_belongs_to_other_user():
    client = _client()
    with patch(
        "db_core.execute_sql_query",
        return_value={"user_id": _OTHER},
    ):
        r = client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons?include_stuck=true"
        )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# 7. Knob de runtime — auto-registrado en _KNOBS_REGISTRY
# ---------------------------------------------------------------------------
def test_knob_registered_in_registry_after_call():
    """Tras llamar al endpoint, el knob debe quedar en `_KNOBS_REGISTRY`
    (auto-registro vía `_env_float` — convención del repo P3-NEW-D)."""
    client = _client()
    plan_row = _ownership_row()
    queue_rows = []
    with patch(
        "db_core.execute_sql_query",
        side_effect=_mock_db_chain(plan_row, queue_rows),
    ):
        client.get(
            f"/api/plans/{_PLAN_ID}/blocked_reasons?include_stuck=true"
        )
    from knobs import _KNOBS_REGISTRY
    assert "MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS" in _KNOBS_REGISTRY
    entry = _KNOBS_REGISTRY["MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS"]
    assert entry["type"] == "float"
    assert entry["default"] == 3.0
