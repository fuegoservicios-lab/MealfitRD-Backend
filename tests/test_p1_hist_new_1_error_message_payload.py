"""[P1-HIST-NEW-1 · 2026-05-09] Tests del payload `metrics.error_message`
en el endpoint `/{plan_id}/chunk-metrics`.

Bug original (audit profundo Historial 2026-05-09):
    El frontend del tab Métricas descartaba `error_message` aunque el
    backend ya lo devolvía. El fix es 100% client-side (render del
    chip "Error: <truncated>" con tooltip), pero este test cierra el
    cross-link del marker (`test_p2_hist_audit_14_marker_test_link`
    requiere `tests/test_p1_hist_new_1*.py`) Y protege el campo
    contra un refactor accidental del SELECT que lo borre.

Cobertura backend (este archivo):
    1. Anchor del marker en el endpoint.
    2. SQL del SELECT incluye `m.error_message`.
    3. Build del dict `metrics` incluye la key `error_message`.
    4. Response shape: `metrics.error_message` viaja como string o None.
    5. `_has_metrics` considera `error_message` poblado como señal
       de "metrics commiteados" (sino, plan que solo tuvo
       error_message sin otros campos perdería el dict completo).

Cobertura frontend (separada — `History.p1_error_message_render.test.js`):
    Render del chip + truncate + tooltip + clases CSS.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CHUNK_X = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _base_chunk_row(**overrides):
    """Fila representativa del LEFT JOIN q+m+ul del endpoint
    chunk-metrics. Cubre todas las keys que el handler lee — un
    overrides selectivo permite simular escenarios específicos
    (chunk failed con error_message, chunk completed sin
    metrics, etc.) sin verbosidad."""
    base = {
        "chunk_id": _CHUNK_X,
        "week_number": 1,
        "days_offset": 0,
        "days_count": 3,
        "status": "failed",
        "quality_tier": None,
        "attempts": 3,
        "chunk_kind": "first_chunk",
        "lag_seconds_at_pickup": 60,
        "effective_lag_seconds_at_pickup": 60,
        "expected_preemption_seconds": None,
        "reservation_status": "ok",
        "escalated_at": None,
        "learning_persisted_at": None,
        "dead_letter_reason": "recovery_exhausted",
        "dead_lettered_at": None,
        "chunk_created_at": None,
        "chunk_updated_at": None,
        "learning_metrics": None,
        "duration_ms": None,
        "was_degraded": None,
        "retries": 3,
        "metrics_lag_seconds": None,
        "learning_repeat_pct": None,
        "rejection_violations": None,
        "allergy_violations": None,
        "pantry_snapshot_age_hours": None,
        "error_message": None,
        "is_rolling_refill": None,
        "metrics_created_at": None,
        "blocking_lock_chunk_id": None,
        "blocking_lock_age_seconds": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    """Marker debe estar en `frontend/src/pages/History.jsx` —
    el fix es 100% client-side. Test backend valida el payload
    pero el render vive en el frontend."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists(), (
        f"History.jsx no existe en {history_jsx} — el fix es client-side."
    )
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P1-HIST-NEW-1" in text, (
        "Marker `P1-HIST-NEW-1` debe aparecer en History.jsx donde "
        "vive el fix."
    )


# ---------------------------------------------------------------------------
# 2. SQL del SELECT incluye `m.error_message`
# ---------------------------------------------------------------------------
def test_sql_select_includes_error_message():
    """El SELECT del LEFT JOIN entre q (queue) y m (metrics) debe
    pedir `m.error_message`. Sin esto, el backend no devolvería el
    campo y el render del frontend nunca dispararía aunque la lógica
    esté correcta."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(r"m\.error_message", norm), (
        "SELECT debe incluir `m.error_message` para que el frontend "
        "pueda renderizar el chip diagnóstico."
    )


# ---------------------------------------------------------------------------
# 3. Build del dict `metrics` incluye la key
# ---------------------------------------------------------------------------
def test_metrics_dict_includes_error_message_key():
    """El handler Python debe popular `metrics["error_message"]`
    cuando el row tiene `m.error_message` no-NULL. Drift detection:
    si un refactor del SQL deja `m.error_message` pero el handler
    olvida la key, el frontend recibe `metrics` sin `error_message`."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # `"error_message": r.get("error_message")` (o equivalente).
    assert re.search(
        r'["\']error_message["\']\s*:\s*r\.get\(\s*["\']error_message["\']',
        src,
    ), (
        "El dict `metrics_obj` debe asignar `error_message` desde el "
        "row del SELECT. Refactors deben preservar esa línea."
    )


# ---------------------------------------------------------------------------
# 4. Response shape: error_message en metrics
# ---------------------------------------------------------------------------
def test_response_includes_error_message_when_populated():
    """End-to-end: row con `error_message='boom'` → response devuelve
    `chunks[0].metrics.error_message == 'boom'`."""
    fake_row = _base_chunk_row(
        status="failed",
        error_message="AssertionError: pantry diff > threshold",
        # Otra key non-null para que `_has_metrics` sea True.
        retries=2,
        metrics_created_at="2026-05-09T10:00:00Z",
    )

    def _fake(query, params=None, **kwargs):
        # Primer call: ownership check (fetch_one).
        if "FROM meal_plans" in query and "fetch_one" not in str(query):
            return {"id": _PLAN_A}
        # Segundo call: el SELECT principal con LEFT JOIN.
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200, r.text
    body = r.json()
    chunks = body.get("chunks") or []
    assert len(chunks) == 1
    metrics = chunks[0].get("metrics")
    assert metrics is not None, (
        "metrics debe estar presente cuando hay al menos un campo "
        "no-NULL (incluyendo error_message)."
    )
    assert metrics.get("error_message") == "AssertionError: pantry diff > threshold"


def test_response_metrics_null_when_no_error_message_and_no_other_fields():
    """Si TODOS los campos del LEFT JOIN m están NULL, `metrics` es
    None (no `{...todo None}`). Garantiza que el frontend distinga
    'no commit' de 'commit con todo vacío'."""
    fake_row = _base_chunk_row(
        status="pending",
        error_message=None,
        retries=None,
        metrics_created_at=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunks = r.json().get("chunks") or []
    assert len(chunks) == 1
    assert chunks[0].get("metrics") is None, (
        "Sin campos commiteados (incluyendo error_message=None), "
        "metrics debe ser None — distingue 'no commit' de 'all-None commit'."
    )


# ---------------------------------------------------------------------------
# 5. error_message dispara `_has_metrics`
# ---------------------------------------------------------------------------
def test_error_message_alone_triggers_has_metrics():
    """Caso edge: chunk con SOLO `error_message` poblado (otros
    campos NULL). El dict `metrics` debe construirse igual — sin
    esto, un chunk failed con solo el exception text perdería el
    chip en el frontend porque `metrics` saldría null."""
    fake_row = _base_chunk_row(
        status="failed",
        error_message="TimeoutError",
        # Todos los demás campos del m.* son None.
        duration_ms=None,
        was_degraded=None,
        retries=None,
        metrics_lag_seconds=None,
        learning_repeat_pct=None,
        rejection_violations=None,
        allergy_violations=None,
        pantry_snapshot_age_hours=None,
        is_rolling_refill=None,
        metrics_created_at=None,
    )

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-metrics")

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert chunk["metrics"] is not None, (
        "error_message poblado debe ser señal suficiente para "
        "construir metrics — sin esto, chunks failed con solo el "
        "exception perderían el chip diagnóstico."
    )
    assert chunk["metrics"]["error_message"] == "TimeoutError"
