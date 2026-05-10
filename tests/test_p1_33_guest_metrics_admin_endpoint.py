"""[P1-33] Tests para los helpers `get_guest_metrics_status` /
`force_set_guest_metrics_enabled` y los admin endpoints
`/admin/guest-metrics`, `/admin/guest-metrics/probe`,
`/admin/guest-metrics/force`.

Bug original (audit P1-33):
  El flag `_GUEST_METRICS_ENABLED` se decidía 1× al startup del worker
  vía `verify_pipeline_metrics_guest_insert()`. Si el probe fallaba (e.g.,
  schema drift, migración no aplicada), el flag quedaba en False DURANTE
  TODA la vida del proceso — las métricas de pipelines de guests no se
  persistían y SRE no tenía visibilidad ni control:
    - No había forma de inspeccionar el estado actual sin SSH al pod.
    - No había forma de re-ejecutar el probe tras una corrección manual
      de schema sin reiniciar el worker (perdiendo conexiones cacheadas
      + pipelines en vuelo).
    - No había forma de forzar enable/disable durante incident response.

Fix:
  1. Tracking adicional en graph_orchestrator: `_GUEST_METRICS_LAST_PROBE_AT`,
     `_GUEST_METRICS_LAST_PROBE_RESULT`, `_GUEST_METRICS_LAST_ERROR`,
     `_GUEST_METRICS_LAST_SOURCE` (default | probe | admin_force),
     `_GUEST_METRICS_LAST_REASON`. Lock para reads/writes.
  2. `get_guest_metrics_status() -> dict`: snapshot read-only.
  3. `force_set_guest_metrics_enabled(enabled, reason)`: admin override.
  4. 3 endpoints en routers/plans.py:
     - `GET /admin/guest-metrics`: snapshot.
     - `POST /admin/guest-metrics/probe`: re-run probe on-demand.
     - `POST /admin/guest-metrics/force`: override manual.

Cobertura:
  - test_get_guest_metrics_status_signature_and_default
  - test_force_set_enabled_true_updates_state
  - test_force_set_enabled_false_updates_state
  - test_force_set_records_reason_and_source
  - test_force_set_validates_bool_coercion
  - test_probe_success_populates_tracking_fields
  - test_probe_failure_populates_error_field
  - test_admin_endpoint_get_returns_snapshot
  - test_admin_endpoint_get_requires_admin_token
  - test_admin_endpoint_force_validates_body_enabled_field
  - test_admin_endpoint_force_invokes_helper
  - test_admin_endpoint_probe_invokes_verify_function
  - test_documentation_p1_33_present
"""
import inspect
import time
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import graph_orchestrator


# ---------------------------------------------------------------------------
# 1. Helpers en graph_orchestrator.
# ---------------------------------------------------------------------------
def test_get_guest_metrics_status_signature_and_default():
    """`get_guest_metrics_status()` debe ser callable sin args y retornar
    dict con keys esperados."""
    assert hasattr(graph_orchestrator, "get_guest_metrics_status")
    snap = graph_orchestrator.get_guest_metrics_status()
    assert isinstance(snap, dict)
    expected_keys = {
        "enabled", "last_probe_at", "last_probe_result", "last_error",
        "last_source", "last_reason",
    }
    assert expected_keys <= set(snap.keys()), (
        f"P1-33: keys faltantes en snapshot: {expected_keys - set(snap.keys())}"
    )


def test_force_set_enabled_true_updates_state():
    """`force_set_guest_metrics_enabled(True)` debe propagar al flag."""
    # Reset state.
    graph_orchestrator.force_set_guest_metrics_enabled(False, reason="test_reset")
    assert graph_orchestrator._is_guest_metrics_enabled() is False

    snap = graph_orchestrator.force_set_guest_metrics_enabled(True, reason="test_enable")
    assert snap["enabled"] is True
    assert graph_orchestrator._is_guest_metrics_enabled() is True


def test_force_set_enabled_false_updates_state():
    """`force_set_guest_metrics_enabled(False)` también propaga."""
    graph_orchestrator.force_set_guest_metrics_enabled(True, reason="test_reset")
    snap = graph_orchestrator.force_set_guest_metrics_enabled(False, reason="incident")
    assert snap["enabled"] is False


def test_force_set_records_reason_and_source():
    """El override debe registrar reason + source='admin_force' para auditoría."""
    snap = graph_orchestrator.force_set_guest_metrics_enabled(
        True, reason="manual fix applied"
    )
    assert snap["last_source"] == "admin_force"
    assert snap["last_reason"] == "manual fix applied"


def test_force_set_validates_bool_coercion():
    """Valores truthy/falsy se coercen a bool (no aceptamos None
    silencioso)."""
    snap = graph_orchestrator.force_set_guest_metrics_enabled(1, reason=None)
    assert snap["enabled"] is True
    snap = graph_orchestrator.force_set_guest_metrics_enabled(0, reason=None)
    assert snap["enabled"] is False
    snap = graph_orchestrator.force_set_guest_metrics_enabled("", reason=None)
    assert snap["enabled"] is False


# ---------------------------------------------------------------------------
# 2. Probe populates tracking fields.
# ---------------------------------------------------------------------------
def test_probe_success_populates_tracking_fields():
    """`verify_pipeline_metrics_guest_insert()` exitoso debe poblar
    last_probe_at, last_probe_result=True, last_source='probe',
    last_error=None."""
    t_before = time.time()
    with patch("db_core.execute_sql_write"):
        result = graph_orchestrator.verify_pipeline_metrics_guest_insert()
    t_after = time.time()

    assert result is True
    snap = graph_orchestrator.get_guest_metrics_status()
    assert snap["enabled"] is True
    assert snap["last_probe_result"] is True
    assert snap["last_error"] is None
    assert snap["last_source"] == "probe"
    assert snap["last_probe_at"] is not None
    assert t_before <= snap["last_probe_at"] <= t_after


def test_probe_failure_populates_error_field():
    """Si el probe falla, last_error debe contener el mensaje de error."""
    def _explode(*args, **kwargs):
        raise RuntimeError("simulated DB drift: NOT NULL constraint")

    with patch("db_core.execute_sql_write", side_effect=_explode):
        result = graph_orchestrator.verify_pipeline_metrics_guest_insert()

    assert result is False
    snap = graph_orchestrator.get_guest_metrics_status()
    assert snap["enabled"] is False
    assert snap["last_probe_result"] is False
    assert snap["last_error"] is not None
    assert "simulated DB drift" in snap["last_error"]
    assert "RuntimeError" in snap["last_error"]
    assert snap["last_source"] == "probe"

    # Restaurar para no contaminar otros tests.
    with patch("db_core.execute_sql_write"):
        graph_orchestrator.verify_pipeline_metrics_guest_insert()


# ---------------------------------------------------------------------------
# 3. Admin endpoints.
# ---------------------------------------------------------------------------
def _build_test_client(monkeypatch):
    """Construye un TestClient con CRON_SECRET configurado.

    El router de plans ya define `prefix="/api/plans"` internamente, así
    que `include_router` se hace SIN prefix duplicado."""
    monkeypatch.setenv("CRON_SECRET", "test-token-p1-33")
    from fastapi import FastAPI
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)  # router ya trae su prefix
    return TestClient(app)


def test_admin_endpoint_get_returns_snapshot(monkeypatch):
    """`GET /admin/guest-metrics` con token válido devuelve el snapshot."""
    client = _build_test_client(monkeypatch)
    # Forzar a un estado conocido.
    graph_orchestrator.force_set_guest_metrics_enabled(True, reason="test_get")

    r = client.get(
        "/api/plans/admin/guest-metrics",
        headers={"Authorization": "Bearer test-token-p1-33"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is True
    assert body["last_source"] in ("admin_force", "probe", "default")
    assert "last_probe_at" in body


def test_admin_endpoint_get_requires_admin_token(monkeypatch):
    """Sin Authorization válido → 401/403."""
    client = _build_test_client(monkeypatch)
    # Sin header.
    r = client.get("/api/plans/admin/guest-metrics")
    assert r.status_code in (401, 403)
    # Token incorrecto.
    r = client.get(
        "/api/plans/admin/guest-metrics",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code in (401, 403)


def test_admin_endpoint_force_validates_body_enabled_field(monkeypatch):
    """POST /force sin `enabled` en body o con tipo no-bool → 400."""
    client = _build_test_client(monkeypatch)
    headers = {"Authorization": "Bearer test-token-p1-33"}

    # Sin enabled.
    r = client.post("/api/plans/admin/guest-metrics/force",
                    headers=headers, json={"reason": "x"})
    assert r.status_code == 400

    # enabled como string.
    r = client.post("/api/plans/admin/guest-metrics/force",
                    headers=headers, json={"enabled": "true"})
    assert r.status_code == 400

    # enabled como int.
    r = client.post("/api/plans/admin/guest-metrics/force",
                    headers=headers, json={"enabled": 1})
    assert r.status_code == 400


def test_admin_endpoint_force_invokes_helper(monkeypatch):
    """POST /force con body válido invoca `force_set_guest_metrics_enabled`
    y devuelve el snapshot."""
    client = _build_test_client(monkeypatch)
    headers = {"Authorization": "Bearer test-token-p1-33"}

    r = client.post(
        "/api/plans/admin/guest-metrics/force",
        headers=headers,
        json={"enabled": False, "reason": "incident response"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is False
    assert body["last_source"] == "admin_force"
    assert body["last_reason"] == "incident response"

    # Restaurar.
    client.post(
        "/api/plans/admin/guest-metrics/force",
        headers=headers,
        json={"enabled": True, "reason": "test_cleanup"},
    )


def test_admin_endpoint_probe_invokes_verify_function(monkeypatch):
    """POST /probe invoca `verify_pipeline_metrics_guest_insert` y devuelve
    el snapshot post-probe con `probe_executed=True`."""
    client = _build_test_client(monkeypatch)
    headers = {"Authorization": "Bearer test-token-p1-33"}

    with patch("db_core.execute_sql_write") as mock_write:
        r = client.post(
            "/api/plans/admin/guest-metrics/probe",
            headers=headers,
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["probe_executed"] is True
    assert body["probe_result"] is True
    assert body["enabled"] is True
    # El probe debió haber emitido al menos un INSERT (+ DELETE).
    assert mock_write.call_count >= 1


# ---------------------------------------------------------------------------
# 4. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_33_present():
    """Marker `[P1-33]` debe estar en graph_orchestrator y routers/plans."""
    src_go = inspect.getsource(graph_orchestrator)
    assert "[P1-33]" in src_go, (
        "P1-33: falta marker en graph_orchestrator.py."
    )
    from routers import plans
    src_plans = inspect.getsource(plans)
    assert "[P1-33]" in src_plans, (
        "P1-33: falta marker en routers/plans.py."
    )


def test_documentation_mentions_admin_or_endpoint_or_probe():
    """El comentario debe explicar el rationale: admin / endpoint /
    probe / sin restart / SRE."""
    src_go = inspect.getsource(graph_orchestrator)
    p133_idx = src_go.find("[P1-33]")
    window = src_go[p133_idx : p133_idx + 2500]
    needles = ["admin", "endpoint", "probe", "restart", "redeploy", "SRE",
               "operable", "observable"]
    found = any(n.lower() in window.lower() for n in needles)
    assert found, (
        f"P1-33: el comentario debe explicar el rationale (admin / SRE / "
        f"sin restart). Encontrado: {window[:300]!r}"
    )
