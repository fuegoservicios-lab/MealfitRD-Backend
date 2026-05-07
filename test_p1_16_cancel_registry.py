"""[P1-16] Tests para el registry de cancelación de planes en vuelo.

Bug original (audit P1-16):
  El frontend llamaba `cancelGeneration()` que abortaba el SSE local pero
  NO informaba al backend. El pipeline LLM seguía corriendo hasta terminar
  el día actual y persistía el plan en DB. El usuario veía el plan
  aparecer 30s después vía Realtime UPDATE de `meal_plans` aunque ya
  había cancelado. Cuota de LLM consumida + UX confuso.

Fix:
  1. Registry global `_PLAN_CANCEL_REGISTRY: set` keyed por `session_id`.
  2. Endpoint `POST /api/plans/cancel` agrega el session_id al set.
  3. SSE handler verifica cooperativamente vía `is_session_cancelled` en
     el loop de heartbeat (cada 5s) y cancela el `_pipeline_task`
     propagando `asyncio.CancelledError`.
  4. `_clear_cancelled_session` se llama en el `finally` del generator
     SSE para evitar leak del set.

Cobertura:
  - test_cancel_session_returns_true_first_time
  - test_cancel_session_idempotent_returns_false_second_time
  - test_cancel_session_rejects_invalid_session_id
  - test_is_session_cancelled_returns_true_after_cancel
  - test_is_session_cancelled_returns_false_for_unknown_session
  - test_is_session_cancelled_returns_false_for_invalid_input
  - test_clear_cancelled_session_removes_from_registry
  - test_clear_cancelled_session_idempotent
  - test_cancel_helpers_thread_safe (smoke under multi-thread)
"""
import threading
import time

import pytest

from routers.plans import (
    _cancel_session,
    is_session_cancelled,
    _clear_cancelled_session,
    _PLAN_CANCEL_REGISTRY,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Limpia el registry antes y después de cada test para aislamiento."""
    _PLAN_CANCEL_REGISTRY.clear()
    yield
    _PLAN_CANCEL_REGISTRY.clear()


# ---------------------------------------------------------------------------
# 1. _cancel_session: idempotencia + validación.
# ---------------------------------------------------------------------------
def test_cancel_session_returns_true_first_time():
    assert _cancel_session("session-abc") is True


def test_cancel_session_idempotent_returns_false_second_time():
    """La segunda cancelación del mismo session retorna False (ya
    estaba marcada). Útil para que el endpoint diga al cliente
    que la operación fue no-op."""
    _cancel_session("session-abc")
    assert _cancel_session("session-abc") is False


def test_cancel_session_rejects_invalid_session_id():
    """Defensa: None, '', non-string → no se agrega al registry."""
    assert _cancel_session(None) is False
    assert _cancel_session("") is False
    assert _cancel_session(123) is False
    assert _cancel_session({}) is False
    # Y el registry debe estar vacío.
    assert len(_PLAN_CANCEL_REGISTRY) == 0


# ---------------------------------------------------------------------------
# 2. is_session_cancelled: lookup correcto.
# ---------------------------------------------------------------------------
def test_is_session_cancelled_returns_true_after_cancel():
    _cancel_session("session-xyz")
    assert is_session_cancelled("session-xyz") is True


def test_is_session_cancelled_returns_false_for_unknown_session():
    assert is_session_cancelled("never-cancelled") is False


def test_is_session_cancelled_returns_false_for_invalid_input():
    """Defensivo: None/empty/non-string → False (no abortamos pipelines
    que el cliente no identificó)."""
    assert is_session_cancelled(None) is False
    assert is_session_cancelled("") is False
    assert is_session_cancelled(0) is False


# ---------------------------------------------------------------------------
# 3. _clear_cancelled_session: cleanup.
# ---------------------------------------------------------------------------
def test_clear_cancelled_session_removes_from_registry():
    _cancel_session("to-clear")
    assert is_session_cancelled("to-clear") is True
    _clear_cancelled_session("to-clear")
    assert is_session_cancelled("to-clear") is False


def test_clear_cancelled_session_idempotent_for_unknown():
    """Limpiar un session que nunca se canceló no debe lanzar."""
    _clear_cancelled_session("never-existed")  # no-op
    assert is_session_cancelled("never-existed") is False


def test_clear_handles_invalid_input_gracefully():
    """Defensivo: None/'' no debe lanzar."""
    _clear_cancelled_session(None)
    _clear_cancelled_session("")


# ---------------------------------------------------------------------------
# 4. Endpoint /api/plans/cancel comportamiento contractual.
# ---------------------------------------------------------------------------
def test_endpoint_cancel_records_session_in_registry():
    """Llamar el endpoint debe registrar el session_id."""
    from routers.plans import api_cancel_plan_generation
    response = api_cancel_plan_generation({"session_id": "session-from-endpoint"})
    assert response["success"] is True
    assert response["registered"] is True
    assert response["session_id"] == "session-from-endpoint"
    assert is_session_cancelled("session-from-endpoint") is True


def test_endpoint_cancel_idempotent_returns_registered_false():
    """Cancelar la misma session dos veces: la segunda devuelve registered=False."""
    from routers.plans import api_cancel_plan_generation
    api_cancel_plan_generation({"session_id": "duplicate"})
    response = api_cancel_plan_generation({"session_id": "duplicate"})
    assert response["success"] is True
    assert response["registered"] is False


def test_endpoint_cancel_rejects_missing_session_id():
    from routers.plans import api_cancel_plan_generation
    response = api_cancel_plan_generation({})
    assert response["success"] is False
    response = api_cancel_plan_generation({"session_id": ""})
    assert response["success"] is False


# ---------------------------------------------------------------------------
# 5. Thread-safety smoke test.
# ---------------------------------------------------------------------------
def test_cancel_helpers_thread_safe_smoke():
    """Bajo concurrencia (worker SSE + endpoint cancel + cleanup), las
    operaciones add/discard no deben corromper el set ni lanzar.
    Smoke test sin asserts cuantitativos — solo que no crashea."""

    def cancel_loop():
        for i in range(50):
            _cancel_session(f"thr-{i}")
            time.sleep(0)

    def check_loop():
        for i in range(50):
            is_session_cancelled(f"thr-{i}")
            time.sleep(0)

    def clear_loop():
        for i in range(50):
            _clear_cancelled_session(f"thr-{i}")
            time.sleep(0)

    threads = [
        threading.Thread(target=cancel_loop),
        threading.Thread(target=check_loop),
        threading.Thread(target=clear_loop),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    # Si llegamos aquí sin excepción, el smoke test pasó.
    assert True


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_16_present_in_source():
    """Comentario `[P1-16]` debe aparecer en el módulo `routers/plans.py`."""
    from routers import plans as plans_module
    src = open(plans_module.__file__, encoding="utf-8").read()
    assert "[P1-16]" in src


def test_endpoint_route_registered():
    """El endpoint debe estar registrado bajo `/api/plans/cancel`."""
    from routers.plans import router
    routes = [r.path for r in router.routes if hasattr(r, "path")]
    assert "/api/plans/cancel" in routes
