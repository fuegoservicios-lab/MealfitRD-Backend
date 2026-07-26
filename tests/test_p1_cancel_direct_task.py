"""[P1-CANCEL-DIRECT-TASK · 2026-07-09] El botón Cancelar cancela DE VERDAD el pipeline.

Forense plan vivo e691498e (2026-07-09 00:44): el usuario clickeó Cancelar 8s tras arrancar, pero el plan
se generó igual (7.5 min, 2 intentos) y se persistió. Causa: el frontend hace `POST /api/plans/cancel`
fire-and-forget Y aborta el SSE casi simultáneamente. El abort dispara `asyncio.CancelledError` (disconnect)
en el backend → path deep-search "Pipeline sigue corriendo" (P1-DEEP-SEARCH-PIPELINE, keep-running para
cierre-de-pestaña). El endpoint `/cancel` solo REGISTRABA el cancel y confiaba en que el SSE loop lo
detectara "en la próxima iteración" — que NO ocurre si el stream ya cerró → el pipeline nunca se cancelaba.

Fix: el endpoint cancela la `asyncio.Task` del pipeline DIRECTAMENTE vía registry `session_id → (task, loop)`,
con `loop.call_soon_threadsafe(task.cancel)` (el endpoint es sync/threadpool → cross-thread-safe). Preserva
deep-search para disconnect SIN cancel explícito (cierre de pestaña NO llama al endpoint). Knob
`MEALFIT_CANCEL_CANCELS_PIPELINE` (default True).
"""
import os
from unittest.mock import MagicMock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PLANS = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-CANCEL-DIRECT-TASK" in _PLANS


def test_knob_default_on():
    assert 'CANCEL_CANCELS_PIPELINE' in _PLANS and 'MEALFIT_CANCEL_CANCELS_PIPELINE' in _PLANS


def test_registry_and_helpers_defined():
    assert "_PIPELINE_TASK_REGISTRY" in _PLANS
    assert "def _register_pipeline_task(" in _PLANS
    assert "def _cancel_pipeline_task_for_session(" in _PLANS


def test_endpoint_cancels_task_directly():
    """El endpoint /cancel debe invocar la cancelación DIRECTA de la task (no solo registrar)."""
    i_ep = _PLANS.index("def api_cancel_plan_generation")
    window = _PLANS[i_ep:i_ep + 4500]  # el endpoint tiene un docstring largo (~40 líneas)
    assert "_cancel_pipeline_task_for_session(" in window


def test_task_registered_after_creation():
    """La task del pipeline se registra tras crearse (para poder cancelarla)."""
    assert "_register_pipeline_task(" in _PLANS
    i_create = _PLANS.index("_pipeline_task = asyncio.create_task(run_pipeline())")
    i_reg = _PLANS.index("_register_pipeline_task(", i_create)
    assert i_reg - i_create < 800, "registrar la task cerca de su creación"


def test_uses_call_soon_threadsafe():
    """Cross-thread: el endpoint sync debe cancelar vía loop.call_soon_threadsafe (NO task.cancel() directo)."""
    i = _PLANS.index("def _cancel_pipeline_task_for_session(")
    window = _PLANS[i:i + 900]
    assert "call_soon_threadsafe" in window


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def p():
    import routers.plans as _p
    return _p


def test_cancel_live_task_schedules_cancel(p, monkeypatch):
    monkeypatch.setattr(p, "CANCEL_CANCELS_PIPELINE", True, raising=False)
    task = MagicMock()
    task.done.return_value = False
    loop = MagicMock()
    p._register_pipeline_task("sess-live", task, loop)
    assert p._cancel_pipeline_task_for_session("sess-live") is True
    loop.call_soon_threadsafe.assert_called_once_with(task.cancel)


def test_cancel_done_task_is_noop(p):
    task = MagicMock()
    task.done.return_value = True
    loop = MagicMock()
    p._register_pipeline_task("sess-done", task, loop)
    assert p._cancel_pipeline_task_for_session("sess-done") is False
    loop.call_soon_threadsafe.assert_not_called()


def test_cancel_unknown_session_is_noop(p):
    assert p._cancel_pipeline_task_for_session("nope-not-registered") is False


def test_unregister_removes_entry(p):
    task = MagicMock(); task.done.return_value = False
    loop = MagicMock()
    p._register_pipeline_task("sess-x", task, loop)
    p._unregister_pipeline_task("sess-x")
    assert p._cancel_pipeline_task_for_session("sess-x") is False


def test_knob_off_does_not_cancel(p, monkeypatch):
    monkeypatch.setattr(p, "CANCEL_CANCELS_PIPELINE", False, raising=False)
    task = MagicMock(); task.done.return_value = False
    loop = MagicMock()
    p._register_pipeline_task("sess-off", task, loop)
    assert p._cancel_pipeline_task_for_session("sess-off") is False
    loop.call_soon_threadsafe.assert_not_called()
