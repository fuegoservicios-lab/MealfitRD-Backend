"""[P1-PIPELINE-CONCURRENCY-CAP · 2026-07-09] Ceiling global sobre el nº de pipelines
de generación concurrentes en el proceso `--workers 1`. LLM_MAX_CONCURRENT acota las
*llamadas* LLM, no los *objetos* pipeline: sin este cap, N usuarios distintos (y guests,
que ya evitan el guard per-user) crean N tasks `arun_plan_pipeline` vivas → OOM-creep
bajo lentitud del provider (bg_executor.py:11-14). Fix: contador in-process + admisión
503 en el punto de arranque SSE + release en el done-callback ya existente.

Test híbrido: funcional (semántica acquire/release del contador) + parser (wiring en el
endpoint + knob + anchor), mismo patrón que el resto de la Fase 0.
"""
import os

import routers.plans as plans

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _plans_src():
    return _read(_BACKEND, "routers", "plans.py")


# ---------------------------------------------------------------- funcional

def _reset_counter():
    plans._ACTIVE_PIPELINES["count"] = 0


def test_acquire_up_to_limit_then_rejects():
    assert hasattr(plans, "_try_acquire_pipeline_slot"), "falta _try_acquire_pipeline_slot"
    assert hasattr(plans, "_release_pipeline_slot"), "falta _release_pipeline_slot"
    _reset_counter()
    _saved = plans._MAX_CONCURRENT_PIPELINES
    try:
        plans._MAX_CONCURRENT_PIPELINES = 2
        assert plans._try_acquire_pipeline_slot() is True   # 1
        assert plans._try_acquire_pipeline_slot() is True   # 2 (en el cap)
        assert plans._try_acquire_pipeline_slot() is False  # rechazado
    finally:
        plans._MAX_CONCURRENT_PIPELINES = _saved
        _reset_counter()


def test_release_frees_a_slot():
    _reset_counter()
    _saved = plans._MAX_CONCURRENT_PIPELINES
    try:
        plans._MAX_CONCURRENT_PIPELINES = 1
        assert plans._try_acquire_pipeline_slot() is True
        assert plans._try_acquire_pipeline_slot() is False
        plans._release_pipeline_slot()
        assert plans._try_acquire_pipeline_slot() is True
    finally:
        plans._MAX_CONCURRENT_PIPELINES = _saved
        _reset_counter()


def test_release_never_goes_negative():
    _reset_counter()
    # release de más NO debe bajar el contador por debajo de 0 (si no, se "gana" capacidad fantasma)
    plans._release_pipeline_slot()
    plans._release_pipeline_slot()
    assert plans._ACTIVE_PIPELINES["count"] == 0


# ---------------------------------------------------------------- parser (wiring)

def test_cap_is_a_knob():
    assert "MEALFIT_MAX_CONCURRENT_PLAN_PIPELINES" in _plans_src(), (
        "el ceiling debe venir de un knob MEALFIT_* (rollback/tuning sin redeploy)"
    )


def test_gate_before_create_task():
    """La admisión (503) debe ir ANTES de asyncio.create_task(run_pipeline()) — si no,
    ya se creó la task antes de rechazar."""
    src = _plans_src()
    pos_gate = src.find("_try_acquire_pipeline_slot()")
    pos_create = src.find("_pipeline_task = asyncio.create_task(run_pipeline())")
    assert pos_gate >= 0 and pos_create >= 0, "faltan el gate o el create_task"
    assert pos_gate < pos_create, "el gate de capacidad debe ir antes de crear la task del pipeline"


def test_returns_503_server_busy():
    src = _plans_src()
    assert '"code": "server_busy_generating"' in src, "el rechazo por capacidad debe ser 503 server_busy_generating"
    assert "status_code=503" in src, "el rechazo por capacidad debe usar HTTP 503 (retryable)"


def test_slot_released_in_done_callback():
    """El slot debe liberarse en _on_pipeline_task_done (corre en natural/error/cancel)
    para no leakear capacidad si el SSE cerró temprano."""
    src = _plans_src()
    dc = src[src.find("def _on_pipeline_task_done"):]
    dc = dc[: dc.find("\n        _pipeline_task.add_done_callback") + 200] if "_pipeline_task.add_done_callback" in dc else dc[:4000]
    assert "_release_pipeline_slot()" in dc, "el done-callback debe liberar el slot del pipeline"


def test_marker_present():
    assert "P1-PIPELINE-CONCURRENCY-CAP" in _plans_src(), "falta el tooltip-anchor P1-PIPELINE-CONCURRENCY-CAP"
