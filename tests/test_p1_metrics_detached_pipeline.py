"""[P1-METRICS-DETACHED-PIPELINE · 2026-07-29] Las métricas per-run no pueden depender del ciclo de
vida del REQUEST.

Bug medido en producción (2026-07-29, corr=5cbced82): el pipeline logueó
    📐 [P2-SOLVER-CLAMP-ACTION] 8/12 meals con clamp del solver saturado
    📐 [P2-SOLVER-CONVERGENCE-METRIC] 6/12 meals NO convergidos
a las 18:52:08, y `pipeline_metrics` NO tiene ninguna de las dos filas. Tampoco salta el
`[METRICS] Insert falló`, porque `_save` nunca llegó a ejecutarse.

Causa: `_emit_progress` encolaba el insert en el `BackgroundTasks` de FastAPI, cuyas tareas corren
al terminar la RESPUESTA. Pero `P1-DEEP-SEARCH-PIPELINE` mantiene el pipeline vivo como task
detached DESPUÉS de que el cliente cierra el SSE — precisamente el caso que ese feature existe para
soportar. Todo `add_task` posterior se encola en un objeto que nadie va a ejecutar.

El sesgo es el peor posible: se pierden las métricas de las corridas MÁS LARGAS, que son las que más
interesa medir y las que más probablemente pierden al cliente por el camino. Es PREEXISTENTE (la
corrida de las 13:52, anterior a los despliegues del día, perdió la suya igual) y el sink está sano
(46-203 filas/hora), así que no era una caída del sink sino una fuga selectiva.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_exists_and_defaults_to_executor():
    import graph_orchestrator as go
    assert 'METRICS_ALWAYS_EXECUTOR = _env_bool("MEALFIT_METRICS_ALWAYS_EXECUTOR", True)' in _GO
    assert go.METRICS_ALWAYS_EXECUTOR is True, \
        "default ON: perder telemetría en silencio es peor que un thread extra del pool acotado"


def test_emit_progress_bypasses_request_background_tasks():
    seg = _GO[_GO.index("def _emit_progress"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "bg_tasks = None if METRICS_ALWAYS_EXECUTOR else state.get(\"background_tasks\")" in seg, \
        "con el knob ON no debe tocarse el BackgroundTasks del request"
    assert "_METRICS_EXECUTOR.submit" in seg, "el executor dedicado sigue siendo el camino real"


def test_knob_is_declared_before_use():
    """Un knob leído antes de definirse sería NameError en cada métrica — tragado por el
    `except Exception` de preparación y convertido en pérdida silenciosa, que es justo el modo de
    fallo que este fix viene a cerrar."""
    assert _GO.index("METRICS_ALWAYS_EXECUTOR = _env_bool") < \
        _GO.index("bg_tasks = None if METRICS_ALWAYS_EXECUTOR")


class _FakeBG:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *a, **kw):
        self.tasks.append(fn)


def test_metric_goes_to_executor_not_to_background_tasks(monkeypatch):
    """Funcional: con el knob ON, un `BackgroundTasks` presente en el state NO recibe la tarea."""
    import graph_orchestrator as go

    submitted = []

    class _FakeExec:
        def submit(self, fn, *a, **kw):
            submitted.append(fn)

    bg = _FakeBG()
    monkeypatch.setattr(go, "_METRICS_EXECUTOR", _FakeExec())
    monkeypatch.setattr(go, "METRICS_ALWAYS_EXECUTOR", True)

    state = {"form_data": {"user_id": "u-1", "session_id": "s-1"}, "background_tasks": bg}
    go._emit_progress(state, "metric", {"node": "solver_convergence", "confidence": 0.5,
                                        "metadata": {"total_meals": 12}})

    assert not bg.tasks, "con el pipeline detached, el BackgroundTasks del request es un agujero negro"
    assert submitted, "la métrica debe irse por el executor, que no depende del request"


def test_rollback_restores_background_tasks(monkeypatch):
    import graph_orchestrator as go
    bg = _FakeBG()
    monkeypatch.setattr(go, "METRICS_ALWAYS_EXECUTOR", False)
    state = {"form_data": {"user_id": "u-1", "session_id": "s-1"}, "background_tasks": bg}
    go._emit_progress(state, "metric", {"node": "solver_clamp", "confidence": 0.1, "metadata": {}})
    assert bg.tasks, "con el knob OFF vuelve el comportamiento previo (rollback sin redeploy)"
