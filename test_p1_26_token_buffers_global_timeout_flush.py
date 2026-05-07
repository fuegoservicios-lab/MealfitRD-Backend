"""[P1-26] Tests para `_flush_all_token_buffers` y su invocación en el path
de degradación graceful global de `arun_plan_pipeline`.

Bug original (audit P1-26):
  `_emit_progress` acumula tokens streamed del LLM en
  `state["_token_buffers"][day_key]` para coalescing — solo despacha al
  callback SSE cuando el buffer supera _PROGRESS_TOKEN_FLUSH_BYTES o pasa
  _PROGRESS_TOKEN_FLUSH_MS desde el último flush. Eventos de transición
  (`day_completed`, `phase`, `metric`, etc.) flushean implícitamente.

  Per-day, `_safe_gen` tiene un finally que flushea el buffer del día
  cuando la coroutine de generación termina (success, exception,
  cancelación cooperativa). Pero existe un bucket especial `_default`
  para eventos sin `day` key (phase events emitidos desde nodos NO-de-
  generación: assemble_plan, review_plan, adversarial_judge), que NO
  tiene un finally simétrico.

  Cuando el global timeout
  (`asyncio.wait_for(run_graph(), timeout=GLOBAL_PIPELINE_TIMEOUT_S)`)
  dispara, los buffers per-day SÍ flushean vía cancellation cooperativa,
  pero `_default` puede quedar con tokens pendientes que se pierden al
  caer al fallback. La UI muestra texto cortado (e.g. "Generando recetas
  saludables y…") y debe esperar al evento del fallback total.

Fix:
  1. Helper `_flush_all_token_buffers(state) -> int` que itera TODAS las
     keys de `state["_token_buffers"]` y flushea cada una. Best-effort
     (excepción de un callback no aborta el resto). Idempotente.
  2. Llamada en el except del global timeout en `arun_plan_pipeline`,
     antes de evaluar el fallback. Garantiza que cualquier token
     pendiente llegue al cliente SSE antes del switch a fallback.

Cobertura:
  - test_helper_exists_and_signature
  - test_helper_returns_zero_when_no_buffers
  - test_helper_returns_zero_when_no_callback
  - test_helper_returns_zero_when_buffers_not_dict
  - test_helper_flushes_all_pending_day_buffers
  - test_helper_flushes_default_bucket
  - test_helper_skips_empty_buffers
  - test_helper_continues_after_callback_failure
  - test_global_timeout_invokes_flush_all
  - test_documentation_p1_26_present
"""
import inspect
import logging
from unittest.mock import MagicMock

import pytest

import graph_orchestrator
from graph_orchestrator import _flush_all_token_buffers


_HELPER_SRC = inspect.getsource(_flush_all_token_buffers)
_FULL_SRC = inspect.getsource(graph_orchestrator)


# ---------------------------------------------------------------------------
# 1. Signature y contrato.
# ---------------------------------------------------------------------------
def test_helper_exists_and_signature():
    """`_flush_all_token_buffers(state) -> int` exportado a nivel módulo."""
    assert callable(_flush_all_token_buffers)
    sig = inspect.signature(_flush_all_token_buffers)
    params = list(sig.parameters.keys())
    assert params == ["state"], f"P1-26: signature inesperada: {params}"


# ---------------------------------------------------------------------------
# 2. Casos triviales.
# ---------------------------------------------------------------------------
def test_helper_returns_zero_when_no_buffers():
    """Sin buffers en state → retorna 0, no toca callback."""
    cb = MagicMock()
    state = {"progress_callback": cb}
    assert _flush_all_token_buffers(state) == 0
    cb.assert_not_called()


def test_helper_returns_zero_when_no_callback():
    """Sin callback → retorna 0 (no hay a quién flushear)."""
    state = {"_token_buffers": {"_default": {"text": "abc", "last_flush": 0}}}
    assert _flush_all_token_buffers(state) == 0


def test_helper_returns_zero_when_buffers_not_dict():
    """Si `_token_buffers` es None / lista / corrupto → no crashea, retorna 0."""
    cb = MagicMock()
    for bad in (None, [], "weird", 42):
        state = {"progress_callback": cb, "_token_buffers": bad}
        assert _flush_all_token_buffers(state) == 0
    cb.assert_not_called()


def test_helper_returns_zero_when_state_not_dict():
    """`state` no-dict → retorna 0 sin crashear."""
    assert _flush_all_token_buffers(None) == 0
    assert _flush_all_token_buffers([]) == 0
    assert _flush_all_token_buffers("garbage") == 0


# ---------------------------------------------------------------------------
# 3. Flush correcto de múltiples buckets.
# ---------------------------------------------------------------------------
def test_helper_flushes_all_pending_day_buffers():
    """Dos días con texto pendiente → ambos se flushean (count = 2),
    el callback recibe DOS eventos `token` con sus chunks."""
    received = []

    def cb(payload):
        received.append(payload)

    state = {
        "progress_callback": cb,
        "_token_buffers": {
            1: {"text": "día uno parcial", "last_flush": 0},
            2: {"text": "día dos parcial", "last_flush": 0},
        },
    }
    flushed = _flush_all_token_buffers(state)
    assert flushed == 2
    assert len(received) == 2
    chunks = sorted(p["data"]["chunk"] for p in received)
    assert chunks == ["día dos parcial", "día uno parcial"]
    # Buffers vacíos tras flush.
    assert state["_token_buffers"][1]["text"] == ""
    assert state["_token_buffers"][2]["text"] == ""


def test_helper_flushes_default_bucket():
    """El bucket `_default` (eventos sin day key) DEBE flushearse — es el
    caso clave del bug P1-26."""
    received = []

    def cb(payload):
        received.append(payload)

    state = {
        "progress_callback": cb,
        "_token_buffers": {
            "_default": {"text": "phase event accumulated", "last_flush": 0},
        },
    }
    flushed = _flush_all_token_buffers(state)
    assert flushed == 1
    assert received[0]["data"]["day"] == "_default"
    assert received[0]["data"]["chunk"] == "phase event accumulated"


def test_helper_skips_empty_buffers():
    """Buffers vacíos NO incrementan el counter ni invocan callback."""
    received = []

    def cb(payload):
        received.append(payload)

    state = {
        "progress_callback": cb,
        "_token_buffers": {
            1: {"text": "", "last_flush": 0},
            2: {"text": "tiene texto", "last_flush": 0},
            3: {"text": "", "last_flush": 0},
        },
    }
    flushed = _flush_all_token_buffers(state)
    assert flushed == 1  # Solo el bucket 2 tenía contenido.
    assert len(received) == 1
    assert received[0]["data"]["chunk"] == "tiene texto"


# ---------------------------------------------------------------------------
# 4. Resiliencia ante fallos del callback.
# ---------------------------------------------------------------------------
def test_helper_continues_after_callback_failure():
    """Si el callback de un día explota, los demás días aún flushean.
    Best-effort: un cliente SSE roto en pleno flush no debe abortar el
    drenaje de los buffers restantes."""
    received_ok = []
    fail_count = [0]

    def cb(payload):
        # Fallar la primera invocación, ok después.
        if fail_count[0] == 0:
            fail_count[0] += 1
            raise RuntimeError("simulated callback boom")
        received_ok.append(payload)

    state = {
        "progress_callback": cb,
        "_token_buffers": {
            1: {"text": "primer día", "last_flush": 0},
            2: {"text": "segundo día", "last_flush": 0},
        },
    }
    # No debe lanzar.
    _flush_all_token_buffers(state)
    # Al menos el segundo día debió flushearse OK.
    assert any(p["data"]["chunk"] == "segundo día" for p in received_ok), (
        "P1-26: tras fallo del primer flush, el segundo debe seguir."
    )


# ---------------------------------------------------------------------------
# 5. Defensa estructural: el global timeout invoca el helper.
# ---------------------------------------------------------------------------
def test_global_timeout_invokes_flush_all():
    """En el except del global timeout (`arun_plan_pipeline`), el helper
    `_flush_all_token_buffers(final_state)` debe invocarse ANTES del
    fallback. Sin esto, los buffers se pierden al construir el fallback."""
    arun_src = inspect.getsource(graph_orchestrator.arun_plan_pipeline)
    # Buscar el except del wait_for global y verificar que el helper se
    # invoca dentro.
    timeout_idx = arun_src.find("EXTREME GRACEFUL DEGRADATION")
    assert timeout_idx > -1, (
        "No se encontró el log marker del except global del timeout."
    )
    # Tomar una ventana de 2000 chars desde ese punto.
    window = arun_src[timeout_idx : timeout_idx + 2000]
    assert "_flush_all_token_buffers(" in window, (
        f"P1-26: el except del global timeout debe invocar "
        f"`_flush_all_token_buffers(final_state)` antes del fallback. "
        f"Source window: {window[:400]!r}"
    )


def test_global_timeout_flushes_before_plan_fallback():
    """Defensa de orden: el flush debe aparecer ANTES del bloque que
    construye `_get_extreme_fallback_plan`. Si el orden se invierte, el
    fallback ocurre primero y el SSE recibe el `plan_result` final con
    los tokens stranded sin entregar."""
    arun_src = inspect.getsource(graph_orchestrator.arun_plan_pipeline)
    flush_idx = arun_src.find("_flush_all_token_buffers(")
    fallback_idx = arun_src.find("_get_extreme_fallback_plan(")
    assert flush_idx > -1, "P1-26: flush call no encontrada en arun_plan_pipeline"
    assert fallback_idx > -1, "fallback call no encontrada"
    assert flush_idx < fallback_idx, (
        f"P1-26: el flush ({flush_idx}) debe ir ANTES del fallback "
        f"({fallback_idx}). Si se invierte, los tokens pendientes se "
        f"pierden al construir el fallback plan."
    )


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_26_present():
    """Comentario `[P1-26]` debe documentar el helper + invocación."""
    assert "[P1-26]" in _FULL_SRC, (
        "P1-26: falta marker que documente el flush global del timeout."
    )


def test_documentation_mentions_default_bucket_or_global_timeout():
    """El comentario debe mencionar el rationale específico: bucket
    `_default` perdido en global timeout. Sin esto, un futuro lector
    podría pensar que el helper duplica el finally per-day y borrarlo."""
    p126_idx = _HELPER_SRC.find("[P1-26]")
    assert p126_idx > -1
    window = _HELPER_SRC[p126_idx : p126_idx + 2500]
    needles = ["_default", "global timeout", "GLOBAL_PIPELINE_TIMEOUT",
               "wait_for", "graceful", "fallback"]
    assert any(n in window.lower() for n in (n.lower() for n in needles)), (
        "P1-26: el comentario debe explicar el rationale (bucket _default / "
        "global timeout / fallback)."
    )
