"""[P1-SSE-QUEUE-BOUNDED · 2026-07-09] La cola de progreso SSE de /analyze/stream
debe estar ACOTADA (maxsize). Pre-fix era `asyncio.Queue()` sin límite: si el
cliente se desconecta el pipeline sigue corriendo (P1-DEEP-SEARCH-PIPELINE) y el
producer sigue emitiendo eventos por toda la ventana wall-clock (~10 min) →
acumulación en RAM proporcional al nº de streams abandonados. Fix: cap + drop
best-effort del evento de progreso cuando la cola está llena (el plan igual
persiste y se recupera via /pending-status), preservando la entrega del sentinel
`_done` a un consumer vivo (drop-oldest-then-put).

Parser-based (mismo patrón que test_p1_dedup_recent_plan): ancla el contrato en
el source para que un rename/regresión falle el test antes de tocar producción.
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _plans_src():
    return _read(_BACKEND, "routers", "plans.py")


def test_progress_queue_is_bounded():
    src = _plans_src()
    assert "asyncio.Queue(maxsize=" in src, "la progress_queue SSE debe construirse con maxsize (acotada)"
    # No debe quedar la construcción bare unbounded para progress_queue.
    assert "progress_queue: asyncio.Queue = asyncio.Queue()\n" not in src, (
        "quedó una progress_queue unbounded (asyncio.Queue() sin maxsize)"
    )


def test_queue_maxsize_is_a_knob():
    src = _plans_src()
    assert "MEALFIT_SSE_PROGRESS_QUEUE_MAXSIZE" in src, (
        "el maxsize de la cola SSE debe venir de un knob MEALFIT_* (rollback sin redeploy)"
    )


def test_producer_handles_full_queue_gracefully():
    src = _plans_src()
    # El producer corre en un thread-pool y agenda put_nowait via call_soon_threadsafe;
    # con maxsize, put_nowait lanza asyncio.QueueFull EN EL LOOP → debe capturarse ahí
    # (el try/except que envuelve call_soon_threadsafe NO cubre el put agendado).
    assert "asyncio.QueueFull" in src, (
        "el producer debe capturar asyncio.QueueFull para no romper el event loop cuando la cola está llena"
    )
    # marker anchor
    assert "P1-SSE-QUEUE-BOUNDED" in src, "falta el tooltip-anchor P1-SSE-QUEUE-BOUNDED"


def test_done_sentinel_survives_full_queue():
    """El evento `_done` cierra el generador SSE; si la cola está llena al final del
    run, debe entregarse igual (drop-oldest-then-put) para no colgar el stream hasta
    el timeout de 5s del consumer."""
    src = _plans_src()
    seg = src[src.find("P1-SSE-QUEUE-BOUNDED"):]
    assert "get_nowait()" in seg, (
        "el push del sentinel _done bajo cola llena debe hacer drop-oldest (get_nowait) antes del put"
    )
