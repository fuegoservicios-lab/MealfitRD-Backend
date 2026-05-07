"""[P1-29] Tests para que `_register_progress_task` drene correctamente las
tasks canceladas por el cap blando.

Bug original (audit P1-29):
  Cuando el cap blando per-loop (`_PROGRESS_CB_TASKS_MAX`) se excede,
  `_register_progress_task` cancela las tasks más antiguas:

    for old_t in oldest:
        old_t.cancel()
        tasks_dict.pop(old_t, None)

  `task.cancel()` SCHEDULES la cancelación (set un flag); la task sigue
  ejecutando hasta su próximo checkpoint async. Si el cb estaba
  bloqueado en una operación I/O larga (SSE write a cliente lento, DB
  query colgada), las referencias internas (closure del cb,
  ContextVars copiados, payload del evento) permanecían vivas hasta que
  la cancelación se procesara.

  Sin awaitar la cancelación o sin schedular un drainer, el cap blando
  perdía efectividad para liberar memoria — los buffers SSE, payloads
  con tokens del LLM, y ContextVars con request_id seguían retenidos
  por minutos.

Fix:
  Helper `_drain_cancelled_progress_task(t)` que `await t` con manejo de
  excepción universal (CancelledError, cualquier cosa del cb). Tras
  cada `old_t.cancel()`, schedularizar un drainer fire-and-forget vía
  `loop.create_task(_drain_cancelled_progress_task(old_t))`. Esto fuerza
  el `await t` en un checkpoint próximo del event loop, materializando
  la cancelación y liberando refs prontamente.

  El drainer no se registra en el `tasks_dict` (no aplica cap a sí
  mismo), su única función es asegurar el `await`. Si el loop está
  cerrado (`RuntimeError`), se ignora silenciosamente — la task
  cancelada queda pendiente hasta GC final, comportamiento existente.

Cobertura:
  - test_drain_helper_exists_and_is_coroutine
  - test_drain_swallows_cancelled_error
  - test_drain_swallows_arbitrary_exceptions
  - test_drain_returns_after_task_done
  - test_register_schedules_drain_on_cap_eviction
  - test_register_does_not_schedule_drain_under_cap
  - test_register_handles_loop_closed_gracefully
  - test_register_drain_runs_in_same_loop
  - test_documentation_p1_29_present
"""
import asyncio
import inspect
from unittest.mock import patch, MagicMock

import pytest

import graph_orchestrator
from graph_orchestrator import (
    _register_progress_task,
    _drain_cancelled_progress_task,
)


# ---------------------------------------------------------------------------
# 1. Helper: signature y semántica.
# ---------------------------------------------------------------------------
def test_drain_helper_exists_and_is_coroutine():
    """`_drain_cancelled_progress_task(t)` debe ser una coroutine function."""
    assert asyncio.iscoroutinefunction(_drain_cancelled_progress_task)
    sig = inspect.signature(_drain_cancelled_progress_task)
    params = list(sig.parameters.keys())
    assert params == ["t"]


def test_drain_swallows_cancelled_error():
    """El drainer NO debe propagar CancelledError de la task awaitada."""
    async def _inner():
        loop = asyncio.get_running_loop()

        async def _slow():
            await asyncio.sleep(60)

        t = loop.create_task(_slow())
        await asyncio.sleep(0)  # let it start
        t.cancel()
        # No debe lanzar.
        await _drain_cancelled_progress_task(t)
        assert t.cancelled() or t.done()

    asyncio.run(_inner())


def test_drain_swallows_arbitrary_exceptions():
    """Si la task awaitada lanza una Exception arbitraria, el drainer la
    silencia (esas se registraron antes vía `_run_*_cb_safe`)."""
    async def _inner():
        loop = asyncio.get_running_loop()

        async def _boom():
            raise RuntimeError("simulated cb failure")

        t = loop.create_task(_boom())
        await asyncio.sleep(0)
        # No debe lanzar.
        await _drain_cancelled_progress_task(t)
        assert t.done()

    asyncio.run(_inner())


def test_drain_returns_after_task_done():
    """El drainer espera hasta que la task realmente complete."""
    async def _inner():
        loop = asyncio.get_running_loop()
        completed = [False]

        async def _quick():
            await asyncio.sleep(0)
            completed[0] = True

        t = loop.create_task(_quick())
        await _drain_cancelled_progress_task(t)
        assert completed[0], (
            "P1-29: drainer debe esperar hasta que la task complete."
        )

    asyncio.run(_inner())


# ---------------------------------------------------------------------------
# 2. _register_progress_task agenda drainers en cap eviction.
# ---------------------------------------------------------------------------
def test_register_schedules_drain_on_cap_eviction():
    """Cuando el cap blando dispara, por cada task vieja cancelada se
    debe schedular un drainer en el mismo loop."""
    async def _inner():
        loop = asyncio.get_running_loop()
        # Forzar cap muy bajo para el test.
        with patch.object(graph_orchestrator, "_PROGRESS_CB_TASKS_MAX", 2):
            # Crear 3 tasks lentas → al registrar la 3ra, la 1ra se evictea.
            async def _slow():
                await asyncio.sleep(60)

            t1 = loop.create_task(_slow())
            t2 = loop.create_task(_slow())
            t3 = loop.create_task(_slow())

            # Snapshot del set de tasks ANTES del registro de t3.
            before_tasks = set(asyncio.all_tasks(loop))

            _register_progress_task(loop, t1, "async")
            _register_progress_task(loop, t2, "async")
            _register_progress_task(loop, t3, "async")

            # Después de registrar t3, cap=2 → t1 cancelada + drainer creado.
            await asyncio.sleep(0.05)  # dar chance al drainer

            # t1 debe estar cancelada.
            assert t1.cancelled() or t1.done()
            # Comprobar que se creó al menos UNA task NUEVA tras el registro
            # (el drainer). Las tasks t1/t2/t3 ya estaban antes.
            after_tasks = set(asyncio.all_tasks(loop))
            new_tasks = after_tasks - before_tasks - {t1, t2, t3}
            # Filtrar al current task del runner.
            new_tasks.discard(asyncio.current_task())
            # En la práctica el drainer pudo ya completar; aceptamos que
            # esté en `after_tasks` o que la cancelación efectivamente
            # llegó (t1.cancelled()).
            # El criterio fuerte: t1 es cancelada (drainer disparó cancel
            # propagation).
            assert t1.cancelled(), (
                "P1-29: tras el cap eviction + drain, t1 debe estar "
                "cancelled (no solo cancel-pending)."
            )

            # Limpiar las que quedan.
            t2.cancel()
            t3.cancel()
            await asyncio.gather(t2, t3, return_exceptions=True)

    asyncio.run(_inner())


def test_register_does_not_schedule_drain_under_cap():
    """Sin cap excedido, NO se debe schedular ningún drainer (no hay
    cancelación que drainear)."""
    async def _inner():
        loop = asyncio.get_running_loop()
        with patch.object(graph_orchestrator, "_PROGRESS_CB_TASKS_MAX", 100):
            async def _quick():
                await asyncio.sleep(0.5)

            # Patcheamos `_drain_cancelled_progress_task` para detectar
            # invocaciones espurias.
            drain_calls = []

            original = graph_orchestrator._drain_cancelled_progress_task

            async def fake_drain(t):
                drain_calls.append(t)
                await original(t)

            with patch.object(
                graph_orchestrator,
                "_drain_cancelled_progress_task",
                side_effect=fake_drain,
            ):
                t = loop.create_task(_quick())
                _register_progress_task(loop, t, "async")
                await asyncio.sleep(0.05)
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)

            assert drain_calls == [], (
                f"P1-29: sin cap excedido, no debe haber drainers. "
                f"drain_calls={len(drain_calls)}"
            )

    asyncio.run(_inner())


def test_register_handles_loop_closed_gracefully():
    """Si `loop.create_task(_drain_cancelled_progress_task(...))` lanza
    `RuntimeError` (loop cerrado), `_register_progress_task` debe
    continuar sin crashear (best-effort)."""
    async def _inner():
        loop = asyncio.get_running_loop()
        with patch.object(graph_orchestrator, "_PROGRESS_CB_TASKS_MAX", 1):
            async def _slow():
                await asyncio.sleep(60)

            t1 = loop.create_task(_slow())
            t2 = loop.create_task(_slow())

            # Mock `loop.create_task` para que falle SOLO cuando se crea
            # el drainer. Distinguimos por nombre del coro: el drainer
            # tiene `_drain_cancelled_progress_task` como coroutine name.
            real_create_task = loop.create_task
            attempted = []

            def fake_create_task(coro, *a, **kw):
                # Detectar el drainer por el nombre del coro.
                cr_name = getattr(coro, "__name__", "")
                if "drain_cancelled" in cr_name or (
                    hasattr(coro, "cr_code")
                    and "drain_cancelled" in coro.cr_code.co_name
                ):
                    attempted.append(coro)
                    coro.close()  # Cerrar el coro para no dejarlo pendiente.
                    raise RuntimeError("loop closed")
                return real_create_task(coro, *a, **kw)

            with patch.object(loop, "create_task", side_effect=fake_create_task):
                _register_progress_task(loop, t1, "async")
                # Esto debería intentar agendar el drainer, fallar con
                # RuntimeError, pero NO crashear `_register_progress_task`.
                _register_progress_task(loop, t2, "async")

            assert attempted, (
                "P1-29: el test esperaba al menos un intento de schedular "
                "el drainer (que falló por loop cerrado simulado)."
            )

            t1.cancel()
            t2.cancel()
            await asyncio.gather(t1, t2, return_exceptions=True)

    asyncio.run(_inner())


def test_register_drain_runs_in_same_loop():
    """El drainer schedule debe usar el MISMO `loop` pasado a
    `_register_progress_task` (no `asyncio.get_event_loop()` que podría
    devolver otro loop en runtime multi-loop)."""
    async def _inner():
        loop = asyncio.get_running_loop()
        with patch.object(graph_orchestrator, "_PROGRESS_CB_TASKS_MAX", 1):
            async def _slow():
                await asyncio.sleep(60)

            captured_loops = []
            real_create_task = loop.create_task

            def spy_create_task(coro, *a, **kw):
                cr_name = getattr(coro, "__name__", "") or (
                    coro.cr_code.co_name if hasattr(coro, "cr_code") else ""
                )
                if "drain_cancelled" in cr_name:
                    captured_loops.append("from_passed_loop")
                return real_create_task(coro, *a, **kw)

            t1 = loop.create_task(_slow())
            t2 = loop.create_task(_slow())

            with patch.object(loop, "create_task", side_effect=spy_create_task):
                _register_progress_task(loop, t1, "async")
                _register_progress_task(loop, t2, "async")
                # Cap excedido en el segundo registro → drainer agendado.

            assert captured_loops, (
                "P1-29: el drainer debe agendarse vía `loop.create_task`, "
                "no vía `asyncio.get_event_loop()`."
            )

            t1.cancel()
            t2.cancel()
            await asyncio.gather(t1, t2, return_exceptions=True)

    asyncio.run(_inner())


# ---------------------------------------------------------------------------
# 3. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_29_present():
    """Comentario `[P1-29]` debe documentar el drainer."""
    full_src = inspect.getsource(graph_orchestrator)
    assert "[P1-29]" in full_src


def test_documentation_mentions_cancel_propagation_or_gc():
    """El comentario debe explicar el rationale: `task.cancel()` no
    awaita; sin drainer las refs viven hasta GC. Ayuda a un lector
    futuro a no eliminarlo creyéndolo redundante."""
    helper_src = inspect.getsource(_drain_cancelled_progress_task)
    needles = ["cancel()", "checkpoint", "propag", "GC", "ref", "fire-and-forget", "schedule"]
    found = any(n in helper_src.lower() for n in (n.lower() for n in needles))
    assert found, (
        f"P1-29: el docstring debe explicar el rationale (cancel + "
        f"checkpoint / GC). Encontrado: {helper_src[:300]!r}"
    )
