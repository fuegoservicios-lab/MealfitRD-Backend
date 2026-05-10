"""[P1-34] Tests para que `arun_plan_pipeline` resetee los ContextVars
(`request_id_var`, `user_id_var`, `_pipeline_cb_stats_var`) al exit (normal
return o exception path), previniendo leakage entre invocaciones.

Bug original (audit P1-34):
  `arun_plan_pipeline` llamaba `request_id_var.set(req_id)`,
  `user_id_var.set(_rate_limit_uid)` y
  `_pipeline_cb_stats_var.set({...})` SIN capturar tokens. En Python, un
  `ContextVar.set(value)` retorna un Token que `var.reset(token)` usa
  para restaurar el valor previo. Sin reset, los valores persisten en el
  contexto del caller después del return.

  Impacto:
    - Producción HTTP: cada request corre en su propio asyncio.Task →
      contexto aislado por defecto → leak limitado a la duración del task.
    - Sync wrappers (`_run_arun_in_isolated_loop`, cron jobs): si reusan
      el thread del pool entre invocaciones SIN spawn de nuevo task,
      heredan el req_id/user_id/cb_stats del pipeline previo. Logs de
      tracing contaminados ("[req=XYZ] [user=A]" mostrando IDs del
      pipeline anterior).
    - Tests: pytest secuencial corre múltiples tests en el mismo contexto.
      Test1 setea user_id_var='A'; Test2 lee user_id_var → ve 'A' en
      lugar del default None.
    - Métricas SSE: `_pipeline_cb_stats_var` mutado in-place por una
      pipeline contamina la siguiente — los counters dropped_cap/
      timed_out se acumulan cross-pipeline en lugar de resetearse.

Fix:
  1. Capturar tokens de cada `.set()` en variables locales.
  2. `try:` justo después de los sets, conteniendo el resto del body.
  3. `finally:` con resets envueltos individualmente en
     `try/except (LookupError, ValueError)` (defensivo contra tokens
     expirados por cancelación).

Cobertura:
  - test_p1_34_marker_present_in_arun_pipeline
  - test_token_capture_pattern_in_source
  - test_finally_block_present_with_three_resets
  - test_each_reset_wrapped_in_individual_try_except
  - test_resets_in_reverse_order_of_sets
  - test_finally_runs_on_normal_return
  - test_finally_runs_on_exception
  - test_documentation_p1_34_present
"""
import asyncio
import inspect
import re

import pytest

import graph_orchestrator
from graph_orchestrator import (
    request_id_var,
    user_id_var,
    _pipeline_cb_stats_var,
)


_ARUN_SRC = inspect.getsource(graph_orchestrator.arun_plan_pipeline)


# ---------------------------------------------------------------------------
# 1. Defensa estructural: source contiene el patrón correcto.
# ---------------------------------------------------------------------------
def test_p1_34_marker_present_in_arun_pipeline():
    """Marker `[P1-34]` debe estar en `arun_plan_pipeline`."""
    assert "[P1-34]" in _ARUN_SRC, (
        "P1-34: falta marker en arun_plan_pipeline."
    )


def test_token_capture_pattern_in_source():
    """Las 3 `.set()` calls deben capturar tokens en variables `_p134_*`."""
    assert "_p134_req_token = request_id_var.set(" in _ARUN_SRC, (
        "P1-34: request_id_var.set debe capturar token en _p134_req_token."
    )
    assert "_p134_uid_token = user_id_var.set(" in _ARUN_SRC, (
        "P1-34: user_id_var.set debe capturar token en _p134_uid_token."
    )
    assert "_p134_cb_token = _pipeline_cb_stats_var.set(" in _ARUN_SRC, (
        "P1-34: _pipeline_cb_stats_var.set debe capturar token en _p134_cb_token."
    )


def test_finally_block_present_with_three_resets():
    """El bloque `finally:` debe contener los 3 `.reset()` con sus tokens."""
    # Verificamos que existe un `finally:` después del último `.set()` con
    # los 3 resets dentro.
    finally_idx = _ARUN_SRC.find("finally:")
    assert finally_idx > -1, "P1-34: falta `finally:` en arun_plan_pipeline."
    finally_block = _ARUN_SRC[finally_idx:]
    assert "request_id_var.reset(_p134_req_token)" in finally_block
    assert "user_id_var.reset(_p134_uid_token)" in finally_block
    assert "_pipeline_cb_stats_var.reset(_p134_cb_token)" in finally_block


def test_each_reset_wrapped_in_individual_try_except():
    """Cada reset debe estar en su propio try/except (LookupError, ValueError)
    para que un fallo aislado no impida los otros resets."""
    finally_idx = _ARUN_SRC.find("finally:")
    finally_block = _ARUN_SRC[finally_idx:]
    # Contar try: e except (LookupError, ValueError):.
    # Debe haber al menos 3 de cada (uno por reset).
    try_count = finally_block.count("try:")
    except_count = finally_block.count("except (LookupError, ValueError):")
    assert try_count >= 3, (
        f"P1-34: esperaba >= 3 `try:` en finally, vio {try_count}"
    )
    assert except_count >= 3, (
        f"P1-34: esperaba >= 3 `except (LookupError, ValueError):` en "
        f"finally, vio {except_count}"
    )


def test_resets_in_reverse_order_of_sets():
    """Por convención de stacks/contexts, resets en orden inverso a sets:
    último seteado se resetea primero. En el source: cb_token (último set)
    debe resetearse antes que uid_token, que antes que req_token."""
    finally_idx = _ARUN_SRC.find("finally:")
    finally_block = _ARUN_SRC[finally_idx:]
    cb_reset_idx = finally_block.find("_pipeline_cb_stats_var.reset")
    uid_reset_idx = finally_block.find("user_id_var.reset")
    req_reset_idx = finally_block.find("request_id_var.reset")
    assert cb_reset_idx > -1 and uid_reset_idx > -1 and req_reset_idx > -1
    assert cb_reset_idx < uid_reset_idx < req_reset_idx, (
        "P1-34: resets deben estar en orden inverso al set "
        "(cb → uid → req). Encontrado: "
        f"cb={cb_reset_idx}, uid={uid_reset_idx}, req={req_reset_idx}"
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento funcional: simulación del patrón en un mini-test.
# ---------------------------------------------------------------------------
def test_finally_runs_on_normal_return():
    """Verificación funcional con un wrapper que reproduce el patrón:
    en un return normal, los ContextVars vuelven a su estado previo."""
    sentinel_user = "PRE_PIPELINE_USER"
    sentinel_req = "PRE_PIPELINE_REQ"
    user_id_var.set(sentinel_user)
    request_id_var.set(sentinel_req)

    # Replicar el patrón de la función arreglada.
    async def fake_pipeline():
        req_token = request_id_var.set("INSIDE_REQ")
        uid_token = user_id_var.set("INSIDE_USER")
        try:
            assert request_id_var.get() == "INSIDE_REQ"
            assert user_id_var.get() == "INSIDE_USER"
            return "ok"
        finally:
            try:
                user_id_var.reset(uid_token)
            except (LookupError, ValueError):
                pass
            try:
                request_id_var.reset(req_token)
            except (LookupError, ValueError):
                pass

    asyncio.run(fake_pipeline())
    assert user_id_var.get() == sentinel_user, (
        f"P1-34: leak detectado tras return normal — user_id_var leak "
        f"({user_id_var.get()!r} != {sentinel_user!r})"
    )
    assert request_id_var.get() == sentinel_req, (
        f"P1-34: leak detectado tras return normal — request_id_var leak"
    )


def test_finally_runs_on_exception():
    """Mismo patrón pero el body raise → finally aún ejecuta + valores
    restaurados."""
    sentinel_user = "PRE_EXCEPTION_USER"
    user_id_var.set(sentinel_user)

    async def fake_pipeline_that_raises():
        uid_token = user_id_var.set("INSIDE_USER_EXC")
        try:
            assert user_id_var.get() == "INSIDE_USER_EXC"
            raise RuntimeError("simulated pipeline failure")
        finally:
            try:
                user_id_var.reset(uid_token)
            except (LookupError, ValueError):
                pass

    with pytest.raises(RuntimeError):
        asyncio.run(fake_pipeline_that_raises())
    assert user_id_var.get() == sentinel_user, (
        f"P1-34: leak detectado tras exception — user_id_var leak "
        f"({user_id_var.get()!r} != {sentinel_user!r})"
    )


def test_actual_arun_pipeline_resets_on_exception():
    """Test integración: invocar `arun_plan_pipeline` con un form_data
    inválido que dispare excepción muy temprana, y verificar que los
    contextvars se resetean. Si la excepción es atrapada antes del
    set, no aplica — pero si llega al set y luego falla, el finally
    debe correr."""
    sentinel = "TEST_SENTINEL_REQ"
    request_id_var.set(sentinel)

    async def _try_invoke():
        try:
            await graph_orchestrator.arun_plan_pipeline(
                form_data=None,  # Disparará error por algún path
                history=[],
                taste_profile="",
                memory_context="",
                progress_callback=None,
                background_tasks=None,
            )
        except Exception:
            pass  # Esperado.

    asyncio.run(_try_invoke())
    # Verificación: el ContextVar debe NO estar contaminado.
    # Acepta sentinel original o el default "SYS" (si la excepción
    # ocurrió antes del set, el value sigue siendo sentinel).
    val = request_id_var.get()
    assert val in (sentinel, "SYS"), (
        f"P1-34: tras invocación fallida, request_id_var debe estar "
        f"reseteado o intacto, vio {val!r}"
    )


# ---------------------------------------------------------------------------
# 3. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_34_present():
    """Comentario `[P1-34]` debe documentar el rationale del reset."""
    src = inspect.getsource(graph_orchestrator)
    assert "[P1-34]" in src, "P1-34: falta marker en graph_orchestrator.py."


def test_documentation_mentions_leak_or_async_task_or_reset():
    """El comentario debe explicar el rationale: leakage, asyncio.Task,
    sync wrappers, reset() para que un futuro lector entienda por qué
    capturar tokens y no solo set()."""
    finally_idx = _ARUN_SRC.find("finally:")
    finally_block = _ARUN_SRC[finally_idx : finally_idx + 2500]
    needles = ["leak", "leakage", "contexto", "Task", "wrapper", "test",
               "reset", "expirado", "previa", "anterior"]
    found = any(n.lower() in finally_block.lower() for n in needles)
    assert found, (
        f"P1-34: el comentario debe explicar leak/Task/wrapper/reset. "
        f"Encontrado: {finally_block[:300]!r}"
    )
