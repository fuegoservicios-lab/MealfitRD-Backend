"""[P1-28] Tests para que `acquire_user_and_global` enforce un cap combinado
sobre la suma de las esperas per-user + global.

Bug original (audit P1-28):
  `acquire_user_and_global(user_id)` (y su async pair) componen los dos
  semáforos:
    1. `PER_USER_LLM_SEMAPHORE.acquire(user_id)` — espera hasta
       `LLM_USER_MAX_WAIT_S` (default 30s) antes de degradar a local.
    2. `LLM_SEMAPHORE.acquire()` — espera hasta `LLM_MAX_WAIT_S`
       (default 90s) antes de degradar a local.

  Cada semáforo aplica su bound INDEPENDIENTEMENTE. En el peor caso
  (per-user saturado + global saturado), el caller pagaba
  `30s + 90s = 120s` por una sola llamada LLM. Sin un cap combinado, el
  caller no podía expresar "no me hagas esperar más de X total". Bajo
  carga, los 30s de espera per-user que ya señalaban saturación se
  pagaban DOS veces (per-user saturado → otro 90s de global saturado).

Fix:
  Knob env nuevo `LLM_COMBINED_MAX_WAIT_S` con default = suma de los
  individuales (preserva comportamiento previo). En la composición,
  tras adquirir per-user (o degradar a su local), se mide elapsed:
    - Si elapsed >= COMBINED_MAX_WAIT: degradar global a su
      `_local_semaphore` directamente, sin tocar Redis. Esto evita el
      segundo wait largo cuando el primer wait ya consumió el budget.
    - Caso contrario: adquirir global normal.

  Counters de observabilidad (`_LLM_BUDGET_STATS`) permiten a SRE
  detectar cuándo la degradación dispara (señal de mala calibración o
  saturación crónica).

Cobertura:
  - test_combined_max_wait_knob_exposed
  - test_combined_max_wait_default_is_sum_of_individuals
  - test_budget_stats_counters_exposed
  - test_acquire_under_budget_proceeds_through_global
  - test_acquire_over_budget_degrades_to_local_global
  - test_aacquire_over_budget_degrades_to_alocal_global
  - test_budget_exceeded_increments_counter
  - test_budget_exceeded_logs_warning
  - test_documentation_p1_28_present
"""
import asyncio
import inspect
import logging
import time
from contextlib import contextmanager, asynccontextmanager
from unittest.mock import patch, MagicMock

import pytest

import graph_orchestrator
from graph_orchestrator import (
    acquire_user_and_global,
    aacquire_user_and_global,
)


# ---------------------------------------------------------------------------
# 1. Knob y counters expuestos.
# ---------------------------------------------------------------------------
def test_combined_max_wait_knob_exposed():
    """`LLM_COMBINED_MAX_WAIT_S` debe estar exportado a nivel módulo."""
    assert hasattr(graph_orchestrator, "LLM_COMBINED_MAX_WAIT_S")
    assert isinstance(graph_orchestrator.LLM_COMBINED_MAX_WAIT_S, int)
    assert graph_orchestrator.LLM_COMBINED_MAX_WAIT_S > 0


def test_combined_max_wait_default_is_sum_of_individuals():
    """Default = `LLM_USER_MAX_WAIT_S + LLM_MAX_WAIT_S` (sin override de env).
    Esto preserva el comportamiento histórico cuando no se setea el knob."""
    user_max = graph_orchestrator.LLM_USER_MAX_WAIT_S
    global_max = graph_orchestrator.LLM_MAX_WAIT_S
    assert graph_orchestrator.LLM_COMBINED_MAX_WAIT_S == user_max + global_max, (
        f"P1-28: default LLM_COMBINED_MAX_WAIT_S debería ser {user_max + global_max} "
        f"(suma de individuales), vio {graph_orchestrator.LLM_COMBINED_MAX_WAIT_S}"
    )


def test_budget_stats_counters_exposed():
    """`get_llm_budget_stats_snapshot()` debe devolver dict con counters
    `combined_budget_exceeded` y `combined_total_warnings`."""
    assert hasattr(graph_orchestrator, "get_llm_budget_stats_snapshot")
    snap = graph_orchestrator.get_llm_budget_stats_snapshot()
    assert isinstance(snap, dict)
    assert "combined_budget_exceeded" in snap
    assert "combined_total_warnings" in snap


# ---------------------------------------------------------------------------
# 2. Path normal: dentro del budget, global se adquiere normalmente.
# ---------------------------------------------------------------------------
def test_acquire_under_budget_proceeds_through_global():
    """Per-user adquiere instantáneamente → elapsed=0 < budget → global se
    adquiere por el path normal (NO se degrada a local)."""
    counter_before = graph_orchestrator.get_llm_budget_stats_snapshot()[
        "combined_budget_exceeded"
    ]
    # Mock: per-user yields instantáneo; LLM_SEMAPHORE.acquire() yields
    # también instantáneo. Verificamos que entra al CM normal.
    @contextmanager
    def fast_per_user(user_id):
        yield

    @contextmanager
    def fast_global():
        yield

    fake_per_user_sem = MagicMock()
    fake_per_user_sem.acquire = fast_per_user
    fake_global_sem = MagicMock()
    fake_global_sem.acquire = fast_global
    fake_global_sem._local_semaphore = MagicMock()
    fake_global_sem._local_semaphore.__enter__ = MagicMock()
    fake_global_sem._local_semaphore.__exit__ = MagicMock()

    with patch("graph_orchestrator.PER_USER_LLM_SEMAPHORE", fake_per_user_sem), \
         patch("graph_orchestrator.LLM_SEMAPHORE", fake_global_sem):
        with acquire_user_and_global("user-A"):
            pass

    # El local_semaphore NO debe haberse usado (path normal de global).
    fake_global_sem._local_semaphore.__enter__.assert_not_called()
    counter_after = graph_orchestrator.get_llm_budget_stats_snapshot()[
        "combined_budget_exceeded"
    ]
    assert counter_after == counter_before, (
        "P1-28: contador de degradación NO debe incrementar en path normal."
    )


# ---------------------------------------------------------------------------
# 3. Path degradado: budget excedido tras per-user → global degrada a local.
# ---------------------------------------------------------------------------
def test_acquire_over_budget_degrades_to_local_global():
    """Si per-user consume >= LLM_COMBINED_MAX_WAIT_S, global debe usar
    su `_local_semaphore` directamente sin invocar `LLM_SEMAPHORE.acquire()`.

    Estrategia: ponemos el budget en 0s — cualquier tiempo positivo de
    espera tras per-user (incluso microsegundos) supera el budget. Es
    estable: no requiere mockear time.monotonic ni controlar el orden de
    ejecución asyncio."""
    @contextmanager
    def fast_per_user(user_id):
        yield

    fake_per_user_sem = MagicMock()
    fake_per_user_sem.acquire = fast_per_user

    fake_global_sem = MagicMock()
    fake_global_sem.acquire = MagicMock()  # NO debe invocarse
    fake_global_sem._local_semaphore = MagicMock()
    fake_global_sem._local_semaphore.__enter__ = MagicMock()
    fake_global_sem._local_semaphore.__exit__ = MagicMock(return_value=False)

    with patch.object(graph_orchestrator, "PER_USER_LLM_SEMAPHORE", fake_per_user_sem), \
         patch.object(graph_orchestrator, "LLM_SEMAPHORE", fake_global_sem), \
         patch.object(graph_orchestrator, "LLM_COMBINED_MAX_WAIT_S", 0):
        with acquire_user_and_global("user-A"):
            pass

    # `acquire()` del global NO debe haberse llamado.
    fake_global_sem.acquire.assert_not_called()
    # `_local_semaphore` SÍ debe haberse usado.
    fake_global_sem._local_semaphore.__enter__.assert_called_once()
    fake_global_sem._local_semaphore.__exit__.assert_called_once()


def test_aacquire_over_budget_degrades_to_alocal_global():
    """Versión async: per-user "lento" (budget=0) → global usa `_alocal_acquire()`."""
    @asynccontextmanager
    async def fast_per_user_async(user_id):
        yield

    alocal_entered = [False]

    @asynccontextmanager
    async def fake_alocal():
        alocal_entered[0] = True
        yield

    fake_per_user_sem = MagicMock()
    fake_per_user_sem.aacquire = fast_per_user_async

    fake_global_sem = MagicMock()
    fake_global_sem.aacquire = MagicMock()  # NO debe invocarse
    fake_global_sem._alocal_acquire = fake_alocal

    async def _runner():
        async with aacquire_user_and_global("user-A"):
            pass

    with patch.object(graph_orchestrator, "PER_USER_LLM_SEMAPHORE", fake_per_user_sem), \
         patch.object(graph_orchestrator, "LLM_SEMAPHORE", fake_global_sem), \
         patch.object(graph_orchestrator, "LLM_COMBINED_MAX_WAIT_S", 0):
        asyncio.run(_runner())

    fake_global_sem.aacquire.assert_not_called()
    assert alocal_entered[0], (
        "P1-28: en over-budget async, _alocal_acquire debe ser usado."
    )


# ---------------------------------------------------------------------------
# 4. Counter + warning telemetría.
# ---------------------------------------------------------------------------
def test_budget_exceeded_increments_counter():
    """Cada vez que la degradación dispara, `combined_budget_exceeded`
    debe incrementar."""
    counter_before = graph_orchestrator.get_llm_budget_stats_snapshot()[
        "combined_budget_exceeded"
    ]

    @contextmanager
    def fast_per_user(user_id):
        yield

    fake_per_user_sem = MagicMock()
    fake_per_user_sem.acquire = fast_per_user
    fake_global_sem = MagicMock()
    fake_global_sem._local_semaphore = MagicMock()
    fake_global_sem._local_semaphore.__enter__ = MagicMock()
    fake_global_sem._local_semaphore.__exit__ = MagicMock(return_value=False)

    with patch.object(graph_orchestrator, "PER_USER_LLM_SEMAPHORE", fake_per_user_sem), \
         patch.object(graph_orchestrator, "LLM_SEMAPHORE", fake_global_sem), \
         patch.object(graph_orchestrator, "LLM_COMBINED_MAX_WAIT_S", 0):
        with acquire_user_and_global("user-A"):
            pass

    counter_after = graph_orchestrator.get_llm_budget_stats_snapshot()[
        "combined_budget_exceeded"
    ]
    assert counter_after == counter_before + 1, (
        f"P1-28: contador de degradación debe incrementar 1. "
        f"before={counter_before}, after={counter_after}"
    )


def test_budget_exceeded_logs_warning(caplog):
    """La degradación debe emitir un warning con elapsed + budget para
    correlación con dashboards."""
    @contextmanager
    def fast_per_user(user_id):
        yield

    fake_per_user_sem = MagicMock()
    fake_per_user_sem.acquire = fast_per_user
    fake_global_sem = MagicMock()
    fake_global_sem._local_semaphore = MagicMock()
    fake_global_sem._local_semaphore.__enter__ = MagicMock()
    fake_global_sem._local_semaphore.__exit__ = MagicMock(return_value=False)

    with patch.object(graph_orchestrator, "PER_USER_LLM_SEMAPHORE", fake_per_user_sem), \
         patch.object(graph_orchestrator, "LLM_SEMAPHORE", fake_global_sem), \
         patch.object(graph_orchestrator, "LLM_COMBINED_MAX_WAIT_S", 0), \
         caplog.at_level(logging.WARNING, logger="graph_orchestrator"):
        with acquire_user_and_global("user-A"):
            pass

    p128_logs = [r for r in caplog.records if "[P1-28]" in r.getMessage()]
    assert p128_logs, (
        f"P1-28: esperaba warning [P1-28]. Logs: "
        f"{[r.getMessage()[:120] for r in caplog.records]}"
    )
    msg = p128_logs[0].getMessage()
    # Debe incluir el budget (0s en este test).
    assert "0s" in msg or "0.0s" in msg or "≥" in msg, (
        f"P1-28: msg debe incluir el budget, vio: {msg!r}"
    )


# ---------------------------------------------------------------------------
# 5. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_28_present():
    """Comentario `[P1-28]` debe documentar el cap combinado."""
    full_src = inspect.getsource(graph_orchestrator)
    assert "[P1-28]" in full_src


def test_documentation_mentions_combined_or_budget():
    """El comentario debe explicar el rationale: cap combinado / budget /
    suma de individuales. Sin esto un futuro lector podría borrar el guard
    pensando que cada semáforo ya tiene su propio bound."""
    src_acquire = inspect.getsource(acquire_user_and_global)
    needles = ["combined", "budget", "suma", "cap", "degradar", "2×", "doble"]
    found = any(n in src_acquire.lower() for n in (n.lower() for n in needles))
    assert found, (
        f"P1-28: el comentario debe explicar combined / budget / "
        f"degradar. Encontrado: {src_acquire[:300]!r}"
    )
