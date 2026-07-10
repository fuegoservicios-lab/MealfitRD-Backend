"""[P3-3 · 2026-05-08] Tests del tracking de lag en el heartbeat daemon
del chunk pickup.

Bug original (audit 2026-05-07):
  El thread daemon `_heartbeat_loop` en cron_tasks.py refresca
  `chunk_user_locks.heartbeat_at` cada `_HB_INTERVAL` segundos. Si Python
  pausa el thread (GC stop-the-world, contención de event loop, scheduler
  stall del SO) entre `stop_event.wait()` y `_do_update()`, el delta real
  entre updates supera el intervalo esperado silenciosamente — siempre
  que el UPDATE eventualmente tenga éxito, no había señal de la anomalía.

  El counter `consecutive_failures` solo cuenta UPDATEs que fallan; un
  UPDATE LENTO pero exitoso era invisible. Si la latencia se acumula y
  cruza `_HB_STALE_MIN`, zombie rescue mata el chunk sin previo warning.

Fix (observabilidad pura, sin cambio de comportamiento):
  1. State extendido con `max_lag_seconds`, `total_updates`, `lagged_updates`,
     `started_at`.
  2. `_do_update()` mide `delta = now - last_heartbeat_at` ANTES del UPDATE.
     Si `delta > _HB_INTERVAL * 1.5` → incrementa `lagged_updates`.
     Si `delta > _HB_INTERVAL * 2` → log WARNING [P3-3/HEARTBEAT-LAG].
  3. `finally:` block en `_heartbeat_loop` emite stats finales:
     - log INFO o WARNING (según anomaly) con max_lag/avg_lag/totals.
     - INSERT best-effort a `pipeline_metrics` con confidence=0.0 cuando
       lag es anómalo (gate visual para dashboards Grafana).

Cobertura:
  - State inicial incluye los nuevos campos con defaults seguros.
  - `_do_update()` mide delta y actualiza max_lag.
  - Threshold `_HB_INTERVAL * 1.5` para lagged_updates.
  - WARNING solo en lag extremo (> 2x interval).
  - `finally:` block en el loop garantiza stats al exit (graceful, exception, GC).
  - INSERT a pipeline_metrics best-effort (no aborta si falla).
  - Stats failure no enmascara el exit reason del thread.
"""
import pathlib
import re

import pytest


# [P1-E2E-FIXTURE-NEON · 2026-07-10] parents[1]: cron_tasks.py vive en backend/, no en tests/.
_CRON = pathlib.Path(__file__).resolve().parents[1] / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def heartbeat_block(cron_source) -> str:
    """Aísla el bloque de `_heartbeat_loop` y la state init."""
    start = cron_source.find('"last_heartbeat_at": datetime.now(timezone.utc)')
    assert start != -1, "State init de heartbeat no encontrada"
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    assert end != -1, "End del bloque heartbeat no encontrado"
    return cron_source[start:end]


# ---------------------------------------------------------------------------
# 1. State extendido con campos de tracking
# ---------------------------------------------------------------------------
def test_state_includes_max_lag_seconds(heartbeat_block):
    assert '"max_lag_seconds": 0.0' in heartbeat_block, (
        "State inicial debe incluir `max_lag_seconds: 0.0`."
    )


def test_state_includes_total_updates(heartbeat_block):
    assert '"total_updates": 0' in heartbeat_block, (
        "State inicial debe incluir `total_updates: 0`."
    )


def test_state_includes_lagged_updates(heartbeat_block):
    assert '"lagged_updates": 0' in heartbeat_block, (
        "State inicial debe incluir `lagged_updates: 0` (counter de ticks "
        "donde delta excedió 1.5x el intervalo esperado)."
    )


def test_state_includes_started_at(heartbeat_block):
    """`started_at` se usa para computar avg_lag al exit."""
    assert '"started_at": datetime.now(timezone.utc)' in heartbeat_block


# ---------------------------------------------------------------------------
# 2. _do_update mide delta antes del UPDATE
# ---------------------------------------------------------------------------
def test_do_update_measures_delta_before_sql(heartbeat_block):
    """`_do_update()` debe computar el delta antes del SQL UPDATE para no
    contar la latencia de la query como lag (sería over-estimate)."""
    do_update_match = re.search(
        r"def _do_update\(\):.*?(?=\n\s{16}\w)",
        heartbeat_block,
        re.DOTALL,
    )
    assert do_update_match is not None, "Función _do_update no encontrada"
    block = do_update_match.group(0)
    # El cálculo del delta debe aparecer ANTES del execute_sql_write.
    delta_idx = block.find("_delta_s = ")
    sql_idx = block.find("execute_sql_write")
    assert delta_idx != -1, "Cálculo de delta no encontrado"
    assert sql_idx != -1, "execute_sql_write no encontrado"
    assert delta_idx < sql_idx, (
        "El delta debe medirse ANTES del UPDATE para no contar latencia "
        "de la query como lag (over-estimate)."
    )


def test_do_update_uses_1p5_threshold_for_lagged(heartbeat_block):
    """`lagged_updates` cuenta ticks con delta > 1.5x el intervalo —
    holgura suficiente para no spamear pero detecta anomalías reales."""
    assert "_HB_INTERVAL * 1.5" in heartbeat_block, (
        "Threshold debe ser `_HB_INTERVAL * 1.5` (50% over expected). "
        "Más estricto = falsos positivos por jitter normal."
    )


def test_do_update_warns_on_2x_threshold(heartbeat_block):
    """WARNING solo en lag extremo (>2x) para no spamear ante GC normal."""
    assert "_HB_INTERVAL * 2" in heartbeat_block, (
        "WARNING debe disparar solo cuando delta > 2x interval — señal "
        "fuerte de GC pause / DB latency / scheduler stall."
    )


def test_warning_log_tag_is_p3_3(heartbeat_block):
    """Log tag debe ser `[P3-3/HEARTBEAT-LAG]` para grep-ability."""
    assert "[P3-3/HEARTBEAT-LAG]" in heartbeat_block


# ---------------------------------------------------------------------------
# 3. Counter total_updates incrementa solo en éxito
# ---------------------------------------------------------------------------
def test_total_updates_increments_inside_try_block(heartbeat_block):
    """`total_updates` solo debe incrementarse tras un UPDATE OK (dentro
    del try, después del execute_sql_write). Si se incrementa en el
    cálculo del delta o en el except, el counter es engañoso."""
    do_update_match = re.search(
        r"def _do_update\(\):.*?(?=\n\s{16}\w)",
        heartbeat_block,
        re.DOTALL,
    )
    block = do_update_match.group(0)
    # `state["total_updates"] += 1` debe aparecer después de last_heartbeat_at = ...
    inc_idx = block.find('state["total_updates"] += 1')
    last_hb_idx = block.find('state["last_heartbeat_at"] = datetime.now')
    assert inc_idx != -1, "Increment de total_updates no encontrado"
    assert last_hb_idx != -1
    assert last_hb_idx < inc_idx, (
        "total_updates debe incrementarse DESPUÉS de last_heartbeat_at — "
        "ambos van en el path de éxito (post-execute_sql_write)."
    )


# ---------------------------------------------------------------------------
# 4. finally block emite stats finales
# ---------------------------------------------------------------------------
def test_finally_block_emits_stats(cron_source):
    """El loop tiene `finally:` que emite stats finales — ejecuta en TODOS
    los exits (graceful stop, exception, GC), garantizando observabilidad
    total del lifecycle del heartbeat thread."""
    start = cron_source.find("def _heartbeat_loop(stop_event, state):")
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    block = cron_source[start:end]
    assert "[P3-3/HEARTBEAT-STATS]" in block, (
        "Stats finales con tag `[P3-3/HEARTBEAT-STATS]` debe loguearse "
        "al exit del thread."
    )
    assert "finally:" in block, (
        "Bloque `finally:` requerido para garantizar emit en TODOS los exits."
    )


def test_finally_emits_pipeline_metrics_on_anomaly(cron_source):
    """Si `_is_anomalous` (lagged>0 o max_lag>2x), INSERT best-effort
    a pipeline_metrics con confidence=0.0 (gate visual para dashboards)."""
    start = cron_source.find("def _heartbeat_loop(stop_event, state):")
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    block = cron_source[start:end]
    assert "INSERT INTO pipeline_metrics" in block, (
        "INSERT a pipeline_metrics requerido en `finally:` para anomaly."
    )
    assert "_chunk_heartbeat_lag" in block, (
        "node label de la métrica debe ser `_chunk_heartbeat_lag` para "
        "filtrado en Grafana."
    )


def test_pipeline_metrics_insert_is_best_effort(cron_source):
    """El INSERT a pipeline_metrics debe estar en try/except interno —
    si falla (DB caída, schema drift), no aborta el thread cleanup."""
    start = cron_source.find("def _heartbeat_loop(stop_event, state):")
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    block = cron_source[start:end]
    # Buscar el patrón try / except dentro del finally.
    finally_idx = block.find("finally:")
    finally_block = block[finally_idx:]
    insert_idx = finally_block.find("INSERT INTO pipeline_metrics")
    assert insert_idx != -1
    # Buscar `try:` antes del INSERT.
    pre_insert = finally_block[:insert_idx]
    assert "try:" in pre_insert, (
        "INSERT a pipeline_metrics debe estar dentro de try/except interno "
        "para no abortar el cleanup del thread si la DB está caída."
    )


def test_stats_failure_does_not_mask_exit_reason(cron_source):
    """Si el computo de stats lanza excepción, debe estar capturada con
    `logger.debug` para no enmascarar el motivo real del exit del thread
    (que está logueado en el `except _outer_err` superior)."""
    start = cron_source.find("def _heartbeat_loop(stop_event, state):")
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    block = cron_source[start:end]
    finally_idx = block.find("finally:")
    finally_block = block[finally_idx:]
    # El except externo del finally debe ser logger.debug, no logger.error.
    assert "no se pudo emitir stats finales" in finally_block, (
        "El except del cómputo de stats debe loguear con debug, mensaje "
        "claro de que stats es secundario al exit reason del thread."
    )


# ---------------------------------------------------------------------------
# 5. Smoke: anomaly criteria
# ---------------------------------------------------------------------------
def test_anomaly_criteria_is_lagged_or_max_lag(cron_source):
    """`_is_anomalous` debe ser TRUE si hubo al menos un lagged_update O
    si max_lag superó 2x interval. Esos son criterios independientes:
    un único spike grande es señal incluso si los demás ticks fueron OK."""
    start = cron_source.find("def _heartbeat_loop(stop_event, state):")
    end = cron_source.find("_heartbeat_thread = _threading.Thread(", start)
    block = cron_source[start:end]
    assert re.search(
        r"_is_anomalous\s*=\s*_lagged\s*>\s*0\s*or\s*_max_lag\s*>\s*_HB_INTERVAL\s*\*\s*2",
        block,
    ), (
        "Criterio de anomalía debe ser `_lagged > 0 or _max_lag > _HB_INTERVAL * 2`."
    )
