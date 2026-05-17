"""[P1-BG-THREAD-TIMEOUT · 2026-05-15] SSOT para ejecución de background
tasks fire-and-forget con timeout duro + alerta en `system_alerts` cuando
se rebasa.

Bug observado (audit production-readiness 2026-05-15):
    `backend/routers/chat.py` lanzaba `threading.Thread(target=fn, daemon=True).start()`
    en dos sitios (título de chat línea 335; `bg_tasks` summarize + facts
    extraction línea 436). Sin timeout, si Gemini/Supabase se cuelgan
    (rate limit upstream, pool exhaustion, network blip) el thread daemon
    vive hasta que el proceso reinicia. Bajo carga de 100+ chats concurrentes
    + un blip de 5min, eso acumula cientos de threads zombies → memory +
    GIL pressure → degradación gradual del worker que Easypanel termina
    OOM-killeando. Clase de bug "el servidor empieza lento después de un día
    y nadie sabe por qué".

Fix:
    - `ThreadPoolExecutor` compartido bounded (default 16 workers).
    - `submit_bg_task` lanza el task + un watcher daemon que llama
      `future.result(timeout=...)` y emite alert `bg_task_timeout:<name>`
      + log si timeout exceeds.
    - Knobs `MEALFIT_BG_TASK_MAX_WORKERS` y `MEALFIT_BG_TASK_TIMEOUT_S`
      auto-registrados en `_KNOBS_REGISTRY` vía `_env_int` (visible en
      `/health/version`).

Limitaciones honestas:
    - Python no expone interrupción segura de threads — `future.cancel()`
      solo funciona si el task aún no empezó. Si Gemini está esperando
      response del LLM, el thread no se puede matar; el watcher emite la
      alerta y el thread terminará eventualmente. Bounded pool + alerts
      sustituyen al "kill" duro.
    - Tasks que SÍ terminan dentro del timeout y luego lanzan excepción
      se loguean al `error` level (no propagan al caller — fire-and-forget).

Tooltip-anchor: P1-BG-THREAD-TIMEOUT.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

from knobs import _env_int

logger = logging.getLogger(__name__)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


# Knobs auto-registrados (visibles en `/health/version`).
_MAX_WORKERS = _clamp(_env_int("MEALFIT_BG_TASK_MAX_WORKERS", 16), 4, 64)
_DEFAULT_TIMEOUT_S = _clamp(_env_int("MEALFIT_BG_TASK_TIMEOUT_S", 120), 30, 600)

# Pool global compartido. Thread-safe en sí mismo (`Executor.submit` es
# thread-safe). Lazy init no se requiere — `ThreadPoolExecutor` no abre
# threads hasta que reciba el primer submit.
_executor = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="mealfit-bg",
)


def _persist_bg_task_timeout_alert(task_name: str, timeout_s: int) -> None:
    """Best-effort UPSERT en `system_alerts`. Cualquier excepción se loguea
    pero NO propaga al watcher (el watcher debe terminar limpio aunque la
    alerta falle — no queremos un segundo timeout watch encadenado)."""
    try:
        from db_core import execute_sql_write

        alert_key = f"bg_task_timeout:{task_name}"
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'bg_task_timeout', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                resolved_at = NULL
            """,
            (
                alert_key,
                f"Background task `{task_name}` excedió timeout {timeout_s}s",
                (
                    f"El task `{task_name}` no terminó dentro del timeout "
                    f"configurado ({timeout_s}s). Patrón habitual: LLM/Supabase "
                    f"cuelga upstream. Si recurre, revisar logs del worker y "
                    f"considerar bumpear `MEALFIT_BG_TASK_TIMEOUT_S` o "
                    f"investigar el cuello de botella."
                ),
                json.dumps(
                    {"task_name": task_name, "timeout_s": timeout_s},
                    ensure_ascii=False,
                ),
                json.dumps([]),
            ),
        )
    except Exception as e:
        logger.warning(
            f"[P1-BG-THREAD-TIMEOUT] No se pudo persistir alert "
            f"bg_task_timeout:{task_name}: {e}"
        )


def submit_bg_task(
    fn: Callable[..., Any],
    *args: Any,
    task_name: str,
    timeout_s: Optional[int] = None,
    **kwargs: Any,
) -> Future:
    """Submit fire-and-forget al pool compartido + watcher para alert si
    el task excede `timeout_s`.

    Args:
        fn: callable a ejecutar.
        *args, **kwargs: forwardeados al callable.
        task_name: identificador estable usado en `alert_key=bg_task_timeout:<name>`.
            Usa snake_case y describe la operación (`chat_title_generation`,
            `chat_sse_bg_tasks`, etc.).
        timeout_s: override del default global (`MEALFIT_BG_TASK_TIMEOUT_S=120`).
            Clamp [10, 1800].

    Returns:
        El `Future`. Caller NO debe `await` ni `.result()` — el watcher
        gestiona el lifecycle. El Future se retorna por si caller quiere
        encadenar `add_done_callback` u observabilidad adicional.
    """
    effective_timeout = _clamp(
        timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S,
        10,
        1800,
    )
    future: Future = _executor.submit(fn, *args, **kwargs)

    def _watch() -> None:
        try:
            future.result(timeout=effective_timeout)
        except FutureTimeoutError:
            logger.warning(
                f"[P1-BG-THREAD-TIMEOUT] task `{task_name}` excedió "
                f"timeout {effective_timeout}s — emitiendo alert + "
                f"cancel best-effort"
            )
            _persist_bg_task_timeout_alert(task_name, effective_timeout)
            future.cancel()
        except Exception as e:
            logger.error(
                f"[P1-BG-THREAD-TIMEOUT] task `{task_name}` falló: {e}"
            )

    threading.Thread(
        target=_watch,
        daemon=True,
        name=f"mealfit-bg-watch-{task_name}",
    ).start()
    return future
