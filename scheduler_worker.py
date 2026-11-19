# -*- coding: utf-8 -*-
"""[P1-PERF-AUDIT-2026-11-18] Proceso dedicado para los 83 crons de APScheduler.

Desacopla los trabajos periódicos del proceso HTTP/SSE de Uvicorn (`mealfit-backend.service`),
eliminando la contención de GIL y habilitando `--workers 2+` en el servidor web.

Activación controlada sin riesgo:
  1. Arrancar `mealfit-scheduler.service` (adquiere el leader advisory lock).
  2. Configurar `MEALFIT_RUN_SCHEDULER_IN_WEB=false` en el entorno del servicio web.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("scheduler_worker")


def main() -> int:
    from apscheduler.executors.pool import ThreadPoolExecutor as _APThreadPoolExecutor
    from apscheduler.schedulers.background import BackgroundScheduler
    import db_core
    from app import (
        _acquire_scheduler_leader_lock,
        _release_scheduler_leader_lock,
        _scheduler_alert_listener,
        run_proactive_checks,
    )
    from cron_tasks import _add_job_jittered, register_plan_chunk_scheduler
    from hydration_reminders import run_hydration_checks
    from proactive_agent import MINUTOS_ENTRE_TICKS

    if db_core.connection_pool is not None:
        db_core.connection_pool.open(wait=True, timeout=15.0)
    if db_core.chat_checkpoint_pool is not None:
        db_core.chat_checkpoint_pool.open(wait=False)

    is_leader, leader_conn = _acquire_scheduler_leader_lock()
    if not is_leader:
        logger.warning("⏰ [scheduler_worker] Otro proceso posee el leader lock; saliendo limpio.")
        return 0

    stop_event = threading.Event()

    def _handle_sig(signum, _frame):
        logger.info(f"🛑 [scheduler_worker] Señal {signum} recibida — iniciando apagado limpio.")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    scheduler = BackgroundScheduler(
        executors={"default": _APThreadPoolExecutor(max_workers=32)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
    )
    try:
        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
        scheduler.add_listener(
            _scheduler_alert_listener,
            EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_EXECUTED,
        )
    except Exception as e:
        logger.warning(f"[scheduler_worker] listener no registrado: {e}")

    _add_job_jittered(
        scheduler, run_proactive_checks, "cron",
        minute=f"*/{MINUTOS_ENTRE_TICKS}", id="proactive_meal_reminders", replace_existing=True,
    )
    _add_job_jittered(
        scheduler, run_hydration_checks, "cron",
        minute=32, id="hydration_reminders", replace_existing=True,
    )
    register_plan_chunk_scheduler(scheduler)
    scheduler.start()
    logger.info("✅ [scheduler_worker] APScheduler iniciado en proceso dedicado.")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=5.0)
    finally:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        _release_scheduler_leader_lock(leader_conn)
        if db_core.connection_pool is not None:
            db_core.connection_pool.close()
        if db_core.chat_checkpoint_pool is not None:
            db_core.chat_checkpoint_pool.close()
        logger.info("👋 [scheduler_worker] Apagado completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
