# backend/graph_orchestrator.py
"""
Orquestación LangGraph: Flujo Map-Reduce multi-agente para generación de planes nutricionales.
Planificador → Generadores Paralelos (×3 días) → Ensamblador → Revisor Médico → (loop si falla)
"""

import os
import time
import json
from typing import TypedDict, Optional, Callable, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import threading
from db_plans import search_similar_plan

# Mejora 1: Semaphore Distribuido Global para backpressure
# P1-10: `time` y `uuid` ya están al inicio del módulo; no reimportar.
import uuid
import asyncio
import weakref
from contextlib import contextmanager, asynccontextmanager


# ============================================================
# P1-NEW-2: Knobs de producción configurables vía variables de entorno
# ------------------------------------------------------------
# Antes, valores como `LLM_SEMAPHORE.max_concurrent`, `HARD_CEILING`,
# `GLOBAL_TIMEOUT`, etc. estaban hardcoded en el código. Cambiarlos en
# respuesta a un incidente de producción (proveedor LLM lento, Redis caído,
# cuota agotada, traffic spike) requería un PR + redeploy. Ahora cada knob
# crítico se lee de una env var con prefijo `MEALFIT_*` al import del módulo,
# manteniendo defaults idénticos al comportamiento previo.
#
# Convención: `MEALFIT_<DOMINIO>_<SUBJETO>[_S]` donde sufijo `_S` indica
# unidad en segundos. Tipos validados (int/float). Valor inválido → warning
# y fallback al default. Los valores efectivos se loguean al startup.
# ============================================================
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            f"[KNOBS] env {name}={raw!r} no es int. Usando default={default}."
        )
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            f"[KNOBS] env {name}={raw!r} no es float. Usando default={default}."
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    """P1-NEW-1: parser laxo de booleanos. Acepta `1/true/yes/on` (case-insensitive)
    como verdadero; cualquier otro valor no vacío como falso. Default si vacío.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- LLM backpressure / Semáforo distribuido ---
# Slots concurrentes globales (Redis sorted set + threading/asyncio Semaphore local).
# Subir si tienes cuota generosa del proveedor; bajar si ves rate-limits.
LLM_MAX_CONCURRENT          = _env_int  ("MEALFIT_LLM_MAX_CONCURRENT",          4)
# TTL del lock en Redis: cuánto retiene un slot un worker antes de que el
# garbage collector del semáforo lo considere abandonado.
LLM_REDIS_LOCK_TIMEOUT_S    = _env_int  ("MEALFIT_LLM_REDIS_LOCK_TIMEOUT_S",    120)
# Bound máximo de espera por un slot Redis antes de degradar al semáforo
# local (P0-B). Subir si los pipelines son muy largos; bajar si la cola
# Redis es propensa a saturarse.
LLM_MAX_WAIT_S              = _env_int  ("MEALFIT_LLM_MAX_WAIT_S",              90)

# --- P1-NEW-1: Rate limit per-user (capa SOBRE el semáforo global) ---
# Antes, un único usuario disparando N requests concurrentes (botón regenerar
# en bucle, bug en frontend, abuso) consumía todos los slots globales y
# bloqueaba a otros usuarios. Esta capa adquiere PRIMERO un slot per-user
# y DESPUÉS el global: si el usuario satura su cuota, su request espera sin
# tomar slot global. Permite multi-tenancy justa.
# Bypass automático para `user_id` None / "guest" → preserva comportamiento
# previo en cron jobs, batch, llamadas internas.
LLM_PER_USER_ENABLED        = _env_bool ("MEALFIT_LLM_PER_USER_ENABLED",        True)
LLM_MAX_PER_USER            = _env_int  ("MEALFIT_LLM_MAX_PER_USER",            2)
LLM_USER_LOCK_TIMEOUT_S     = _env_int  ("MEALFIT_LLM_USER_LOCK_TIMEOUT_S",     60)
# Espera máxima por un slot per-user antes de degradar al semáforo global
# (más corto que LLM_MAX_WAIT_S — un usuario saturado debe ceder rápido para
# que el grafo continúe vía global).
LLM_USER_MAX_WAIT_S         = _env_int  ("MEALFIT_LLM_USER_MAX_WAIT_S",         30)

# --- Circuit breaker LLM ---
CB_FAILURE_THRESHOLD        = _env_int  ("MEALFIT_CB_FAILURE_THRESHOLD",        3)
CB_RESET_TIMEOUT_S          = _env_int  ("MEALFIT_CB_RESET_TIMEOUT_S",          30)
CB_LOCAL_HEALTH_TTL_S       = _env_float("MEALFIT_CB_LOCAL_HEALTH_TTL_S",       1.0)

# --- Hedging per-day (generate_days_parallel_node) ---
# `HEDGE_AFTER_BASE_S`: tiempo soft antes de lanzar el intento especulativo.
# `HARD_CEILING_S`: tiempo duro tras el cual cancelamos primary+hedge y damos error.
HEDGE_AFTER_BASE_S          = _env_float("MEALFIT_HEDGE_AFTER_BASE_S",          45.0)
HARD_CEILING_S              = _env_float("MEALFIT_HARD_CEILING_S",              170.0)

# --- Retry policy + timeout global del pipeline ---
MAX_ATTEMPTS                = _env_int  ("MEALFIT_MAX_ATTEMPTS",                2)
MIN_RETRY_BUDGET_S          = _env_int  ("MEALFIT_MIN_RETRY_BUDGET_S",          180)
# [P1-FIX-BUDGET] Subido de 600s → 720s. Análisis del incidente real:
# pipeline normal terminó en 230s con 370s remaining; threshold (180+80+125=385s)
# cortó retry por margen de 15s. Con `GLOBAL_PIPELINE_TIMEOUT_S=600s`, cualquier
# primer intento que tomara >215s denegaba retry — caso muy común dado que el
# happy path (sin retry) toma 200-260s.
#
# Con 720s, un primer intento que tome hasta 335s deja presupuesto suficiente
# (720-335=385s = threshold exacto), recuperando ~120s de "ventana real de
# retry". Trade-off: en el peor caso (retry usado), el usuario espera hasta
# 12 min en lugar de 10. En el caso normal (sin retry, ~95% de pipelines)
# ningún cambio de UX porque `wait_for` solo cancela si excede el cap.
#
# Este es un knob operacional: bajar a 600 si el SLA exige <10min hard cap,
# subir a 900+ si se prioriza la calidad del retry sobre la latencia tail.
GLOBAL_PIPELINE_TIMEOUT_S   = _env_int  ("MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S",   720)
# P1-NEW-5: margen para fases POST-retry (assemble + review_plan + guardrails
# P0-1/P0-2). Antes, `MIN_RETRY_BUDGET_S` solo cubría "reflexión + planner +
# generate_days" (~180s). Pero después del retry todavía corren assemble (~5s)
# + review_plan (~70s con sus retries internos) + guardrails (~5s) ≈ 80s.
# Sin este margen, podíamos aprobar un retry con `remaining=200s`, agotar el
# budget en la generación, y disparar `wait_for(timeout=GLOBAL_PIPELINE_TIMEOUT_S)`
# a mitad del review — cancelando el grafo y perdiendo TODO el trabajo del
# segundo intento. Con margin, el guard preserva el primer intento si el retry
# no cabe sin riesgo.
RETRY_SAFETY_MARGIN_S       = _env_int  ("MEALFIT_RETRY_SAFETY_MARGIN_S",       80)

# P1-A6: delta de cobertura para HEDGING activo en el retry.
# ------------------------------------------------------------
# Antes, el guard de `should_retry` validaba `remaining ≥ MIN_RETRY_BUDGET_S +
# RETRY_SAFETY_MARGIN_S` (=260s default). Pero `MIN_RETRY_BUDGET_S=180s`
# representa el peor caso del HAPPY PATH del retry: reflection (~10-30s) +
# planner (~5-15s) + generate_days_parallel cuando el primary termina rápido
# (cerca de `HEDGE_AFTER_BASE_S=45s`).
#
# Bajo carga (LLM provider lento, hedge_after_base agotado, hedge especulativo
# corriendo), `generate_days_parallel_node` puede llegar a `HARD_CEILING_S=170s`
# en wall-clock — ~125s adicionales sobre el escenario "hedge no se disparó".
# Si aprobamos un retry con remaining justo por encima del threshold previo
# y el provider está lento, agotamos el budget en `generate_days` y disparamos
# `wait_for(GLOBAL_PIPELINE_TIMEOUT_S)` a mitad de `review_plan` — cancelando
# todo el trabajo del segundo intento (mismo modo de fallo que P1-NEW-5
# corrigió para el path post-retry).
#
# Default: `HARD_CEILING_S - HEDGE_AFTER_BASE_S` = ~125s con los defaults
# actuales (170 - 45). Conservador / safety-first: prefiere preservar el
# primer intento antes que aprobar un retry que probablemente se cancelará.
#
# Trade-off operacional:
#   - Subir = más restrictivo: con `GLOBAL_PIPELINE_TIMEOUT_S=600s` el
#     threshold es 385s, así que tras ~215s el guard corta retries.
#   - Bajar = más permisivo pero más riesgo de cancelación bajo carga.
#   - Set a `0` para volver al comportamiento previo (sin cobertura de hedging).
RETRY_HEDGE_BUDGET_DELTA_S  = _env_float("MEALFIT_RETRY_HEDGE_BUDGET_DELTA_S",
                                         max(0.0, HARD_CEILING_S - HEDGE_AFTER_BASE_S))

# --- Fact-check ---
FACT_CHECK_TOOL_TIMEOUT_S   = _env_float("MEALFIT_FACT_CHECK_TOOL_TIMEOUT_S",   20.0)

# --- Self-critique correction timeout (P1-FIX-CRITIQUE) ---
# Timeout por día corregido en `self_critique_node._correct_single_day`.
# Antes era hardcoded 70s; observado en producción: el último día corregido
# (Día 3 cuando se corrigen 3 días) sufría peak de carga del proveedor LLM
# y disparaba TimeoutError → mantenía el día original SIN corregir, justamente
# el día que tenía las violaciones más graves (8.5 lonjas de pavo en cena en
# incidente real). El día sin corregir luego era flageado por `review_plan_node`
# pero ya sin presupuesto para retry → plan rechazado entregado al usuario.
# Subir a 90s da margen de cola del proveedor (~25-30% headroom adicional).
# Las 3 correcciones corren en `asyncio.gather`, así que el costo wall-clock
# es el max de las 3, no la suma — subir el cap individual no mueve el total.
CRITIQUE_FIX_TIMEOUT_S      = _env_float("MEALFIT_CRITIQUE_FIX_TIMEOUT_S",      90.0)

# --- Progress callbacks (SSE) ---
PROGRESS_CB_MAX_PENDING     = _env_int  ("MEALFIT_PROGRESS_CB_MAX_PENDING",     1000)
PROGRESS_CB_TIMEOUT_S       = _env_float("MEALFIT_PROGRESS_CB_TIMEOUT_S",       10.0)

# --- LLM cache TTL ---
LLM_CACHE_TTL_S             = _env_int  ("MEALFIT_LLM_CACHE_TTL_S",             300)

# ============================================================
# P0-A3: Versionado del schema del plan cacheado.
# ------------------------------------------------------------
# Antes, `semantic_cache_check_node` aceptaba cualquier plan en `meal_plans`
# que pasara los filtros de similaridad / frescura / médicos. Si un deploy
# cambiaba el shape del plan (campo nuevo obligatorio, lista renombrada,
# claves removidas en `assemble_plan_node`), el cache servía planes con
# shape viejo → frontend crasheaba al hacer `.shopping_list_weekly.map()`
# sobre `undefined`, o `assemble_plan_node` los reescribía parcialmente
# dejando estructura mixta.
#
# Solución: cada plan generado se marca con `_cache_schema_version` igual a
# la versión actual del orquestador. `semantic_cache_check_node` descarta
# (continue) candidatos cuya versión no matche. Bumpear `CACHE_SCHEMA_VERSION`
# en cualquier deploy que altere el contrato de `plan_data` invalida
# automáticamente todos los planes pre-deploy sin DROP TABLE ni migración.
#
# Convención: `vN` strings, monotónicos. Bump cuando:
#   - Se añade campo top-level requerido por el frontend.
#   - Se renombra/elimina clave en `days[].meals[]`.
#   - Cambia el formato de `aggregated_shopping_list_*`.
#   - Se altera el contrato de `_pantry_degraded_summary`, `_review_issues`,
#     `_active_learning_signals`, etc.
#
# `_LEGACY_CACHE_SCHEMA_VERSION` define cómo tratar planes pre-existentes
# en producción que no tienen el flag — se les asigna esta versión por
# compatibilidad. En el primer deploy se usa "v1" (= la versión actual);
# todos los planes existentes son tratados como compatibles. En el SIGUIENTE
# deploy con cambio de schema, se sube `CACHE_SCHEMA_VERSION` a "v2" y los
# planes legacy quedan automáticamente invalidados.
# ============================================================
CACHE_SCHEMA_VERSION: str = "v1"
_LEGACY_CACHE_SCHEMA_VERSION: str = "v1"

# --- P1-NEW-6: Sync wrapper bounded ---
# Antes, `run_plan_pipeline` (sync API para cron/batch) spawneaba un
# `threading.Thread` nuevo por cada call, cada uno con su propio event loop
# vía `asyncio.run`. N callers concurrentes → N threads + N loops + N copias
# del contexto, y mutaciones concurrentes a globales como `_PROGRESS_CB_TASKS`
# generaban race conditions (cap-drop iteraba/cancelaba tasks de OTROS loops).
# Ahora todo el path sync va por un executor dedicado con concurrencia
# acotada: sobre N callers concurrentes, N-MAX se serializan en cola.
SYNC_WRAPPER_MAX_WORKERS    = _env_int  ("MEALFIT_SYNC_WRAPPER_MAX_WORKERS",    4)

# --- P1-Q7 / P1-A7: Strict mode del sync wrapper (deadlock protection) ---
# `run_plan_pipeline` (sync) llamado desde código async bloquea el thread
# caller con `future.result()` — pero ese thread es el que ejecuta el event
# loop, así que el loop queda CONGELADO durante todo el pipeline (~2-5 min).
# Otras requests/tasks del worker quedan en standby. El executor del wrapper
# eventualmente termina y libera, así que NO es deadlock infinito, pero
# sí degrada catastróficamente la latencia tail de TODO el worker.
#
# P1-A7: default flipped a `True`. Auditoría exhaustiva (grep `run_plan_pipeline\(`
# sobre `backend/`) confirmó que TODOS los call sites legítimos viven en
# funciones `def` sync (no `async def`):
#   - `proactive_agent.py` → `_trigger_week2_background_generation` (sync).
#   - `cron_tasks.py` → `_validate_and_retry_initial_chunk_against_pantry`
#     (sync, x2) y `_refill_emergency_backup_plan` (sync).
#   - `tools.py` → `execute_generate_new_plan` (sync).
#   - `routers/plans.py` → `api_analyze` (FastAPI sync handler, corre en
#     threadpool — sin loop activo en el thread).
# Strict mode no rompe ninguno: en threadpools / cron / scripts NO hay
# `asyncio.get_running_loop()` activo, así que `RuntimeError` solo se levanta
# si en el futuro alguien introduce una llamada DESDE `async def` (regresión
# real, queremos fail-fast).
#
# `True` (default actual): lanza `RuntimeError` explícito redirigiendo a
#   `arun_plan_pipeline`. Previene regresiones futuras donde un dev nuevo
#   meta `result = run_plan_pipeline(...)` desde un handler async.
# `False`: warning ruidoso con stacktrace, sigue funcionando. Útil como
#   escape hatch operacional si aparece un caller legacy no documentado en
#   producción y se necesita tiempo para migrarlo. Bajar vía env var:
#       MEALFIT_SYNC_WRAPPER_STRICT_MODE=0
SYNC_WRAPPER_STRICT_MODE    = _env_bool ("MEALFIT_SYNC_WRAPPER_STRICT_MODE",   True)

# --- P1-NEW-3: Drain de métricas en shutdown ---
# Antes, el atexit hook llamaba `shutdown(wait=False, cancel_futures=True)`,
# descartando silenciosamente las métricas encoladas (~50/pipeline × N
# pipelines en flight) cuando llegaba SIGTERM (rolling deploy, autoscaling).
# Eso degradaba el meta-learning: A/B Thompson Sampling, holistic score
# history, drift detection perdían señal con cada deploy.
# Ahora el shutdown intenta drenar la cola hasta `MEALFIT_METRICS_SHUTDOWN_DRAIN_S`
# segundos antes de cancelar lo que quede. K8s grace period típico es 30s,
# así que default 5s deja margen al resto de cleanup (LangGraph, workers).
METRICS_SHUTDOWN_DRAIN_S    = _env_int  ("MEALFIT_METRICS_SHUTDOWN_DRAIN_S",    5)

# --- P1-Q9: Drain bounded de OTROS executors críticos en SIGTERM ---
# Antes solo el `_METRICS_EXECUTOR` tenía drain; los demás hacían
# `cancel_futures=True` abrupto, lo que cancela tareas en cola pero NO
# puede matar threads en vuelo (Python no permite interrupt de thread).
# Bajo rolling deploy con jobs en flight, los sockets/connections quedaban
# half-closed visibles al cliente. Ahora cada executor drena con deadline
# antes del cancel forzoso. Suma total ≤ 18s, bien dentro del K8s grace
# period típico (30s).
#
# - Fact-check: 8s. Tools clínicas tardan 5-15s; 8s permite que la mayoría
#   complete y cierre socket HTTP del provider gracefully.
# - DB: 3s. DB ops son 50-300ms; 3s acomoda ~10 secuencialmente.
# - Progress CB: 2s. Callbacks son rápidos (10ms-2s); preservar último
#   burst de eventos al cliente SSE antes del exit.
FACT_CHECK_SHUTDOWN_DRAIN_S = _env_int  ("MEALFIT_FACT_CHECK_SHUTDOWN_DRAIN_S", 8)
DB_SHUTDOWN_DRAIN_S         = _env_int  ("MEALFIT_DB_SHUTDOWN_DRAIN_S",         3)
PROGRESS_CB_SHUTDOWN_DRAIN_S= _env_int  ("MEALFIT_PROGRESS_CB_SHUTDOWN_DRAIN_S",2)


def _log_active_knobs():
    """P1-NEW-2: Loguea una sola vez los knobs efectivos al import.

    Útil para confirmar overrides de producción (`MEALFIT_*` env vars) en los
    logs de startup del worker. Si un knob fue sobre-escrito vía env var, el
    valor efectivo aparecerá aquí — sin tener que SSH-ear al pod a leer env.
    """
    knobs = {
        "LLM_MAX_CONCURRENT": LLM_MAX_CONCURRENT,
        "LLM_REDIS_LOCK_TIMEOUT_S": LLM_REDIS_LOCK_TIMEOUT_S,
        "LLM_MAX_WAIT_S": LLM_MAX_WAIT_S,
        "LLM_PER_USER_ENABLED": LLM_PER_USER_ENABLED,
        "LLM_MAX_PER_USER": LLM_MAX_PER_USER,
        "LLM_USER_LOCK_TIMEOUT_S": LLM_USER_LOCK_TIMEOUT_S,
        "LLM_USER_MAX_WAIT_S": LLM_USER_MAX_WAIT_S,
        "CB_FAILURE_THRESHOLD": CB_FAILURE_THRESHOLD,
        "CB_RESET_TIMEOUT_S": CB_RESET_TIMEOUT_S,
        "CB_LOCAL_HEALTH_TTL_S": CB_LOCAL_HEALTH_TTL_S,
        "HEDGE_AFTER_BASE_S": HEDGE_AFTER_BASE_S,
        "HARD_CEILING_S": HARD_CEILING_S,
        "MAX_ATTEMPTS": MAX_ATTEMPTS,
        "MIN_RETRY_BUDGET_S": MIN_RETRY_BUDGET_S,
        "GLOBAL_PIPELINE_TIMEOUT_S": GLOBAL_PIPELINE_TIMEOUT_S,
        "RETRY_SAFETY_MARGIN_S": RETRY_SAFETY_MARGIN_S,
        "RETRY_HEDGE_BUDGET_DELTA_S": RETRY_HEDGE_BUDGET_DELTA_S,
        "FACT_CHECK_TOOL_TIMEOUT_S": FACT_CHECK_TOOL_TIMEOUT_S,
        "CRITIQUE_FIX_TIMEOUT_S": CRITIQUE_FIX_TIMEOUT_S,
        "PROGRESS_CB_MAX_PENDING": PROGRESS_CB_MAX_PENDING,
        "PROGRESS_CB_TIMEOUT_S": PROGRESS_CB_TIMEOUT_S,
        "LLM_CACHE_TTL_S": LLM_CACHE_TTL_S,
        "METRICS_SHUTDOWN_DRAIN_S": METRICS_SHUTDOWN_DRAIN_S,
        "FACT_CHECK_SHUTDOWN_DRAIN_S": FACT_CHECK_SHUTDOWN_DRAIN_S,
        "DB_SHUTDOWN_DRAIN_S": DB_SHUTDOWN_DRAIN_S,
        "PROGRESS_CB_SHUTDOWN_DRAIN_S": PROGRESS_CB_SHUTDOWN_DRAIN_S,
        "SYNC_WRAPPER_MAX_WORKERS": SYNC_WRAPPER_MAX_WORKERS,
        "SYNC_WRAPPER_STRICT_MODE": SYNC_WRAPPER_STRICT_MODE,
    }
    logging.getLogger(__name__).info(
        "[KNOBS] graph_orchestrator activos: "
        + ", ".join(f"{k}={v}" for k, v in knobs.items())
    )


_log_active_knobs()


class DistributedLLMSemaphore:
    """Semáforo distribuido usando Redis Sorted Sets para backpressure global.
    Aplica límite de concurrencia a través de múltiples workers Gunicorn/Uvicorn.
    Fallback a threading.Semaphore local si Redis no está disponible."""
    def __init__(self, max_concurrent=4, timeout_seconds=120, max_wait_seconds=90):
        self.max_concurrent = max_concurrent
        self.timeout = timeout_seconds
        # P0-B: Cap de tiempo máximo que el caller espera por un slot Redis
        # antes de degradar al `_local_semaphore`. Sin este bound, el loop
        # `while not acquired: sleep(1)` puede esperar indefinidamente si la
        # cola Redis se mantiene saturada (workers colgados, deadlock
        # distribuido), excediendo el GLOBAL_TIMEOUT del pipeline (600s) y
        # dejando sockets HTTP del cliente colgados.
        # Default 90s: deja margen para un retry completo del pipeline
        # (~180s budget) y aún así es <<600s del global timeout.
        self.max_wait_seconds = max_wait_seconds
        self.key = "semaphore:llm_global"
        # P1-X1: `_local_semaphore` es un `threading.Semaphore` proceso-wide
        # — único cap real cross-thread / cross-loop bajo Redis-down.
        #
        # Antes (P0-6) convivía con un mapa per-loop de `asyncio.Semaphore`
        # (`_async_semaphores: WeakKeyDictionary`). El motivo original era que
        # un `asyncio.Semaphore` está bound al loop donde se creó, así que
        # BackgroundTasks de FastAPI o `_SYNC_WRAPPER_EXECUTOR` con su propio
        # loop no podían reusar el del loop principal. Pero ese arreglo creó
        # otro problema: cada loop tenía su PROPIO `Semaphore(max_concurrent)`,
        # así que con N loops vivos la concurrencia efectiva pasaba de
        # `max_concurrent` a `max_concurrent × N`. Justo bajo outage de Redis
        # (peor momento para sobrecargar al proveedor LLM), el cap se evaporaba.
        #
        # Solución P1-X1: el path async adquiere este mismo `threading.Semaphore`
        # vía `_alocal_acquire` con busy-poll no bloqueante (`acquire(blocking=False)`
        # + `asyncio.sleep(0.05)`). `threading.Semaphore` es thread-safe y no
        # bound a loop, así que cualquier loop/thread comparte el cap real.
        self._local_semaphore = threading.Semaphore(max_concurrent)

    @asynccontextmanager
    async def _alocal_acquire(self):
        """P1-X1: adquiere `_local_semaphore` desde código async sin bloquear
        el event loop ni ocupar threads del default executor.

        Estrategia: busy-poll con `acquire(blocking=False)`. Si no hay slot,
        `asyncio.sleep(0.05)` cede el loop sin retener ningún thread. Latencia
        adicional de wake-up: hasta ~50ms (irrelevante en pipelines LLM de
        60-90s; bajo Redis-down con saturación, trade-off aceptable).

        Cancellation-safe: si el caller es cancelado durante un `asyncio.sleep`,
        la última `acquire(blocking=False)` retornó False (no se retuvo slot)
        y no hay leak. Si fue cancelado tras adquirir, el `finally` libera.
        """
        while not self._local_semaphore.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            self._local_semaphore.release()

    @contextmanager
    def acquire(self):
        # P0-4: Prevención de congelamiento del event loop
        try:
            loop = asyncio.get_running_loop()
            is_running = loop.is_running()
        except RuntimeError:
            is_running = False
            
        if is_running:
            raise RuntimeError(
                "P0-4: Bloqueo de event loop detectado. "
                "DistributedLLMSemaphore.acquire() sync llamado desde un event loop activo. "
                "Usa .aacquire() o las variantes async (.ainvoke, .agenerate) en su lugar."
            )

        from cache_manager import redis_client
        if not redis_client:
            with self._local_semaphore:
                yield
            return

        req_id = str(uuid.uuid4())
        acquired = False
        # P0-B: Bound de espera. Si Redis está vivo pero la cola está saturada
        # (workers hung, deadlock distribuido), evita esperar más allá de
        # `max_wait_seconds` y degrada al semáforo local. Usamos `time.monotonic()`
        # —no `time.time()`— para no ser sensibles a saltos del reloj del SO.
        wait_started = time.monotonic()
        try:
            while not acquired:
                try:
                    now = time.time()
                    # 1. Limpiar locks expirados (evita leaks por workers caídos)
                    redis_client.zremrangebyscore(self.key, "-inf", now - self.timeout)

                    # 2. Intentar adquirir agregando a la cola
                    redis_client.zadd(self.key, {req_id: now})

                    # 3. Verificar posición en la cola
                    rank = redis_client.zrank(self.key, req_id)

                    if rank is not None and rank < self.max_concurrent:
                        acquired = True
                        break
                    else:
                        # Cola llena, retirarse y esperar
                        redis_client.zrem(self.key, req_id)
                        # P0-B: chequeo de bound ANTES del sleep. Si excedimos
                        # el max_wait, ya estamos limpios (acabamos de zrem) y
                        # podemos degradar al local sin estado pendiente en Redis.
                        elapsed = time.monotonic() - wait_started
                        if elapsed >= self.max_wait_seconds:
                            logger.warning(
                                f"P0-B: cola Redis del LLM_SEMAPHORE saturada "
                                f"({elapsed:.1f}s ≥ {self.max_wait_seconds}s sin obtener slot). "
                                "Degradando a local semaphore para evitar deadlock distribuido."
                            )
                            with self._local_semaphore:
                                yield
                            return
                        time.sleep(1.0)
                except Exception as e:
                    logger.warning(f"Redis semaphore error: {e}. Fallback to local semaphore.")
                    with self._local_semaphore:
                        yield
                    return

            yield

        finally:
            if acquired:
                from cache_manager import redis_client
                if redis_client:
                    try:
                        redis_client.zrem(self.key, req_id)
                    except Exception:
                        pass

    @asynccontextmanager
    async def aacquire(self):
        from cache_manager import redis_async_client
        if not redis_async_client:
            async with self._alocal_acquire():
                yield
            return

        req_id = str(uuid.uuid4())
        acquired = False
        # P0-B: Bound de espera (idéntico al sync `acquire`). Cierra la ventana
        # donde el await colgaba indefinidamente esperando un slot Redis cuando
        # otros workers están hung con su contador local de salud todavía OK.
        wait_started = time.monotonic()
        try:
            while not acquired:
                try:
                    now = time.time()
                    # 1. Limpiar locks expirados (evita leaks por workers caídos)
                    await redis_async_client.zremrangebyscore(self.key, "-inf", now - self.timeout)

                    # 2. Intentar adquirir agregando a la cola
                    await redis_async_client.zadd(self.key, {req_id: now})

                    # 3. Verificar posición en la cola
                    rank = await redis_async_client.zrank(self.key, req_id)

                    if rank is not None and rank < self.max_concurrent:
                        acquired = True
                        break
                    else:
                        # Cola llena, retirarse y esperar
                        await redis_async_client.zrem(self.key, req_id)
                        # P0-B: chequeo de bound ANTES del sleep. Después del
                        # zrem ya no estamos en la cola Redis, así que el
                        # fallback a local no deja estado pendiente.
                        elapsed = time.monotonic() - wait_started
                        if elapsed >= self.max_wait_seconds:
                            logger.warning(
                                f"P0-B: cola Redis del LLM_SEMAPHORE saturada "
                                f"({elapsed:.1f}s ≥ {self.max_wait_seconds}s sin obtener slot). "
                                "Degradando a local async semaphore para evitar deadlock distribuido."
                            )
                            async with self._alocal_acquire():
                                yield
                            return
                        await asyncio.sleep(1.0)
                except Exception as e:
                    logger.warning(f"Redis semaphore aasync error: {e}. Fallback to local async semaphore.")
                    async with self._alocal_acquire():
                        yield
                    return

            yield

        finally:
            if acquired:
                from cache_manager import redis_async_client
                if redis_async_client:
                    try:
                        await redis_async_client.zrem(self.key, req_id)
                    except Exception:
                        pass


# P1-NEW-2: knobs configurables vía env (`MEALFIT_LLM_*`).
LLM_SEMAPHORE = DistributedLLMSemaphore(
    max_concurrent=LLM_MAX_CONCURRENT,
    timeout_seconds=LLM_REDIS_LOCK_TIMEOUT_S,
    max_wait_seconds=LLM_MAX_WAIT_S,
)


# ============================================================
# P1-NEW-1: Semáforo per-user (rate limit por tenant)
# ------------------------------------------------------------
# Capa SOBRE el `LLM_SEMAPHORE` global. Adquirir SIEMPRE en orden:
#   1. per-user (este semáforo)
#   2. global  (`LLM_SEMAPHORE`)
# Razonamiento del orden: si invertimos, un usuario que excede su cuota
# tomaría slots globales y luego se bloquearía esperando su slot per-user
# — desperdicio de slots globales que otros usuarios podrían usar. Adquirir
# per-user primero garantiza que el usuario "se autocontiene" antes de
# competir por la cuota compartida.
#
# Bypass automático cuando:
#   - el knob `MEALFIT_LLM_PER_USER_ENABLED` está en false
#   - `user_id` es None (cron, batch, llamada interna sin tenant atribuible)
#   - `user_id == "guest"` (usuarios anónimos comparten cuota global, no se
#     les aplica rate limit per-tenant — se puede invertir vía override)
# ============================================================
class DistributedPerUserSemaphore:
    """Semáforo distribuido per-user. Misma estructura Redis sorted-set que
    `DistributedLLMSemaphore` pero con KEY por user_id. Soporta fallback a
    semáforo local (threading / asyncio per-loop) cuando Redis no está
    disponible — la limitación per-user sigue funcionando dentro del worker
    pero se pierde la coordinación cross-worker, lo cual es aceptable como
    degradación graceful.
    """

    def __init__(self, *, max_per_user: int, lock_timeout_s: int,
                 max_wait_s: int, enabled: bool):
        self.max_per_user = int(max_per_user)
        self.timeout = int(lock_timeout_s)
        self.max_wait_s = int(max_wait_s)
        self.enabled = bool(enabled)
        self._key_prefix = "semaphore:llm:user"
        # P1-X1: dict[user_id → threading.Semaphore] proceso-wide. Único cap
        # real cross-thread / cross-loop bajo Redis-down. Antes existía además
        # un mapa per-loop de `asyncio.Semaphore` (`_local_async`) que fragaba
        # el cap cuando había múltiples loops vivos en el proceso (p.ej.
        # `_SYNC_WRAPPER_EXECUTOR` con N workers + BackgroundTasks de FastAPI):
        # cada loop tenía su propio Semaphore(max_per_user) por usuario, así
        # que la concurrencia efectiva por usuario era max_per_user × N en vez
        # de max_per_user. Ahora el path async adquiere el mismo
        # `threading.Semaphore` vía `_alocal_acquire` (busy-poll), y el cap
        # se respeta cross-loop. El lock protege solo INSERTS al dict
        # (microsegundos); el wait sobre cada Semaphore individual no toma
        # este lock — es no bloqueante (`acquire(blocking=False)`).
        self._local_sync: dict = {}
        self._local_sync_lock = threading.Lock()

    def _bypass(self, user_id) -> bool:
        if not self.enabled:
            return True
        if user_id is None:
            return True
        if not isinstance(user_id, str):
            user_id = str(user_id)
        return user_id == "" or user_id == "guest"

    def _redis_key(self, user_id: str) -> str:
        return f"{self._key_prefix}:{user_id}"

    def _get_local_sync(self, user_id: str) -> threading.Semaphore:
        with self._local_sync_lock:
            sem = self._local_sync.get(user_id)
            if sem is None:
                sem = threading.Semaphore(self.max_per_user)
                self._local_sync[user_id] = sem
            return sem

    @asynccontextmanager
    async def _alocal_acquire(self, user_id: str):
        """P1-X1: adquiere el `threading.Semaphore` per-user (proceso-wide)
        desde código async. Mismo patrón que `DistributedLLMSemaphore._alocal_acquire`:
        busy-poll con `acquire(blocking=False)` + `asyncio.sleep(0.05)` — sin
        bloquear el event loop ni ocupar threads del default executor.

        Reemplaza el viejo `_get_local_async` que devolvía `asyncio.Semaphore`
        per-loop (P0-6-style). Ese arreglo fragaba el cap real cuando había
        múltiples loops vivos en el proceso (ver comentario del __init__).
        """
        sem = self._get_local_sync(user_id)
        while not sem.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            sem.release()

    @contextmanager
    def acquire(self, user_id):
        """Sync acquire. Bypass si user_id no atribuible o knob desactivado."""
        if self._bypass(user_id):
            yield
            return

        # Validación de uso desde event loop activo (mismo guard que P0-4 global)
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "P1-NEW-1: PerUserSem.acquire() sync llamado desde event loop activo. "
                    "Usa .aacquire(user_id) o las variantes async."
                )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise

        from cache_manager import redis_client
        if not redis_client:
            with self._get_local_sync(user_id):
                yield
            return

        key = self._redis_key(user_id)
        req_id = str(uuid.uuid4())
        acquired = False
        wait_started = time.monotonic()
        try:
            while not acquired:
                try:
                    now = time.time()
                    redis_client.zremrangebyscore(key, "-inf", now - self.timeout)
                    redis_client.zadd(key, {req_id: now})
                    rank = redis_client.zrank(key, req_id)
                    if rank is not None and rank < self.max_per_user:
                        acquired = True
                        break
                    redis_client.zrem(key, req_id)
                    elapsed = time.monotonic() - wait_started
                    if elapsed >= self.max_wait_s:
                        logger.warning(
                            f"P1-NEW-1: PerUserSem saturado para user={user_id!r} "
                            f"({elapsed:.1f}s ≥ {self.max_wait_s}s). Degradando a local."
                        )
                        with self._get_local_sync(user_id):
                            yield
                        return
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"PerUserSem Redis error: {e}. Fallback local.")
                    with self._get_local_sync(user_id):
                        yield
                    return
            yield
        finally:
            if acquired:
                try:
                    redis_client.zrem(key, req_id)
                except Exception:
                    pass

    @asynccontextmanager
    async def aacquire(self, user_id):
        """Async acquire per-user. Bypass si knob desactivado o user_id no atribuible."""
        if self._bypass(user_id):
            yield
            return

        from cache_manager import redis_async_client
        if not redis_async_client:
            async with self._alocal_acquire(user_id):
                yield
            return

        key = self._redis_key(user_id)
        req_id = str(uuid.uuid4())
        acquired = False
        wait_started = time.monotonic()
        try:
            while not acquired:
                try:
                    now = time.time()
                    await redis_async_client.zremrangebyscore(key, "-inf", now - self.timeout)
                    await redis_async_client.zadd(key, {req_id: now})
                    rank = await redis_async_client.zrank(key, req_id)
                    if rank is not None and rank < self.max_per_user:
                        acquired = True
                        break
                    await redis_async_client.zrem(key, req_id)
                    elapsed = time.monotonic() - wait_started
                    if elapsed >= self.max_wait_s:
                        logger.warning(
                            f"P1-NEW-1: PerUserSem saturado para user={user_id!r} "
                            f"({elapsed:.1f}s ≥ {self.max_wait_s}s). Degradando a local async."
                        )
                        async with self._alocal_acquire(user_id):
                            yield
                        return
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"PerUserSem Redis async error: {e}. Fallback local async.")
                    async with self._alocal_acquire(user_id):
                        yield
                    return
            yield
        finally:
            if acquired:
                try:
                    await redis_async_client.zrem(key, req_id)
                except Exception:
                    pass


# P1-NEW-1: knobs vía env (`MEALFIT_LLM_PER_USER_*`).
PER_USER_LLM_SEMAPHORE = DistributedPerUserSemaphore(
    max_per_user=LLM_MAX_PER_USER,
    lock_timeout_s=LLM_USER_LOCK_TIMEOUT_S,
    max_wait_s=LLM_USER_MAX_WAIT_S,
    enabled=LLM_PER_USER_ENABLED,
)


@contextmanager
def acquire_user_and_global(user_id):
    """P1-NEW-1: composición sync — adquiere per-user antes que global.

    El orden importa: per-user PRIMERO. Si invertimos, un usuario saturando
    su cuota tomaría slots globales y luego bloquearía esperando per-user,
    desperdiciando slots compartidos.
    """
    with PER_USER_LLM_SEMAPHORE.acquire(user_id):
        with LLM_SEMAPHORE.acquire():
            yield


@asynccontextmanager
async def aacquire_user_and_global(user_id):
    """P1-NEW-1: composición async, mismo orden que la versión sync."""
    async with PER_USER_LLM_SEMAPHORE.aacquire(user_id):
        async with LLM_SEMAPHORE.aacquire():
            yield


class ChatGoogleGenerativeAI(_ChatGoogleGenerativeAI):
    """Wrapper para aplicar backpressure transparente a TODAS las llamadas
    LLM en LangGraph (síncrono y asíncrono).

    P1-NEW-1: ahora usa el helper compuesto `acquire_user_and_global` /
    `aacquire_user_and_global` que aplica primero el rate limit per-user
    (vía `PER_USER_LLM_SEMAPHORE`) y después el slot global. El `user_id`
    se lee del `user_id_var` (ContextVar) — `arun_plan_pipeline` lo setea
    al entrar; otros callers (cron, batch) sin contextvar configurado
    obtienen `None` → bypass del rate limit per-user (preserva comportamiento
    previo). Para callers que quieran rate limit explícito sin pasar por
    `arun_plan_pipeline`, basta con `user_id_var.set(uid)` antes del .ainvoke.
    """
    def invoke(self, *args, **kwargs):
        with acquire_user_and_global(user_id_var.get()):
            return super().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        with acquire_user_and_global(user_id_var.get()):
            yield from super().stream(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with acquire_user_and_global(user_id_var.get()):
            return super().generate(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        async with aacquire_user_and_global(user_id_var.get()):
            return await super().ainvoke(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        async with aacquire_user_and_global(user_id_var.get()):
            async for chunk in super().astream(*args, **kwargs):
                yield chunk

    async def agenerate(self, *args, **kwargs):
        async with aacquire_user_and_global(user_id_var.get()):
            return await super().agenerate(*args, **kwargs)

# P1-10: Imports de stdlib y third-party subidos a nivel módulo para evitar
# reimport repetido en hot paths (workers paralelos, retries, loops por meal).
# `uuid` ya importado arriba (línea ~20); no reimportar.
import concurrent.futures
from datetime import datetime, timezone, timedelta
import random
import re as _re
import contextvars
import builtins
import copy
import hashlib
import unicodedata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Mejora 8: ContextVar para Distributed Tracing
request_id_var = contextvars.ContextVar("request_id", default="SYS")
# P1-NEW-1: ContextVar para rate-limit per-user. Lo setea `arun_plan_pipeline`
# al entrar y se propaga automáticamente a tasks hijas (asyncio.Task hereda
# el contexto al crearse) y a callbacks despachados vía `asyncio.to_thread`
# (`run_in_executor` con default executor preserva contextvars desde 3.7+).
# Default `None` → bypass del rate limit (preservar comportamiento de cron
# jobs, batch y otros callers que no setean el var).
user_id_var = contextvars.ContextVar("user_id", default=None)

# [P1-6] Mapeo emoji → log level. La convención del codebase (verificada con
# un censo de los 209 sitios `print(` del módulo) usa consistentemente
# emojis como señal implícita de severidad: 🚨/❌/🛑/🚫 para fallas,
# ⚠️/🛡/⏰/⏳/🟠 para degradación, el resto (✅/📊/🔄/🔬/🔗/⚡/🔪/etc.)
# para info. Centralizar el mapping en `custom_print` permite migrar TODOS
# los call sites a `logger.X` con UN solo cambio en lugar de 209 ediciones
# riesgosas — preserva la semántica de severidad existente y añade
# timestamp + level + module name vía el formatter de `app.py:23`.
#
# Ambas variantes (con y sin Variation Selector-16 `️`) están listadas
# porque diferentes editores / inputs producen una u otra para el mismo
# emoji visual. `str.startswith(tuple)` matchea cualquiera.
_LOG_LEVEL_PREFIXES_ERROR = (
    "🚨", "❌", "🛑", "🚫",
)
_LOG_LEVEL_PREFIXES_WARNING = (
    "⚠️", "⚠",
    "🛡️", "🛡",
    "⏰", "⏳", "🟠",
)


def custom_print(*args, **kwargs):
    """[P1-6] Wrapper que despacha a `logger` según severidad inferida del
    prefijo emoji. ANTES escribía a `builtins.print(msg)` → stdout sin nivel
    ni timestamp, perdiendo filtrado en Grafana/Loki y la correlación de
    severidad. Mantiene el prefijo `[request_id]` que ya añadía para tracear
    requests; el formatter del root logger añade timestamp + level + module.
    """
    req_id = request_id_var.get()
    msg = " ".join(str(a) for a in args)
    msg = f"[{req_id}] {msg}"

    # Detectar emoji severidad después del prefijo `[req_id] ` y de
    # cualquier whitespace inicial (los banners ASCII suelen empezar con `\n`).
    body = msg.split("] ", 1)[1] if "] " in msg else msg
    body_stripped = body.lstrip()

    if body_stripped.startswith(_LOG_LEVEL_PREFIXES_ERROR):
        logger.error(msg)
    elif body_stripped.startswith(_LOG_LEVEL_PREFIXES_WARNING):
        logger.warning(msg)
    else:
        logger.info(msg)


# Rebind del builtin: TODAS las llamadas `print(...)` en este módulo (209 al
# momento del fix P1-6) se enrutan vía `custom_print` → `logger.X`. Los
# call sites no requieren cambio. Las llamadas directas a `logger.warning/
# info/error/critical` que ya existían en el módulo se preservan tal cual
# (no pasan por `custom_print`); ambos paths convergen en el mismo root logger.
print = custom_print
# NOTA: NO importar 'from agent import ...' a nivel de módulo → causa import circular
# (app → agent → tools → graph_orchestrator → agent). Se usa lazy import donde se necesite.
from cpu_tasks import _validar_repeticiones_cpu_bound, _normalize_meal_name
from constants import (
    normalize_ingredient_for_tracking, strip_accents,
    TECHNIQUE_FAMILIES, ALL_TECHNIQUES, TECH_TO_FAMILY, SUPPLEMENT_NAMES,
    # P1-10: subidos para evitar reimport en hot paths
    PLAN_CHUNK_SIZE, validate_ingredients_against_pantry, safe_fromisoformat,
    # P1-11: vocabularios canónicos centralizados (técnicas + stopwords).
    # Ambos se usan en assemble_plan_node y _calculate_complexity_score; antes
    # `RECIPE_INGREDIENT_STOPWORDS` vivía hardcodeado en el orquestador con
    # riesgo de divergir de COMPLEX_TECHNIQUE_KEYWORDS al añadir vocabulario.
    COMPLEX_TECHNIQUE_KEYWORDS, RECIPE_INGREDIENT_STOPWORDS,
)
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from db import get_recent_techniques, get_recent_meals_from_plans, check_meal_plan_generated_today, search_user_facts, search_visual_diary, get_user_facts_by_metadata
from nutrition_calculator import get_nutrition_targets

# P1-10: `threading` ya importado al inicio del módulo (línea ~16); no reimportar.
logger = logging.getLogger(__name__)

from cache_manager import redis_client, redis_async_client
from db_core import execute_sql_query, execute_sql_write, aexecute_sql_query, aexecute_sql_write

class LLMCircuitBreaker:
    """Circuit breaker distribuido usando Redis INCR atómico.
    Seguro para multi-worker (Gunicorn/uvicorn --workers N).
    Fallback a DB si Redis no está disponible.

    P0-3: la cache local del estado "healthy" usaba 10s hardcodeados, lo cual
    abría una ventana de carrera entre N workers paralelos: el primer fallo
    invalidaba el flag local pero los workers que ya pasaron `acan_proceed()`
    seguían invocando contra un proveedor saturado. Ahora:
      - `local_health_ttl` es parametrizable (default 1.0s — granularidad
        adecuada para apps con paralelismo alto).
      - `_failure_propagated_at` registra el instante de la última falla, y
        `acan_proceed()` lo respeta como signal local fresca antes de tocar
        Redis (invalidación instantánea entre workers del mismo proceso).
      - Método sync `_is_locally_unhealthy_fresh()` permite double-check
        ultra-rápido (~1µs) en hot paths antes de cada `ainvoke()`.

    P1-Q1: el lock que protege el path de fallback DB (read-modify-write de
    `app_kv_store`) era un `asyncio.Lock` lazy-init. `asyncio.Lock` queda
    bound al primer event loop que lo adquiere; bajo `_SYNC_WRAPPER_EXECUTOR`
    con varios workers (cada uno con su propio loop vía `asyncio.run`) +
    Redis-down (degrada al fallback DB), llamadas desde un loop ≠ primero
    lanzaban `RuntimeError: Lock attached to different event loop`. La
    excepción NO estaba envuelta en try/except dentro de los métodos del CB,
    así que propagaba al caller (`invoke_planner` etc.) y disparaba retries
    de tenacity por motivo equivocado — ruido en logs y falsos failures.
    Mismo problema que P1-X1 cerró para `DistributedLLMSemaphore`.

    Solución: reusar `self._lock` (`threading.Lock`, ya existente para el
    path sync) desde código async vía busy-poll no bloqueante
    (`acquire(blocking=False)` + `asyncio.sleep(0.05)`). `threading.Lock`
    no está bound a loop, así que cualquier loop/thread comparte el cap real;
    además sync y async ahora compiten por el MISMO mutex, eliminando la
    ventana de race donde un sync `with self._lock` y un async path corrían
    en paralelo sobre el mismo estado DB.
    """
    def __init__(self, failure_threshold=3, reset_timeout=30, local_health_ttl=1.0,
                 model_name: str | None = None):
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._local_health_ttl = float(local_health_ttl)  # P0-3: configurable, antes 10s hardcoded
        # P1-Q3: namespacing de keys por modelo. Si `model_name` es None, las
        # keys quedan como antes ("cb:llm:failures" / "cb:llm:open" /
        # "llm_circuit_breaker") — preservando compatibilidad para callers
        # legacy que no atribuyen modelo. Si se especifica, las keys se
        # sufijan con `:<model>` permitiendo aislar el estado de cada modelo
        # en Redis y en `app_kv_store`.
        self.model_name = model_name
        _key_suffix = f":{model_name}" if model_name else ""
        self._failures_key = f"cb:llm:failures{_key_suffix}"
        self._open_key = f"cb:llm:open{_key_suffix}"
        # Key de DB fallback (`app_kv_store`). El sufijo solo se aplica si hay
        # model_name; el legacy "llm_circuit_breaker" se mantiene para el CB
        # global, evitando migración de datos existentes.
        self._db_kv_key = f"llm_circuit_breaker{_key_suffix}"
        # P1-Q1: `_lock` es un `threading.Lock` proceso-wide compartido por
        # los paths sync (`with self._lock`) y async (`async with self._alock_acquire()`).
        # Antes el async usaba un `asyncio.Lock` separado y lazy-init, que se
        # asociaba al primer loop en adquirirlo y rompía bajo multi-loop. Ver
        # docstring de la clase para detalles.
        self._lock = threading.Lock()  # Fallback para modo sin-Redis (sync + async)
        self._local_state_lock = threading.Lock() # P0-3: Lock unificado seguro (sync/async) para variables locales
        self._local_healthy = True     # Optimización para no golpear la DB si está sano
        self._last_db_check = 0        # TTL para el estado DB local
        self._failure_propagated_at = 0.0  # P0-3: timestamp de la última falla (cross-worker fast invalidation)

    @asynccontextmanager
    async def _alock_acquire(self):
        """P1-Q1: adquiere `self._lock` (threading.Lock) desde código async sin
        bloquear el event loop ni quedar bound a un loop específico.

        Mismo patrón que `DistributedLLMSemaphore._alocal_acquire` (P1-X1):
        busy-poll con `acquire(blocking=False)` + `asyncio.sleep(0.05)`.
        Latencia de wake-up: hasta ~50ms — irrelevante en este path porque
        solo se ejerce bajo Redis-down (fallback DB es 5-50ms por query) y
        la sección crítica (read-modify-write del KV store) es <100ms.

        Cancellation-safe: si el caller es cancelado durante un `asyncio.sleep`,
        la última `acquire(blocking=False)` retornó False (no se retuvo el
        lock) y no hay leak. Si fue cancelado tras adquirir, el `finally`
        libera.

        Coexiste con sync `with self._lock` — ambos compiten por el mismo
        mutex, eliminando la posible race entre sync `record_failure` (de un
        thread) y async `arecord_failure` (de un loop) corriendo simultáneos
        sobre el mismo estado DB.
        """
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            self._lock.release()

    def _is_locally_unhealthy_fresh(self) -> bool:
        """P0-3: Double-check ultra-rápido sin I/O (~1µs).

        Útil para llamar JUSTO ANTES de `ainvoke()` o dentro de loops de agent
        para abortar una llamada cuando otro worker paralelo ya registró fallo.
        Retorna True si el flag local dice unhealthy Y la falla es reciente
        (dentro de la ventana del reset_timeout).
        """
        with self._local_state_lock:
            healthy = self._local_healthy
            propagated_at = self._failure_propagated_at
        if healthy:
            return False
        # Si la falla local fue muy antigua, no confiamos en el flag — dejar que
        # `acan_proceed()` re-chequee Redis. Aquí solo cubrimos la ventana corta.
        return (time.time() - propagated_at) < self.reset_timeout

    def record_failure(self):
        with self._local_state_lock:
            self._local_healthy = False
            self._failure_propagated_at = time.time()
        if redis_client:
            try:
                failures = redis_client.incr(self._failures_key)
                redis_client.expire(self._failures_key, self.reset_timeout)
                if failures >= self.threshold:
                    redis_client.set(self._open_key, "1", ex=self.reset_timeout)
                return
            except Exception as e:
                logger.warning(f"Redis CB write error: {e}")
        # Fallback DB (best-effort, no atómico pero funcional)
        with self._lock:
            try:
                state = self._get_db_state()
                state["failures"] = state.get("failures", 0) + 1
                state["last_failure"] = time.time()
                if state["failures"] >= self.threshold:
                    state["is_open"] = True
                self._save_db_state(state)
            except Exception as e:
                logger.warning(f"DB CB write error: {e}")

    def record_success(self):
        with self._local_state_lock:
            is_healthy = self._local_healthy
            last_check = self._last_db_check
        if is_healthy and (time.time() - last_check) < self._local_health_ttl:
            return  # Debounce: si localmente creemos que está sano y chequeamos hace poco, no golpear DB

        if redis_client:
            try:
                redis_client.delete(self._failures_key, self._open_key)
                with self._local_state_lock:
                    self._local_healthy = True
                    self._last_db_check = time.time()
                return
            except Exception as e:
                logger.warning(f"Redis CB reset error: {e}")
        with self._lock:
            try:
                state = self._get_db_state()
                if state.get("failures", 0) > 0 or state.get("is_open", False):
                    self._save_db_state({"failures": 0, "last_failure": 0, "is_open": False})
                with self._local_state_lock:
                    self._local_healthy = True
                    self._last_db_check = time.time()
            except Exception as e:
                logger.warning(f"DB CB reset error: {e}")

    def can_proceed(self) -> bool:
        with self._local_state_lock:
            is_healthy = self._local_healthy
            last_check = self._last_db_check
            propagated_at = self._failure_propagated_at
        # P0-3: short-circuit cuando el fallo local es fresco. Antes este path
        # iba a Redis, dando una ventana de race entre el momento en que un
        # worker registra el fallo y la propagación a Redis observable por otros.
        if not is_healthy and (time.time() - propagated_at) < self.reset_timeout:
            return False
        if is_healthy and (time.time() - last_check) < self._local_health_ttl:
            return True # Asumimos sano si se verificó recientemente

        if redis_client:
            try:
                is_open = redis_client.get(self._open_key)
                if not is_open:
                    with self._local_state_lock:
                        self._local_healthy = True
                        self._last_db_check = time.time()
                return not is_open  # Si la key expiró (reset_timeout), redis.get = None → True
            except Exception as e:
                logger.warning(f"Redis CB read error: {e}")
        with self._lock:
            try:
                state = self._get_db_state()
                is_open = state.get("is_open", False)
                if not is_open:
                    with self._local_state_lock:
                        self._local_healthy = True
                        self._last_db_check = time.time()
                    return True
                if time.time() - state.get("last_failure", 0) > self.reset_timeout:
                    return True
                return False
            except Exception:
                return True  # Fail-open si todo falla

    def _get_db_state(self):
        try:
            # P1-Q3: key namespaced por modelo vía `self._db_kv_key`
            res = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (self._db_kv_key,), fetch_one=True)
            if res:
                return res["value"] if isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception:
            pass
        return {"failures": 0, "last_failure": 0, "is_open": False}

    def _save_db_state(self, state):
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (self._db_kv_key, json.dumps(state))  # P1-Q3
        )

    async def arecord_failure(self):
        with self._local_state_lock:
            self._local_healthy = False
            self._failure_propagated_at = time.time()
        if redis_async_client:
            try:
                failures = await redis_async_client.incr(self._failures_key)
                await redis_async_client.expire(self._failures_key, self.reset_timeout)
                if failures >= self.threshold:
                    await redis_async_client.set(self._open_key, "1", ex=self.reset_timeout)
                return
            except Exception as e:
                logger.warning(f"Redis async CB write error: {e}")
        async with self._alock_acquire():
            try:
                state = await self._aget_db_state()
                state["failures"] = state.get("failures", 0) + 1
                state["last_failure"] = time.time()
                if state["failures"] >= self.threshold:
                    state["is_open"] = True
                await self._asave_db_state(state)
            except Exception as e:
                logger.warning(f"DB async CB write error: {e}")

    async def arecord_success(self):
        with self._local_state_lock:
            is_healthy = self._local_healthy
            last_check = self._last_db_check
        if is_healthy and (time.time() - last_check) < self._local_health_ttl:
            return  # Debounce

        if redis_async_client:
            try:
                await redis_async_client.delete(self._failures_key, self._open_key)
                with self._local_state_lock:
                    self._local_healthy = True
                    self._last_db_check = time.time()
                return
            except Exception as e:
                logger.warning(f"Redis async CB reset error: {e}")
        async with self._alock_acquire():
            try:
                state = await self._aget_db_state()
                if state.get("failures", 0) > 0 or state.get("is_open", False):
                    await self._asave_db_state({"failures": 0, "last_failure": 0, "is_open": False})
                with self._local_state_lock:
                    self._local_healthy = True
                    self._last_db_check = time.time()
            except Exception as e:
                logger.warning(f"DB async CB reset error: {e}")

    async def acan_proceed(self) -> bool:
        with self._local_state_lock:
            is_healthy = self._local_healthy
            last_check = self._last_db_check
            propagated_at = self._failure_propagated_at
        # P0-3: short-circuit cuando el fallo local es fresco — invalidación
        # instantánea cross-worker dentro del mismo proceso. Cierra la ventana
        # de carrera donde un worker que esperaba el semáforo del LLM seguía
        # invocando aunque otro ya hubiera registrado fallo.
        if not is_healthy and (time.time() - propagated_at) < self.reset_timeout:
            return False
        if is_healthy and (time.time() - last_check) < self._local_health_ttl:
            return True

        if redis_async_client:
            try:
                is_open = await redis_async_client.get(self._open_key)
                if not is_open:
                    with self._local_state_lock:
                        self._local_healthy = True
                        self._last_db_check = time.time()
                return not is_open
            except Exception as e:
                logger.warning(f"Redis async CB read error: {e}")
        async with self._alock_acquire():
            try:
                state = await self._aget_db_state()
                is_open = state.get("is_open", False)
                if not is_open:
                    with self._local_state_lock:
                        self._local_healthy = True
                        self._last_db_check = time.time()
                    return True
                if time.time() - state.get("last_failure", 0) > self.reset_timeout:
                    return True
                return False
            except Exception:
                return True

    async def _aget_db_state(self):
        try:
            # P1-Q3: key namespaced por modelo vía `self._db_kv_key`
            res = await aexecute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (self._db_kv_key,), fetch_one=True)
            if res:
                return res["value"] if isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception:
            pass
        return {"failures": 0, "last_failure": 0, "is_open": False}

    async def _asave_db_state(self, state):
        await aexecute_sql_write(
            "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (self._db_kv_key, json.dumps(state))  # P1-Q3
        )

# P1-NEW-2: knobs configurables vía env (`MEALFIT_CB_*`).
# Singleton "global" sin model_name — usado como fallback cuando el caller
# no atribuye modelo (`_get_circuit_breaker(None)` lo devuelve) y para
# preservar compatibilidad con tools/callers legacy. Sus keys de Redis
# y DB siguen siendo `cb:llm:failures` / `cb:llm:open` / `llm_circuit_breaker`
# (sin sufijo) — ver `LLMCircuitBreaker.__init__` (P1-Q3).
_circuit_breaker = LLMCircuitBreaker(
    failure_threshold=CB_FAILURE_THRESHOLD,
    reset_timeout=CB_RESET_TIMEOUT_S,
    local_health_ttl=CB_LOCAL_HEALTH_TTL_S,
)


# ============================================================
# P1-Q3: Registry de circuit breakers per-modelo.
# ------------------------------------------------------------
# Antes había un único `_circuit_breaker` global compartido por TODAS las
# llamadas LLM (gemini-3-flash-preview + gemini-3.1-pro-preview + cualquier
# otro modelo futuro). Si gemini-pro tenía rate-limit transitorio (común al
# usarlo en perfiles clínicos complejos vía `_route_model`) pero flash
# estaba sano, el CB se abría globalmente y bloqueaba AMBOS modelos —
# cascade failure innecesario que desperdiciaba la capacidad disponible
# del proveedor para los perfiles fáciles.
#
# Ahora cada modelo tiene su propio CB con keys Redis/DB namespaced por
# nombre de modelo (`cb:llm:failures:gemini-3-flash-preview`, etc.). El
# router de modelos (`_route_model`) puede seguir despachando libremente:
# si pro está saturado, los perfiles complejos caen al fallback matemático
# (P0-1) sin afectar el throughput de los perfiles fáciles que usan flash.
#
# Compatibilidad: `_circuit_breaker` (singleton sin sufijo) se conserva
# para callers que no atribuyen modelo. `_get_circuit_breaker(None)` lo
# devuelve. `_get_circuit_breaker("gemini-3-flash-preview")` construye y
# cachea una instancia per-model lazy.
# ============================================================
_CIRCUIT_BREAKERS_BY_MODEL: dict = {}
_CIRCUIT_BREAKERS_LOCK = threading.Lock()


def _get_circuit_breaker(model: str | None = None) -> LLMCircuitBreaker:
    """P1-Q3: devuelve el CB para un modelo específico, construyéndolo lazy.

    Si `model` es None / vacío, devuelve `_circuit_breaker` (global) para
    preservar la semántica previa. Las instancias per-model usan los mismos
    knobs (`MEALFIT_CB_*`) pero sus keys Redis y DB están namespaced por
    nombre de modelo, aislando completamente el estado de fallos.

    Double-checked locking: el lookup sin lock cubre el 99.9% de los casos
    (CB ya construido tras la primera llamada por modelo). El lock solo se
    toma en la PRIMERA invocación por modelo en el proceso.

    Thread-safe: el dict `_CIRCUIT_BREAKERS_BY_MODEL` solo se MUTA bajo
    lock; las lecturas concurrentes ven una snapshot consistente porque
    Python GC garantiza que el ref a la entrada no se libera mientras el
    caller la usa.
    """
    if not model:
        return _circuit_breaker
    cb = _CIRCUIT_BREAKERS_BY_MODEL.get(model)
    if cb is not None:
        return cb
    with _CIRCUIT_BREAKERS_LOCK:
        cb = _CIRCUIT_BREAKERS_BY_MODEL.get(model)
        if cb is not None:
            return cb
        cb = LLMCircuitBreaker(
            failure_threshold=CB_FAILURE_THRESHOLD,
            reset_timeout=CB_RESET_TIMEOUT_S,
            local_health_ttl=CB_LOCAL_HEALTH_TTL_S,
            model_name=model,
        )
        _CIRCUIT_BREAKERS_BY_MODEL[model] = cb
        logger.info(
            f"[CB] P1-Q3: nueva instancia per-model creada para {model!r} "
            f"(keys Redis: {cb._failures_key}, {cb._open_key})."
        )
        return cb


def get_circuit_breaker_snapshot() -> dict:
    """P1-Q3: snapshot read-only del estado de todos los CBs activos.

    Útil para periodic scraping (Prometheus exporter), shutdown logging,
    o tests de regresión. Devuelve un dict {model_name|"_global": health_info}.
    No reads Redis/DB: solo el flag local `_local_healthy` que ya se
    refresca con cada `acan_proceed`/`record_*`.
    """
    snapshot = {
        "_global": {
            "healthy": _circuit_breaker._local_healthy,
            "last_failure_age_s": (
                round(time.time() - _circuit_breaker._failure_propagated_at, 1)
                if _circuit_breaker._failure_propagated_at > 0 else None
            ),
        }
    }
    with _CIRCUIT_BREAKERS_LOCK:
        items = list(_CIRCUIT_BREAKERS_BY_MODEL.items())
    for model, cb in items:
        snapshot[model] = {
            "healthy": cb._local_healthy,
            "last_failure_age_s": (
                round(time.time() - cb._failure_propagated_at, 1)
                if cb._failure_propagated_at > 0 else None
            ),
        }
    return snapshot


# P0-5: Estructura fuerte de tasks de callback de progreso (fire-and-forget).
# `loop.create_task()` solo guarda weak refs en el event loop, así que tasks
# largas (SSE write a un cliente lento, DB hit) podían ser GC-eadas a mitad
# de ejecución haciendo que la UI nunca recibiera el evento. Aquí mantenemos
# una referencia fuerte hasta que el task termine (vía add_done_callback).
#
# P1-A: Antes era `set`, sin orden de inserción. Pasa a `dict` (Python 3.7+
# preserva orden de insert), lo que permite cancelar los más antiguos cuando
# excede el cap blando. Sin este cap, un cliente SSE lento o un callback
# colgado acumulaba tasks indefinidamente — fuga de memoria progresiva por
# request bajo condiciones adversas. El timeout en `_run_async_cb_safe` es
# la primera línea de defensa; el cap aquí es la red de seguridad para
# eventos de transición de estado en los que el timeout no se aplica.
#
# P1-NEW-6: Antes había UN dict global compartido por TODOS los event loops
# vivos en el proceso. Bajo el sync wrapper concurrente (varios threads, cada
# uno con su asyncio.run), múltiples loops mutaban el mismo dict en paralelo:
#   - El cap-drop iteraba `_PROGRESS_CB_TASKS` y `cancel()`-eaba tasks que
#     pertenecían a OTROS loops → comportamiento undefined (cancel cross-loop).
#   - Race entre `add_done_callback` que pop()-ea desde el loop dueño y el
#     cap-drop que pop()-ea desde un loop ajeno.
# Ahora el registro es per-loop vía `WeakKeyDictionary`: cada loop tiene su
# propio sub-dict {task → None}, y el cap-drop solo afecta tasks del MISMO
# loop. La cuenta del cap se mantiene per-loop, no global. Cuando un loop se
# cierra (asyncio.run termina), su sub-dict se libera automáticamente.
_PROGRESS_CB_TASKS_BY_LOOP: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PROGRESS_CB_TASKS_BY_LOOP_LOCK = threading.Lock()


def _get_progress_cb_tasks_for_loop(loop) -> dict:
    """P1-NEW-6: devuelve el sub-dict {task → None} para `loop`, creándolo
    on-demand bajo lock. Cada loop tiene su propio registro aislado.

    El lock protege solo el INSERT al WeakKeyDictionary (microsegundos por
    loop, una sola vez). El acceso al sub-dict luego es lock-free porque
    asyncio es single-threaded por loop.
    """
    sub = _PROGRESS_CB_TASKS_BY_LOOP.get(loop)
    if sub is not None:
        return sub
    with _PROGRESS_CB_TASKS_BY_LOOP_LOCK:
        sub = _PROGRESS_CB_TASKS_BY_LOOP.get(loop)
        if sub is None:
            sub = {}
            _PROGRESS_CB_TASKS_BY_LOOP[loop] = sub
        return sub


# P1-NEW-2: cap configurable vía `MEALFIT_PROGRESS_CB_MAX_PENDING` y
# `MEALFIT_PROGRESS_CB_TIMEOUT_S`. Defaults idénticos al hardcode anterior.
# El cap se aplica per-loop (P1-NEW-6) — cada loop puede tener hasta
# _PROGRESS_CB_TASKS_MAX tasks pendientes antes de empezar a cancelar.
_PROGRESS_CB_TASKS_MAX = PROGRESS_CB_MAX_PENDING  # cap blando per-loop
_PROGRESS_CB_TIMEOUT = PROGRESS_CB_TIMEOUT_S      # timeout duro por callback async (P1-A)


# `atexit` se importa inline (no en el bloque de imports del módulo) para
# mantener autocontenido el conjunto de patches de pool/shutdown. Hoisted
# al tope del bloque de executors (P1-Q9) — antes vivía después del primer
# atexit.register, pero ahora múltiples executors necesitan registrar drain
# helpers que requieren el módulo importado primero.
import atexit as _atexit  # noqa: E402


# ============================================================
# P1-NEW-3 / P1-Q9: _DrainableThreadPoolExecutor
# ------------------------------------------------------------
# Subclass de `ThreadPoolExecutor` que rastrea las futures encoladas/en-vuelo
# para poder hacer drain bounded en shutdown.
#
# Por qué subclassing y no composición: `loop.run_in_executor(executor, ...)`
# y otras APIs de asyncio/concurrent.futures aceptan instancias de
# `ThreadPoolExecutor`. Una clase wrapper que delegue por composición
# rompería ese contrato. Subclassing preserva isinstance() chequeos.
#
# Costo extra por submit: un `add_done_callback` y dos ops a un set
# (~microsegundos). Negligible vs el trabajo que cada submit hace.
#
# P1-Q9: hoisted ARRIBA del primer executor consumidor (era debajo de
# `_FACT_CHECK_EXECUTOR`, lo cual obligaba a usar `ThreadPoolExecutor` plano
# para fact-check). Ahora los 4 executors críticos pueden usar la versión
# drainable directamente.
# ============================================================
class _DrainableThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor con tracking de futures pendientes para drain
    determinista en `atexit`. Drop-in replacement: cualquier callsite que
    espere un `ThreadPoolExecutor` sigue funcionando sin cambios.
    """

    def __init__(self, *args, name: str = "drainable", **kwargs):
        super().__init__(*args, **kwargs)
        self._drain_name = name
        self._drain_pending: set = set()
        self._drain_lock = threading.Lock()

    def submit(self, fn, /, *args, **kwargs):
        fut = super().submit(fn, *args, **kwargs)
        with self._drain_lock:
            self._drain_pending.add(fut)
        # Discard al terminar para no leak. `add_done_callback` se ejecuta en
        # un worker del pool si el future ya completó, o cuando complete; en
        # ambos casos el discard del set es seguro bajo el lock.
        fut.add_done_callback(self._drain_discard)
        return fut

    def _drain_discard(self, fut: concurrent.futures.Future):
        with self._drain_lock:
            self._drain_pending.discard(fut)

    def drain_with_deadline(self, deadline_s: float) -> tuple[int, int]:
        """Espera hasta `deadline_s` que terminen las futures encoladas.
        Devuelve `(done, abandoned)` con conteos finales. No cancela nada.
        """
        with self._drain_lock:
            pending = list(self._drain_pending)
        if not pending:
            return 0, 0
        done_set, not_done_set = concurrent.futures.wait(
            pending, timeout=deadline_s
        )
        return len(done_set), len(not_done_set)


def _shutdown_drainable_executor(
    executor: _DrainableThreadPoolExecutor, name: str, drain_s: int
) -> None:
    """P1-Q9: helper genérico para shutdown bounded de cualquier executor drainable.

    Extraído del antiguo `_shutdown_metrics_with_drain` (P1-NEW-3) para
    aplicar el mismo patrón a TODOS los executors críticos bajo SIGTERM.
    Antes solo `_METRICS_EXECUTOR` tenía drain; los demás (`_FACT_CHECK`,
    `_DB`, `_PROGRESS_CB`) hacían `cancel_futures=True` abrupto, lo cual
    cancela queued pero NO mata threads en vuelo (Python no permite
    interrupt). Bajo rolling deploy con jobs in-flight, los sockets/conn
    quedaban half-closed visibles al cliente.

    Secuencia:
      1. `shutdown(wait=False)` — bloquear nuevas submissions sin esperar.
      2. `drain_with_deadline(drain_s)` — esperar hasta N seg que las
         in-flight terminen naturalmente y cierren sockets gracefully.
      3. `cancel()` futures aún en cola (no comenzadas) para acotar lo
         que `_python_exit` esperará por workers non-daemon.
      4. Logging diferenciado: WARNING si hubo abandono; INFO si todo OK.
    """
    executor.shutdown(wait=False)
    done, abandoned = executor.drain_with_deadline(drain_s)
    if abandoned > 0:
        cancelled = 0
        with executor._drain_lock:
            for fut in list(executor._drain_pending):
                if fut.cancel():
                    cancelled += 1
        logging.getLogger(__name__).warning(
            f"[SHUTDOWN] {name}: drain timeout tras {drain_s}s — "
            f"{abandoned} tasks no terminadas ({cancelled} canceladas "
            f"pre-ejecución; {abandoned - cancelled} en vuelo, esperarán "
            f"_python_exit). Drenadas con éxito: {done}."
        )
    elif done > 0:
        logging.getLogger(__name__).info(
            f"[SHUTDOWN] {name}: {done} tasks drenadas exitosamente antes del exit."
        )


# P1-7: Executor dedicado para fact-checking médico.
# `consultar_base_datos_medica.invoke()` internamente hace clinical_llm.invoke()
# síncrono que tarda 5-15s por call. Ejecutarlo vía `asyncio.to_thread` usaba
# el thread pool default (típicamente ~32 workers compartidos con TODA la app:
# RAG search, embeddings, SQL writes, etc.). Bajo carga concurrente (N pipelines
# en paralelo), el pool default se saturaba con fact-checks lentos y otras
# operaciones rápidas se encolaban detrás de ellos.
# Solución: pool dedicado con max_workers=2. Limita la concurrencia de fact-checks
# globalmente sin afectar el resto de operaciones to_thread. Si llegan 5 pipelines
# simultáneos, los primeros 2 corren, los otros 3 se encolan en este executor
# (no en el pool default), preservando throughput de operaciones más rápidas.
#
# P1-Q9: variante drainable. Bajo SIGTERM (rolling deploy, autoscaling), antes
# se cancelaban abruptamente los fact-checks en vuelo dejando sockets HTTP
# half-closed con el provider clínico. Ahora `_shutdown_drainable_executor`
# espera hasta `FACT_CHECK_SHUTDOWN_DRAIN_S` (~8s default) para que la mayoría
# complete y cierre sockets gracefully.
_FACT_CHECK_EXECUTOR = _DrainableThreadPoolExecutor(
    max_workers=2, thread_name_prefix="fact-check", name="fact-check"
)
_atexit.register(
    _shutdown_drainable_executor,
    _FACT_CHECK_EXECUTOR, "_FACT_CHECK_EXECUTOR", FACT_CHECK_SHUTDOWN_DRAIN_S,
)


# P1-E: Executor dedicado para inserts de métricas (`pipeline_metrics`).
# Antes, `_emit_progress` con event="metric" spawneaba un `threading.Thread`
# fire-and-forget por cada métrica cuando no había `background_tasks`
# (FastAPI). Bajo carga (~50 métricas por pipeline × N pipelines concurrentes),
# eso producía explosión de threads — varios cientos vivos a la vez,
# saturando el OS y haciendo que cada thread.start() costara más que el
# propio insert.
# Con un pool de 4 workers, las métricas se serializan en cola hacia los
# 4 threads dedicados. La cola crece pero los threads no, manteniendo el
# costo por métrica acotado y predecible.
#
# P1-NEW-3: variante drainable. El atexit hook intenta drenar la cola
# por hasta `METRICS_SHUTDOWN_DRAIN_S` antes de cancelar. Antes, un SIGTERM
# de K8s descartaba ~50 métricas/pipeline en vuelo silenciosamente, degradando
# señal de meta-learning (A/B sampler, holistic score history, drift).
# P1-Q9: ahora delega al helper genérico `_shutdown_drainable_executor`
# (mismo patrón aplicado consistentemente a los 4 executors críticos).
_METRICS_EXECUTOR = _DrainableThreadPoolExecutor(
    max_workers=4, thread_name_prefix="metrics", name="metrics"
)
_atexit.register(
    _shutdown_drainable_executor,
    _METRICS_EXECUTOR, "_METRICS_EXECUTOR", METRICS_SHUTDOWN_DRAIN_S,
)


# P0-NEW-1: Executor dedicado para I/O DB síncrona dentro de nodos async.
# Antes, ~15 callsites llamaban `get_user_profile`, `execute_sql_query`,
# `get_recent_meals_from_plans`, `humanize_plan_ingredients`, etc. de forma
# síncrona desde `arun_plan_pipeline` y nodos del grafo. Cada call (50-300ms)
# congelaba el event loop, degradando SSE streaming, callbacks de progreso,
# health-checks y latencia tail de TODAS las requests del worker.
# Solución: pool dedicado max_workers=8. Bounded para evitar saturación del
# OS bajo cientos de pipelines paralelos. Aislado del pool default para que
# operaciones cortas (`asyncio.to_thread` de helpers CPU) no compitan con
# DB I/O.
#
# P1-Q9: variante drainable. Bajo SIGTERM antes se cancelaban abruptamente
# las queries DB en vuelo (UPDATE de health_profile, INSERT de plan, etc.)
# dejando transacciones rolled-back silenciosamente. Ahora se drena hasta
# `DB_SHUTDOWN_DRAIN_S` (~3s default) para que las queries cortas commiteen.
_DB_EXECUTOR = _DrainableThreadPoolExecutor(
    max_workers=8, thread_name_prefix="db-io", name="db-io"
)
_atexit.register(
    _shutdown_drainable_executor,
    _DB_EXECUTOR, "_DB_EXECUTOR", DB_SHUTDOWN_DRAIN_S,
)


# P1-X2: Executor dedicado para callbacks de progreso síncronos.
# ------------------------------------------------------------
# Antes, `_dispatch_progress_callback` despachaba sync callbacks al default
# executor con `loop.run_in_executor(None, _run_sync_cb_safe, ...)` SIN
# timeout. La rama async ya tenía cap duro (`_PROGRESS_CB_TIMEOUT`, P1-A),
# pero la sync no — un callback sync hung del cliente (SSE writer bloqueante,
# DB lookup colgado, file IO lento) ocupaba indefinidamente un slot del
# default `ThreadPoolExecutor` y competía con DB queries, embeddings, RAG
# search y demás operaciones `asyncio.to_thread` del worker, degradando
# latencia tail de TODO el proceso.
#
# Solución (asimétrica → simétrica con la rama async):
#   - Pool dedicado max_workers=4: aísla los callbacks sync del default pool.
#     Si todos los workers se cuelgan, solo bloquea callbacks futuros, no
#     otras operaciones del worker.
#   - Watchdog vía `_run_sync_cb_with_watchdog`: aplica `asyncio.wait_for`
#     con `_PROGRESS_CB_TIMEOUT` y registra `_inc_cb_stat("timed_out")` si
#     el cb excede el cap. Limitación conocida: Python no permite matar
#     threads, así que el thread underlying sigue ejecutando hasta que el
#     callback termine naturalmente; el slot del pool se libera entonces.
#     Pero al menos liberamos al observer (la task del loop) y registramos
#     señal en métricas.
#
# P1-Q9: variante drainable. Bajo SIGTERM antes se cancelaba el último burst
# de eventos SSE al cliente (e.g., el evento `phase=review` o `day_completed`
# de los días finales). Ahora se drena hasta `PROGRESS_CB_SHUTDOWN_DRAIN_S`
# (~2s) para preservar la última experiencia del usuario antes del shutdown.
_PROGRESS_CB_EXECUTOR = _DrainableThreadPoolExecutor(
    max_workers=4, thread_name_prefix="progress-cb", name="progress-cb"
)
_atexit.register(
    _shutdown_drainable_executor,
    _PROGRESS_CB_EXECUTOR, "_PROGRESS_CB_EXECUTOR", PROGRESS_CB_SHUTDOWN_DRAIN_S,
)


# P1-NEW-6: Executor dedicado al sync wrapper `run_plan_pipeline`.
# Antes, cada llamada al wrapper sync (cron / batch / scripts) spawneaba un
# `threading.Thread` directo + `asyncio.run` que crea un loop nuevo cada vez.
# Bajo N callers concurrentes desde el mismo proceso (cron paralelizado, app
# que usa ThreadPoolExecutor para invocar el wrapper, etc.) eso producía:
#   - N threads sin bound (presión sobre el OS, contención de GIL)
#   - N event loops simultáneos mutando globales como `_PROGRESS_CB_TASKS`
#     (ver fix per-loop más abajo)
#   - Cada loop con su propio `asyncio.Semaphore` per-loop en `DistributedLLMSemaphore`
#     fragando el cap real (P1-X1 lo cerró usando `threading.Semaphore` proceso-wide
#     con busy-poll; ya no hay fuga de concurrencia, pero la presión de threads
#     y loops desacotados sigue siendo motivo del pool dedicado).
# Solución: pool dedicado max_workers=`SYNC_WRAPPER_MAX_WORKERS` (default 4),
# configurable vía env. N+1, N+2... callers se serializan en cola del executor
# en lugar de spawnear threads adicionales. Same shutdown semantics que
# `_FACT_CHECK_EXECUTOR` (los jobs son LLM-largos, no vale la pena drenar
# en SIGTERM — la request padre ya está cancelada).
_SYNC_WRAPPER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=SYNC_WRAPPER_MAX_WORKERS, thread_name_prefix="sync-wrap"
)
_atexit.register(_SYNC_WRAPPER_EXECUTOR.shutdown, wait=False, cancel_futures=True)


async def _adb(fn, *args, **kwargs):
    """P0-NEW-1: Despacha una función DB síncrona al `_DB_EXECUTOR` dedicado
    sin bloquear el event loop.

    Patrón de uso:
        profile = await _adb(get_user_profile, user_id)
        rows = await _adb(execute_sql_query, sql, params, fetch_all=True)

    Razonamiento: `loop.run_in_executor` no acepta kwargs nativamente; los
    serializamos vía `functools.partial` solo cuando hay kwargs (path raro).
    El path sin kwargs (mayoría) llama directo para evitar overhead de partial.

    P1-X3: copia el contexto del caller con `contextvars.copy_context()` y lo
    despacha vía `ctx.run`. `loop.run_in_executor` SOLO propaga ContextVars
    cuando el executor es el default (None); para `_DB_EXECUTOR` los workers
    no veían `request_id_var` ni `user_id_var`, así que `custom_print` mostraba
    `[SYS]` en logs de DB y la rate-limit per-user (si la función ejecutada
    invoca al LLM) se evaluaba con `user_id=None` (bypass involuntario).
    """
    loop = asyncio.get_running_loop()
    # P1-X3: snapshot del contexto del caller. Ligero (solo referencias).
    ctx = contextvars.copy_context()
    if kwargs:
        from functools import partial
        return await loop.run_in_executor(_DB_EXECUTOR, ctx.run, partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_DB_EXECUTOR, ctx.run, fn, *args)


# P0-A: Cap de timeout por tool-call individual de fact-checking. La tool
# internamente hace `clinical_llm.invoke()` síncrono (5-15s típico). Sin este
# cap, una llamada colgada bloqueaba indefinidamente uno de los 2 threads del
# pool dedicado — dos colgadas paralizaban TODO el fact-checking de la app.
# Es independiente del cap del LLM orquestador (`_safe_ainvoke timeout=30s`):
# aquí acotamos cada *tool call individual* dentro del agent loop del fact-checker.
# P1-NEW-2: configurable vía `MEALFIT_FACT_CHECK_TOOL_TIMEOUT_S`. Default 20.0.
_FACT_CHECK_TOOL_TIMEOUT = FACT_CHECK_TOOL_TIMEOUT_S


async def _run_fact_check_tool(tool, args, *, timeout: float = _FACT_CHECK_TOOL_TIMEOUT):
    """P1-7 + P0-A: Despacha la tool de fact-checking a su executor dedicado
    con timeout duro por call.

    No usar `asyncio.to_thread` aquí — eso usaría el pool default y saturaría
    operaciones más rápidas. Este executor está aislado a max_workers=2.

    P0-A: Si la tool excede `timeout`, propagamos `asyncio.TimeoutError` al caller
    para que libere el await y registre el fallo. Limitación conocida:
    `ThreadPoolExecutor` no puede matar el thread underlying — el slot del pool
    queda ocupado hasta que `tool.invoke` retorne naturalmente. El cap aguas
    abajo en `tools_medical` (LLM con timeout) cierra esa segunda ventana.
    Aquí nos limitamos a liberar el caller asíncrono y darle al circuit breaker
    la oportunidad de marcar la falla.

    `asyncio.shield` evita que la cancelación que dispara `wait_for` propague
    al future del executor (que igual no puede matar el thread, pero así
    dejamos el future en estado consistente para que el thread pueda completar
    su trabajo y liberar el slot eventualmente).
    """
    loop = asyncio.get_running_loop()
    # P1-X3: propagar ContextVars (request_id, user_id) al thread del pool
    # dedicado. `_FACT_CHECK_EXECUTOR` es custom — `loop.run_in_executor` no
    # propaga el contexto a executors no-default. Sin esto, los logs internos
    # de la tool clínica (`consultar_base_datos_medica`) pierden el request_id.
    ctx = contextvars.copy_context()
    future = loop.run_in_executor(_FACT_CHECK_EXECUTOR, ctx.run, tool.invoke, args)
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
    except asyncio.TimeoutError:
        future.cancel()
        raise


# ============================================================
# P0-4: GRACEFUL CANCELLATION HELPER PARA ainvoke()
# ------------------------------------------------------------
# Antes el código usaba `asyncio.wait_for(llm.ainvoke(...), timeout=N)`
# directamente. Cuando dispara el TimeoutError, `wait_for` cancela el task
# pero retorna inmediatamente sin esperar a que el cliente HTTP corra su
# cleanup (cierre de socket, liberación de connection pool slot, __aexit__
# del context manager). Bajo timeouts repetidos o hedging activo, esto deja
# conexiones half-closed que agotan FDs del OS ("too many open files") y
# saturan el connection pool del provider.
#
# Patrón: shieldear el task dentro de wait_for y, en caso de timeout,
# cancelarlo explícitamente Y awaitear su resolución antes de re-raise.
# Esto le da al cliente HTTP la oportunidad de procesar la cancelación
# (cerrar socket, devolver slot al pool) antes de que el caller continúe.
# ============================================================
async def _swallow_cancelled_task(task: asyncio.Task) -> None:
    """P1-NEW-7: espera a que `task` resuelva tras un `cancel()`, silenciando
    cualquier excepción del cleanup (CancelledError, OSError de cierre de
    socket, errores del cliente HTTP).

    Le da a la task la oportunidad de ejecutar su finally (cerrar socket,
    devolver el slot del connection pool, liberar locks internos del SDK)
    sin que esa excepción enmascare la causa raíz que el caller propaga.
    Si la task ya completó, `await` retorna inmediatamente — el helper es
    barato (~µs) en el path de error normal del LLM.
    """
    try:
        await task
    except BaseException:
        pass


async def _safe_ainvoke(llm, payload, *, timeout: float):
    """Invoca `llm.ainvoke(payload)` con timeout DURO y cancelación graceful.

    P0-4: Cierra el resource leak de sockets HTTP huérfanos cuando dispara
    el timeout. Cualquier excepción del cleanup post-cancelación se silencia
    (no enmascara la causa raíz que el caller debe manejar).

    P1-NEW-7: extiende el cleanup a CUALQUIER salida no-normal, no solo
    `TimeoutError`. Antes, `asyncio.shield(task)` protegía la task interna
    cuando el caller era cancelado externamente — quedaba corriendo hasta
    natural completion, reteniendo el slot del `LLM_SEMAPHORE` y consumiendo
    tokens del proveedor para un resultado que nadie iba a leer.
    Cancelaciones externas comunes que disparaban el leak:
      - `GLOBAL_PIPELINE_TIMEOUT_S` disparando `wait_for(run_graph(), ...)`
        en `arun_plan_pipeline` (línea ~7526).
      - Hedge winner cancelando al hermano perdedor en
        `generate_days_parallel_node` (línea ~3160).
      - Request abort del cliente HTTP / SIGTERM del worker.
      - Sibling fallando en `asyncio.gather(...)` sin `return_exceptions=True`.
    Ahora cualquier `BaseException` distinta a un return normal cancela la
    task interna explícitamente y espera su resolución antes de re-raise —
    misma semántica de cleanup que la rama `TimeoutError`. `BaseException`
    (no `Exception`) porque desde Py3.8 `CancelledError` ya no hereda de
    `Exception` y un `except Exception` la dejaría escapar sin cleanup.
    """
    task = asyncio.create_task(llm.ainvoke(payload))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        # P0-4: timeout duro. Cancelar la task interna y esperar a que libere
        # el socket antes de propagar al caller.
        task.cancel()
        await _swallow_cancelled_task(task)
        raise
    except BaseException:
        # P1-NEW-7: cancelación externa (CancelledError) o cualquier otra
        # excepción del LLM/SDK. `task.cancel()` es idempotente — si la task
        # ya completó (caso "ainvoke raised"), es no-op y el await retorna
        # inmediatamente. Evita el leak donde `shield` mantenía viva la
        # llamada al LLM tras un cancel externo.
        task.cancel()
        await _swallow_cancelled_task(task)
        raise

class PersistentLLMCache:
    """Implementa un diccionario persistente para reemplazar la caché en memoria"""
    def __init__(self, ttl_seconds=300):
        self.ttl = ttl_seconds

    def __contains__(self, key):
        return self.get(key) is not None

    def get(self, key, default=None):
        if redis_client:
            try:
                val = redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        try:
            res = execute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - interval '%s seconds'",
                (key, self.ttl), fetch_one=True
            )
            if res:
                return res["value"] if isinstance(res["value"], list) or isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception as e:
            logger.warning(f"DB cache read error: {e}")
        return default

    async def aget(self, key, default=None):
        if redis_async_client:
            try:
                val = await redis_async_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis async cache read error: {e}")
        try:
            res = await aexecute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - interval '%s seconds'",
                (key, self.ttl), fetch_one=True
            )
            if res:
                return res["value"] if isinstance(res["value"], list) or isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception as e:
            logger.warning(f"DB async cache read error: {e}")
        return default

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        if redis_client:
            try:
                redis_client.setex(key, self.ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        try:
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, json.dumps(value))
            )
        except Exception as e:
            logger.warning(f"DB cache write error: {e}")

    async def aset(self, key, value):
        if redis_async_client:
            try:
                await redis_async_client.setex(key, self.ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis async cache write error: {e}")
        try:
            await aexecute_sql_write(
                "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, json.dumps(value))
            )
        except Exception as e:
            logger.warning(f"DB async cache write error: {e}")

# Brecha 1 Fix: Estado Persistente (Redis / Supabase)
# P1-NEW-2: TTL configurable vía `MEALFIT_LLM_CACHE_TTL_S`. Default 300s.
_LLM_CACHE = PersistentLLMCache(ttl_seconds=LLM_CACHE_TTL_S)
CACHE_TTL_SECONDS = LLM_CACHE_TTL_S

# ============================================================
# ESTADO COMPARTIDO DEL GRAFO
# ============================================================
class PlanState(TypedDict):
    request_id: str
    # Inputs (se setean al inicio)
    form_data: dict
    taste_profile: str
    nutrition: dict
    history_context: str
    reflection_directive: Optional[str]
    compressed_context: Optional[str]
    
    # Semantic Caching
    semantic_cache_hit: bool
    cached_plan_data: Optional[dict]
    profile_embedding: Optional[list]

    # Plan generado (output del generador)
    plan_result: Optional[dict]
    
    # Esqueleto del planificador (fase map)
    plan_skeleton: Optional[dict]
    
    # Adversarial Self-Play (Mejora 1)
    candidate_a: Optional[dict]
    candidate_b: Optional[dict]
    adversarial_rationale: Optional[str]
    _ab_temp_meta: Optional[dict]  # Par de temperaturas AB usado en el ciclo actual
    
    # Revisión médica
    review_passed: bool
    review_feedback: str
    
    # Control de flujo
    attempt: int
    user_facts: str
    rejection_reasons: list[str]
    _rejection_severity: Optional[str]
    
    # Callback de progreso para SSE streaming (opcional)
    progress_callback: Optional[Any]  # Callable o None
    background_tasks: Optional[Any]   # FastAPI BackgroundTasks

    # GAP B: Contexto cacheado para reducir latencia
    _cached_context: Optional[dict]

    # Brecha 2: Corrección Quirúrgica
    _affected_days: Optional[list[int]]

    # Presupuesto de tiempo — para decisiones de retry conscientes del tiempo
    pipeline_start: Optional[float]

    # P1-G: Buffers de coalescing de tokens streaming (P1-1).
    # Mutado in-place desde `_emit_progress` cuando event="token". Antes, sin
    # esta declaración, LangGraph en strict-schema mode lo filtraba al pasar
    # `state` entre nodos, perdiendo los buffers acumulados silenciosamente.
    # Forma: `{ day_key: {"text": str, "last_flush": float} }`.
    # `_flush_token_buffer_for_day` (P1-B) lo lee y limpia en el path de
    # excepción de `_safe_gen` para garantizar entrega al cliente.
    #
    # [P1-4] Tipado como `dict` (no `Optional[dict]`). La inicialización en
    # `arun_plan_pipeline:9299` siempre lo setea a `{}` con un comentario
    # explícito de que `None` rompería `setdefault` downstream. Permitir
    # `None` en el TYPE era una trampa de mantenimiento: un test directo del
    # nodo o un refactor futuro que escribiera `state["_token_buffers"] = None`
    # (válido por type) reventaba `_emit_progress` con AttributeError. Tipo
    # estricto + guard defensivo runtime en `_emit_progress` cierran ambas
    # capas (type-checker estático + safety runtime) sin tener que coordinar
    # cambios futuros.
    _token_buffers: dict



# ============================================================
# SCHEMAS (importados del módulo canónico schemas.py)
# ============================================================
from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel, PlanSkeletonModel, SingleDayPlanModel


# ============================================================
# PROMPTS (importados del paquete prompts/)
# ============================================================
from prompts.plan_generator import (
    GENERATOR_SYSTEM_PROMPT,
    build_nutrition_context,
    build_correction_context,
    build_unified_behavioral_profile,
    build_rag_context,
    # [P0-FORM-3] Inyecta motivación personal del usuario al planner + day generator.
    build_motivation_context,
    build_time_context,
    build_technique_injection,
    build_supplements_context,
    build_grocery_duration_context,
    build_pantry_context,
    build_prices_context,
    build_adherence_context,
    build_success_patterns_context,
    build_temporal_adherence_context,
    build_skeleton_quality_context,
    build_fatigue_context,
    build_quality_hint_context,
    build_weight_history_context,
    build_liked_meals_context,
    build_chunk_lessons_context,
    build_prev_chunk_adherence_context,
    build_pantry_correction_context,
)
from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT
from prompts.planner import PLANNER_SYSTEM_PROMPT
from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_assignment_context


# ============================================================
# HELPERS: Selección de técnicas y contextos compartidos
# ============================================================

def _flatten_ingredient(ing) -> str:
    """Asegura que un ingrediente sea un string plano, incluso si el LLM devuelve una lista anidada."""
    if isinstance(ing, list):
        return " ".join(str(x) for x in ing)
    return str(ing) if ing is not None else ""

def _calculate_complexity_score(plan: dict) -> float:
    """Calcula la complejidad promedio del plan (1 a 10).
    Factores: cantidad de ingredientes, longitud de receta, técnicas complejas.

    P1-11: vocabulario expandido (`COMPLEX_TECHNIQUE_KEYWORDS` desde constants)
    cubre sinónimos, conjugaciones y términos en inglés. Además normaliza
    `recipe` defensivamente: el LLM a veces devuelve `list[str]` (steps) o
    `None` y el `.split` antiguo lanzaba AttributeError silencioso.
    """
    total_score = 0
    meal_count = 0
    days = plan.get("days", [])

    for day in days:
        for meal in day.get("meals", []):
            meal_count += 1
            score = 1.0 # Base score

            # 1. Por cantidad de ingredientes
            ing_count = len(meal.get("ingredients", []))
            if ing_count > 8: score += 3.0
            elif ing_count > 5: score += 1.5
            elif ing_count <= 3: score -= 0.5

            # 2. Por pasos/longitud de receta — normalizar a string primero
            recipe_raw = meal.get("recipe", "")
            if isinstance(recipe_raw, list):
                # Cada elemento es un paso → unimos con \n para que el conteo
                # de "pasos" sea fiable.
                recipe = "\n".join(str(s) for s in recipe_raw)
            elif recipe_raw is None:
                recipe = ""
            else:
                recipe = str(recipe_raw)

            steps = len([s for s in recipe.split('\n') if s.strip()])
            if steps > 5: score += 3.0
            elif steps > 3: score += 1.5
            elif steps <= 2: score -= 0.5

            # 3. Técnicas complejas — vocabulario canónico desde constants
            recipe_lower = recipe.lower()
            if any(tech in recipe_lower for tech in COMPLEX_TECHNIQUE_KEYWORDS):
                score += 2.0

            # Limitar score entre 1 y 10
            total_score += min(10.0, max(1.0, score))

    if meal_count == 0: return 0.0
    return round(total_score / meal_count, 1)

def _emit_progress(state: PlanState, event: str, data: dict):
    """Emite un evento de progreso si hay callback registrado, y guarda métricas de observabilidad."""
    if event == "metric":
        try:
            uid = state.get("form_data", {}).get("user_id", "guest")
            sid = state.get("form_data", {}).get("session_id", "unknown")
            _resolved_uid = uid if uid != "guest" else None
            # P1-Q10: si el probe del startup detectó que pipeline_metrics
            # rechaza inserts con user_id=NULL (schema drift no fixed por la
            # migración), skipear guest metrics gracefully en lugar de
            # disparar 50× IntegrityError por pipeline. El probe ya logueó
            # CRITICAL al startup con remediation steps.
            if _resolved_uid is None and not _is_guest_metrics_enabled():
                return
            metrics_data = {
                "user_id": _resolved_uid,
                "session_id": sid,
                "node": data.get("node", "unknown"),
                "duration_ms": int(data.get("duration_ms", 0)),
                "retries": data.get("retries", state.get("attempt", 1) - 1),
                "tokens_estimated": data.get("tokens_estimated", 0),
                "confidence": float(data.get("confidence", 0.0)),
                "metadata": data.get("metadata", {})
            }
            # Fire and forget
            def _save():
                try:
                    from db_core import execute_sql_write
                    import json
                    query = """
                    INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    params = (
                        metrics_data["user_id"],
                        metrics_data["session_id"],
                        metrics_data["node"],
                        metrics_data["duration_ms"],
                        metrics_data["retries"],
                        metrics_data["tokens_estimated"],
                        metrics_data["confidence"],
                        json.dumps(metrics_data["metadata"])
                    )
                    execute_sql_write(query, params)
                except Exception as e:
                    # P1-Q10: log diagnóstico mejorado — antes era genérico
                    # `Failed to insert metric: {e}`. Ahora incluye el `node`
                    # y si el caller era guest (la dimensión más útil para
                    # diagnosticar pérdida sistemática vs incidente puntual).
                    logger.error(
                        f"[METRICS] Insert falló para node={metrics_data['node']!r} "
                        f"(guest={metrics_data['user_id'] is None}): "
                        f"{type(e).__name__}: {e}"
                    )
            
            bg_tasks = state.get("background_tasks")
            if bg_tasks and hasattr(bg_tasks, "add_task"):
                bg_tasks.add_task(_save)
            else:
                # P1-E: pool dedicado en lugar de un Thread por métrica.
                # Bajo carga (~50 métricas/pipeline × N pipelines concurrentes),
                # `Thread().start()` repetido saturaba el OS — el costo de spawn
                # superaba al del propio insert. El executor con max_workers=4
                # serializa las métricas en cola hacia 4 threads reusables;
                # los threads no proliferan, solo crece la cola interna.
                # `submit` no bloquea (returns Future inmediatamente).
                # P1-X3: propagar ContextVars (request_id) al worker; sin esto,
                # cualquier log producido dentro de `_save` (ej. `Failed to
                # insert metric`) saldría con `[SYS]` en vez del request_id
                # del pipeline que originó la métrica.
                ctx = contextvars.copy_context()
                _METRICS_EXECUTOR.submit(ctx.run, _save)
        except Exception as e:
            logger.error(f"⚠️ Error preparing metric insertion: {e}")

    cb = state.get("progress_callback")
    if cb:
        # P1-1: Coalescing de tokens streaming.
        # `token` se emite por cada chunk del LLM (~5-10 chars) durante streaming
        # paralelo de N días. Eso son 200-1000 eventos por pipeline → 200-1000
        # tasks fire-and-forget que saturan el callback SSE de Uvicorn.
        # Acumulamos chunks en un buffer per-day y solo flusheamos cuando:
        #   - el buffer supera _PROGRESS_TOKEN_FLUSH_BYTES caracteres, O
        #   - pasaron _PROGRESS_TOKEN_FLUSH_MS desde el último flush, O
        #   - llega un evento de transición (token_reset/day_completed/tool_call/
        #     phase/metric/...) que indica fin del segmento de streaming.
        # Reduce ~20-50× la frecuencia de eventos sin perder información:
        # el callback recibe el mismo evento "token" con `chunk` más grande.
        if event == "token" and isinstance(data, dict):
            day_key = data.get("day", "_default")
            chunk = data.get("chunk", "")
            # [P1-4] Guard defensivo: `state.setdefault("_token_buffers", {})`
            # NO protegía el caso donde la key existe pero su valor es None
            # (setdefault solo inserta si la key falta). El initial_state
            # siempre setea `{}` y el TYPE ahora es `dict` (no Optional), pero
            # un test directo del nodo, un refactor que escriba None
            # explícitamente, o un merge raro de LangGraph podía dejar el
            # estado en `{"_token_buffers": None}` y reventar `setdefault`
            # downstream con AttributeError. `isinstance(_, dict)` cubre None,
            # listas, strings, cualquier basura — y reasigna de vuelta al
            # state para que el próximo emit no reintente la rehidratación.
            buffers = state.get("_token_buffers")
            if not isinstance(buffers, dict):
                buffers = {}
                state["_token_buffers"] = buffers
            buf = buffers.setdefault(day_key, {"text": "", "last_flush": 0.0})
            buf["text"] += chunk
            now = time.monotonic()
            if buf["last_flush"] == 0.0:
                buf["last_flush"] = now
            should_flush = (
                len(buf["text"]) >= _PROGRESS_TOKEN_FLUSH_BYTES
                or (now - buf["last_flush"]) * 1000 >= _PROGRESS_TOKEN_FLUSH_MS
            )
            if not should_flush:
                return  # Acumular más antes de despachar
            # Flush: emitir chunk consolidado
            payload = {"event": "token", "data": {"day": day_key, "chunk": buf["text"]}}
            buf["text"] = ""
            buf["last_flush"] = now
            _dispatch_progress_callback(cb, payload, event=event)
            return

        # P1-1: Eventos de transición → flush implícito de cualquier buffer
        # pendiente del mismo día (preserva orden secuencial token → completed).
        # P1-B: lógica extraída a `_flush_token_buffer_for_day` para reusarla
        # desde el path de excepción de `_generate_day_hedged`.
        if event in _PROGRESS_FLUSH_TRIGGERS and isinstance(data, dict):
            _flush_token_buffer_for_day(state, data.get("day", "_default"))
            # Cae al dispatch normal del evento original abajo

        payload = {"event": event, "data": data}
        _dispatch_progress_callback(cb, payload, event=event)


# P1-1: Umbrales de coalescing del stream de tokens. Tunear si el cliente SSE
# muestra "tirones" (subir _PROGRESS_TOKEN_FLUSH_MS) o queda muy entrecortado
# (bajar _PROGRESS_TOKEN_FLUSH_BYTES).
_PROGRESS_TOKEN_FLUSH_BYTES = 200      # ~25-40 tokens del LLM
_PROGRESS_TOKEN_FLUSH_MS = 250         # smooth-feel para el cliente
_PROGRESS_FLUSH_TRIGGERS = frozenset({
    "token_reset", "day_completed", "tool_call", "phase", "metric", "day_started"
})


def _register_progress_task(loop, task, kind: str) -> None:
    """P1-X2: registra una task de progreso en el sub-dict per-loop y aplica
    el cap blando (`_PROGRESS_CB_TASKS_MAX`).

    Compartido entre la rama async (callbacks coroutine) y la sync (watchdog
    sobre `_PROGRESS_CB_EXECUTOR`). Antes la lógica vivía inline en el path
    async; el path sync no contaba para el cap, así que un cliente sync que
    encolara miles de eventos no era acotado a nivel de tasks del loop.

    Conserva las garantías P0-5 (ref fuerte hasta done) y P1-NEW-6 (sub-dict
    per-loop, no global) — ambas siguen funcionando porque el helper opera
    sobre `tasks_dict` que `_get_progress_cb_tasks_for_loop` resuelve para el
    loop activo.
    """
    tasks_dict = _get_progress_cb_tasks_for_loop(loop)
    tasks_dict[task] = None
    # P1-A: cap blando per-loop. Si el sub-dict excede el cap, cancelar los
    # más antiguos del MISMO loop. dict preserva orden de inserción → los
    # primeros keys son los más viejos.
    if len(tasks_dict) > _PROGRESS_CB_TASKS_MAX:
        n_drop = len(tasks_dict) - _PROGRESS_CB_TASKS_MAX
        oldest = []
        for old_t in tasks_dict:
            oldest.append(old_t)
            if len(oldest) >= n_drop:
                break
        for old_t in oldest:
            old_t.cancel()
            tasks_dict.pop(old_t, None)
        # P1-NEW-4: registrar la pérdida en counters observables.
        _inc_cb_stat("dropped_cap", n_drop)
        if _should_log_cb_error(kind):
            logger.warning(
                f"[PROGRESS] P1-A: cap excedido per-loop "
                f"({len(tasks_dict) + n_drop} tasks > "
                f"{_PROGRESS_CB_TASKS_MAX}). Cancelando {n_drop} más "
                f"antiguos. Probable cliente SSE lento o callback hung."
            )
    # `tasks_dict.pop` corre en el done_callback del task que vive en
    # ESTE loop, así que la mutación es safe (no cross-loop).
    task.add_done_callback(lambda t, td=tasks_dict: td.pop(t, None))


async def _run_sync_cb_with_watchdog(cb, payload):
    """P1-X2: ejecuta un callback sync en `_PROGRESS_CB_EXECUTOR` con cap duro
    de `_PROGRESS_CB_TIMEOUT`.

    Antes el sync se despachaba con `loop.run_in_executor(None, ...)` SIN
    timeout — un cliente sync hung (SSE writer bloqueante, DB lookup colgado)
    ocupaba un slot del default executor indefinidamente y degradaba latencia
    tail de TODO el worker.

    Ahora:
      - Pool dedicado (`_PROGRESS_CB_EXECUTOR`, max_workers=4) aísla del default.
      - `asyncio.shield` evita que la cancelación del wait_for cancele el
        future del executor mientras el thread underlying sigue corriendo
        (Python no permite matar threads).
      - Si excede el timeout, registramos `_inc_cb_stat("timed_out")` y
        liberamos al observer; el thread sigue ejecutando hasta completar
        naturalmente y libera el slot del pool entonces.
      - El logging usa el throttling existente (`_should_log_cb_error`).

    `_run_sync_cb_safe` ya silencia excepciones del cb propiamente y registra
    `failed_sync`; aquí solo cubrimos el caso "el cb sigue corriendo más allá
    del cap aceptable".
    """
    loop = asyncio.get_running_loop()
    # P1-X3: propagar ContextVars (request_id, user_id, _pipeline_cb_stats_var)
    # al thread del pool. Sin esto, `_run_sync_cb_safe` invoca `_inc_cb_stat`
    # que muta el dict per-pipeline vía `_pipeline_cb_stats_var.get()` — pero
    # el ContextVar estaría en su default `None`, así que las pérdidas de
    # eventos sync NO se contabilizarían en la métrica `progress_cb` de
    # `arun_plan_pipeline` (rompía el conteo P1-NEW-4).
    ctx = contextvars.copy_context()
    fut = loop.run_in_executor(_PROGRESS_CB_EXECUTOR, ctx.run, _run_sync_cb_safe, cb, payload)
    try:
        await asyncio.wait_for(asyncio.shield(fut), timeout=_PROGRESS_CB_TIMEOUT)
    except asyncio.TimeoutError:
        _inc_cb_stat("timed_out")
        if _should_log_cb_error("sync"):
            event = payload.get("event") if isinstance(payload, dict) else "?"
            logger.warning(
                f"[PROGRESS] P1-X2: sync callback timeout "
                f"(>{_PROGRESS_CB_TIMEOUT:.0f}s, event={event!r}). "
                f"El thread del pool sigue ejecutando hasta natural completion. "
                f"Total fallos sync: {_CB_ERROR_LOG_COUNTER['sync']}."
            )
    except Exception:
        # Excepciones del propio cb las registra `_run_sync_cb_safe`; cualquier
        # cosa que llegue aquí es de la infraestructura del executor — silenciar
        # para no derrumbar el dispatcher.
        pass


def _dispatch_progress_callback(cb, payload, *, event: str):
    """P0-5/P1-1/P1-X2: Despacha un payload al callback sin bloquear el event loop.

    Extraído del cuerpo de `_emit_progress` para que el batching de tokens
    pueda reusar la misma lógica (despachar buffer pendiente + despachar
    evento actual). Comportamiento:
      - cb async      → create_task (fire-and-forget en el loop, ref fuerte)
      - cb sync       → watchdog en `_PROGRESS_CB_EXECUTOR` con timeout
                        (P1-X2: simétrico con la rama async)
      - sin loop      → cb sync corre inline, cb async se descarta con warning
    """
    try:
        loop = asyncio.get_running_loop()
        if asyncio.iscoroutinefunction(cb):
            # P0-5: Mantener referencia fuerte para evitar GC prematuro.
            # `loop.create_task()` solo guarda weak refs; sin esta protección
            # tasks de >100ms (SSE write, DB hit) podían desaparecer a mitad
            # de ejecución y la UI nunca recibía el evento.
            t = loop.create_task(_run_async_cb_safe(cb, payload))
            _register_progress_task(loop, t, "async")
        else:
            # P1-X2: dispatch sync con watchdog. El cb se ejecuta en el pool
            # dedicado, y la task watchdog (que vive en este loop) propaga
            # el resultado / timeout. Cuenta para el cap blando per-loop
            # igual que la rama async.
            t = loop.create_task(_run_sync_cb_with_watchdog(cb, payload))
            _register_progress_task(loop, t, "sync")
    except RuntimeError:
        # No hay loop activo (caller sync standalone). Para callbacks sync
        # ejecutamos inline para preservar el contrato previo. Para
        # coroutines no se puede agendar; se descartan con warning explícito.
        if asyncio.iscoroutinefunction(cb):
            logger.warning(
                "[PROGRESS] Callback async ignorado (sin event loop activo). "
                f"event={event!r}. Si llamas el pipeline desde código sync, "
                "usa un callback sync."
            )
        else:
            try:
                cb(payload)
            except Exception as e:
                logger.warning(f"[PROGRESS] Callback sync inline falló: {e!r} (event={event!r})")


def _flush_token_buffer_for_day(state, day_key) -> bool:
    """P1-B: Flushea el buffer de tokens pendiente para un día.

    Despacha como evento `token` cualquier texto acumulado en
    `state["_token_buffers"][day_key]` y limpia el buffer. Idempotente:
    si no hay buffer o está vacío, no hace nada.

    Llamadores:
      - `_emit_progress` ante eventos de transición (day_completed, phase, etc.)
        — preserva orden secuencial token → completed.
      - `_generate_day_hedged` en `finally` outermost — antes, si
        `generate_single_day` lanzaba mid-stream (timeout, retry exhausted,
        cancelación por hedge ganador), los tokens acumulados en el buffer
        se perdían y la UI mostraba texto cortado al final del día.

    Retorna True si flusheó contenido.
    """
    cb = state.get("progress_callback") if isinstance(state, dict) else None
    if not cb:
        return False
    buffers = state.get("_token_buffers")
    if not buffers or day_key not in buffers:
        return False
    pending = buffers[day_key].get("text", "")
    if not pending:
        return False
    buffers[day_key]["text"] = ""
    buffers[day_key]["last_flush"] = time.monotonic()
    _dispatch_progress_callback(
        cb,
        {"event": "token", "data": {"day": day_key, "chunk": pending}},
        event="token",
    )
    return True


# P0-5: Throttling de logs de errores de callback. Un callback roto puede
# disparar miles de veces por pipeline; loguear cada falla satura el output.
# Mantenemos un contador y solo logueamos las primeras N + 1 cada 100.
_CB_ERROR_LOG_COUNTER = {"async": 0, "sync": 0}
_CB_ERROR_LOG_THRESHOLD = 5  # primeras 5 fallas siempre se loguean
_CB_ERROR_LOG_SAMPLE = 100   # luego, 1 de cada 100

def _should_log_cb_error(kind: str) -> bool:
    n = _CB_ERROR_LOG_COUNTER.get(kind, 0) + 1
    _CB_ERROR_LOG_COUNTER[kind] = n
    if n <= _CB_ERROR_LOG_THRESHOLD:
        return True
    return n % _CB_ERROR_LOG_SAMPLE == 0


# ============================================================
# P1-NEW-4: Counters de pérdida/falla de eventos SSE para observabilidad.
# ------------------------------------------------------------
# Antes, las cancelaciones por cap (P1-A) y los timeouts/excepciones en
# callbacks tenían throttled logging pero no exportaban señal numérica.
# Una regresión que provocara cancelaciones masivas (cliente SSE roto,
# callback hung) pasaba desapercibida hasta reportes manuales — el
# dashboard de Grafana no podía alertar.
#
# Tracking en dos niveles:
#   1. Cumulativo process-wide (`_PROGRESS_CB_STATS`): sirve para scrape
#      externo, periodic logging o tests de regresión.
#   2. Per-pipeline vía contextvar (`_pipeline_cb_stats_var`): cada llamada
#      a `arun_plan_pipeline` setea un dict fresco; tasks descendientes
#      (asyncio.Task, run_in_executor) heredan el contexto y mutan el mismo
#      dict. Al final del pipeline, `arun_plan_pipeline` emite una métrica
#      "progress_cb" con los conteos para esta request, persistida en
#      `pipeline_metrics` y queryable post-hoc.
#
# Categorías:
#   - "dropped_cap":  tasks canceladas por exceder _PROGRESS_CB_TASKS_MAX
#   - "timed_out":    tasks async que superaron _PROGRESS_CB_TIMEOUT
#   - "failed_async": cualquier excepción en callbacks coroutine
#   - "failed_sync":  cualquier excepción en callbacks sync (en executor)
# ============================================================
_PROGRESS_CB_STATS: dict = {
    "dropped_cap": 0,
    "timed_out": 0,
    "failed_async": 0,
    "failed_sync": 0,
}
_PROGRESS_CB_STATS_LOCK = threading.Lock()

# Per-pipeline mutable dict referenciado vía contextvar. `arun_plan_pipeline`
# asigna uno nuevo al entrar; tasks descendientes ven el mismo objeto.
_pipeline_cb_stats_var = contextvars.ContextVar(
    "pipeline_cb_stats", default=None
)


def _inc_cb_stat(kind: str, n: int = 1) -> None:
    """P1-NEW-4: incrementa counter `kind` en cumulativo + per-pipeline.

    Thread-safe vía lock — los callbacks SSE se despachan a threads del
    executor default y mutan concurrentemente. El lock protege el dict
    cumulativo Y la mutación del per-pipeline en la misma sección crítica.
    """
    if n <= 0:
        return
    with _PROGRESS_CB_STATS_LOCK:
        _PROGRESS_CB_STATS[kind] = _PROGRESS_CB_STATS.get(kind, 0) + n
        ppl = _pipeline_cb_stats_var.get()
        if ppl is not None:
            ppl[kind] = ppl.get(kind, 0) + n


def get_progress_cb_stats_snapshot() -> dict:
    """P1-NEW-4: snapshot read-only de los counters cumulativos.

    Útil para periodic scraping (Prometheus exporter), shutdown logging,
    o tests de regresión. Devuelve una copia para que el caller no pueda
    mutar el estado interno.
    """
    with _PROGRESS_CB_STATS_LOCK:
        return dict(_PROGRESS_CB_STATS)


async def _run_async_cb_safe(cb, payload):
    """P0-5 + P1-A: Wrapper que silencia excepciones del callback async
    (fire-and-forget) y aplica timeout duro.

    Antes la excepción se silenciaba sin log → silent failures con UI congelada.
    Ahora logueamos con throttling (primeras 5 + 1/100) para visibilidad sin saturar.

    P1-A: añadimos timeout = `_PROGRESS_CB_TIMEOUT` (10s). Sin esto, un cliente
    SSE hung dejaba la task viva indefinidamente, la cual sería retenida por
    `_PROGRESS_CB_TASKS` hasta agotar memoria. El timeout cancela la coroutine
    interna y libera el slot del set.
    """
    try:
        await asyncio.wait_for(cb(payload), timeout=_PROGRESS_CB_TIMEOUT)
    except asyncio.TimeoutError:
        # P1-NEW-4: registrar timeout antes del log throttled para no perder
        # señal cuando el throttling silencia el warning (>5 fallos).
        _inc_cb_stat("timed_out")
        if _should_log_cb_error("async"):
            event = payload.get("event") if isinstance(payload, dict) else "?"
            logger.warning(
                f"[PROGRESS] P1-A: Async callback timeout "
                f"(>{_PROGRESS_CB_TIMEOUT:.0f}s, event={event!r}). "
                f"Probable cliente SSE lento o socket hung. "
                f"Total fallos async: {_CB_ERROR_LOG_COUNTER['async']}."
            )
    except Exception as e:
        # P1-NEW-4: idem — counter se incrementa siempre, log con throttling.
        _inc_cb_stat("failed_async")
        if _should_log_cb_error("async"):
            event = payload.get("event") if isinstance(payload, dict) else "?"
            logger.warning(
                f"[PROGRESS] Async callback falló (event={event!r}): {e!r}. "
                f"Total fallos async: {_CB_ERROR_LOG_COUNTER['async']}."
            )


def _run_sync_cb_safe(cb, payload):
    """P0-5: Wrapper que silencia excepciones del callback sync (despachado a executor).

    Igual que el async: con throttled logging para visibilidad de silent failures.
    """
    try:
        cb(payload)
    except Exception as e:
        # P1-NEW-4: registrar el fallo siempre; el log es throttled.
        _inc_cb_stat("failed_sync")
        if _should_log_cb_error("sync"):
            event = payload.get("event") if isinstance(payload, dict) else "?"
            logger.warning(
                f"[PROGRESS] Sync callback falló (event={event!r}): {e!r}. "
                f"Total fallos sync: {_CB_ERROR_LOG_COUNTER['sync']}."
            )


def _select_techniques(user_id: str | None, successful_techniques: list = None, abandoned_techniques: list = None) -> list:
    """Selecciona 3 técnicas de cocción diversificadas por familia con decaimiento temporal y cruzado de éxito."""
    technique_freq = {}
    if user_id:
        try:
            recent_techs = get_recent_techniques(user_id, limit=6)
            now_utc = datetime.now(timezone.utc)
            decay_factor = 0.9
            for t, created_at_str in recent_techs:
                days_elapsed = 0
                if created_at_str:
                    try:
                        if created_at_str.endswith("Z"):
                            dt = datetime.fromisoformat(created_at_str[:-1]).replace(tzinfo=timezone.utc)
                        else:
                            dt = datetime.fromisoformat(created_at_str)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        days_elapsed = max(0, (now_utc - dt).days)
                    except Exception:
                        pass
                decayed_weight = decay_factor ** days_elapsed
                technique_freq[t] = technique_freq.get(t, 0) + decayed_weight
            if technique_freq:
                print(f"🔍 [TÉCNICAS] Frecuencias con decaimiento temporal: { {k: round(v, 2) for k, v in technique_freq.items()} }")
        except Exception as e:
            print(f"⚠️ [TÉCNICAS] Error consultando DB, usando pesos uniformes: {e}")

    selected_techniques = []
    used_families = set()
    _pool_t = [(t, 1.0 / (technique_freq.get(t, 0) + 1)) for t in ALL_TECHNIQUES]
    
    if successful_techniques or abandoned_techniques:
        successful_techniques = successful_techniques or []
        abandoned_techniques = abandoned_techniques or []
        new_pool = []
        for t, w in _pool_t:
            if t in abandoned_techniques:
                w *= 0.1 # Penalización severa
            elif t in successful_techniques:
                w *= 2.0 # Boost por éxito comprobado
            new_pool.append((t, w))
        _pool_t = new_pool

    while len(selected_techniques) < 3 and _pool_t:
        cross_family_pool = [(t, w) for t, w in _pool_t if TECH_TO_FAMILY.get(t) not in used_families]
        active_pool = cross_family_pool if cross_family_pool else _pool_t
        pick = random.choices([x[0] for x in active_pool], weights=[x[1] for x in active_pool], k=1)[0]
        selected_techniques.append(pick)
        used_families.add(TECH_TO_FAMILY.get(pick, ""))
        _pool_t = [(t, w) for t, w in _pool_t if t != pick]

    print(f"👨‍🍳 [TÉCNICAS] Seleccionadas (familias diversas): {[f'{t} ({TECH_TO_FAMILY.get(t)})' for t in selected_techniques]}")
    return selected_techniques


def _build_shared_context(state: PlanState, force_rebuild: bool = False) -> dict:
    """Construye todos los bloques de contexto compartidos entre nodos."""
    if not force_rebuild and state.get("_cached_context"):
        return state["_cached_context"]

    form_data = state["form_data"]
    nutrition = state["nutrition"]
    review_feedback = state.get("review_feedback", "")
    user_facts = state.get("user_facts", "")
    history_context = state.get("compressed_context") or state.get("history_context", "")
    taste_profile = state.get("taste_profile", "")

    _uid = form_data.get("user_id") or form_data.get("session_id")
    if _uid == "guest": _uid = None

    rejection_reasons = state.get("rejection_reasons", [])

    from ai_helpers import get_deterministic_variety_prompt
    variety_prompt = get_deterministic_variety_prompt(history_context, form_data, user_id=_uid, rejection_reasons=rejection_reasons)
    adherence_hint = form_data.get("_adherence_hint", "")
    meal_level_adherence = form_data.get("_meal_level_adherence", {})
    ignored_meal_types = form_data.get("_ignored_meal_types", [])
    cold_start_recs = form_data.get("_cold_start_recommendations", [])
    succ_techs = form_data.get("successful_techniques", [])
    aban_techs = form_data.get("abandoned_techniques", [])
    abandoned_reasons = form_data.get("_abandoned_reasons", {})
    emotional_state = form_data.get("_emotional_state", None)
    successful_tone_strategies = form_data.get("_successful_tone_strategies", [])
    previous_plan_quality = form_data.get("_previous_plan_quality")
    fatigued_ingredients = form_data.get("fatigued_ingredients", [])
    quality_hint = form_data.get("_quality_hint", "")
    drastic_strategy = form_data.get("_drastic_change_strategy", None)
    weight_history = form_data.get("weight_history", [])
    nudge_conversion_rates = form_data.get("_nudge_conversion_rates", {})
    frustrated_meal_types = form_data.get("_frustrated_meal_types", [])
    liked_meals = form_data.get("_liked_meals", [])
    liked_flavor_profiles = form_data.get("_liked_flavor_profiles", [])
    allergies = nutrition.get("alergias", [])
    cold_start_recs = form_data.get("_cold_start_recommendations", [])
    llm_retrospective = form_data.get("_llm_retrospective", "")
    chunk_lessons = form_data.get("_chunk_lessons")
    prev_chunk_adherence = form_data.get("_prev_chunk_adherence")

    return {
        "user_id": _uid,
        "quality_context": build_skeleton_quality_context(previous_plan_quality, meal_level_adherence),
        "quality_hint_context": build_quality_hint_context(quality_hint, drastic_strategy),
        "chunk_lessons_context": build_chunk_lessons_context(chunk_lessons),
        "prev_chunk_adherence_context": build_prev_chunk_adherence_context(prev_chunk_adherence),
        "weight_history_context": build_weight_history_context(weight_history),
        "nutrition_context": build_nutrition_context(nutrition),
        "adherence_context": build_adherence_context(adherence_hint, meal_level_adherence, ignored_meal_types, abandoned_reasons, emotional_state, successful_tone_strategies, nudge_conversion_rates, frustrated_meal_types),
        "success_patterns_context": build_success_patterns_context(succ_techs, aban_techs, []),
        "temporal_adherence_context": build_temporal_adherence_context(form_data.get("day_of_week_adherence", {})),
        "unified_behavioral_profile": build_unified_behavioral_profile(user_facts, fatigued_ingredients, liked_meals, liked_flavor_profiles, cold_start_recs, allergies, llm_retrospective),
        # [P0-FORM-3] `motivation` se sanea río arriba en `_sanitize_form_data_recursive`,
        # así que llega aquí ya seguro contra prompt injection. El builder retorna ""
        # si el campo está vacío/whitespace → no-op transparente para usuarios que
        # no completaron el step (aunque desde P0-FORM-3 es required en el backend).
        "motivation_context": build_motivation_context(form_data),
        "rag_context": "",
        "fatigue_context": build_fatigue_context(fatigued_ingredients),
        "liked_meals_context": build_liked_meals_context(liked_meals),
        "correction_context": build_correction_context(review_feedback),
        "pantry_correction_context": build_pantry_correction_context(form_data.get("_pantry_correction", "")),
        "time_context": build_time_context(),
        "variety_prompt": variety_prompt,
        "supplements_context": build_supplements_context(form_data),
        "grocery_duration_context": build_grocery_duration_context(form_data),
        "pantry_context": build_pantry_context(form_data),
        "prices_context": build_prices_context(),
        "taste_profile": taste_profile,
        "history_context": history_context,
    }


def _route_model(form_data: dict, attempt: int = 1, force_fast: bool = False) -> str:
    """Mejora 1: Ruteo Dinámico de Modelos (Cost/Latency Routing)."""
    if force_fast:
        return "gemini-3-flash-preview"

    # En reintentos, ya NO forzamos pro-preview: era demasiado lento (90s+ solo el planner)
    # y consumía el budget global antes de que los días pudieran terminar. Mantenemos el
    # ruteo dinámico por complejidad para retries también.
        
    medical = form_data.get("medicalConditions", [])
    allergies = form_data.get("allergies", [])
    # P1-A: leer del array canónico `dislikes` (mergeado con `otherDislikes` por
    # `_merge_other_text_fields` antes de cualquier nodo). El nombre legacy
    # `dislikedIngredients` quedó huérfano: ningún cliente lo escribe, por lo
    # que la rama "muchos dislikes → escalar a PRO" nunca se activaba para
    # usuarios reales y el ruteo de complejidad sólo escalaba vía medical/allergies.
    disliked = form_data.get("dislikes", [])
    
    # Consideramos "complejo" si tiene condiciones médicas serias, o muchas alergias
    is_complex = False
    if isinstance(medical, list) and len(medical) > 0:
        if isinstance(medical[0], str) and medical[0].lower() not in ["ninguna", "none", "n/a", ""]:
            is_complex = True
        elif not isinstance(medical[0], str):
            is_complex = True # Si no es string (ej. lista/dict anidado), asumimos complejidad
    elif isinstance(medical, str) and medical.lower() not in ["ninguna", "none", "n/a", ""]:
        is_complex = True
        
    if isinstance(allergies, list) and len(allergies) > 1:
        is_complex = True
    if isinstance(disliked, list) and len(disliked) > 3:
        is_complex = True
        
    if is_complex:
        print("🔀 [ROUTER] Perfil CLÍNICO complejo detectado. Enrutando a modelo PRO (gemini-3.1-pro-preview).")
        return "gemini-3.1-pro-preview"
    else:
        print("🔀 [ROUTER] Perfil FÁCIL detectado. Enrutando a modelo FLASH (gemini-3-flash-preview).")
        return "gemini-3-flash-preview"


# ============================================================
async def context_compression_node(state: PlanState) -> dict:
    """Mejora 4: Comprime el exceso de contexto para evitar Lost in the Middle."""
    history_context = state.get("history_context", "")
    
    # Solo comprimir si el contexto es demasiado largo (ej. > 2000 caracteres)
    if len(history_context) < 2000:
        return {"compressed_context": history_context}
        
    print(f"🗜️ [COMPRESIÓN] Contexto masivo detectado ({len(history_context)} caracteres). Comprimiendo...")
    
    from langchain_google_genai import ChatGoogleGenerativeAI

    # P1-Q3: capturar el modelo para usar el CB per-modelo
    _compressor_model = _route_model(state.get("form_data", {}), force_fast=True)
    _cb = _get_circuit_breaker(_compressor_model)
    compressor_llm = ChatGoogleGenerativeAI(
        model=_compressor_model,
        temperature=0.0,
        max_retries=1
    )
    
    prompt = f"""
Eres un sintetizador experto de historiales clínicos y de preferencias.
Se te proporciona un contexto de historial extremadamente largo sobre un usuario (reglas médicas, alergias, platos recientes, despensa, notas de adherencia).
Tu trabajo es comprimir todo esto en una lista estricta, condensada y clara de "Reglas Duras" y "Preferencias Suaves".

Elimina palabras innecesarias. Mantén OBLIGATORIAMENTE todas las alergias, condiciones médicas, ingredientes vetados y las advertencias de monotonía/fatiga.
No inventes nada nuevo. Solo resume lo proporcionado.

Contexto Original:
---
{history_context}
---

Devuelve ÚNICAMENTE el contexto comprimido en viñetas directas.
"""
    try:
        if not await _cb.acan_proceed():  # P1-Q3: CB per-modelo
            print("⚠️ [COMPRESIÓN] Circuit Breaker OPEN. Saltando compresión.")
            return {"compressed_context": history_context}

        # P0-4: Hard timeout con cancelación graceful (cleanup de sockets HTTP)
        result = await _safe_ainvoke(compressor_llm, prompt, timeout=30.0)
        await _cb.arecord_success()  # P1-Q3

        # Gemini 3.x puede devolver content como list[dict] (content blocks) o str.
        # Normalizar SIEMPRE a string para evitar crashes downstream (.lower(), .split(), etc.)
        raw = result.content
        if isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            compressed_text = "".join(parts).strip()
        elif isinstance(raw, str):
            compressed_text = raw.strip()
        else:
            compressed_text = str(raw).strip()

        # Sanity check: si la compresión devolvió algo absurdamente corto, usar el original
        if len(compressed_text) < 50:
            print(f"⚠️ [COMPRESIÓN] Resultado sospechoso ({len(compressed_text)} chars). Usando contexto original.")
            return {"compressed_context": history_context}

        print(f"🗜️ [COMPRESIÓN] Contexto reducido a {len(compressed_text)} caracteres.")
        return {"compressed_context": compressed_text}
    except Exception as e:
        await _cb.arecord_failure()  # P1-Q3: CB per-modelo
        print(f"⚠️ [COMPRESIÓN] Error comprimiendo contexto: {e}. Usando original.")
        return {"compressed_context": history_context}

# ============================================================
# NODO 1: PLANIFICADOR (Fase Map — esqueleto liviano)
# ============================================================
async def plan_skeleton_node(state: PlanState) -> dict:
    attempt = state.get("attempt", 1)
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    print(f"\n{'='*60}")
    print(f"📋 [PLANIFICADOR] Diseñando estructura del plan (intento #{attempt})...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "skeleton", "message": "Diseñando la estructura del plan..."})
    start_time = time.time()

    ctx = _build_shared_context(state, force_rebuild=True)
    _uid = ctx["user_id"]

    # Seleccionar técnicas de cocción
    succ_techs = form_data.get("successful_techniques", [])
    aban_techs = form_data.get("abandoned_techniques", [])
    
    # GAP 3: Mutación real en Retry — las técnicas del intento anterior se tratan como "abandonadas"
    if attempt > 1:
        previous_skeleton = state.get("plan_skeleton") or {}
        previous_techniques = previous_skeleton.get("_selected_techniques", [])
        aban_techs = list(set(aban_techs + previous_techniques))
        print(f"🔀 [RETRY MUTATION] Técnicas del intento anterior bloqueadas: {previous_techniques}")
        
    # [GAP 1 FIX] Bloquear la última técnica del chunk anterior si es continuación
    last_tech = form_data.get("_last_technique")
    if last_tech and attempt == 1:
        aban_techs = list(set(aban_techs + [last_tech]))
        print(f"🔄 [CHUNK CONTINUATION] Técnica anterior bloqueada para este chunk: {last_tech}")

    selected_techniques = _select_techniques(_uid, succ_techs, aban_techs)

    random_seed = random.randint(10000, 99999)

    days_offset = form_data.get("_days_offset", 0)
    # P1-10: datetime/timezone/timedelta a nivel módulo
    start_date_str = form_data.get("_plan_start_date")
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str)
        except Exception:
            start_date = datetime.now(timezone.utc)
    else:
        start_date = datetime.now(timezone.utc)

    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    # [GAP A FIX: Hacer el pipeline dinámico según _days_to_generate]
    # P1-10: PLAN_CHUNK_SIZE a nivel módulo
    days_in_chunk = form_data.get("_days_to_generate", PLAN_CHUNK_SIZE) or PLAN_CHUNK_SIZE
    days_in_chunk = max(1, days_in_chunk) # [GAP 1 FIX] Validar >= 1
    
    techniques_lines = []
    for i in range(days_in_chunk):
        global_day = days_offset + i + 1
        target_date = start_date + timedelta(days=global_day - 1)
        day_name = dias_es[target_date.weekday()]
        
        # Rotar técnicas cíclicamente si hay más días que técnicas base (que son 3)
        tech_idx = i % len(selected_techniques) if selected_techniques else 0
        tech = selected_techniques[tech_idx] if selected_techniques else 'Libre'
        techniques_lines.append(f"• Día {global_day} ({day_name}): {tech}")
        
    techniques_str = "\n".join(techniques_lines)

    prompt_text = (
        f"Analiza la siguiente información del usuario y diseña el ESQUELETO de un plan de {days_in_chunk} alternativas/días.\n"
        f"Semilla de generación aleatoria: {random_seed}\n\n"
        f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
        f"{ctx['quality_context']}\n{ctx['quality_hint_context']}\n{ctx['chunk_lessons_context']}\n{ctx['prev_chunk_adherence_context']}\n{ctx['weight_history_context']}\n{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
        f"{ctx['unified_behavioral_profile']}\n{ctx['correction_context']}\n{ctx['pantry_correction_context']}\n{ctx['history_context']}\n"
        f"{ctx['variety_prompt']}\n{ctx['pantry_context']}\n{ctx['prices_context']}\n"
        f"{ctx['adherence_context']}\n{ctx['success_patterns_context']}\n"
        f"{ctx['temporal_adherence_context']}\n"
        f"{ctx['motivation_context']}\n\n"
        f"Técnicas de cocción asignadas (una por día):\n"
        f"{techniques_str}\n\n"
        f"{PLANNER_SYSTEM_PROMPT}"
    )

    if state.get("reflection_directive"):
        prompt_text = f"🧠 DIRECTIVA META-LEARNING (PRIORIDAD MÁXIMA):\n{state['reflection_directive']}\n\n" + prompt_text

    rejection_reasons = state.get("rejection_reasons", [])
    if attempt > 1 and rejection_reasons:
        prompt_text += f"\n\n🚨 ATENCIÓN (INTENTO {attempt}): El intento anterior fue RECHAZADO por estas razones:\n"
        for reason in rejection_reasons:
            prompt_text += f" - {reason}\n"
        prompt_text += "\nDEBES MUTAR DRÁSTICAMENTE tu estrategia anterior. Cambia los ingredientes seleccionados, simplifica la preparación si se te pidió, y evita repetir los patrones fallidos."

    if form_data.get("_pipeline_drift_alert"):
        prompt_text += "\n\n🚨 ALERTA DE AUTO-OPTIMIZACIÓN: El sistema ha detectado que tus últimos planes requirieron múltiples correcciones. DEBES SER EXTREMADAMENTE ESTRICTO y generar un plan seguro, coherente y visualmente atractivo desde el primer intento."

    if form_data.get("_auto_simplify"):
        prompt_text += "\n\n⚠️ INSTRUCCIÓN DE SIMPLIFICACIÓN: Los planes complejos recientes han sido rechazados médicamente. Usa ingredientes BÁSICOS y métodos de cocción SENCILLOS. Minimiza el riesgo de desbalance."

    if days_offset > 0:
        # Mostramos el rango real a generar.
        prior_meals = form_data.get("_chunk_prior_meals", [])
        # Mantener lista corta para no saturar el prompt (25 platos máx, ~300 tokens)
        prior_meals_str = ", ".join(prior_meals[-25:]) if prior_meals else "no disponibles"
        prompt_text = (
            f"\n\n🗓️ GENERACIÓN EN BACKGROUND — CONTINUACIÓN DEL PLAN:\n"
            f"Estás generando los DÍAS {days_offset + 1} al {days_offset + days_in_chunk} de un plan de largo plazo.\n"
            f"Los días 1 al {days_offset} ya fueron generados y el usuario los está viendo.\n"
            f"Platos ya generados en días previos (NO repetir ninguno): {prior_meals_str}\n"
            f"REGLA OBLIGATORIA: Numera los días comenzando desde {days_offset + 1}. "
            f"Garantiza variedad total respecto a TODOS los días previos, no solo la última semana.\n\n"
        ) + prompt_text

    is_re_roll = form_data.get("_is_same_day_reroll", False)
    
    # Mutación estratégica de temperatura: 0.7 en intento 1, 0.9 en intento 2 para forzar variabilidad extrema
    weak_signal_mod = -0.1 if form_data.get("_chunk_lessons", {}).get("weak_signal") else 0.0
    base_temp = (0.95 if is_re_roll else (0.7 if attempt == 1 else 0.9)) + weak_signal_mod

    # Ruteo dinámico por complejidad (antes se forzaba pro-preview en retry pero era
    # demasiado lento — consumía budget y bloqueaba la generación paralela).
    planner_model = _route_model(form_data, attempt)
    if attempt > 1:
        print(f"🔀 [RETRY MUTATION] Modelo '{planner_model}' + temp={base_temp} para intento {attempt}")

    planner_llm = ChatGoogleGenerativeAI(
        model=planner_model,
        temperature=base_temp,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=0,
        timeout=45
    ).with_structured_output(PlanSkeletonModel)

    # P1-Q3: CB per-modelo. Si pro está saturado, perfiles complejos paran;
    # los simples (que usan flash) no se ven afectados.
    _planner_cb = _get_circuit_breaker(planner_model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [PLANIFICADOR] Reintento #{retry_state.attempt_number}...")
    )
    async def invoke_planner():
        if not await _planner_cb.acan_proceed():
            raise Exception(f"Circuit Breaker OPEN para {planner_model} - LLM cascade failure prevented")
        try:
            print(f"⏳ [PLANIFICADOR] Generando esqueleto del plan...")
            # P0-4: Hard timeout con cancelación graceful. El constructor pone
            # timeout=45 pero el SDK no siempre lo respeta con sockets colgados.
            # tenacity hará 3 retries, con cap explícito de 50s/intento mantenemos
            # el peor caso ~150s. _safe_ainvoke garantiza cleanup del socket.
            res = await _safe_ainvoke(planner_llm, prompt_text, timeout=50.0)
            await _planner_cb.arecord_success()
            return res
        except Exception as e:
            await _planner_cb.arecord_failure()
            raise e

    response = await invoke_planner()

    duration = round(time.time() - start_time, 2)
    print(f"✅ [PLANIFICADOR] Esqueleto generado en {duration}s")

    if hasattr(response, "model_dump"):
        skeleton = response.model_dump()
    elif isinstance(response, dict):
        skeleton = response
    else:
        skeleton = response.dict()

    # Guardar técnicas seleccionadas para persistencia
    skeleton["_selected_techniques"] = selected_techniques

    # ── Scrub determinista del skeleton: enforce caps de proteínas restringidas ──
    # El planner LLM a veces ignora el cap del prompt e incluye atún/embutidos en múltiples
    # días. Aquí lo enforzamos a nivel estructural antes de que los workers reciban el pool.
    _SKELETON_RESTRICTED = ['atún', 'atun', 'salami', 'longaniza', 'chorizo']
    _EMBUTIDO_KEYS = ['salami', 'longaniza', 'chorizo']

    skel_days = skeleton.get('days', [])

    # 1. Cada proteína restringida aparece en MÁXIMO 1 día — remover del resto
    for restricted in _SKELETON_RESTRICTED:
        days_with_it = [
            i for i, d in enumerate(skel_days)
            if any(restricted in (p or '').lower() for p in d.get('protein_pool', []))
        ]
        if len(days_with_it) > 1:
            keep_idx = days_with_it[0]
            for idx in days_with_it[1:]:
                original = skel_days[idx].get('protein_pool', [])
                filtered = [p for p in original if restricted not in (p or '').lower()]
                removed = [p for p in original if p not in filtered]
                if removed:
                    skel_days[idx]['protein_pool'] = filtered
                    print(f"🧹 [SKELETON SCRUB] Día {skel_days[idx].get('day')}: "
                          f"eliminadas {removed} (cap '{restricted}' ya asignado al Día {skel_days[keep_idx].get('day')})")

    # 2. No combinar atún + embutido en el mismo día — remover embutidos si coexisten con atún
    for d in skel_days:
        pool = d.get('protein_pool', [])
        pool_lower = ' '.join((p or '').lower() for p in pool)
        has_atun = 'atún' in pool_lower or 'atun' in pool_lower
        embutidos_in_pool = [p for p in pool if any(emb in (p or '').lower() for emb in _EMBUTIDO_KEYS)]
        if has_atun and embutidos_in_pool:
            d['protein_pool'] = [p for p in pool if p not in embutidos_in_pool]
            print(f"🧹 [SKELETON SCRUB] Día {d.get('day')}: eliminados embutidos "
                  f"{embutidos_in_pool} (conflicto con atún presente)")

    # 3. Fallback: si algún pool quedó vacío tras scrub, inyectar Lentejas como proteína segura
    for d in skel_days:
        if not d.get('protein_pool'):
            d['protein_pool'] = ['Lentejas']
            print(f"⚠️ [SKELETON SCRUB] Día {d.get('day')}: protein_pool vacío tras scrub, "
                  f"inyectado 'Lentejas' como fallback")

    _emit_progress(state, "metric", {
        "node": "plan_skeleton",
        "duration_ms": int(duration * 1000),
        "retries": attempt - 1,
        "tokens_estimated": len(prompt_text) // 4,
        "confidence": 0.95
    })

    return {
        "plan_skeleton": skeleton,
        "attempt": attempt,
        "_cached_context": ctx
    }


# ============================================================
# NODO 2: GENERADORES PARALELOS (Fase Reduce — 3 días simultáneos)
# ============================================================
async def generate_days_parallel_node(state: PlanState) -> dict:
    """Genera los 7 días completos en PARALELO usando el esqueleto del planificador."""
    skeleton = state["plan_skeleton"]
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    print(f"\n{'='*60}")
    print(f"🚀 [GENERADORES PARALELOS] Lanzando 3 workers para generar las opciones...\n")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "parallel_generation", "message": "Generando las 3 opciones en paralelo..."})

    # GAP 5: Invalidador de caché en retries
    ctx = _build_shared_context(state, force_rebuild=state.get("attempt", 1) > 1)
    skeleton_days = skeleton.get("days", [])

    is_re_roll = form_data.get("_is_same_day_reroll", False)
    weak_signal_mod = -0.1 if form_data.get("_chunk_lessons", {}).get("weak_signal") else 0.0
    base_temp = (0.95 if is_re_roll else 0.7) + weak_signal_mod
    attempt = state.get("attempt", 1)

    # --- INICIO MEJORA 2: Extraer días reciclados ANTES del worker para inyectar contexto ---
    # P1-10: PLAN_CHUNK_SIZE a nivel módulo
    days_in_chunk = form_data.get("_days_to_generate", PLAN_CHUNK_SIZE) or PLAN_CHUNK_SIZE
    days_in_chunk = max(1, days_in_chunk) # [GAP 1 FIX] Validar >= 1
    affected_days = state.get("_affected_days") or []
    surgical_mode = attempt > 1 and affected_days and len(affected_days) < len(skeleton_days[:days_in_chunk])
    
    recycled_days_context = ""
    recycled_days_cache = {} # day_num -> recycled_day
    
    if surgical_mode:
        print(f"🔪 [SURGICAL FIX] Reciclando días válidos. Regenerando SOLO los días afectados: {affected_days}")
        previous_days = state.get("plan_result", {}).get("days", [])
        for i, skel_day in enumerate(skeleton_days[:days_in_chunk]):
            day_num = i + 1
            if day_num not in affected_days:
                r_day = next((d for d in previous_days if d.get("day") == day_num), None)
                if r_day:
                    recycled_days_cache[day_num] = r_day
                    
        if recycled_days_cache:
            recycled_summary = []
            for d in recycled_days_cache.values():
                meals_summary = []
                for m in d.get("meals", []):
                    meals_summary.append(f"{m.get('meal')}: {m.get('name')} (Ingredientes: {', '.join(m.get('ingredients', []))})")
                recycled_summary.append(f"DÍA {d.get('day')} (YA APROBADO):\n" + "\n".join(f"  - {ms}" for ms in meals_summary))
            
            recycled_days_context = (
                "\n\n🚨 [ATENCIÓN: CORRECCIÓN QUIRÚRGICA] 🚨\n"
                "ESTOS DÍAS YA FUERON GENERADOS Y APROBADOS. PARA EVITAR QUE EL PLAN FALLE POR REPETICIÓN, "
                "DEBES ASEGURARTE DE NO REPETIR LAS MISMAS PROTEÍNAS PRINCIPALES O PREPARACIONES DE LOS SIGUIENTES DÍAS:\n"
                + "\n".join(recycled_summary) + 
                "\n------------------------------------------------------------\n"
            )
    # --- FIN MEJORA 2 ---

    async def generate_single_day(skeleton_day: dict, day_num: int, temp_override: float = None) -> dict:
        """Worker: genera un solo día completo."""
        _emit_progress(state, "day_started", {"day": day_num, "message": f"Generando Día {day_num}..."})
        day_start = time.time()

        days_offset = form_data.get("_days_offset", 0)
        global_day = days_offset + day_num

        # P1-10: datetime/timezone/timedelta están a nivel módulo
        start_date_str = form_data.get("_plan_start_date")
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
            except Exception:
                start_date = datetime.now(timezone.utc)
        else:
            start_date = datetime.now(timezone.utc)

        dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        target_date = start_date + timedelta(days=global_day - 1)
        day_name = dias_es[target_date.weekday()]

        assignment_context = build_day_assignment_context(skeleton_day, day_num, day_name=day_name)

        random_seed = random.randint(10000, 99999)

        prompt_text = (
            f"Genera las comidas completas para el DÍA {day_num} del plan.\n"
            f"Semilla de generación aleatoria: {random_seed}\n\n"
            f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
            f"{ctx['quality_context']}\n{ctx['quality_hint_context']}\n{ctx['chunk_lessons_context']}\n{ctx['prev_chunk_adherence_context']}\n"
            f"{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
            f"{ctx['unified_behavioral_profile']}\n{ctx['correction_context']}\n{ctx['pantry_correction_context']}\n"
            f"{ctx['supplements_context']}\n{ctx['grocery_duration_context']}\n"
            f"{ctx['pantry_context']}\n{ctx['adherence_context']}\n{ctx['success_patterns_context']}\n"
            f"{ctx['temporal_adherence_context']}\n"
            f"{ctx['motivation_context']}\n"
            f"{assignment_context}\n"
            f"{recycled_days_context}\n"
            f"{DAY_GENERATOR_SYSTEM_PROMPT}"
        )

        day_model = _route_model(form_data, attempt)
        # P1-Q3: CB per-modelo. Cada día puede usar pro o flash según routing;
        # la salud de cada modelo se rastrea independientemente.
        _day_cb = _get_circuit_breaker(day_model)

        day_llm = ChatGoogleGenerativeAI(
            model=day_model,
            temperature=temp_override if temp_override is not None else (base_temp if attempt == 1 else (base_temp + 0.1)),
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_retries=0,
            timeout=90
        )

        # Enlazar herramientas (Mejora 3)
        from tools_nutrition import NUTRITION_TOOLS, consultar_nutricion
        day_llm_with_tools = day_llm.bind_tools(NUTRITION_TOOLS)

        # Inject the schema strictly so the model returns raw JSON
        schema_dict = SingleDayPlanModel.model_json_schema()
        streaming_prompt = prompt_text + f"\n\nDEBES DEVOLVER ESTRICTAMENTE UN JSON VÁLIDO QUE CUMPLA CON ESTE ESQUEMA (NO incluyas bloques de markdown como ```json):\n{json.dumps(schema_dict)}"

        # P1-10: HumanMessage/AIMessage/ToolMessage están a nivel módulo

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
            before_sleep=lambda rs: print(f"⚠️  [DÍA {day_num}] Reintento #{rs.attempt_number}...")
        )
        async def invoke_day():
            if not await _day_cb.acan_proceed():
                raise Exception(f"Circuit Breaker OPEN para {day_model} - LLM cascade failure prevented")
            try:
                messages = [HumanMessage(content=streaming_prompt)]
                accumulated_json = ""

                # Agent loop (hasta 4 interacciones de herramientas permitidas)
                for _agent_iter in range(4):
                    # P0-3: Double-check rápido en cada iteración. Cerramos la
                    # ventana de race entre N días paralelos: si otro worker
                    # registró fallo mientras esperábamos el semáforo del LLM
                    # o en una iteración anterior, abortamos antes de invocar.
                    # `_is_locally_unhealthy_fresh()` es ~1µs (sin I/O).
                    if _day_cb._is_locally_unhealthy_fresh():  # P1-Q3
                        raise Exception(f"Circuit Breaker tripped during agent loop (P0-3) para {day_model}")
                    ai_msg = None
                    async for chunk in day_llm_with_tools.astream(messages):
                        if chunk.content:
                            c_text = ""
                            if isinstance(chunk.content, list):
                                for c_block in chunk.content:
                                    if isinstance(c_block, dict) and "text" in c_block:
                                        c_text += c_block["text"]
                                    elif isinstance(c_block, str):
                                        c_text += c_block
                            else:
                                c_text = str(chunk.content)

                            if c_text:
                                accumulated_json += c_text
                                _emit_progress(state, "token", {"day": day_num, "chunk": c_text})

                        if chunk.tool_call_chunks:
                            for tcc in chunk.tool_call_chunks:
                                if tcc.get("name"):
                                    _emit_progress(state, "tool_call", {"day": day_num, "tool": tcc.get("name")})

                        if ai_msg is None:
                            ai_msg = chunk
                        else:
                            ai_msg += chunk

                    if ai_msg and ai_msg.tool_calls:
                        messages.append(ai_msg)

                        for tool_call in ai_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            if tool_name == "consultar_nutricion":
                                print(f"🔧 [DÍA {day_num}] Tool Call: consultar_nutricion({tool_args})")
                                try:
                                    # P0-3: Despachar la tool sync a thread pool. Aunque la
                                    # tool actual es CPU-puro (lookup en dict), el agent loop
                                    # corre con N días en paralelo + retries: cualquier
                                    # evolución a I/O (USDA API, Supabase) congelaría el
                                    # event loop de todos los workers simultáneamente.
                                    result = await asyncio.to_thread(consultar_nutricion.invoke, tool_args)
                                except Exception as e:
                                    result = f"Error ejecutando herramienta: {str(e)}"
                                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

                        _emit_progress(state, "token_reset", {"day": day_num})
                        accumulated_json = ""
                        continue
                    else:
                        # Generación finalizada sin más tools
                        break
                        
                await _day_cb.arecord_success()  # P1-Q3

                # Parse the accumulated JSON
                clean_json = accumulated_json.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                parsed_dict = json.loads(clean_json)
                return SingleDayPlanModel(**parsed_dict).model_dump()
            except Exception as e:
                await _day_cb.arecord_failure()  # P1-Q3
                raise e

        day_result = await invoke_day()

        # Forzar day number y name correctos
        day_result["day"] = day_num
        day_result["day_name"] = day_name

        # ── Scrub determinista de proteínas restringidas no autorizadas ──
        # Elimina de `ingredients` cualquier proteína restringida que el LLM
        # haya añadido como complemento sin que el planner la asignara.
        # No toca `recipe` para no crear nuevos errores de bidireccionalidad;
        # el revisor médico actúa como red de seguridad si la recipe la menciona.
        from prompts.day_generator import _RESTRICTED_PROTEIN_KEYS
        _pool_lower = ' '.join(skeleton_day.get('protein_pool', [])).lower()
        _unauthorized_keys = {k for k in _RESTRICTED_PROTEIN_KEYS if k not in _pool_lower}

        if _unauthorized_keys:
            for _meal in day_result.get('meals', []):
                _original = _meal.get('ingredients', [])
                _cleaned, _removed = [], []
                for _ing in _original:
                    _ing_lower = _ing.lower()
                    if any(_uk in _ing_lower for _uk in _unauthorized_keys):
                        _removed.append(_ing)
                    else:
                        _cleaned.append(_ing)
                if _removed:
                    _meal['ingredients'] = _cleaned
                    print(f"🚫 [DÍA {day_num}] Proteínas no autorizadas eliminadas de "
                          f"'{_meal.get('name')}': {_removed}")

        # ── Scrub de unidades prohibidas ("ramita", "pizca", "chorrito", etc.) ──
        # La regla 8 del day_generator las prohíbe, pero el LLM las incluye a veces.
        # El revisor médico las flagea porque su parser trata la unidad como nombre del
        # ingrediente. Aquí las normalizamos a unidades medibles reales.
        _UNIT_REPLACEMENTS = [
            (r'\bramitas?\b',    'cda'),   # ramita/ramitas → cda
            (r'\bpizcas?\b',     'cdta'),  # pizca/pizcas → cdta
            (r'\bchorritos?\b',  'cda'),   # chorrito → cda
            (r'\btoques?\b',     'cdta'),  # toque → cdta
            (r'\bchin\b',        'cdta'),  # chin → cdta
            (r'\bpuñados?\b',    'cda'),   # puñado → cda
            (r'\bhojitas?\b',    'cdta'),  # hojita → cdta
        ]
        for _meal in day_result.get('meals', []):
            _new_ings = []
            for _ing in _meal.get('ingredients', []):
                _scrubbed = _ing
                for _pattern, _repl in _UNIT_REPLACEMENTS:
                    _scrubbed = _re.sub(_pattern, _repl, _scrubbed, flags=_re.IGNORECASE)
                if _scrubbed != _ing:
                    print(f"🧹 [DÍA {day_num}] Unidad normalizada: '{_ing}' → '{_scrubbed}'")
                _new_ings.append(_scrubbed)
            _meal['ingredients'] = _new_ings

        day_duration = round(time.time() - day_start, 2)
        print(f"✅ [DÍA {day_num}] Generado en {day_duration}s")
        _emit_progress(state, "day_completed", {"day": day_num})

        _emit_progress(state, "metric", {
            "node": f"generate_day_{day_num}",
            "duration_ms": int(day_duration * 1000),
            "retries": attempt - 1,
            "tokens_estimated": len(prompt_text) // 4,
            "confidence": 0.90
        })

        return day_result

    # ── Hedged execution per-day ──
    # Si un día tarda >hedge_after segundos, se lanza un retry especulativo en paralelo.
    # Gana el primero que termine. Ceiling duro por día: HARD_CEILING segundos.
    # Esto elimina el patrón anterior "120s timeout → cancelar → retry sequential" que
    # acumulaba ~90s extra cuando un día era lento.
    #
    # P0-D: hedge_after escala con days_in_chunk para no saturar el LLM_SEMAPHORE
    # (max_concurrent=4 distribuido). Análisis del problema:
    #   - Para chunks de 7 días: 7 primaries simultáneos compitiendo por 4 slots
    #     → 3 quedan en queue. A los 45s, 7 hedges adicionales se sumaban
    #     → 14 tasks compitiendo por 4 slots. Tareas excedían HARD_CEILING por
    #     starvation, no por lentitud real del LLM.
    #   - Adversarial play duplica la concurrencia (×2 candidates), agravando.
    # Fix:
    #   - hedge_after_base=45s para chunks ≤4d. Cada día adicional suma 10s
    #     (5d=55s, 7d=75s, 14d adversarial=135s). Da tiempo a primaries en
    #     queue para liberar slots antes de inyectar hedges.
    #   - HEDGE_MAX_CONCURRENT cap absoluto: solo 2 hedges concurrentes por
    #     pipeline. Si el counter ya está en el cap, el día espera SOLO al
    #     primary hasta HARD_CEILING. Garantiza que primaries en queue tengan
    #     slots libres del semáforo.
    # P1-NEW-2: knobs configurables vía `MEALFIT_HEDGE_AFTER_BASE_S`
    # y `MEALFIT_HARD_CEILING_S`. Defaults idénticos al hardcode previo.
    # Las variables locales mantienen los nombres originales para no romper
    # el resto de la lógica/logs del nodo.
    HEDGE_AFTER_BASE = HEDGE_AFTER_BASE_S
    HARD_CEILING = HARD_CEILING_S
    hedge_after = HEDGE_AFTER_BASE + max(0, days_in_chunk - 4) * 10.0
    HEDGE_MAX_CONCURRENT = max(1, LLM_SEMAPHORE.max_concurrent // 2)
    # Counter compartido entre todos los `_generate_day_hedged` del nodo.
    # Lista de un elemento para mutar desde el closure sin `nonlocal` (que
    # rompería si en futuro se mueve la fn fuera de generate_days_parallel_node).
    hedge_in_flight = [0]

    async def _generate_day_hedged(skel_day: dict, day_num: int, temp_override: float = None) -> dict:
        day_start_time = time.time()
        primary = asyncio.create_task(generate_single_day(skel_day, day_num, temp_override))

        # 1. Esperar primary hasta el soft timeout (adaptativo P0-D)
        done, _ = await asyncio.wait({primary}, timeout=hedge_after)
        if primary in done:
            return primary.result()  # propaga excepciones normales

        elapsed = round(time.time() - day_start_time, 1)

        # 2. P0-D: chequear cap de hedges concurrentes ANTES de lanzar.
        # asyncio es single-threaded por loop → check + increment es atómico.
        if hedge_in_flight[0] >= HEDGE_MAX_CONCURRENT:
            print(f"⚠️ [HEDGE] Día {day_num} lleva >{elapsed}s pero "
                  f"hedge limiter saturado ({hedge_in_flight[0]}/{HEDGE_MAX_CONCURRENT} "
                  f"hedges en flight). Esperando solo primary hasta HARD_CEILING.")
            remaining = max(5.0, HARD_CEILING - (time.time() - day_start_time))
            done, _ = await asyncio.wait({primary}, timeout=remaining)
            if primary in done:
                return primary.result()  # propaga excepciones normales del primary
            primary.cancel()
            try:
                await primary
            except BaseException:
                pass
            raise TimeoutError(
                f"Día {day_num}: primary excedió ceiling de {HARD_CEILING}s "
                f"(hedge skipped por límite P0-D de concurrencia)"
            )

        # 3. Hay slot disponible → reservar e inyectar hedge especulativo
        hedge_in_flight[0] += 1
        print(f"🪁 [HEDGE] Día {day_num} lleva >{elapsed}s. Lanzando intento especulativo "
              f"({hedge_in_flight[0]}/{HEDGE_MAX_CONCURRENT} hedges activos).")
        hedge = asyncio.create_task(generate_single_day(skel_day, day_num, temp_override))

        racing = {primary, hedge}
        last_exc: Optional[Exception] = None

        try:
            while racing:
                remaining = max(5.0, HARD_CEILING - (time.time() - day_start_time))
                done, pending = await asyncio.wait(
                    racing, return_when=asyncio.FIRST_COMPLETED, timeout=remaining
                )
                if not done:
                    # Ambos pasaron el ceiling duro — cancelar y fallar
                    for t in pending:
                        t.cancel()
                    raise TimeoutError(
                        f"Día {day_num}: primary y hedge excedieron ceiling de {HARD_CEILING}s"
                    )

                for t in done:
                    racing.discard(t)
                    try:
                        result = t.result()
                        # Ganador — cancelar al perdedor
                        for pt in pending:
                            pt.cancel()
                        winner = "primary" if t is primary else "hedge"
                        print(f"🏁 [HEDGE] Día {day_num} ganador: {winner}")
                        return result
                    except Exception as e:
                        last_exc = e
                        name = "primary" if t is primary else "hedge"
                        print(f"⚠️ [HEDGE] Día {day_num}: {name} falló ({type(e).__name__}). Esperando al otro.")
                racing = pending

            raise last_exc or RuntimeError(f"Día {day_num}: ambos intentos fallaron sin excepción")
        finally:
            # P0-D: liberar slot SIEMPRE (éxito, falla o cancelación) para que
            # otros días puedan hedge. El finally se ejecuta antes de que
            # cualquier excepción propague al caller.
            hedge_in_flight[0] -= 1

    # Lanzar generaciones en paralelo (con hedging per-day)
    parallel_start = time.time()
    generated_days = []

    async def _safe_gen(skel_day, day_num, temp_override=None):
        try:
            result = await _generate_day_hedged(skel_day, day_num, temp_override)
            return day_num, result, None
        except Exception as e:
            return day_num, None, e
        finally:
            # P1-B: flush buffer de tokens del día. Cubre TODOS los paths:
            # success normal, excepción capturada (TimeoutError del HARD_CEILING,
            # retry exhausted en invoke_day), y cancelación cooperativa (hedge
            # ganador cancela al perdedor mid-stream). Sin esto, los tokens
            # acumulados desde el último flush automático se perdían y la UI
            # mostraba texto cortado al final del día fallido. Idempotente:
            # si `day_completed` ya flusheó por la ruta normal, noop.
            _flush_token_buffer_for_day(state, day_num)

    use_adversarial = form_data.get("_use_adversarial_play", False)

    # --- P1: AUTO-ACTIVACIÓN AUTÓNOMA del Adversarial Self-Play ---
    # Si el flag no fue seteado explícitamente, decidimos basándonos en la salud del pipeline.
    if not use_adversarial and attempt == 1:
        health_profile = form_data.get("health_profile", {})
        _auto_reasons = []

        # Condición 1: Quality Score bajo sostenido (< 0.5 por 2+ ciclos)
        qh = health_profile.get("quality_history_chunks", [])
        if isinstance(qh, list) and len(qh) >= 2 and all(s < 0.5 for s in qh[-2:]):
            _auto_reasons.append(f"quality_low_sustained (últimos 2: {qh[-2:]})")

        # Condición 2: Alta varianza en Attribution Tracker (el sistema no sabe qué funciona)
        attr_tracker = health_profile.get("attribution_tracker", {})
        if len(attr_tracker) >= 3:
            scores = [v.get("avg_score", 0.5) for v in attr_tracker.values() if isinstance(v, dict)]
            if scores:
                mean_s = sum(scores) / len(scores)
                variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                if variance > 0.06:
                    _auto_reasons.append(f"attribution_high_variance ({variance:.3f})")

        # Condición 3: Historial de rechazos médicos frecuentes
        rejection_patterns = health_profile.get("rejection_patterns", [])
        if isinstance(rejection_patterns, list) and len(rejection_patterns) >= 5:
            _auto_reasons.append(f"frequent_rejections ({len(rejection_patterns)} patterns)")

        if _auto_reasons:
            use_adversarial = True
            print(f"⚔️ [ADVERSARIAL AUTO-ACTIVATE] Activado autónomamente por: {', '.join(_auto_reasons)}")

    # --- GAP 2: Desactivar adversarial si la calibración es mala ---
    health_profile = form_data.get("health_profile", {})
    judge_calib = health_profile.get("judge_calibration", {})
    if judge_calib.get("total", 0) >= 5 and judge_calib.get("score", 1.0) < 0.5:
        print("🛑 [ADVERSARIAL JUDGE] Desactivado automáticamente. Score de calibración muy bajo (< 0.5).")
        use_adversarial = False

    # Si es el intento 2 (retry tras revisión médica), desactivar adversarial por tiempo.
    if attempt > 1:
        use_adversarial = False

    async def _generate_candidate(temp_override=None):
        generated_days = []
        day_coros = []
        for i, skel_day in enumerate(skeleton_days[:days_in_chunk]):
            day_num = i + 1

            if surgical_mode and day_num not in affected_days:
                recycled_day = recycled_days_cache.get(day_num)
                if recycled_day:
                    generated_days.append(recycled_day)
                    if temp_override is None or temp_override == 0.7:
                        print(f"♻️ [SURGICAL FIX] Día {day_num} reciclado con éxito.")
                    continue
                else:
                    if temp_override is None or temp_override == 0.7:
                        print(f"⚠️ [SURGICAL FIX] No se encontró el Día {day_num} para reciclar, forzando regeneración.")

            day_coros.append(_safe_gen(skel_day, day_num, temp_override))

        failed_days = []
        if day_coros:
            if len(day_coros) == 1:
                results = [await day_coros[0]]
            else:
                results = await asyncio.gather(*day_coros)
                
            for day_num, result, err in results:
                if result is not None:
                    generated_days.append(result)
                else:
                    err_name = type(err).__name__ if err else "unknown"
                    print(f"❌ [DÍA {day_num}] Falló definitivamente tras hedging: {err_name}")
                    failed_days.append(day_num)

        if failed_days and generated_days:
            # P1-10: copy ya está importado a nivel módulo
            valid_day_template = generated_days[0]
            for f_day in failed_days:
                cloned = copy.deepcopy(valid_day_template)
                cloned["day"] = f_day
                generated_days.append(cloned)
                print(f"⚠️ [FALLBACK EXTREMO] Día {f_day} clonado a partir del Día {valid_day_template.get('day', 1)}")
        elif failed_days and not generated_days:
            # P0-2: Evitar propagar días vacíos si todos los workers fallan
            raise RuntimeError("Todos los workers de generación de días fallaron. Forzando fallback matemático global.")

        generated_days.sort(key=lambda d: d.get("day", 0))
        return generated_days

    candidate_a = None
    candidate_b = None

    _ab_pair_selected = _AB_TEMP_PAIRS[1]  # default: balanced
    if use_adversarial:
        _ab_user_id = form_data.get("user_id", "guest")
        # P0-NEW-1.a: variante async — antes `_select_ab_temp_pair` (sync) bloqueaba
        # el event loop ~30-150ms en cada generación adversarial, retrasando todos
        # los SSE callbacks de los workers paralelos del nodo.
        _ab_pair_selected = await _aselect_ab_temp_pair(_ab_user_id) if _ab_user_id != "guest" else _AB_TEMP_PAIRS[1]
        _ab_temp_a = _ab_pair_selected["temp_a"]
        _ab_temp_b = _ab_pair_selected["temp_b"]
        print(f"⚔️ [ADVERSARIAL SELF-PLAY] Activado. Par '{_ab_pair_selected['label']}': A={_ab_temp_a} / B={_ab_temp_b}")
        cand_a_coro = _generate_candidate(temp_override=_ab_temp_a)
        cand_b_coro = _generate_candidate(temp_override=_ab_temp_b)
        # P1-2: return_exceptions=True para que un fallo en uno no aborte el otro.
        # Antes, si cand_b_coro fallaba, asyncio.gather propagaba la excepción y
        # candidate_a (que pudo haber tenido éxito) se perdía. Ahora degradamos
        # graciosamente a single-candidate si solo falla uno.
        results = await asyncio.gather(cand_a_coro, cand_b_coro, return_exceptions=True)
        candidate_a, candidate_b = results
        a_failed = isinstance(candidate_a, BaseException)
        b_failed = isinstance(candidate_b, BaseException)
        if a_failed and b_failed:
            print(f"❌ [ADVERSARIAL] Ambos candidatos fallaron. "
                  f"A={type(candidate_a).__name__}, B={type(candidate_b).__name__}. "
                  "Re-raise para activar fallback global.")
            raise candidate_a  # consistente con _generate_candidate cuando todo falla
        if a_failed:
            print(f"⚠️ [ADVERSARIAL] Candidato A falló ({type(candidate_a).__name__}); "
                  "promoviendo Candidato B como único.")
            candidate_a = candidate_b
            candidate_b = None
        elif b_failed:
            print(f"⚠️ [ADVERSARIAL] Candidato B falló ({type(candidate_b).__name__}); "
                  "continuando solo con Candidato A.")
            candidate_b = None
    else:
        candidate_a = await _generate_candidate(temp_override=None)

    parallel_duration = round(time.time() - parallel_start, 2)
    print(f"✅ [PARALELO] Días generados en {parallel_duration}s")

    return {
        "candidate_a": {
            "days": candidate_a,
            "_skeleton": skeleton,
            "_parallel_duration": parallel_duration,
        },
        "candidate_b": {
            "days": candidate_b,
            "_skeleton": skeleton,
            "_parallel_duration": parallel_duration,
        } if candidate_b else None,
        "plan_result": {
            "days": candidate_a,
            "_skeleton": skeleton,
            "_parallel_duration": parallel_duration,
        },
        "_ab_temp_meta": _ab_pair_selected,
    }


# ============================================================
# A/B ONLINE PARA TEMPERATURA DEL ADVERSARIAL JUDGE
# ============================================================
_AB_TEMP_PAIRS = [
    {"label": "conservative", "temp_a": 0.5,  "temp_b": 0.7},
    {"label": "balanced",     "temp_a": 0.7,  "temp_b": 0.95},
    {"label": "adventurous",  "temp_a": 0.8,  "temp_b": 1.0},
]
_AB_MIN_SAMPLES_PER_PAIR = 5  # explorar cada par al menos N veces antes de converger

# P1-Q5: query incluye age_days computado en DB (más eficiente que parsear
# timestamps en Python por fila) y baja el LIMIT de 90 → 50. Con decay
# exponencial las muestras >30 días aportan ≤22% del peso de una reciente,
# así que LIMIT mayor solo añadía ruido sin valor decisional.
_AB_TEMP_QUERY = (
    "SELECT metadata->>'pair_label' AS pair_label, "
    "metadata->>'winner' AS winner, "
    "EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days "
    "FROM pipeline_metrics "
    "WHERE user_id = %s "
    "  AND node = 'adversarial_judge' "
    "  AND metadata->>'pair_label' IS NOT NULL "
    "ORDER BY created_at DESC LIMIT 50"
)

# P1-Q5: factor de decay exponencial por antigüedad en días.
# - 0.95**14 ≈ 0.49  (2 semanas → ~50% del peso)
# - 0.95**30 ≈ 0.21  (1 mes → ~21%)
# - 0.95**60 ≈ 0.05  (2 meses → ~5%, aporte marginal)
# Más conservador que `_select_techniques` (0.9) porque las muestras del A/B
# son escasas (~50 max por usuario) y queremos preservar señal histórica
# sin perder responsividad al cambio reciente.
_AB_TEMP_DECAY_FACTOR = 0.95


def _compute_ab_temp_pair_from_rows(rows: list) -> dict:
    """P0-NEW-1.a: Lógica pura de Thompson Sampling sin I/O.

    Extraída de `_select_ab_temp_pair` para que la decisión sea idéntica entre
    la versión sync (legacy) y async (`_aselect_ab_temp_pair`). Recibe las
    filas ya cargadas de pipeline_metrics y devuelve el par seleccionado.

    P1-Q5: cada win se pondera por `_AB_TEMP_DECAY_FACTOR ** age_days`. Antes
    el sampler contaba todas las observaciones por igual: un usuario con 90
    pipelines a lo largo de 6 meses veía pesos uniformes — el sampler tardaba
    eternamente en responder a cambios de patrón conductual (ej. usuario que
    perdió peso, ganó motivación, cambió de país/estación) porque las
    preferencias antiguas dominaban. Con decay 0.95/día, observaciones >30
    días aportan ≤22% y >60 días <5%, así el sampler converge rápido a
    cambios recientes sin descartar completamente la señal histórica.

    Tracking dual:
      - `n_raw`: conteo INT sin pesar — usado por la fase de exploración
        (round-robin hasta `_AB_MIN_SAMPLES_PER_PAIR` muestras por par). Si
        usáramos pesos aquí, un par con muchas observaciones ANTIGUAS podría
        parecer bajo-explorado (peso decayed bajo) cuando ya tiene cobertura
        histórica suficiente — distorsionando la decisión de exploración.
      - `wins_a/wins_b`: FLOATS pesados por decay — usados para Thompson
        Sampling (Beta posterior) en la fase de explotación.
    """
    stats = {
        p["label"]: {"wins_a": 0.0, "wins_b": 0.0, "n_raw": 0}
        for p in _AB_TEMP_PAIRS
    }
    for row in rows:
        lbl = row.get("pair_label")
        if lbl not in stats:
            continue
        # P1-Q5: parsear age_days defensivo (driver puede devolver Decimal,
        # float, str). max(0, ...) protege contra clock skew negativo.
        try:
            age_days = max(0.0, float(row.get("age_days") or 0))
        except (TypeError, ValueError):
            age_days = 0.0
        weight = _AB_TEMP_DECAY_FACTOR ** age_days
        key = "wins_a" if row.get("winner") == "candidate_a" else "wins_b"
        stats[lbl][key] += weight
        stats[lbl]["n_raw"] += 1

    # Exploración: usa conteo RAW (sin decay) para garantizar cobertura mínima.
    totals_raw = {lbl: s["n_raw"] for lbl, s in stats.items()}
    min_total_raw = min(totals_raw.values())

    if min_total_raw < _AB_MIN_SAMPLES_PER_PAIR:
        under_explored = [p for p in _AB_TEMP_PAIRS if totals_raw[p["label"]] == min_total_raw]
        selected = random.choice(under_explored)
        print(f"🔬 [AB-TEMP] Exploración: par '{selected['label']}' ({totals_raw[selected['label']]}/{_AB_MIN_SAMPLES_PER_PAIR} muestras)")
        return selected

    # Explotación: inferir preferencia conservador/creativo del usuario usando
    # wins WEIGHTED — refleja preferencia RECIENTE, no acumulado histórico.
    total_a_wins = sum(s["wins_a"] for s in stats.values())
    total_b_wins = sum(s["wins_b"] for s in stats.values())
    total_all = total_a_wins + total_b_wins
    conservative_ratio = total_a_wins / total_all if total_all > 0 else 0.5

    samples = {}
    for p in _AB_TEMP_PAIRS:
        lbl = p["label"]
        s = stats[lbl]
        total_w = s["wins_a"] + s["wins_b"]
        # Wins del candidato alineado con la preferencia global del usuario
        if conservative_ratio >= 0.6:
            aligned_wins = s["wins_a"]   # prefiere conservador → candidate_a es el valioso
        elif conservative_ratio <= 0.4:
            aligned_wins = s["wins_b"]   # prefiere creativo → candidate_b es el valioso
        else:
            aligned_wins = (s["wins_a"] + s["wins_b"]) / 2.0  # equilibrado → ambos aportan
        # P1-Q5: betavariate acepta floats. El `+1` actúa como prior Beta(1,1)
        # uniforme — Bayesian smoothing que evita que Beta(0, 0) sea inválido
        # cuando un par tiene wins_a/b muy decayed (cercanos a 0).
        samples[lbl] = random.betavariate(
            aligned_wins + 1,
            max(0.0, total_w - aligned_wins) + 1,
        )

    best_label = max(samples, key=lambda k: samples[k])
    selected = next(p for p in _AB_TEMP_PAIRS if p["label"] == best_label)
    print(f"🎯 [AB-TEMP] Explotación: par '{selected['label']}' "
          f"(ratio_conservador={conservative_ratio:.2f}, "
          f"wins_weighted_total={total_all:.1f} sobre {sum(totals_raw.values())} muestras raw)")
    return selected


def _select_ab_temp_pair(user_id: str) -> dict:
    """Sync legacy. Mantenido para callers externos (CLI, tests, scripts).
    Producción async usa `_aselect_ab_temp_pair` para no bloquear el loop.
    """
    try:
        rows = execute_sql_query(_AB_TEMP_QUERY, (user_id,), fetch_all=True) or []
    except Exception as e:
        print(f"⚠️ [AB-TEMP] Error leyendo historial: {e}")
        rows = []
    return _compute_ab_temp_pair_from_rows(rows)


async def _aselect_ab_temp_pair(user_id: str) -> dict:
    """P0-NEW-1.a: variante async que usa `aexecute_sql_query` para no congelar
    el event loop. Misma semántica que `_select_ab_temp_pair`. Se llama desde
    `generate_days_parallel_node` (async) cuando el adversarial self-play está
    activo. La consulta se ejecuta sobre `pipeline_metrics` (~90 filas) y
    típicamente toma 30-150ms — sync bloqueaba ese tiempo todos los SSE
    callbacks y demás coroutines del worker.
    """
    try:
        rows = await aexecute_sql_query(_AB_TEMP_QUERY, (user_id,), fetch_all=True) or []
    except Exception as e:
        print(f"⚠️ [AB-TEMP] Error leyendo historial: {e}")
        rows = []
    return _compute_ab_temp_pair_from_rows(rows)


class AdversarialJudgeResult(BaseModel):
    winner: Literal["candidate_a", "candidate_b"] = Field(description="El candidato seleccionado como ganador.")
    rationale: str = Field(description="Explicación breve de por qué este candidato maximiza la adherencia.")

def _is_candidate_valid(cand) -> bool:
    """P1-2: Un candidato es estructuralmente válido si tiene al menos un día
    con al menos un meal real.

    Antes, el judge prompt incluía candidatos corruptos como bloques vacíos —
    la asimetría hacía que el juez sesgara hacia el candidato detallado sin
    saber que el otro estaba inválido. También evitamos llamar al juez si
    sabemos que la decisión es trivial (un candidato es la única opción real).
    """
    if not isinstance(cand, dict):
        return False
    days = cand.get("days")
    if not isinstance(days, list) or not days:
        return False
    for d in days:
        if not isinstance(d, dict):
            continue
        meals = d.get("meals")
        if isinstance(meals, list) and len(meals) > 0:
            return True
    return False


def _log_adversarial_metric(state, form_data, *, winner, rationale, pair_meta,
                            duration_ms=0, tokens_estimated=0,
                            reason="judged", include_pair=True,
                            extra_metadata: dict | None = None):
    """P1-F: Persiste métrica del adversarial judge incluyendo paths silenciosos.

    Antes solo el path normal (LLM-decided) escribía en `pipeline_metrics`.
    Los paths donde se promovía un candidato sin invocar al LLM (otro inválido,
    juez crasheó, ambos corruptos) eran silenciosos: el A/B Thompson Sampling
    de `_select_ab_temp_pair` perdía esas observaciones — los pares con alta
    temperatura que generaban candidatos corruptos NO se penalizaban porque
    su mala señal nunca llegaba a la tabla.

    Args:
        winner: "candidate_a" | "candidate_b" — quién quedó como plan_result,
            incluso si la promoción fue por default.
        reason: discriminador del path para futuro filtrado/análisis:
          - "judged": LLM-juez decidió.
          - "adversary_a_invalid" | "adversary_b_invalid": el otro candidato
            estaba estructuralmente corrupto.
          - "both_invalid": ambos corruptos (winner arbitrario, sin señal AB
            real → `include_pair=False` para excluir del sampler).
          - "judge_failed": el LLM del juez crasheó/timeout; default a A.
        include_pair: si False, omite `pair_label` para que el sampler
          (`_select_ab_temp_pair`) ignore este registro vía su filtro
          `metadata->>'pair_label' IS NOT NULL`.

    Persistencia vía `_METRICS_EXECUTOR` (P1-E) — no bloquea, no crea threads.
    """
    user_id = form_data.get("user_id", "guest") if form_data else "guest"
    if not user_id or user_id == "guest":
        return  # no persistimos para guests; mantiene comportamiento existente

    metadata = {
        "winner": winner,
        "rationale": rationale,
        "winner_reason": reason,  # P1-F: discriminador para análisis posterior
    }
    if include_pair:
        meta = pair_meta or {}
        metadata["pair_label"] = meta.get("label", "balanced")
        metadata["temp_a"] = meta.get("temp_a", 0.7)
        metadata["temp_b"] = meta.get("temp_b", 0.95)
        metadata["winning_temperature"] = (
            metadata["temp_a"] if winner == "candidate_a" else metadata["temp_b"]
        )
    # P0-NEW-1.g: campos opcionales (ej. loser_snapshot del path "judged").
    # Los reservados se ignoran para no permitir overrides accidentales del
    # discriminador de path.
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in {"winner", "rationale", "winner_reason"}:
                metadata[k] = v

    def _persist():
        try:
            from db_core import execute_sql_write
            execute_sql_write(
                "INSERT INTO pipeline_metrics (user_id, session_id, node, "
                "duration_ms, tokens_estimated, confidence, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    state.get("request_id") if isinstance(state, dict) else None,
                    "adversarial_judge",
                    int(duration_ms),
                    int(tokens_estimated),
                    1.0,
                    json.dumps(metadata),
                ),
            )
        except Exception as e:
            logger.warning(
                f"[ADVERSARIAL JUDGE] P1-F: error persistiendo métrica "
                f"(reason={reason}): {e}"
            )

    # P1-X3: propagar ContextVars (request_id) al worker. El logger de error
    # dentro de `_persist` muestra request_id correcto, y futuras métricas que
    # lean `_pipeline_cb_stats_var` desde el worker mantienen consistencia.
    ctx = contextvars.copy_context()
    _METRICS_EXECUTOR.submit(ctx.run, _persist)


# P1-X6: Cap de tamaño del prompt del adversarial judge.
# ------------------------------------------------------------
# Antes el prompt se armaba con `_compress_candidate` truncando ingredientes
# a 6 y recipes a 120 chars. Para chunks chicos (3-7 días) eso producía
# prompts de 3-5K tokens, dentro de cualquier modelo. Pero para chunks de
# 14 días en modo adversarial (×2 candidatos × 4 meals/día ≈ 11K chars por
# candidato + behavioral_profile + adherence + fatigue ≈ 22-30K chars total),
# se acercaba o excedía el contexto de Gemini Flash sin fallback explícito.
# Resultado: structured output corrupto o `InvalidArgument` del SDK; el except
# del nodo defaulteaba a candidate_a sin distinguir de un fallo verdadero,
# perdiendo señal AB legítima.
#
# Estrategia 2-pass:
#   1. Comprimir con caps default (6 ingredientes, 120 chars de recipe).
#   2. Si el prompt total excede `_ADVERSARIAL_PROMPT_CHAR_BUDGET`, recomprimir
#      con caps agresivos (3 ingredientes, 60 chars). Pierde detalle pero
#      preserva la decisión AB.
#   3. Si AÚN excede tras trim agresivo, saltar el juez y promover candidate_a
#      con `reason="prompt_too_large"`. `include_pair=False` excluye este
#      registro del Thompson Sampling porque el winner es arbitrario (no LLM
#      decided), pero el `winner_reason` queda en metadata para debugging.
#
# 18000 chars ≈ 4500 tokens, cómodo bajo el contexto de Flash (~32K) incluso
# considerando overhead del structured output schema. Suficiente para chunks
# de 14 días con caps agresivos. No subir sin reevaluar el caso PRO modelo
# (Gemini Pro tiene 1M+ pero el costo por token es 2-3× Flash).
_ADVERSARIAL_PROMPT_CHAR_BUDGET = 18000


async def adversarial_judge_node(state: PlanState) -> dict:
    """Evalúa dos candidatos de plan y selecciona el que mejor se adapte al perfil conductual del usuario."""
    candidate_b = state.get("candidate_b")
    if not candidate_b:
        # Adversarial mode was not triggered, fast-path
        return {}

    candidate_a = state.get("candidate_a")
    form_data = state["form_data"]
    # P1-F: leer pair_meta UPFRONT para que los paths silenciosos también lo
    # adjunten a sus métricas. El path normal lo lee redundantemente más abajo
    # (preserva código existente sin refactor invasivo).
    _ab_pair_meta = state.get("_ab_temp_meta") or {}

    # P1-2: Validar candidatos antes de invocar al LLM-juez. Si alguno está
    # corrupto (sin días o días sin meals), no hay decisión real que tomar:
    # promovemos el válido directamente y ahorramos un LLM call innecesario.
    a_valid = _is_candidate_valid(candidate_a)
    b_valid = _is_candidate_valid(candidate_b)
    if not (a_valid and b_valid):
        if a_valid and not b_valid:
            print("⚠️ [ADVERSARIAL JUDGE] Candidato B inválido (sin días/meals). "
                  "Promoviendo Candidato A sin invocar al juez.")
            # P1-F: registrar la observación. Que B haya salido corrupto es
            # señal AB legítima — penaliza al par con la temperatura más alta
            # que produjo un candidato no-renderizable.
            _log_adversarial_metric(
                state, form_data,
                winner="candidate_a",
                rationale="Candidato B descartado por estructura inválida.",
                pair_meta=_ab_pair_meta,
                reason="adversary_b_invalid",
            )
            return {
                "plan_result": candidate_a,
                "adversarial_rationale": "Candidato B descartado por estructura inválida.",
            }
        if b_valid and not a_valid:
            print("⚠️ [ADVERSARIAL JUDGE] Candidato A inválido (sin días/meals). "
                  "Promoviendo Candidato B sin invocar al juez.")
            _log_adversarial_metric(
                state, form_data,
                winner="candidate_b",
                rationale="Candidato A descartado por estructura inválida.",
                pair_meta=_ab_pair_meta,
                reason="adversary_a_invalid",
            )
            return {
                "plan_result": candidate_b,
                "adversarial_rationale": "Candidato A descartado por estructura inválida.",
            }
        # Ambos corruptos: devolvemos A; los guardrails P0-1/P0-2 downstream
        # (review_plan_node + arun_plan_pipeline) repararán o forzarán fallback.
        print("⚠️ [ADVERSARIAL JUDGE] Ambos candidatos inválidos. Devolviendo A; "
              "P0-2 guardrail lo reparará downstream.")
        # P1-F: registrar SIN pair_label para que el sampler lo ignore — el
        # winner aquí es arbitrario y no aporta señal AB real (ambos lados
        # fallaron, no se puede atribuir al par). Mantenemos `winner_reason`
        # para visibilidad/debugging fuera del sampler.
        _log_adversarial_metric(
            state, form_data,
            winner="candidate_a",
            rationale="Ambos candidatos corruptos; reparación delegada a P0-2.",
            pair_meta=_ab_pair_meta,
            reason="both_invalid",
            include_pair=False,
        )
        return {
            "plan_result": candidate_a,
            "adversarial_rationale": "Ambos candidatos corruptos; reparación delegada a P0-2.",
        }

    print(f"\n{'='*60}")
    print(f"⚖️ [ADVERSARIAL JUDGE] Evaluando Candidato A vs Candidato B...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "adversarial_judging", "message": "Seleccionando el mejor plan candidato..."})
    start_time = time.time()

    # P1-Q3: capturar modelo del juez para CB per-modelo
    _judge_model = _route_model(form_data, force_fast=False)  # Necesitamos buen razonamiento
    _judge_cb = _get_circuit_breaker(_judge_model)
    judge_llm = ChatGoogleGenerativeAI(
        model=_judge_model,
        temperature=0.2,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=1
    ).with_structured_output(AdversarialJudgeResult)

    ctx = state.get("_cached_context", {})
    behavioral_profile = ctx.get("unified_behavioral_profile", "")
    adherence_context = ctx.get("adherence_context", "")
    fatigue_context = ctx.get("fatigue_context", "")

    # P1-2: Compresión enriquecida — antes solo se exponían ingredientes, dejando
    # al juez sin visibilidad de complejidad culinaria, tiempo de prep, técnica
    # ni densidad calórica. Eso degradaba decisiones sobre adherencia conductual
    # (un usuario fatigado prefiere recetas más simples; un usuario motivado puede
    # con técnicas complejas).
    # P1-X6: caps parametrizados para permitir trim agresivo si el prompt excede
    # `_ADVERSARIAL_PROMPT_CHAR_BUDGET`. Defaults (6, 120) preservan el comportamiento
    # previo en chunks pequeños/medianos; el caller hace 2-pass con (3, 60) si el
    # prompt total supera el budget.
    def _compress_candidate(cand, ing_cap: int = 6, recipe_cap: int = 120):
        # P1-2: Si el candidato está vacío o sin estructura mínima, retornar
        # placeholder explícito en lugar de string vacío. Antes, un candidato
        # corrupto producía resumen "" y el juez sesgaba hacia el otro sin
        # saber que el primero era inválido.
        if not _is_candidate_valid(cand):
            return "[CANDIDATO CORRUPTO O VACÍO — sin días/meals válidos]"
        summary = []
        for d in cand.get("days", []):
            if not isinstance(d, dict):
                continue
            day_meals = d.get("meals")
            if not isinstance(day_meals, list) or not day_meals:
                # P1-2: Día sin meals — saltar para no producir entradas vacías
                # que descalibren el juicio del LLM.
                continue
            day_str = f"Día {d.get('day')}:"
            for m in day_meals:
                if not isinstance(m, dict):
                    continue
                # Normalizar recipe: puede ser str o list[str]
                recipe_raw = m.get("recipe", "")
                if isinstance(recipe_raw, list):
                    recipe_raw = " | ".join(str(s) for s in recipe_raw)
                recipe_str = str(recipe_raw)
                recipe_snippet = (recipe_str[:recipe_cap] + "…") if len(recipe_str) > recipe_cap else recipe_str

                ingredients = m.get("ingredients", []) or []
                ing_str = ", ".join(str(i) for i in ingredients[:ing_cap])
                if len(ingredients) > ing_cap:
                    ing_str += f", …(+{len(ingredients) - ing_cap} más)"

                prep = m.get("prep_time", "?")
                diff = m.get("difficulty", "?")
                cals = m.get("cals", "?")

                day_str += (
                    f"\n  - {m.get('meal')}: {m.get('name')} "
                    f"[{cals}kcal | prep: {prep} | dificultad: {diff}]"
                    f"\n    Ingredientes: {ing_str}"
                    f"\n    Técnica: {recipe_snippet}"
                )
            summary.append(day_str)
        return "\n".join(summary)

    # P1-X6: armado del prompt con cap de tamaño en 2-pass.
    # Helper local para no duplicar el template entre el pase default y el agresivo.
    def _build_judge_prompt(summary_a: str, summary_b: str) -> str:
        return f"""
Eres un juez clínico experto en nutrición y adherencia conductual.
Tu objetivo es seleccionar el MEJOR de dos planes candidatos para un usuario, maximizando la probabilidad de que NO abandone el plan.

--- PERFIL CONDUCTUAL Y RESTRICCIONES ---
{behavioral_profile}
{adherence_context}
{fatigue_context}

--- CANDIDATO A (Conservador) ---
{summary_a}

--- CANDIDATO B (Creativo) ---
{summary_b}

EVALÚA (en este orden de prioridad):
1. SEGURIDAD: ¿Cuál respeta mejor las reglas estrictas (alergias, disgustos, fatiga)?
2. ADHERENCIA CONDUCTUAL: ¿Cuál se adapta mejor al estado emocional y al historial?
3. CARGA COGNITIVA Y DE TIEMPO: usa `prep`, `dificultad` y la `Técnica` para juzgar
   complejidad real. Si el usuario tiene baja adherencia, premia recetas <15 min y
   dificultad fácil. Si el usuario está motivado/alta adherencia, técnicas más
   elaboradas son aceptables.
4. VARIEDAD CULTURAL: revisa los snippets de Técnica para detectar repetición de
   métodos de cocción (3 días seguidos al horno = monótono).

Selecciona el ganador ("candidate_a" o "candidate_b") y da una breve justificación
mencionando explícitamente cuál criterio (1-4) fue decisivo.
"""

    # Pass 1: caps default (6 ingredientes, 120 chars de recipe).
    summary_a = _compress_candidate(candidate_a)
    summary_b = _compress_candidate(candidate_b)
    prompt = _build_judge_prompt(summary_a, summary_b)

    # Pass 2: si excedimos el budget, recomprimir agresivo (3 ing, 60 chars).
    if len(prompt) > _ADVERSARIAL_PROMPT_CHAR_BUDGET:
        print(f"⚠️ [ADVERSARIAL JUDGE] P1-X6: prompt {len(prompt)}c > budget "
              f"{_ADVERSARIAL_PROMPT_CHAR_BUDGET}c. Aplicando trim agresivo "
              f"(ing_cap=3, recipe_cap=60).")
        summary_a = _compress_candidate(candidate_a, ing_cap=3, recipe_cap=60)
        summary_b = _compress_candidate(candidate_b, ing_cap=3, recipe_cap=60)
        prompt = _build_judge_prompt(summary_a, summary_b)

    # Skip: si tras trim agresivo aún excede, no invocar al juez. Promover A
    # como default arbitrario (sin señal AB → `include_pair=False`).
    if len(prompt) > _ADVERSARIAL_PROMPT_CHAR_BUDGET:
        print(f"🛑 [ADVERSARIAL JUDGE] P1-X6: prompt aún excede budget tras trim "
              f"({len(prompt)}c > {_ADVERSARIAL_PROMPT_CHAR_BUDGET}c). "
              f"Saltando juez y promoviendo Candidato A.")
        _log_adversarial_metric(
            state, form_data,
            winner="candidate_a",
            rationale=(
                f"Skip judge: prompt {len(prompt)}c excede budget "
                f"{_ADVERSARIAL_PROMPT_CHAR_BUDGET}c tras trim agresivo."
            ),
            pair_meta=_ab_pair_meta,
            duration_ms=int((time.time() - start_time) * 1000),
            reason="prompt_too_large",
            include_pair=False,
        )
        return {
            "plan_result": candidate_a,
            "adversarial_rationale": (
                f"Skip judge: prompt excede budget ({len(prompt)}c). "
                "Default a Candidato A."
            ),
        }

    try:
        if not await _judge_cb.acan_proceed():  # P1-Q3
            raise Exception(f"Circuit Breaker OPEN para {_judge_model} - LLM cascade failure prevented")

        # P0-4: Hard timeout con cancelación graceful. Decisión binaria con
        # structured output, debería responder rápido. Si excede 30s, default a
        # candidate_a (manejado en except).
        result: AdversarialJudgeResult = await _safe_ainvoke(
            judge_llm, prompt, timeout=30.0
        )
        await _judge_cb.arecord_success()  # P1-Q3
        
        winner_key = result.winner
        rationale = result.rationale
        print(f"🏆 [ADVERSARIAL JUDGE] Ganador: {winner_key}. Razón: {rationale}")
        
        # P0-NEW-1.g: persistir el resultado vía `_log_adversarial_metric`
        # (despachado a `_METRICS_EXECUTOR`, no bloquea el event loop). Antes
        # este path llamaba `execute_sql_write` síncrono directo, lo que añadía
        # 30-150ms bloqueando el loop en el camino feliz del adversarial. Ahora
        # consistente con paths silenciosos (P1-F) — todos pasan por el mismo
        # helper. `loser_snapshot` se preserva vía `extra_metadata`.
        user_id = form_data.get("user_id", "guest")
        if user_id != "guest":
            loser_key = "candidate_b" if winner_key == "candidate_a" else "candidate_a"
            loser_cand = candidate_b if loser_key == "candidate_b" else candidate_a
            _log_adversarial_metric(
                state, form_data,
                winner=winner_key,
                rationale=rationale,
                pair_meta=state.get("_ab_temp_meta") or {},
                duration_ms=int((time.time() - start_time) * 1000),
                tokens_estimated=len(prompt) // 4,
                reason="judged",
                extra_metadata={"loser_snapshot": _compress_candidate(loser_cand)},
            )

        # Update the plan_result to be the winner's payload
        winner_cand = state.get(winner_key)
        if isinstance(winner_cand, dict):
            winner_cand["_adversarial_winner"] = winner_key
        return {
            "plan_result": winner_cand,
            "adversarial_rationale": rationale
        }
        
    except Exception as e:
        await _judge_cb.arecord_failure()  # P1-Q3
        print(f"⚠️ [ADVERSARIAL JUDGE] Falló la evaluación: {e}. Defaulting to Candidate A.")
        # P1-F: registrar el fallo del juez como observación AB. El default a A
        # es arbitrario (no LLM-decided), así que `include_pair=False` excluye
        # del Thompson Sampling para no contaminar las stats del par usado.
        _log_adversarial_metric(
            state, form_data,
            winner="candidate_a",
            rationale=f"Fallo en evaluación: {e}",
            pair_meta=state.get("_ab_temp_meta") or {},
            duration_ms=int((time.time() - start_time) * 1000),
            reason="judge_failed",
            include_pair=False,
        )
        return {
            "plan_result": candidate_a,
            "adversarial_rationale": f"Fallo en evaluación: {e}"
        }


# ============================================================
# NODO POST-GENERACIÓN: SELF-CRITIQUE (GAP 3)
# ============================================================
class CritiqueEvaluation(BaseModel):
    visual_score: int = Field(description="Atractivo Visual (1-10)")
    diversity_score: int = Field(description="Diversidad Real de sabores (1-10)")
    cultural_score: int = Field(description="Coherencia Cultural Dominicana (1-10)")
    temperature_score: int = Field(description="Balance de Temperaturas (1-10)")
    slot_coherence_score: int = Field(description="Coherencia comida↔horario (1-10): ¿Las meriendas son snacks ligeros y no platos fuertes? ¿La cena evita repetir proteína/carbo del almuerzo?")
    needs_correction: bool = Field(description="True si >=2 scores son < 6, o si algún score es < 4")
    suggestions: str = Field(description="Si needs_correction es True, especifica exactamente qué cambiar.")

class CorrectedDays(BaseModel):
    days: list[SingleDayPlanModel] = Field(description="Lista de los 3 días con las correcciones aplicadas.")

# Número máximo de días que el self-critique corregirá en un solo run.
# P1-1: Antes era constante a 2 — para chunks de 7-15 días dejaba 5+ días con
# problemas de diversidad/cultura sin corregir. Ahora se calcula dinámicamente:
# ver `_compute_self_critique_max_days`. Esta constante es un fallback histórico.
_SELF_CRITIQUE_MAX_DAYS = 2

# Cap absoluto del self-critique. No subir de 4 sin reevaluar contención del
# LLM_SEMAPHORE local (max_concurrent=4) y del budget global de 600s.
_SELF_CRITIQUE_HARD_CEILING = 4


def _compute_self_critique_max_days(state: PlanState) -> int:
    """P1-1: Cap dinámico para correcciones del self-critique.

    Escala con `_days_to_generate` y respeta el budget temporal restante: si
    quedan <180s antes del GLOBAL_TIMEOUT, se reduce el cap para no comprometer
    review_plan/assemble downstream.

    Para chunks pequeños (≤3 días) corregimos TODO el chunk (cubrir 100% de los
    problemas detectados). Para chunks grandes (4+) cap a la mitad para no
    consumir todo el budget en correcciones — el revisor médico es el gatekeeper
    real y necesita ~70s al final.
    """
    form_data = state.get("form_data", {}) or {}
    try:
        days_in_chunk = max(1, int(form_data.get("_days_to_generate", 3) or 3))
    except (TypeError, ValueError):
        days_in_chunk = 3

    # Cap base: para chunks chicos (≤3) corregir todo el chunk; para chunks
    # grandes, mitad. Acotado por el hard ceiling (4) y mínimo 2.
    if days_in_chunk <= 3:
        base_cap = min(_SELF_CRITIQUE_HARD_CEILING, max(2, days_in_chunk))
    else:
        base_cap = min(_SELF_CRITIQUE_HARD_CEILING, max(2, days_in_chunk // 2))

    pipeline_start = state.get("pipeline_start")
    if pipeline_start:
        elapsed = time.time() - pipeline_start
        # P1-NEW-2: GLOBAL_PIPELINE_TIMEOUT_S del knob (default 600s).
        remaining = GLOBAL_PIPELINE_TIMEOUT_S - elapsed
        # Reservar tiempo para review_plan (~70s) + assemble + retry margin.
        if remaining < 120:
            return min(base_cap, 1)
        if remaining < 180:
            return min(base_cap, 2)

    return base_cap

# Staples de desayuno/merienda que tienden a repetirse silenciosamente entre días.
# El ANTI MODE-COLLAPSE solo rota proteína/carbo principal, no estos.
# Map: etiqueta canónica → aliases buscados (case-insensitive, sin acentos).
#
# [P1-FIX-STAPLES] Incidente real: pipeline entregó plan con `yuca en 6 comidas`
# y `piña en 5 comidas` en 3 días. El reviewer médico lo rechazó por
# "Repetición excesiva de ingredientes principales" pero `_count_staple_repetitions`
# (consumido por self-critique antes del reviewer) NO incluía yuca ni piña en
# este map → el critique no pudo flagear el patrón, no hubo corrección
# determinística previa, y el reviewer rechazó tarde (sin presupuesto para
# retry). Añadidas ambas para cerrar el gap critique↔reviewer:
#   - `yuca`: carbohidrato dominicano de alta rotación (tubérculo barato +
#     contexto cultural). Aparece comúnmente en almuerzo/cena y en mangú-style
#     desayunos. Sin esta entrada el critique daba diversity_score alto a
#     planes con yuca en 4-6 comidas.
#   - `piña`: fruta tropical de alta densidad calórica que el LLM a veces
#     usa como "merienda saludable" en planes de ganancia muscular. Sin la
#     entrada, repetir piña 5 veces en 3 días pasaba como variedad aceptable.
_STAPLE_INGREDIENT_ALIASES = {
    "avena": ["avena"],
    "claras de huevo": ["clara de huevo", "claras de huevo", "clara"],
    "pan integral": ["pan integral", "pan de centeno", "pan de molde"],
    "yogurt griego": ["yogurt griego", "yogur griego", "yogurt", "yogur"],
    "queso blanco": ["queso blanco", "queso fresco"],
    "queso ricotta": ["ricotta"],
    "lechosa": ["lechosa", "papaya"],
    "guineo": ["guineo", "banano"],
    "platano maduro": ["platano maduro", "plátano maduro"],
    "aguacate": ["aguacate"],
    "tortilla integral": ["tortilla integral", "tortilla de trigo"],
    "yuca": ["yuca"],
    "pina": ["pina", "piña"],
}


def _count_staple_repetitions(days: list) -> dict:
    """Cuenta en cuántos días distintos aparece cada staple. Devuelve solo staples
    que aparecen en >=2 días (señal de mode-collapse a nivel de staples).
    """
    # P1-10: unicodedata a nivel módulo

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode("ascii")
        return s

    aliases_norm = {label: [_norm(a) for a in als] for label, als in _STAPLE_INGREDIENT_ALIASES.items()}
    day_counts: dict = {}
    for day in days:
        text_blob = ""
        for meal in day.get("meals", []):
            text_blob += " " + meal.get("name", "")
            for ing in meal.get("ingredients", []) or []:
                text_blob += " " + str(ing)
        text_norm = _norm(text_blob)
        for label, alias_list in aliases_norm.items():
            if any(a in text_norm for a in alias_list):
                day_counts[label] = day_counts.get(label, 0) + 1
    return {k: v for k, v in day_counts.items() if v >= 2}


# Proteínas y carbohidratos principales para detectar overlap almuerzo/cena
# del mismo día. Match por substring sobre nombre+ingredientes normalizados.
_MAIN_PROTEIN_ALIASES = {
    "pollo": ["pollo", "pechuga"],
    "cerdo": ["cerdo", "lomo de cerdo", "chuleta"],
    "res": ["res", "carne molida", "bistec"],
    "pescado": ["pescado", "tilapia", "salmon", "merluza", "mero", "dorado"],
    "atun": ["atun"],
    "huevo": ["huevo", "claras"],
    "gandules": ["gandules"],
    "habichuelas": ["habichuela", "frijoles", "frijol"],
    "lentejas": ["lentejas"],
    "yogurt": ["yogurt", "yogur"],
}

_MAIN_CARB_ALIASES = {
    "arroz": ["arroz"],
    "platano verde": ["platano verde", "guineo verde"],
    "platano maduro": ["platano maduro", "amarillo maduro"],
    "batata": ["batata"],
    "yuca": ["yuca"],
    "name": ["name", "ñame"],
    "papa": ["papa", "papas"],
    "casabe": ["casabe"],
    "pan": ["pan integral", "pan de molde", "pan de centeno"],
    "avena": ["avena"],
    "pasta": ["pasta", "espagueti", "fideos", "lasagna"],
}

# Técnicas/palabras clave que delatan un plato fuerte cuando aparecen
# en una merienda. Si el nombre o la receta contiene alguna, la merienda
# está disfrazada de mini-almuerzo.
_HEAVY_TECHNIQUE_KEYWORDS = [
    "salteado", "locrio", "asopao", "guisado", "guisada",
    "sancocho", "estofado", "encebollado", "al horno con",
    "croquetas", "mofongo", "moro", "pastelon",
]


def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFD", str(s).lower()).encode("ascii", "ignore").decode("ascii")


def _detect_main_items(meal: dict, aliases: dict) -> set:
    """Devuelve el conjunto de etiquetas canónicas (proteínas o carbos) presentes
    en una comida, mirando nombre + lista de ingredientes."""
    blob = _norm_text(meal.get("name", ""))
    for ing in meal.get("ingredients", []) or []:
        blob += " " + _norm_text(ing)
    found = set()
    for label, alias_list in aliases.items():
        if any(_norm_text(a) in blob for a in alias_list):
            found.add(label)
    return found


def _detect_slot_incoherence(days: list) -> list:
    """Detecta dos clases de incoherencia por slot:
       1. Almuerzo y cena del mismo día comparten proteína o carbo principal.
       2. Una merienda tiene técnica/nombre de plato fuerte.
    Devuelve lista de strings legibles para inyectar al prompt del crítico.
    """
    issues: list = []
    for day in days:
        day_num = day.get("day", "?")
        meals = day.get("meals", []) or []
        by_slot = {}
        for m in meals:
            slot = _norm_text(m.get("meal", ""))
            by_slot[slot] = m

        # 1. Overlap almuerzo/cena
        lunch = by_slot.get("almuerzo") or by_slot.get("lunch")
        dinner = by_slot.get("cena") or by_slot.get("dinner")
        if lunch and dinner:
            shared_protein = _detect_main_items(lunch, _MAIN_PROTEIN_ALIASES) & \
                             _detect_main_items(dinner, _MAIN_PROTEIN_ALIASES)
            shared_carb = _detect_main_items(lunch, _MAIN_CARB_ALIASES) & \
                          _detect_main_items(dinner, _MAIN_CARB_ALIASES)
            if shared_protein:
                issues.append(
                    f"Día {day_num}: almuerzo y cena comparten proteína principal "
                    f"({', '.join(sorted(shared_protein))}). Cambia la proteína de la cena."
                )
            if shared_carb:
                issues.append(
                    f"Día {day_num}: almuerzo y cena comparten carbohidrato principal "
                    f"({', '.join(sorted(shared_carb))}). Cambia el carbo de la cena."
                )

        # 2. Merienda con técnica de plato fuerte
        snack = by_slot.get("merienda") or by_slot.get("snack")
        if snack:
            blob = _norm_text(snack.get("name", ""))
            recipe = snack.get("recipe", "")
            if isinstance(recipe, list):
                recipe = " ".join(str(r) for r in recipe)
            blob += " " + _norm_text(recipe)
            heavy_hits = [kw for kw in _HEAVY_TECHNIQUE_KEYWORDS if kw in blob]
            if heavy_hits:
                issues.append(
                    f"Día {day_num}: la merienda parece un plato fuerte "
                    f"(detectado: {', '.join(heavy_hits)}). Conviértela en snack ligero "
                    f"(yogurt+fruta, batido, casabe+queso, sándwich pequeño, fruta+mani)."
                )
    return issues


async def self_critique_node(state: PlanState) -> dict:
    """Evalúa los días generados y aplica correcciones in-place si hay deficiencias."""
    partial = state.get("plan_result", {})
    days = partial.get("days", [])
    if not days:
        return {}

    # Skip en retries: en intento 2 no tenemos presupuesto para 100-170s extras de
    # evaluación+corrección. El revisor médico es el gatekeeper que importa; el
    # self-critique es quality-of-life y solo debe correr en el primer intento.
    if state.get("attempt", 1) > 1:
        print(f"⏭️ [SELF-CRITIQUE] Saltado en intento {state.get('attempt')} (budget-aware).")
        return {}

    print(f"\n{'='*60}")
    print(f"🧐 [SELF-CRITIQUE] Evaluando calidad post-generación...")
    print(f"{'='*60}")
    
    _emit_progress(state, "phase", {"phase": "critique", "message": "Evaluando atractivo y coherencia del plan..."})
    start_time = time.time()

    # P1-Q3: capturar modelo del evaluator para CB per-modelo
    _evaluator_model = _route_model(state.get("form_data", {}), force_fast=True)
    _evaluator_cb = _get_circuit_breaker(_evaluator_model)
    evaluator_llm = ChatGoogleGenerativeAI(
        model=_evaluator_model,
        temperature=0.1,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=1
    ).with_structured_output(CritiqueEvaluation)

    days_json = json.dumps(days, ensure_ascii=False)
    
    def _compress_for_evaluation(days_list):
        summary = []
        for day in days_list:
            for meal in day.get("meals", []):
                recipe = meal.get("recipe", "")
                if isinstance(recipe, list) and len(recipe) > 0:
                    recipe_str = str(recipe[0])
                else:
                    recipe_str = str(recipe)
                summary.append({
                    "day": day.get("day"),
                    "meal": meal.get("meal"),
                    "name": meal.get("name"),
                    "cals": meal.get("cals"),
                    "ingredients_count": len(meal.get("ingredients", [])),
                    "technique_hint": recipe_str[:80] + "..." if recipe_str else ""
                })
        return json.dumps(summary, ensure_ascii=False)

    days_summary_json = _compress_for_evaluation(days)

    # Conteo determinístico de staples repetidos (pre-LLM hard signal).
    # Cubre el blind spot del compresor: el LLM no ve los ingredientes, así que
    # le damos esta tabla calculada por código para que penalice con certeza.
    staple_repetitions = _count_staple_repetitions(days)
    suggested_day_hint = ""
    if staple_repetitions:
        items_str = ", ".join([f"'{k}' en {v} días" for k, v in staple_repetitions.items()])
        print(f"🔁 [SELF-CRITIQUE] Staples repetidos detectados: {items_str}")
        staples_block = (
            f"\n⚠️ STAPLES REPETIDOS DETECTADOS (conteo determinístico, no opinable): {items_str}\n"
            f"   Por cada staple en >=2 días, BAJA diversity_score a 4 o menos y especifica en suggestions "
            f"qué día cambiar para variar el staple.\n"
        )
        # Mencionar día(s) afectado(s) para que el extractor de día (regex 'Día N')
        # pueda apuntar la corrección. Día 1 es default razonable.
        suggested_day_hint = "Día 1"
    else:
        staples_block = ""

    # Detector determinístico de incoherencia por slot:
    # - almuerzo y cena del mismo día con misma proteína/carbo principal
    # - meriendas que en realidad son platos fuertes
    # Estas señales son hechos calculados por código; el LLM no puede negociarlos.
    slot_issues = _detect_slot_incoherence(days)
    if slot_issues:
        joined = "\n   - " + "\n   - ".join(slot_issues)
        print(f"🍽️ [SELF-CRITIQUE] Incoherencias de slot detectadas:{joined}")
        slot_block = (
            f"\n⚠️ INCOHERENCIAS POR SLOT DETECTADAS (conteo determinístico, no opinable):{joined}\n"
            f"   Por CADA incoherencia listada arriba, BAJA slot_coherence_score a 4 o menos, "
            f"marca needs_correction=True, y en suggestions especifica el día y qué cambiar.\n"
        )
        # Si no hay otra pista de día, usa el primero detectado.
        if not suggested_day_hint:
            m = _re.search(r'[Dd]ía\s*(\d+)', slot_issues[0])
            suggested_day_hint = f"Día {m.group(1)}" if m else "Día 1"
    else:
        slot_block = ""

    # Brecha 4: Inyección de Contexto de Usuario
    form_data = state.get("form_data", {})
    adherence = form_data.get("_adherence_hint", "")
    emotional = form_data.get("_emotional_state", "")

    user_context = ""
    if adherence == "low":
        user_context += "\nNOTA CRÍTICA: Este usuario tiene BAJA adherencia. Un plan extremadamente simple, repetitivo y fácil de cocinar es BUENO para él. NO penalices el plan por falta de variedad excesiva o simplicidad."
    if emotional == "needs_comfort":
        user_context += "\nNOTA CRÍTICA: Este usuario reportó necesitar 'comfort food'. Platos calientes, densos y reconfortantes son POSITIVOS. NO los penalices por falta de frescura o ensaladas."

    prompt = f"""
    Eres un Crítico Culinario Experto. Evalúa el siguiente plan de comidas (días generados):
    {days_summary_json}
    {staples_block}{slot_block}{user_context}

    Evalúa del 1 al 10:
    1. Atractivo visual (¿Se lee apetitoso o son combinaciones raras?)
    2. Diversidad real de sabores. Penaliza con score <=4 si:
       - Se repite la misma proteína o guarnición principal con nombres distintos.
       - Un staple (avena, claras, pan, yogurt, queso, lechosa, guineo, plátano maduro, aguacate, tortilla)
         aparece en 2+ días (ver bloque 'STAPLES REPETIDOS' arriba si está presente).
    3. Coherencia cultural Dominicana (¿El desayuno tiene sentido? ¿La cena es coherente?)
    4. Balance de temperaturas (¿Hay 3 días seguidos de ensaladas frías o todo es sopa?)
    5. Coherencia comida↔horario (slot_coherence_score):
       - MERIENDAS deben ser SNACKS LIGEROS (yogurt+fruta, batido, casabe+queso, sándwich pequeño, fruta+mani).
         Si una merienda es "Salteado de…", "Locrio de…", "Pechuga al grill con puré", o cualquier mini-almuerzo, BAJA este score a ≤4.
       - CENA NO debe repetir la PROTEÍNA PRINCIPAL ni el CARBOHIDRATO PRINCIPAL del almuerzo del mismo día. Si los repite, BAJA este score a ≤4.
       - Si el bloque 'INCOHERENCIAS POR SLOT' arriba lista hallazgos, son hechos: BAJA slot_coherence_score a ≤4 obligatoriamente.

    Si DOS O MÁS scores son < 6, o si ALGÚN score es < 4, marca needs_correction=True y da instrucciones CLARAS Y CORTAS de qué cambiar, mencionando explícitamente el día (ej. "Día 2").
    {f"Pista: empieza por {suggested_day_hint} si necesitas elegir cuál corregir primero." if suggested_day_hint else ""}
    """
    
    try:
        if not await _evaluator_cb.acan_proceed():  # P1-Q3
            raise Exception(f"Circuit Breaker OPEN para {_evaluator_model} - LLM cascade failure prevented")
        try:
            # P0-4: Hard timeout con cancelación graceful. Self-critique es
            # opcional/quality-of-life; 30s es suficiente para una evaluación
            # con structured output.
            critique: CritiqueEvaluation = await _safe_ainvoke(
                evaluator_llm, prompt, timeout=30.0
            )
            await _evaluator_cb.arecord_success()  # P1-Q3
        except Exception as e:
            await _evaluator_cb.arecord_failure()  # P1-Q3
            raise e
        print(f"📊 [SELF-CRITIQUE] Scores -> Visual: {critique.visual_score}, Diversidad: {critique.diversity_score}, Cultural: {critique.cultural_score}, Temp: {critique.temperature_score}, Slot: {critique.slot_coherence_score}")
        
        if critique.needs_correction:
            print(f"⚠️ [SELF-CRITIQUE] Problemas detectados. Sugerencias: {critique.suggestions}")

            # Parsear qué días necesitan corrección desde el texto de sugerencias.
            # P1-1: regex \d+ (no \d) — antes "Día 10" se parseaba como "Día 1".
            mentioned = list(dict.fromkeys(
                int(d) for d in _re.findall(r'[Dd]ía\s*(\d+)', critique.suggestions)
            ))
            if not mentioned:
                mentioned = [1]  # Default: corregir día 1 si no se menciona ninguno
            critique_max_days = _compute_self_critique_max_days(state)
            print(f"🔧 [SELF-CRITIQUE] Corrigiendo días afectados: {mentioned[:critique_max_days]} "
                  f"(cap dinámico={critique_max_days})")

            # P1-Q3: capturar modelo del corrector para CB per-modelo
            _corrector_model = _route_model(form_data, force_fast=True)
            _corrector_cb = _get_circuit_breaker(_corrector_model)
            corrector_llm = ChatGoogleGenerativeAI(
                model=_corrector_model,
                temperature=0.3,
                google_api_key=os.environ.get("GEMINI_API_KEY"),
                max_retries=0,
                timeout=80,
            ).with_structured_output(SingleDayPlanModel)

            ctx = _build_shared_context(state)

            # Extraer asignaciones del skeleton para inyectarlas en cada corrector
            _skeleton = state.get("plan_result", {}).get("_skeleton", {})
            _skeleton_days = _skeleton.get("days", [])

            async def _correct_single_day(day_num: int):
                target_day = next((d for d in days if d.get("day") == day_num), None)
                if not target_day:
                    return day_num, None
                if not await _corrector_cb.acan_proceed():  # P1-Q3
                    print(f"⚠️ [SELF-CRITIQUE] Circuit Breaker OPEN ({_corrector_model}). Saltando corrección Día {day_num}.")
                    return day_num, None
                try:
                    print(f"🔧 [SELF-CRITIQUE] Corrigiendo Día {day_num}...")

                    # Incluir asignación del skeleton para que el corrector respete proteínas/carbos
                    skeleton_day = next((d for d in _skeleton_days if d.get("day") == day_num), {})
                    skeleton_block = ""
                    if skeleton_day:
                        from prompts.day_generator import build_day_assignment_context
                        skeleton_block = f"\n⚠️ ASIGNACIÓN OBLIGATORIA DEL PLANIFICADOR (no la ignores):\n{build_day_assignment_context(skeleton_day, day_num)}"

                    correction_prompt = f"""Eres un nutricionista chef. Corrige SOLO el Día {day_num} del plan alimenticio.

PROBLEMA DETECTADO: {critique.suggestions}

RESTRICCIONES NUTRICIONALES (respétalas siempre):
{ctx['nutrition_context']}
{skeleton_block}
REGLA BIDIRECCIONAL CRÍTICA:
- Todo ingrediente en `ingredients` DEBE aparecer nombrado en la receta (Mise en place, El Toque de Fuego o Montaje).
- Todo alimento nombrado en la receta DEBE estar en `ingredients`.
- Si un ingrediente no se usa en la receta, elimínalo de `ingredients`.

Día {day_num} actual (JSON):
{json.dumps(target_day, ensure_ascii=False)}

Devuelve el Día {day_num} corregido con EXACTAMENTE la misma estructura JSON y los mismos targets calóricos."""

                    # P0-4: Hard timeout con cancelación graceful.
                    # [P1-FIX-CRITIQUE] Configurable vía `MEALFIT_CRITIQUE_FIX_TIMEOUT_S`
                    # (default 90s). Antes era hardcoded 70s y el último día de la
                    # corrección paralela (típicamente Día 3) sufría peak de carga
                    # del proveedor LLM → TimeoutError → día sin corregir, justo el
                    # que tenía las violaciones más graves. 45s antes de eso fue el
                    # primer fix; 90s es el cap actual. Como las correcciones corren
                    # en `asyncio.gather`, subir el cap individual NO suma latencia
                    # — el wall-clock total es el max de las 3 ramas, no la suma.
                    corrected_result: SingleDayPlanModel = await _safe_ainvoke(
                        corrector_llm, correction_prompt, timeout=CRITIQUE_FIX_TIMEOUT_S
                    )
                    await _corrector_cb.arecord_success()  # P1-Q3
                    if corrected_result:
                        corrected_day = corrected_result.model_dump()
                        corrected_day["day"] = day_num
                        print(f"✅ [SELF-CRITIQUE] Día {day_num} corregido exitosamente.")
                        return day_num, corrected_day
                except asyncio.TimeoutError:
                    print(f"⏱️ [SELF-CRITIQUE] Timeout corrigiendo Día {day_num} ({CRITIQUE_FIX_TIMEOUT_S:.0f}s). Manteniendo original.")
                except Exception as e:
                    await _corrector_cb.arecord_failure()  # P1-Q3
                    print(f"⚠️ [SELF-CRITIQUE] Error corrigiendo Día {day_num}: {e}. Manteniendo original.")
                return day_num, None

            days_to_fix = mentioned[:critique_max_days]
            correction_results = await asyncio.gather(
                *[_correct_single_day(d) for d in days_to_fix]
            )

            corrected_any = False
            for day_num, corrected_day in correction_results:
                if corrected_day is not None:
                    for i, d in enumerate(days):
                        if d.get("day") == day_num:
                            days[i] = corrected_day
                            break
                    corrected_any = True

            if corrected_any:
                partial["days"] = days
        else:
            print("✅ [SELF-CRITIQUE] El plan pasó la evaluación visual y de coherencia.")
            
    except Exception as e:
        print(f"⚠️ [SELF-CRITIQUE] Error crítico durante la evaluación/corrección: {e}")
        raise e  # Bubble up to trigger graph fallback
        
    duration = round(time.time() - start_time, 2)
    print(f"⏱️ [SELF-CRITIQUE] Completado en {duration}s")
    
    _emit_progress(state, "metric", {
        "node": "self_critique",
        "duration_ms": int(duration * 1000),
        "retries": 0,
        "tokens_estimated": len(str(days_json)) // 4,
        "confidence": 0.98,
        "metadata": {"needs_correction": critique.needs_correction if 'critique' in locals() else False}
    })
    
    return {"plan_result": partial}


# ============================================================
# NODO 3: ENSAMBLADOR (combina días + shopping list + macros del calculador)
# ============================================================
class ParsedIngredient(BaseModel):
    original_string: str = Field(description="El string original exactamente como se pasó")
    qty: float = Field(description="Cantidad numérica. Convertir fracciones a decimales. Si dice 'un puñado' = 1.0, 'medio' = 0.5. Si no hay, usar 1.0", default=1.0)
    unit: str = Field(description="Unidad de medida (g, ml, lb, taza, cda, unidad, etc.). Dejar vacío si no aplica.", default="")
    base_name: str = Field(description="Nombre base limpio del ingrediente (ej. 'pollo', 'cebolla'). Sin adjetivos de preparación si es posible.")

class BatchParsedIngredients(BaseModel):
    parsed_ingredients: list[ParsedIngredient] = Field(description="Lista de ingredientes parseados")

async def assemble_plan_node(state: PlanState) -> dict:
    """Ensambla el plan final combinando skeleton + días paralelos + datos del calculador, o re-ensambla un plan en caché."""
    nutrition = state["nutrition"]
    form_data = state["form_data"]

    print(f"\n{'='*60}")
    print(f"🔧 [ENSAMBLADOR] Combinando plan final...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "assembly", "message": "Ensamblando tu plan final..."})
    start_time = time.time()

    # P0-X1 (defensa en profundidad): el flag `semantic_cache_hit` solo aplica
    # en el PRIMER intento. Si llegamos aquí en attempt>=2 con el flag aún en
    # True (caso patológico donde retry_reflection_node no lo reseteó por algún
    # bug futuro), ignoramos el cache y usamos el plan_result regenerado por el
    # LLM. Garantiza que un retry SIEMPRE refleje el trabajo del segundo paso.
    #
    # [P1-ORQ-3] Sanity check defensivo del invariante de cache_hit: el contrato
    # de `semantic_cache_check_node` (líneas ~6354-6645) es retornar
    # `semantic_cache_hit=True` SOLO con un `cached_plan_data` no-None y con
    # días extraíbles. Si llegamos aquí con el flag True pero el candidato es
    # inválido (None, no-dict, o sin "days"/"plan" extraíble), el invariante
    # upstream se rompió por bug futuro, race en LangGraph state merge, o
    # corrupción del estado. ANTES, este path entregaba silenciosamente
    # `{"days": []}` al usuario — un "plan" vacío que rompía downstream. AHORA
    # detectamos la corrupción, emitimos warning crítico, y degradamos al path
    # LLM (`plan_result`). Si plan_result también está vacío (porque LLM se
    # saltó por cache), el resultado degradado no es peor que el bug original;
    # pero el log permite a operadores detectar la corrupción y alertar.
    use_cache_assembly = state.get("semantic_cache_hit") and state.get("attempt", 1) == 1
    days = []
    cached_plan = None
    if use_cache_assembly:
        raw_cached = state.get("cached_plan_data")
        if not isinstance(raw_cached, dict):
            print(
                f"🚨 [P1-ORQ-3] semantic_cache_hit=True pero "
                f"cached_plan_data={type(raw_cached).__name__} (esperado dict). "
                f"Bypass cache → degradando a plan_result fallback."
            )
            use_cache_assembly = False
        else:
            cached_plan = raw_cached
            # Extracción robusta de 'days' para manejar estructuras anidadas o fallbacks
            days = cached_plan.get("days", [])
            if not days and "plan" in cached_plan:
                p_obj = cached_plan["plan"]
                if isinstance(p_obj, dict):
                    if "days" in p_obj:
                        days = p_obj["days"]
                    elif "day_1" in p_obj:
                        days = [
                            {"day": 1, **p_obj.get("day_1", {})},
                            {"day": 2, **p_obj.get("day_2", {})},
                            {"day": 3, **p_obj.get("day_3", {})}
                        ]
            if not days:
                print(
                    f"🚨 [P1-ORQ-3] semantic_cache_hit=True pero el cache no "
                    f"tiene días extraíbles (keys={list(cached_plan.keys())}). "
                    f"Bypass cache → degradando a plan_result fallback."
                )
                use_cache_assembly = False

    if use_cache_assembly:
        skeleton = {}
        result = {
            "main_goal": nutrition.get("goal_label", ""),
            "insights": (cached_plan.get("insights") if isinstance(cached_plan, dict) else None) or [],
            "days": days,
        }
    else:
        partial = state.get("plan_result", {})
        skeleton = partial.get("_skeleton", state.get("plan_skeleton", {}))
        result = {
            "main_goal": skeleton.get("main_goal", nutrition.get("goal_label", "")),
            "insights": skeleton.get("insights", []),
            "days": partial.get("days", []),
        }

    # Sanitizer de suplementos: el schema `SingleDayPlanModel.supplements` permite
    # al LLM rellenar el campo aunque el usuario no haya activado `includeSupplements`.
    # Como el day_generator prompt no menciona suplementos cuando están apagados,
    # el LLM ocasionalmente los inventa por su cuenta (visto: 'Proteína en polvo'
    # apareciendo en planes con includeSupplements=False). Lo limpiamos de raíz aquí
    # en dos sitios: el campo `supplements` Y los `ingredients` de cada meal (donde
    # el LLM también ha colado proteína en polvo, creatina, etc.).
    if not form_data.get("includeSupplements"):
        _supp_keywords = (
            "proteína en polvo", "proteina en polvo", "whey", "caseína", "caseina",
            "creatina", "creatine", "bcaa", "glutamina", "pre-entreno", "preentreno",
            "pre workout", "colágeno hidrolizado", "colageno hidrolizado",
            "multivitamínico", "multivitaminico", "omega-3", "omega 3",
            "vitamina d3", "magnesio en cápsulas", "magnesio en capsulas",
        )
        _stripped_supps = 0
        _stripped_ings = 0
        for _d in result.get("days") or []:
            if _d.get("supplements"):
                _stripped_supps += len(_d["supplements"])
                _d["supplements"] = []
            for _m in _d.get("meals") or []:
                _ings = _m.get("ingredients") or []
                if not _ings:
                    continue
                _filtered = []
                for _ing in _ings:
                    _ing_norm = str(_ing).lower()
                    if any(kw in _ing_norm for kw in _supp_keywords):
                        _stripped_ings += 1
                        continue
                    _filtered.append(_ing)
                _m["ingredients"] = _filtered
        if _stripped_supps or _stripped_ings:
            print(
                f"💊 [SUPPLEMENTS] Eliminados {_stripped_supps} en campo supplements + "
                f"{_stripped_ings} colados en ingredients (includeSupplements=false)."
            )

    # Renumerar días y asignar day_name obligatoriamente (para parchar planes del semantic_cache)
    days_offset = form_data.get("_days_offset", 0)

    # P1-10: datetime/timezone/timedelta a nivel módulo
    start_date_str = form_data.get("_plan_start_date")
    if start_date_str:
        try:
            start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        except Exception:
            start_dt = datetime.now(timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc)
    
    # Ajustar al timezone local del usuario (tzOffset viene en minutos, ej. 240 = UTC-4).
    # [P1-ORQ-10] Coalesce simétrico con `cron_tasks.py:2013-2017` y
    # `db_profiles.py:209-214`. ANTES, este nodo solo leía `tzOffset` — el
    # nombre que el frontend envía vía `Plan.jsx:419`. Pero `health_profile`
    # persiste el campo como `tz_offset_minutes` (ver `_postprocess_pipeline_result`
    # en `routers/plans.py:913`), y los snapshots de cron jobs / proactive_agent
    # / scripts internos que reconstruyen `form_data` desde el perfil del
    # usuario llegaban acá con SOLO `tz_offset_minutes` presente → defaultaba
    # a 0 → `target_date` calculado en UTC en lugar del huso local → `day_name`
    # desfasado hasta 1 día completo para usuarios en UTC±8/12. El plan se
    # entregaba con "Lunes" cuando para el usuario era domingo o martes.
    tz_offset_minutes = (
        form_data.get("tzOffset")
        or form_data.get("tz_offset_minutes")
        or 0
    )
    if tz_offset_minutes:
        try:
            start_dt = start_dt - timedelta(minutes=int(tz_offset_minutes))
        except Exception:
            pass
        
    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    
    for i, day in enumerate(result.get("days", [])):
        # Forzar numeración 1-based confiable — el LLM a veces devuelve day=0 (0-indexed)
        # o lo omite, lo que corrompía target_date y day_name.
        day["day"] = days_offset + i + 1

        target_date = start_dt + timedelta(days=days_offset + i)
        day["day_name"] = dias_es[target_date.weekday()]
    
    injected_names = [d.get("day_name") for d in result.get("days", [])]
    print(f"📅 [DAY NAMES] Inyectados: {injected_names} (start={start_dt.isoformat()}, tzOffset={tz_offset_minutes})")

    # Post-proceso: forzar valores exactos del calculador
    result["calories"] = nutrition.get("total_daily_calories", nutrition["target_calories"])
    active_macros = nutrition.get("total_daily_macros", nutrition["macros"])
    result["macros"] = {
        "protein": active_macros["protein_str"],
        "carbs": active_macros["carbs_str"],
        "fats": active_macros["fats_str"],
    }
    result["main_goal"] = nutrition["goal_label"]

    # =========================================================
    # OPTIMIZACIÓN DETERMINISTA POST-ASSEMBLY (GAP 4)
    # =========================================================
    days = result.get("days", [])
    
    # Normalizar claves para compatibilidad con cachés antiguos o fallbacks
    for d in days:
        for m in d.get("meals", []):
            if "calories" in m and "cals" not in m: m["cals"] = m.pop("calories")
            if "description" in m and "desc" not in m: m["desc"] = m.pop("description")
            if "instructions" in m and "recipe" not in m: m["recipe"] = m.pop("instructions")
            
            # Auto-fill missing required keys to pass Pydantic Validation & Frontend
            if "meal" not in m: m["meal"] = m.get("name", "Comida").split(" ")[0] if " " in m.get("name", "") else m.get("name", "Comida")
            if "time" not in m: m["time"] = "Flexible"
            if "prep_time" not in m: m["prep_time"] = "15 min"
            if "macros" not in m: m["macros"] = ["Plan Matemático"]
            if "ingredients" not in m: m["ingredients"] = ["Proteína magra al gusto", "Carbohidratos complejos", "Vegetales mixtos"]
            if "difficulty" not in m: m["difficulty"] = "Fácil"
            if "desc" not in m: m["desc"] = "Comida saludable y balanceada."
            if "recipe" not in m: m["recipe"] = ["Mise en place: Preparar todo", "El Toque de Fuego: Cocinar", "Montaje: Servir"]
            
    target_cals = result["calories"]

    # P1-Q6: guardrail anti ZeroDivisionError + balancing inseguro.
    # `target_cals` viene de `nutrition.get("total_daily_calories", nutrition["target_calories"])`.
    # Si `get_nutrition_targets` devolvió 0 (BMR negativo en perfiles edge case,
    # bug en el calculador, perfil corrupto), las 3 secciones siguientes
    # producen resultados sin sentido y la sección 1.5 lanza `ZeroDivisionError`
    # en `... / target_cals` (línea ~4372). También la sección 1.6 calcula
    # `min_cal/max_cal = int(0 * pct) = 0`, que dispara la coherencia de
    # comidas a "redistribuir" toda comida >0 hacia un target imposible.
    # Saltamos todo el balancing y dejamos el plan tal cual del LLM — el
    # revisor médico downstream lo flageará si está fuera de rango calórico.
    _balancing_safe = isinstance(target_cals, (int, float)) and target_cals > 0
    if not _balancing_safe:
        logger.warning(
            f"P1-Q6: target_cals={target_cals!r} inválido (<=0 o no numérico) "
            f"en assemble_plan. Saltando macro balancing, redistribución estricta "
            f"y coherencia de comidas. Probable bug en get_nutrition_targets "
            f"para este perfil — revisar BMR/TDEE calculation upstream."
        )

    # 1. Macro Balancing Post-Assembly
    # Ajusta porciones si un día se desvió del target calórico por más de 100 kcal.
    for day in days:
        if not _balancing_safe:  # P1-Q6
            break
        day_meals = day.get("meals", [])
        if not day_meals:
            continue

        day_cals = sum(m.get("cals", 0) for m in day_meals)
        diff = day_cals - target_cals
        
        # Tolerancia de 100 kcal
        if abs(diff) > 100:
            # Seleccionar la comida más grande para absorber el impacto
            largest_meal = max(day_meals, key=lambda m: m.get("cals", 0))
            # Ajuste máximo de 150 kcal para no distorsionar demasiado la receta
            adjustment = -diff if abs(diff) <= 150 else (-150 if diff > 0 else 150)

            # P1-C: clamp a [0, ∞). Caso de borde: meal["cals"] inicial pequeño
            # (ej. 50) con adjustment muy negativo (-150) producía -100. Bajo
            # este clamp el resultado se acota a 0; loguea warning si la
            # corrección alcanzó el floor para alertar de condiciones extremas
            # (target absurdo, meal mal generado, etc.).
            cals_before_adj = largest_meal.get("cals", 0)
            new_cals = cals_before_adj + adjustment
            if new_cals < 0:
                logger.warning(
                    f"P1-C: ajuste de macro balancing produciría cals<0 en Día "
                    f"{day.get('day')} '{largest_meal.get('meal')}': "
                    f"{cals_before_adj} + ({adjustment}) = {new_cals}. Clamp a 0."
                )
                new_cals = 0
            largest_meal["cals"] = new_cals

            # Ajustar macros (60% del ajuste vía carbs @4kcal/g, 40% vía fats @9kcal/g)
            carb_adj_kcal = abs(adjustment) * 0.6
            fat_adj_kcal = abs(adjustment) * 0.4
            carb_delta = int(carb_adj_kcal / 4)  # 1g carb = 4 kcal
            fat_delta = int(fat_adj_kcal / 9)     # 1g fat = 9 kcal
            if adjustment < 0:
                largest_meal["carbs"] = max(0, largest_meal.get("carbs", 0) - carb_delta)
                largest_meal["fats"] = max(0, largest_meal.get("fats", 0) - fat_delta)
            else:
                # P1-C: defensive clamp; en path positivo no debería ser
                # necesario salvo que macros vinieran corruptos de fases
                # previas. max(0, ...) simétrico al path negativo.
                largest_meal["carbs"] = max(0, largest_meal.get("carbs", 0) + carb_delta)
                largest_meal["fats"] = max(0, largest_meal.get("fats", 0) + fat_delta)
                
            # Brecha 5: Aviso de ajuste de porciones
            recipe_text = largest_meal.get("recipe", "")
            if isinstance(recipe_text, list):
                recipe_text = " ".join(recipe_text)
            disclaimer = f"\n⚠️ Nota del Nutricionista AI: Las cantidades de los ingredientes fueron ajustadas matemáticamente para corregir un desvío de {abs(adjustment)} kcal."
            if "Nota del Nutricionista AI" not in recipe_text:
                largest_meal["recipe"] = recipe_text + disclaimer
                
            print(f"⚖️ [MACRO BALANCING] Día {day.get('day')}: Desviación {diff}kcal -> Ajustado {adjustment}kcal en '{largest_meal.get('meal')}'")

    # 1.5 GAP 1: Guardrail Nutricional Estricto (Redistribución Proporcional Forzada)
    # Si después del soft-balancing la desviación sigue siendo >10%, forzar redistribución matemática.
    for day in days:
        if not _balancing_safe:  # P1-Q6: división `... / target_cals` requiere target>0
            break
        day_meals = day.get("meals", [])
        if not day_meals:
            continue
        day_cals = sum(m.get("cals", 0) for m in day_meals)
        if day_cals == 0:
            continue
        deviation_pct = abs(day_cals - target_cals) / target_cals
        if deviation_pct > 0.10:  # >10% desviación residual
            scale_factor = target_cals / max(day_cals, 1)
            for meal in day_meals:
                # P1-C: clamps defensivos. Si una fase previa dejó valores
                # negativos (sección 1 sin clamp era el caso clásico, ya
                # corregido), multiplicar por scale_factor positivo seguiría
                # siendo negativo. El max(0, ...) cierra esa propagación.
                meal["cals"] = max(0, int(meal.get("cals", 0) * scale_factor))
                meal["protein"] = max(0, int(meal.get("protein", 0) * scale_factor))
                meal["carbs"] = max(0, int(meal.get("carbs", 0) * scale_factor))
                meal["fats"] = max(0, int(meal.get("fats", 0) * scale_factor))
                
                # Brecha 5: Aviso de ajuste de porciones drástico
                if scale_factor < 0.85 or scale_factor > 1.15:
                    recipe_text = meal.get("recipe", "")
                    if isinstance(recipe_text, list):
                        recipe_text = " ".join(recipe_text)
                    disclaimer = f"\n⚠️ Nota del Nutricionista AI: Las porciones fueron escaladas un {abs(1 - scale_factor)*100:.0f}% matemáticamente para cumplir tu meta estricta."
                    if "Nota del Nutricionista AI" not in recipe_text:
                        meal["recipe"] = recipe_text + disclaimer
                        
            new_total = sum(m.get("cals", 0) for m in day_meals)
            print(f"🔒 [STRICT NUTRITION] Día {day.get('day')}: Redistribución forzada ({deviation_pct*100:.0f}% desviación, {day_cals}→{new_total} kcal)")

    # 1.6 GAP A: Guardrail de Coherencia Macro por Comida Individual
    # Asegura que cada comida individual tenga macros y calorías lógicas para su momento del día.
    for day in days:
        if not _balancing_safe:  # P1-Q6: min_cal/max_cal serían 0 → toda comida marcada "exceso"
            break
        day_meals = day.get("meals", [])
        if len(day_meals) < 2:
            continue
            
        # Usamos porcentajes dinámicos del target calórico en lugar de valores fijos
        # para soportar dietas extremas (ej. 1200 kcal vs 3500 kcal).
        MEAL_PCT_RANGES = {
            "desayuno": (0.15, 0.35),
            "almuerzo": (0.25, 0.45),
            "merienda": (0.05, 0.25),
            "cena": (0.15, 0.35),
        }
        
        for meal in day_meals:
            meal_type = str(meal.get("meal", "")).lower()
            # Encontrar el rango correspondiente basado en el nombre de la comida
            matched_key = next((k for k in MEAL_PCT_RANGES.keys() if k in meal_type), None)
            min_pct, max_pct = MEAL_PCT_RANGES.get(matched_key, (0.10, 0.40)) if matched_key else (0.10, 0.40)
            
            min_cal = int(target_cals * min_pct)
            max_cal = int(target_cals * max_pct)
            current_cals = meal.get("cals", 0)
            
            if current_cals < min_cal:
                deficit = min_cal - current_cals
                largest_meal = max(day_meals, key=lambda m: m.get("cals", 0))
                if largest_meal != meal:
                    largest_type = str(largest_meal.get("meal", "")).lower()
                    l_matched_key = next((k for k in MEAL_PCT_RANGES.keys() if k in largest_type), None)
                    l_min_pct = MEAL_PCT_RANGES.get(l_matched_key, (0.10, 0.40))[0] if l_matched_key else 0.10
                    
                    available = largest_meal.get("cals", 0) - int(target_cals * l_min_pct)
                    transfer = min(deficit, max(0, available))
                    if transfer > 0:
                        largest_cals_before = largest_meal.get("cals", 0)

                        # P1-C: clamp defensivo. transfer está acotado por
                        # `min(deficit, max(0, available))` así que en teoría
                        # `largest_cals - transfer >= min_floor`. Pero si
                        # `largest_meal["cals"]` venía corrupto (negativo de
                        # fases previas), el resultado podía quedar negativo.
                        largest_meal["cals"] = max(0, largest_meal.get("cals", 0) - transfer)
                        meal["cals"] = max(0, meal.get("cals", 0) + transfer)
                        print(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy bajo ({current_cals}kcal). Transferidos {transfer}kcal desde '{largest_meal.get('meal')}'")

                        scale_up = meal["cals"] / max(current_cals, 1)
                        meal["protein"] = max(0, int(meal.get("protein", 0) * scale_up))
                        meal["carbs"] = max(0, int(meal.get("carbs", 0) * scale_up))
                        meal["fats"] = max(0, int(meal.get("fats", 0) * scale_up))

                        scale_down = largest_meal["cals"] / max(largest_cals_before, 1)
                        largest_meal["protein"] = max(0, int(largest_meal.get("protein", 0) * scale_down))
                        largest_meal["carbs"] = max(0, int(largest_meal.get("carbs", 0) * scale_down))
                        largest_meal["fats"] = max(0, int(largest_meal.get("fats", 0) * scale_down))

            elif current_cals > max_cal:
                excess = current_cals - max_cal
                smallest_meal = min([m for m in day_meals if m != meal], key=lambda m: m.get("cals", float('inf')))
                if smallest_meal:
                    smallest_cals_before = smallest_meal.get("cals", 0)

                    # P1-C: clamp defensivo simétrico al ramal anterior.
                    # En teoría meal["cals"] - excess = max_cal >= 0, pero si
                    # current_cals fue mutado entre el check y aquí (no debería
                    # bajo asyncio single-thread, pero defensivo) o si vino
                    # negativo de antes, el clamp lo previene.
                    meal["cals"] = max(0, meal.get("cals", 0) - excess)
                    smallest_meal["cals"] = max(0, smallest_meal.get("cals", 0) + excess)
                    print(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy alto ({current_cals}kcal). Transferidos {excess}kcal hacia '{smallest_meal.get('meal')}'")

                    scale_down = meal["cals"] / max(current_cals, 1)
                    meal["protein"] = max(0, int(meal.get("protein", 0) * scale_down))
                    meal["carbs"] = max(0, int(meal.get("carbs", 0) * scale_down))
                    meal["fats"] = max(0, int(meal.get("fats", 0) * scale_down))
                    
                    scale_up = smallest_meal["cals"] / max(smallest_cals_before, 1)
                    # P1-C: clamps defensivos simétricos al ramal anterior.
                    smallest_meal["protein"] = max(0, int(smallest_meal.get("protein", 0) * scale_up))
                    smallest_meal["carbs"] = max(0, int(smallest_meal.get("carbs", 0) * scale_up))
                    smallest_meal["fats"] = max(0, int(smallest_meal.get("fats", 0) * scale_up))


    # 2. Ingredient Consolidation Intelligence
    # Consolida compras. Si hay variaciones leves en ingredientes (e.g. 200g pollo vs 230g pollo), se promedian a un estándar.
    ingredient_map = {}
    all_raw_ingredients = set()
    ing_locations = []
    
    for d_idx, day in enumerate(days):
        for m_idx, meal in enumerate(day.get("meals", [])):
            for i_idx, ing in enumerate(meal.get("ingredients", [])):
                ing_clean = _flatten_ingredient(ing).strip()
                if ing_clean:
                    all_raw_ingredients.add(ing_clean)
                    ing_locations.append({
                        "raw": ing_clean,
                        "d_idx": d_idx, "m_idx": m_idx, "i_idx": i_idx
                    })
                    
    parsed_dict = {}
    if all_raw_ingredients:
        print(f"📦 [CONSOLIDATION] Parseando {len(all_raw_ingredients)} ingredientes únicos (Regex Fast-Path)...")
        parse_start = time.time()
        
        import re
        for raw in all_raw_ingredients:
            raw_lower = raw.lower().strip()
            fractions = {'½': '0.5', '⅓': '0.33', '⅔': '0.67', '¼': '0.25', '¾': '0.75'}
            for f_char, f_val in fractions.items():
                raw_lower = raw_lower.replace(f_char, f_val)
            word_to_num = {'un ': '1 ', 'una ': '1 ', 'unos ': '1 ', 'unas ': '1 ', 'medio ': '0.5 ', 'media ': '0.5 ', 'dos ': '2 ', 'tres ': '3 '}
            for w, n in word_to_num.items():
                if raw_lower.startswith(w):
                    raw_lower = raw_lower.replace(w, n, 1)

            qty = 1.0
            qty_match = re.match(r'^([\d\.,/]+(?:\s*-\s*[\d\.,/]+)?)(?:\s+|$|(?=[a-zñA-ZÑ]))', raw_lower)
            if qty_match:
                qty_str = qty_match.group(1).strip()
                raw_lower = raw_lower[len(qty_match.group(0)):].strip()
                if '-' in qty_str:
                    qty_str = qty_str.split('-')[-1].strip()
                qty_str = qty_str.replace(',', '.')
                try:
                    if '/' in qty_str:
                        qty = float(qty_str.split('/')[0]) / float(qty_str.split('/')[1])
                    else:
                        qty = float(qty_str)
                except ValueError:
                    pass

            units_regex = r'^(g|ml|lb|lbs|tazas?|cda?s?|cucharadas?|cditas?|cucharaditas?|oz|onzas?|dientes?(?:\s+de)?|puñado|pizca|rebanadas?|filetes?|porción|porcion|unidades?|piezas?)\b'
            unit_match = re.match(units_regex, raw_lower)
            unit = ""
            if unit_match:
                unit = unit_match.group(1).strip()
                raw_lower = raw_lower[len(unit_match.group(0)):].strip()
                if unit.endswith(" de"):
                    unit = unit[:-3]
                    
            if raw_lower.startswith("de "):
                raw_lower = raw_lower[3:].strip()
                
            name = raw_lower.strip()
            if name:
                parsed_dict[raw] = ParsedIngredient(original_string=raw, qty=qty, unit=unit, base_name=name)    
    for loc in ing_locations:
        raw = loc["raw"]
        parsed = parsed_dict.get(raw)
        if parsed:
            qty = parsed.qty
            unit = parsed.unit.lower().strip()
            name = parsed.base_name.lower().strip()
            
            # Ignorar ingredientes muy pequeños en g/ml (especias, aceites)
            if unit in ["g", "ml"] and qty < 20: 
                continue
                
            key = f"{name}_{unit}"
            if key not in ingredient_map:
                ingredient_map[key] = []
            ingredient_map[key].append({
                "d_idx": loc["d_idx"], "m_idx": loc["m_idx"], "i_idx": loc["i_idx"],
                "qty": qty, "unit": unit, "original_name": parsed.base_name
            })
    
    for key, occurrences in ingredient_map.items():
        if len(occurrences) >= 2:
            avg_qty = sum(o["qty"] for o in occurrences) / len(occurrences)
            unit = occurrences[0]["unit"]
            
            # Redondeos lógicos por tipo de unidad
            if unit in ["g", "ml"]:
                standard_qty = max(25, int(round(avg_qty / 25.0) * 25))
            elif unit in ["lb", "lbs", "taza", "tazas"]:
                standard_qty = max(0.5, round(avg_qty * 2) / 2.0)
            else:
                standard_qty = max(1 if not unit else 0.5, round(avg_qty * 2) / 2.0)
                
            standard_qty_str = str(int(standard_qty)) if standard_qty == int(standard_qty) else f"{standard_qty:.1f}"
            
            for o in occurrences:
                # Solo consolidar si la desviación no es masiva (max 30% de diferencia)
                if abs(o["qty"] - standard_qty) / max(0.1, o["qty"]) <= 0.30:
                    old_ing = days[o["d_idx"]]["meals"][o["m_idx"]]["ingredients"][o["i_idx"]]
                    
                    if not o["unit"]:
                        new_ing = f"{standard_qty_str} {o['original_name']}"
                    else:
                        new_ing = f"{standard_qty_str} {o['unit']} de {o['original_name']}"
                        
                    days[o["d_idx"]]["meals"][o["m_idx"]]["ingredients"][o["i_idx"]] = new_ing
                    print(f"📦 [CONSOLIDATION] Unificado '{old_ing}' -> '{new_ing}'")
    # =========================================================

    # Calcular shopping lists
    # Solo usar user_id real (autenticado); session_id no tiene inventory en DB
    _uid = form_data.get("user_id")
    if not _uid or _uid == "guest": _uid = None

    from shopping_calculator import get_shopping_list_delta
    try:
        household = max(1, int(form_data.get("householdSize", 1) or 1))
        if household > 1:
            print(f"👨‍👩‍👧‍👦 [HOUSEHOLD] Escalando lista de compras ×{household} personas")

        # P0-NEW-1.b: paralelizar las 3 multiplicidades en `_DB_EXECUTOR`. Antes
        # corrían secuenciales bloqueando el event loop ~150-600ms total
        # (cada `get_shopping_list_delta` hace queries a inventory + pricing).
        # Ahora: 3 calls en paralelo → latencia ≈ max(latencias) y el loop libre
        # para SSE callbacks y otras coroutines del worker.
        if _uid:
            aggr_list_7, aggr_list_15, aggr_list_30 = await asyncio.gather(
                _adb(get_shopping_list_delta, _uid, result, is_new_plan=True, structured=True, multiplier=1.0 * household),
                _adb(get_shopping_list_delta, _uid, result, is_new_plan=True, structured=True, multiplier=2.0 * household),
                _adb(get_shopping_list_delta, _uid, result, is_new_plan=True, structured=True, multiplier=4.0 * household),
            )
        else:
            aggr_list_7, aggr_list_15, aggr_list_30 = [], [], []

        grocery_duration = form_data.get("groceryDuration", "weekly")
        if grocery_duration == "biweekly":
            aggr_list = aggr_list_15
        elif grocery_duration == "monthly":
            aggr_list = aggr_list_30
        else:
            aggr_list = aggr_list_7

        result["aggregated_shopping_list"] = aggr_list
        result["aggregated_shopping_list_weekly"] = aggr_list_7
        result["aggregated_shopping_list_biweekly"] = aggr_list_15
        result["aggregated_shopping_list_monthly"] = aggr_list_30
    except Exception as e:
        import traceback
        print(f"⚠️ [SHOPPING MATH] Error agregando lista delta: {e}")
        traceback.print_exc()
        result["aggregated_shopping_list"] = []
        result["aggregated_shopping_list_weekly"] = []
        result["aggregated_shopping_list_biweekly"] = []
        result["aggregated_shopping_list_monthly"] = []

    # Humanizar ingredientes a medidas caseras dominicanas para la UI (display-only)
    # P0-NEW-1.b: la humanización aplica regex y lookups por ingrediente sobre
    # todo el plan (~21 meals × ~5 ingredientes); aunque es CPU-bound, llamarla
    # sync desde async bloqueaba el loop. Despachada al executor dedicado.
    try:
        from humanize_ingredients import humanize_plan_ingredients
        result = await _adb(humanize_plan_ingredients, result)
    except Exception as e:
        print(f"⚠️ [HUMANIZE] Error al humanizar ingredientes: {e}")

    # Guardar técnicas seleccionadas para persistencia en DB
    result["_selected_techniques"] = skeleton.get("_selected_techniques", [])

    print(f"✅ [ENSAMBLADOR] Plan final ensamblado")

    # Brecha 1: Skeleton Fidelity Validation (Correctiva)
    # Se salta para días reciclados en surgical mode: esos días fueron generados con el
    # skeleton ANTERIOR y compararlos contra el skeleton NUEVO produce falsos positivos.
    skeleton_fidelity_errors = []
    skeleton_days = skeleton.get("days", [])
    affected_days_set = set(state.get("_affected_days") or [])

    for i, day in enumerate(result.get("days", [])):
        day_num = day.get("day", i + 1)

        # Si hay affected_days seteados (surgical mode) y este día NO fue regenerado → skip
        if affected_days_set and day_num not in affected_days_set:
            continue

        skeleton_day = next((s for s in skeleton_days if s.get("day") == day_num), {})
        assigned_proteins = [_flatten_ingredient(p).lower() for p in skeleton_day.get("protein_pool", [])]

        day_ingredients_text = " ".join(
            _flatten_ingredient(ing).lower() for meal in day.get("meals", [])
            for ing in meal.get("ingredients", [])
        )

        missing_proteins = [p for p in assigned_proteins if p not in day_ingredients_text]
        if len(missing_proteins) >= 2:
            msg = f"Día {day_num} omitió múltiples proteínas clave asignadas: {missing_proteins}"
            print(f"⚠️ [SKELETON FIDELITY] {msg}")
            skeleton_fidelity_errors.append(msg)

    if skeleton_fidelity_errors:
        result["_skeleton_fidelity_errors"] = skeleton_fidelity_errors

    # Brecha 3 y 4: Coherencia Intra-Receta y Receta <-> Ingredientes (Determinista)
    recipe_coherence_errors = []
    common_proteins = ["pollo", "res", "cerdo", "salmón", "salmon", "pescado", "atún", "atun", "huevo", "huevos", "pavo", "camarón", "camaron"]
    import re
    
    # P1-11: Stopwords para extracción de "core noun" de ingredientes.
    # Centralizado en constants.RECIPE_INGREDIENT_STOPWORDS — antes vivía
    # hardcodeado aquí con ~80 términos in-place y riesgo de divergir de
    # COMPLEX_TECHNIQUE_KEYWORDS al añadir vocabulario nuevo.
    stopwords = RECIPE_INGREDIENT_STOPWORDS


    # Brecha 4: Validación de Proteína en Ingredientes (con sinónimos)
    protein_synonyms = {
        "pescado": ["pescado", "chillo", "dorado", "mero", "salmón", "salmon", "tilapia", "bacalao", "atún", "atun", "sardina"],
        "pollo": ["pollo", "pechuga", "muslo", "alitas"],
        "res": ["res", "carne", "filete", "sirloin", "churrasco", "molida"],
        "cerdo": ["cerdo", "chuleta", "lomo", "masita"],
        "huevo": ["huevo", "huevos", "clara", "claras", "yema"],
        "huevos": ["huevo", "huevos", "clara", "claras", "yema"],
        "pavo": ["pavo", "pechuga de pavo", "jamón de pavo"],
        "camarón": ["camarón", "camarones", "camaron"],
        "camarones": ["camarón", "camarones", "camaron"]
    }
    
    for day in result.get("days", []):
        for meal in day.get("meals", []):
            ingredients = [_flatten_ingredient(i).lower() for i in meal.get("ingredients", [])]
            recipe = meal.get("recipe", "")
            if isinstance(recipe, list):
                recipe = " ".join(recipe)
            recipe = recipe.lower()
            
            # Brecha 4: Validación de Proteína en Ingredientes
            for cp, synonyms in protein_synonyms.items():
                pattern = r'\b' + re.escape(cp) + r'\b'
                if re.search(pattern, recipe):
                    # La receta menciona la proteína (ej. 'pescado'). Buscar si algún sinónimo está en los ingredientes.
                    if not any(any(re.search(r'\b' + re.escape(syn) + r'\b', ing) for syn in synonyms) for ing in ingredients):
                        msg = f"Día {day.get('day')}, {meal.get('name')}: La receta indica '{cp}' pero no hay ningún ingrediente equivalente (ej. {', '.join(synonyms[:3])}) listado."
                        recipe_coherence_errors.append(msg)
            
            # Brecha 3: Validación de Completitud Estructural
            completion_pattern = r'\b(sirve|servir|montaje|monta|emplata|emplatar|empaca|empacar|disfruta|disfrutar|agrega)\b'
            if not re.search(completion_pattern, recipe):
                msg = f"Día {day.get('day')}, {meal.get('name')}: La receta parece incompleta, falta un paso final (ej: 'Servir', 'Montaje' o 'Empacar')."
                recipe_coherence_errors.append(msg)
                
            # Brecha 3: Validación de Sustantivos de Ingredientes en Instrucciones
            for ing in ingredients:
                clean_ing = re.sub(r'[\d\.,\(\)/\-]', ' ', ing)
                words = [w.strip() for w in clean_ing.split() if w.strip() and len(w.strip()) > 2]
                core_nouns = [w for w in words if w not in stopwords]
                
                if core_nouns:
                    core_noun = core_nouns[0]
                    if len(core_noun) >= 4:
                        prefix = core_noun[:min(5, len(core_noun))]
                        permissive_pattern = r'\b' + re.escape(prefix) + r'[a-z]*\b'
                        if not re.search(permissive_pattern, recipe):
                            msg = f"Día {day.get('day')}, {meal.get('name')}: El ingrediente principal '{core_noun}' está listado pero no se menciona en las instrucciones de la receta."
                            recipe_coherence_errors.append(msg)
                    
    if recipe_coherence_errors:
         result["_recipe_coherence_errors"] = list(set(recipe_coherence_errors))

    # Schema Validation Post-Assembly
    # P1-8: Si el schema canónico falla, marcar el plan como inválido. El revisor
    # médico detecta esta marca y eleva a severity=critical, lo que encadena con
    # el guardrail P0-1 al final del pipeline para entregar fallback matemático
    # en vez de un plan que el frontend no puede renderizar.
    try:
        # P1-10: PlanModel ya está importado a nivel módulo
        PlanModel(**result)
    except Exception as e:
        err_msg = str(e)[:500]
        print(f"🚨 [ASSEMBLY VALIDATION] Plan corrupto post-assembly detectado: {err_msg}")
        result["_schema_invalid"] = True
        result["_schema_errors"] = err_msg

    duration = round(time.time() - start_time, 2)
    _emit_progress(state, "metric", {
        "node": "assemble_plan",
        "duration_ms": int(duration * 1000),
        "retries": state.get("attempt", 1) - 1,
        "tokens_estimated": 0,
        "confidence": 1.0
    })

    return {
        "plan_result": result
    }


# ============================================================
# NODO 2: AGENTE REVISOR MÉDICO
# ============================================================
def _auto_patch_ingredient_coherence(plan: dict, errors: list) -> int:
    """Elimina ingredientes listados que no aparecen en las instrucciones (cosmético, seguro).
    Retorna el número de ingredientes eliminados efectivamente.
    """
    patched = 0
    for error in errors:
        m = _re.search(r"D[ií]a (\d+), (.+?): El ingrediente principal '(.+?)' está listado", error)
        if not m:
            continue
        day_num = int(m.group(1))
        meal_name = m.group(2).strip()
        core_noun = m.group(3).strip().lower()
        for day in plan.get("days", []):
            if day.get("day") != day_num:
                continue
            for meal in day.get("meals", []):
                if meal.get("name", "").strip() != meal_name:
                    continue
                ings = meal.get("ingredients", [])
                orig_len = len(ings)
                meal["ingredients"] = [i for i in ings if core_noun not in i.lower()]
                if len(meal["ingredients"]) < orig_len:
                    patched += 1
                break
    return patched


class ReviewResult(BaseModel):
    approved: bool = Field(description="True si el plan es seguro médicamente")
    issues: list[str] = Field(default_factory=list, description="Lista de problemas encontrados")
    severity: Literal["none", "minor", "high", "critical"] = Field(
        default="none",
        description=(
            "Política de severidad: "
            "'none' = aprobado; "
            "'minor' = retry posible (problemas regenerables: variedad, repetición, complejidad); "
            "'high' = no-retry pero no peligroso médicamente (fallo de despensa estricta, conflictos no-recuperables); "
            "'critical' = peligro médico (alergia/condición violada) → descartar plan y forzar fallback."
        ),
    )
    affected_days: list[int] = Field(default_factory=list, description="Lista de números de día (1, 2, o 3) afectados por los problemas. Vacío si aplica a todos o a ninguno.")


# P1-6: Política única de severidad — la mayor gana cuando hay acumulación.
# Antes, múltiples validaciones (LLM + assembly + pantry + anti-rep + complexity)
# se piseaban entre sí; un `severity = "minor"` posterior podía degradar un
# `"critical"` previo del LLM, causando retries indebidos sobre rechazos médicos.
_SEVERITY_RANK = {"none": 0, "minor": 1, "high": 2, "critical": 3}


def _severity_max(current: str, new: str) -> str:
    """Devuelve la severidad mayor entre `current` y `new` según _SEVERITY_RANK."""
    cur_rank = _SEVERITY_RANK.get(current or "none", 0)
    new_rank = _SEVERITY_RANK.get(new or "none", 0)
    return new if new_rank > cur_rank else (current or "none")


async def review_plan_node(state: PlanState) -> dict:
    """Revisa el plan generado para verificar seguridad médica."""
    plan = state["plan_result"]
    form_data = state["form_data"]
    taste_profile = state.get("taste_profile", "")
    attempt = state.get("attempt", 1)
    
    print(f"\n{'='*60}")
    print(f"🩺 [AGENTE REVISOR MÉDICO] Verificando plan (intento #{attempt})...")
    print(f"{'='*60}")
    
    _emit_progress(state, "phase", {"phase": "review", "message": "Verificación médica en curso..."})
    
    # Extraer restricciones del usuario.
    # P1-1: `otherAllergies`/`otherConditions` ya fueron mergeados en
    # `arun_plan_pipeline` vía `_merge_other_text_fields` (línea ~8047),
    # antes de cualquier nodo del grafo. Aquí solo leemos los arrays
    # canónicos — la lógica de split-CSV+dedup vive en una sola fuente.
    allergies = form_data.get("allergies", [])
    medical_conditions = form_data.get("medicalConditions", [])
    diet_type = form_data.get("dietType", "balanced")
    dislikes = form_data.get("dislikes", [])
    
    # Extraer todos los ingredientes del plan para revisión y validaciones
    all_ingredients = []
    all_meals_summary = []
    days = plan.get("days", [])
    if not days and plan.get("meals"):
        days = [{"day": 1, "meals": plan.get("meals")}] # Fallback seguro durante generación

    for day_obj in days:
        day_num = day_obj.get("day", "?")
        for meal in day_obj.get("meals", []):
            meal_name = meal.get("name", "Sin nombre")
            ingredients = meal.get("ingredients", [])
            all_ingredients.extend(ingredients)
            all_meals_summary.append(f"- Día {day_num} | {meal.get('meal', '?')}: {meal_name} → Ingredientes: {', '.join(ingredients)}")

    start_time = time.time()
    
    # Si no hay restricciones, aprobar el chequeo médico automáticamente (bypasseando el LLM)
    if not allergies and not medical_conditions and diet_type == "balanced" and not dislikes and not taste_profile:
        print("✅ [REVISOR] Sin restricciones declaradas → Bypassing LLM Reviewer, procediendo a validaciones deterministas.")
        approved = True
        issues = []
        severity = "none"
        fact_check_report = "N/A"
    else:
        # ============================================================
        # FASE 1: AGENTE DE FACT-CHECKING (INVESTIGACIÓN CLÍNICA)
        # ============================================================
        fact_check_report = "Sin hallazgos adicionales."
        if allergies or medical_conditions:
            from tools_medical import consultar_base_datos_medica
            # P1-10: SystemMessage/HumanMessage/ToolMessage a nivel módulo

            print("🔬 [FACT-CHECKING] Iniciando investigación de alergias/condiciones...")
            # P1-Q3: capturar modelo del fact-checker para CB per-modelo
            _fact_checker_model = _route_model(form_data, force_fast=True)
            _fact_checker_cb = _get_circuit_breaker(_fact_checker_model)
            fact_checker_llm = ChatGoogleGenerativeAI(
                model=_fact_checker_model,
                temperature=0.0,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            ).bind_tools([consultar_base_datos_medica])
            
            fc_sys_prompt = "Eres un investigador clínico. Revisa las alergias y condiciones médicas frente a los ingredientes presentados. Usa tu herramienta para investigar posibles reacciones cruzadas, contraindicaciones o interacciones peligrosas. Cuando termines o si no ves riesgo, responde con un REPORTE CLÍNICO CONCISO con tus hallazgos."
            
            fc_messages = [
                SystemMessage(content=fc_sys_prompt),
                HumanMessage(content=f"Alergias: {allergies}\nCondiciones: {medical_conditions}\nDieta: {diet_type}\nIngredientes a evaluar: {all_ingredients}")
            ]
            
            for step in range(4): # Límite de 4 iteraciones
                try:
                    # P0-4: Hard timeout con cancelación graceful. 4 iteraciones
                    # secuenciales sin tope explícito podían colgarse acumulativamente;
                    # 30s/iter mantiene peor caso del fact-check ~120s + tool calls.
                    fc_response = await _safe_ainvoke(
                        fact_checker_llm, fc_messages, timeout=30.0
                    )
                    fc_messages.append(fc_response)

                    if not fc_response.tool_calls:
                        # P1-7: Normalizar content (Gemini 3.x puede devolver
                        # list[dict] de content blocks o str) y validar que el
                        # reporte tenga sustancia. Antes, un content vacío/truncado
                        # se aceptaba tal cual y el revisor médico recibía un
                        # reporte degenerado, perdiendo el guard clínico.
                        raw_content = fc_response.content
                        if isinstance(raw_content, list):
                            parts = []
                            for block in raw_content:
                                if isinstance(block, dict):
                                    parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    parts.append(block)
                            normalized = "".join(parts).strip()
                        elif isinstance(raw_content, str):
                            normalized = raw_content.strip()
                        else:
                            normalized = str(raw_content).strip() if raw_content is not None else ""

                        if len(normalized) < 30:
                            print(f"⚠️ [FACT-CHECK] Reporte sospechosamente corto "
                                  f"({len(normalized)} chars). Usando fallback precautorio.")
                            fact_check_report = (
                                "Sin hallazgos clínicos verificables. "
                                "Asumir precaución estándar para las restricciones declaradas."
                            )
                        else:
                            fact_check_report = normalized
                        # P1-NEW-8: el LLM convergió sin excepción. Registrar
                        # éxito cierra la asimetría histórica (solo failures se
                        # grababan en el CB del fact-checker) y permite resetear
                        # el contador de fallos cuando el proveedor recupera, en
                        # lugar de esperar al `reset_timeout` natural (~30s) de
                        # la última falla. Aplica también a la rama A1 (reporte
                        # degenerado) — el LLM respondió, el fallback es decisión
                        # nuestra de fail-safe semántico, no falla del proveedor.
                        # Symmetric con `invoke_planner`/`invoke_day`/`invoke_with_retry`.
                        await _fact_checker_cb.arecord_success()
                        break

                    for tool_call in fc_response.tool_calls:
                        if tool_call["name"] == "consultar_base_datos_medica":
                            # P0-3 + P1-7: La tool internamente hace clinical_llm.invoke()
                            # síncrono (5-15s por call). Antes despachábamos al pool
                            # default vía to_thread, que se compartía con TODA la app
                            # (RAG, embeddings, SQL): bajo N pipelines concurrentes, el
                            # pool se saturaba con fact-checks lentos y operaciones más
                            # rápidas se encolaban detrás. Ahora usamos un executor
                            # dedicado max_workers=2 (_FACT_CHECK_EXECUTOR), limitando
                            # globalmente la concurrencia de fact-checks sin penalizar
                            # otras operaciones to_thread.
                            # P0-A: cada tool-call individual tiene cap duro de
                            # _FACT_CHECK_TOOL_TIMEOUT (20s). Si la tool cuelga,
                            # registramos fallo del circuit breaker e inyectamos un
                            # reporte precautorio al agent loop — el revisor médico
                            # debe actuar fail-closed (asumir riesgo no descartado)
                            # en lugar de aceptar silencio como "todo bien".
                            try:
                                tool_res = await _run_fact_check_tool(
                                    consultar_base_datos_medica, tool_call["args"]
                                )
                                print(f"🔍 [FACT-CHECK] Consulta DB: {tool_call['args']} -> {str(tool_res)[:80]}...")
                            except asyncio.TimeoutError:
                                # P1-Q3: el timeout es de la TOOL clínica (que internamente
                                # invoca a `clinical_llm`), no del fact_checker_llm
                                # propiamente. Atribuimos al CB del fact_checker como
                                # mejor approximación (es el LLM que orquesta la tool).
                                await _fact_checker_cb.arecord_failure()
                                tool_res = (
                                    f"TIMEOUT en consulta clínica (>{_FACT_CHECK_TOOL_TIMEOUT:.0f}s) "
                                    f"para argumentos {tool_call['args']}. No se pudo verificar "
                                    "interacciones. Asumir PRECAUCIÓN MÁXIMA: tratar como riesgo "
                                    "no descartado y restringir uso del ingrediente/condición."
                                )
                                print(f"⏱️ [FACT-CHECK] Tool timeout (>{_FACT_CHECK_TOOL_TIMEOUT:.0f}s) "
                                      f"en {tool_call['args']}. Inyectando reporte precautorio.")
                            fc_messages.append(ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=str(tool_res)
                            ))
                except Exception as fc_e:
                    # P1-NEW-8: cualquier excepción del agent loop (timeout del
                    # `_safe_ainvoke`, rate-limit del proveedor, parse error,
                    # cancelación, etc.) cuenta como fallo del LLM orquestador
                    # del fact-checker. Antes solo el timeout de la TOOL clínica
                    # se atribuía al CB (línea ~5146); el LLM principal podía
                    # caer y la asimetría dejaba el CB sin la señal completa.
                    # Symmetric con cómo `invoke_planner`/`invoke_day`/`invoke_with_retry`
                    # registran failure en su except handler.
                    await _fact_checker_cb.arecord_failure()
                    print(f"⚠️ [FACT-CHECK] Error durante la investigación: {fc_e}")
                    fact_check_report = f"Error en la investigación: {str(fc_e)}. Asumir precaución máxima."
                    break
            else:
                # P1-7: El loop terminó sin break — el fact-checker no convergió en
                # 4 iteraciones (siguió pidiendo tool calls). Mantenemos el reporte
                # inicial pero lo logueamos para detectar bucles patológicos.
                print(f"⚠️ [FACT-CHECK] No convergió en 4 iteraciones (loop de tool calls). "
                      f"Usando reporte conservador inicial.")

        # ============================================================
        # FASE 2: REVISIÓN DETERMINISTA FINAL
        # ============================================================
        review_prompt = f"""
{REVIEWER_SYSTEM_PROMPT}

--- RESTRICCIONES DEL PACIENTE ---
Alergias declaradas: {json.dumps(allergies) if allergies else "Ninguna"}
Condiciones médicas: {json.dumps(medical_conditions) if medical_conditions else "Ninguna"}
Tipo de dieta: {diet_type}
Alimentos que no le gustan: {json.dumps(dislikes) if dislikes else "Ninguno"}

--- REPORTE DE INVESTIGACIÓN CLÍNICA (FACT-CHECKING) ---
{fact_check_report}

--- PERFIL DE GUSTOS (SI EXISTE) ---
{taste_profile if taste_profile else "Sin perfil de gustos disponible."}

--- PLAN A REVISAR ---
Calorías totales: {plan.get("calories")} kcal

Comidas e ingredientes:
{chr(10).join(all_meals_summary)}

--- TODOS LOS INGREDIENTES DEL PLAN ---
{json.dumps(all_ingredients)}

Responde ÚNICAMENTE con el JSON de revisión.
"""
        
        # P1-Q3: capturar modelo del reviewer para CB per-modelo
        _reviewer_model = _route_model(state.get("form_data", {}), attempt=1)
        _reviewer_cb = _get_circuit_breaker(_reviewer_model)
        reviewer_llm = ChatGoogleGenerativeAI(
            model=_reviewer_model,
            temperature=0.1,  # Temperatura muy baja para ser preciso
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_retries=0,
            timeout=60
        ).with_structured_output(ReviewResult)

        # Invocar con reintentos automáticos
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
            before_sleep=lambda retry_state: print(f"⚠️  [REVISOR] Reintento #{retry_state.attempt_number}...")
        )
        async def invoke_with_retry():
            if not await _reviewer_cb.acan_proceed():  # P1-Q3
                raise Exception(f"Circuit Breaker OPEN para {_reviewer_model} - LLM cascade failure prevented")
            try:
                # P0-4: Hard timeout con cancelación graceful. El revisor es el
                # LLM más pesado del pipeline (prompt grande + structured output).
                # Constructor tiene timeout=60 pero el SDK no siempre lo respeta;
                # cap explícito a 70s/intento. Worst case con tenacity (3 intentos):
                # ~210s + backoff = aceptable vs el GLOBAL_TIMEOUT de 600s.
                res = await _safe_ainvoke(
                    reviewer_llm, review_prompt, timeout=70.0
                )
                await _reviewer_cb.arecord_success()  # P1-Q3
                return res
            except Exception as e:
                await _reviewer_cb.arecord_failure()  # P1-Q3
                raise e
        
        try:
            result: ReviewResult = await invoke_with_retry()
            approved = result.approved
            issues = result.issues
            severity = result.severity
            llm_affected_days = result.affected_days
        except Exception as e:
            print(f"⚠️  [REVISOR] Error en structured output, RECHAZANDO por defecto (Fail-Closed): {e}")
            approved = False
            issues = ["Error en la estructura del revisor médico. Forzando regeneración por seguridad clínica."]
            severity = "critical"
            llm_affected_days = []

    duration = round(time.time() - start_time, 2)

    # P1-8: Schema-invalid es un rechazo no negociable.
    # Si assemble_plan_node marcó el plan como inválido contra PlanModel, el
    # frontend no podrá renderizarlo. Elevar a critical para que P0-1 dispare
    # el fallback matemático con disclaimer en lugar de propagar basura.
    if plan.get("_schema_invalid"):
        schema_errors = plan.get("_schema_errors", "Plan no cumple el schema canónico")
        print(f"🚨 [REVISOR] Schema inválido detectado por assemble_plan: {schema_errors}")
        approved = False
        issues.append(
            f"SCHEMA INVÁLIDO: el plan no cumple la estructura esperada y no es renderizable. "
            f"Detalles: {schema_errors}"
        )
        severity = _severity_max(severity, "critical")

    # Brechas 1 y 4: Errores deterministas del ensamblador
    skeleton_fidelity_errors = plan.get("_skeleton_fidelity_errors", [])
    coherence_errors = plan.get("_recipe_coherence_errors", [])

    # Separar errores cosméticos (ingrediente listado sin aparecer en instrucciones) de errores estructurales.
    # Los cosméticos se parchean en-place; los estructurales siguen requiriendo regeneración.
    COHERENCE_PATCHABLE_MARKER = "está listado pero no se menciona en las instrucciones"
    patchable_errors = [e for e in coherence_errors if COHERENCE_PATCHABLE_MARKER in e]
    structural_coherence_errors = [e for e in coherence_errors if COHERENCE_PATCHABLE_MARKER not in e]

    if patchable_errors:
        n_patched = _auto_patch_ingredient_coherence(plan, patchable_errors)
        print(f"🩹 [AUTO-PATCH] {n_patched}/{len(patchable_errors)} ingredientes huérfanos eliminados de listas (no aparecían en instrucciones).")

    assembly_errors = skeleton_fidelity_errors + structural_coherence_errors
    if assembly_errors:
        print(f"❌ [REVISOR] Errores deterministas de ensamblaje detectados: {assembly_errors}")
        approved = False
        issues.extend(assembly_errors)
        # P1-6: severidad diferenciada por TIPO de error.
        # Antes ambos tipos se marcaban como "minor", lo que activaba retry incluso
        # cuando el LLM había omitido proteínas prescritas por el skeleton — un
        # retry produciría el mismo error con alta probabilidad (consume budget de
        # tiempo y attempts sin beneficio, terminando en fallback matemático).
        #   - skeleton_fidelity_errors → "high": el LLM ignoró la asignación de
        #     proteína; retry tiende a repetir el patrón. Mejor abortar y entregar
        #     plan marcado para que el usuario decida regenerar manualmente.
        #   - structural_coherence_errors → "minor": ingrediente listado pero
        #     no usado, receta incompleta, etc. Corregible vía retry.
        # _severity_max preserva el máximo previo (un "critical" del LLM no se
        # degrada a "high" por un assembly_error secundario).
        if skeleton_fidelity_errors:
            severity = _severity_max(severity, "high")
        else:
            severity = _severity_max(severity, "minor")
    
    # ============================================================
    # VALIDACIÓN DETERMINISTA DE DESPENSA Y ANTI-REPETICIÓN (Post-LLM)
    # Verifica que el plan cumpla restricciones de inventario y no repita platos.
    # ============================================================
    if approved:
        # 1. Validación Estricta de Despensa (Pantry Guardrail)
        # P0-7: La heurística previa exigía `has_pantry AND has_previous_meals`
        # para activar la validación, lo cual dejaba sin validar cualquier
        # llamada directa (cron, batch, tests, API públicos) que pasara
        # `current_pantry_ingredients` SIN previous_meals — el plan podía
        # incluir ingredientes que el usuario no tiene (riesgo de plan
        # inutilizable con potencial impacto en salud si se ignoran las
        # restricciones declaradas).
        #
        # Nueva semántica (más segura por defecto):
        #   - Si el caller pasa pantry → SE VALIDA estrictamente. Asumimos
        #     que enviar la despensa es señal de intención de respetarla.
        #   - Opt-out explícito: `_pantry_advisory_only=True` permite tratar
        #     la pantry como "sugerencia/contexto" sin enforce (para callers
        #     que quieren mantener el comportamiento blando previo, p.ej.
        #     primera generación donde el usuario aún no compró).
        #   - Flags explícitos `_is_rotation_reroll` y `_strict_pantry_required`
        #     siguen forzando validación aunque no haya pantry presente.
        is_rotation = bool(form_data.get("_is_rotation_reroll", False))
        is_strict_required = bool(form_data.get("_strict_pantry_required", False))
        has_pantry = bool(form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list"))
        pantry_advisory_only = bool(form_data.get("_pantry_advisory_only", False))
        needs_pantry_validation = (
            not pantry_advisory_only
            and (is_rotation or is_strict_required or has_pantry)
        )

        if needs_pantry_validation:
            # P1-10: validate_ingredients_against_pantry a nivel módulo
            current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])

            clean_pantry = []
            if current_pantry and isinstance(current_pantry, list):
                clean_pantry = [item.strip() for item in current_pantry if item and isinstance(item, str) and len(item) > 2]

            if clean_pantry:
                # Auto-marcar la intención para downstream (logging, métricas, decisiones
                # de severity en should_retry). No mutamos form_data original; solo state.
                if not is_rotation and not is_strict_required:
                    print(f"🔄 [PANTRY GUARD] Rotación implícita detectada (pantry + previous_meals). Validando estricto.")

                val_result = validate_ingredients_against_pantry(all_ingredients, clean_pantry, strict_quantities=False)
                if val_result is not True:
                    approved = False
                    issues.append(val_result)  # val_result es el string de error generado por constants.py
                    # P1-6: max — preservar critical si ya estaba marcado
                    severity = _severity_max(severity, "high")
                    print(f"🚨 [PANTRY GUARD] Validación fallida en Revisor Médico.")
                else:
                    print(f"✅ [PANTRY GUARD] Todos los ingredientes cumplen con la despensa.")

    # 2. Validación Anti-Repetición
    if approved:
        try:
            user_id = form_data.get("user_id") or form_data.get("session_id")
            if user_id and user_id != "guest":

                # P0-NEW-1.c: `get_recent_meals_from_plans` es sync (DB query).
                # Antes bloqueaba el event loop ~50-200ms; ahora despachado al
                # `_DB_EXECUTOR` para no congelar SSE callbacks ni otros pipelines
                # del worker durante la fase de revisión médica.
                recent_meal_names = await _adb(get_recent_meals_from_plans, user_id, 3)
                if recent_meal_names:
                    repeated_meals = await asyncio.to_thread(_validar_repeticiones_cpu_bound, recent_meal_names, days)
                    
                    # Umbral: cero tolerancia, pero excluir nombres genéricos de desayuno
                    generic_ignores = ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa']
                    filtered_repeated = [rm for rm in repeated_meals if not any(g in rm for g in generic_ignores)]
                    
                    if len(filtered_repeated) > 0:
                        approved = False
                        issues.append(
                            f"REPETICIÓN DETECTADA: Los siguientes platos principales ya aparecieron en planes recientes y deben ser reemplazados por alternativas completamente diferentes: {', '.join(repeated_meals)}."
                        )
                        # P1-6: max — preservar critical/high si ya estaba marcado
                        severity = _severity_max(severity, "minor")
                        print(f"🔄 [ANTI-REPETICIÓN] {len(repeated_meals)} platos repetidos detectados: {repeated_meals}")
                    else:
                        print(f"✅ [ANTI-REPETICIÓN] Sin repeticiones detectadas contra {len(recent_meal_names)} platos recientes.")
            else:
                # Fallback para guests: validar contra el history_context in-memory
                history_ctx = state.get("history_context", "") if isinstance(state, dict) else ""
                if history_ctx and days:
                    # Extraer nombres de platos del history_context usando patrón común
                    # Los planes en history usan formato: "- NombrePlato" o "name: NombrePlato"
                    guest_recent = []
                    for line in history_ctx.split("\n"):
                        line = line.strip()
                        if line.startswith("- ") and len(line) > 5 and not line[2:].strip().startswith("["):
                            candidate = line[2:].strip()
                            # Filtrar líneas que parecen ingredientes (contienen cantidades)
                            if not _re.match(r'^\d', candidate) and len(candidate.split()) <= 8:
                                guest_recent.append(candidate)
                    if guest_recent:
                        # P1-H: usar `asyncio.to_thread` (pool default optimizado
                        # con ~32 workers reusados) en lugar de crear/destruir un
                        # `ThreadPoolExecutor(max_workers=2)` por cada request de
                        # guest. Antes el `with`-context spawneaba 2 threads, los
                        # destruía al salir del bloque, y volvía a hacerlo en cada
                        # validación — churn de threads bajo tráfico de guests.
                        # `asyncio.to_thread` reusa el pool global del event loop y
                        # libera el await sin bloquear el loop, consistente con el
                        # path autenticado (línea 4015 aprox.) que usa el mismo helper.
                        repeated_meals = await asyncio.to_thread(
                            _validar_repeticiones_cpu_bound,
                            guest_recent,
                            days,
                        )
                        filtered_guest_repeated = [rm for rm in repeated_meals if not any(g in rm for g in ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa'])]
                        if len(filtered_guest_repeated) > 0:
                            approved = False
                            issues.append(
                                f"REPETICIÓN DETECTADA (Guest): {', '.join(filtered_guest_repeated)}. Regenerar con variantes diferentes."
                            )
                            # P1-6: max — preservar critical/high si ya estaba marcado
                            severity = _severity_max(severity, "minor")
                            print(f"🔄 [ANTI-REPETICIÓN GUEST] {len(filtered_guest_repeated)} platos repetidos detectados")
        except Exception as e:
            print(f"⚠️ [ANTI-REPETICIÓN] Error en validación (no bloqueante): {e}")
            
    # 3. Validación de Complejidad (Feedback Loop Enforcement)
    if approved:
        try:
            adherence_hint = form_data.get("_adherence_hint", "")
            if adherence_hint == "low":
                complexity_score = _calculate_complexity_score(plan)
                print(f"📉 [COMPLEXITY GUARD] Adherencia baja. Score de complejidad del plan generado: {complexity_score}/10")
                if complexity_score > 4.5:
                    approved = False
                    issues.append(
                        f"COMPLEJIDAD EXCESIVA: El usuario tiene baja adherencia. El plan actual es demasiado complejo (Score: {complexity_score}/10). Simplifica radicalmente las recetas usando menos ingredientes y menos pasos (máx 3 pasos simples). Evita el horno."
                    )
                    # P1-6: max — preservar critical/high si ya estaba marcado
                    severity = _severity_max(severity, "minor")
                    print(f"🚨 [COMPLEXITY GUARD] Plan rechazado por ser muy complejo para el nivel actual del usuario.")
                else:
                    print(f"✅ [COMPLEXITY GUARD] Plan validado. Score: {complexity_score}/10 (Adecuado para baja adherencia).")
        except Exception as e:
            print(f"⚠️ [COMPLEXITY GUARD] Error en validación (no bloqueante): {e}")

    _emit_progress(state, "metric", {
        "node": "review_plan",
        "duration_ms": int(duration * 1000) if 'duration' in locals() else 0,
        "retries": state.get("attempt", 1) - 1,
        "tokens_estimated": 0,
        "confidence": 1.0 if approved else 0.0,
        "metadata": {"issues": len(issues)}
    })
    
    if approved:
        print(f"✅ [REVISOR MÉDICO] Plan APROBADO en {duration}s ✅")
        return {
            "review_passed": True,
            "review_feedback": "",
            "rejection_reasons": []
        }
    else:
        feedback = "\n".join([f"• {issue}" for issue in issues])
        print(f"❌ [REVISOR MÉDICO] Plan RECHAZADO en {duration}s (Severidad: {severity})")
        print(f"   Problemas encontrados:")
        for issue in issues:
            print(f"   ❌ {issue}")
            
        # Persistir patrones de rechazo en la base de datos para aprendizaje continuo (GAP 1)
        # [P1-ORQ-1] Read-modify-write atómico vía advisory lock. Antes, el patrón
        # get + mutate + update perdía entries de `rejection_patterns` cuando
        # dos pipelines del mismo user_id rechazaban simultáneamente — el cron
        # `_refill_emergency_backup_plan` corriendo en paralelo con una request
        # del usuario era el caso típico. P0-NEW-1.c: el atomic helper sigue
        # siendo sync (psycopg blocking), despachado al `_DB_EXECUTOR` para no
        # bloquear el event loop ni retrasar el retry inmediato del grafo.
        user_id = form_data.get("user_id")
        if user_id and user_id != "guest":
            try:
                from db_profiles import update_user_health_profile_atomic

                def _rejection_mutator(hp):
                    rejection_patterns = list(hp.get("rejection_patterns", []) or [])
                    # Evitar duplicados exactos y mantener historial manejable
                    for issue in issues:
                        if issue not in rejection_patterns:
                            rejection_patterns.append(issue)
                    # Mantener solo los últimos 10 patrones
                    hp["rejection_patterns"] = rejection_patterns[-10:]

                new_hp = await _adb(update_user_health_profile_atomic, user_id, _rejection_mutator)
                if new_hp is not None:
                    print(f"💾 [META-LEARNING] Patrones de rechazo persistidos en DB.")
            except Exception as e:
                print(f"⚠️ [META-LEARNING] Error persistiendo patrones de rechazo: {e}")
                
        # Brecha 2: Determinar días afectados para Surgical Fix
        days_in_chunk = int(form_data.get("_days_to_generate", 3))
        final_affected_days = set(llm_affected_days) if 'llm_affected_days' in locals() else set()
        for issue in issues:
            # Capturar explícitamente menciones a días (soporta chunks > 3 días)
            for day_num_str in _re.findall(r'(?i)D[íi]a[s]?\s*(\d+)', issue):
                day_int = int(day_num_str)
                if 1 <= day_int <= days_in_chunk:
                    final_affected_days.add(day_int)
                
        return {
            "review_passed": False,
            "review_feedback": feedback,
            "rejection_reasons": issues,
            "_rejection_severity": severity,
            "_affected_days": list(final_affected_days)
        }


# ============================================================
# DECISIÓN CONDICIONAL: ¿Repetir o finalizar?
# ============================================================
def should_retry(state: PlanState) -> str:
    """Decide si regenerar el plan o enviarlo al usuario.

    P1-NEW-2: la política se configura vía env vars `MEALFIT_MAX_ATTEMPTS`,
    `MEALFIT_MIN_RETRY_BUDGET_S`, `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` y
    `MEALFIT_RETRY_SAFETY_MARGIN_S`. Las constantes módulo-nivel se usan
    directamente; antes había shadows locales con valores hardcoded.

    P1-NEW-5: budget simétrico. Antes el guard solo verificaba que el retry
    en sí (reflexión + planner + days) cupiera en `MIN_RETRY_BUDGET_S`,
    pero IGNORABA las fases POST-retry (assemble + review_plan + guardrails)
    que pueden tomar hasta 80s adicionales. Si el retry consumía todo el
    `remaining`, `wait_for(GLOBAL_PIPELINE_TIMEOUT_S)` disparaba mid-review
    y cancelaba TODO el segundo intento. Ahora exigimos
    `remaining >= MIN_RETRY_BUDGET_S + RETRY_SAFETY_MARGIN_S` antes de
    aprobar un retry — si no cabe sin riesgo, preservamos el primer intento.

    P1-A6: Cobertura del peor caso de HEDGING en `generate_days_parallel_node`.
    `MIN_RETRY_BUDGET_S` representa el escenario "happy" donde el primary
    termina cerca de `HEDGE_AFTER_BASE_S` (~45s). Bajo carga el wall-clock
    real puede llegar a `HARD_CEILING_S` (~170s), ~125s extras. Si aprobamos
    un retry justo por encima del threshold previo y el provider está lento,
    `wait_for(GLOBAL_PIPELINE_TIMEOUT_S)` se dispara mid-`generate_days` o
    mid-`review_plan` y cancela el segundo intento entero. Ahora añadimos
    `RETRY_HEDGE_BUDGET_DELTA_S` (default = `HARD_CEILING_S - HEDGE_AFTER_BASE_S`)
    al threshold para preservar margen ante hedging. Bajar este knob a 0 vía
    env var restaura el comportamiento previo; subirlo aumenta seguridad pero
    reduce capacidad de retries en pipelines tardíos.
    """
    # Mínimo de tiempo necesario para la GENERACIÓN del retry: reflexión +
    # planificador + 3 días paralelos. NO incluye assemble + review_plan
    # (esos viven en `RETRY_SAFETY_MARGIN_S`).
    MIN_RETRY_BUDGET_SECONDS = MIN_RETRY_BUDGET_S
    GLOBAL_TIMEOUT = GLOBAL_PIPELINE_TIMEOUT_S
    SAFETY_MARGIN = RETRY_SAFETY_MARGIN_S  # P1-NEW-5
    HEDGE_DELTA = RETRY_HEDGE_BUDGET_DELTA_S  # P1-A6

    if state.get("review_passed", False):
        print("✅ [ORQUESTADOR] Revisión aprobada → Enviando al usuario.")
        return "end"

    severity = state.get("_rejection_severity", "minor")

    # Brecha 6 + P0-2: Retry por Severidad
    # 'critical' = peligro médico (alergia/condición). Abortar y delegar a P0-1 guardrail
    # para entregar fallback matemático con disclaimer.
    # 'high' = no-recuperable por retry (ej. violación de despensa estricta: la despensa
    # no cambia entre intentos, regenerar produciría el mismo error o un loop).
    if severity == "critical":
        print("🚨 [ORQUESTADOR] Rechazo CRÍTICO → No tiene sentido reintentar con el mismo contexto. Abortando temprano.")
        return "end"
    if severity == "high":
        print("🛑 [ORQUESTADOR] Rechazo HIGH (no-recuperable por retry) → Abortando y entregando plan marcado.")
        return "end"

    # P1-X4: default `attempt=1` para alinear con el resto del archivo
    # (initial_state lo setea a 1; otros nodos leen `state.get("attempt", 1)`).
    # En el path normal no cambia el comportamiento — `initial_state` siempre
    # incluye attempt=1 — pero cierra una trampa de mantenimiento (tests
    # directos del nodo con state vacío y refactors futuros).
    if state.get("attempt", 1) >= MAX_ATTEMPTS:
        if not state.get("review_passed", False):
            print(f"🚨 [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada → Tolerando y enviando mejor versión disponible.")
            return "end"
        print(f"⚠️  [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado → Enviando mejor versión disponible.")
        return "end"

    # P1-NEW-5: Guard de presupuesto SIMÉTRICO. Cubre el retry completo
    # (generación + assemble + review + guardrails). Antes solo cubría la
    # generación, lo que permitía aprobar retries que disparaban TimeoutError
    # mid-review y se perdían enteros.
    # P1-A6: añadimos `HEDGE_DELTA` al threshold para cubrir el peor caso de
    # hedging en `generate_days_parallel_node` (HARD_CEILING vs HEDGE_AFTER_BASE).
    #
    # [P1-ORQ-4] Fail-safe ante `pipeline_start` ausente o inválido. ANTES, el
    # check `if start:` era truthy: si `pipeline_start` venía None, 0, o no
    # numérico (caller manual, test con state mínimo, refactor futuro que
    # rompió el contrato del initial_state), se SALTABA todo el guard →
    # `should_retry` siempre retornaba "retry" sin chequear budget →
    # reintentos sin límite hasta que `asyncio.wait_for(GLOBAL_PIPELINE_TIMEOUT_S)`
    # cancelaba TODO el pipeline desde fuera (perdiendo el mejor intento).
    # AHORA tratamos cualquier `pipeline_start` no-numérico-positivo como
    # "presupuesto desconocido = agotado" → preservamos mejor versión
    # disponible y emitimos warning para que operadores detecten el estado
    # inconsistente. El initial_state (línea ~9174) SIEMPRE setea un
    # `time.time()` válido; si llegamos aquí con None, hay un bug upstream
    # que vale la pena loguear.
    start = state.get("pipeline_start")
    if not isinstance(start, (int, float)) or start <= 0:
        print(
            f"⚠️  [ORQUESTADOR] pipeline_start={start!r} (tipo {type(start).__name__}) "
            f"inválido — no puedo calcular budget de retry. Tratando como "
            f"agotado para evitar bypass del guard de timeout. Preservando "
            f"mejor versión disponible."
        )
        return "end"

    elapsed = time.time() - start
    remaining = GLOBAL_TIMEOUT - elapsed
    budget_threshold = MIN_RETRY_BUDGET_SECONDS + SAFETY_MARGIN + HEDGE_DELTA
    if remaining < budget_threshold:
        print(
            f"⏰ [ORQUESTADOR] Sin presupuesto para retry "
            f"({remaining:.0f}s restantes < {budget_threshold:.0f}s mínimo: "
            f"{MIN_RETRY_BUDGET_SECONDS}s generación + {SAFETY_MARGIN}s "
            f"margen post-retry + {HEDGE_DELTA:.0f}s cobertura hedging). "
            f"Preservando mejor versión disponible."
        )
        return "end"

    print("🔄 [ORQUESTADOR] Revisión fallida → Regenerando plan con correcciones...")
    return "retry"


# ============================================================
# NODO 0: REFLEXIÓN META-LEARNING
# ============================================================
# P1-10: `pydantic.BaseModel` y `Field` ya importados al inicio del módulo
# (línea ~13); no reimportar.

class ReflectionResult(BaseModel):
    reflection: str = Field(description="Diagnóstico de la causa raíz en una oración.")

async def reflection_node(state: PlanState) -> dict:
    """Analiza POR QUÉ el ciclo anterior tuvo bajo rendimiento (o alto rendimiento)."""
    form_data = state["form_data"]
    user_id = form_data.get("user_id") or form_data.get("session_id", "guest")
    
    # Solo reflexionamos si hay métricas de un ciclo previo
    quality_score = form_data.get("_previous_plan_quality")
    if quality_score is None:
        return {} # Skip, es usuario nuevo o no hay datos recientes
        
    meal_adherence = form_data.get("_meal_adherence", "Sin datos granulares")
    successful = form_data.get("successful_techniques", "Ninguna")
    abandoned = form_data.get("abandoned_techniques", "Ninguna")
    fatigued = form_data.get("fatigued_ingredients", "Ninguno")

    # P1-1: Hash corto del estado conductual para invalidación de caché precisa
    # P1-10: hashlib a nivel módulo
    behavioral_state = f"{meal_adherence}_{successful}_{abandoned}_{fatigued}".encode('utf-8')
    behavior_hash = hashlib.md5(behavioral_state).hexdigest()[:8]

    # Brecha 5: Validación de Caché de Reflexión (TTL manejado por PersistentLLMCache)
    cache_key = f"reflection_{user_id}_{quality_score}_{behavior_hash}"
    cached_val = await _LLM_CACHE.aget(cache_key)
    if cached_val is not None:
        print(f"⚡ [CACHE HIT] Reutilizando reflexión anterior para el score ({quality_score}).")
        return cached_val

    print(f"🤔 [META-LEARNING] Reflexionando sobre el plan anterior (Quality: {quality_score})...")
    start_time = time.time()
    
    try:
        # P1-Q3: capturar modelo del reflector para CB per-modelo
        _reflector_model = _route_model(form_data, force_fast=True)  # Modelo rápido para reflexiones
        _reflector_cb = _get_circuit_breaker(_reflector_model)
        reflector_llm = ChatGoogleGenerativeAI(
            model=_reflector_model,
            temperature=0.2,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_retries=1
        ).with_structured_output(ReflectionResult)
        
        prompt = f"""
        El plan anterior del usuario tuvo un quality score de {quality_score} (0.0 a 1.0).
        Datos de adherencia por comida: {meal_adherence}
        Técnicas exitosas: {successful}
        Técnicas abandonadas (que no funcionaron): {abandoned}
        Ingredientes fatigados (monotonía detectada): {fatigued}
        
        Diagnostica la causa raíz del rendimiento del plan en UNA oración.
        Ejemplo si es bajo: "El usuario abandona las cenas porque son demasiado complejas para preparar después del trabajo."
        Ejemplo si es alto: "El usuario tiene excelente adherencia cuando se usan ingredientes básicos con cocción al horno."
        """
        
        if not await _reflector_cb.acan_proceed():  # P1-Q3
            raise Exception(f"Circuit Breaker OPEN para {_reflector_model} - LLM cascade failure prevented")
        try:
            # P0-4: Hard timeout con cancelación graceful. Diagnóstico de una
            # sola oración con structured output, debería completar en <15s;
            # cap defensivo a 30s.
            result = await _safe_ainvoke(reflector_llm, prompt, timeout=30.0)
            await _reflector_cb.arecord_success()  # P1-Q3
        except Exception as e:
            await _reflector_cb.arecord_failure()  # P1-Q3
            raise e
        reflection_text = result.reflection
        print(f"💡 [META-LEARNING] Diagnóstico: {reflection_text}")
        
        # --- PERSISTENCIA Y CARGA HISTÓRICA (GAP 1) ---
        user_id = form_data.get("user_id") or form_data.get("session_id")
        historical_reflections_text = ""
        
        if user_id and user_id != "guest":
            # [P1-ORQ-1] Read-modify-write atómico vía advisory lock (FOR UPDATE
            # sobre la fila). Antes, get_user_profile + mutate + update_user_health_profile
            # eran 2 roundtrips separados — bajo concurrencia del mismo user_id
            # (2 tabs regenerando, cron + manual), dos pipelines podían appendear
            # cada uno su `reflection_history` y el segundo UPDATE pisaba al primero.
            # P0-NEW-1.d: el atomic helper sigue siendo sync (psycopg blocking),
            # despachado al `_DB_EXECUTOR` para no bloquear el event loop.
            from db_profiles import update_user_health_profile_atomic

            def _reflection_mutator(hp):
                # `nonlocal` para emitir el texto del historial PRE-append al
                # caller — el mutator es la única vista consistente del estado
                # bajo lock.
                nonlocal historical_reflections_text
                reflection_history = list(hp.get("reflection_history") or [])

                # Cargar historial anterior para el prompt actual (las últimas 5)
                if reflection_history:
                    historical_reflections_text = "\n\n--- 📚 HISTORIAL DE APRENDIZAJE (Últimos diagnósticos) ---\n"
                    for r in reflection_history[-5:]:
                        historical_reflections_text += f"- {r.get('date', '')[:10]}: {r.get('diagnosis', '')} (Calidad: {r.get('quality_score', 'N/A')})\n"
                    historical_reflections_text += "INSTRUCCIÓN: Ten en cuenta estos patrones históricos al diseñar el plan actual.\n----------------------------------------------------------------------\n"

                # Guardar la nueva reflexión
                new_reflection = {
                    "date": datetime.now(timezone.utc).isoformat(),
                    "quality_score": quality_score,
                    "diagnosis": reflection_text
                }
                reflection_history.append(new_reflection)

                # Mantener solo las últimas 10 reflexiones para no saturar
                if len(reflection_history) > 10:
                    reflection_history = reflection_history[-10:]

                hp["reflection_history"] = reflection_history

            new_hp = await _adb(update_user_health_profile_atomic, user_id, _reflection_mutator)
            if new_hp is not None:
                print(f"💾 [META-LEARNING] Reflexión guardada en el perfil de {user_id}")
        
        new_context = state.get("history_context", "") + historical_reflections_text + f"\n\n--- 🧠 REFLEXIÓN META-LEARNING (CICLO ACTUAL) ---\nEl agente ha diagnosticado el resultado del ciclo anterior:\n{reflection_text}\nINSTRUCCIÓN: Usa este diagnóstico para ajustar el diseño de este nuevo plan.\n----------------------------------------------------------------------\n"
        
        duration = round(time.time() - start_time, 2)
        _emit_progress(state, "metric", {
            "node": "reflection",
            "duration_ms": int(duration * 1000),
            "retries": 0,
            "tokens_estimated": len(prompt) // 4,
            "confidence": 1.0,
            "metadata": {"rejection_count": len(state.get("rejection_reasons", []))}
        })

        result_dict = {
            "history_context": new_context,
            "reflection_directive": reflection_text
        }
        # Cachear de forma asíncrona (Mejora 1)
        await _LLM_CACHE.aset(cache_key, result_dict)
        return result_dict
    except Exception as e:
        print(f"⚠️ [META-LEARNING] Error en nodo de reflexión: {e}")
        return {}

# ============================================================
# NODO 0: PRE-FLIGHT OPTIMIZATION (GAP D)
# ============================================================
async def preflight_optimization_node(state: PlanState) -> dict:
    """Lee métricas históricas y el score holístico (Meta-Learning) para auto-ajustar parámetros del pipeline."""
    form_data = state["form_data"]
    user_id = form_data.get("user_id")
    if not user_id or user_id == "guest": return {}

    # P1-10: copy a nivel módulo
    new_form_data = copy.deepcopy(form_data)
    auto_adjusted = False
    
    try:
        from db_profiles import get_user_profile
        import json

        # P1-5: Centralizar lectura de health_profile vía db_profiles.get_user_profile
        # (mismo path que reflection_node, review_plan_node y la persistencia del
        # holistic_score). Antes este nodo usaba SQL raw, lo que (a) duplicaba
        # parsing JSON, (b) saltaba la lógica de graceful-degradation que aplica
        # get_user_profile, (c) podía leer estados desfasados respecto a otros
        # nodos si Supabase tenía réplicas de lectura.
        # P0-NEW-1.e: get_user_profile sync bloqueaba el loop ~50-200ms al inicio
        # del pipeline. Despachado al `_DB_EXECUTOR` para no congelar callbacks.
        # Mejora 7: Leer el last_pipeline_score (Holistic Score) del perfil canónico
        profile = await _adb(get_user_profile, user_id)
        if profile and profile.get("health_profile"):
            # P1-5: parse defensivo. Antes, un health_profile guardado como string
            # JSON corrupto (truncado, encoding mixto) causaba JSONDecodeError que
            # burbujeaba al `except Exception` general → todo el nodo se descarta
            # (preflight pierde su capacidad de auto-simplify y meta-learning para
            # ese usuario hasta que el JSON se corrija manualmente). Ahora si
            # falla el parse, defaulteamos a {} con warning y dejamos que el
            # resto del nodo siga funcionando con la rama de pipeline_metrics.
            raw_hp = profile["health_profile"]
            if isinstance(raw_hp, dict):
                hp = raw_hp
            else:
                try:
                    hp = json.loads(raw_hp) if raw_hp else {}
                    if not isinstance(hp, dict):
                        # JSON válido pero no es un objeto (ej. lista, número)
                        logger.warning(
                            f"[PREFLIGHT] health_profile de user_id={user_id} es JSON "
                            f"válido pero tipo {type(hp).__name__} (esperado dict). Defaulting a {{}}."
                        )
                        hp = {}
                except (json.JSONDecodeError, ValueError, TypeError) as parse_err:
                    logger.warning(
                        f"[PREFLIGHT] health_profile de user_id={user_id} corrupto "
                        f"({type(parse_err).__name__}: {parse_err}). Defaulting a {{}} y "
                        "continuando con análisis de pipeline_metrics."
                    )
                    hp = {}
            # P1-12: Filtrar score_history por antigüedad antes de tomar decisiones.
            # Antes este nodo leía `last_pipeline_score` aislado: un score de
            # hace 3 meses (mala adherencia ya superada) seguía activando
            # auto_simplify aunque el usuario hubiera mejorado. Ahora filtramos
            # entradas con timestamp dentro de los últimos PIPELINE_SCORE_FRESHNESS_DAYS
            # y usamos el último score reciente. Si no hay scores frescos
            # (>14d sin uso o cuenta nueva), no auto-ajustamos por score y
            # dejamos que la rama de pipeline_metrics decida.
            PIPELINE_SCORE_FRESHNESS_DAYS = 14
            score_history = hp.get("pipeline_score_history") or []
            recent_scores = []
            if isinstance(score_history, list):
                cutoff = datetime.now(timezone.utc) - timedelta(days=PIPELINE_SCORE_FRESHNESS_DAYS)
                for entry in score_history:
                    if not isinstance(entry, dict):
                        continue
                    ts_raw = entry.get("ts")
                    if not ts_raw:
                        continue
                    try:
                        ts_dt = safe_fromisoformat(ts_raw) if isinstance(ts_raw, str) else None
                    except Exception:
                        ts_dt = None
                    if ts_dt is None:
                        continue
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    if ts_dt >= cutoff:
                        recent_scores.append(entry)

            # Backward-compat: si no hay history (cuentas migradas), recurrir a
            # last_pipeline_score plano (sin garantía de frescura).
            last_score = None
            score_source = None
            if recent_scores:
                last_score = recent_scores[-1].get("score")
                score_source = f"history reciente ({len(recent_scores)} entradas en últimos {PIPELINE_SCORE_FRESHNESS_DAYS}d)"
            elif score_history:
                # Hay history pero todos los scores son obsoletos → no auto-ajustar
                print(f"🧊 [PREFLIGHT] {len(score_history)} entradas en pipeline_score_history "
                      f"pero todas con antigüedad >{PIPELINE_SCORE_FRESHNESS_DAYS}d. Ignorando para auto-ajuste.")
            else:
                last_score = hp.get("last_pipeline_score")  # legacy
                if last_score is not None:
                    score_source = "last_pipeline_score (legacy, sin timestamp)"

            if last_score is not None:
                print(f"🧠 [META-LEARNING] score detectado: {last_score} (fuente: {score_source})")
                if last_score < 0.65:
                    print(f"🔧 [PREFLIGHT] Score histórico bajo ({last_score}). Activando auto_simplify y drift_alert para maximizar estabilidad.")
                    new_form_data["_auto_simplify"] = True
                    new_form_data["_pipeline_drift_alert"] = True
                    auto_adjusted = True
                elif last_score > 0.90:
                    print(f"✨ [PREFLIGHT] Score histórico alto ({last_score}). Permitiendo mayor libertad creativa.")
                    new_form_data["_creative_freedom"] = True
                    auto_adjusted = True

        # Respaldar con análisis de pipeline_metrics recientes si no se ajustó por score
        # P0-NEW-1.e: usar `aexecute_sql_query` async nativo (asyncpg), no
        # despachado al executor — drivers async son más eficientes que envolver
        # el sync driver en un thread pool.
        if not auto_adjusted:
            recent_data = await aexecute_sql_query(
                "SELECT node, duration_ms, confidence, metadata FROM pipeline_metrics WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
                (user_id,), fetch_all=True
            )
            
            if recent_data:
                # P1-5: parse defensivo de metadata. Si una sola fila tiene
                # JSON corrupto, antes crasheaba todo el cálculo de avg_critique
                # y caía al except general. Ahora cada fila se parsea aislada;
                # las corruptas se cuentan como "no needs_correction" (dict vacío).
                def _parse_metadata(m):
                    raw = m.get("metadata")
                    if isinstance(raw, dict):
                        return raw
                    if not isinstance(raw, str) or not raw:
                        return {}
                    try:
                        parsed = json.loads(raw)
                        return parsed if isinstance(parsed, dict) else {}
                    except (json.JSONDecodeError, ValueError, TypeError):
                        return {}

                critique_metrics = [m for m in recent_data if m["node"] == "self_critique"]
                avg_critique_corrections = 0
                if critique_metrics:
                    avg_critique_corrections = sum(
                        1 for m in critique_metrics
                        if _parse_metadata(m).get("needs_correction")
                    ) / len(critique_metrics)
                
                review_metrics = [m for m in recent_data if m["node"] == "review_plan"]
                avg_rejections = 0
                if review_metrics:
                    avg_rejections = sum(1 for m in review_metrics if float(m.get("confidence", 1.0)) == 0.0) / len(review_metrics)
                    
                    if avg_critique_corrections > 0.5 or avg_rejections > 0.6:
                        print(f"📈 [PREFLIGHT] Auto-ajuste por métricas: {int(avg_rejections*100)}% rechazos, {int(avg_critique_corrections*100)}% correcciones.")
                        new_form_data["_pipeline_drift_alert"] = True
                        if avg_rejections > 0.6:
                            new_form_data["_auto_simplify"] = True
                        auto_adjusted = True

    except Exception as e:
        print(f"⚠️ [PREFLIGHT] Error leyendo métricas de meta-learning: {e}")
        
    return {"form_data": new_form_data} if auto_adjusted else {}

# ============================================================
# CONSTRUCTOR DEL GRAFO
# ============================================================
async def retry_reflection_node(state: PlanState) -> dict:
    """Inyecta contexto de rechazo como directiva para el retry (GAP 1).

    P0-X1: resetea `semantic_cache_hit` y `cached_plan_data` al entrar al retry.
    Sin esto, si el primer intento provino de un cache-hit y `review_plan_node`
    lo rechazó (p.ej. pantry guard, anti-repetición, complejidad), el retry
    regeneraba el plan completo (skeleton + days paralelos + adversarial +
    self-critique) pero al volver a `assemble_plan_node` éste leía el flag
    aún en True y reusaba `cached_plan_data` (el plan rechazado), descartando
    el `plan_result` recién generado. El retry quedaba como no-op funcional.
    """
    reasons = state.get("rejection_reasons", [])
    attempt = state.get("attempt", 1) + 1

    update_data = {
        "attempt": attempt,
        # P0-X1: forzar que el segundo intento use SIEMPRE el plan regenerado
        # por el LLM, no el cacheado del primer intento.
        "semantic_cache_hit": False,
        "cached_plan_data": None,
        # [P1-ORQ-2] Reset explícito del directive previo en CADA invocación de
        # retry_reflection_node. Sin esto, si attempt 3 entra al retry sin
        # `rejection_reasons` frescas (ej. retry disparado por timeout interno,
        # fallo del adversarial judge, o error de schema — rutas donde
        # `should_retry` decide reintentar pero el state no tiene issues nuevos
        # del review_node), el `if reasons:` no entraba, `update_data` no
        # incluía la key, y LangGraph hacía merge del state preservando el
        # directive viejo de attempt 2 → planner recibía señal contradictoria
        # del intento anterior, generando planes no deterministas.
        # Con None explícito, garantizamos directive limpio por defecto;
        # solo se sobrescribe abajo si hay `reasons` frescas que inyectar.
        # El consumer en `_build_planner_prompt` (línea ~2764) usa
        # `state.get("reflection_directive")` con truthy check → None es safe.
        "reflection_directive": None,
    }
    if reasons:
        directive = f"El plan anterior fue RECHAZADO por: {'; '.join(reasons)}. MUTA DRÁSTICAMENTE la estrategia."
        print(f"🔄 [RETRY REFLECTION] Intento {attempt}. Directiva inyectada: {directive}")
        update_data["reflection_directive"] = directive
    return update_data

def _is_cached_plan_schema_compatible(cand_data: dict) -> bool:
    """P0-A3: chequea que el plan cacheado fue generado con el schema actual.

    Lee `_cache_schema_version` del plan candidato; si está ausente, asume
    `_LEGACY_CACHE_SCHEMA_VERSION` (planes pre-deploy del fix). Si no matchea
    `CACHE_SCHEMA_VERSION` actual, devuelve `False` y el caller debe descartar.

    No hace strict equality contra una lista de versiones aceptadas: bumpear
    `CACHE_SCHEMA_VERSION` invalida de golpe todo lo anterior. Eso es la
    semántica deseada — un cambio de schema NO debe servir planes viejos
    aunque "casi" matcheen, porque el frontend depende de la nueva forma.
    """
    if not isinstance(cand_data, dict):
        return False
    cached_version = cand_data.get("_cache_schema_version") or _LEGACY_CACHE_SCHEMA_VERSION
    return cached_version == CACHE_SCHEMA_VERSION


# ============================================================
# P1-ORQ-2: Validación de compatibilidad de despensa para semantic cache
# ------------------------------------------------------------
# Threshold de Jaccard distance sobre el SET normalizado de ingredientes de
# pantry. >0.2 = el usuario reemplazó >20% de su despensa desde que se generó
# el plan cacheado → regenerar produce un plan mejor adaptado.
PANTRY_DRIFT_THRESHOLD: float = 0.2


def _normalize_pantry_set(form_data: dict) -> frozenset:
    """P1-ORQ-2: normaliza la pantry de un form a un set canónico para
    comparación entre cache y request actual.

    `current_pantry_ingredients` y `current_shopping_list` se tratan como una
    sola intención (mismo `or` que `review_plan_node:5603`). Cada item se
    normaliza con `normalize_ingredient_for_tracking` para colapsar variantes
    de cantidad/unidad/sinónimos a un término base canónico — "pollo 500g" y
    "pollo 600g" producen ambos "pollo", lo que hace la comparación robusta a
    cambios de cantidad sin descartar planes con drift trivial.

    Returns:
        frozenset de strings canónicos (lower, sin acentos, sin cantidades).
        Vacío si no hay pantry o el formato es inesperado.
    """
    if not isinstance(form_data, dict):
        return frozenset()
    pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list") or []
    if not isinstance(pantry, list):
        return frozenset()
    from constants import normalize_ingredient_for_tracking
    out = set()
    for raw in pantry:
        if not isinstance(raw, str) or not raw.strip():
            continue
        canonical = normalize_ingredient_for_tracking(raw)
        if canonical:
            out.add(canonical)
    return frozenset(out)


def _pantry_cache_discard_reason(actual_form: dict, cached_form: dict) -> Optional[str]:
    """P1-ORQ-2: si el plan cacheado NO debería servirse a la request actual
    por incompatibilidad de despensa, devuelve string descriptivo del motivo.
    Devuelve None si el cache hit es válido respecto a pantry.

    Reglas:
      - Asimetría: pantry presente AHORA + ausente en cache → discard
        (el plan cacheado fue generado sin awareness de pantry; servirlo
        dispara el rechazo de `review_plan_node` con severity="high" garantizado).
      - Drift: ambos presentes + Jaccard distance > `PANTRY_DRIFT_THRESHOLD`
        (>20% de items normalizados distintos) → discard.
      - Ambos vacíos: no aplica filtro (flujo no-pantry, cache hit válido).
      - Solo cache tiene pantry: tampoco aplica filtro — el plan FUE generado
        con pantry awareness pero el caller actual no exige nada al respecto;
        servirlo es seguro (downstream no activará pantry guard).
    """
    curr_set = _normalize_pantry_set(actual_form)
    cache_set = _normalize_pantry_set(cached_form)

    if curr_set and not cache_set:
        return (
            f"intención de despensa: actual tiene {len(curr_set)} ingredientes, "
            f"cache fue generado sin pantry"
        )

    if curr_set and cache_set:
        intersection = curr_set & cache_set
        union = curr_set | cache_set
        jaccard_dist = 1.0 - (len(intersection) / len(union)) if union else 0.0
        if jaccard_dist > PANTRY_DRIFT_THRESHOLD:
            added = len(curr_set - cache_set)
            removed = len(cache_set - curr_set)
            return (
                f"drift de despensa: jaccard_dist={jaccard_dist:.2f} > "
                f"{PANTRY_DRIFT_THRESHOLD} (+{added} items, -{removed} items vs cache)"
            )

    return None


async def semantic_cache_check_node(state: PlanState) -> dict:
    """Busca un plan similar en la base de datos usando similitud de coseno para saltar la generación LLM.

    P0-NEW-1.f: nodo migrado a `async def`. Antes corría sync dentro del grafo
    LangGraph (que es async): `search_similar_plan` (vector search en pgvector,
    50-300ms) + `get_recent_meals_from_plans` (DB query, 50-200ms) +
    `_validar_repeticiones_cpu_bound` (CPU intenso) bloqueaban el event loop
    al inicio de cada pipeline. Ahora cada I/O se despacha al `_DB_EXECUTOR`
    o a un thread (CPU-bound), liberando el loop para callbacks SSE y otros
    pipelines paralelos.
    """
    profile_embedding = state.get("profile_embedding")
    actual_form_data = state.get("form_data", {})

    # 1. No usar caché si el usuario pidió un re-roll forzado hoy (quiere algo distinto explícitamente)
    is_reroll = actual_form_data.get("_is_same_day_reroll") or actual_form_data.get("_is_rotation_reroll")
    if is_reroll:
        return {"semantic_cache_hit": False, "cached_plan_data": None}

    # [P1-ORQ-6] Observabilidad de embedding ausente. ANTES, si `profile_embedding`
    # llegaba None/empty (servicio de embeddings down — Vertex/OpenAI HTTP
    # failure, throttling — o excepción silenciada en el bloque RAG de
    # `arun_plan_pipeline` líneas ~8869-8975 que dejaba `query_emb=None`),
    # el `if profile_embedding:` de abajo era falsy → fall-through silencioso
    # al return final → el cache miss aparecía como "no había candidato similar"
    # sin distinguir de "no se pudo BUSCAR candidato". Operadores veían cache
    # hit rate caer erráticamente y no podían correlacionar con incidentes
    # del servicio de embeddings. AHORA: para usuarios autenticados (no guests),
    # emitimos warning explícito antes de short-circuit. Guests se omiten:
    # nunca tienen `user_id` real ni embedding persistido, ruido sin valor
    # accionable.
    if not profile_embedding:
        _uid = actual_form_data.get("user_id")
        if _uid and _uid != "guest":
            logger.warning(
                f"🟠 [P1-ORQ-6] Cache miss para user_id={_uid}: profile_embedding "
                f"ausente o vacío (tipo={type(profile_embedding).__name__}). "
                f"Esperado para guests; sospechoso para auth users — posible "
                f"degradación del servicio de embeddings (Vertex/OpenAI), "
                f"throttling, o excepción silenciada en el bloque RAG de "
                f"arun_plan_pipeline (líneas ~8869-8975). Si la frecuencia "
                f"sube, revisar logs de `get_embedding` y status de proveedor."
            )
        return {"semantic_cache_hit": False, "cached_plan_data": None}

    if profile_embedding:
        # P1-2: Búsqueda semántica ampliada (límite 10) para poder aplicar filtros de frescura y médicos.
        # P0-NEW-1.f: vector search es sync (pgvector via psycopg2). Despachado al executor.
        # [P1-ORQ-5] Limit subido de 5 → 10 para compensar pollution de planes
        # con `_cache_schema_version` obsoleta tras un bump. Antes, si los 5
        # candidatos más cercanos eran legacy-version, el post-filter en
        # `_is_cached_plan_schema_compatible` (línea ~6449) los descartaba todos
        # → cache miss garantizado durante la transición. Con HNSW ANN, traer
        # 10 vs 5 candidatos cuesta ~microsegundos extra en el RPC; el costo
        # de transferir 10 plan_data JSONB (~10-50KB c/u) es ~0.5-1ms — trade-off
        # ampliamente favorable contra perder un cache hit que ahorraría 30-90s
        # de generación LLM. Bumpear a 20+ si el conteo de stale del startup
        # log (`P1-ORQ-5`) supera el ratio actual/legacy ≥ 2:1.
        similar_plans = await _adb(search_similar_plan, profile_embedding, 0.98, 10)
        
        valid_plan = None
        plan_data = None
        
        if similar_plans:
            # P1-10: datetime/timezone/safe_fromisoformat a nivel módulo
            now_utc = datetime.now(timezone.utc)
            
            for plan_candidate in similar_plans:
                cand_data = plan_candidate.get("plan_data")
                if not cand_data or not isinstance(cand_data, dict):
                    continue

                # P0-A3: descartar planes con schema incompatible ANTES de las
                # validaciones más caras (anti-repetición, fact-check). Un plan
                # con shape viejo aunque matchee médicamente y por similaridad
                # romperá el frontend o `assemble_plan_node`. Evita gastar CPU
                # en candidatos que serán rechazados de todas formas.
                if not _is_cached_plan_schema_compatible(cand_data):
                    _cached_v = cand_data.get("_cache_schema_version") or _LEGACY_CACHE_SCHEMA_VERSION
                    print(f"🗑️ [SEMANTIC CACHE] P0-A3: plan descartado por schema incompatible "
                          f"(cached={_cached_v!r} ≠ current={CACHE_SCHEMA_VERSION!r}). "
                          f"Probable cambio de contrato post-deploy.")
                    continue

                # 🛡️ Evitar que la caché semántica sirva planes de Fallback Matemático
                is_fallback = False
                for day in cand_data.get("days", []):
                    if "Contingencia de Emergencia" in day.get("daily_summary", "") or "Matemáticamente" in day.get("daily_summary", ""):
                        is_fallback = True
                        break

                if is_fallback:
                    continue

                # P1-4: Validar tamaño de chunk y offset antes de aceptar el caché.
                # Antes el cache podía servir un plan de 3 días cuando el caller
                # pedía 7, o un plan con offset=0 cuando se pedían días 8-14.
                # El frontend renderizaba incompleto / fuera de orden.
                # P1-10: PLAN_CHUNK_SIZE a nivel módulo
                try:
                    expected_days = int(actual_form_data.get("_days_to_generate") or PLAN_CHUNK_SIZE)
                    expected_offset = int(actual_form_data.get("_days_offset") or 0)
                except (TypeError, ValueError):
                    expected_days = PLAN_CHUNK_SIZE
                    expected_offset = 0

                cached_days_count = len(cand_data.get("days", []) or [])
                if cached_days_count != expected_days:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por tamaño de chunk "
                          f"({cached_days_count}d ≠ {expected_days}d esperado).")
                    continue

                _cached_form_for_offset = cand_data.get("form_data") or \
                    cand_data.get("metadata", {}).get("form_data", {}) or {}
                try:
                    cached_offset = int(_cached_form_for_offset.get("_days_offset") or 0)
                except (TypeError, ValueError):
                    cached_offset = 0
                if cached_offset != expected_offset:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por offset "
                          f"({cached_offset} ≠ {expected_offset} esperado).")
                    continue

                # P1-2 (a): Filtro temporal de 30 días de frescura
                created_at_str = plan_candidate.get("created_at") or cand_data.get("created_at")
                if created_at_str:
                    try:
                        created_at_dt = safe_fromisoformat(created_at_str)
                        if created_at_dt.tzinfo is None:
                            created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
                        days_old = (now_utc - created_at_dt).days
                        if days_old > 30:
                            print(f"🗑️ [SEMANTIC CACHE] Plan descartado por antigüedad ({days_old} días > 30).")
                            continue
                    except Exception:
                        pass

                # P1-Q4: Validar drift del target nutricional (calorías + macros).
                # ------------------------------------------------------------
                # Antes el cache solo validaba allergies/medical/diet/dislikes
                # pero NO los macros del calculador. Si el usuario cambió
                # `weight`, `mainGoal` (ej. "perder peso" → "mantener"),
                # `activityLevel` o cualquier input que afecte
                # `get_nutrition_targets`, el cache servía un plan calculado
                # para macros OBSOLETOS. `assemble_plan_node` reescribe los
                # macros AGREGADOS del plan con los nuevos valores (línea ~4039)
                # pero NO redistribuye los `cals/protein/carbs/fats` por meal
                # individual — el plan queda inconsistente: macros agregados
                # del nuevo target, comidas distribuidas para el viejo target.
                #
                # Tolerancia 5%: cubre redondeos del calculador (BMR + TDEE +
                # ajustes de actividad) sin descartar planes con drift trivial.
                # Cambios mayores (e.g. "+500 kcal/día por bulking", goal flip,
                # cambio de actividad) producen ≥10% — descartar es seguro.
                # ------------------------------------------------------------
                _curr_nutrition = state.get("nutrition", {}) or {}
                current_target_cal = (
                    _curr_nutrition.get("total_daily_calories")
                    or _curr_nutrition.get("target_calories")
                    or 0
                )
                cached_target_cal = cand_data.get("calories") or 0
                try:
                    current_target_cal = int(current_target_cal)
                    cached_target_cal = int(cached_target_cal)
                except (TypeError, ValueError):
                    current_target_cal = cached_target_cal = 0

                if current_target_cal > 0 and cached_target_cal > 0:
                    cal_diff_pct = abs(current_target_cal - cached_target_cal) / current_target_cal
                    if cal_diff_pct > 0.05:
                        print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de target "
                              f"calórico (cached={cached_target_cal}kcal, "
                              f"current={current_target_cal}kcal, "
                              f"diff={cal_diff_pct*100:.1f}% > 5%). "
                              f"Probable cambio de peso/goal/actividad.")
                        continue

                # P1-Q4: comparación de macros individuales. Un plan high-protein
                # no debe servirse a un usuario cuyo target ahora es high-carb,
                # aunque las calorías totales coincidan por compensación cruzada.
                # `nutrition["macros"]` en estado: dict con keys `protein_g/carbs_g/fats_g`
                # (ints). En el plan cacheado: dict con keys `protein/carbs/fats`
                # como strings tipo "150g" (`assemble_plan_node` línea ~4042).
                def _parse_macro_g(val):
                    """Extrae cantidad numérica de un macro. Acepta int/float/str."""
                    if isinstance(val, (int, float)):
                        return int(val)
                    if not isinstance(val, str):
                        return 0
                    digits = "".join(c for c in val if c.isdigit())
                    return int(digits) if digits else 0

                _curr_macros = (
                    _curr_nutrition.get("total_daily_macros")
                    or _curr_nutrition.get("macros")
                    or {}
                )
                _cached_macros = cand_data.get("macros") or {}
                if isinstance(_curr_macros, dict) and isinstance(_cached_macros, dict):
                    macro_drift = []
                    for cached_key, current_key in (
                        ("protein", "protein_g"),
                        ("carbs", "carbs_g"),
                        ("fats", "fats_g"),
                    ):
                        cur_g = _curr_macros.get(current_key) or _parse_macro_g(
                            _curr_macros.get(f"{cached_key}_str")
                        )
                        cached_g = _parse_macro_g(_cached_macros.get(cached_key))
                        if cur_g and cached_g:
                            macro_diff = abs(cur_g - cached_g) / cur_g
                            if macro_diff > 0.05:
                                macro_drift.append(
                                    f"{cached_key} (cached={cached_g}g, "
                                    f"current={cur_g}g, diff={macro_diff*100:.1f}%)"
                                )
                    if macro_drift:
                        print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de "
                              f"distribución de macros: {', '.join(macro_drift)}. "
                              f"Probable cambio de goal o estrategia nutricional.")
                        continue

                # P1-2 (b) + P1-4: Compatibilidad Médica Estricta — FAIL-SAFE.
                # Antes, si `cached_form` era None/{} (planes legacy o con
                # estructura corrupta), TODO el bloque de validación se saltaba
                # silenciosamente y el plan se aceptaba sin verificar alergias
                # ni condiciones. Eso permitía servir un plan de un usuario sin
                # alergias a otro con alergia al maní (riesgo de salud directo).
                # Ahora: si no hay form_data válido en el caché, DESCARTAMOS
                # el plan en lugar de aceptarlo a ciegas.
                cached_form = cand_data.get("form_data") or cand_data.get("metadata", {}).get("form_data")
                if not isinstance(cached_form, dict) or not cached_form:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por form_data ausente o inválido "
                          f"(no se puede verificar compatibilidad médica — fail-safe P1-4).")
                    continue

                def _normalize(val):
                    if not val: return ""
                    if isinstance(val, str): return ",".join(sorted([v.strip().lower() for v in val.split(",")]))
                    if isinstance(val, list): return ",".join(sorted([str(v).strip().lower() for v in val]))
                    return ""

                curr_allergies = _normalize(actual_form_data.get("allergies"))
                cache_allergies = _normalize(cached_form.get("allergies"))
                curr_medical = _normalize(actual_form_data.get("medicalConditions"))
                cache_medical = _normalize(cached_form.get("medicalConditions"))
                # P1-4: También validar dietType y dislikes — un plan vegetariano
                # no debe servirse a un omnívoro con preferencias de carne, ni un
                # plan con cebolla a alguien que la rechaza explícitamente.
                curr_diet = _normalize(actual_form_data.get("dietType"))
                cache_diet = _normalize(cached_form.get("dietType"))
                curr_dislikes = _normalize(actual_form_data.get("dislikes"))
                cache_dislikes = _normalize(cached_form.get("dislikes"))
                # P1-ORQ-7: validar cookingTime y budget — ortogonal a los
                # filtros nutricionales (cal/macros) y médicos (allergies/
                # medical/diet/dislikes) ya cubiertos arriba. ANTES, un usuario
                # que cambiaba `cookingTime="plenty"` → `"none"` (porque su
                # agenda cambió y ahora declara no tener tiempo) recibía un
                # cache hit con recetas elaboradas de 60 min — el target
                # calórico no se mueve, así que `cal_diff_pct` no descarta;
                # tampoco las macros (la distribución es la misma); tampoco
                # los filtros médicos. El plan se servía con tactical surface
                # incompatible (instrucciones de cocina largas para alguien
                # que ahora pide 5-min meals) → adherencia destruida sin que
                # el reviewer médico flageara nada (no es safety, es UX).
                # Mismo patrón con `budget`: low → unlimited (o viceversa)
                # cambia el set de ingredientes del catálogo dominicano que
                # `_get_fast_filtered_catalogs` (constants.py:1250) ofrece al
                # LLM, pero no toca calorías/macros. Cache-hit servía recetas
                # de presupuesto incorrecto.
                # Comparación case-insensitive vía `_normalize`. Ambos campos
                # son enums cortos (`none`/`30min`/`1hour`/`plenty` y
                # `low`/`medium`/`high`/`unlimited`) — drift de typing es
                # improbable y la igualdad estricta es segura.
                curr_cooking = _normalize(actual_form_data.get("cookingTime"))
                cache_cooking = _normalize(cached_form.get("cookingTime"))
                curr_budget = _normalize(actual_form_data.get("budget"))
                cache_budget = _normalize(cached_form.get("budget"))

                if curr_allergies != cache_allergies:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de alergias post-hoc.")
                    continue
                if curr_medical != cache_medical:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de condiciones médicas post-hoc.")
                    continue
                if curr_diet != cache_diet:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de tipo de dieta.")
                    continue
                if curr_dislikes != cache_dislikes:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de rechazos/dislikes.")
                    continue
                if curr_cooking != cache_cooking:
                    print(f"🗑️ [SEMANTIC CACHE] P1-ORQ-7: Plan descartado por "
                          f"cambio de cookingTime (cached={cache_cooking!r}, "
                          f"current={curr_cooking!r}). Recetas en caché "
                          f"probablemente no encajan con la nueva capacidad de cocina.")
                    continue
                if curr_budget != cache_budget:
                    print(f"🗑️ [SEMANTIC CACHE] P1-ORQ-7: Plan descartado por "
                          f"cambio de budget (cached={cache_budget!r}, "
                          f"current={curr_budget!r}). Catálogo de ingredientes "
                          f"en caché incompatible con el nuevo presupuesto.")
                    continue

                # P1-ORQ-2: Compatibilidad de despensa. Antes este nodo no
                # validaba pantry — un cache hit servía un plan generado sin
                # awareness de despensa a un usuario que ahora la tenía
                # configurada, disparando rechazo con severity="high" en
                # `review_plan_node` y entregando ámbar evitable. La lógica
                # completa está en `_pantry_cache_discard_reason` (módulo).
                _pantry_discard = _pantry_cache_discard_reason(actual_form_data, cached_form)
                if _pantry_discard:
                    print(f"🗑️ [SEMANTIC CACHE] Plan descartado por {_pantry_discard} "
                          f"(P1-ORQ-2). Regenerar producirá plan pantry-aware.")
                    continue

                valid_plan = plan_candidate
                plan_data = cand_data
                break
                
        if valid_plan and plan_data:
            similar_plan = valid_plan
            
            # 🛡️ Validar Anti-Repetición ANTES de aceptar el caché
            user_id = actual_form_data.get("user_id") or actual_form_data.get("session_id")
            if user_id and user_id != "guest":
                try:
                    from db import get_recent_meals_from_plans
                    from cpu_tasks import _validar_repeticiones_cpu_bound
                    # P0-NEW-1.f: get_recent_meals_from_plans (DB) y
                    # _validar_repeticiones_cpu_bound (CPU intenso) ahora despachadas
                    # al executor / thread pool para no bloquear el loop.
                    recent_meal_names = await _adb(get_recent_meals_from_plans, user_id, 3)
                    if recent_meal_names:
                        # Extraer nombres de platos del plan cacheado
                        cached_meal_names = []
                        for day in plan_data.get("days", []):
                            for meal in day.get("meals", []):
                                if isinstance(meal, dict) and meal.get("name"):
                                    cached_meal_names.append(meal.get("name"))

                        # Validar — la función espera days_plan (lista de días con 'meals'), no nombres planos
                        repeated_meals = await asyncio.to_thread(
                            _validar_repeticiones_cpu_bound,
                            recent_meal_names,
                            plan_data.get("days", []),
                        )
                        generic_ignores = ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa']
                        filtered_repeated = [rm for rm in repeated_meals if not any(g in rm for g in generic_ignores)]
                        
                        # Si más de la mitad de los platos están repetidos, descartar el caché
                        if len(filtered_repeated) > 0 and len(filtered_repeated) >= len(cached_meal_names) / 2:
                            print(f"⚠️ [SEMANTIC CACHE] Plan rechazado por anti-repetición ({len(filtered_repeated)}/{len(cached_meal_names)} platos repetidos). Forzando IA.")
                            return {"semantic_cache_hit": False, "cached_plan_data": None}
                except Exception as e:
                    print(f"⚠️ [SEMANTIC CACHE] Error validando anti-repetición del caché: {e}")

            print(f"🚀 [SEMANTIC CACHE HIT] Plan idéntico encontrado (Similitud: {similar_plan.get('similarity', 0):.3f}). Saltando LLM.")
            return {
                "semantic_cache_hit": True,
                "cached_plan_data": plan_data
            }
                
    return {"semantic_cache_hit": False, "cached_plan_data": None}

def check_cache_hit(state: PlanState) -> str:
    """Edge condicional: Si hubo cache hit, vamos directo al ensamblador."""
    if state.get("semantic_cache_hit"):
        return "hit"
    return "miss"

def build_plan_graph() -> StateGraph:
    """Construye y compila el grafo de orquestación LangGraph Map-Reduce."""

    graph = StateGraph(PlanState)

    # Agregar nodos del pipeline Map-Reduce
    graph.add_node("preflight_optimization", preflight_optimization_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("context_compression", context_compression_node)
    graph.add_node("semantic_cache_check", semantic_cache_check_node)
    graph.add_node("retry_reflection", retry_reflection_node)
    graph.add_node("plan_skeleton", plan_skeleton_node)
    graph.add_node("generate_days_parallel", generate_days_parallel_node)
    graph.add_node("adversarial_judge", adversarial_judge_node)
    graph.add_node("self_critique", self_critique_node)
    graph.add_node("assemble_plan", assemble_plan_node)
    graph.add_node("review_plan", review_plan_node)

    # Definir flujo: preflight -> reflection -> compression -> cache_check -> skeleton/assemble
    graph.set_entry_point("preflight_optimization")
    graph.add_edge("preflight_optimization", "reflection")
    graph.add_edge("reflection", "context_compression")
    graph.add_edge("context_compression", "semantic_cache_check")
    
    # Conditional edge for Semantic Cache
    graph.add_conditional_edges(
        "semantic_cache_check",
        check_cache_hit,
        {
            "hit": "assemble_plan",
            "miss": "plan_skeleton"
        }
    )
    
    graph.add_edge("plan_skeleton", "generate_days_parallel")
    graph.add_edge("generate_days_parallel", "adversarial_judge")
    graph.add_edge("adversarial_judge", "self_critique")
    graph.add_edge("self_critique", "assemble_plan")
    graph.add_edge("assemble_plan", "review_plan")
    graph.add_edge("retry_reflection", "plan_skeleton")

    # Edge condicional: revisor decide si regenerar o terminar
    graph.add_conditional_edges(
        "review_plan",
        should_retry,
        {
            "retry": "retry_reflection",  # Vuelve a reflexión ligera en caso de rechazo (GAP 1)
            "end": END
        }
    )

    return graph.compile()

# ============================================================
# P1-Q2: Lazy-init thread-safe del grafo compilado.
# ------------------------------------------------------------
# Antes: `_PLAN_GRAPH = build_plan_graph()` corría al import del módulo.
# Si LangGraph cambiaba de API, una constante referenciada se renombraba,
# o un import upstream fallaba (schemas, prompts, tools_*), el `ImportError`
# derribaba el worker entero al cargar — sin telemetría útil más allá del
# traceback de `import graph_orchestrator`. Outage total invisible hasta que
# el orquestador (Kubernetes / Gunicorn master) reportara worker-down, y
# sin posibilidad de servir healthchecks "starting" o `liveness` mientras
# el problema se diagnosticaba.
#
# Ahora: el grafo se construye on-demand en `_get_plan_graph()` con
# double-checked locking + try/except. La PRIMERA request paga la latencia
# de compile (típicamente <100ms; LangGraph compile es CPU-puro), pero el
# worker arranca aunque LangGraph esté en mal estado. Si el build falla en
# runtime, la excepción propaga al `arun_plan_pipeline.except`, que dispara
# el fallback matemático (P0-1) — el cliente recibe un plan de contingencia
# en lugar de un 5xx, y se loguea CRITICAL para alerting inmediato.
#
# Reintentos: si una request falla al construir, las siguientes reintentan
# automáticamente en la próxima entrada al lock. Útil cuando el motivo es
# transitorio (módulo upstream que se reimportó después del primer intento,
# config dinámica que se cargó mid-flight, etc.).
# ============================================================
_PLAN_GRAPH = None  # P1-Q2: lazy, antes era eager `build_plan_graph()`
_PLAN_GRAPH_LOCK = threading.Lock()
_PLAN_GRAPH_BUILD_FAILURES = 0  # P1-Q2: counter observable para alerting

# P1-A4: contador acumulado de invalidaciones manuales/programáticas del grafo.
# Una invalidación significa "el grafo cacheado se descartó intencionalmente y
# la próxima request reconstruirá". Métrica útil para correlacionar con incidentes:
# spikes de invalidations + builds_failures alto suele indicar corrupción upstream
# (prompts/, schemas, langgraph version mismatch).
_PLAN_GRAPH_INVALIDATIONS_TOTAL = 0
# P1-A4: timestamp de la última invalidación + razón. Solo último evento; útil
# para `/api/system/health/plan-graph` y debug en triage de incidentes. Sin
# historial — Grafana captura el time-series desde el counter monótono.
_PLAN_GRAPH_LAST_INVALIDATION_TS: float | None = None
_PLAN_GRAPH_LAST_INVALIDATION_REASON: str | None = None


def _get_plan_graph():
    """P1-Q2: devuelve el grafo compilado, construyéndolo lazy + thread-safe.

    Double-checked locking: el check sin lock cubre el 99.9% de los casos
    (singleton ya construido, costo ~ns), el lock solo se toma en la PRIMERA
    request por proceso. Cualquier excepción durante el build se loguea como
    CRITICAL y se propaga al caller — `arun_plan_pipeline` la captura en su
    `except` global y dispara el fallback matemático (P0-1), de modo que el
    cliente recibe un plan de contingencia en lugar de un 5xx.

    Si una request previa falló al construir, las siguientes reintentan
    automáticamente. El counter `_PLAN_GRAPH_BUILD_FAILURES` permite alertar
    sobre fallos sostenidos vs un blip transitorio.
    """
    global _PLAN_GRAPH, _PLAN_GRAPH_BUILD_FAILURES
    if _PLAN_GRAPH is not None:
        return _PLAN_GRAPH
    with _PLAN_GRAPH_LOCK:
        if _PLAN_GRAPH is not None:
            return _PLAN_GRAPH
        try:
            graph = build_plan_graph()
        except Exception as e:
            _PLAN_GRAPH_BUILD_FAILURES += 1
            logger.critical(
                f"[STARTUP] P1-Q2: build_plan_graph() falló "
                f"(intento #{_PLAN_GRAPH_BUILD_FAILURES}): "
                f"{type(e).__name__}: {e}. Las requests caerán al fallback "
                f"matemático hasta que el problema se resuelva. Revisa: "
                f"schema de PlanState, LangGraph version, imports de nodos "
                f"(prompts/, schemas, tools_*)."
            )
            raise
        _PLAN_GRAPH = graph
        logger.info(
            "[STARTUP] P1-Q2: LangGraph plan_graph compilado exitosamente "
            f"(failures previos: {_PLAN_GRAPH_BUILD_FAILURES})."
        )
        return _PLAN_GRAPH


def is_plan_graph_ready() -> bool:
    """P1-Q2: helper para readiness probes (Kubernetes / load balancer).

    Devuelve True si el grafo está construido y listo para servir. Útil para
    gating de readiness: el orquestador puede llamar `warm_plan_graph()` al
    startup para forzar la construcción y luego usar este helper en el
    `/ready` endpoint para no enrutar tráfico hasta que el grafo esté listo.
    """
    return _PLAN_GRAPH is not None


def warm_plan_graph() -> bool:
    """P1-Q2: fuerza la construcción del grafo (warm-up al startup).

    Llamar desde un `lifespan` handler de FastAPI o un init script para mover
    la latencia de compile fuera del path de la primera request real.
    Devuelve True si la construcción tuvo éxito; False si falló (el caller
    puede decidir si hacer fail-fast o continuar y dejar que el fallback
    matemático cubra). No re-lanza la excepción para no derribar el startup
    cuando se invoca desde código non-critical.
    """
    try:
        _get_plan_graph()
        return True
    except Exception:
        # Ya está logueado como CRITICAL en `_get_plan_graph`. No re-lanzar
        # para que el caller pueda decidir tolerar y servir requests vía
        # fallback (mejor que crashear el proceso).
        return False


def invalidate_plan_graph(reason: str = "manual") -> dict:
    """P1-A4: descarta el grafo cacheado para forzar reconstrucción on-demand.

    Casos de uso:
      - Hot-reload: tras editar un prompt, schema o nodo en runtime sin
        redeployar (debugging / incidente en producción), invalidar fuerza
        que la próxima request use el código actualizado en lugar del
        snapshot cargado en `_PLAN_GRAPH`.
      - Recover de corrupción detectada: si un componente upstream se
        modificó por monkeypatch y el grafo cacheado quedó referenciando un
        objeto stale, esta función limpia el estado.
      - Operacional: endpoint admin (`POST /api/system/admin/plan-graph/invalidate`)
        con auth Bearer permite a SRE/oncall reiniciar el grafo sin tocar
        el proceso/contenedor.
      - Tests: facilita aislar cada test sin reimportar el módulo entero.

    Diseño:
      - Toma `_PLAN_GRAPH_LOCK` para coordinarse con `_get_plan_graph`. Si
        un build estaba en curso, se completa primero y luego invalida —
        ese resultado se descarta inmediatamente, lo cual es OK (es la
        semántica deseada de "tira lo que tengas, voy a empezar de cero").
      - Incrementa `_PLAN_GRAPH_INVALIDATIONS_TOTAL` (counter monótono para
        time-series en Grafana).
      - Persiste `reason` y `timestamp` para `get_plan_graph_status()`.
      - NO reconstruye inmediatamente: retorna y la próxima request paga
        la latencia de compile. Caller que quiera warm-up post-invalidate
        debe llamar `warm_plan_graph()` después.
      - Idempotente: invalidar dos veces seguidas suma 2 al counter pero
        no falla aunque el grafo ya estuviera en None.

    Args:
        reason: descripción corta del motivo (loggeada + retornada). Útil
            para auditoría — qué disparó cada invalidación. Default `"manual"`.

    Returns:
        Snapshot `dict` post-invalidación (mismas keys que `get_plan_graph_status`).
    """
    global _PLAN_GRAPH, _PLAN_GRAPH_INVALIDATIONS_TOTAL
    global _PLAN_GRAPH_LAST_INVALIDATION_TS, _PLAN_GRAPH_LAST_INVALIDATION_REASON

    safe_reason = (reason or "manual").strip()[:200] or "manual"
    with _PLAN_GRAPH_LOCK:
        was_built = _PLAN_GRAPH is not None
        _PLAN_GRAPH = None
        _PLAN_GRAPH_INVALIDATIONS_TOTAL += 1
        _PLAN_GRAPH_LAST_INVALIDATION_TS = time.time()
        _PLAN_GRAPH_LAST_INVALIDATION_REASON = safe_reason
        invalidations_now = _PLAN_GRAPH_INVALIDATIONS_TOTAL

    logger.info(
        f"[ORQUESTADOR] P1-A4: plan_graph invalidado "
        f"(reason={safe_reason!r}, was_built={was_built}, "
        f"invalidations_total={invalidations_now}). "
        f"Próxima request reconstruirá el grafo."
    )
    return get_plan_graph_status()


def get_plan_graph_status() -> dict:
    """[P1-9] Snapshot estructurado del estado de salud del grafo LangGraph.

    Útil para health endpoints / dashboards / alerting. Lo que `/ready` (en
    `app.py`) hace en modo binario (200 si listo, 503 si no), este helper lo
    expone con detalle:

    - `ready`: bool. True si el grafo está compilado (`_PLAN_GRAPH is not None`).
    - `build_failures`: int. Conteo acumulado de intentos de build fallidos.
      Una vez compilado exitosamente, se mantiene > 0 si hubo fallos previos —
      útil para detectar inestabilidad histórica aunque el grafo se haya
      recuperado.
    - `status`: literal `"ready"` | `"not_ready"`.
    - `message`: descripción human-readable para incluir en respuestas API.

    No hace `warm_plan_graph()` desde aquí: ese trabajo lo hace el lifespan
    de FastAPI al startup. Si el caller necesita forzar build, debe usar
    `warm_plan_graph()` explícitamente.

    P1-A4: Campos adicionales:
    - `invalidations_total`: counter monótono incrementado por cada llamada
      a `invalidate_plan_graph()`. Time-series-friendly para Grafana.
    - `last_invalidation_ts` / `last_invalidation_reason`: snapshot del último
      evento (None si nunca se invalidó). Útil en triage para correlacionar
      con tracebacks CRITICAL en logs.
    """
    ready = _PLAN_GRAPH is not None
    return {
        "ready": ready,
        "build_failures": _PLAN_GRAPH_BUILD_FAILURES,
        "invalidations_total": _PLAN_GRAPH_INVALIDATIONS_TOTAL,
        "last_invalidation_ts": _PLAN_GRAPH_LAST_INVALIDATION_TS,
        "last_invalidation_reason": _PLAN_GRAPH_LAST_INVALIDATION_REASON,
        "status": "ready" if ready else "not_ready",
        "message": (
            "LangGraph plan_graph compilado y listo para servir."
            if ready
            else (
                "LangGraph plan_graph no está compilado. Requests al pipeline "
                "caerán al fallback matemático. Revisa logs CRITICAL para el "
                "traceback del último build fallido."
            )
        ),
    }


# ============================================================
# P1-Q10: Probe + flag para inserts de métricas con user_id NULL.
# ------------------------------------------------------------
# Antes, cuando `_emit_progress` (event="metric") o `_log_adversarial_metric`
# o `_persist_sanitize_metric` intentaban INSERT con `user_id=None` para
# pipelines de guests, el resultado dependía del schema de `pipeline_metrics`:
#   - Si `user_id` permitía NULL → todo OK.
#   - Si quedó como `NOT NULL` (estado en algunos entornos por crear la tabla
#     fuera de migrations/) → cada insert lanzaba `IntegrityError` capturada
#     por el `except Exception` genérico del emitter, logueada como
#     `logger.error("Failed to insert metric: ...")` SIN distinguir entre
#     "DB caída" vs "schema bug" vs "constraint violado". Resultado: pérdida
#     silenciosa del 100% de la señal de meta-learning del segmento guest.
#
# La migración en `app.py` lifespan (P1-Q10) hace `ALTER COLUMN user_id DROP
# NOT NULL` idempotente, pero este probe es la red de seguridad que verifica
# que efectivamente quedó así — y si no, deja el flag `_GUEST_METRICS_ENABLED`
# en False para que los emitters skipeen guest inserts gracefully en lugar
# de fallar 50× por pipeline. El probe se llama 1× al startup desde el
# lifespan, NO por cada request.
# ============================================================
_GUEST_METRICS_ENABLED = True  # default optimista; el probe lo desactiva si schema mal


def _is_guest_metrics_enabled() -> bool:
    """P1-Q10: snapshot del flag para emitters. Lectura atómica (bool en GIL)."""
    return _GUEST_METRICS_ENABLED


def verify_pipeline_metrics_guest_insert() -> bool:
    """P1-Q10: probe de schema de `pipeline_metrics` para detectar drift.

    Hace un INSERT de prueba con `user_id=NULL` envuelto en transacción y
    rollback explícito (no deja datos residuales). Si el INSERT falla con
    IntegrityError (NOT NULL constraint) u otro error de schema:
      - Loguea CRITICAL con remediation steps explícitos.
      - Setea `_GUEST_METRICS_ENABLED = False` para que los emitters skipeen
        guest inserts en lugar de fallar 50× por pipeline.
      - Retorna False.

    Si el INSERT funciona, retorna True (default `_GUEST_METRICS_ENABLED=True`
    se preserva).

    Llamar 1× al startup desde el `lifespan` de FastAPI, después de la
    migración (que es la que crea/repara el schema). Esta función NO crea
    la tabla — solo verifica que la migración funcionó.
    """
    global _GUEST_METRICS_ENABLED
    try:
        # Probe: INSERT + ROLLBACK manual. Usamos `execute_sql_query` con un
        # bloque explícito de DO + ROLLBACK para no requerir API transaccional
        # nueva. Más simple: INSERT y luego DELETE inmediato del row de prueba.
        # `node='__schema_probe__'` distingue el row si por alguna razón el
        # DELETE falla (queda como artefacto identificable, no contamina
        # analytics).
        from db_core import execute_sql_write
        probe_node = "__schema_probe_p1_q10__"
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, "
            "duration_ms, retries, tokens_estimated, confidence, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                None,  # ← el caso que estamos probando: guest insert
                "__startup_probe__",
                probe_node,
                0, 0, 0, 0.0,
                json.dumps({"probe": True, "note": "P1-Q10 schema verification"}),
            ),
        )
        # Cleanup: borrar el row de prueba. Best-effort; si falla solo deja
        # un row identificable que no afecta analytics (filtrable por node).
        try:
            execute_sql_write(
                "DELETE FROM pipeline_metrics WHERE node = %s AND session_id = %s",
                (probe_node, "__startup_probe__"),
            )
        except Exception as cleanup_err:
            logger.warning(
                f"[STARTUP] P1-Q10: probe row no cleanup-eado "
                f"(node={probe_node!r}): {cleanup_err}. Filtrar manualmente "
                f"con DELETE FROM pipeline_metrics WHERE node='{probe_node}'."
            )
        _GUEST_METRICS_ENABLED = True
        logger.info(
            "[STARTUP] P1-Q10: pipeline_metrics acepta inserts con user_id=NULL "
            "(guest metrics habilitadas)."
        )
        return True
    except Exception as probe_err:
        _GUEST_METRICS_ENABLED = False
        logger.critical(
            f"[STARTUP] P1-Q10: pipeline_metrics RECHAZA inserts con user_id=NULL "
            f"({type(probe_err).__name__}: {probe_err}). Las métricas de pipelines "
            f"de guests NO se persistirán durante esta vida del proceso. "
            f"Remediation:\n"
            f"  1. Verificar que la migración del lifespan corrió: "
            f"     `\\d pipeline_metrics` debe mostrar `user_id ... null`.\n"
            f"  2. Si quedó NOT NULL, ejecutar manualmente: "
            f"     `ALTER TABLE pipeline_metrics ALTER COLUMN user_id DROP NOT NULL;`\n"
            f"  3. Reiniciar el worker tras la fix; este probe re-evalúa al startup."
        )
        return False


def _persist_holistic_score_sync(user_id: str, holistic_score: float,
                                  attempts: int, review_passed: bool) -> None:
    """P1-X5: persistencia sync del holistic score, despachada al
    `_METRICS_EXECUTOR` desde `arun_plan_pipeline`.

    Antes este bloque corría inline al final de `arun_plan_pipeline` con dos
    awaits secuenciales (`_adb(get_user_profile)` + `_adb(update_user_health_profile)`),
    añadiendo ~100-300ms al p99 perceptible al cliente por una operación de
    meta-learning que NO necesita bloquear la respuesta.

    Despachado a `_METRICS_EXECUTOR` (sync executor proceso-wide):
      - El thread sobrevive al cierre del loop del caller (importante para
        el sync wrapper / cron donde `asyncio.run` cancelaría tasks pendientes).
      - El executor tiene drain bounded vía `_shutdown_drainable_executor`
        (P1-NEW-3 / P1-Q9) — bajo SIGTERM se drena hasta `METRICS_SHUTDOWN_DRAIN_S`.
      - ContextVars del caller propagados via `ctx.run` (P1-X3) → logs
        muestran el `request_id_var` del pipeline original.

    Best-effort: cualquier excepción se silencia con un warning. El holistic
    score es señal de meta-learning, no crítica para entregar el plan al cliente.
    """
    try:
        # [P1-ORQ-1] Read-modify-write atómico para evitar lost-update bajo
        # concurrencia del mismo user_id (2 tabs regenerando, cron + acción
        # manual del usuario). Antes, el patrón get → mutate → update perdía
        # entries de `pipeline_score_history` cuando dos pipelines del mismo
        # usuario commiteaban simultáneamente.
        from db_profiles import update_user_health_profile_atomic

        def _holistic_mutator(hp):
            hp["last_pipeline_score"] = holistic_score
            hp["last_pipeline_attempts"] = attempts
            score_history = hp.get("pipeline_score_history") or []
            if not isinstance(score_history, list):
                score_history = []
            score_history.append({
                "score": holistic_score,
                "attempts": attempts,
                "ts": datetime.now(timezone.utc).isoformat(),
                "review_passed": bool(review_passed),
            })
            # Mantener solo los últimos 10 (rolling window).
            hp["pipeline_score_history"] = score_history[-10:]

        new_hp = update_user_health_profile_atomic(user_id, _holistic_mutator)
        if new_hp is None:
            return
        print(f"💾 [HOLISTIC SCORE] Guardado en health_profile "
              f"(historial: {len(new_hp.get('pipeline_score_history', []))} entradas).")
    except Exception as db_err:
        print(f"⚠️ [HOLISTIC SCORE] Error guardando en DB: {db_err}")


# ============================================================
# P1-Q8: Sanitización anti-prompt-injection (recursiva, observable)
# ------------------------------------------------------------
# Antes la sanitización solo cubría 5 campos fijos (mainGoal, otherAllergies,
# otherConditions, struggles, notes). Eso dejaba abiertos vectores como
# items individuales de `dislikes`/`medicalConditions` (si el frontend acepta
# texto libre), `customMealNotes` y cualquier campo nuevo que se añada al
# form sin actualizar el whitelist. Además, no resistía a evasiones triviales
# con caracteres invisibles ("ig​nore previous" insertando ZWSP entre
# letras).
#
# Ahora:
#   - `_sanitize_form_data_recursive` walks dicts y listas a cualquier
#     profundidad, sanitizando TODO string con `len >= _SANITIZE_MIN_LEN`.
#     Threshold mínimo evita false positives sobre IDs cortos / enums.
#   - `_sanitize_text_normalize` strip de zero-width chars + bidi controls
#     ANTES del NFKD → un atacante no puede slip patrones evadiendo con
#     caracteres invisibles ni con override de dirección de texto.
#   - Hits acumulados con `path/pattern/snippet` → log + métrica persistida
#     en `pipeline_metrics` para alerting (Grafana/Prometheus query sobre
#     `node='sanitize_hits'` filtrado por `user_id` detecta abuse patterns).
#
# Vocabulario module-level (frozenset) para permitir reutilización futura
# desde otros módulos (chat, RAG, fact-extractor) sin redefinir. Centralizar
# en `constants.py` queda como follow-up — por ahora mantenemos aquí para
# minimizar el blast radius de este cambio.
# ============================================================
_PROMPT_INJECTION_PATTERNS: frozenset = frozenset({
    # Inglés
    "ignore previous", "ignore above", "ignore all", "disregard",
    "system prompt", "you are now", "act as", "new instructions",
    "override", "forget everything", "reveal your", "output all",
    # Español
    "ignora", "olvida", "nuevas instrucciones", "actua como",
    "eres ahora", "revela tu", "imprime tu",
})

# P1-Q8: caracteres invisibles que un atacante puede usar para evasion:
#   - U+200B..U+200D: zero-width space, non-joiner, joiner
#   - U+FEFF: ZWNBSP (BOM mid-string)
#   - U+202A..U+202E: bidi explicit overrides (LRE, RLE, PDF, LRO, RLO)
#   - U+2066..U+2069: bidi isolates (LRI, RLI, FSI, PDI)
# NFKD no los descompone — hay que strippearlos explícitamente antes.
_INVISIBLE_CHAR_RX = _re.compile(r"[​-‍﻿‪-‮⁦-⁩]")

# P1-Q8: mínimo de longitud para considerar sanitización. Patrones de
# injection son sustantivos ("ignore previous instructions and..."); strings
# cortos típicamente son IDs, enums (vegan/balanced/keto), códigos. Threshold
# 20 evita false positives sobre identificadores sin perder cobertura.
_SANITIZE_MIN_LEN = 20


# ============================================================
# P0-A2: Whitelist de claves internas con prefijo `_` que el backend escribe
# legítimamente en `form_data`/`pipeline_data` antes de pasar al pipeline.
# ------------------------------------------------------------
# Antes, claves como `_quality_hint`, `_emotional_state`, `_days_to_generate`,
# `_use_adversarial_play` se leían directo del payload del cliente (deepcopy de
# `form_data`). Un cliente malicioso/scraper podía:
#   - Setear `_quality_hint="drastic_change"` y forzar la estrategia "cambio
#     radical" del orquestador (línea ~7041), distorsionando las métricas de
#     aprendizaje y manipulando el plan recibido.
#   - Setear `_emotional_state="needs_comfort"` para forzar comfort food.
#   - Setear `_days_to_generate=9999` y disparar trabajo desproporcionado del
#     pipeline (fallback matemático con N días, asignación de memoria, etc.).
#   - Setear `_use_adversarial_play=True` y forzar el judge adversarial gratis.
#
# Defensa de doble capa:
#   1. `routers/plans.py` (capa estricta, sin whitelist): justo después del
#      `pipeline_data = dict(data)`, stripea TODAS las `_keys` del cliente
#      antes de que el backend escriba las suyas. Es el filtro principal.
#   2. `arun_plan_pipeline` (capa con whitelist, defense-in-depth): para callers
#      no oficiales (cron_tasks, proactive_agent, scripts internos legacy) que
#      pueden no haber pasado por la capa 1. Permite las claves de esta
#      whitelist y stripea cualquier otra.
#
# Mantenimiento: cuando se introduce una `_key` nueva escrita por el backend,
# AGREGARLA aquí. Buscar con: `grep -nE "(form_data|pipeline_data)\\[\"_\\w+\"\\]\\s*=" backend/`
# ============================================================
_TRUSTED_INTERNAL_FORM_KEYS: frozenset = frozenset({
    # routers/plans.py — derivadas de la request (no del payload)
    "_plan_start_date",
    "_days_to_generate",

    # cron_tasks.py — pantry / scheduling / reintentos
    "_fresh_pantry_source",
    "_tz_drift_at_pantry_refresh_minutes",
    "_tz_major_drift_at_pantry_refresh",
    "_pantry_snapshot_age_hours",
    "_live_fetch_backoff_durations_ms",
    "_pantry_paused",
    "_requires_pantry_review",
    "_pantry_flexible_mode",
    "_pantry_advisory_only",
    "_inventory_live_degraded",
    "_pantry_degraded_reason",
    "_pantry_correction",
    "_is_emergency_generation",
    "_plan_start_date_fallback_logged",
    "_pantry_diversity_warning",
    "_pantry_warning_sent",
    "_learning_signal_strength",
    "_inventory_activity_proxy_used",
    "_inventory_activity_mutations",
    "_force_variety",
    "_blocked_techniques",
    "_learning_forced",
    "_sparse_logging_proxy",
    "_learning_forced_reason",
    "_failed_chunk_learning_disabled",
    "_failed_chunk_predecessor_weeks",
    "_days_offset",
    "_chunk_prior_meals",
    "_force_technique_variety",
    "_learning_window_starved",
    "_chunk_lessons",
    "_last_technique",
    "_prev_chunk_adherence",
    "_pantry_drift_warning",
    "_drift_retries",
    "_pantry_quantity_violations",
    "_strict_post_gen_required",
    "_pantry_supplement_required",

    # cron_tasks.py — `inject_learning_signals_from_profile` (señales meta-learning)
    "_meal_level_adherence",
    "_meal_level_adherence_long",
    "_adherence_ema_hint",
    "_adherence_hint",
    "_abandoned_reasons",
    "_emotional_state",
    "_previous_plan_quality",
    "_quality_hint",
    "_drastic_change_strategy",
    "_ignored_meal_types",
    "_frustrated_meal_types",
    "_cold_start_recommendations",
    "_liked_meals",
    "_liked_flavor_profiles",
    "_llm_retrospective",
    "_use_adversarial_play",

    # graph_orchestrator.py — flags internos (preflight + reroll)
    "_auto_simplify",
    "_pipeline_drift_alert",
    "_creative_freedom",
    "_is_rotation_reroll",
    "_is_same_day_reroll",
})


# P0-A2: cap absoluto al campo `_days_to_generate`. El orquestador nunca debe
# generar más de esto en un solo run del pipeline; chunks adicionales se
# encolan por `cron_tasks` con su propia gestión de carga. Cliente que envíe
# `_days_to_generate=9999` queda capeado a `PLAN_CHUNK_SIZE`.
_MAX_DAYS_TO_GENERATE = PLAN_CHUNK_SIZE


def _strip_untrusted_internal_keys(
    form_data: dict,
    *,
    allow_set: frozenset | None = None,
    log_prefix: str = "STRIP",
) -> list:
    """P0-A2: elimina del dict (in-place) las claves con prefijo `_` que NO
    estén en `allow_set`. Devuelve la lista de claves dropeadas.

    Si `allow_set` es None → modo ESTRICTO: stripea TODA `_key`. Útil para
    `routers/plans.py` justo después de copiar el payload del cliente, antes
    de que el backend inyecte sus propias claves legítimas.

    Si `allow_set` es la whitelist (`_TRUSTED_INTERNAL_FORM_KEYS`) → modo
    PERMISIVO: solo stripea claves desconocidas. Útil para `arun_plan_pipeline`
    como defense-in-depth tras callers que pueden haberse saltado la capa
    estricta de `routers/plans.py` (cron_tasks, proactive_agent, etc.).

    Las claves dropeadas se loguean a nivel WARNING para alerting; persistir
    métrica queda fuera de scope (low-volume signal — un cliente normal nunca
    envía `_keys` y un atacante coordinado se detecta por cardinalidad de
    sanitize_hits ya existente).
    """
    if not isinstance(form_data, dict):
        return []
    dropped = []
    for key in list(form_data.keys()):
        if not isinstance(key, str) or not key.startswith("_"):
            continue
        if allow_set is not None and key in allow_set:
            continue
        dropped.append(key)
        del form_data[key]
    if dropped:
        logger.warning(
            f"🛡️ [{log_prefix}] P0-A2: stripped {len(dropped)} clave(s) interna(s) "
            f"no autorizada(s) del payload: {dropped[:20]}"
            f"{' ...(truncado)' if len(dropped) > 20 else ''}"
        )
    return dropped


def _enforce_days_to_generate_cap(form_data: dict, *, log_prefix: str = "CAP") -> bool:
    """P0-A2: garantiza que `_days_to_generate` no exceda el máximo permitido.

    Acepta valores `int` o convertibles a `int`. Valores no-numéricos, negativos
    o mayores al cap se reemplazan por `_MAX_DAYS_TO_GENERATE`. Retorna `True`
    si modificó el valor (para logging/alerting).

    Importante: este enforce corre DESPUÉS del strip de claves no whitelisted
    — si la clave no estaba en `_TRUSTED_INTERNAL_FORM_KEYS` ya fue eliminada
    y este helper no encuentra nada que capear.
    """
    if not isinstance(form_data, dict):
        return False
    raw = form_data.get("_days_to_generate")
    if raw is None:
        return False
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"🛡️ [{log_prefix}] P0-A2: `_days_to_generate={raw!r}` no es int. "
            f"Reemplazando por {_MAX_DAYS_TO_GENERATE}."
        )
        form_data["_days_to_generate"] = _MAX_DAYS_TO_GENERATE
        return True
    if val < 1 or val > _MAX_DAYS_TO_GENERATE:
        logger.warning(
            f"🛡️ [{log_prefix}] P0-A2: `_days_to_generate={val}` fuera de rango "
            f"[1, {_MAX_DAYS_TO_GENERATE}]. Capeando a {_MAX_DAYS_TO_GENERATE}."
        )
        form_data["_days_to_generate"] = _MAX_DAYS_TO_GENERATE
        return True
    return False


# ============================================================
# P1-A8: Validación de enum para hints de meta-learning.
# ------------------------------------------------------------
# Antes, el orquestador leía `_quality_hint`, `_emotional_state`,
# `_drastic_change_strategy`, `_adherence_hint`, `_adherence_ema_hint` con
# patrones tipo `if quality_hint == "drastic_change":`. Si una cron de
# aprendizaje persistía un valor inesperado (rename interno, feature flag
# de A/B testing, typo en una constante), el bloque hacía `if X == ...` sin
# match y silenciosamente NO inyectaba ninguna estrategia → la señal de
# meta-learning se perdía sin warning, sin fallback, sin métrica.
#
# P0-A2 ya stripea las claves `_` que el cliente puede inyectar. P1-A8 cubre
# el caso ortogonal: la clave VIENE del backend (whitelisted), pero su VALOR
# está fuera del enum esperado. Dos paths:
#   1. Bug interno: `cron_tasks` o `inject_learning_signals_from_profile`
#      escribe un valor mal tipeado (ej. "drastic-change" con guion en lugar
#      de underscore). Sin validación, perdemos la señal sin alertar.
#   2. Drift de schema: el código del orquestador agrega una nueva rama y
#      olvida actualizar el enum aquí. La validación falla en CI/staging y
#      el operador se da cuenta antes de prod.
#
# Comportamiento: valor desconocido → log WARNING + clear del campo (None).
# Preferimos perder la señal con observabilidad que ejecutar una rama no
# intencionada.
#
# Mantenimiento: cuando `cron_tasks` agregue un valor nuevo (ej. otra
# estrategia drástica), AGREGARLO aquí. Buscar con: `grep -nE
# "(_quality_hint|_emotional_state|_drastic_change_strategy|_adherence_hint|_adherence_ema_hint)"
# backend/cron_tasks.py backend/graph_orchestrator.py`.
# ============================================================
_FORM_HINT_ENUMS: dict[str, frozenset[str]] = {
    "_quality_hint": frozenset({
        "drastic_change", "increase_complexity",
        "break_plateau", "simplify_urgently",
    }),
    "_drastic_change_strategy": frozenset({
        "ethnic_rotation", "texture_swap", "protein_shock",
    }),
    "_emotional_state": frozenset({
        "needs_comfort", "ready_for_challenge",
    }),
    "_adherence_hint": frozenset({"low", "high"}),
    "_adherence_ema_hint": frozenset({
        "temporary_dip", "drastic_change", "improving",
    }),
}


def _validate_form_hint_enums(
    form_data: dict,
    *,
    log_prefix: str = "ORQUESTADOR",
) -> list[tuple[str, str]]:
    """P1-A8: valida que cada hint enum del form_data tenga un valor permitido.

    Para cada clave en `_FORM_HINT_ENUMS`:
      - Si está ausente o `None` → no-op.
      - Si su valor está en la whitelist → no-op (caso normal).
      - Si su valor NO está en la whitelist → log WARNING con el valor offending
        y la lista de valores válidos, y CLEAR el campo (set a None). El
        orquestador después leerá `None` y skipeará la rama silenciosamente,
        idéntico al comportamiento esperado para "no hay señal".

    Devuelve la lista de `(key, bad_value)` para que el caller pueda emitir
    métrica si lo desea. El return vacío es el path normal.

    No re-lanza; degradación silenciosa de la señal con observabilidad.
    """
    if not isinstance(form_data, dict):
        return []
    bad = []
    for key, allowed in _FORM_HINT_ENUMS.items():
        val = form_data.get(key)
        if val is None or val == "":
            continue
        if isinstance(val, str) and val in allowed:
            continue
        bad.append((key, str(val)[:60]))
        logger.warning(
            f"🛡️ [{log_prefix}] P1-A8: `{key}={val!r}` fuera del enum permitido "
            f"({sorted(allowed)}). Limpiando a None — la señal se descarta. "
            f"Posibles causas: (1) cron escribió valor mal tipeado, (2) drift "
            f"de schema sin actualizar `_FORM_HINT_ENUMS`."
        )
        form_data[key] = None
    return bad


# ============================================================
# P1-1 / P1-2: Merge de campos `other*` (texto libre del wizard) en sus arrays
# canónicos
# ------------------------------------------------------------
# Antes, `otherAllergies`/`otherConditions` solo se mergeaban DENTRO de
# `review_plan_node` (lectura local de `allergies`/`medical_conditions`), y
# `otherDislikes`/`otherStruggles` no se mergeaban EN NINGÚN sitio del
# backend — quedaban huérfanos como texto crudo en el JSON dump del prompt.
# Eso significaba que las capas previas leían los arrays SIN el texto libre:
#   - `semantic_cache_check_node` (línea ~6451) → cache hit puede servir un
#     plan con el alérgeno o ingrediente rechazado a otro usuario que matchea
#     por arrays; review (en el caso de allergies) luego rechaza → retry → 2×
#     pipeline. Para `dislikes` el review no flagea (no es safety-critical),
#     así que el plan se servía con el ingrediente rechazado.
#   - `_get_fast_filtered_catalogs` (`ai_helpers.py:137`) → catálogo no excluye
#     el ingrediente textual, así el LLM lo ofrece como protein/carb/veggie
#     candidate.
#   - RAG dynamic_query (`arun_plan_pipeline:~8559-8664`) → no busca contexto
#     histórico del alérgeno/dislike/struggle textual.
#
# Ahora el merge se aplica UNA VEZ al inicio del pipeline (en `arun_plan_pipeline`
# justo después de `_validate_form_hint_enums`), y todos los nodos downstream
# leen los arrays canónicos ya unificados. Eliminamos la duplicación local del
# review_plan_node.
#
# Cobertura por field (P1-1 = allergies/medicalConditions; P1-2 = dislikes/struggles):
#   - `allergies/otherAllergies`         → P1-1 (safety-critical)
#   - `medicalConditions/otherConditions` → P1-1 (safety-critical)
#   - `dislikes/otherDislikes`           → P1-2 (UX/calidad)
#   - `struggles/otherStruggles`         → P1-2 (RAG context + JSON dump)
#
# Reglas de parseo (idénticas al merge legacy del review):
#   - Split por commas, strip whitespace, descarta vacíos.
#   - Sin normalización de capitalización: el LLM y el reviewer aceptan ambos
#     casos. La normalización para comparación la hace cada caller (catalog
#     usa `strip_accents().lower()`, cache usa `.lower()`).
# ============================================================
_OTHER_TEXT_FIELD_MAP = (
    ("allergies",         "otherAllergies"),
    ("medicalConditions", "otherConditions"),
    # P1-2: el wizard también captura `otherDislikes` (paso QDislikes, P1-B5)
    # y `otherStruggles` (paso QStruggles). Antes de este merge ambos campos
    # quedaban huérfanos en backend — solo aparecían como texto crudo en el
    # JSON dump del prompt. Con el merge:
    #   - `dislikes` ← lo respetan `_get_fast_filtered_catalogs` (filtro de
    #     catálogo dominicano), `semantic_cache_check_node` (comparación con
    #     plan cacheado, evita servir un plan con apio a alguien que escribió
    #     "Apio" en el textbox), RAG dynamic_query (línea ~8659).
    #   - `struggles` ← lo respeta RAG dynamic_query (línea ~8662) para traer
    #     contexto de obstáculos textualmente declarados; el resto de los nodos
    #     leen `struggles` solo vía el JSON dump del prompt (informacional),
    #     pero la unión sigue siendo útil para coherencia.
    ("dislikes",          "otherDislikes"),
    ("struggles",         "otherStruggles"),
)

# P0-FORM-1: sentinels exclusivos que el wizard usa para "no aplica".
# Comparación case-insensitive contra el array canónico. Si el array contiene
# alguno de estos valores, el `other*` asociado se descarta antes del merge:
# preserva la promesa de exclusividad mutua que el frontend declara con
# `toggleArrayWithExclusiveSentinel` y evita que un texto libre stale
# (escrito ANTES de marcar "Ninguna") contamine el array canónico downstream.
# Femenino/masculino porque QAllergies/QMedical usan "Ninguna" y
# QStruggles/QDislikes usan "Ninguno".
#
# [P1-FORM-2] CONTRATO CON FRONTEND: este frozenset DEBE estar alineado con
# `SENTINEL_VALUES` exportado por `frontend/src/config/sentinels.js`
# (después de aplicar `.lower()`). El SSOT vive en el frontend porque los
# strings se renderizan al usuario (chip labels); el backend solo los
# reconoce post-payload. Si en el futuro se añade un nuevo sentinel en el
# frontend (ej. "Sin alergia"), DEBE añadirse acá en lowercase
# ("sin alergia") o reaparece la contradicción de P0-FORM-1: el array
# llega `["Sin alergia", "Maní"]` al LLM como ambos verdaderos.
# El "none" extra cubre clientes legacy / no oficiales que envían el inglés.
_SENTINEL_NONE_VALUES = frozenset({"ninguna", "ninguno", "none"})


def _merge_other_text_fields(form_data: dict) -> int:
    """P1-1: mergea cada `other*` (string CSV) en su array canónico, in-place.

    Mutación in-place sobre `form_data` (que `arun_plan_pipeline` ya deep-copió
    del payload del cliente, así que el dict del caller no se ve afectado).
    Idempotente: si una llamada previa ya mergeó los `other*`, esta segunda
    invocación no duplica entradas (dedup case-insensitive).

    No clearamos el campo `other*` original tras mergear: preserva la shape
    de `form_data` para sanitize y para cualquier consumidor legacy que aún
    lo lea como referencia textual cruda. La duplicación es benigna porque
    todos los nodos críticos leen el array canónico.

    P0-FORM-1: si el array canónico contiene un sentinel "Ninguna"/"Ninguno",
    descarta el `other*` asociado y limpia el campo (`form_data[other_field] = ""`)
    para que sanitize y consumidores legacy no vean la contradicción. Antes,
    `allergies=["Ninguna"]` + `otherAllergies="Maní"` (escrito antes de marcar
    "Ninguna" en el frontend) producía `allergies=["Ninguna","Maní"]` —
    contradicción de seguridad médica que el LLM podía resolver ignorando la
    alergia real.

    Returns:
        Número de entradas nuevas añadidas (telemetría útil para validar
        que el merge ocurrió cuando el usuario rellena texto libre). Cero
        en el caso normal (sin texto libre).
    """
    if not isinstance(form_data, dict):
        return 0
    added_total = 0
    logger = logging.getLogger(__name__)
    for canonical, other_field in _OTHER_TEXT_FIELD_MAP:
        other_text = form_data.get(other_field)
        if not isinstance(other_text, str) or not other_text.strip():
            continue

        # Defensive: el array canónico podría venir None o como CSV legacy.
        existing_raw = form_data.get(canonical)
        if existing_raw is None:
            existing = []
        elif isinstance(existing_raw, list):
            existing = list(existing_raw)
        else:
            # Cliente legacy mandó CSV en el array → normalizar a list[str].
            existing = [s.strip() for s in str(existing_raw).split(",") if s.strip()]

        # P0-FORM-1: si el sentinel "Ninguna"/"Ninguno" ya está en el array,
        # el usuario declaró exclusividad. Descarta el `other*` y limpia el
        # campo para que la shape del form quede consistente con la intención.
        has_sentinel = any(
            isinstance(item, str) and item.strip().lower() in _SENTINEL_NONE_VALUES
            for item in existing
        )
        if has_sentinel:
            logger.warning(
                f"[P0-FORM-1] descartando `{other_field}`={other_text!r}: "
                f"`{canonical}` contiene sentinel exclusivo "
                f"({[i for i in existing if isinstance(i, str) and i.strip().lower() in _SENTINEL_NONE_VALUES]}). "
                f"Frontend debió limpiar el texto libre al marcar el sentinel."
            )
            form_data[other_field] = ""
            continue

        # Set para dedup case-insensitive. Idempotente cross-call y dentro del
        # mismo CSV ("Maní, mani" → solo agrega "Maní").
        existing_lower = {str(item).strip().lower() for item in existing if item}
        for raw in other_text.split(","):
            cleaned = raw.strip()
            if not cleaned:
                continue
            cleaned_lower = cleaned.lower()
            if cleaned_lower in existing_lower:
                continue
            existing.append(cleaned)
            existing_lower.add(cleaned_lower)
            added_total += 1
        form_data[canonical] = existing
    return added_total


def _sanitize_text_normalize(text: str) -> str:
    """P1-Q8: normaliza un string para detección anti-injection.

    Pipeline:
      1. Strip caracteres invisibles (ZWSP/ZWJ/ZWNJ/BOM/bidi controls) que
         un atacante puede usar para insertar separadores invisibles entre
         letras y evadir el matching ("ig\\u200bnore previous").
      2. NFKD: descompone unicode a forma canónica — colapsa "ígnoré" → "ignore"
         con acentos separables, "fiˌˌ" → "fi" con ligaduras descomponibles.
      3. ASCII encode/decode con errors='ignore': descarta el resto de no-ASCII.
      4. lower: case-insensitive matching.

    Limitaciones conocidas: NO cubre homoglyphs entre scripts (Cyrillic 'а'
    U+0430 sigue distinto de Latin 'a' U+0061 tras NFKD). Para eso harían
    falta tablas de confusables de Unicode TR39 — fuera de scope de este P1.
    """
    cleaned = _INVISIBLE_CHAR_RX.sub("", text)
    return unicodedata.normalize("NFKD", cleaned).encode("ASCII", "ignore").decode("utf-8").lower()


def _detect_injection_in_text(text: str) -> str | None:
    """P1-Q8: devuelve el patrón matched si hay injection, o None.

    Skip strings cortos (`< _SANITIZE_MIN_LEN`) — patrones reales son
    sustantivos y matches sobre cadenas pequeñas son típicamente false
    positives (IDs, enums, valores cortos del schema).
    """
    if not isinstance(text, str) or len(text) < _SANITIZE_MIN_LEN:
        return None
    norm = _sanitize_text_normalize(text)
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in norm:
            return pattern
    return None


def _sanitize_form_data_recursive(data, *, path: str = "", hits: list | None = None) -> list:
    """P1-Q8: recursivamente sanitiza strings en dicts/listas IN-PLACE.

    Mutación in-place — mismo contrato que el sanitize previo (solo afecta
    a `actual_form_data`, ya copiada vía deepcopy en el caller).

    Reemplaza strings que matcheen patrones por "" y acumula `hits` con
    `{path, pattern, snippet}` para logging y persistencia de métrica.
    Devuelve la lista de hits acumulados (también mutada in-place).

    `path` notation: `field`, `field.nested`, `list[i]`, `list[i].sub` —
    útil para localizar exactamente DÓNDE viene la inyección, especialmente
    en estructuras anidadas que el whitelist anterior nunca cubría.
    """
    if hits is None:
        hits = []
    if isinstance(data, dict):
        for key, val in list(data.items()):
            sub_path = f"{path}.{key}" if path else str(key)
            if isinstance(val, str):
                pattern = _detect_injection_in_text(val)
                if pattern:
                    hits.append({"path": sub_path, "pattern": pattern, "snippet": val[:80]})
                    data[key] = ""
            elif isinstance(val, (dict, list)):
                _sanitize_form_data_recursive(val, path=sub_path, hits=hits)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            sub_path = f"{path}[{i}]"
            if isinstance(item, str):
                pattern = _detect_injection_in_text(item)
                if pattern:
                    hits.append({"path": sub_path, "pattern": pattern, "snippet": item[:80]})
                    data[i] = ""
            elif isinstance(item, (dict, list)):
                _sanitize_form_data_recursive(item, path=sub_path, hits=hits)
    return hits


def _sanitize_external_text_block(text: str, path: str, hits: list) -> str:
    """P0-A1: sanitiza un bloque de texto provisto por fuentes externas
    (memory_context, history_context construido desde DB, facts del RAG).

    A diferencia de `_sanitize_form_data_recursive` (in-place sobre dict/list),
    este helper opera sobre un string puro y retorna el valor saneado:
      - Si detecta un patrón de injection → registra hit en `hits` y devuelve "".
        El caller decide si concatena o saltea esa sección. La degradación es
        silenciosa para el pipeline (no aborta) — preferimos perder una sección
        de contexto antes que entregar al LLM una orden envenenada.
      - Si no detecta nada → devuelve el texto sin cambios.

    Antes, `_sanitize_form_data_recursive` solo cubría `actual_form_data`. El
    `memory_context` (parámetro del caller, persistido en `chat_sessions`) y los
    strings que `history_context` agregaba desde DB (recent_meals, consumed,
    facts del RAG) viajaban al prompt sin filtrar. Un atacante que persistió
    texto venenoso en una sesión previa, en un fact del RAG o vía `proactive_agent`
    podía saltarse toda la defensa P1-Q8 sin tocar el form_data del request actual.
    """
    if not isinstance(text, str) or not text:
        return text
    pattern = _detect_injection_in_text(text)
    if pattern:
        hits.append({"path": path, "pattern": pattern, "snippet": text[:80]})
        return ""
    return text


def _enqueue_sanitize_metric(
    *,
    node: str,
    hits: list,
    user_id_for_metric,
    session_for_metric: str,
    skip_persist: bool,
) -> None:
    """P0-A1 / P1-Q8: encola persistencia async de métrica `sanitize_hits` en
    `pipeline_metrics`.

    Extraído como helper module-level para que tanto la sanitización de
    `actual_form_data` (P1-Q8) como la de `memory_context` / `history_context` /
    RAG (P0-A1) compartan el mismo contrato y misma propagación de ContextVars
    (request_id, user_id) al worker del `_METRICS_EXECUTOR`.

    `node` distingue la fuente en Grafana:
      - "sanitize_hits"           → form_data del request (P1-Q8 original).
      - "sanitize_hits_external"  → memory + history + RAG (P0-A1).

    `skip_persist=True` se respeta para guests cuando el probe del startup
    deshabilitó inserts con user_id=NULL (P1-Q10): se loguea pero no se persiste.
    """
    if not hits:
        return
    if skip_persist:
        logger.debug(
            f"[SANITIZE] P0-A1/P1-Q10: skip persist {node} "
            f"(guest_disabled_by_probe). hits={len(hits)}"
        )
        return

    _hits_snapshot = list(hits)  # cerrar ref para closure

    def _persist():
        try:
            execute_sql_write(
                "INSERT INTO pipeline_metrics (user_id, session_id, node, "
                "duration_ms, retries, tokens_estimated, confidence, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id_for_metric,
                    session_for_metric,
                    node,
                    0,
                    0,
                    0,
                    0.0,  # confidence=0.0 → flag binario "hit detected"
                    json.dumps({
                        "hits_count": len(_hits_snapshot),
                        "patterns": sorted({h["pattern"] for h in _hits_snapshot}),
                        "paths": sorted({h["path"] for h in _hits_snapshot}),
                    }),
                ),
            )
        except Exception as _persist_err:
            logger.warning(
                f"[SANITIZE] P0-A1/P1-Q8: error persistiendo métrica {node}: {_persist_err}"
            )

    try:
        # P1-X3: propagar ContextVars (request_id, user_id) al worker.
        _ctx_metric = contextvars.copy_context()
        _METRICS_EXECUTOR.submit(_ctx_metric.run, _persist)
    except Exception as _enq_err:
        logger.warning(f"[SANITIZE] P0-A1/P1-Q8: error encolando métrica {node}: {_enq_err}")


# ============================================================
# P1-A5: Helpers extraídos del monolito `arun_plan_pipeline`.
# ------------------------------------------------------------
# Antes, las funciones siguientes vivían como closures locales dentro de
# `arun_plan_pipeline` capturando `nutrition`, `requested_days`, `initial_state`
# del scope exterior. Eso impedía:
#   - Testearlas unitariamente con fixtures (tenían que invocarse el pipeline
#     completo con LLMs mockeados solo para ejercitar guardrails).
#   - Reutilizarlas desde otros módulos (services.py, cron_tasks.py).
#   - Localizar bugs: cualquier fix dentro del pipeline tocaba 1200 líneas.
#
# Ahora cada helper tiene firma explícita y vive a nivel módulo. El
# comportamiento es idéntico — solo cambió el scope. Riesgos abordados:
#   - Cierres sobre `nutrition`/`requested_days` → ahora kwargs explícitos.
#   - Side-effects sobre `final_state` mutables → preservados, mismas mutaciones.
#   - Logs (`print`) → mantenidos textualmente para no romper grep alerting.
# ============================================================


def _build_fallback_day(nutr: dict, day_number: int) -> dict:
    """P1-A5: extraída del closure local de `arun_plan_pipeline`.

    Construye un día fallback determinista (3 comidas balanceadas).
    """
    target_cal = nutr.get('target_calories', 2000)
    macros_dict = nutr.get('macros', {})
    target_pro = macros_dict.get('protein_g', 150)
    target_car = macros_dict.get('carbs_g', 200)
    target_fat = macros_dict.get('fats_g', 60)

    def create_meal(name, cal_ratio, p_ratio, c_ratio, f_ratio, desc, meal_type):
        return {
            "meal": meal_type,
            "time": "Flexible",
            "name": name,
            "desc": f"Fallback: {desc}",
            "prep_time": "15 min",
            "difficulty": "Fácil",
            "cals": int(target_cal * cal_ratio),
            "protein": int(target_pro * p_ratio),
            "carbs": int(target_car * c_ratio),
            "fats": int(target_fat * f_ratio),
            "macros": ["Plan Matemático"],
            "ingredients": ["1 porción de proteína magra", "1 porción de carbohidratos", "Vegetales al gusto"],
            "recipe": ["Mise en place: Preparar todo", "El Toque de Fuego: Cocinar la proteína", "Montaje: Servir"]
        }

    meal1 = create_meal("Huevos y Avena", 0.3, 0.3, 0.3, 0.3, "Huevos revueltos con avena cocida.", "Desayuno")
    meal2 = create_meal("Pollo y Arroz", 0.4, 0.4, 0.4, 0.4, "Pechuga a la plancha con arroz blanco y vegetales.", "Almuerzo")
    meal3 = create_meal("Pescado y Batata", 0.3, 0.3, 0.3, 0.3, "Filete de pescado al horno con batata asada.", "Cena")

    return {
        "day": day_number,
        "daily_summary": "Plan de Contingencia de Emergencia (Generado Matemáticamente)",
        "total_calories": target_cal,
        "total_protein": target_pro,
        "total_carbs": target_car,
        "total_fats": target_fat,
        "meals": [meal1, meal2, meal3],
    }


def _get_extreme_fallback_plan(nutr: dict, goal: str, num_days: int = 3, day_offset: int = 0) -> dict:
    """P1-A5: extraída del closure local de `arun_plan_pipeline`.

    Fallback matemático determinista para evitar caídas del sistema.

    P0-1: `num_days` ahora se respeta — antes estaba hardcodeado a 3 y entregaba
    plans truncados al frontend cuando el usuario pidió 7. `day_offset` permite
    continuar la numeración cuando se rellenan días faltantes a un plan parcial.
    """
    target_cal = nutr.get('target_calories', 2000)
    macros_dict = nutr.get('macros', {})
    target_pro = macros_dict.get('protein_g', 150)
    target_car = macros_dict.get('carbs_g', 200)
    target_fat = macros_dict.get('fats_g', 60)

    safe_num_days = max(1, int(num_days or 1))
    days = [_build_fallback_day(nutr, day_offset + i + 1) for i in range(safe_num_days)]

    return {
        "main_goal": goal,
        "insights": ["Este es un plan de contingencia generado matemáticamente debido a indisponibilidad temporal de la IA."],
        "calories": target_cal,
        "macros": {
            "protein": f"{target_pro}g",
            "carbs": f"{target_car}g",
            "fats": f"{target_fat}g"
        },
        "days": days,
        # P0-C: shopping lists vacías para mantener contrato con el frontend.
        # El path normal (assemble_plan_node) siempre setea las 4 claves —
        # cuando `_uid` es None o falla `get_shopping_list_delta`, también
        # caen a `[]`. Los planes de contingencia siguen esa misma semántica:
        # arrays vacíos en lugar de claves ausentes (que causarían
        # `undefined.map(...)` o errores de acceso en el cliente).
        "aggregated_shopping_list": [],
        "aggregated_shopping_list_weekly": [],
        "aggregated_shopping_list_biweekly": [],
        "aggregated_shopping_list_monthly": [],
        # P1-9: el flag debe vivir DENTRO del plan, no solo en final_state.
        # Antes, `_is_fallback` se seteaba solo en `final_state["_is_fallback"]`
        # y el frontend (que recibe `final_state["plan_result"]`) no podía
        # distinguir un plan IA de un plan de contingencia. P0-1 añade flags
        # adicionales (_critical_rejection, _review_issues, etc.) cuando aplica.
        "_is_fallback": True,
        "_review_disclaimer": (
            "Este es un plan de contingencia generado matemáticamente debido a "
            "indisponibilidad temporal de la IA. Por favor regenera más tarde."
        ),
    }


def _is_day_valid(day: dict) -> bool:
    """P1-A5: extraída del closure local de `arun_plan_pipeline`.

    Un día es válido si tiene `meals` como lista no vacía.
    """
    if not isinstance(day, dict):
        return False
    meals = day.get("meals")
    return isinstance(meals, list) and len(meals) > 0


def _is_plan_complete(plan: dict, requested_days: int) -> bool:
    """P1-A5: extraída del closure local. `requested_days` ahora explícito.

    ¿El plan tiene la cantidad de días pedida y todos con comidas?
    """
    if not isinstance(plan, dict):
        return False
    days = plan.get("days") or []
    if len(days) < requested_days:
        return False
    return all(_is_day_valid(d) for d in days)


def _repair_partial_plan(plan: dict, *, nutrition: dict, requested_days: int) -> bool:
    """P1-A5: extraída del closure local. `nutrition` y `requested_days`
    ahora kwargs explícitos.

    P0-2: Repara un plan parcial in-place. Retorna True si reparó algo.

    Antes, el bloque except solo inyectaba macros y rellenaba días faltantes,
    pero NO reemplazaba días existentes que quedaron vacíos (ej. cuando el
    ensamblador insertó skeleton-days sin meals). Esto entregaba al frontend
    un plan con `days: [{meals: []}, ...]` indistinguible de un plan válido,
    crashearlo al renderizar.

    Reparaciones:
      1. Inyecta calorías y macros agregados si faltan.
      2. Reemplaza cada día inválido (sin meals, sin estructura) con un
         `_build_fallback_day` matemático.
      3. Rellena días faltantes hasta `requested_days`.
      4. Marca `_is_fallback=True` y disclaimer si reparó cualquier cosa.
    """
    if not isinstance(plan, dict):
        return False

    repaired = False

    # 1) Macros agregados
    if "calories" not in plan or not plan.get("calories"):
        target_cal = nutrition.get('target_calories', 2000)
        macros_dict = nutrition.get('macros', {})
        plan["calories"] = target_cal
        plan["macros"] = {
            "protein": f"{macros_dict.get('protein_g', 150)}g",
            "carbs": f"{macros_dict.get('carbs_g', 200)}g",
            "fats": f"{macros_dict.get('fats_g', 60)}g",
        }
        repaired = True
        print("🛡️ [P0-2] Macros agregados faltantes inyectados.")

    # 2) Días: reemplazar inválidos + rellenar faltantes
    existing = plan.get("days")
    if not isinstance(existing, list):
        existing = []
        repaired = True

    new_days = []
    replaced_count = 0
    for idx, d in enumerate(existing[:requested_days]):
        if _is_day_valid(d):
            # Asegurar numeración consistente sin destruir contenido válido
            if d.get("day") != idx + 1:
                d["day"] = idx + 1
            new_days.append(d)
        else:
            new_days.append(_build_fallback_day(nutrition, idx + 1))
            replaced_count += 1
            repaired = True

    # Rellenar faltantes
    filled_count = 0
    while len(new_days) < requested_days:
        new_days.append(_build_fallback_day(nutrition, len(new_days) + 1))
        filled_count += 1
        repaired = True

    plan["days"] = new_days

    if replaced_count or filled_count:
        print(f"🛡️ [P0-2] Días reparados: {replaced_count} reemplazados (vacíos/inválidos), "
              f"{filled_count} rellenados (faltantes). Total: {len(new_days)}/{requested_days}.")

    # 3) P0-C: Asegurar shopping lists para no romper el contrato con el frontend.
    # `assemble_plan_node` setea estas 4 claves en el path normal (incluso a `[]`
    # cuando falla el cálculo). Si llegamos a `_repair_partial_plan` es porque el
    # plan no completó ese paso o lo perdió, así que cada clave faltante o
    # corrupta se normaliza a `[]` (mismo contrato que cuando _uid=None).
    _SHOPPING_KEYS = (
        "aggregated_shopping_list",
        "aggregated_shopping_list_weekly",
        "aggregated_shopping_list_biweekly",
        "aggregated_shopping_list_monthly",
    )
    shopping_repaired = 0
    for key in _SHOPPING_KEYS:
        if not isinstance(plan.get(key), list):
            plan[key] = []
            shopping_repaired += 1
    if shopping_repaired:
        repaired = True
        print(f"🛡️ [P0-C] Shopping lists normalizadas: {shopping_repaired}/4 claves "
              f"ausentes o no-list reparadas a `[]`.")

    if repaired:
        plan["_is_fallback"] = True
        plan.setdefault(
            "_review_disclaimer",
            "El plan se completó con valores de contingencia matemáticos por "
            "indisponibilidad temporal de la IA. Por favor regenera más tarde."
        )

    return repaired


def _compute_pipeline_holistic_score_and_emit(
    final_state: dict,
    *,
    nutrition: dict,
    actual_form_data: dict,
    initial_state: dict,
    pipeline_duration: float,
) -> None:
    """P1-A5: extraída del bloque "GAP 4" de `arun_plan_pipeline`.

    Calcula el `holistic_score` (retry penalty + review bonus + cal score),
    emite la métrica `pipeline_holistic`, emite la métrica `progress_cb` si
    hubo pérdidas SSE en este pipeline, y dispara persistencia async del
    score al perfil del usuario via `_METRICS_EXECUTOR`.

    No retorna nada — todos los efectos son fire-and-forget (emit_progress
    queda en cola del executor SSE; persist_holistic_score queda en cola del
    metrics executor con drain en SIGTERM cubierto por P1-NEW-3).

    Encerrado en try/except global: cualquier error en el cálculo del score
    se loguea pero no afecta al plan entregado al cliente.

    [P1-1] DEBE invocarse DESPUÉS de `_apply_critical_review_guardrails` y
    `_apply_final_defense_guardrails` para que el score refleje el plan
    REALMENTE entregado al usuario. Si el guardrail reemplazó el plan por un
    fallback (`_is_fallback=True`), clampeamos `holistic_score` a 0.0 — desde
    la perspectiva del usuario la IA falló y recibió un plan de contingencia
    genérico, así que `preflight_optimization_node` en la siguiente request
    debe leerlo como "intento previo desastroso → estrategia conservadora",
    no como "happy path porque cal_score salió 1.0 sobre el fallback".
    """
    try:
        plan = final_state.get("plan_result")
        if plan and isinstance(plan, dict):
            # [P1-1] Detección de fallback entregado. Cubre los tres caminos
            # que estampan `_is_fallback=True` upstream:
            #   - `_apply_critical_review_guardrails` por schema_invalid o
            #     rechazo médico crítico (línea ~8256).
            #   - `_apply_final_defense_guardrails` cuando el plan vino vacío
            #     o incompleto (vía `_get_extreme_fallback_plan` /
            #     `_repair_partial_plan`).
            #   - `arun_plan_pipeline.except` cuando el grafo entero lanzó.
            delivered_was_fallback = bool(plan.get("_is_fallback"))

            # 1. Penalizar retries (cada retry = -25% en este componente)
            retry_penalty = max(0.0, 1.0 - (final_state.get("attempt", 1) - 1) * 0.25)
            # 2. Bonus por aprobación médica
            review_bonus = 1.0 if final_state.get("review_passed") else 0.5
            # 3. Desviación calórica promedio real vs target
            # P1-A9: días con `day_cals == 0` ahora penalizan en lugar de
            # omitirse del promedio. Antes, un día sin meals (insertado por el
            # ensamblador como skeleton vacío, o por un fallback parcial) NO se
            # incluía en `cal_deviations` y el average de los días restantes
            # quedaba artificialmente alto — el `cal_score` ignoraba la peor
            # falla del plan. Ahora cada día sin calorías cuenta como deviación
            # 1.0 (peor caso saturado, mismo techo que un día con desviación
            # ≥100% real). `days_with_zero_cals` se publica en el metadata para
            # alerting independiente del score agregado.
            cal_deviations = []
            days_with_zero_cals = 0
            target_cal = nutrition.get("target_calories") or 0
            for day in plan.get("days", []):
                day_cals = sum(m.get("cals", 0) for m in day.get("meals", []))
                if day_cals <= 0:
                    days_with_zero_cals += 1
                    cal_deviations.append(1.0)  # peor caso saturado
                elif target_cal > 0:
                    cal_deviations.append(abs(day_cals - target_cal) / target_cal)
                else:
                    # Sin target_cal no podemos calcular deviation real;
                    # tampoco es un día vacío. Conservador: deviación 0
                    # (no penalizar al score por dato faltante upstream).
                    cal_deviations.append(0.0)
            avg_deviation = sum(cal_deviations) / len(cal_deviations) if cal_deviations else 0
            cal_score = max(0.0, 1.0 - avg_deviation * 5)  # 20% desviación = score 0

            holistic_score = round(retry_penalty * 0.3 + review_bonus * 0.3 + cal_score * 0.4, 3)

            # [P1-1] Clamp a 0.0 si el plan entregado fue un fallback. Sin esto,
            # un fallback con macros perfectos (cal_score=1.0 por construcción)
            # más review_bonus=0.5 producía holistic_score≈0.55 — lectura
            # falsamente "decente" para `preflight_optimization_node`. La
            # semántica deseada es: "el usuario recibió plan de contingencia,
            # el LLM falló completamente". El bonus de retry/review se preserva
            # SOLO en metadata (`raw_holistic_pre_clamp`) para análisis post-hoc.
            raw_holistic_pre_clamp = holistic_score
            if delivered_was_fallback:
                holistic_score = 0.0

            _emit_progress(initial_state, "metric", {
                "node": "pipeline_holistic",
                "duration_ms": int(pipeline_duration * 1000),
                "retries": final_state.get("attempt", 1) - 1,
                "tokens_estimated": 0,
                "confidence": holistic_score,
                "metadata": {
                    "attempts": final_state.get("attempt", 1),
                    "review_passed": final_state.get("review_passed"),
                    "avg_cal_deviation": round(avg_deviation, 3),
                    # P1-A9: cardinalidad para alerting. Un valor >0 indica
                    # que el plan entregado tenía días sin meals — señal de
                    # que skeletons quedaron sin rellenar o que `_repair_partial_plan`
                    # se saltó. Grafana puede graficar la tendencia
                    # independientemente del `confidence` global.
                    "days_with_zero_cals": days_with_zero_cals,
                    "pipeline_duration_s": pipeline_duration,
                    # [P1-1] Distingue en Grafana "score=0.0 por fallback
                    # forzado" vs "score=0.0 por cal_deviation extremo".
                    # `raw_holistic_pre_clamp` preserva el valor que el score
                    # tendría sin clamp para análisis de cuán cerca/lejos
                    # estuvo el LLM antes del rechazo crítico.
                    "delivered_was_fallback": delivered_was_fallback,
                    "raw_holistic_pre_clamp": raw_holistic_pre_clamp,
                }
            })
            print(
                f"📊 [HOLISTIC SCORE] Pipeline Quality: {holistic_score:.3f} "
                f"(retry={retry_penalty:.2f}, review={review_bonus:.2f}, "
                f"cal={cal_score:.2f}, zero_cal_days={days_with_zero_cals}, "
                f"fallback={delivered_was_fallback})"
            )

            # P1-NEW-4: emitir métrica de pérdidas de eventos SSE para esta
            # request. Solo persiste si hubo señal (>0) — evita ruido en la
            # tabla `pipeline_metrics` para el caso happy path. `confidence`
            # se usa como flag binario: 1.0 = sin pérdidas, 0.0 = hubo drops.
            ppl_cb_stats = _pipeline_cb_stats_var.get()
            if ppl_cb_stats:
                # Snapshot bajo lock para consistencia con increments concurrentes
                with _PROGRESS_CB_STATS_LOCK:
                    snapshot = dict(ppl_cb_stats)
                if any(v > 0 for v in snapshot.values()):
                    total_loss = sum(snapshot.values())
                    _emit_progress(initial_state, "metric", {
                        "node": "progress_cb",
                        "duration_ms": int(pipeline_duration * 1000),
                        "retries": final_state.get("attempt", 1) - 1,
                        "tokens_estimated": 0,
                        # confidence=0.0 si hubo dropped_cap (peor pérdida —
                        # eventos enteros descartados); =0.5 si solo timeouts
                        # o failed (delivery degradado pero sin pérdida total).
                        "confidence": 0.0 if snapshot.get("dropped_cap", 0) > 0 else 0.5,
                        "metadata": {**snapshot, "total_loss": total_loss},
                    })
                    print(
                        f"⚠️ [PROGRESS CB STATS] Pérdidas SSE en este pipeline: "
                        f"dropped_cap={snapshot.get('dropped_cap', 0)}, "
                        f"timed_out={snapshot.get('timed_out', 0)}, "
                        f"failed_async={snapshot.get('failed_async', 0)}, "
                        f"failed_sync={snapshot.get('failed_sync', 0)}"
                    )

            # GAP 3 + P1-12: Persistir el holistic score en la DB
            # Antes solo se guardaba `last_pipeline_score`, sobreescribiendo en cada
            # chunk: 3 chunks consecutivos perdían los 2 primeros valores y preflight
            # no podía distinguir mejora/deterioro/estabilidad. Ahora mantenemos:
            #   - `last_pipeline_score`: lo que ya consume preflight_optimization_node
            #     (no romper API).
            #   - `pipeline_score_history`: lista deslizante de los últimos 10 chunks
            #     con score + attempts + timestamp + review_passed. Útil para
            #     análisis de tendencia (drift), counterfactual y debugging.
            user_id = actual_form_data.get("user_id") or actual_form_data.get("session_id")
            if user_id and user_id != "guest":
                # P1-X5: dispatch fire-and-forget al `_METRICS_EXECUTOR`. Antes
                # este bloque hacía dos awaits secuenciales `_adb(get_user_profile)`
                # + `_adb(update_user_health_profile)` ANTES del `return final_state["plan_result"]`,
                # añadiendo ~100-300ms al p99 perceptible al cliente por meta-learning
                # que no necesita bloquear la respuesta.
                #
                # `_METRICS_EXECUTOR` (sync, proceso-wide) sobrevive al cierre del
                # loop del caller — importante para el sync wrapper (cron, batch)
                # donde `asyncio.run` cancelaría tasks pendientes en el loop.
                # Drain bounded en SIGTERM (P1-NEW-3) garantiza que la métrica se
                # persiste antes del exit hasta `METRICS_SHUTDOWN_DRAIN_S` segundos.
                # ContextVars (P1-X3) propagados via `ctx.run`.
                ctx = contextvars.copy_context()
                _METRICS_EXECUTOR.submit(
                    ctx.run,
                    _persist_holistic_score_sync,
                    user_id,
                    holistic_score,
                    final_state.get("attempt", 1),
                    bool(final_state.get("review_passed")),
                )

    except Exception as e:
        print(f"⚠️ [HOLISTIC SCORE] Error calculando score: {e}")


def _apply_critical_review_guardrails(
    final_state: dict,
    *,
    nutrition: dict,
    actual_form_data: dict,
    requested_days: int,
) -> None:
    """P1-A5: extraída del bloque "P0-1 / P1-8 GUARDRAIL" de `arun_plan_pipeline`.

    Aplica los guardrails médicos críticos sobre `final_state["plan_result"]`
    in-place. Tres caminos:
      - schema_invalid (P1-8) → reemplaza por fallback marcado con
        `_critical_rejection=True` y disclaimer de "estructura inválida".
      - rechazo médico CRÍTICO → idem con disclaimer de "violación de alergia/
        condición médica declarada".
      - rechazo no-crítico → marca el plan con `_review_failed_but_delivered=True`
        para que el frontend muestre banner ámbar.
      - happy path (review_passed=True, sin schema_invalid) → no-op.

    El plan original (no-fallback) se reemplaza con un fallback nuevo;
    `_profile_embedding` del intento original se preserva.

    Mismas mutaciones que el bloque inline previo. No retorna.
    """
    review_passed = final_state.get("review_passed", False)
    rejection_severity = final_state.get("_rejection_severity")
    rejection_reasons = final_state.get("rejection_reasons", []) or []
    plan_result = final_state.get("plan_result")
    already_fallback = isinstance(plan_result, dict) and plan_result.get("_is_fallback")
    schema_invalid = isinstance(plan_result, dict) and plan_result.get("_schema_invalid")

    needs_critical_fallback = isinstance(plan_result, dict) and not already_fallback and (
        schema_invalid
        or (not review_passed and rejection_severity == "critical")
    )

    if needs_critical_fallback:
        cause = (
            "SCHEMA INVÁLIDO (plan no renderizable por el frontend)"
            if schema_invalid else "rechazo médico CRÍTICO"
        )
        print(f"🚨 [P0-1/P1-8 GUARDRAIL] Fallback forzado por: {cause}. "
              f"Generando plan matemático ({requested_days} días). "
              f"review_passed={review_passed}, severity={rejection_severity}, "
              f"razones={rejection_reasons}")
        fallback_plan = _get_extreme_fallback_plan(
            nutrition,
            actual_form_data.get("mainGoal", "Salud General"),
            num_days=requested_days,
        )
        fallback_plan["_is_fallback"] = True
        fallback_plan["_critical_rejection"] = True
        fallback_plan["_review_issues"] = rejection_reasons or [cause]
        fallback_plan["_review_severity"] = "critical"
        if schema_invalid:
            # Preservar el detalle del error de schema para debugging downstream
            fallback_plan["_schema_errors"] = plan_result.get("_schema_errors", "schema inválido")
            fallback_plan["_review_disclaimer"] = (
                "El plan generado por la IA tenía estructura inválida y no pudo ser entregado. "
                "Este es un plan de contingencia matemático. "
                "Por favor regenera más tarde."
            )
        else:
            fallback_plan["_review_disclaimer"] = (
                "El sistema detectó violaciones críticas (alergias o condiciones médicas) "
                "en el plan generado por la IA y lo descartó por seguridad. "
                "Este es un plan de contingencia matemático. "
                "Por favor regenera o revisa tus restricciones declaradas."
            )
        # Preservar embedding y señales de aprendizaje del intento original
        if plan_result.get("_profile_embedding"):
            fallback_plan["_profile_embedding"] = plan_result["_profile_embedding"]
        final_state["plan_result"] = fallback_plan
    elif not review_passed and isinstance(plan_result, dict) and not already_fallback:
        print(f"⚠️ [P0-1 TRANSPARENCY] Plan entregado tras rechazo no-crítico. "
              f"Marcando para visibilidad en cliente. Severidad: {rejection_severity}")
        plan_result["_review_failed_but_delivered"] = True
        plan_result["_review_issues"] = rejection_reasons
        plan_result["_review_severity"] = rejection_severity or "minor"
        plan_result["_review_disclaimer"] = (
            "Este plan no superó completamente la verificación médica automática. "
            "Las observaciones encontradas son no-críticas, pero te recomendamos "
            "regenerarlo o revisarlo con tu nutricionista."
        )


def _apply_final_defense_guardrails(
    final_state: dict,
    *,
    nutrition: dict,
    actual_form_data: dict,
    requested_days: int,
) -> None:
    """P1-A5: extraída del bloque "P0-2 GUARDRAIL FINAL" + active learning +
    cache schema version de `arun_plan_pipeline`.

    Tres responsabilidades cohesivas (todas son post-pipeline, defense-in-depth):
      1. P0-2 GUARDRAIL: si el plan terminó vacío o incompleto pese a graph
         success, generar fallback total o reparar parcial.
      2. Inyectar `_profile_embedding` desde state al plan (para que el caller
         lo persista junto con el plan en `meal_plans`).
      3. MEJORA 2: snapshot de señales de aprendizaje activas para Attribution
         Tracker.
      4. P0-A3: estampar `_cache_schema_version` para versionado de cache.

    Punto único de salida → cubre TODOS los paths del pipeline (LLM exitoso,
    fallback total por excepción, fallback parcial reparado, guardrail crítico
    de schema/médico).
    """
    plan_final = final_state.get("plan_result")
    if not isinstance(plan_final, dict) or not plan_final:
        print(f"🛡️ [P0-2 GUARDRAIL] Plan ausente tras pipeline exitoso. "
              f"Generando fallback completo ({requested_days} días).")
        # P1-9: `_get_extreme_fallback_plan` ya setea `_is_fallback=True` en el plan.
        final_state["plan_result"] = _get_extreme_fallback_plan(
            nutrition,
            actual_form_data.get("mainGoal", "Salud General"),
            num_days=requested_days,
        )
    elif not _is_plan_complete(plan_final, requested_days):
        days_count = len(plan_final.get("days") or [])
        invalid_count = sum(1 for d in (plan_final.get("days") or []) if not _is_day_valid(d))
        print(f"🛡️ [P0-2 GUARDRAIL] Plan terminó incompleto pese a graph success. "
              f"Días: {days_count}/{requested_days}, inválidos: {invalid_count}. Reparando.")
        # P1-9: `_repair_partial_plan` ya setea el flag en plan_final si repara.
        _repair_partial_plan(plan_final, nutrition=nutrition, requested_days=requested_days)

    # Inyectar profile_embedding para que el caller lo guarde en la BD
    if final_state.get("profile_embedding") and final_state.get("plan_result"):
        final_state["plan_result"]["_profile_embedding"] = final_state["profile_embedding"]

    # MEJORA 2: Extraer snapshot COMPLETO de señales activas para el Attribution Tracker.
    # CADA señal que se inyecta al LLM debe estar aquí para que el Attribution Tracker
    # pueda correlacionar su presencia con el Quality Score resultante y tomar decisiones
    # de pruning/counterfactual informadas.
    active_learning_signals = {
        # --- Señales originales (corregida la key de cold_start) ---
        "quality_hint": actual_form_data.get("_quality_hint"),
        "adherence_hint": actual_form_data.get("_adherence_hint"),
        "emotional_state": actual_form_data.get("_emotional_state"),
        "drastic_strategy": actual_form_data.get("_drastic_change_strategy"),
        "cold_start": bool(actual_form_data.get("_cold_start_recommendations")),
        # --- Señales de alto impacto que estaban invisibles al tracker ---
        "has_fatigued_ingredients": bool(actual_form_data.get("fatigued_ingredients")),
        "has_liked_meals": bool(actual_form_data.get("_liked_meals")),
        "has_liked_flavor_profiles": bool(actual_form_data.get("_liked_flavor_profiles")),
        "has_weight_history": bool(actual_form_data.get("weight_history")),
        "has_llm_retrospective": bool(actual_form_data.get("_llm_retrospective")),
        "has_frustrated_meal_types": bool(actual_form_data.get("_frustrated_meal_types")),
        "has_abandoned_reasons": bool(actual_form_data.get("_abandoned_reasons")),
        "adversarial_play_active": bool(actual_form_data.get("_use_adversarial_play")),
        "has_temporal_adherence": bool(actual_form_data.get("day_of_week_adherence")),
    }
    # Solo guardar las que no sean nulas, vacías o False
    active_learning_signals = {k: v for k, v in active_learning_signals.items() if v}
    if active_learning_signals and final_state.get("plan_result") and isinstance(final_state["plan_result"], dict):
        final_state["plan_result"]["_active_learning_signals"] = active_learning_signals

    # P0-A3: estampar la versión del schema del plan en el resultado final.
    # Punto único de salida → cubre TODOS los paths (LLM exitoso, fallback
    # total por excepción, fallback parcial reparado, guardrail de schema
    # invalido, guardrail de rechazo crítico). El plan_data persistido en
    # `meal_plans` lleva el flag, y en futuras requests `_is_cached_plan_schema_compatible`
    # podrá distinguir planes pre-deploy (legacy) de los post-deploy.
    # Bumpear `CACHE_SCHEMA_VERSION` invalida automáticamente todo lo previo.
    if isinstance(final_state.get("plan_result"), dict):
        final_state["plan_result"]["_cache_schema_version"] = CACHE_SCHEMA_VERSION


# ============================================================
# FUNCIÓN PÚBLICA: Ejecutar el pipeline completo
# ============================================================
async def arun_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None, background_tasks=None) -> dict:
    """
    Ejecuta el pipeline completo de generación de planes (Map-Reduce):
    Calculador → Planificador → Generadores Paralelos (×3) → Ensamblador → Revisor Médico → (loop si falla)
    
    Args:
        progress_callback: Función opcional que recibe dicts de progreso para streaming SSE.
    """
    print("\n" + "🔗" * 30)
    print("🔗 [LANGGRAPH] Iniciando Pipeline Multi-Agente")
    print("🔗" * 30)
    
    pipeline_start = time.time()

    # 0. Copia segura de form_data para no mutar origenes (evita guardar vars temporales en DB)
    # P1-10: copy a nivel módulo
    actual_form_data = copy.deepcopy(form_data)

    # 0.0 P0-A2: defense-in-depth contra claves internas inyectadas por el cliente.
    # `routers/plans.py` ya stripea TODAS las `_keys` del payload del cliente
    # (capa estricta) antes de inyectar las legítimas del backend. Acá aplicamos
    # la capa con whitelist como red de seguridad para callers que NO pasaron
    # por el router (cron_tasks, proactive_agent, scripts internos): cualquier
    # `_key` que no esté en `_TRUSTED_INTERNAL_FORM_KEYS` se elimina con log
    # WARNING. Después capeamos `_days_to_generate` para evitar que un caller
    # legacy mande un valor inflado y dispare trabajo desproporcionado.
    _strip_untrusted_internal_keys(
        actual_form_data,
        allow_set=_TRUSTED_INTERNAL_FORM_KEYS,
        log_prefix="ORQUESTADOR",
    )
    _enforce_days_to_generate_cap(actual_form_data, log_prefix="ORQUESTADOR")

    # 0.0.1 P1-A8: validación de enum para hints de meta-learning. Cubre el
    # caso ortogonal a P0-A2: la clave SÍ está en la whitelist (legítima) pero
    # su VALOR está fuera del enum esperado (bug en cron, drift de schema,
    # typo). Valor desconocido → WARNING + clear (set a None) — el orquestador
    # leerá None y skipeará la rama, idéntico a "no hay señal". Preferimos
    # perder la señal con observabilidad que ejecutar una rama no intencionada.
    _validate_form_hint_enums(actual_form_data, log_prefix="ORQUESTADOR")

    # 0.0.2 P1-1 / P1-2: merge de los cuatro campos `other*` (texto libre del
    # wizard) en sus arrays canónicos para que TODAS las capas downstream los
    # respeten: semantic_cache_check (comparación con plan cacheado), filtro
    # de catálogo (`_get_fast_filtered_catalogs` en `ai_helpers.py`), RAG
    # dynamic_query, prompt JSON dump, review_plan_node. Cobertura:
    #   - P1-1 (safety): otherAllergies → allergies, otherConditions → medicalConditions
    #   - P1-2 (UX):     otherDislikes  → dislikes,  otherStruggles  → struggles
    # Antes el merge SOLO ocurría dentro de review_plan_node y SOLO para
    # alergias/condiciones; dislikes/struggles textuales no se mergeaban en
    # ningún sitio. Ahora una sola fuente de verdad antes de la sanitización
    # (que después recorrerá los arrays mergeados como si vinieran del cliente).
    _added_other = _merge_other_text_fields(actual_form_data)
    if _added_other:
        logger.info(
            f"🔗 [ORQUESTADOR] P1-1/P1-2: mergeadas {_added_other} entrada(s) "
            f"de `other*` en arrays canónicos "
            f"(allergies/medicalConditions/dislikes/struggles)."
        )

    # 0.1 Sanitización anti-prompt-injection.
    # P1-Q8: ahora RECURSIVA sobre todo `actual_form_data` (dicts + listas a
    # cualquier profundidad), no solo los 5 campos del whitelist anterior.
    # Cubre items individuales de `dislikes`/`medicalConditions`/`allergies`
    # con texto libre, campos custom futuros del schema, y fields anidados.
    # También resiste evasiones con caracteres invisibles (ZWSP/bidi controls)
    # vía `_sanitize_text_normalize`. Threshold de longitud (`_SANITIZE_MIN_LEN`)
    # evita false positives en IDs/enums cortos.
    sanitize_hits = _sanitize_form_data_recursive(actual_form_data)

    # P1-Q10: pre-resolver user_id/session_id que comparten ambos sanitize paths
    # (form_data + memory/history/RAG). Si es guest y el probe deshabilitó guest
    # metrics, skip persistencia (los logs por hit individual aún se mantienen
    # — no perdemos visibilidad operacional, solo el row agregado).
    _user_id_for_metric = (
        actual_form_data.get("user_id") or actual_form_data.get("session_id")
    )
    if _user_id_for_metric == "guest":
        _user_id_for_metric = None
    _session_for_metric = actual_form_data.get("session_id", "unknown")
    _skip_sanitize_persist = (
        _user_id_for_metric is None and not _is_guest_metrics_enabled()
    )

    if sanitize_hits:
        for _hit in sanitize_hits:
            logger.warning(
                f"🛡️ [SANITIZE] P1-Q8: prompt injection detectado en "
                f"'{_hit['path']}': pattern={_hit['pattern']!r}, "
                f"snippet={_hit['snippet']!r}"
            )
        # P1-Q8: persistir métrica agregada para alerting. Una entrada con
        # `node='sanitize_hits'` por pipeline donde hubo al menos un match.
        # Grafana puede graficar volumen + cardinalidad de patrones para
        # detectar abuse patterns o ataques sostenidos por usuario.
        _enqueue_sanitize_metric(
            node="sanitize_hits",
            hits=sanitize_hits,
            user_id_for_metric=_user_id_for_metric,
            session_for_metric=_session_for_metric,
            skip_persist=_skip_sanitize_persist,
        )

    # P0-A1: lista acumulativa para hits provenientes de fuentes EXTERNAS al
    # request (memory_context del caller, history_context construido desde DB,
    # facts del RAG). Se persiste como `node='sanitize_hits_external'` al final
    # de la construcción de contexto para distinguir en Grafana el origen del
    # ataque (cliente directo vs. dato persistido envenenado).
    external_sanitize_hits: list = []


    # 1. Pre-calcular nutrición
    nutrition = get_nutrition_targets(actual_form_data)
    
    # 2. Preparar contexto del historial (memoria inteligente y platos recientes)
    history_context = ""
    if memory_context:
        # P0-A1: sanitizar memory_context. Lo construye `build_memory_context()`
        # desde `chat_sessions` (texto que el usuario y agentes previos escribieron
        # en chats), así que un mensaje de chat venenoso persistido en sesiones
        # anteriores podía inyectar instrucciones al LLM sin pasar nunca por
        # `_sanitize_form_data_recursive`. Si hay match, degradamos toda la
        # sección a "" — preferible a entregar al LLM una orden envenenada.
        memory_context = _sanitize_external_text_block(
            memory_context, "memory_context", external_sanitize_hits
        )
    if memory_context:
        if len(memory_context) > 10000:
            logger.warning(f"⚠️ [ORQUESTADOR] memory_context excede los 10k caracteres ({len(memory_context)}). Truncando para evitar exceder tokens.")
            memory_context = memory_context[:10000] + "\n...[TRUNCADO POR LIMITE DE TOKENS]..."
        history_context = memory_context + "\n"
        
    # [P1-4] Eliminado: parámetro `previous_ai_error` y su bloque de auto-
    # corrección externa. Cero callers reales pasaban un valor distinto de
    # None (routers/plans.py, cron_tasks.py, proactive_agent.py todos lo
    # omitían), y el grafo ya hace reflection/retry intra-pipeline vía
    # `reflection_node` + `retry_reflection_node` + `should_retry`. El bloque
    # solo era ruido en el contexto si alguien lo cableara externamente —
    # mejor tener una sola fuente de verdad para la corrección automática
    # (la del grafo, que ve estado completo y no solo el último error).

    # P1-4 (consistencia): leer de `actual_form_data` igual que el resto del
    # pipeline. `user_id`/`session_id` no son fields que el merge ni la
    # sanitización toquen, así que el VALOR es idéntico al del param `form_data`,
    # pero esta convención evita que un futuro field nuevo sanitizado (custom
    # text, perfil cifrado, etc.) cause divergencia silenciosa entre el path
    # de RAG y el de DB lookup.
    user_id = actual_form_data.get("user_id") or actual_form_data.get("session_id")
    if user_id == "guest":
        user_id = None

    # Nuevo motor anti-repetición robusto: Query directo a la base de datos
    if user_id:
        try:
            # P0-NEW-1.h: query DB sync. Despachada al `_DB_EXECUTOR` para no
            # congelar el loop al inicio del pipeline (~50-200ms por usuario).
            recent_meals = await _adb(get_recent_meals_from_plans, user_id, 5)
            if recent_meals:
                # P0-A1: sanitizar nombres/descripciones in-place antes de serializar
                # al prompt. Un plan persistido con texto envenenado (cliente legacy,
                # bug histórico de sanitización, dato corrupto) podía dispararse
                # como instrucción al LLM vía este JSON dump. Strings con match
                # quedan vacíos; el JSON resultante mantiene shape para no romper
                # el resto del prompt.
                _sanitize_form_data_recursive(
                    recent_meals,
                    path="history_context.recent_meals",
                    hits=external_sanitize_hits,
                )
                history_context += (
                    "\n\n--- HISTORIAL RECIENTE (PLATOS YA GENERADOS) ---\n"
                    "Estos platos fueron generados recientemente para el usuario:\n"
                    f"{json.dumps(recent_meals, ensure_ascii=False)}\n"
                    "🚨 REGLA DE ORO OBLIGATORIA: Puedes reutilizar los mismos INGREDIENTES de estos platos para optimizar tus ingredientes, PERO ESTÁ ESTRICTAMENTE PROHIBIDO repetir el mismo PLATO O PREPARACIÓN EXACTA.\n"
                    "Por ejemplo: Si dice 'Mangú de Plátano', NO uses Mangú, pero sí puedes usar el plátano para un Mofongo o Plátano Hervido.\n"
                    "Cambia la forma de cocinarlos y combínalos distinto. NO repitas el mismo nombre o concepto de plato en toda la semana (a menos que el usuario lo pida).\n"
                    "----------------------------------------------------------------------"
                )
        except Exception as e:
            print(f"⚠️ Error recuperando comidas recientes desde db: {e}")
            
        try:
            from db_facts import get_consumed_meals_since
            # P1-10: datetime/timezone/timedelta a nivel módulo

            since_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            # P0-NEW-1.h: get_consumed_meals_since (DB sync). Despachado al executor.
            consumed = await _adb(get_consumed_meals_since, user_id, since_date)

            if consumed:
                # P0-A1: sanitizar nombres de platos consumidos antes de listarlos
                # al LLM. `meal_tracking` puede contener texto que el usuario tipeó
                # manualmente al loggear — vector simétrico al que ya se cubrió en
                # `actual_form_data` con P1-Q8.
                _sanitize_form_data_recursive(
                    consumed,
                    path="history_context.consumed",
                    hits=external_sanitize_hits,
                )
                consumed_names = [f"- {m.get('meal_name')} ({m.get('calories')} kcals)" for m in consumed]
                history_context += (
                    "\n\n--- FEEDBACK EVOLUTIVO (ÚLTIMOS 7 DÍAS) ---\n"
                    "El usuario hizo tracking de los siguientes platos en la ventana JIT anterior:\n"
                    f"{chr(10).join(consumed_names)}\n"
                    "INSTRUCCIÓN: Observa qué tipos de platos tuvieron éxito. "
                    "Incentiva opciones similares pero NO repitas las mismas recetas exactas.\n"
                    "----------------------------------------------------------------------\n"
                )
        except Exception as e:
            print(f"⚠️ Error recuperando feedback evolutivo de 7 días: {e}")
            
    # 2.14 --- MEJORA 1: FEEDBACK DE CALIDAD CON CONSECUENCIAS ---
    quality_hint = actual_form_data.get("_quality_hint")
    drastic_strategy = actual_form_data.get("_drastic_change_strategy", None)
    if quality_hint == "drastic_change":
        history_context += "\n\n🚨 ALERTA DE CALIDAD: CAMBIO RADICAL NECESARIO 🚨\nLos últimos planes generados han tenido una adherencia y satisfacción extremadamente baja.\n"
        if drastic_strategy == "ethnic_rotation":
            history_context += "INSTRUCCIÓN OBLIGATORIA (ESTRATEGIA ÉTNICA): Cambia radicalmente el perfil étnico de los sabores.\n"
        elif drastic_strategy == "texture_swap":
            history_context += "INSTRUCCIÓN OBLIGATORIA (ESTRATEGIA DE TEXTURAS): Cambia radicalmente las texturas de las comidas (de guisos a crujientes/horneados).\n"
        elif drastic_strategy == "protein_shock":
            history_context += "INSTRUCCIÓN OBLIGATORIA (ESTRATEGIA DE PROTEÍNAS): Usa fuentes de proteína completamente diferentes a las usuales.\n"
        else:
            history_context += "INSTRUCCIÓN OBLIGATORIA: Cambia radicalmente la estrategia. Simplifica las preparaciones, utiliza 'comfort food' segura y conocida, y evita recetas complejas o ingredientes que no sean básicos.\n"
        history_context += "Abandona la estrategia de los días anteriores por completo.\n----------------------------------------------------------------------\n"
    elif quality_hint == "increase_complexity":
        history_context += (
            "\n\n🌟 EXCELENTE ADHERENCIA: LUZ VERDE PARA MAYOR COMPLEJIDAD 🌟\n"
            "El usuario ha mostrado una adherencia y satisfacción excelentes durante los últimos ciclos.\n"
            "INSTRUCCIÓN: Tienes permiso para aumentar gradualmente la complejidad culinaria y la variedad. Puedes introducir nuevos ingredientes, fusiones o técnicas de preparación un poco más elaboradas para mantener el interés y evitar la monotonía.\n"
            "----------------------------------------------------------------------\n"
        )
    elif quality_hint == "break_plateau":
        history_context += (
            "\n\n⚠️ ALERTA: PLATEAU DE ADHERENCIA SILENCIOSO DETECTADO ⚠️\n"
            "El usuario ha mantenido un nivel de calidad y adherencia mediocre pero constante durante varios ciclos. Está cayendo en monotonía sin quejarse explícitamente.\n"
            "INSTRUCCIÓN OBLIGATORIA: Rompe el patrón actual. Introduce al menos 2 ingredientes o proteínas que el usuario no haya consumido recientemente y cambia las técnicas de cocción principales. Sorprende al usuario para reactivar su interés.\n"
            "----------------------------------------------------------------------\n"
        )

    # 2.14.2 --- GAP 1 FIX: LAS 6 SEÑALES DE CRON QUE NO LLEGABAN AL LLM ---
    # _inject_advanced_learning_signals() las calcula y las guarda en pipeline_data/form_data,
    # pero sin este bloque el compresor de contexto nunca las ve y el LLM las ignoraba.

    # Señal 1: Adherencia EMA granular por tipo de comida
    _meal_level_adherence = actual_form_data.get("_meal_level_adherence", {})
    if _meal_level_adherence:
        skipped_meals = [m for m, r in _meal_level_adherence.items() if isinstance(r, (int, float)) and r < 0.4]
        low_meals_fmt = [f"{m.capitalize()} ({r*100:.0f}%)" for m, r in _meal_level_adherence.items() if isinstance(r, (int, float)) and r < 0.7]
        if skipped_meals or low_meals_fmt:
            history_context += (
                "\n\n--- 🎯 [CRON] ADHERENCIA EMA GRANULAR POR COMIDA (DATOS REALES) ---\n"
            )
            if low_meals_fmt:
                history_context += f"Comidas con baja adherencia histórica: {', '.join(low_meals_fmt)}\n"
            if skipped_meals:
                skipped_str = ", ".join([m.capitalize() for m in skipped_meals])
                history_context += (
                    f"COMIDAS SISTEMÁTICAMENTE SALTADAS (<40%): {skipped_str}.\n"
                    "INSTRUCCIÓN OBLIGATORIA: Para esas comidas, diseña opciones ULTRA-SIMPLES (frutas, yogurt, batidos, grab-and-go). "
                    "Si no se pueden simplificar, redistribuye sus macros al resto del día.\n"
                )
            history_context += "----------------------------------------------------------------------\n"
            print(f"✅ [GAP 1] Señal 1: Adherencia EMA granular inyectada → {_meal_level_adherence}")

    # Señal 1b: EMA largo plazo — tendencia estacional (~30 días)
    _meal_level_adherence_long = actual_form_data.get("_meal_level_adherence_long", {})
    if _meal_level_adherence_long:
        low_long = {m: r for m, r in _meal_level_adherence_long.items() if isinstance(r, (int, float)) and r < 0.6}
        high_long = {m: r for m, r in _meal_level_adherence_long.items() if isinstance(r, (int, float)) and r >= 0.75}
        if low_long or high_long:
            long_block = "\n\n--- 📈 [CRON] EMA LARGO PLAZO — TENDENCIA ESTACIONAL (30 DÍAS) ---\n"
            if low_long:
                low_fmt = ", ".join([f"{m.capitalize()} ({r*100:.0f}%)" for m, r in low_long.items()])
                long_block += (
                    f"Comidas con adherencia históricamente baja (tendencia 30 días): {low_fmt}.\n"
                    "INSTRUCCIÓN ESTRATÉGICA: Estas comidas son problemáticas a largo plazo, no solo este ciclo. "
                    "Rediseña su estructura base: simplifica, rota ingredientes clave, o elimina la comida del plan si es ignorada crónicamente.\n"
                )
            if high_long:
                high_fmt = ", ".join([f"{m.capitalize()} ({r*100:.0f}%)" for m, r in high_long.items()])
                long_block += (
                    f"Comidas con adherencia históricamente alta (anclas sólidas): {high_fmt}.\n"
                    "INSTRUCCIÓN: Mantén estas comidas como pilares del plan. Introduce variaciones de ellas, no sustitutos radicales.\n"
                )
            long_block += "----------------------------------------------------------------------\n"
            history_context += long_block
            print(f"✅ [GAP 1] Señal 1b: EMA largo plazo inyectado → low={list(low_long)}, high={list(high_long)}")

    # Señal 2: Razones causales de abandono de comidas
    _abandoned_reasons = actual_form_data.get("_abandoned_reasons", {})
    if _abandoned_reasons:
        reason_map = {
            'no_time': 'Falta de tiempo → Recetas <10 minutos, batch-cook o grab-and-go.',
            'no_ingredients': 'Faltaron ingredientes → Usar SOLO ingredientes de despensa disponible.',
            'not_hungry': 'Falta de hambre → Reducir volumen, alta densidad calórica (nueces, aceite, batidos).',
            'didnt_like': 'No le gustó → Apegarse estrictamente a sus likes conocidos.',
            'ate_out': 'Comió fuera → Sugerir alternativa tipo "fake-away" irresistible.',
        }
        reason_lines = [f"  - {mt.capitalize()}: {reason_map.get(r, r)}" for mt, r in _abandoned_reasons.items()]
        history_context += (
            "\n\n--- 🧠 [CRON] DIAGNÓSTICO CAUSAL DE ABANDONO (OBLIGATORIO) ---\n"
            "El usuario explicó POR QUÉ abandonó estas comidas en el ciclo anterior:\n"
            + "\n".join(reason_lines)
            + "\nINSTRUCCIÓN: Aplica cada ajuste específico indicado para esa comida. No ignorar.\n"
            "----------------------------------------------------------------------\n"
        )
        print(f"✅ [GAP 1] Señal 2: Razones de abandono inyectadas → {_abandoned_reasons}")

    # Señal 3: Estado emocional detectado por nudge_outcomes
    _emotional_state = actual_form_data.get("_emotional_state")
    if _emotional_state == 'needs_comfort':
        history_context += (
            "\n\n--- ❤️ [CRON] ESTADO EMOCIONAL: USUARIO NECESITA CONFORT ---\n"
            "El usuario ha manifestado frustración, culpa o agobio recientemente (detectado vía nudge_outcomes).\n"
            "INSTRUCCIÓN OBLIGATORIA: Diseña 'comfort food' saludable. Platos calientes, cremosos, reconfortantes. "
            "La comida debe sentirse como un abrazo, no como una dieta estricta. Evita ensaladas frías o comidas aburridas.\n"
            "----------------------------------------------------------------------\n"
        )
        print("✅ [GAP 1] Señal 3: Estado emocional 'needs_comfort' inyectado")
    elif _emotional_state == 'ready_for_challenge':
        history_context += (
            "\n\n--- 🔥 [CRON] ESTADO EMOCIONAL: USUARIO MOTIVADO Y LISTO PARA RETOS ---\n"
            "El usuario está altamente motivado y positivo (detectado vía nudge_outcomes).\n"
            "INSTRUCCIÓN: Introduce recetas más desafiantes, nuevos perfiles de sabor y técnicas culinarias avanzadas. "
            "Diseña comidas enfocadas en máximo rendimiento y variedad para mantener el impulso.\n"
            "----------------------------------------------------------------------\n"
        )
        print("✅ [GAP 1] Señal 3: Estado emocional 'ready_for_challenge' inyectado")

    # Señal 4: Tipos de comida cuyos nudges son sistemáticamente ignorados
    _ignored_meal_types = actual_form_data.get("_ignored_meal_types", [])
    if _ignored_meal_types:
        ignored_str = ", ".join([m.capitalize() for m in _ignored_meal_types])
        history_context += (
            f"\n\n--- 🔔 [CRON] RECORDATORIOS IGNORADOS SISTEMÁTICAMENTE ---\n"
            f"El usuario NO responde a los recordatorios de estas comidas: {ignored_str}.\n"
            "INSTRUCCIÓN OBLIGATORIA: Diseña esas comidas como 'grab-and-go' instantáneo (preparación <5 min). "
            "Si no es posible hacerlas instantáneas, elimínalas del plan y redistribuye sus macros.\n"
            "----------------------------------------------------------------------\n"
        )
        print(f"✅ [GAP 1] Señal 4: Tipos de comida ignorados inyectados → {_ignored_meal_types}")

    # Señal 5: Patrón de adherencia por día de la semana
    _day_of_week_adherence = actual_form_data.get("day_of_week_adherence", {})
    if _day_of_week_adherence:
        low_days = [d for d, r in _day_of_week_adherence.items() if isinstance(r, (int, float)) and r <= 0.6]
        if low_days:
            low_days_str = ", ".join(low_days)
            history_context += (
                f"\n\n--- 📆 [CRON] PERFIL CONDUCTUAL POR DÍA DE LA SEMANA ---\n"
                f"El usuario tiende a fallar o abandonar la dieta los: {low_days_str}.\n"
                "INSTRUCCIÓN CRÍTICA: Para esos días específicos, diseña EXCLUSIVAMENTE 'comfort food' saludable "
                "(hamburguesas fit, wraps rápidos, bowl express). Tiempo de preparación <10 minutos, sin recetas complejas.\n"
                "----------------------------------------------------------------------\n"
            )
            print(f"✅ [GAP 1] Señal 5: Días de baja adherencia inyectados → {low_days}")

    # Señal 6: Técnicas de cocción exitosas y abandonadas (basado en consumo real)
    _succ_techs = actual_form_data.get("successful_techniques", [])
    _aban_techs = actual_form_data.get("abandoned_techniques", [])
    if _succ_techs or _aban_techs:
        tech_block = "\n\n--- 🎯 [CRON] PATRONES DE ÉXITO Y FRACASO (DATOS REALES DE CONSUMO) ---\n"
        if _succ_techs:
            tech_block += f"✅ TÉCNICAS QUE SÍ FUNCIONAN (el usuario las comió): {', '.join(set(_succ_techs))}\n"
            tech_block += "   → Fomenta activamente estas técnicas. El usuario tiene alta probabilidad de adherirse.\n"
        if _aban_techs:
            tech_block += f"❌ TÉCNICAS ABANDONADAS (el usuario las ignoró): {', '.join(set(_aban_techs))}\n"
            tech_block += "   → EVITA estas técnicas en este ciclo. Causaron fricción y reducen la adherencia.\n"
        tech_block += "----------------------------------------------------------------------\n"
        history_context += tech_block
        print(f"✅ [GAP 1] Señal 6: Técnicas éxito/abandono inyectadas → succ={_succ_techs}, aban={_aban_techs}")

    # Señal 7: Snapshot de calidad global — adherencia escalar + score previo + tendencia
    # (_adherence_hint ya se usa en complexity_guard pero no se narraba aqui;
    #  _previous_plan_quality + quality_history solo iban al reflection_node lateral.)
    _adherence_hint = actual_form_data.get("_adherence_hint")
    _prev_quality = actual_form_data.get("_previous_plan_quality")
    _quality_hist = actual_form_data.get("quality_history", [])
    if _adherence_hint or isinstance(_prev_quality, (int, float)) or (isinstance(_quality_hist, list) and len(_quality_hist) >= 2):
        q_block = "\n\n--- 📊 [CRON] SNAPSHOT DE CALIDAD Y ADHERENCIA GLOBAL ---\n"
        if isinstance(_prev_quality, (int, float)):
            q_block += f"Quality Score del ciclo anterior: {_prev_quality:.2f} / 1.00\n"
        if isinstance(_quality_hist, list) and len(_quality_hist) >= 2:
            hist_fmt = " → ".join([f"{q:.2f}" for q in _quality_hist[-5:]])
            try:
                trend = "subiendo" if _quality_hist[-1] > _quality_hist[0] else ("bajando" if _quality_hist[-1] < _quality_hist[0] else "plana")
            except Exception:
                trend = "sin definir"
            q_block += f"Historial (últimos {len(_quality_hist[-5:])} ciclos): {hist_fmt} — tendencia {trend}\n"
        if _adherence_hint == "low":
            q_block += (
                "ADHERENCIA RECIENTE: BAJA (<30% comidas loggeadas). "
                "INSTRUCCIÓN: Simplifica radicalmente. Máximo 3 pasos por receta, ingredientes básicos, sin horno.\n"
            )
        elif _adherence_hint == "high":
            q_block += (
                "ADHERENCIA RECIENTE: ALTA (>80% comidas loggeadas). "
                "INSTRUCCIÓN: Luz verde para introducir variedad y técnicas nuevas sin miedo al rechazo.\n"
            )
        # Hint contextual dual-EMA: diagnóstico más fino combinando corto y largo plazo
        _adherence_ema_hint = actual_form_data.get("_adherence_ema_hint")
        _ema_hint_instructions = {
            "temporary_dip": (
                "DIAGNÓSTICO DUAL-EMA: CAÍDA TEMPORAL — el usuario bajó su adherencia recientemente "
                "pero tiene buen historial a largo plazo.\n"
                "INSTRUCCIÓN: NO entres en pánico ni simplifiques agresivamente. "
                "Ofrece 'comfort food' saludable: platos calientes, reconfortantes y familiares. "
                "Es un bache puntual; mantén la estructura del plan y refuerza con opciones irresistibles.\n"
            ),
            "drastic_change": (
                "DIAGNÓSTICO DUAL-EMA: ABANDONO CRÓNICO — adherencia baja tanto a corto como a largo plazo.\n"
                "INSTRUCCIÓN: Intervención drástica. Rediseña el 100% del plan con platos completamente distintos. "
                "Prioriza opciones grab-and-go, sin cocción, o de un solo paso. "
                "El plan actual no está funcionando; cambia la estrategia de fondo.\n"
            ),
            "improving": (
                "DIAGNÓSTICO DUAL-EMA: RECUPERACIÓN ACTIVA — el usuario mejoró recientemente tras un historial bajo.\n"
                "INSTRUCCIÓN: Refuerza el progreso. Mantén la simplicidad que está funcionando pero "
                "introduce 1-2 novedades atractivas para capitalizar la motivación positiva. "
                "No retrocedas a recetas que causaron el abandono previo.\n"
            ),
        }
        if _adherence_ema_hint in _ema_hint_instructions:
            q_block += _ema_hint_instructions[_adherence_ema_hint]
        q_block += "----------------------------------------------------------------------\n"
        history_context += q_block
        print(f"✅ [GAP 1] Señal 7: Snapshot calidad/adherencia inyectado (hint={_adherence_hint}, ema_hint={_adherence_ema_hint}, prev={_prev_quality})")

    # Señal 8: Platos recurrentes de alta adherencia (variaciones bienvenidas)
    _frequent_meals = actual_form_data.get("frequent_meals", [])
    if isinstance(_frequent_meals, list) and _frequent_meals:
        freq_fmt = ", ".join(_frequent_meals[:5])
        history_context += (
            "\n\n--- 🔁 [CRON] PLATOS RECURRENTES DE ALTA ADHERENCIA ---\n"
            f"El usuario consume estos platos repetidamente con éxito: {freq_fmt}.\n"
            "INSTRUCCIÓN: Incluye al menos 1 VARIACIÓN (no copia literal) de estos platos en la semana. "
            "Son anclas de adherencia probadas; respetar su estructura base reduce el riesgo de abandono.\n"
            "----------------------------------------------------------------------\n"
        )
        print(f"✅ [GAP 1] Señal 8: Platos recurrentes inyectados → {_frequent_meals[:5]}")

    # Señal 9: Tipos de comida que generan frustración emocional (más fuerte que "ignorados")
    _frustrated = actual_form_data.get("_frustrated_meal_types", [])
    if isinstance(_frustrated, list) and _frustrated:
        fru_fmt = ", ".join([m.capitalize() for m in _frustrated])
        history_context += (
            "\n\n--- 😤 [CRON] COMIDAS QUE GENERAN FRUSTRACIÓN ---\n"
            f"El usuario expresó frustración/agobio asociados a estas comidas: {fru_fmt}.\n"
            "INSTRUCCIÓN OBLIGATORIA: Rediseña estas comidas con placer explícito (sabor, textura cremosa, calor). "
            "Evita presentaciones 'fit/dieta tradicional'. Si no logras reencantarlas, elimínalas del plan.\n"
            "----------------------------------------------------------------------\n"
        )
        print(f"✅ [GAP 1] Señal 9: Comidas frustrantes inyectadas → {_frustrated}")



    # 2.14.1 --- MEJORA 5 y GAP 5: FATIGA DE INGREDIENTES Y CATEGORÍAS ---
    fatigued_ingredients = actual_form_data.get("fatigued_ingredients", [])
    if fatigued_ingredients:
        fatigued_str = ", ".join(fatigued_ingredients).upper()
        history_context += (
            "\n\n⚠️ ALERTA CRÍTICA: FATIGA DE INGREDIENTES Y MACRONUTRIENTES ⚠️\n"
            f"El sistema ha detectado monotonía severa. El usuario ha consumido en exceso lo siguiente en los últimos 14 días: {fatigued_str}.\n"
            "INSTRUCCIÓN OBLIGATORIA: REDUCE DRÁSTICAMENTE o elimina por completo el uso de estos ingredientes o categorías nutricionales en este plan.\n"
            "Si ves una '[CATEGORÍA]' (ej. '[CATEGORÍA] AVES'), significa que el usuario está fatigado de TODA esa familia de alimentos (pollo, pavo, etc.). "
            "Debes rotar hacia una categoría completamente distinta (ej. pescados, carnes rojas, o fuentes vegetales).\n"
            "Prioriza la variación inmediata sustituyéndolos por otras proteínas/carbohidratos.\n"
            "----------------------------------------------------------------------\n"
        )


    # 2.15 --- REGENERACIÓN DEL MISMO DÍA (RECHAZO EXPLÍCITO) ---
    previous_meals = actual_form_data.get("previous_meals", [])
    if previous_meals and user_id:
        try:
            
            # Check if this is a Pantry Rotation vs a Full Rejected Plan Regeneration
            is_rotation = bool(actual_form_data.get("current_pantry_ingredients") or actual_form_data.get("current_shopping_list"))

            # Si el plan anterior se generó en el mismo día, interpretamos como RECHAZO o ROTACIÓN
            # P0-NEW-1.h: DB sync. Despachado al executor.
            if await _adb(check_meal_plan_generated_today, user_id):
                if is_rotation:
                    print("🔄 [ROTACIÓN DE PLATOS] Generando nuevas recetas ESTRICTAMENTE con la misma despensa.")
                    
                    current_pantry = actual_form_data.get("current_pantry_ingredients") or actual_form_data.get("current_shopping_list", [])
                    pantry_list_str = ", ".join(current_pantry) if current_pantry else "Ninguno detectado."
                    
                    history_context += (
                        f"\n\n🚨 INSTRUCCIÓN DE ROTACIÓN DE MENÚ 🚨\n"
                        f"El usuario solicitó 'Rotar Platos'. EVITA estas preparaciones anteriores:\n{', '.join(previous_meals)}\n"
                        f"DEBES inventar nuevas recetas pero OBLIGATORIAMENTE usando SOLO los ingredientes permitidos en la despensa base actual: {pantry_list_str}.\n"
                        f"ESTÁ ESTRICTAMENTE PROHIBIDO INVENTAR INGREDIENTES QUE NO ESTÉN EN ESTA LISTA EXACTA.\n"
                        f"----------------------------------------------------------------------\n"
                    )
                    actual_form_data["_is_rotation_reroll"] = True
                else:
                    print("🔄 [REGENERACIÓN] Usuario solicitó 'Generar Nueva Opción' el mismo día = RECHAZO del menú actual.")
                    
                    history_context += (
                        f"\n\n🚨 INSTRUCCIÓN DE VARIEDAD (RE-ROLL) 🚨\n"
                        f"El usuario quiere cambiar completamente las opciones de hoy:\n{', '.join(previous_meals)}\n"
                        f"REGLA CREATIVA: Inventa preparaciones inéditas. Cambia el método de cocción, la combinación o el corte para sorprender al usuario con algo nuevo.\n"
                        f"----------------------------------------------------------------------\n"
                    )
                    actual_form_data["_is_same_day_reroll"] = True
            else:
                print("🌅 [NUEVO DÍA] Generación para un nuevo día iniciada.")
        except Exception as e:
            print(f"⚠️ Error validando regeneración del mismo día: {e}")


    # 2.5 Buscar Hechos y Diario Visual en Memoria Vectorial (RAG multimodal)
    # P0-ORQ-1: inicializar `query_emb` ANTES del `if user_id:` para que
    # guests, RAG fallidos y cualquier path que no entre al bloque dejen una
    # variable definida (None) en lugar de depender de `'query_emb' in locals()`
    # más abajo. El check de `locals()` enmascaraba degradaciones silenciosas
    # del cache semántico (profile_embedding=None sin alerta).
    user_facts_text = ""
    visual_facts_text = ""
    facts_data_sorted = []
    visual_list = []
    query_emb = None
    if user_id:
        try:

            # 1. Recuperación estricta (Metadata JSONB) - ALERGIAS, CONDICIONES, RECHAZOS
            # P1-3: Las 3 lookups son independientes y cada una hace I/O contra
            # Supabase (~50-200ms). Antes corrían secuenciales bloqueando el
            # event loop (~150-600ms total). Ahora se paralelizan en thread pool
            # → latencia ≈ max(latencias) y no se bloquea el loop.
            strict_facts_text = ""
            alergias, rechazos, condiciones = await asyncio.gather(
                asyncio.to_thread(get_user_facts_by_metadata, user_id, 'category', 'alergia'),
                asyncio.to_thread(get_user_facts_by_metadata, user_id, 'category', 'rechazo'),
                asyncio.to_thread(get_user_facts_by_metadata, user_id, 'category', 'condicion_medica'),
            )

            # P0-A1: sanitizar facts del RAG estricto antes de inyectarlos al
            # prompt. `user_facts` se puebla desde `proactive_agent` (extracción
            # automática de chats), uploads de usuario y migraciones legacy;
            # cualquier vector envenenado podía colarse como "ALERGIA ESTRICTA"
            # o "CONDICIÓN MÉDICA" — secciones que el revisor médico respeta.
            # Filtramos los facts cuyo texto quedó vacío tras sanitización: la
            # cadena "🔴 ALERGIAS ESTRICTAS:\n  - " sin contenido confunde al LLM.
            def _sanitize_fact_list(items, path_prefix):
                if not items:
                    return items
                cleaned = []
                for idx, _f in enumerate(items):
                    fact_text = _f.get("fact") if isinstance(_f, dict) else None
                    if not isinstance(fact_text, str):
                        cleaned.append(_f)
                        continue
                    sanitized = _sanitize_external_text_block(
                        fact_text, f"{path_prefix}[{idx}].fact", external_sanitize_hits
                    )
                    if sanitized:
                        _f["fact"] = sanitized
                        cleaned.append(_f)
                    # Si quedó vacío → drop completo del fact
                return cleaned

            alergias = _sanitize_fact_list(alergias, "history_context.rag.alergias")
            rechazos = _sanitize_fact_list(rechazos, "history_context.rag.rechazos")
            condiciones = _sanitize_fact_list(condiciones, "history_context.rag.condiciones")

            if alergias:
                strict_facts_text += "🔴 ALERGIAS ESTRICTAS (PROHIBIDO USAR):\n" + "\n".join([f"  - {a['fact']}" for a in alergias]) + "\n"
            if rechazos:
                strict_facts_text += "🔴 RECHAZOS (NO USAR):\n" + "\n".join([f"  - {r['fact']}" for r in rechazos]) + "\n"
            if condiciones:
                strict_facts_text += "⚠️ CONDICIONES MÉDICAS (ADAPTAR PLAN):\n" + "\n".join([f"  - {c['fact']}" for c in condiciones]) + "\n"

            if strict_facts_text:
                user_facts_text += "=== REGLAS MÉDICAS Y DE GUSTO ABSOLUTAS (Extraídas de Base de Datos Estructurada) ===\n"
                user_facts_text += strict_facts_text + "=================================================================================\n\n"

            # 2. Buscar hechos textuales — QUERY DINÁMICA (Vectorial) para contexto general.
            # P1-4: leer de `actual_form_data` (deep-copied, sanitizado y con `other*`
            # mergeados vía `_merge_other_text_fields` línea ~8164), NO del parámetro
            # `form_data` original. Antes este bloque leía el param crudo, así que
            # un usuario con "Maní" en `otherAllergies` o "Apio" en `otherDislikes`
            # generaba una query RAG sin esos términos → no traíamos contexto
            # histórico relevante (preferencias, episodios previos, facts del
            # alérgeno/dislike). El bug era ortogonal a P1-1/P1-2: aunque las otras
            # capas (cache, catalog, review) ya respetaban el merge, RAG seguía
            # ciega a los textos libres.
            dynamic_parts = []
            if actual_form_data.get("mainGoal"):
                dynamic_parts.append(f"Objetivo: {actual_form_data['mainGoal']}")
            if actual_form_data.get("allergies"):
                _ral = actual_form_data["allergies"]
                allergies = _ral if isinstance(_ral, list) else [_ral]
                dynamic_parts.append(f"Alergias: {', '.join(allergies)}")
            if actual_form_data.get("medicalConditions"):
                _rmc = actual_form_data["medicalConditions"]
                conditions = _rmc if isinstance(_rmc, list) else [_rmc]
                dynamic_parts.append(f"Condiciones: {', '.join(conditions)}")
            if actual_form_data.get("dietType"):
                dynamic_parts.append(f"Dieta: {actual_form_data['dietType']}")
            if actual_form_data.get("dislikes"):
                _rdl = actual_form_data["dislikes"]
                dislikes = _rdl if isinstance(_rdl, list) else [_rdl]
                dynamic_parts.append(f"No le gusta: {', '.join(dislikes)}")
            if actual_form_data.get("struggles"):
                _rst = actual_form_data["struggles"]
                struggles = _rst if isinstance(_rst, list) else [_rst]
                dynamic_parts.append(f"Obstáculos: {', '.join(struggles)}")
            
            dynamic_query = ". ".join(dynamic_parts) if dynamic_parts else "Preferencias de comida, restricciones médicas, gustos y síntomas digestivos del usuario"
            print(f"🔍 [RAG] Query dinámica: {dynamic_query}")
            
            # Brecha 5: Validación de Caché RAG
            # P1-3: aget (no get) — el call site está dentro de arun_plan_pipeline
            # async; .get síncrono bloqueaba el event loop esperando Redis (típicamente
            # 5-50ms, pero acumulativo si Redis está lento o caído con DB fallback).
            rag_cache_key = f"rag_{user_id}_{dynamic_query}"
            cached_rag = await _LLM_CACHE.aget(rag_cache_key)
            if cached_rag is not None:
                print(f"⚡ [CACHE HIT] Reutilizando contexto RAG para la misma query dinámica.")
                if len(cached_rag) == 3:
                    facts_data_sorted, visual_list, query_emb = cached_rag
                else:
                    facts_data_sorted, visual_list = cached_rag
                    # P1-3: get_embedding hace I/O bloqueante (HTTP a Vertex/OpenAI)
                    query_emb = await asyncio.to_thread(get_embedding, dynamic_query)
            else:
                # P1-3: Paralelizar las dos ramas independientes (text + visual).
                # Antes corrían secuenciales bloqueando el event loop:
                #   get_embedding → search_user_facts → get_multimodal_embedding → search_visual_diary
                # Latencia total ≈ suma de las 4 (típicamente 600-1500ms) + congelamiento del loop.
                # Ahora corren en paralelo en thread pool: latencia ≈ max(rama_text, rama_visual)
                # y el loop sigue libre para otras tasks (otros pipelines concurrentes).
                CATEGORY_PRIORITY_WEIGHTS = {
                    "alergia": 0, "condicion_medica": 1, "rechazo": 2,
                    "dieta": 3, "objetivo": 4, "preferencia": 5, "sintoma_temporal": 6
                }
                def get_fact_weight(fact_item):
                    meta = fact_item.get("metadata", {})
                    if isinstance(meta, dict):
                        cat = meta.get("category", "")
                        return CATEGORY_PRIORITY_WEIGHTS.get(cat, 7)
                    return 7

                async def _fetch_text_facts():
                    emb = await asyncio.to_thread(get_embedding, dynamic_query)
                    if not emb:
                        return [], None
                    facts = await asyncio.to_thread(
                        search_user_facts, user_id, emb,
                        query_text=dynamic_query, threshold=0.5, limit=10,
                    )
                    return (facts or []), emb

                async def _fetch_visual_facts():
                    v_emb = await asyncio.to_thread(get_multimodal_embedding, dynamic_query)
                    if not v_emb:
                        return []
                    v_data = await asyncio.to_thread(
                        search_visual_diary, user_id, v_emb, threshold=0.5, limit=10,
                    )
                    return v_data or []

                (facts_data, query_emb), visual_data = await asyncio.gather(
                    _fetch_text_facts(),
                    _fetch_visual_facts(),
                )

                if facts_data:
                    # === PRIORIZACIÓN POR CATEGORÍA (Anti-Poda Bruta) ===
                    facts_data_sorted = sorted(facts_data, key=get_fact_weight)
                    print(f"🧠 [RAG] Hechos textuales recuperados: {len(facts_data)} (ordenados por prioridad de categoría)")

                if visual_data:
                    visual_list = [f"• {item['description']}" for item in visual_data]
                    print(f"📸 [VISUAL RAG] Entradas visuales recuperadas: {len(visual_data)}")

                # P1-3: aset (no __setitem__) — async-safe. La tupla se serializa
                # como lista vía json.dumps; el destructuring upstream funciona igual.
                await _LLM_CACHE.aset(rag_cache_key, (facts_data_sorted, visual_list, query_emb))

            # P0-A1: sanitizar facts generales y visuales tras cargar (cache hit o
            # miss). Cubre ambos paths con una sola pasada — incluso si el cache
            # devuelve data sucia persistida antes del fix, queda saneada antes de
            # llegar al LLM. Costo: <1ms por fact (regex sobre patrones cortos).
            if facts_data_sorted:
                cleaned_facts = []
                for idx, _item in enumerate(facts_data_sorted):
                    fact_text = _item.get("fact") if isinstance(_item, dict) else None
                    if not isinstance(fact_text, str):
                        cleaned_facts.append(_item)
                        continue
                    sanitized = _sanitize_external_text_block(
                        fact_text,
                        f"history_context.rag.facts_data[{idx}].fact",
                        external_sanitize_hits,
                    )
                    if sanitized:
                        _item["fact"] = sanitized
                        cleaned_facts.append(_item)
                facts_data_sorted = cleaned_facts

            if visual_list:
                cleaned_visual = []
                for idx, _vline in enumerate(visual_list):
                    if not isinstance(_vline, str):
                        cleaned_visual.append(_vline)
                        continue
                    sanitized = _sanitize_external_text_block(
                        _vline,
                        f"history_context.rag.visual_list[{idx}]",
                        external_sanitize_hits,
                    )
                    if sanitized:
                        cleaned_visual.append(sanitized)
                visual_list = cleaned_visual

        except Exception as e:
            print(f"⚠️ [RAG] Error recuperando memoria: {e}")
            
    # === PRUNING FACT-BY-FACT (Basado en Tokens) ===
    # Estrategia: Los hechos estrictos (alergias, condiciones, rechazos) son NON-NEGOTIABLE.
    # Luego se agregan hechos generales y visuales uno por uno calculando ~4 caracteres por token.
    MAX_CONTEXT_TOKENS = 30000
    
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # 1. Strict facts son siempre incluidos (non-negotiable)
    full_rag_context = user_facts_text  # Ya contiene alergias, rechazos, condiciones
    
    # 2. Calcular presupuesto restante para hechos generales + visuales
    current_tokens = estimate_tokens(full_rag_context)
    remaining_tokens = MAX_CONTEXT_TOKENS - current_tokens
    
    if remaining_tokens > 0 and facts_data_sorted:
        header = "--- CONTEXTO GENERAL (Memoria Semántica, ordenado por prioridad) ---\n"
        general_section = header
        included_count = 0
        skipped_count = 0
        
        current_section_tokens = estimate_tokens(general_section)
        
        for item in facts_data_sorted:
            fact_line = f"• {item['fact']}\n"
            fact_tokens = estimate_tokens(fact_line)
            
            if current_section_tokens + fact_tokens <= remaining_tokens:
                general_section += fact_line
                current_section_tokens += fact_tokens
                included_count += 1
            else:
                skipped_count += 1
        
        if included_count > 0:
            full_rag_context += general_section
            remaining_tokens -= current_section_tokens
            if skipped_count > 0:
                print(f"✂️ [PRUNING] {skipped_count} hechos de baja prioridad descartados completos (límite tokens)")
    
    # 3. Agregar hechos visuales con el presupuesto restante
    if remaining_tokens > 25 and visual_list:  # ~100 chars
        visual_header = "\n--- INVENTARIO Y DIARIO VISUAL ---\nEl usuario subió fotos de estos alimentos:\n"
        visual_section = visual_header
        visual_section_tokens = estimate_tokens(visual_header)
        
        for vline in visual_list:
            candidate = vline + "\n"
            candidate_tokens = estimate_tokens(candidate)
            if visual_section_tokens + candidate_tokens <= remaining_tokens:
                visual_section += candidate
                visual_section_tokens += candidate_tokens
            else:
                break
        
        if len(visual_section) > len(visual_header):
            full_rag_context += visual_section
    
    if full_rag_context.strip():
        print(f"✅ [PRUNING] Contexto final: {estimate_tokens(full_rag_context)} tokens aprox (fact-by-fact, sin cortes)")
    # ====================================================

    # P0-A1: persistir métrica agregada de hits externos (memory_context +
    # history_context + RAG facts/visual). Una sola entrada por pipeline con
    # `node='sanitize_hits_external'` para distinguir en Grafana del path de
    # form_data (P1-Q8 = `node='sanitize_hits'`). Si todos los hits ya están
    # logueados individualmente más arriba, esto solo agrega el row de
    # alerting/cardinalidad. Sin hits → no-op.
    if external_sanitize_hits:
        for _hit in external_sanitize_hits:
            logger.warning(
                f"🛡️ [SANITIZE] P0-A1: prompt injection detectado en fuente externa "
                f"'{_hit['path']}': pattern={_hit['pattern']!r}, "
                f"snippet={_hit['snippet']!r}"
            )
        _enqueue_sanitize_metric(
            node="sanitize_hits_external",
            hits=external_sanitize_hits,
            user_id_for_metric=_user_id_for_metric,
            session_for_metric=_session_for_metric,
            skip_persist=_skip_sanitize_persist,
        )

    # 3. Estado inicial del grafo
    req_id = str(uuid.uuid4())[:8]
    request_id_var.set(req_id)
    # P1-NEW-1: setear user_id en ContextVar para que el wrapper
    # `ChatGoogleGenerativeAI` lo lea y aplique rate limit per-user.
    # Resolución del user_id efectivo:
    #   - `user_id` real (autenticado) → rate limit estricto contra otros
    #     pipelines del mismo usuario.
    #   - `session_id` (anónimo no-guest) → tratado como user_id efectivo
    #     para que un visitante no autenticado no pueda monopolizar slots
    #     vía múltiples sesiones.
    #   - `"guest"` o nada → None → bypass (semántica de tenant compartido).
    _rate_limit_uid = (
        actual_form_data.get("user_id")
        or actual_form_data.get("session_id")
    )
    if _rate_limit_uid == "guest" or not _rate_limit_uid:
        _rate_limit_uid = None
    user_id_var.set(_rate_limit_uid)

    # P1-NEW-4: dict mutable per-pipeline para trackear pérdidas de eventos
    # SSE. Las tasks descendientes (asyncio.Task, run_in_executor) heredan
    # el contexto y mutan este mismo objeto. Al final del pipeline, los
    # totales se emiten como métrica `progress_cb`.
    _pipeline_cb_stats_var.set({
        "dropped_cap": 0,
        "timed_out": 0,
        "failed_async": 0,
        "failed_sync": 0,
    })

    # [P1-ORQ-9] Cierre de schema drift: TODOS los campos declarados en
    # `PlanState` (líneas ~1827-1885) deben aparecer en `initial_state` con
    # un default explícito. Antes faltaban 9 campos opcionales; LangGraph en
    # strict-schema mode (modo recomendado para producción) puede filtrar
    # writes a keys no presentes en el initial_state, descartando silenciosamente
    # actualizaciones de nodos downstream. La línea 1877-1885 del propio
    # archivo documenta este riesgo: "Antes, sin esta declaración, LangGraph en
    # strict-schema mode lo filtraba al pasar `state` entre nodos, perdiendo
    # los buffers acumulados de forma silenciosa". El mismo riesgo existe para
    # campos declarados pero no inicializados. Defaults:
    #   - Optional[T] → None excepto donde el patrón existente requiere otro
    #     valor (ver `_token_buffers` abajo).
    #   - Todos los consumidores leen vía `state.get(...)` con fallback,
    #     así que None es semánticamente equivalente a "key missing" para
    #     ellos. Los nodos que ESCRIBEN estos campos (review, adversarial,
    #     compress_history, etc.) ahora tienen su update garantizado a no
    #     ser filtrado por LangGraph.
    initial_state: PlanState = {
        "request_id": req_id,
        "form_data": actual_form_data,
        "taste_profile": taste_profile,
        "nutrition": nutrition,
        "history_context": history_context,
        # [P1-ORQ-9] reflection_directive: escrito por `retry_reflection_node`
        # (línea ~6256) tras una review fallida. Consumer en `_build_planner_prompt`
        # (línea ~2764) usa truthy check → None es safe. P1-ORQ-2 ya garantizaba
        # reset en cada retry; este init asegura que LangGraph no filtre el write.
        "reflection_directive": None,
        # [P1-ORQ-9] compressed_context: escrito por nodos de compresión de
        # historial (líneas 2613, 2646, 2671, 2674, 2678) cuando el chunk de
        # historia es demasiado grande para inyectar crudo al prompt. Consumer
        # en `_build_shared_context` (línea 2498) usa `state.get(...) or
        # state.get("history_context", "")` — None pasa al fallback original.
        "compressed_context": None,
        "user_facts": full_rag_context,
        "semantic_cache_hit": False,
        "cached_plan_data": None,
        "profile_embedding": query_emb,
        "plan_result": None,
        "plan_skeleton": None,
        # [P1-ORQ-9] candidate_a/b + adversarial_rationale + _ab_temp_meta:
        # escritos por el nodo adversarial self-play (línea 3467+). Consumers
        # downstream (judge, assemble) usan `.get()` con fallback. None safe.
        "candidate_a": None,
        "candidate_b": None,
        "adversarial_rationale": None,
        "_ab_temp_meta": None,
        "review_passed": False,
        "review_feedback": "",
        "attempt": 1,
        "rejection_reasons": [],
        # [P1-ORQ-9] _rejection_severity: escrito por `review_node` (línea
        # 5834) cuando rechaza con clasificación crítica/high/minor. Consumer
        # en `should_retry` (línea 5883) usa `state.get(..., "minor")` —
        # con None presente, el get devuelve None (no "minor"), pero los
        # downstream checks `severity == "critical"` y `severity == "high"`
        # fallan idénticamente con None y con "minor", así que falls through
        # a la rama de retry — comportamiento funcional equivalente.
        "_rejection_severity": None,
        # P0-ORQ-3: declarado en PlanState (línea ~1872) y escrito por
        # `review_node` (línea ~5794) para alimentar la corrección quirúrgica
        # (`assemble_plan_node` línea ~5087, `_chunked_generation` línea ~2933).
        # Sin este key en `initial_state`, LangGraph en strict-schema mode podía
        # filtrar la escritura del nodo → set de días afectados queda vacío en
        # el read → toda regeneración quirúrgica validaba TODOS los días contra
        # el skeleton (no solo los regenerados) → falsos positivos de fidelidad
        # rechazaban planes válidos. Inicializado a None = "sin surgical fix
        # pendiente" (semántica equivalente a `state.get(...) or []`).
        "_affected_days": None,
        # [P1-ORQ-9] _cached_context: cacheo de los blocks de contexto del
        # prompt para reducir latencia de retries (los blocks no cambian
        # entre attempt 1 y 2). Escrito por `_build_shared_context` (línea
        # 2909) la primera vez. Consumer en líneas 2491-2492 usa truthy check
        # antes de retornar — None / {} ambos falsy → rebuild. Safe.
        "_cached_context": None,
        "progress_callback": progress_callback,
        "background_tasks": background_tasks,
        "pipeline_start": pipeline_start,
        # [P1-ORQ-9] _token_buffers: dict mutado in-place vía
        # `state.setdefault("_token_buffers", {})` desde `_emit_progress`
        # (línea 2081). IMPORTANTE: inicializar a {} (no a None). Si fuera
        # None, `state.setdefault("_token_buffers", {})` retornaría None
        # (clave existe), y luego `None.setdefault(day_key, ...)` lanzaría
        # AttributeError. Con {} preservamos el patrón existente: setdefault
        # devuelve el dict existente y la mutación per-day funciona.
        "_token_buffers": {},
    }
    
    # 4. Ejecutar el grafo con Timeout Global y Graceful Degradation (Mejora 5)
    latest_state = [initial_state]
    
    async def run_graph():
        # Usamos astream con stream_mode="values".
        # P1-Q2: `_get_plan_graph()` construye el grafo lazy en la primera
        # request por proceso. Si el build falla, propaga la excepción al
        # `except` de afuera que dispara el fallback matemático (P0-1) en
        # lugar de devolver 5xx al cliente.
        plan_graph = _get_plan_graph()
        async for event in plan_graph.astream(initial_state, stream_mode="values"):
            latest_state[0] = event
        return latest_state[0]

    # P0-1: cantidad de días que el caller solicitó. El plan final SIEMPRE debe
    # tener este número de días, ya sea generado por la IA, fallback puro, o
    # mezcla (parcial + relleno fallback). Antes el fallback hardcodeaba 3 y el
    # frontend recibía planes truncados sin advertencia.
    # P1-A5: helpers de fallback (`_build_fallback_day`, `_get_extreme_fallback_plan`,
    # `_is_day_valid`, `_is_plan_complete`, `_repair_partial_plan`) ahora viven
    # a nivel módulo. `nutrition` y `requested_days` se les pasan como kwargs.
    #
    # [P1-ORQ-8] Logging explícito cuando triggea la corrección defensiva.
    # ANTES, el cómputo `int(... or PLAN_CHUNK_SIZE) or PLAN_CHUNK_SIZE`
    # corregía silenciosamente strings no-numéricos (TypeError no atrapado),
    # `"0"` (int(=0) → falsy → segundo or), y negativos (`if < 1`) sin avisar.
    # El router ahora valida en boundary (`_validate_total_days` → 422), pero
    # la corrección defensiva sigue aquí porque otros callers acceden al
    # orquestador sin pasar por router (cron_tasks, proactive_agent, sync
    # wrapper). Esos callers deben enviar valores válidos; si llegamos a
    # corregir aquí, hay un bug upstream que vale la pena loguear.
    _raw_days = actual_form_data.get("_days_to_generate")
    try:
        requested_days = int(_raw_days) if _raw_days is not None else PLAN_CHUNK_SIZE
    except (TypeError, ValueError):
        logger.warning(
            f"🟠 [P1-ORQ-8] _days_to_generate={_raw_days!r} (tipo "
            f"{type(_raw_days).__name__}) no parseable a int. Defaulteando a "
            f"{PLAN_CHUNK_SIZE}. Bug upstream: caller no-router (cron / "
            f"proactive_agent / sync wrapper) envió tipo inválido. El router "
            f"valida vía `_validate_total_days` → 422."
        )
        requested_days = PLAN_CHUNK_SIZE
    if requested_days < 1:
        logger.warning(
            f"🟠 [P1-ORQ-8] _days_to_generate={_raw_days!r} → {requested_days} "
            f"(< 1). Corrigiendo a {PLAN_CHUNK_SIZE}. Bug upstream: caller "
            f"no-router envió valor cero/negativo. El router valida vía "
            f"`_validate_total_days` → 422 para clientes oficiales."
        )
        requested_days = PLAN_CHUNK_SIZE

    try:
        # Ejecutar asíncronamente con un timeout global (sin saltos de hilo)
        # P1-NEW-2: timeout configurable vía `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S`.
        final_state = await asyncio.wait_for(run_graph(), timeout=GLOBAL_PIPELINE_TIMEOUT_S)
    except Exception as e:
        print(f"🚨 [EXTREME GRACEFUL DEGRADATION] Error crítico en pipeline ({type(e).__name__}): {e}")
        final_state = latest_state[0] if latest_state else {}
        plan_partial = final_state.get("plan_result")
        # P0-1/P0-2: si no hay plan parcial usable, fallback total con la cantidad
        # de días solicitada. Si hay plan parcial (aunque sea con días vacíos o
        # incompletos), repararlo en lugar de descartar el trabajo del LLM.
        if not isinstance(plan_partial, dict) or not plan_partial:
            print(f"🛡️ [FALLBACK] Generando plan de emergencia matemático ({requested_days} días, by-pass de LLM)...")
            # P1-9: `_get_extreme_fallback_plan` ya setea `_is_fallback=True` dentro
            # del plan retornado. No duplicar en final_state — `arun_plan_pipeline`
            # retorna `final_state["plan_result"]` al caller, así que el flag en
            # final_state nunca se lee externamente y solo agregaba ambigüedad.
            final_state["plan_result"] = _get_extreme_fallback_plan(
                nutrition,
                actual_form_data.get("mainGoal", "Salud General"),
                num_days=requested_days,
            )
            final_state["attempt"] = 1
            final_state["review_passed"] = True
        else:
            # P1-9: `_repair_partial_plan` ya setea `plan_partial["_is_fallback"]=True`
            # cuando hace cualquier reparación. Nada más que hacer aquí.
            _repair_partial_plan(plan_partial, nutrition=nutrition, requested_days=requested_days)
    
    pipeline_duration = round(time.time() - pipeline_start, 2)

    print(f"\n{'🔗' * 30}")
    print(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
    print(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
    print("🔗" * 30 + "\n")

    # P0-1 / P1-8 / P1-A5: guardrail crítico de rechazo médico + transparencia
    # de rechazo no-crítico. Lógica idéntica al bloque inline previo, ahora
    # encapsulada en `_apply_critical_review_guardrails`.
    _apply_critical_review_guardrails(
        final_state,
        nutrition=nutrition,
        actual_form_data=actual_form_data,
        requested_days=requested_days,
    )

    # P0-2 / MEJORA 2 / P0-A3 / P1-A5: defensa en profundidad final + active
    # learning signals + cache schema version. Encapsula los tres bloques que
    # cierran el pipeline antes del return.
    _apply_final_defense_guardrails(
        final_state,
        nutrition=nutrition,
        actual_form_data=actual_form_data,
        requested_days=requested_days,
    )

    # GAP 4 / P1-A5 / [P1-1]: Score holístico + persistencia async.
    # ANTES, esta llamada ocurría ANTES de los dos guardrails de arriba: si el
    # plan original era reemplazado por fallback (rechazo médico crítico,
    # `_schema_invalid`, plan vacío), `last_pipeline_score` se persistía con la
    # confianza del plan REJECTED — `preflight_optimization_node` leía esa
    # señal en la próxima request y elegía estrategia como si la IA hubiera
    # tenido un happy path. AHORA corre DESPUÉS de los guardrails y detecta
    # `_is_fallback=True` para clampear el score a 0.0; el `delivered_was_fallback`
    # del metadata permite a Grafana distinguir score=0.0 por fallback vs por
    # otras razones (cal_score colapsado, etc.).
    _compute_pipeline_holistic_score_and_emit(
        final_state,
        nutrition=nutrition,
        actual_form_data=actual_form_data,
        initial_state=initial_state,
        pipeline_duration=pipeline_duration,
    )

    # [P1-5] Última línea de defensa antes del return. ANTES,
    # `return final_state["plan_result"]` con bracket access lanzaba KeyError
    # opaco al caller (FastAPI → 500 sin contexto) si la key se perdía. Las
    # defensas existentes la garantizan en condiciones normales:
    #   - `initial_state` (P1-ORQ-9) la setea a None.
    #   - `_apply_final_defense_guardrails` arriba reemplaza None/empty por
    #     `_get_extreme_fallback_plan(...)` antes de llegar acá.
    # PERO si llegamos aquí sin un dict válido (refactor futuro que mueva un
    # guardrail dentro de un `if`, regresión de LangGraph que filtre keys
    # tras el merge final, test directo del pipeline con state mutado), un
    # crash silencioso degrada peor que un fallback marcado. CRITICAL log
    # dispara alerting inmediato; el cliente recibe un plan en vez de 5xx.
    plan_to_return = final_state.get("plan_result")
    if not isinstance(plan_to_return, dict) or not plan_to_return:
        logger.critical(
            f"🚨 [P1-5] Pipeline terminó SIN plan_result válido pese a los "
            f"dos guardrails arriba (tipo={type(plan_to_return).__name__}, "
            f"truthy={bool(plan_to_return)}). Bug upstream en "
            f"`_apply_critical_review_guardrails` o `_apply_final_defense_guardrails`. "
            f"Generando último fallback de emergencia ({requested_days} días) "
            f"para no devolver None / KeyError al cliente."
        )
        plan_to_return = _get_extreme_fallback_plan(
            nutrition,
            actual_form_data.get("mainGoal", "Salud General"),
            num_days=requested_days,
        )
        # Marcar para que el caller (router/cron) NO lo persista como plan
        # real. `_is_fallback` ya lo setea `_get_extreme_fallback_plan`, pero
        # añadimos una marca específica de este path para que Grafana pueda
        # distinguir el origen del fallback en alerts.
        plan_to_return["_p1_5_emergency_return"] = True

    return plan_to_return

def _run_arun_in_isolated_loop(arun_kwargs: dict, ctx: contextvars.Context) -> dict:
    """P1-NEW-6: Helper ejecutado en `_SYNC_WRAPPER_EXECUTOR`.

    Crea un coroutine fresco y un event loop fresco DENTRO del worker thread
    del executor — `asyncio.run` no puede compartirse entre llamadas porque
    cierra el loop al terminar. `ctx.run(asyncio.run, coro)` propaga los
    contextvars del caller (request_id, user_id, etc.) sin mezclarlos con
    otros workers del pool: cada submission opera en su propia copia.

    Recibe `arun_kwargs` (no el coroutine) porque crear el coroutine en el
    thread caller y pasarlo al worker provoca `RuntimeError: coroutine ...
    was never awaited / attached to different loop` en algunos casos border.
    Crearlo aquí garantiza que el loop que lo ejecuta es el mismo donde nació.
    """
    coro = arun_plan_pipeline(**arun_kwargs)
    return ctx.run(asyncio.run, coro)


def run_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None, background_tasks=None) -> dict:
    """Wrapper síncrono para mantener compatibilidad con cron/callers no-async.

    P1-NEW-6: bajo carga concurrente, todos los callers comparten un executor
    bounded (`_SYNC_WRAPPER_EXECUTOR`, default 4 workers). Los callers N+1...
    se serializan en cola en lugar de spawnear threads + event loops sin bound,
    evitando contención de GIL y races sobre globales mutados desde múltiples
    loops simultáneos.

    Path rápido: si NO hay un event loop activo (ej. script CLI directo),
    `asyncio.run` se ejecuta inline — sin overhead del executor.

    ⚠️ P1-Q7: NO LLAMAR DESDE CÓDIGO ASYNC ⚠️
    --------------------------------------------
    Llamar este wrapper desde una task asyncio (handler async de FastAPI,
    BackgroundTask, otro loop running) bloquea el thread caller con
    `future.result()`. Pero ese thread es el que ejecuta el event loop, así
    que el loop queda CONGELADO durante todo el pipeline (~2-5 min). Otras
    requests/tasks del worker quedan en standby — degradación catastrófica
    de latencia tail aunque NO sea un deadlock infinito.

    En su lugar, desde código async usa el coroutine directamente:

        # ❌ MAL (desde una def async o BackgroundTask):
        result = run_plan_pipeline(form_data, ...)

        # ✅ BIEN:
        result = await arun_plan_pipeline(form_data, ...)

    Modos de detección (controlados por `MEALFIT_SYNC_WRAPPER_STRICT_MODE`):
      - `True` (default actual, P1-A7): lanza `RuntimeError` explícito
        redirigiendo a `arun_plan_pipeline`. Auditoría confirmó cero callers
        legacy desde código async; cualquier `RuntimeError P1-Q7` es una
        regresión nueva, no falsa alarma.
      - `False`: warning ruidoso con stacktrace, sigue funcionando. Escape
        hatch operacional si aparece un caller legacy no documentado en
        producción y se necesita tiempo para migrarlo. Bajar vía env var:
            MEALFIT_SYNC_WRAPPER_STRICT_MODE=0
        Buscar el log `[SYNC WRAPPER] P1-Q7: ...` para encontrar callers.
    """
    arun_kwargs = dict(
        form_data=form_data,
        history=history,
        taste_profile=taste_profile,
        memory_context=memory_context,
        progress_callback=progress_callback,
        background_tasks=background_tasks,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # P1-Q7: foot-gun detection. Estamos siendo llamados desde código que
        # tiene un event loop ACTIVO. Capturamos stacktrace del caller para
        # facilitar localizar la regresión, y aplicamos política configurada.
        import traceback as _tb
        stack_summary = "".join(
            _tb.format_list(_tb.extract_stack(limit=8)[:-1])  # excluir esta frame
        )
        msg = (
            f"P1-Q7: run_plan_pipeline() llamado desde código async (event loop "
            f"activo detectado). Esto bloquea el loop ~2-5min durante el "
            f"pipeline, congelando otras requests del worker. Migra a:\n"
            f"    result = await arun_plan_pipeline(...)\n"
            f"Stacktrace del caller:\n{stack_summary}"
        )
        if SYNC_WRAPPER_STRICT_MODE:
            raise RuntimeError(msg)
        logger.warning(f"[SYNC WRAPPER] {msg}")

        # Path "running loop" (legacy / non-strict): delegar al pool con
        # concurrencia acotada. `future.result()` bloquea el thread caller
        # (mismo contrato que el `t.join()` previo), pero los workers del
        # pool están bounded. Bajo N callers > SYNC_WRAPPER_MAX_WORKERS, el
        # N+1 espera en cola del executor — no se inflan threads adicionales.
        ctx = contextvars.copy_context()
        future = _SYNC_WRAPPER_EXECUTOR.submit(_run_arun_in_isolated_loop, arun_kwargs, ctx)
        return future.result()
    else:
        # Path "standalone": script CLI sin loop activo. Sin overhead del pool.
        return asyncio.run(arun_plan_pipeline(**arun_kwargs))
