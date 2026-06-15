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
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek. El wrapper local
# `ChatDeepSeek` (abajo) subclasea el cliente base para backpressure +
# instrumentación; el router por tier vive en llm_provider.
from llm_provider import (
    ChatDeepSeek as _ChatDeepSeekBase,
    DEEPSEEK_FLASH,
    DEEPSEEK_PRO,
    PAID_TIERS,
    get_user_tier,
)
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception
import logging
import threading
from db_plans import search_similar_plan

# Mejora 1: Semaphore Distribuido Global para backpressure
# P1-10: `time` y `uuid` ya están al inicio del módulo; no reimportar.
import uuid
import asyncio
import weakref
import contextvars
import functools
from contextlib import contextmanager, asynccontextmanager


# ============================================================
# [P1-COST-INSTRUMENTATION-PHASE2 · 2026-05-16] ContextVar para etiquetar
# llamadas LLM con el NODO del pipeline donde se originan. Phase 1 dejó la
# columna `llm_usage_events.node` 100% NULL ("unknown") porque la signature
# de `_safe_ainvoke` no incluía el caller — modificar 30+ callsites era
# invasivo. Phase 2 usa contextvar: cada nodo de LangGraph setea su nombre
# al entrar (via decorator `@_node_label("nombre")`), y propaga
# automáticamente a CUALQUIER `_safe_ainvoke` invocada en su scope
# (incluso a través de `asyncio.create_task` por la semántica de
# Python 3.7+ que copia el contexto a tasks hijas).
#
# El `_emit_llm_usage_event_best_effort` lee el var y lo pasa a
# `db_profiles.log_llm_usage_event(node=...)`. Si una llamada LLM ocurre
# FUERA del pipeline (e.g. chat agent tools), el var queda en None → la
# fila persiste con `node=NULL` (mismo comportamiento que phase 1 — no
# regresión).
#
# Tooltip-anchor: P1-COST-INSTRUMENTATION-PHASE2-CONTEXTVAR
_current_node_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mealfit_current_node", default=None
)


def _node_label(name: str):
    """Decorator para nodos async del orquestador. Setea el contextvar al
    entrar, lo resetea al salir — incluso si la función levanta excepción.

    Uso:
        @_node_label("planner")
        async def plan_skeleton_node(state): ...

    El ContextVar propaga a:
      - Llamadas async dentro del cuerpo del nodo.
      - Tasks lanzadas via `asyncio.create_task` o `asyncio.gather`
        (Python 3.7+ copia el contexto).
      - Llamadas síncronas vía `asyncio.to_thread` (también copian contexto).

    Si el nodo no es async, devolver una versión sync del decorator.
    """
    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                token = _current_node_var.set(name)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _current_node_var.reset(token)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                token = _current_node_var.set(name)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _current_node_var.reset(token)
            return sync_wrapper
    return decorator


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
# [P3-NEW-D · 2026-05-08] Registry global de knobs auto-inventariados.
# ------------------------------------------------------------
# Antes, `_log_active_knobs()` mantenía un dict hardcoded de ~30 knobs
# visibles. Con 50+ knobs activos en el módulo (algunos viviendo en helpers,
# otros añadidos sin actualizar el dict), el log de startup era incompleto:
# un override `MEALFIT_LLM_MAX_PER_USER=5` aparecía, pero
# `MEALFIT_LLM_COMBINED_MAX_WAIT_S` podía no estar listado, y el operador
# no podía confirmar que el override tomó efecto sin grep manual al
# código fuente. Drift entre lo que el operador cree configurado y lo
# realmente leído por el proceso.
#
# Ahora cada llamada a `_env_int/_env_float/_env_bool` registra el knob
# con su default, valor crudo del env (o None) y valor parseado final.
# `_log_active_knobs()` itera el registry completo en startup y emite
# inventario ordenado con highlight de overrides activos.
#
# Observabilidad pura, cero cambio de comportamiento.
#
# [P2-1 · 2026-05-08] Helpers extraídos a `backend/knobs.py` para que módulos
# importados POR este (notablemente `constants.py`) puedan registrar sus knobs
# en el mismo registry sin generar import circular. Las re-exports de abajo
# preservan los call sites históricos `from graph_orchestrator import _env_int`.
# ============================================================
from knobs import (  # noqa: E402  (re-export bloque doc-arriba)
    _KNOBS_REGISTRY,
    _register_knob,
    _env_int,
    _env_float,
    _env_bool,
    _env_str,
    get_knobs_registry_snapshot,
)


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
# [P2-ORCH-8 · 2026-05-28] Default 2 → 3 para igualar PLAN_CHUNK_SIZE (=3): cada
# generación dispara PLAN_CHUNK_SIZE días en paralelo (gather), pero con max=2 el
# 3er día siempre quedaba en cola → day-gen efectivamente 2-wide en vez de 3-wide
# (~una ola extra de 50-90s de wall-clock en el nodo que es >50% de la latencia).
# El abuso cross-session sigue acotado por el cap global (LLM_MAX_CONCURRENT=4) +
# el guard 409 de /pending-status. PLAN_CHUNK_SIZE se importa más abajo (~1344);
# la reconciliación con warning vive allí. Tooltip-anchor: P2-ORCH-8.
LLM_MAX_PER_USER            = _env_int  ("MEALFIT_LLM_MAX_PER_USER",            3)
LLM_USER_LOCK_TIMEOUT_S     = _env_int  ("MEALFIT_LLM_USER_LOCK_TIMEOUT_S",     60)
# [P3-PROD-AUDIT-3 · 2026-05-30] Cota del dict in-process `_local_sync` de
# DistributedPerUserSemaphore. En modo Redis-DOWN (REDIS_URL ausente o Redis
# caído — degradación soportada por cache_manager) el dict acumula un
# threading.Semaphore por user_id DISTINTO sin eviction → fuga lenta pero
# ilimitada en un proceso long-lived (--workers 1). Al exceder el cap se purgan
# entradas IDLE (todos sus permits disponibles). Clamp generoso.
LLM_PER_USER_LOCAL_CACHE_MAX = _env_int(
    "MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX", 4096, validator=lambda v: 64 <= v <= 1_000_000
)
# Espera máxima por un slot per-user antes de degradar al semáforo global
# (más corto que LLM_MAX_WAIT_S — un usuario saturado debe ceder rápido para
# que el grafo continúe vía global).
LLM_USER_MAX_WAIT_S         = _env_int  ("MEALFIT_LLM_USER_MAX_WAIT_S",         30)
# [P2-ORCH-11 · 2026-05-28] Cota del busy-poll LOCAL (fallback cuando Redis está
# caído). Los paths Redis ya degradan vía LLM_MAX_WAIT_S/LLM_USER_MAX_WAIT_S, pero
# el busy-poll local giraba a 20Hz SIN cota → bajo Redis-down + saturación
# sostenida, N corrutinas quemaban CPU del event loop hasta el techo de 720s del
# pipeline. Al expirar: stat + warning + backpressure error (fail-fast → el handler
# global entrega fallback). Default 120s, clamp [1, 3600]. Tooltip-anchor: P2-ORCH-11.
LLM_LOCAL_MAX_WAIT_S        = _env_int  ("MEALFIT_LLM_LOCAL_MAX_WAIT_S",        120,
                                         validator=lambda v: 1 <= v <= 3600)

# [P1-28] Cap COMBINADO sobre la composición per-user → global.
# `acquire_user_and_global` antes podía esperar hasta
# `LLM_USER_MAX_WAIT_S + LLM_MAX_WAIT_S` (default 30+90=120s) en el peor
# caso porque cada semáforo aplicaba su propio bound de espera de forma
# independiente. Bajo carga, un usuario saturando per-user que luego
# encontraba el global saturado pagaba el peor caso completo. Ahora,
# cuando el tiempo gastado en per-user agota >=`LLM_COMBINED_MAX_WAIT_S`,
# degradamos global a su semáforo local in-process (`_local_semaphore`)
# en lugar de esperar otros 90s.
# Default = SUMA de los dos individuales para preservar el comportamiento
# previo en steady state. Operadores que quieran un cap más estricto
# (e.g. 60s) solo necesitan setear el env var.
LLM_COMBINED_MAX_WAIT_S     = _env_int  ("MEALFIT_LLM_COMBINED_MAX_WAIT_S",
                                          LLM_USER_MAX_WAIT_S + LLM_MAX_WAIT_S)

# --- Circuit breaker LLM ---
# [P2-KNOBS-ENV-INT-NO-VALIDATOR · 2026-05-24] Validators añaden defensa
# contra override accidental: `=0` o negativo en FAILURE_THRESHOLD abre el
# breaker eternamente; `=0` en RESET_TIMEOUT_S deja el breaker open sin
# auto-recovery. Knobs críticos para el flujo LLM — ahora un WARNING al
# load identifica el typo antes de runtime degradado.
CB_FAILURE_THRESHOLD        = _env_int  ("MEALFIT_CB_FAILURE_THRESHOLD",        3,
                                          validator=lambda v: 1 <= v <= 1000)
CB_RESET_TIMEOUT_S          = _env_int  ("MEALFIT_CB_RESET_TIMEOUT_S",          30,
                                          validator=lambda v: 1 <= v <= 86_400)
CB_LOCAL_HEALTH_TTL_S       = _env_float("MEALFIT_CB_LOCAL_HEALTH_TTL_S",       1.0)

# --- Hedging per-day (generate_days_parallel_node) ---
# `HEDGE_AFTER_BASE_S`: tiempo soft antes de lanzar el intento especulativo.
# `HARD_CEILING_S`: tiempo duro tras el cual cancelamos primary+hedge y damos error.
#
# [P1-HEDGE-THRESHOLD-RAISE · 2026-05-21] Default subido 45s → 90s. Razón:
# observación 2026-05-21 chunk 713ff43a, intento #2 — los 3 day_generators
# tardaron 74s, 89s, 104s (rangos normales para gemini-3-flash bajo throttling).
# Con threshold=45s, los 3 días dispararon hedge → "3/3 hedges en flight" →
# saturación del limiter (otros días esperaron sin hedge) Y +3 llamadas Gemini
# innecesarias (los primaries terminaron <30s después del hedge fire).
#
# Con threshold=90s, en ese mismo chunk solo el día 3 (104s) habría disparado
# hedge — ahorro estimado ~2 llamadas Gemini por chunk en condiciones normales.
# Trade-off: si Gemini se atasca de verdad (>90s sin respuesta), el hedge llega
# 45s más tarde. Pero esos chunks son raros y ya están cubiertos por
# `HARD_CEILING_S=170s` que cancela cualquier intento congelado.
#
# Rollback sin redeploy: `MEALFIT_HEDGE_AFTER_BASE_S=45` revierte al
# comportamiento pre-fix-original; `MEALFIT_HEDGE_AFTER_BASE_S=90` al pre-fix-V2.
#
# [P3-COST-CUT-V2 · 2026-05-21] Default subido 90s → 120s. Razón: análisis del
# costo desperdiciado por hedges que fired pero perdieron. Cuando primary gana,
# `pt.cancel()` solo schedula la cancelación — la task hedge sigue corriendo
# hasta el próximo checkpoint async, completa el LLM call, genera ~1600 output
# tokens × $9/M = ~$0.014 desperdiciados por hedge. Plan productivo 2026-05-21
# disparó 2/3 hedges con primary ganador (Día 2 a 143s, Día 3 a 101s con
# threshold=90) → ~$0.028 desperdiciado en ese plan. Con threshold=120s, solo
# Día 2 dispararía hedge (Día 3 a 101s queda por debajo) — ahorro ~$0.014 por
# plan en escenario típico.
# Trade-off: si Gemini se atasca >120s, el hedge llega 30s más tarde que antes.
# Worst case wall-clock peor, pero `HARD_CEILING_S=170s` sigue siendo el cap
# de seguridad.
HEDGE_AFTER_BASE_S          = _env_float("MEALFIT_HEDGE_AFTER_BASE_S",          120.0)
HARD_CEILING_S              = _env_float("MEALFIT_HARD_CEILING_S",              170.0)
# [P2-HEDGE-LIMITER-RAISE · 2026-05-16] Cap absoluto de hedges concurrentes por
# pipeline. Default subido 2 → 3 para que los 3 day_generators paralelos del
# chunk inicial (cycle_start, 3 días) tengan protección simétrica. Pre-fix:
# `max(1, LLM_SEMAPHORE.max_concurrent // 2) = 2` → cuando 3 días están en
# flight, el último que dispara hedge queda sin slot (incidente plan bf6f1383
# 2026-05-16: Día 1 sin hedge → falló → CB OPEN → self_critique bloqueado).
# Clamp [1, max(1, LLM_MAX_CONCURRENT-1)] preserva headroom para primaries.
HEDGE_MAX_CONCURRENT_KNOB   = _env_int  ("MEALFIT_HEDGE_MAX_CONCURRENT",        3)

# --- Retry policy + timeout global del pipeline ---
# [P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Default 2 → 3. Razón: usuarios con
# señal baja (chunk 2 inicial sin historial denso) pueden necesitar más
# intentos para que el LLM converja a un plan coherente. Combinado con la
# notificación visible (`plan_data._quality_degraded` cuando se agotan los
# 3 intentos), el usuario sabe explícitamente cuando el sistema "se rindió"
# en lugar de recibir silenciosamente un plan degradado. Trade-off: +1
# intento = ~50% más de latencia/costo en chunks que reach max_attempts,
# pero esos son <5% del volumen total — los happy paths siguen igual.
MAX_ATTEMPTS                = _env_int  ("MEALFIT_MAX_ATTEMPTS",                3)
# [P2-ORCH-12 · 2026-05-28] Límite de super-steps del grafo LangGraph como
# defensa-en-profundidad. La terminación del retry depende hoy SOLO del contador
# de intentos en retry_reflection_node + should_retry + budget guards. Un refactor
# futuro que re-entre plan_skeleton sin incrementar `attempt` loopearía hasta el
# techo de 720s (12 min/request) en vez de fallar rápido con GraphRecursionError.
# Default 50 (>> el máximo legítimo ~26 super-steps de 3 intentos + marker_regen;
# << un loop infinito). Clamp [10, 200]. Tooltip-anchor: P2-ORCH-12.
GRAPH_RECURSION_LIMIT       = _env_int  ("MEALFIT_GRAPH_RECURSION_LIMIT",       50,
                                         validator=lambda v: 10 <= v <= 200)
# [P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Umbral mínimo de `previous_plan_quality`
# para inyectar señales de preferencia históricas (Señales 7-10 en
# `_inject_history_aware_signals`). Sin este gate, un usuario con Quality
# acumulada baja recibía señales débiles inyectadas como verdades fuertes
# ("Platos recurrentes inyectados" con base 0.33) → el LLM forzaba gustos
# que no eran genuinos. Con el gate, si confidence < 0.40, esas señales se
# OMITEN del prompt y el LLM genera solo en base a:
#   - Macros calculados (BMR/TDEE/target — siempre presentes)
#   - Pantry / shopping list (siempre presentes en chunk 2+)
#   - Alergias / condiciones médicas (siempre presentes)
# Resultado: plan coherente sin inventar preferencias que el usuario nunca
# expresó. Las Señales 1-6 (EMA granular, abandonos, emocional, técnicas)
# siguen self-gating: solo inyectan si su data subyacente existe.
MIN_LEARNING_CONFIDENCE     = _env_float("MEALFIT_MIN_LEARNING_CONFIDENCE",    0.40)
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
# [P1-31] Tamaño del pool dedicado de fact-checking médico
# (`_FACT_CHECK_EXECUTOR`). Antes hardcodeado a 2 → bajo carga (5+
# pipelines concurrentes) generaba backlog de fact-checks que tardaban
# 20-90s extra por encolamiento, sin que un operador pudiera tunear el
# tradeoff (concurrencia LLM clínico vs throughput) sin redeploy. Ahora
# vía env. Default 2 preserva comportamiento previo. Clamp a [1, 16]:
#   - 1: deshabilita paralelismo (debugging, providers con rate limit
#     extremo).
#   - 2 (default): el provider clínico tolera ~2 calls concurrentes sin
#     degradar latency.
#   - >2: solo si el provider tolera más concurrencia (test loads). Cap
#     16 evita config errors absurdos (e.g. "MEALFIT_FACT_CHECK_POOL_SIZE=200")
#     que saturarían threads del proceso sin beneficio.
FACT_CHECK_POOL_SIZE        = max(1, min(16, _env_int("MEALFIT_FACT_CHECK_POOL_SIZE", 2)))
# [P2-ORCH-13 · 2026-05-28] Pool de I/O DB (`_DB_EXECUTOR`) como knob. Antes era
# max_workers=8 hardcoded — el pool COMPARTIDO de TODO el I/O DB sync vía `_adb`.
# El step de assemble emite 1 fetch + 3 deltas concurrentes; con 2-3 pipelines +
# crons la demanda excede 8 y las queries se serializan (tail latency) sin lever
# para subirlo sin redeploy. Clamp [1, 64] (espejo del cap de FACT_CHECK_POOL_SIZE
# — evita configs patológicas). Tooltip-anchor: P2-ORCH-13.
DB_EXECUTOR_MAX_WORKERS     = max(1, min(64, _env_int("MEALFIT_DB_EXECUTOR_MAX_WORKERS", 8)))

# --- Self-critique correction timeout (P1-FIX-CRITIQUE / P4-TIMEOUT-1) ---
# Timeout por día corregido en `self_critique_node._correct_single_day`.
#
# Historia de bumps:
#   - 70s (original): hardcoded. Día 3 (último corregido) sufría peak de
#     cola del proveedor → TimeoutError → día sin corregir, justamente el
#     que tenía las violaciones más graves (8.5 lonjas pavo en cena en
#     incidente real). Plan luego rechazado por review_plan_node sin
#     budget para retry.
#   - 90s (P1-FIX-CRITIQUE): +25-30% headroom para cola normal. Resolvió
#     ~70% de los timeouts de Día 3.
#   - 120s (P4-TIMEOUT-1): bump tras 3 corridas en producción 2026-05-05
#     mostrando 504 DEADLINE_EXCEEDED de Gemini cuando el provider está
#     bajo carga. Patrón observado:
#       - SDK de Gemini hace retry interno con exponential backoff (1.3-2s)
#         tras 504 → +2-5s al wall-clock de la llamada
#       - Si la llamada original ya estaba cerca de 60-70s (común con
#         tools), el retry empuja el total a 85-95s, justo en el cap.
#     Subir a 120s absorbe ~95% de estos casos sin penalizar el happy path.
#
# Costo wall-clock: las 3 correcciones corren en `asyncio.gather`, así que
# el costo es max(3) no sum(3). Subir el cap individual añade hasta +30s en
# el peor caso (los 3 días en cola simultáneamente). En el happy path
# (primary day termina rápido, los demás no necesitan max budget) cero
# impacto. Trade-off favorable.
#
# Sinergia: este bump reduce la frecuencia de `_critique_unresolved`
# markers que P1-SURGICAL-1 detecta para forzar regen en retry. Menos
# markers = menos retries innecesarios = menos costo total.
#
# [P4-TIMEOUT-3] (corrida 2026-05-04 03:26): aún con 120s, observamos
# 3 days timeoutear simultáneamente bajo Gemini cascade-overload. P4-TIMEOUT-2
# añadió circuit breaker (abort tras N timeouts), pero el knob individual
# también sube a 150s — gana 30s extra de tolerancia para los retries
# internos del SDK (1-3 backoffs de 2-5s cada uno). Sigue paliativo (la
# solución estructural es P4-TIMEOUT-2), pero quita el último 5% de casos
# borderline que P4-TIMEOUT-1 dejó sin cubrir. Wall-clock self-critique
# en happy path no cambia (max(3) en gather, igual que antes).
# [P6-TIMEOUT-4] (corrida 2026-05-05 19:58 [c6eaf808]): los 3 días
# timeoutearon en self-critique a 150s, después Día 2 timeoutea OTRA vez en
# P5-MARKER-REGEN. Causa: Gemini API lento (504 DEADLINE_EXCEEDED visto 2×)
# + correctores con prompts largos (sugerencias verbosas del evaluator).
# Bump 150→180s gana 30s extra para días borderline sin afectar happy path
# (gather paralelo, max(3) wall-clock = 1 timeout). Trade-off: peor caso
# pipeline +90s (3 días × 30s extra), mejor caso 0 cambio. P4-TIMEOUT-2 CB
# aún limita explosión bajo cascade-overload.
CRITIQUE_FIX_TIMEOUT_S      = _env_float("MEALFIT_CRITIQUE_FIX_TIMEOUT_S",      180.0)
# [P1-SELF-CRITIQUE-SKIP-CLEAN · 2026-05-28] Si los DOS detectores determinísticos
# del self_critique (`_count_staple_repetitions` + `_detect_slot_incoherence`)
# vienen vacíos, el plan no tiene issues "duros". El evaluador LLM (~30s siempre)
# + sus correcciones (las llamadas más caras del pipeline: p50 ~192s, ~6.4k tokens
# out, ~$3.5 acumulado en prod) solo aportarían pulido subjetivo marginal. Con el
# knob ON saltamos TODO self_critique en ese caso → recorte de latencia+costo en
# la mayoría de planes estructuralmente sanos. Flip a False restaura el evaluador
# en cada plan (comportamiento pre-fix). Las señales determinísticas siguen siendo
# el floor de calidad — solo se omite el pulido subjetivo cuando NO hay nada duro.
SELF_CRITIQUE_SKIP_WHEN_CLEAN = _env_bool("MEALFIT_SELF_CRITIQUE_SKIP_WHEN_CLEAN", True)

# ============================================================
# [P4-TIMEOUT-2] Self-critique circuit breaker
# ------------------------------------------------------------
# P4-TIMEOUT-1 subió `CRITIQUE_FIX_TIMEOUT_S` 90→120s. Cubre el ~95% de
# los 504 cuando el peak es individual, PERO no ayuda cuando Gemini está
# en cascada (provider overload correlacionado). Observado 2026-05-04
# corrida 03:26: los 3 días corrieron en `asyncio.gather` y los 3 se
# timeoutearon a 120s simultáneamente. Wall-clock self-critique = 120s
# (no 360s gracias a gather), pero el costo real fue:
#   - 0 días corregidos
#   - 3 markers `_critique_unresolved` → P1-SURGICAL-1 fuerza regen
#   - retry path completo (otros ~250s) → pipeline ~400s
#
# El waste no es el timeout en sí (ya es paralelo), es seguir esperando
# a los demás cuando ya sabemos que el provider está saturado. Si el
# primer día timeoutea, los siguientes muy probablemente también — son
# el mismo proveedor en el mismo segundo. Mejor abortar temprano y
# proceder al assemble (snapshot fallback de P0-PIPE-1 ya cubre quality),
# dejando que los días no resueltos vayan a regen en retry vía P1-SURGICAL-1.
#
# Threshold = 2 (default): después de 2 timeouts cancelamos los pendientes.
# - Threshold = 1 sería muy agresivo (un timeout aislado podría ser flaky
#   y los otros días sí completarían).
# - Threshold = 3 nunca abortaría con `critique_max_days=3` (el típico).
#
# Trade-off: en peak provider-load aceptamos perder días de corrección
# (el snapshot del intento original sigue válido + retry los regenera),
# a cambio de no quemar 120s de wall-clock esperando fallos seguros.
# ============================================================
CRITIQUE_TIMEOUT_ABORT_THRESHOLD = _env_int("MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD", 2)

# ============================================================
# [2026-05-06] Pro fallback para correcciones del self-critique
# ------------------------------------------------------------
# Cuando Flash hace timeout corrigiendo un día (caso real corrida 16:30:
# 3/3 días con 504 DEADLINE_EXCEEDED), reintentamos con Gemini Pro.
# Pro es ~2x más lento y caro pero más estable bajo prompts complejos
# y bajo provider load. Trade-off:
#   - Costo: ~$0.001 extra por día corregido vía fallback.
#   - Wall-clock: +PRO_FALLBACK_TIMEOUT_S al peor caso (gather paralelo
#     amortigua: max de los días que cayeron a fallback).
#   - Calidad: días que antes quedaban con `_critique_unresolved` ahora
#     se corrigen → menos regens en retry, mejor plan al usuario.
#
# Si Pro también timeoutea o el CB está OPEN, mantenemos el original con
# marker (comportamiento idéntico al pre-fallback). El knob permite
# desactivar el fallback rápido si Pro queda saturado y solo añade latencia.
# ============================================================
CRITIQUE_PRO_FALLBACK_ENABLED   = _env_bool("MEALFIT_CRITIQUE_PRO_FALLBACK_ENABLED",  True)
CRITIQUE_PRO_FALLBACK_TIMEOUT_S = _env_float("MEALFIT_CRITIQUE_PRO_FALLBACK_TIMEOUT_S", 120.0)

# --- Progress callbacks (SSE) ---
PROGRESS_CB_MAX_PENDING     = _env_int  ("MEALFIT_PROGRESS_CB_MAX_PENDING",     1000)
PROGRESS_CB_TIMEOUT_S       = _env_float("MEALFIT_PROGRESS_CB_TIMEOUT_S",       10.0)

# --- LLM cache TTL ---
LLM_CACHE_TTL_S             = _env_int  ("MEALFIT_LLM_CACHE_TTL_S",             300)

# --- Egg-whites cap por meal/día ---
# El planner usa "claras de huevo" como proteína fácil sin mesura. Se vio
# corrida 2026-05-06 02:44 [b0791cfb]: 11 claras en una sola comida, 36 en 3 días,
# revisor médico rechazó CRÍTICO ("riesgo de estrés renal, deficiencia de
# biotina por avidina"). El usuario re-disparó y el sistema iba a generar el
# mismo problema. Guard programático en assemble_plan_node corta el meal a
# MAX_PER_MEAL antes de pasarlo al revisor. Defaults conservadores:
#   - 6 claras/meal ≈ 21g proteína de claras puras (suficiente).
#   - 12 claras/día ≈ 42g (cubre 2 comidas con claras como proteína principal).
MAX_EGG_WHITES_PER_MEAL = _env_int("MEALFIT_MAX_EGG_WHITES_PER_MEAL", 6)
MAX_EGG_WHITES_PER_DAY  = _env_int("MEALFIT_MAX_EGG_WHITES_PER_DAY",  12)
# [P2-EGG-WHITE-MEALS-CAP · 2026-05-16] Tercer cap: limitar el NÚMERO de
# meals/día que usan claras de huevo como PROTEÍNA BASE. Pre-fix los caps
# anteriores (PER_MEAL=6, PER_DAY=12) limitaban cantidad pero NO frecuencia
# → el planner ponía claras en 3-4 meals/día (desayuno + almuerzo aglutinante
# + merienda batido + cena revoltillo) → reviewer médico rechazaba por
# "frecuencia de consumo de claras de huevo es excesivamente alta en
# múltiples comidas" (plan_id=fbd014b2 2026-05-16).
# Cap default 2 meals/día: típicamente desayuno + 1 más. Los meals en
# exceso (el 3ro+ con claras) ven sus claras recortadas a 1 (simbólico,
# como aglutinante de receta) — no se elimina el ingrediente para no
# romper la coherencia de la receta downstream.
MAX_MEALS_WITH_EGG_WHITES = _env_int("MEALFIT_MAX_MEALS_WITH_EGG_WHITES", 2)

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
#
# [P3-1 2026-05-08] Procedimiento de restauración tras activar el escape hatch:
#   1. `grep "[SYNC WRAPPER] P1-Q7:" en logs` para identificar al caller legacy
#      (el log incluye stacktrace completo del frame que entró al wrapper sync).
#   2. Migrar ese caller a `arun_plan_pipeline` (versión async nativa).
#   3. Tras desplegar la migración y verificar que el log dejó de aparecer 24h,
#      eliminar el override en env y reanudar `True`.
# NO dejar `False` permanente: cada caller async que entra por el path sync
# bloquea el event loop por la duración del pipeline (decenas de segundos),
# degradando latencia de TODAS las requests concurrentes. El strict mode
# `True` es la única defensa contra esa regresión silenciosa.
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
    """[P3-NEW-D · 2026-05-08] Loguea inventario completo desde `_KNOBS_REGISTRY`.

    Antes esta función mantenía un dict hardcoded de ~30 knobs. Con 50+ knobs
    activos en el módulo, drift silencioso era inevitable: un override
    `MEALFIT_LLM_COMBINED_MAX_WAIT_S=60` en producción no aparecía en el log
    si nadie había agregado la entrada al dict, y el operador no podía
    confirmar que el override tomó efecto sin grep manual al código.

    Ahora itera `_KNOBS_REGISTRY` (auto-poblado por `_env_int/_env_float/_env_bool`)
    y emite tres líneas:
      1. Resumen: total de knobs registrados + cuántos tienen override via env.
      2. Highlight de overrides activos (lo que un operador investigando
         producción quiere primero — qué cambió respecto al default).
      3. Inventario completo (grep-able, single-line) para auditoría.

    Si un knob nuevo se añade vía `_env_*`, aparece automáticamente. Cierre
    del gap "drift silencioso" detectado en el audit 2026-05-07.
    """
    log = logging.getLogger(__name__)
    if not _KNOBS_REGISTRY:
        log.info("[KNOBS] Registry vacío — ningún _env_* fue invocado.")
        return

    overrides = sorted(
        ((n, i) for n, i in _KNOBS_REGISTRY.items() if i["is_override"]),
        key=lambda kv: kv[0],
    )
    parse_failures = sorted(
        ((n, i) for n, i in _KNOBS_REGISTRY.items() if i.get("parse_failed")),
        key=lambda kv: kv[0],
    )
    all_sorted = sorted(_KNOBS_REGISTRY.items())

    log.info(
        f"[KNOBS] graph_orchestrator activos: {len(all_sorted)} totales, "
        f"{len(overrides)} overrides via env, {len(parse_failures)} parse-failures."
    )
    if overrides:
        sample = ", ".join(f"{n}={i['value']}" for n, i in overrides)
        log.info(f"[KNOBS/OVERRIDE] {sample}")
    if parse_failures:
        sample = ", ".join(
            f"{n}=raw{i['raw']!r}→default{i['default']}" for n, i in parse_failures
        )
        log.warning(f"[KNOBS/PARSE-FAIL] {sample}")

    full = ", ".join(f"{n}={i['value']}" for n, i in all_sorted)
    log.info(f"[KNOBS/INVENTORY] {full}")


# [P3-KNOBS-INVENTORY-LATE-EMIT · 2026-05-15] La invocación de
# `_log_active_knobs()` se movió al FINAL del módulo (post todas las
# asignaciones module-level de knobs vía `_env_*`). Pre-fix la llamada
# vivía aquí, en línea ~489, ANTES de unos 25-30 knobs declarados
# downstream (e.g. `DAY_GEN_RETRY_USE_PRO` línea ~3527,
# `PROMPT_CACHE_SYSTEM_MESSAGE` línea ~3554, `PROMPT_TRIM_FORM_DATA`
# línea ~3639). Esos knobs SÍ se registraban en `_KNOBS_REGISTRY` al
# completar el import, pero `[KNOBS/INVENTORY]` ya había emitido sin
# verlos → operador post-deploy revisaba el log y aparentaba que sus
# fixes no estaban activos (falsa sensación). El comportamiento en
# runtime NO cambiaba; solo era un gap de observabilidad del startup.
#
# El último statement ejecutable del módulo invoca `_log_active_knobs()`
# para que el snapshot capture el estado FINAL del registry.


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
        # [P2-ORCH-11] Cota de espera: sin esto el busy-poll giraba a 20Hz sin
        # límite bajo Redis-down + saturación, hasta el techo de 720s del pipeline.
        _deadline = time.monotonic() + LLM_LOCAL_MAX_WAIT_S
        while not self._local_semaphore.acquire(blocking=False):
            if time.monotonic() >= _deadline:
                _inc_budget_stat("local_wait_timeout")
                logger.warning(
                    f"🛑 [P2-ORCH-11] Busy-poll local del semáforo GLOBAL excedió "
                    f"LLM_LOCAL_MAX_WAIT_S={LLM_LOCAL_MAX_WAIT_S}s (Redis-down + "
                    f"saturación sostenida). Backpressure fail-fast en vez de girar "
                    f"hasta el timeout global del pipeline."
                )
                raise RuntimeError(
                    "LLM global local-semaphore backpressure: max wait exceeded (P2-ORCH-11)"
                )
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
                # [P3-PROD-AUDIT-3 · 2026-05-30] GC oportunista antes de crear una
                # entrada nueva: si el dict excede el cap (típico de un outage
                # prolongado de Redis con muchos usuarios distintos), purga
                # entradas IDLE — un Semaphore con TODOS sus permits disponibles
                # (`_value >= max_per_user`) no tiene holders activos, así que
                # descartarlo no afecta ningún acquire en vuelo. Una entrada en
                # uso jamás se evicta. Peor caso (race con un holder que ya tomó la
                # ref justo antes): ese usuario tiene 2 sems brevemente — el
                # semáforo es backpressure soft, no un control de seguridad.
                if len(self._local_sync) >= LLM_PER_USER_LOCAL_CACHE_MAX:
                    _evicted = 0
                    for _uid in list(self._local_sync.keys()):
                        _s = self._local_sync.get(_uid)
                        if _s is not None and getattr(_s, "_value", 0) >= self.max_per_user:
                            del self._local_sync[_uid]
                            _evicted += 1
                            if len(self._local_sync) < LLM_PER_USER_LOCAL_CACHE_MAX:
                                break
                    if _evicted:
                        logger.debug(
                            f"[P3-PROD-AUDIT-3] GC cache per-user local sem: purgadas "
                            f"{_evicted} entradas idle (cap={LLM_PER_USER_LOCAL_CACHE_MAX})."
                        )
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
        # [P2-ORCH-11] Misma cota que el semáforo global: fail-fast en vez de
        # girar indefinidamente bajo Redis-down + saturación.
        _deadline = time.monotonic() + LLM_LOCAL_MAX_WAIT_S
        while not sem.acquire(blocking=False):
            if time.monotonic() >= _deadline:
                _inc_budget_stat("local_wait_timeout_user")
                logger.warning(
                    f"🛑 [P2-ORCH-11] Busy-poll local del semáforo PER-USER "
                    f"(user={user_id}) excedió LLM_LOCAL_MAX_WAIT_S="
                    f"{LLM_LOCAL_MAX_WAIT_S}s (Redis-down + saturación). Backpressure "
                    f"fail-fast en vez de girar hasta el timeout global."
                )
                raise RuntimeError(
                    "LLM per-user local-semaphore backpressure: max wait exceeded (P2-ORCH-11)"
                )
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


# [P1-28] Counters de observabilidad del cap combinado per-user → global.
# Permiten a SRE monitorear cuándo el budget se agota y la degradación a
# local kicks in. Si `combined_budget_exceeded` crece sostenidamente,
# señal de que `LLM_COMBINED_MAX_WAIT_S` está mal calibrado (subir) o
# que el provider está saturado (revisar Redis/proveedor).
_LLM_BUDGET_STATS: dict = {
    "combined_budget_exceeded": 0,   # Veces que per-user agotó el budget
                                     # combinado y global se degradó a local.
    "combined_total_warnings": 0,    # Veces que el total elapsed superó
                                     # el budget tras yield (caller hizo
                                     # más wait del esperado).
}
_LLM_BUDGET_STATS_LOCK = threading.Lock()


def _inc_budget_stat(kind: str, n: int = 1) -> None:
    if n <= 0:
        return
    with _LLM_BUDGET_STATS_LOCK:
        _LLM_BUDGET_STATS[kind] = _LLM_BUDGET_STATS.get(kind, 0) + n


def get_llm_budget_stats_snapshot() -> dict:
    """[P1-28] Snapshot read-only de los counters del budget combinado."""
    with _LLM_BUDGET_STATS_LOCK:
        return dict(_LLM_BUDGET_STATS)


@contextmanager
def acquire_user_and_global(user_id):
    """P1-NEW-1: composición sync — adquiere per-user antes que global.

    El orden importa: per-user PRIMERO. Si invertimos, un usuario saturando
    su cuota tomaría slots globales y luego bloquearía esperando per-user,
    desperdiciando slots compartidos.

    [P1-28] Cap COMBINADO de espera. Antes la composición podía esperar
    hasta `LLM_USER_MAX_WAIT_S + LLM_MAX_WAIT_S` porque cada semáforo
    aplicaba su bound independientemente. Ahora medimos elapsed tras
    per-user; si ya gastó >=`LLM_COMBINED_MAX_WAIT_S`, degradamos global
    a su `_local_semaphore` en lugar de esperar otros 90s — el caller
    no pagará 2× wait inesperado.
    """
    _t_start = time.monotonic()
    with PER_USER_LLM_SEMAPHORE.acquire(user_id):
        _elapsed_after_per_user = time.monotonic() - _t_start
        if _elapsed_after_per_user >= LLM_COMBINED_MAX_WAIT_S:
            # [P1-28] Budget agotado: degradar global a local in-process
            # para no extender el wait. El local semaphore mantiene back-
            # pressure intra-proceso (sin ir a Redis), y dado que ya
            # esperamos ≥ combined_max_wait, asumimos que la cola Redis
            # no se vacía pronto y un wait adicional no aporta.
            _inc_budget_stat("combined_budget_exceeded")
            logger.warning(
                f"🟠 [P1-28] Combined budget agotado tras per-user "
                f"({_elapsed_after_per_user:.1f}s ≥ "
                f"{LLM_COMBINED_MAX_WAIT_S}s). Degradando global a "
                f"local sin tocar Redis."
            )
            with LLM_SEMAPHORE._local_semaphore:
                yield
            return
        with LLM_SEMAPHORE.acquire():
            yield


@asynccontextmanager
async def aacquire_user_and_global(user_id):
    """P1-NEW-1: composición async, mismo orden que la versión sync.

    [P1-28] Mismo cap combinado que la versión sync. Ver docstring de
    `acquire_user_and_global` para rationale.
    """
    _t_start = time.monotonic()
    async with PER_USER_LLM_SEMAPHORE.aacquire(user_id):
        _elapsed_after_per_user = time.monotonic() - _t_start
        if _elapsed_after_per_user >= LLM_COMBINED_MAX_WAIT_S:
            _inc_budget_stat("combined_budget_exceeded")
            logger.warning(
                f"🟠 [P1-28] Combined budget agotado tras per-user async "
                f"({_elapsed_after_per_user:.1f}s ≥ "
                f"{LLM_COMBINED_MAX_WAIT_S}s). Degradando global a "
                f"local async sin tocar Redis."
            )
            async with LLM_SEMAPHORE._alocal_acquire():
                yield
            return
        async with LLM_SEMAPHORE.aacquire():
            yield


class ChatDeepSeek(_ChatDeepSeekBase):
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
            _ainvoke_start = time.time()
            result = await super().ainvoke(*args, **kwargs)
            # [P1-COST-INSTRUMENTATION-FIX · 2026-05-15] Captura usage_metadata
            # AQUÍ (en el override del LLM raw) cobre TODOS los paths:
            # `.ainvoke()` directo, `.with_structured_output(...).ainvoke()` (que
            # internamente llama a este override antes del OutputParser),
            # `.bind_tools(...).ainvoke()`, etc. El fallback en `_safe_ainvoke`
            # solo captura cuando el result es AIMessage; aquí captura SIEMPRE
            # antes de cualquier wrapping.
            try:
                _emit_llm_usage_event_best_effort(
                    llm=self,
                    result=result,
                    duration_s=time.time() - _ainvoke_start,
                )
            except Exception:
                pass
            return result

    async def astream(self, *args, **kwargs):
        async with aacquire_user_and_global(user_id_var.get()):
            # [P1-COST-INSTRUMENTATION-FIX · 2026-05-15] Acumula chunks para
            # extraer usage_metadata del mensaje final. Sin esto, day_generator
            # (que usa `.astream()` directamente) no se contabiliza en
            # `llm_usage_events` — perdiendo el nodo MÁS costoso del pipeline.
            _astream_start = time.time()
            _accumulated = None
            async for chunk in super().astream(*args, **kwargs):
                if _accumulated is None:
                    _accumulated = chunk
                else:
                    try:
                        _accumulated = _accumulated + chunk
                    except Exception:
                        pass
                yield chunk
            if _accumulated is not None:
                try:
                    _emit_llm_usage_event_best_effort(
                        llm=self,
                        result=_accumulated,
                        duration_s=time.time() - _astream_start,
                    )
                except Exception:
                    pass

    async def agenerate(self, *args, **kwargs):
        async with aacquire_user_and_global(user_id_var.get()):
            _agen_start = time.time()
            result = await super().agenerate(*args, **kwargs)
            # `agenerate` retorna LLMResult con `generations` — extraer del
            # primer generation el AIMessage que sí tiene usage_metadata.
            try:
                gens = getattr(result, "generations", None) or []
                if gens and gens[0]:
                    msg = getattr(gens[0][0], "message", None)
                    if msg is not None:
                        _emit_llm_usage_event_best_effort(
                            llm=self,
                            result=msg,
                            duration_s=time.time() - _agen_start,
                        )
            except Exception:
                pass
            return result

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

# [P2-ORCH-8 · 2026-05-28] Reconciliación per-user vs PLAN_CHUNK_SIZE. Si el
# operador bajó LLM_MAX_PER_USER por debajo de PLAN_CHUNK_SIZE, el day-gen de un
# usuario correrá serializado en ceil(PLAN_CHUNK_SIZE/LLM_MAX_PER_USER) olas.
# No clampeamos (un valor bajo puede ser intencional por multi-tenancy fairness),
# solo alertamos para que sea decisión consciente y no un default accidental.
if LLM_PER_USER_ENABLED and LLM_MAX_PER_USER < PLAN_CHUNK_SIZE:
    _p2_orch_8_waves = -(-PLAN_CHUNK_SIZE // max(1, LLM_MAX_PER_USER))
    logger.warning(
        f"⚠️ [P2-ORCH-8] LLM_MAX_PER_USER={LLM_MAX_PER_USER} < PLAN_CHUNK_SIZE="
        f"{PLAN_CHUNK_SIZE}: el day-gen de un usuario correrá en ~{_p2_orch_8_waves} "
        f"olas seriales en vez de 1 (latencia extra en el nodo más caro). Subir "
        f"MEALFIT_LLM_MAX_PER_USER a >= {PLAN_CHUNK_SIZE} para paralelismo intra-plan pleno."
    )

from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from db import get_recent_techniques, get_recent_meals_from_plans, check_meal_plan_generated_today, search_user_facts, search_visual_diary, get_user_facts_by_metadata
from nutrition_calculator import get_nutrition_targets

# P1-10: `threading` ya importado al inicio del módulo (línea ~16); no reimportar.
#
# [P2-LOGGER-MIGRATION · 2026-05-12] Logger SSOT del módulo. Tras audit
# 2026-05-12 los 250 `print()` directos se convirtieron a `logger.<level>(...)`
# (info/warning/error según emoji prefix ⚠/❌/🚨). Pre-fix los prints
# escapaban del LogRecord pipeline: no respetaban `LOG_LEVEL`, mezclados con
# trazas del scheduler, sin timestamp consistente del logging framework.
# Production-grade backend NO usa print(). Anchor: P2-LOGGER-MIGRATION.
logger = logging.getLogger(__name__)

from cache_manager import redis_client, redis_async_client
from db_core import execute_sql_query, execute_sql_write, aexecute_sql_query, aexecute_sql_write


# ============================================================
# [P1-BESTEFFORT-DB-CB · 2026-05-21] Circuit breaker LOCAL in-process
# para escrituras DB "best-effort" (LLM-CACHE, CB-RESET, AB-TEMP).
#
# Motivación (incidente 2026-05-21 02:08-02:12):
#   El async pool se saturaba bajo carga normal (3 day_generators + adversarial
#   self-play + meta-learning paralelos) y cada best-effort write se quedaba
#   esperando 8s al timeout del pool. Con 8 callsites consecutivos timeoutean
#   → ~64s acumulados de latencia gastada en operaciones cosméticas.
#
#   Peor: el `LLMCircuitBreaker.arecord_success` también falla en su write a
#   DB → el estado del CB principal queda "confundido" → se abre prematuramente
#   → Días 1 y 2 fallan con `Circuit Breaker OPEN para gemini-3.5-flash` aunque
#   el modelo en sí responde normal.
#
# Diseño:
#   - In-process only (Redis/DB caen contigo, no podemos apoyarnos en ellos).
#   - Per-callsite name (registry singleton).
#   - Thread-safe (callers sync + async).
#   - Auto half-open tras `OPEN_DURATION_S` (default 60s).
#   - Solo cuenta "pool timeout" como failure — otros errores son TOLERADOS
#     (puede ser un schema bug, no significa que el pool esté saturado).
#
# Patrón de uso:
#   _cb = _get_be_db_cb("llm_cache_aget")
#   if _cb.is_open():
#       return default                  # fail-fast, no toca pool
#   try:
#       result = await aexecute_sql_query(...)
#       _cb.record_success()
#       return result
#   except Exception as e:
#       if _is_pool_timeout_error(e):
#           _cb.record_pool_timeout()   # solo timeouts cuentan
#       logger.info(f"...: {e}")
#       return default
#
# Knobs: `MEALFIT_BE_DB_CB_FAILURE_THRESHOLD` (default 3),
#        `MEALFIT_BE_DB_CB_OPEN_DURATION_S` (default 60).
# ============================================================
_BE_DB_CB_FAILURE_THRESHOLD = max(1, int(os.environ.get("MEALFIT_BE_DB_CB_FAILURE_THRESHOLD", "3") or "3"))
_BE_DB_CB_OPEN_DURATION_S = max(5, int(os.environ.get("MEALFIT_BE_DB_CB_OPEN_DURATION_S", "60") or "60"))


class _BestEffortDBCircuitBreaker:
    """CB local para writes best-effort. Ver bloque P1-BESTEFFORT-DB-CB arriba."""

    __slots__ = ("name", "_failure_threshold", "_open_duration_s", "_failures", "_opened_at", "_lock")

    def __init__(self, name: str, failure_threshold: int | None = None, open_duration_s: int | None = None):
        self.name = name
        self._failure_threshold = failure_threshold if failure_threshold is not None else _BE_DB_CB_FAILURE_THRESHOLD
        self._open_duration_s = open_duration_s if open_duration_s is not None else _BE_DB_CB_OPEN_DURATION_S
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        """Retorna True si el CB está OPEN y la cooldown aún no ha expirado."""
        with self._lock:
            if self._failures < self._failure_threshold:
                return False
            if (time.time() - self._opened_at) < self._open_duration_s:
                return True
            # Cooldown expiró → half-open: reset failures para reintentar.
            self._failures = 0
            self._opened_at = 0.0
            return False

    def record_success(self) -> None:
        """Cualquier éxito limpia el contador de fallas."""
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_pool_timeout(self) -> None:
        """Solo timeouts del pool cuentan. Otros errores no abren este CB."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold and self._opened_at == 0.0:
                self._opened_at = time.time()
                logger.info(
                    f"🛑 [BE-DB-CB] {self.name!r} OPEN tras {self._failures} pool-timeouts. "
                    f"Skipeando best-effort writes por {self._open_duration_s}s."
                )

    def snapshot(self) -> dict:
        """Estado actual para diagnostics/tests."""
        with self._lock:
            return {
                "name": self.name,
                "failures": self._failures,
                "is_open": (
                    self._failures >= self._failure_threshold
                    and (time.time() - self._opened_at) < self._open_duration_s
                ),
                "opened_at": self._opened_at,
            }


_BE_DB_CB_REGISTRY: dict[str, "_BestEffortDBCircuitBreaker"] = {}
_BE_DB_CB_REGISTRY_LOCK = threading.Lock()


def _get_be_db_cb(name: str) -> "_BestEffortDBCircuitBreaker":
    """Singleton per name. Thread-safe (registry double-check)."""
    cb = _BE_DB_CB_REGISTRY.get(name)
    if cb is not None:
        return cb
    with _BE_DB_CB_REGISTRY_LOCK:
        cb = _BE_DB_CB_REGISTRY.get(name)
        if cb is None:
            cb = _BestEffortDBCircuitBreaker(name)
            _BE_DB_CB_REGISTRY[name] = cb
        return cb


def _is_pool_timeout_error(exc: Exception) -> bool:
    """Detecta el error específico del pool psycopg cuando no hay conexión
    disponible dentro del timeout configurado. Buscamos por el texto canónico
    `couldn't get a connection after X.XX sec` que psycopg_pool emite.
    """
    msg = str(exc).lower() if exc else ""
    return (
        "couldn't get a connection" in msg
        or "couldn’t get a connection" in msg  # apóstrofe curvo, por las dudas
        or "pool is closed" in msg
        or "pool exhausted" in msg
    )


def _is_transient_upstream_error(exc: BaseException) -> bool:
    """[P1-LLM-TRANSIENT-5XX · 2026-05-21] Detecta errores 5xx transitorios
    de Google que NO deben contar como failure en el LLMCircuitBreaker.

    Bug observado 2026-05-21 02:58:37:
      Google retornó `502 Bad Gateway` en la compresión + planner. El CB
      contó esos 3 retries como fallas → abrió `gemini-3.5-flash` por 30s →
      Días 1/2/3 cayeron con `Circuit Breaker OPEN` aunque el modelo
      principal estaba sano (era infra de Google teniendo un hipo).

    Distinto a `_is_rate_limit_error` (429): los 5xx son **del lado de
    Google** (problemas internos suyos), no del usuario/proyecto. La
    estrategia correcta: backoff + retry SIN contaminar el CB. Por eso
    los excluimos del conteo de failures.

    Cubre 502/503/504/INTERNAL/UNAVAILABLE — los códigos transitorios que
    Google documenta como retryable. Match por string + por attributes
    porque LangChain wrappea estos errores de formas inconsistentes entre
    versiones.

    Tooltip-anchor: P1-LLM-TRANSIENT-5XX.
    """
    try:
        _type_name = type(exc).__name__
        # Excepciones canónicas que documentan transient upstream
        if _type_name in (
            "ServiceUnavailable", "InternalServerError", "GatewayTimeout",
            "DeadlineExceeded", "Aborted", "ServerError", "BadGateway",
        ):
            return True
        _msg = str(exc).lower() if exc else ""
        # HTTP code 502/503/504 en el mensaje (LangChain/genai wrappean así)
        if any(code in f" {_msg} " for code in (" 502 ", " 503 ", " 504 ")):
            return True
        if any(s in _msg for s in ("(502)", "(503)", "(504)", '"code":502', '"code":503', '"code":504')):
            return True
        # gRPC / google.api_core status strings
        if "bad gateway" in _msg or "gateway timeout" in _msg or "service unavailable" in _msg:
            return True
        if "internal" in _msg and ("server error" in _msg or "google" in _msg):
            return True
        if "unavailable" in _msg and ("backend" in _msg or "google" in _msg or "service" in _msg):
            return True
        # google.genai ClientError expone `.code` numérico
        _code = getattr(exc, "code", None)
        if _code in (502, 503, 504):
            return True
        return False
    except Exception:
        return False


# ============================================================
# [P1-ORCH-2 · 2026-05-28] Latch global del spending-cap de Gemini en el
# pipeline de generación (espejo del de shopping_calculator para embeddings).
# ------------------------------------------------------------
# El 429 "spending cap" es PERSISTENTE (activo hasta que el operador suba el
# cap, ~30 días). Sin un latch: cada nodo lo reintenta 3x (tenacity) por día
# paralelo → decenas de intentos facturados inútiles + segundos del budget de
# 720s, y los CB per-modelo se abren espuriamente. Además, cuando el síntoma
# que escapa al handler global NO es el 429 literal (sino "Circuit Breaker
# OPEN" o "Todos los workers fallaron"), el string-match perdía la señal y el
# usuario veía el mensaje falso "IA saturada, intenta en 1-2 min".
#
# El latch lo setea el PRIMER nodo que ve el 429; el handler global lo consulta
# aunque el error que escapó sea un síntoma downstream. Reset por restart del
# proceso o al expirar el window (default 1800s). `=0` desactiva el latch.
# Knob: MEALFIT_PLAN_SPEND_CAP_BACKOFF_S.
# Tooltip-anchor: P1-ORCH-2.
# ============================================================
PLAN_SPEND_CAP_BACKOFF_S = _env_int(
    "MEALFIT_PLAN_SPEND_CAP_BACKOFF_S", 1800,
    validator=lambda v: 0 <= v <= 86400,
)
_plan_spend_cap_until: float = 0.0


def _is_plan_spend_cap_error(exc: BaseException) -> bool:
    """[P1-ORCH-2] True si `exc` es el 429 spending-cap de Gemini. Reusa el
    detector canónico de shopping_calculator (lazy import: sin ciclo)."""
    try:
        from shopping_calculator import _is_gemini_spending_cap_error
        return bool(_is_gemini_spending_cap_error(exc))
    except Exception:
        return False


def _note_plan_spend_cap() -> None:
    """[P1-ORCH-2] Activa el latch global del spending-cap por
    PLAN_SPEND_CAP_BACKOFF_S segundos (no-op si el knob es 0)."""
    global _plan_spend_cap_until
    if PLAN_SPEND_CAP_BACKOFF_S > 0:
        _plan_spend_cap_until = time.time() + PLAN_SPEND_CAP_BACKOFF_S


def _plan_spend_cap_active() -> bool:
    """[P1-ORCH-2] True si el latch del spending-cap sigue activo. Lo consulta
    el handler global para no perder la señal cuando escapa un síntoma."""
    return _plan_spend_cap_until > time.time()


async def _record_cb_failure_unless_transient(cb, exc) -> None:
    """[P1-ORCH-1 · 2026-05-28] Registra un fallo en el CB SALVO que el error
    NO refleje salud del modelo:
      - 5xx transitorio de Google (`_is_transient_upstream_error`) — infra de
        Google teniendo un hipo, no el modelo (incidente 2026-05-21).
      - 429 spending-cap (`_is_plan_spend_cap_error`) — billing persistente;
        reintentar no ayuda y abrir el CB agrava el outage (P1-ORCH-2). Además
        activa el latch global para que el handler detecte el cap aunque escape
        un síntoma downstream.
      - timeout/agotamiento del POOL de DB (`_is_pool_timeout_error`) —
        "couldn't get a connection" refleja salud de la DB/pooler, NO del modelo
        LLM. La DB tiene su PROPIO breaker (`_BestEffortDBCircuitBreaker`). Sin
        esta exclusión, una contención del pool async durante el fact-check/
        reviewer abría el CB del modelo → fail-closed → rechazo "critical" falso
        → se descartaba un plan ya generado (incidente 2026-05-28: pool async
        max=10 saturado, reviewer flash-lite CB OPEN). Tooltip-anchor: P1-ORCH-1-DBPOOL.
    Centraliza el patrón que vivía inline solo en planner/day-gen y lo extiende
    a los ~9 nodos LLM restantes (reviewer/judge/evaluator/fact-checker/etc.).
    Tooltip-anchor: P1-ORCH-1.
    """
    # [P1-ORCH-1-DBPOOL · 2026-05-28] La salud de la DB no es la salud del modelo.
    if _is_pool_timeout_error(exc):
        return
    _spend_cap = _is_plan_spend_cap_error(exc)
    if _spend_cap:
        _note_plan_spend_cap()
    if _spend_cap or _is_transient_upstream_error(exc):
        return
    await cb.arecord_failure()


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
        # [P1-27] Fallback DB ATÓMICO. Antes el path era un read-modify-write
        # multi-step:
        #   1. SELECT value FROM app_kv_store WHERE key = ?
        #   2. state['failures'] += 1; if >= threshold: state['is_open'] = True
        #   3. INSERT ... ON CONFLICT DO UPDATE SET value = (re-serialized state)
        # `self._lock` (threading.Lock) garantizaba atomicidad SOLO dentro del
        # mismo proceso. Bajo Gunicorn `--workers N` cada worker tiene su propia
        # instancia de `LLMCircuitBreaker` con su propio lock — dos workers que
        # registraban fallos concurrentes leían `failures=2` cada uno y
        # escribían `failures=3` (lost-update). Resultado: el threshold se
        # cruzaba con DELAY proporcional al número de workers — el CB no se
        # abría cuando debía, dejando seguir requests contra un proveedor
        # saturado.
        # Ahora la SQL hace el INCR del lado del servidor con jsonb_build_object,
        # garantizando atomicidad cross-worker. Mantenemos el `with self._lock`
        # para evitar thundering herd (N threads del mismo proceso lanzando
        # SQL en paralelo cuando una sola serializada basta), pero el lock ya
        # no es necesario para la corrección — la SQL lo es.
        with self._lock:
            try:
                self._atomic_record_failure_db()
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
                # [P1-27] Reset atómico: una sola UPSERT idempotente reemplaza
                # el patrón SELECT-then-conditional-UPDATE. Sin esto, dos
                # workers que registraban éxito tras una racha de fallos
                # podían leer `failures=5,is_open=true` cada uno y escribir
                # el reset; con SQL atómica el último write gana siempre con
                # el estado correcto y no hay ventana de lectura stale.
                self._atomic_reset_db()
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

    def _atomic_record_failure_db(self) -> None:
        """[P1-27] Increment atómico server-side del contador de fallos.

        Reemplaza el patrón inseguro `SELECT → mutate → UPSERT`. La SQL
        construye el `value` JSON desde el valor actual leído por la
        misma fila (`app_kv_store.value` referenciado en la cláusula DO
        UPDATE), garantizando que el INCR no pierda updates bajo dos
        workers concurrentes — Postgres serializa a nivel de fila el
        ON CONFLICT DO UPDATE.

        - En INSERT (primera falla, no existe la fila): inicializa
          {failures: 1, last_failure: now, is_open: 1 >= threshold}.
        - En UPDATE (fila existe): usa COALESCE((value->>'failures')::int, 0)
          + 1 → resistente a JSON corrupto / key faltante.
        """
        now_ts = time.time()
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (
                %s,
                jsonb_build_object(
                    'failures',     1,
                    'last_failure', %s::float,
                    'is_open',      (1 >= %s::int)
                )
            )
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object(
                    'failures',
                        COALESCE((app_kv_store.value->>'failures')::int, 0) + 1,
                    'last_failure',
                        %s::float,
                    'is_open',
                        (COALESCE((app_kv_store.value->>'failures')::int, 0) + 1)
                            >= %s::int
                ),
                updated_at = NOW()
            """,
            (
                self._db_kv_key,
                now_ts, self.threshold,           # INSERT params
                now_ts, self.threshold,           # UPDATE params
            ),
        )

    def _atomic_reset_db(self) -> None:
        """[P1-27] Reset atómico del estado del CB en DB.

        UPSERT idempotente: en cada llamada deja la fila en
        `{failures: 0, last_failure: 0, is_open: false}`. Equivalente
        funcional al patrón previo `if any: save({…zeros…})` pero sin la
        lectura previa, eliminando la ventana de race entre SELECT y
        UPDATE bajo concurrencia multi-worker.
        """
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, '{"failures": 0, "last_failure": 0, "is_open": false}'::jsonb)
            ON CONFLICT (key) DO UPDATE SET
                value = '{"failures": 0, "last_failure": 0, "is_open": false}'::jsonb,
                updated_at = NOW()
            """,
            (self._db_kv_key,),
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
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] Mismo gate que arecord_success.
        # Si el pool está saturado, no perdamos otros 8-12s intentando registrar
        # una falla cuando el LLMCircuitBreaker principal ya tiene el estado
        # propagado in-memory via `_local_healthy=False`. La info se reescribirá
        # cuando el pool vuelva a estar disponible.
        _be_cb = _get_be_db_cb("llm_cb_failure_async")
        if _be_cb.is_open():
            return
        async with self._alock_acquire():
            try:
                # [P1-27] Ver `record_failure` para rationale de atomicidad.
                # Mismo patrón sync: SQL UPSERT con INCR server-side.
                await self._aatomic_record_failure_db()
                _be_cb.record_success()
            except Exception as e:
                if _is_pool_timeout_error(e):
                    _be_cb.record_pool_timeout()
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
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] Gate: si los últimos 3 CB-RESETs
        # timeoutearon en el pool, skipea por 60s. Los próximos record_success
        # reintentarán automáticamente tras la cooldown. Cierre del root cause
        # del incidente 2026-05-21 02:08-02:12 donde `DB async CB write error`
        # bloqueaba 8s × N callsites consecutivos.
        _be_cb = _get_be_db_cb("llm_cb_reset_async")
        if _be_cb.is_open():
            # Best-effort skipped silently; el local_healthy ya marca al CB
            # principal como sano en memoria.
            with self._local_state_lock:
                self._local_healthy = True
                self._last_db_check = time.time()
            return
        async with self._alock_acquire():
            try:
                # [P1-27] Reset atómico async (ver `record_success` sync).
                await self._aatomic_reset_db()
                _be_cb.record_success()
                with self._local_state_lock:
                    self._local_healthy = True
                    self._last_db_check = time.time()
            except Exception as e:
                if _is_pool_timeout_error(e):
                    _be_cb.record_pool_timeout()
                # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info y wording
                # ajustado: el CB reset es BEST-EFFORT (idempotente, el próximo
                # success reintentará). Pre-fix decía "error" + nivel WARNING →
                # el operador (y el user mirando logs) creía que el plan estaba
                # fallando cuando en realidad solo se perdió 1 escritura de
                # observabilidad. La generación procede sin issue.
                logger.info(f"[CB-RESET] DB async no disponible (best-effort, no afecta plan): {e}")

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

    async def _aatomic_record_failure_db(self) -> None:
        """[P1-27] Versión async de `_atomic_record_failure_db`.

        Mismo SQL, ejecutado vía `aexecute_sql_write`. Ver el método sync
        para el rationale de atomicidad cross-worker.
        """
        now_ts = time.time()
        await aexecute_sql_write(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (
                %s,
                jsonb_build_object(
                    'failures',     1,
                    'last_failure', %s::float,
                    'is_open',      (1 >= %s::int)
                )
            )
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object(
                    'failures',
                        COALESCE((app_kv_store.value->>'failures')::int, 0) + 1,
                    'last_failure',
                        %s::float,
                    'is_open',
                        (COALESCE((app_kv_store.value->>'failures')::int, 0) + 1)
                            >= %s::int
                ),
                updated_at = NOW()
            """,
            (
                self._db_kv_key,
                now_ts, self.threshold,
                now_ts, self.threshold,
            ),
        )

    async def _aatomic_reset_db(self) -> None:
        """[P1-27] Versión async de `_atomic_reset_db`. UPSERT idempotente."""
        await aexecute_sql_write(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, '{"failures": 0, "last_failure": 0, "is_open": false}'::jsonb)
            ON CONFLICT (key) DO UPDATE SET
                value = '{"failures": 0, "last_failure": 0, "is_open": false}'::jsonb,
                updated_at = NOW()
            """,
            (self._db_kv_key,),
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
# Solución: pool dedicado. Limita la concurrencia de fact-checks
# globalmente sin afectar el resto de operaciones to_thread. Si llegan 5 pipelines
# simultáneos, los primeros 2 (default) corren, los otros 3 se encolan en este
# executor (no en el pool default), preservando throughput de operaciones más
# rápidas.
#
# [P1-31] `max_workers` ahora viene del knob `FACT_CHECK_POOL_SIZE`
# (`MEALFIT_FACT_CHECK_POOL_SIZE`, default 2). Antes era hardcoded a 2 — un
# operador que detectaba backlog del fact-checker bajo carga no podía subir
# la concurrencia sin redeploy con código modificado. Ahora se tunea por env,
# preservando default histórico.
#
# P1-Q9: variante drainable. Bajo SIGTERM (rolling deploy, autoscaling), antes
# se cancelaban abruptamente los fact-checks en vuelo dejando sockets HTTP
# half-closed con el provider clínico. Ahora `_shutdown_drainable_executor`
# espera hasta `FACT_CHECK_SHUTDOWN_DRAIN_S` (~8s default) para que la mayoría
# complete y cierre sockets gracefully.
_FACT_CHECK_EXECUTOR = _DrainableThreadPoolExecutor(
    max_workers=FACT_CHECK_POOL_SIZE,  # [P1-31]
    thread_name_prefix="fact-check", name="fact-check"
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
    max_workers=DB_EXECUTOR_MAX_WORKERS,  # [P2-ORCH-13] knob MEALFIT_DB_EXECUTOR_MAX_WORKERS (era 8 hardcoded)
    thread_name_prefix="db-io", name="db-io"
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


def _submit_best_effort_metric(fn, **kwargs):
    """[P3-GENCHUNK-SPEED · 2026-06-01] Despacha un emit best-effort de
    métrica/usage-event (`_emit_llm_usage_event_best_effort` /
    `_emit_llm_timeout_metric`) al `_METRICS_EXECUTOR` para que el INSERT
    bloqueante a DB NO corra en el event loop.

    Pre-fix: estos emits (P1-COST-INSTRUMENTATION / P1-LLM-TIMEOUT-METRICS)
    corrían SÍNCRONOS dentro de `_safe_ainvoke`, serializando el loop con un
    round-trip psycopg por CADA llamada LLM exitosa. Peor bajo day-gen
    concurrente: N `_safe_ainvoke` resolviendo casi-simultáneo, cada uno
    bloqueando el loop por turnos antes de que los hermanos pudieran resolver.
    Es el mismo antipatrón sync-DB-on-loop que P0-NEW-1 eliminó de ~15
    callsites; este se añadió después y se escapó.

    Snapshot del contexto del caller (`copy_context`, patrón P1-X3 / L3599)
    preserva los ContextVars (request_id/user_id/`_current_node_var`) en el
    worker — el helper de usage-event lee `_current_node_var` como fallback de
    nodo, así que el snapshot mantiene el tag correcto. Fallback inline si el
    submit falla (pool en shutdown): la señal es best-effort pero no la
    perdemos gratis. Tooltip-anchor: P3-GENCHUNK-SPEED-METRIC-OFFLOAD.
    """
    try:
        ctx = contextvars.copy_context()
        _METRICS_EXECUTOR.submit(ctx.run, lambda: fn(**kwargs))
    except Exception:
        try:
            fn(**kwargs)
        except Exception:
            pass


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
    _safe_ainvoke_started_at = time.time()
    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        # [P1-COST-INSTRUMENTATION · 2026-05-15] Emit financial accounting
        # tick a `llm_usage_events`. Best-effort: cualquier fallo se silencia
        # para no enmascarar el response exitoso al caller. Sin esto, el
        # sistema queda ciego al costo real por modelo/nodo y no se pueden
        # tomar decisiones de optimización (context caching, retry budgets).
        # [P3-GENCHUNK-SPEED · 2026-06-01] OFF-LOOP: el emit hace un INSERT
        # psycopg BLOQUEANTE; correrlo síncrono congelaba el event loop por
        # cada LLM call exitosa (peor bajo day-gen concurrente). Resolvemos el
        # nodo desde el ContextVar AHORA (ventana set por el caller aún activa,
        # p.ej. self_critique_correction L7846+) y lo pasamos explícito para no
        # depender de leerlo en el worker; `copy_context` adentro del helper es
        # belt-and-suspenders para los demás ContextVars.
        try:
            _emit_node = _current_node_var.get()
        except Exception:
            _emit_node = None
        _submit_best_effort_metric(
            _emit_llm_usage_event_best_effort,
            llm=llm,
            result=result,
            duration_s=time.time() - _safe_ainvoke_started_at,
            node=_emit_node,
        )
        return result
    except asyncio.TimeoutError:
        # P0-4: timeout duro. Cancelar la task interna y esperar a que libere
        # el socket antes de propagar al caller.
        task.cancel()
        await _swallow_cancelled_task(task)
        # [P1-LLM-TIMEOUT-METRICS · 2026-05-15] Emit signal estructurada
        # a `pipeline_metrics`. ANTES, los timeouts del LLM (que afectan
        # ~30%+ de planes fallidos en incidentes de cuota Gemini) solo
        # quedaban en logs. Los logs rotan; sin métrica persistente, post-
        # hoc no se puede graficar "cuántos planes timeoutearon hoy por
        # modelo X". El emit es best-effort (try/except silencioso) — un
        # fallo de DB no debe enmascarar el TimeoutError original al caller.
        # [P3-GENCHUNK-SPEED · 2026-06-01] OFF-LOOP por simetría con el success
        # path (este es el error path, raro, pero el INSERT igual bloquearía).
        _submit_best_effort_metric(
            _emit_llm_timeout_metric,
            node="_safe_ainvoke_timeout",
            timeout_threshold_s=timeout,
            actual_wait_s=time.time() - _safe_ainvoke_started_at,
            llm=llm,
        )
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


def _emit_llm_timeout_metric(
    *,
    node: str,
    timeout_threshold_s: float,
    actual_wait_s: float,
    llm=None,
    extra_metadata: dict | None = None,
) -> None:
    """[P1-LLM-TIMEOUT-METRICS · 2026-05-15] Helper SSOT para emitir tick a
    `pipeline_metrics` en TimeoutError catches del LLM.

    Best-effort: cualquier fallo de DB se silencia para no enmascarar el
    TimeoutError original al caller. Extrae model name del llm (si tiene
    `.model` attr — ChatGoogleGenerativeAI lo expone) sin lanzar.
    """
    try:
        model_name = None
        if llm is not None:
            for attr in ("model", "model_name", "_model"):
                try:
                    cand = getattr(llm, attr, None)
                    if isinstance(cand, str) and cand:
                        model_name = cand
                        break
                except Exception:
                    continue
        meta = {
            "timeout_threshold_s": float(timeout_threshold_s),
            "actual_wait_s": float(actual_wait_s),
            "model": model_name,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (NULL, NULL, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (node, int(actual_wait_s * 1000), json.dumps(meta, ensure_ascii=False)),
        )
    except Exception as _e_metric:
        try:
            logger.debug(
                f"[P1-LLM-TIMEOUT-METRICS] emit falló (best-effort): {_e_metric!r}"
            )
        except Exception:
            pass


# [P2-ORCH-14 · 2026-05-28] Idempotencia del emit de usage-events. La misma
# AIMessage puede pasar por DOS capas de emit: el override del LLM raw
# (`ChatGoogleGenerativeAI.ainvoke/astream`, ~1264) Y `_safe_ainvoke` (~2737).
# Sin guard, los callers RAW (compressor/fact_checker que pasan el subclass crudo
# a `_safe_ainvoke`) doble-contarían el costo. La única razón de que hoy no
# doble-cuenten es incidental (los structured runnables no exponen `.model`).
# Stamp per-result (atributo o WeakSet de fallback) → emit exactamente una vez por
# objeto-resultado real. Tooltip-anchor: P2-ORCH-14.
_USAGE_EMIT_SEEN: "weakref.WeakSet" = weakref.WeakSet()


def _usage_was_emitted(result) -> bool:
    """[P2-ORCH-14] True si `result` ya fue contabilizado (no re-emitir)."""
    try:
        if getattr(result, "_mealfit_usage_emitted", False):
            return True
    except Exception:
        pass
    try:
        return result in _USAGE_EMIT_SEEN
    except Exception:
        return False


def _mark_usage_emitted(result) -> None:
    """[P2-ORCH-14] Marca `result` como ya-contabilizado. Intenta atributo
    (bypass de frozen vía object.__setattr__); si falla, WeakSet de identidad."""
    try:
        object.__setattr__(result, "_mealfit_usage_emitted", True)
        return
    except Exception:
        pass
    try:
        _USAGE_EMIT_SEEN.add(result)
    except Exception:
        # Objeto no weakref-able/hashable → no se puede trackear; mejor permitir
        # un emit (best-effort) que perder la fila por completo.
        pass


def _emit_llm_usage_event_best_effort(*, llm, result, duration_s: float, node: str = None) -> None:
    """[P1-COST-INSTRUMENTATION · 2026-05-15] Extrae model + usage_metadata
    de un response exitoso de LangChain ChatGoogleGenerativeAI y persiste
    a `llm_usage_events` via `db_profiles.log_llm_usage_event`.

    Best-effort wrapper: cualquier fallo (parse, DB) se silencia. No
    enmascara el response exitoso al caller.

    Acceso a usage_metadata es defensivo — LangChain expone:
      - `result.usage_metadata` (dict con input_tokens/output_tokens/total_tokens)
      - `result.response_metadata.usage_metadata` (fallback path en algunas
        versiones del SDK).
    Cached tokens vienen como `cached_content_token_count` (Gemini) o
    `input_token_details.cache_read` (LangChain canonical).

    [P3-CHAT-NODE-EXPLICIT · 2026-05-20] `node` ahora aceptable como kwarg
    explícito. Pre-fix: el helper solo resolvía desde `_current_node_var`
    (ContextVar). El chat-flow NO setea ese var → todas sus filas iban
    con `node=NULL` → SRE no podía filtrar costos chat vs plan-gen.
    Caller del chat ahora pasa `node='chat_call_model'` explícito; el
    ContextVar sigue siendo fallback para callsites del pipeline plan-gen
    que ya lo gestionan.
    """
    try:
        # [P2-ORCH-14] Idempotencia: si esta result ya fue contabilizada por otra
        # capa de emit (override raw vs _safe_ainvoke), abortar para no doble-contar.
        if result is not None and _usage_was_emitted(result):
            return
        model_name = None
        for attr in ("model", "model_name", "_model"):
            try:
                cand = getattr(llm, attr, None)
                if isinstance(cand, str) and cand:
                    model_name = cand
                    break
            except Exception:
                continue
        if not model_name:
            return

        usage = None
        try:
            usage = getattr(result, "usage_metadata", None)
        except Exception:
            usage = None
        if not usage:
            try:
                resp_meta = getattr(result, "response_metadata", None) or {}
                usage = resp_meta.get("usage_metadata") if isinstance(resp_meta, dict) else None
            except Exception:
                usage = None
        if not usage or not isinstance(usage, dict):
            return

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cached_tokens = 0
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cache_read") or details.get("cached") or 0
        if not cached_tokens:
            cached_tokens = usage.get("cached_content_token_count", 0) or 0

        # [P1-COST-INSTRUMENTATION-PHASE2 · 2026-05-16] Inyecta el nombre
        # del nodo desde el ContextVar `_current_node_var`. Si la llamada
        # ocurrió fuera de un nodo etiquetado (e.g. agent tools, scripts
        # admin), `node` queda None → fila DB con `node=NULL` (legacy phase 1).
        # [P3-CHAT-NODE-EXPLICIT · 2026-05-20] Si el caller pasó `node`
        # explícito (kwarg), úsalo — tiene prioridad sobre el ContextVar.
        # El chat-flow (agent.py:call_model) lo pasa como 'chat_call_model'
        # porque ese flow NO setea el ContextVar.
        if node:
            current_node = node
        else:
            try:
                current_node = _current_node_var.get()
            except Exception:
                current_node = None

        from db_profiles import log_llm_usage_event
        # [P2-ORCH-14] Marcar ANTES del log (una sola fila por objeto-resultado).
        if result is not None:
            _mark_usage_emitted(result)
        log_llm_usage_event(
            model=model_name,
            node=current_node,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=int(cached_tokens) if cached_tokens else 0,
            metadata={"duration_s": round(float(duration_s), 3)},
        )
    except Exception as _e_emit:
        try:
            logger.debug(
                f"[P1-COST-INSTRUMENTATION] emit usage event falló "
                f"(best-effort): {_e_emit!r}"
            )
        except Exception:
            pass


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
            # [P0-4] Antes: `interval '%s seconds'` con `params=(key, self.ttl)`.
            # Psycopg NO sustituye `%s` dentro de literales SQL — el parser solo
            # cuenta UN placeholder real (`key = %s`) pero recibimos DOS params,
            # disparando `ProgrammingError` SIEMPRE. Resultado: cuando Redis
            # cae, el fallback DB del cache LLM tiene 100% miss garantizado y
            # cada lectura paga el costo del query + parse error silenciado.
            # Fix: `make_interval(secs => %s)` parametriza correctamente.
            res = execute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - make_interval(secs => %s)",
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
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] CB local: si el pool tuvo
        # ≥3 timeouts seguidos, skipea esta call los próximos 60s en lugar
        # de gastar 8-12s del pool timeout en una operación cosmética.
        _be_cb = _get_be_db_cb("llm_cache_aget")
        if _be_cb.is_open():
            return default
        try:
            # [P0-4] Ver comentario equivalente en `get` arriba — mismo bug,
            # mismo fix vía `make_interval(secs => %s)`.
            res = await aexecute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - make_interval(secs => %s)",
                (key, self.ttl), fetch_one=True
            )
            _be_cb.record_success()
            if res:
                return res["value"] if isinstance(res["value"], list) or isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception as e:
            if _is_pool_timeout_error(e):
                _be_cb.record_pool_timeout()
            # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info: cache MISS
            # por pool saturado es BEST-EFFORT (el LLM call procede normal sin
            # cache hit; solo perdemos la optimización de evitar 1 prompt
            # duplicado). El plan NO falla por esto.
            logger.info(f"[LLM-CACHE] DB async miss (best-effort, LLM call procede): {e}")
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
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] Mismo gate que aget: fail-fast
        # cuando pool está saturado.
        _be_cb = _get_be_db_cb("llm_cache_aset")
        if _be_cb.is_open():
            return
        try:
            await aexecute_sql_write(
                "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, json.dumps(value))
            )
            _be_cb.record_success()
        except Exception as e:
            if _is_pool_timeout_error(e):
                _be_cb.record_pool_timeout()
            # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info: cache SET
            # fallido es BEST-EFFORT (perdemos opt de cachear este prompt
            # para próximas calls; no afecta el plan actual).
            logger.info(f"[LLM-CACHE] DB async set skipped (best-effort, no afecta plan): {e}")

# Brecha 1 Fix: Estado Persistente (Redis / DB)
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

    # [P0-PIPE-1] Snapshot del MEJOR intento por severidad de rechazo.
    # ------------------------------------------------------------
    # Antes el orquestador entregaba SIEMPRE el plan del último intento al
    # usuario, incluso cuando era CATASTRÓFICAMENTE peor que un intento
    # previo. Caso real (incidente 2026-05-05):
    #   - Intento #1: rechazo `minor` (pavo procesado, camarones repetidos)
    #     — plan estructuralmente válido pero con observaciones no-críticas.
    #   - Intento #2: rechazo `high` (3 días con skeleton fidelity rotos +
    #     ingredientes que no aparecen en `ingredients_raw`) — plan inválido.
    #   - `should_retry` retornaba "end" → END del grafo → arun retornaba
    #     `final_state["plan_result"]` (intento #2 roto) al cliente.
    #
    # Ahora `review_plan_node` mantiene un snapshot del intento con MENOR
    # severidad (rank: approved < minor < high < critical). Tras el grafo,
    # `_swap_to_best_attempt_if_better` (en `arun_plan_pipeline`) compara
    # current vs best y restaura el best si es mejor — ANTES de aplicar los
    # guardrails de critical/transparency, para que el cliente reciba el
    # plan de mayor calidad disponible.
    #
    # Costo en memoria: 1 deepcopy de plan_result (~50KB por intento).
    # Despreciable comparado con el contexto LLM ya en memoria. La política
    # "preservar el menos roto" es la red de seguridad que P0-1 transparency
    # no provee (transparency solo MARCA el plan; aquí lo SUSTITUIMOS).
    _best_attempt_plan: Optional[dict]
    _best_attempt_severity: Optional[str]
    _best_attempt_reasons: list
    _best_attempt_review_passed: bool
    _best_attempt_number: Optional[int]

    # [P5-MARKER-APPROVED-1] Flag de re-entrada para `surgical_marker_regen_node`.
    # Cuando el reviewer aprueba un plan que tiene `_critique_unresolved` markers
    # en algún día (caso real 2026-05-05 03:54: Día 2 timeoutó la corrección de
    # slot-coherence, reviewer aprobó porque su lente es médico, plan fue al
    # usuario con repetición almuerzo↔cena sin resolver), `should_retry` enruta
    # a `surgical_marker_regen_node` que re-corrige solo esos días con budget
    # fresco. Este flag previene re-entrada infinita: una sola pasada por gate.
    # Reseteado en `retry_reflection_node` para que cada attempt nuevo tenga
    # su propia oportunidad de surgical regen.
    _marker_regen_attempted: bool

    # ============================================================
    # [P2-CANDIDATE-A · 2026-05-08] CONTRATO: claves que NO viven en PlanState.
    # ------------------------------------------------------------
    # Las siguientes claves persisten dentro de `plan_result` (el dict
    # `Optional[dict]` declarado arriba), NO como campos top-level del state.
    # LangGraph strict-schema NO las filtra porque viajan dentro del payload
    # opaco de `plan_result`; pero un futuro refactor que las migrara al nivel
    # del state sin declararlas aquí las perdería silenciosamente entre nodos.
    # Si alguien necesita acceso state-level a cualquiera de estas keys, debe:
    #   1. Añadir el campo a este TypedDict.
    #   2. Inicializarlo en `arun_plan_pipeline` (igual que `_token_buffers`).
    #   3. Migrar los call sites de `result["..."]`/`plan["..."]` a `state[...]`.
    # No basta con escribir `state["nueva_key"] = ...` desde un nodo.
    #
    # Claves vivas en `plan_result` (no en state):
    #
    # COHERENCE (recetas↔lista de compras):
    #   - `_shopping_coherence_block`         (lista crítica de divergencias;
    #     escrita en `assemble_plan_node` ~6975, leída en `review_plan_node`
    #     ~7972; persistida a DB vía `plan_data` cuando `BLOCK_ACTION=degrade`).
    #   - `_shopping_coherence_block_history` (cap 20; escrita ~7000/8019/7752,
    #     leída ~6964/7994/8026; análisis post-mortem P3-NEW-C).
    #     Cada entry tiene `action_taken` ∈ {not_applicable, hydration_error,
    #     degrade, reject_minor, reject_high, post_swap_revalidation};
    #     `history[-1].action_taken` es la fuente única del último resultado
    #     del consumer (NO existe key top-level espejo — drift documentado y
    #     removido en P2-A 2026-05-08). El valor `post_swap_revalidation`
    #     (P2-B 2026-05-08) marca entries emitidas por
    #     `_recompute_aggregates_after_swap` con `swap=True` en modo warn —
    #     telemetría pura post-review, NO indica error.
    #   - `_recipe_coherence_errors`           (lista de errores de coherencia
    #     intra-receta detectados durante validation; consumida por el flujo
    #     de review para decidir severidad).
    #
    # PANTRY (nevera ↔ ingredientes):
    #   - `_pantry_supplement_required`       (categoría 🚨 Compra Urgente
    #     — se escribe a `meal_plans.plan_data` directo en `cron_tasks.py`,
    #     nunca pasa por el state del grafo).
    #
    # CHUNK / PLAN-LEVEL MARKERS:
    #   - `_critique_unresolved`, `_merged_chunk_ids`, `_user_forced_simplified_weeks`
    #     (markers per-day o per-plan persistidos en `days[*]` o `plan_data`
    #     respectivamente; no son state-level).
    #
    # [P1-NEW-1 · 2026-05-11] Sección añadida tras audit 2026-05-11:
    # las siguientes keys ALSO viajan en `plan_result` y NO estaban
    # documentadas — un refactor que las migre a state-level sin
    # declararlas perdería su valor entre nodos (mismo bug class P1-G).
    #
    # RETRY / BEST-ATTEMPT FALLBACK (cuando un retry falla, se entrega
    # el mejor attempt previo en lugar de un fallback matemático):
    #   - `_best_attempt_number`              (índice del attempt elegido).
    #   - `_best_attempt_plan`                (snapshot del plan ganador).
    #   - `_best_attempt_reasons`             (motivos de rechazo registrados).
    #   - `_best_attempt_review_passed`       (bool — review había pasado).
    #   - `_best_attempt_severity`            (severidad del rechazo previo).
    #
    # REVIEW PATH (cuando el plan se entrega con flags al usuario):
    #   - `_review_disclaimer`                (texto explicativo para UI).
    #   - `_review_issues`                    (lista de issues del reviewer).
    #   - `_review_severity`                  (severity ∈ minor/high/critical).
    #   - `_is_fallback`                      (marca planes de fallback
    #                                           matemático tras retries fallidos).
    #
    # SCHEMA VALIDATION:
    #   - `_schema_errors`                    (lista de errores Pydantic).
    #   - `_schema_invalid`                   (bool — fallo de validación).
    #
    # SKELETON (fidelidad estructural pre-day-generation):
    #   - `_skeleton`                         (skeleton del plan, base de days).
    #   - `_skeleton_fidelity_errors`         (lista de divergencias detectadas).
    #
    # MISC HELPERS:
    #   - `_profile_embedding`                (embedding del usuario, usado
    #                                           por semantic cache + RAG).
    #   - `_selected_techniques`              (técnicas culinarias inyectadas
    #                                           al prompt; preservar entre attempts).
    #
    # Si una migración futura (ej. mover coherence handling a un nodo dedicado
    # `coherence_arbiter_node`) requiere visibilidad state-level, declarar el
    # campo en este TypedDict ANTES de tocar el call site — el bug equivalente
    # a P1-G (`_token_buffers` filtrado por strict-schema) se reproduciría.
    #
    # Tests de drift:
    #   - `test_p3_audit_7_plan_result_keys_contract.py` — productor+consumer
    #     para CADA key listada.
    #   - `test_p1_new_1_plan_result_contract_completeness.py` — inverso:
    #     CADA key `_xxx` que aparezca en `plan_result["..."]`/`result["..."]`
    #     del código de producción DEBE estar en este bloque CONTRATO.
    #   - `test_p3_new_8_plan_result_contract_runtime_validator.py` — valida
    #     que `_ensure_plan_result_contract` está cableado al return final
    #     del pipeline + chequea los rangos canónicos en runtime.
    # ============================================================


def _ensure_plan_result_contract(plan_result, *, source: str = "unknown") -> None:
    """[P3-NEW-8 · 2026-05-11] Validador defensivo runtime del contrato
    del `plan_result` documentado en el bloque CONTRATO arriba.

    Por qué existe:
      El test parser-based `test_p1_new_1_plan_result_contract_completeness`
      ya enforza el contrato a CI: cada key `_xxx` en producción DEBE
      estar documentada en el bloque. Pero eso es post-hoc — un refactor
      que mete drift de TIPO/RANGO (ej. `_review_severity="medium"` en
      lugar de uno del set canónico, o `days=dict` en lugar de list) no
      lo atrapa porque la KEY sigue documentada.

      Este helper cierra ese gap: corre justo antes del `return` final
      del pipeline (`arun_plan_pipeline`) y emite WARNING para cada
      violación de rango. Log-only, NEVER raise — un guard runtime que
      crashea el pipeline por type drift sería peor que el drift mismo.

    Validaciones (subset deliberadamente pequeño — solo las keys con
    valor enumerado o tipo estructural fijo. Las keys de string libre
    `_review_disclaimer`/`_schema_errors` no se validan porque no hay
    contrato de rango):

      - `_review_severity` ∈ {"minor", "high", "critical", None}.
      - `_is_fallback` ∈ {True, False, None}.
      - `_review_failed_but_delivered` ∈ {True, False, None}.
      - `days` is list (or None/absent).
      - `_skeleton_fidelity_errors` is list (or None/absent).
      - `_schema_errors` is list-or-str (or None/absent).

    Args:
        plan_result: el dict a validar (puede ser None — log y return).
        source: hint para el log del callsite (e.g., "arun_plan_pipeline_return").

    Returns: None — side-effect-only via logger.warning.

    Idempotente, sin side-effects en `plan_result` (NO mutates).
    """
    if plan_result is None:
        logger.warning(f"[P3-NEW-8/CONTRACT] plan_result=None (source={source})")
        return
    if not isinstance(plan_result, dict):
        logger.warning(
            f"[P3-NEW-8/CONTRACT] plan_result no es dict "
            f"(source={source}, tipo={type(plan_result).__name__})"
        )
        return

    _SEVERITY_SET = {"minor", "high", "critical"}
    sev = plan_result.get("_review_severity")
    if sev is not None and sev not in _SEVERITY_SET:
        logger.warning(
            f"[P3-NEW-8/CONTRACT] _review_severity={sev!r} fuera del set "
            f"canónico {_SEVERITY_SET} (source={source})"
        )

    for bool_key in ("_is_fallback", "_review_failed_but_delivered"):
        val = plan_result.get(bool_key)
        if val is not None and not isinstance(val, bool):
            logger.warning(
                f"[P3-NEW-8/CONTRACT] {bool_key}={val!r} no es bool "
                f"(source={source}, tipo={type(val).__name__})"
            )

    for list_key in ("days", "_skeleton_fidelity_errors", "_review_issues"):
        val = plan_result.get(list_key)
        if val is not None and not isinstance(val, list):
            logger.warning(
                f"[P3-NEW-8/CONTRACT] {list_key} no es lista "
                f"(source={source}, tipo={type(val).__name__})"
            )



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
    build_minimal_correction_context,
    build_correction_context,
    build_unified_behavioral_profile,
    build_rag_context,
    # [P0-FORM-3] Inyecta motivación personal del usuario al planner + day generator.
    build_motivation_context,
    # [P2-AUDIT-5 · 2026-05-10] Hints fisiológicos/emocionales (sleepHours/stressLevel).
    # Antes ambos campos vivían en _REQUIRED_FORM_FIELDS sin consumer downstream —
    # promesa rota del wizard. Ahora se inyectan como hint de tono/sesgo al planner.
    build_sleep_stress_context,
    # [P3-CONDITION-RULES · 2026-06-14] Directivas DM2 (ADA 2026) / ERC (KDIGO 2024) al generador.
    build_medical_condition_context,
    build_time_context,
    build_technique_injection,
    build_supplements_context,
    # [BUDGET-CUSTOM · 2026-05-31] Presupuesto categórico + monto custom RD$ → LLM.
    build_budget_context,
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
    build_pantry_drift_context,
)
from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT
from prompts.planner import PLANNER_SYSTEM_PROMPT
from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_assignment_context


# ============================================================
# HELPERS: Selección de técnicas y contextos compartidos
# ============================================================

def _flatten_ingredient(ing) -> str:
    """Asegura que un ingrediente sea un string plano, incluso si el LLM devuelve una lista anidada.

    [P6-HTML-ENTITY-FIX] Decodifica HTML entities (&oacute;→ó, &iacute;→í,
    &eacute;→é, &ntilde;→ñ, etc.) para evitar items duplicados en aggregate.
    Bug observable PDF 2026-05-05 21:02 ([775ce092]): el corrector LLM emitió
    `'120g de mel&oacute;n'` en algunos días y `'100g de melón'` en otros →
    aggregator los trató como ingredientes DIFERENTES → lista de compras
    mostró 'Melón' Y 'Mel&oacute;n' como entradas separadas, igual para
    'Orégano dominicano' y 'Yautía blanca' (rota la consolidación + display
    al usuario muestra `Yaut&iacute;a blanca` literal).

    `html.unescape` es idempotente (no rompe strings sin entities).
    """
    import html
    if isinstance(ing, list):
        s = " ".join(str(x) for x in ing)
    else:
        s = str(ing) if ing is not None else ""
    return html.unescape(s)

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


async def _drain_cancelled_progress_task(t) -> None:
    """[P1-29] Drena una task de progreso cancelada por el cap blando.

    `task.cancel()` solo SCHEDULES la cancelación (set un flag); la task
    sigue ejecutando hasta su próximo checkpoint async. Si la task estaba
    bloqueada en un callback colgado o en una operación I/O larga, sus
    referencias internas (closures, ContextVars copiados, payload del
    evento) permanecen vivas mientras la cancelación no se propaga
    realmente.

    Este drain `await t` con manejo de excepción asegura que:
      - La task termina antes de que el drainer mismo complete.
      - Excepciones (CancelledError o cualquier otra del cb) se silencian
        — ya fueron contabilizadas por `_run_async_cb_safe` /
        `_run_sync_cb_safe`.
      - El GC del cb + payload ocurre prontamente, sin esperar al ciclo
        completo del event loop.

    Diseñado para fire-and-forget: el caller (`_register_progress_task`)
    crea esta drain task pero NO la awaita — su único propósito es
    asegurar que el `await t` ocurra en algún checkpoint próximo.
    """
    try:
        await t
    except BaseException:
        # CancelledError, Exception del cb, o cualquier cosa: silenciar.
        # `_run_*_cb_safe` ya registró failed_async/failed_sync.
        pass


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

    [P1-29] Cuando el cap blando dispara y cancelamos tasks viejas,
    schedularizamos un drainer (`_drain_cancelled_progress_task`) por cada
    una. Sin esto, `task.cancel()` quedaba fire-and-forget: la task
    cancelada sigue retenida hasta su próximo checkpoint async + el
    procesamiento de la CancelledError. Si el cb estaba bloqueado en una
    op larga, la memoria del payload + ContextVars copiados quedaba viva
    minutos. El drainer fuerza el `await t` en un checkpoint próximo, así
    el cleanup se materializa rápido.
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
            # [P1-29] Drainer fire-and-forget que `await old_t` para que la
            # cancelación se propague y el GC libere refs prontamente.
            try:
                loop.create_task(_drain_cancelled_progress_task(old_t))
            except RuntimeError:
                # Loop cerrado: la cancelación quedará pendiente hasta GC.
                # No es bloqueante para el resto del cap eviction.
                pass
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


def _flush_all_token_buffers(state) -> int:
    """[P1-26] Flushea TODOS los token buffers pendientes en el estado.

    Llamadores típicos:
      - El except del global timeout en `arun_plan_pipeline`
        (`asyncio.wait_for(run_graph(), timeout=GLOBAL_PIPELINE_TIMEOUT_S)`):
        cuando el timeout dispara antes de que el último día complete su
        finally, los buffers per-day SÍ flushean (asyncio cancellation
        respeta finally), pero el buffer `_default` (eventos sin `day`
        key — phase, metric, status emitidos desde nodos NO de
        generate_days) no tiene un finally simétrico. Sin un flush
        explícito al timeout, los tokens acumulados en `_default` se
        pierden y la UI muestra texto cortado a mitad.

      - Cualquier path graceful-degradation que entre al fallback antes
        de un return ordenado.

    Idempotente: si no hay buffers o todos están vacíos, devuelve 0.
    Best-effort: una excepción del callback en un día no aborta el flush
    de los otros días (prevenir que un callback roto bloquee la limpieza).

    Returns:
        int: cantidad de buffers que se flushearon con contenido.
    """
    if not isinstance(state, dict):
        return 0
    cb = state.get("progress_callback")
    if not cb:
        return 0
    buffers = state.get("_token_buffers")
    if not isinstance(buffers, dict) or not buffers:
        return 0
    flushed = 0
    # `list(buffers.keys())` para no iterar el dict mientras se muta
    # (el helper `_flush_token_buffer_for_day` modifica el text in-place
    # pero NO añade/borra keys; aun así blindamos contra refactors futuros).
    for day_key in list(buffers.keys()):
        try:
            if _flush_token_buffer_for_day(state, day_key):
                flushed += 1
        except Exception as _flush_err:
            # Best-effort: log y continuar con los demás buffers.
            logger.debug(
                f"[P1-26] Flush de buffer day_key={day_key!r} falló: "
                f"{_flush_err!r}. Continuando con los demás."
            )
    return flushed


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
                logger.info(f"🔍 [TÉCNICAS] Frecuencias con decaimiento temporal: { {k: round(v, 2) for k, v in technique_freq.items()} }")
        except Exception as e:
            logger.warning(f"⚠️ [TÉCNICAS] Error consultando DB, usando pesos uniformes: {e}")

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

    logger.info(f"👨‍🍳 [TÉCNICAS] Seleccionadas (familias diversas): {[f'{t} ({TECH_TO_FAMILY.get(t)})' for t in selected_techniques]}")
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
    # [P3-CONDITION-RULES · 2026-06-14] Anexa las reglas clínicas por condición (DM2 ADA 2026 /
    # ERC KDIGO 2024) al bloque de reglas determinista que SIEMPRE llega al generador. No-op para
    # usuarios sin condición cubierta (el builder retorna ""). Lever de prompt; el refuerzo
    # determinista (cap proteína renal, piso fibra DM2) ya vive en assemble/micronutrients.
    if CONDITION_RULES_ENABLED:
        try:
            variety_prompt = (variety_prompt or "") + build_medical_condition_context(form_data)
        except Exception:
            pass
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

    # [P1-DREAMING-1 · 2026-06-13 · Fase 4] Bloque del "modelo del usuario"
    # consolidado por el Dreaming, inyectado al generador de planes (gateado por
    # MEALFIT_DREAMING_INJECT_PLAN_ENABLED, default OFF → '' → plan idéntico).
    # Lazy import (dreaming hace lazy-import de LLMCircuitBreaker de este módulo).
    _dream_plan_constraints = ""
    try:
        import dreaming as _dreaming_mod
        _dream_plan_constraints = _dreaming_mod.build_plan_constraints_block(_uid)
    except Exception:
        pass

    return {
        "user_id": _uid,
        "quality_context": build_skeleton_quality_context(previous_plan_quality, meal_level_adherence),
        "quality_hint_context": build_quality_hint_context(quality_hint, drastic_strategy),
        "chunk_lessons_context": build_chunk_lessons_context(chunk_lessons),
        "prev_chunk_adherence_context": build_prev_chunk_adherence_context(prev_chunk_adherence),
        "weight_history_context": build_weight_history_context(weight_history),
        "nutrition_context": build_nutrition_context(nutrition),
        # [P5-PROMPT-D] Versión mínima del bloque para el corrector
        # (self_critique correction y surgical_marker_regen). Quita
        # kinematics + adaptación evolutiva — el corrector solo necesita
        # los targets duros para preservar balance. Reduce ~60-75% del
        # bloque, traduce a ~20-30s menos por corrección bajo provider load.
        "nutrition_context_minimal": build_minimal_correction_context(nutrition),
        "adherence_context": build_adherence_context(adherence_hint, meal_level_adherence, ignored_meal_types, abandoned_reasons, emotional_state, successful_tone_strategies, nudge_conversion_rates, frustrated_meal_types),
        "success_patterns_context": build_success_patterns_context(succ_techs, aban_techs, []),
        "temporal_adherence_context": build_temporal_adherence_context(form_data.get("day_of_week_adherence", {})),
        "unified_behavioral_profile": build_unified_behavioral_profile(user_facts, fatigued_ingredients, liked_meals, liked_flavor_profiles, cold_start_recs, allergies, llm_retrospective),
        # [P0-FORM-3] `motivation` se sanea río arriba en `_sanitize_form_data_recursive`,
        # así que llega aquí ya seguro contra prompt injection. El builder retorna ""
        # si el campo está vacío/whitespace → no-op transparente para usuarios que
        # no completaron el step (aunque desde P0-FORM-3 es required en el backend).
        "motivation_context": build_motivation_context(form_data),
        # [P2-AUDIT-5 · 2026-05-10] sleepHours/stressLevel inyectados como hints de
        # tono/sesgo. Retorna "" si ambos valores son no-accionables (sleep 7-8h,
        # stress Bajo/Moderado) — no contamina el prompt con low-signal data.
        "sleep_stress_context": build_sleep_stress_context(form_data),
        "rag_context": "",
        "fatigue_context": build_fatigue_context(fatigued_ingredients),
        "liked_meals_context": build_liked_meals_context(liked_meals),
        "correction_context": build_correction_context(review_feedback),
        "pantry_correction_context": build_pantry_correction_context(form_data.get("_pantry_correction", "")),
        # [P1-D · cableado P1-CHUNK-LEARN-3 · 2026-05-29] Drift de nevera entre chunks.
        # `_compute_pantry_diff_warning` lo computa en cron_tasks y lo deja en form_data;
        # pre-fix ningún builder lo consumía (dead-write → feature P1-D muerta).
        "pantry_drift_context": build_pantry_drift_context(form_data.get("_pantry_drift_warning")),
        "time_context": build_time_context(),
        "variety_prompt": variety_prompt,
        "supplements_context": build_supplements_context(form_data),
        # [BUDGET-CUSTOM · 2026-05-31] Presupuesto (categórico + monto custom RD$)
        # → ajuste de ingredientes. Pre-fix `budget` no llegaba al LLM.
        "budget_context": build_budget_context(form_data),
        "grocery_duration_context": build_grocery_duration_context(form_data),
        "pantry_context": build_pantry_context(form_data),
        # [P3-GENCHUNK-SPEED · 2026-06-01] El catálogo de precios solo es
        # relevante cuando hay señal de presupuesto (su propio texto dice
        # "si el usuario pide algo económico"). Lo gateamos con el MISMO
        # predicado no-vacío que `build_budget_context` (que ya transporta la
        # señal real de budget al LLM): sin budget declarado, omitimos el bloque
        # — el LLM no necesita el catálogo de precios para diseñar un plan sano,
        # y el bloque crecería sin techo con la tabla master_ingredients.
        # tooltip-anchor: P3-GENCHUNK-SPEED-PRICES-GATE
        "prices_context": build_prices_context() if (str(form_data.get("budget") or "").strip()) else "",
        "taste_profile": taste_profile,
        "history_context": history_context,
    }


# [P4-MODEL-1] Escalación dinámica a Pro para el day generator cuando el
# intento previo falló por skeleton fidelity violation.
# ------------------------------------------------------------
# Patrón observado en producción (corridas 2026-05-05 múltiples):
#   - Planner asigna proteínas distintas (ej. ['Atún', 'Lentejas', 'Huevos']).
#   - Day generator (Flash) las IGNORA, pone pavo procesado o omite proteínas.
#   - Medical reviewer rechaza con HIGH "Día N omitió proteínas asignadas".
#   - Retry permitido por P1-RETRY-CLASSIFY → segundo intento con Flash.
#   - Flash sigue inconsistente → segundo HIGH → P0-PIPE-1 swap al intento #1
#     menos malo, pero el usuario sigue recibiendo plan con disclaimer.
#
# Análisis del trade-off:
#   - Flash: ~30s/día, $0.075/M tokens input. Adherencia ~70%.
#   - Pro: ~75s/día, $1.25/M tokens input. Adherencia ~95%.
#   - Costo Pro = ~17× más por token; latencia ~2.5×.
#
# Política conservadora: escalar a Pro SOLO en retry SOLO cuando la causa
# fue skeleton fidelity. Esto evita pagar Pro para retries por otras
# razones (despensa, alergia, repetición proteína sin omisión asignada).
# Knob default `True`; operador puede desactivar via env si quiere costo
# mínimo a riesgo de más fallos cascading.
#
# Latencia retry esperada con Pro:
#   3 días × (75s Pro − 35s Flash) = +120s vs retry todo en Flash.
#   Pipeline budget 720s, intento #1 típicamente 230-300s, deja
#   ~420-490s. Pro retry consume ~370s → margen de 50-120s. Tight pero OK.
DAY_GEN_RETRY_USE_PRO = _env_bool("MEALFIT_DAY_GEN_RETRY_USE_PRO", True)

# [P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] Habilita el switch idiomático a
# `SystemMessage` + `HumanMessage` para `plan_skeleton` y `generate_single_day`.
# Gemini procesa `SystemMessage` como `system_instruction` separado del user
# content — es el target canónico del implicit prompt caching (Gemini reutiliza
# tokens del system_instruction entre requests consecutivos dentro del TTL,
# cobrando ~25% del input price normal).
#
# Pre-fix: tanto `PLANNER_SYSTEM_PROMPT` (~960 tok) como
# `DAY_GENERATOR_SYSTEM_PROMPT` (~4900 tok) + schema JSON estaban concatenados
# AL FINAL de un único `HumanMessage`. Sin SystemMessage separado, Gemini no
# puede identificar la región estática del prompt → cache hit rate ≈ 0% →
# se paga full input price en cada llamada.
#
# Riesgo: algunos prompts están escritos asumiendo que el modelo lee
# instrucciones al FINAL del payload. Switch a SystemMessage cambia la
# disposición — riesgo de regresión en calidad. Mitigación:
#   - Knob default True pero flippable a False sin redeploy si plan_quality_degraded
#     alerts (invariante I5) suben tras deploy.
#   - SOP post-deploy: generar 2-3 planes consecutivos, query
#     `SELECT model, input_tokens, cached_tokens FROM llm_usage_events
#      ORDER BY created_at DESC LIMIT 10` — `cached_tokens > 0` en runs
#     2+ confirma que el caching está hitando.
#
# ROI esperado: -40-60% del input cost en planes consecutivos (warm cache).
# Si solo 1 plan generándose, no hay hit (cold cache, primer request).
PROMPT_CACHE_SYSTEM_MESSAGE = _env_bool("MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE", True)

# [P1-PROMPT-CACHE-STAGGER · 2026-05-16] Audit 2026-05-16 reveló cache hit
# de solo 16% en Flash regular (vs. 50%+ esperado de P1-PROMPT-CACHE-SYSTEMMSG).
# Root cause: las 3 day_gen llamadas paralelas arrancan SIMULTÁNEAMENTE → todas
# son "primer request" para el cache server-side → ninguna se beneficia del cache
# de las otras. La 1ra puebla el cache pero las 2da y 3ra ya están mid-flight.
#
# Fix opt-in: stagger los días 2+3 por N ms tras lanzar día 1. Trade-off:
#   - Stagger=0 (default actual): comportamiento legacy, paralelismo puro,
#     cache hit ~16% (solo retries dentro de TTL).
#   - Stagger=2000 (recomendado para activar): día 1 fire t=0; día 2 fire
#     t=2000ms (cache populado); día 3 fire t=4000ms. Latencia plan +~4s,
#     cache hit esperado ~50-60% en days 2+3 = $0.20-0.30/plan ahorrado.
#
# [P2-ORCH-1 · 2026-05-28] Default 0 → 1500ms (activo). day_gen es el nodo más
# caro (>50% del gasto) y con stagger=0 los N días disparaban simultáneos → 16%
# cache hit medido vs ~50-60% esperado ($0.20-0.30/plan). Worst-case latencia
# añadida = (N-1)*stagger; con PLAN_CHUNK_SIZE=3 y 1500ms = ~3s, despreciable vs
# GLOBAL_PIPELINE_TIMEOUT_S=720s. Clamp [0,10000] evita que un valor patológico
# serialice day-gen y reviente el timeout. Revertir sin redeploy:
# MEALFIT_DAY_GEN_CACHE_STAGGER_MS=0. Validar con `SELECT cached_tokens FROM
# llm_usage_events WHERE node LIKE '%day%'` (days 2..N deben mostrar cached>0).
# Tooltip-anchor: P2-ORCH-1.
DAY_GEN_CACHE_STAGGER_MS    = _env_int  ("MEALFIT_DAY_GEN_CACHE_STAGGER_MS",     1500,
                                         validator=lambda v: 0 <= v <= 10000)

# [P1-COST-THINKING-CAP · 2026-05-28] Cap del thinking budget de gemini-3.5-flash
# en los nodos de GENERACIÓN/CORRECCIÓN (day-gen + correctores + planner skeleton),
# NO en reviewer/judge/fact-checker (SÍ necesitan razonar → se dejan sin cap; además
# defaultean a flash-lite que ni soporta thinking_config).
# [P1-EVALUATOR-THINKING-CAP · 2026-06-01] El EVALUATOR de self_critique se EXCLUÍA
# de aquí por la misma razón ("necesita razonar"), pero la telemetría de prod
# (`llm_usage_events`: media 6426 / máx 14350 tok de OUTPUT para un schema de 5
# scores + bool + string corto) reveló reasoning runaway — las señales duras
# (slot_issues/staples/monotonía) ya van PRE-calculadas por código al prompt, así
# que el razonamiento extra del LLM es mayormente desperdicio. Decisión revisada
# (consenso 2026-06-01): se capa con un presupuesto DEDICADO y GENEROSO
# (`EVALUATOR_THINKING_BUDGET`, default 4096 = 2x day-gen) que recorta solo el tail
# patológico SIN tocar el razonamiento normal. Redes: el FLOOR determinista de días
# (corrige incoherencias aunque el LLM las omita) + el knob `EVALUATOR_USE_PRO`.
# Diagnóstico 2026-05-28: sin cap, cada call grande del
# day-gen emitía ~16K output tokens (~12K de reasoning) a $9/M → ~80% del costo
# del plan, cuando generar un día es rellenar un schema (no requiere 12K de
# razonamiento). Semántica: -1=dynamic (comportamiento previo, sin cap),
# 0=thinking off, >0=cap en N tokens. Default 2048 (punto medio seguro). Verificar
# calidad A/B vía `llm_usage_events.output_tokens` antes/después; subir el knob si
# la calidad baja. Solo aplica a modelos thinking-capable (gemini-3.5-flash);
# flash-lite NO soporta thinking_config → se omite. Tooltip-anchor: P1-COST-THINKING-CAP.
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] `DAYGEN_THINKING_BUDGET` y
# `EVALUATOR_THINKING_BUDGET` eliminados junto con sus helpers — eran caps
# del reasoning de Gemini (ver nota en el bloque de helpers más abajo).

# [L1-UNBIND-NUTRITION-TOOL · 2026-05-28] Kill-switch para des-enlazar la tool
# consultar_nutricion del day-gen. La tabla de nutrición ya está inyectada en el
# SystemMessage cacheado (_build_nutrition_lookup_instruction) y Z1 instruye no
# llamarla → la tool es redundante. Con False: bind_tools se omite → el modelo NO
# puede emitir tool_calls → cero rondas de tool (garantía estructural sobre Z1, que
# es solo a nivel prompt) + se quita el schema de la tool (~250-400 tok) de cada
# call. Default True (la tool queda como fallback explícito) → flip a False tras
# confirmar por telemetría que las tool-calls cayeron a ~0 (Z1) + spot-check de
# macros por-porción de las proteínas/carbos principales. Tooltip-anchor: L1-UNBIND-NUTRITION-TOOL.
DAYGEN_BIND_NUTRITION_TOOL  = _env_bool ("MEALFIT_DAYGEN_BIND_NUTRITION_TOOL",   True)

# [P1-DEEPSEEK-JSON-MODE · 2026-06-13] DeepSeek-V4 flash, ante el prompt complejo
# del day-generator (schema JSON + mucho contexto), IGNORA la instrucción "responde
# solo JSON" y emite ~18K chars de razonamiento en prosa ("Let me analyze the
# requirements...") → `json.loads` falla → se cuenta como fallo del CB → CB OPEN →
# fallback matemático (test E2E 2026-06-13, 3 días en paralelo todos fallaron así).
# El prompt estaba afinado para Gemini, que sí obedecía. Fix: forzar JSON mode
# (`response_format={"type":"json_object"}`), que GARANTIZA salida JSON válida.
# JSON mode es incompatible con tool-calling streaming → cuando está activo NO se
# bindea la tool de nutrición (de todas formas un mock redundante: la tabla
# autoritativa ya viaja en el SystemMessage cacheado, ver L1-UNBIND-NUTRITION-TOOL).
# Validado en vivo: 17.3s/día con JSON parseable vs >70s de prosa sin parsear.
# Rollback sin redeploy: MEALFIT_DAYGEN_JSON_MODE=False vuelve al tool-calling.
DAYGEN_JSON_MODE = _env_bool("MEALFIT_DAYGEN_JSON_MODE", True)


# [P0-DEEPSEEK-MIGRATION · 2026-06-12] `_thinking_budget_kwargs` y
# `_evaluator_thinking_budget_kwargs` ELIMINADOS (P1-COST-THINKING-CAP /
# P1-EVALUATOR-THINKING-CAP). Eran caps del reasoning de Gemini, que
# facturaba como output a ~$9/M y producía runaways patológicos (day-gen
# llegó a 19,162 tok). DeepSeek-V4 gestiona el thinking nativamente sin
# budget por request, y su output cuesta $0.28–0.87/M (10-30× menos) — el
# problema de costo que motivaba el cap ya no existe. Los knobs
# `MEALFIT_DAYGEN_THINKING_BUDGET` / `MEALFIT_EVALUATOR_THINKING_BUDGET`
# dejaron de leerse.


# [P1-PROMPT-CACHE-STAGGER · 2026-05-16] Pre-computar el SystemMessage del
# day_generator UNA vez al import. ANTES, `schema_dict = SingleDayPlanModel.
# model_json_schema()` se invocaba en cada call de `generate_single_day`
# (línea ~4644) y se concatenaba a `_schema_instruction` para construir
# `day_system_instruction`. Esto introducía dos riesgos:
#   1. Re-computación innecesaria por call (~2-3ms × N días/plan).
#   2. Cualquier variance de Pydantic en el orden de keys entre llamadas
#      rompía la byte-equivalence del SystemMessage → cache miss garantizado.
#
# El sort_keys=True elimina la 2da fuente de variance (orden de keys no-
# determinístico es un edge case de Pydantic raro pero documentado).
# El pre-cómputo al import elimina la 1ra fuente.
#
# Tooltip-anchor: P1-PROMPT-CACHE-STAGGER-MODULE-CONSTANTS
def _build_day_schema_instruction() -> str:
    """Construye la sección del prompt que declara el JSON schema del día.
    sort_keys=True garantiza byte-equivalence entre invocaciones."""
    schema_dict = SingleDayPlanModel.model_json_schema()
    return (
        "\n\nDEBES DEVOLVER ESTRICTAMENTE UN JSON VÁLIDO QUE CUMPLA CON "
        "ESTE ESQUEMA (NO incluyas bloques de markdown como ```json):\n"
        + json.dumps(schema_dict, sort_keys=True)
    )


_DAY_SCHEMA_INSTRUCTION = _build_day_schema_instruction()


# [P3-COST-CUT-V2 · 2026-05-21] Pre-computar tabla de nutrición e inyectarla
# al SystemMessage cacheado del day_generator. ANTES: cada day_worker hacía
# 3-4 `consultar_nutricion` tool_calls, cada uno un LLM roundtrip completo
# (~$0.012/call con cache). Audit 2026-05-21 reveló que `tools_nutrition.py`
# es un MOCK dict de 15 ingredientes y ~50% de las queries del LLM retornan
# "No se encontró" (gandules, yuca, guineo, yogurt griego no están en el
# dict) — el LLM estima igual, pero paga el roundtrip. Inyectando el dict
# en el SystemMessage cacheado:
#   - Cero coste marginal por llamada (cached input $0.15/M vs fresh $1.50/M)
#   - LLM ve la tabla autoritativa sin necesitar tool_call para los 15 conocidos
#   - Para ingredientes NO en tabla: LLM estima desde training knowledge
#     (mismo comportamiento que hoy cuando el tool retorna "no encontrado")
# Tool sigue disponible vía bind_tools — fallback explícito si el LLM lo
# necesita para ingredientes específicos, pero la instrucción del prompt
# desincentiva su uso para los 15 listados.
# Tooltip-anchor: P3-COST-CUT-V2-NUTRITION-TABLE.
def _build_nutrition_lookup_instruction() -> str:
    """Construye la sección del SystemMessage cacheado con la tabla de
    nutrición pre-computada. Llamado UNA vez al import — byte-equivalence
    entre todas las llamadas (cache hit garantizado tras la primera)."""
    from tools_nutrition import MOCK_NUTRITION_DB
    lines = [
        "\n\nTABLA DE NUTRICIÓN PRE-COMPUTADA (valores por 100g, fuente autoritativa):",
        "Usa estos valores DIRECTAMENTE sin invocar `consultar_nutricion` para los",
        "siguientes 15 ingredientes — el roundtrip de tool es innecesario:",
    ]
    for key in sorted(MOCK_NUTRITION_DB.keys()):
        d = MOCK_NUTRITION_DB[key]
        lines.append(
            f"  - {key}: {d['calories']} kcal | "
            f"P {d['protein']}g | C {d['carbs']}g | G {d['fats']}g"
        )
    lines.append(
        "Para ingredientes NO listados arriba (ej. gandules, yuca, guineo, "
        "yogurt griego, mango, ñame, casabe, salami, jamón, sardina): estima "
        "macros usando tu conocimiento general SIN invocar `consultar_nutricion` "
        "(el tool retorna 'no encontrado' para esos casos y desperdicia un "
        "roundtrip). Reglas de orientación rápida:"
    )
    lines.append(
        "  - Tubérculos (yuca, ñame, malanga): ~110-130 kcal/100g, alto en carbos."
    )
    lines.append(
        "  - Legumbres secas cocidas (gandules, garbanzos, lentejas): ~120-130 kcal/100g, P~7-9g, C~22g."
    )
    lines.append(
        "  - Yogurt griego sin azúcar: ~60 kcal/100g, P~10g, C~4g, G~0.4g."
    )
    lines.append(
        "  - Guineo (banana): ~89 kcal/100g, C~23g."
    )
    lines.append(
        "  - Mango: ~60 kcal/100g, C~15g."
    )
    return "\n".join(lines)


_NUTRITION_LOOKUP_INSTRUCTION = _build_nutrition_lookup_instruction()
_DAY_SYSTEM_INSTRUCTION_CACHED = (
    DAY_GENERATOR_SYSTEM_PROMPT
    + _DAY_SCHEMA_INSTRUCTION
    + _NUTRITION_LOOKUP_INSTRUCTION
)


# [P3-COST-CUT-V2 · 2026-05-21] System instruction cacheable del SELF-CRITIQUE
# EVALUATOR. ANTES: el prompt completo (rol + criterios + datos del plan) iba
# como un solo string concatenado a `_safe_ainvoke(evaluator_llm, prompt, ...)`,
# sin SystemMessage separado → NO cache eligible aunque `PROMPT_CACHE_SYSTEM_MESSAGE=True`.
# Audit 2026-05-21 confirmó: planner + day_gen wirean cache OK; este evaluator
# era el único bug de wiring restante. Splitting el prompt en
# {SystemMessage(static), HumanMessage(per-plan data)} permite cache hit del
# rol + criterios (la parte invariante) — ahorra ~$0.003/plan input cost +
# ~10-15s latencia per critique cycle.
# Tooltip-anchor: P3-COST-CUT-V2-EVALUATOR-CACHE.
_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION = """
Eres un Crítico Culinario Experto. Tu tarea es evaluar un plan de comidas (días generados) y emitir 5 scores numéricos del 1 al 10.

CRITERIOS DE EVALUACIÓN:
1. Atractivo visual (visual_score): ¿Se lee apetitoso o son combinaciones raras?

2. Diversidad real de sabores (diversity_score). Penaliza con score <=4 si:
   - Se repite la misma proteína o guarnición principal con nombres distintos.
   - Un staple (avena, claras, pan, yogurt, queso, lechosa, guineo, plátano maduro, aguacate, tortilla)
     aparece en 2+ días (ver bloque 'STAPLES REPETIDOS' en el HumanMessage si está presente).

3. Coherencia cultural Dominicana (cultural_score): ¿El desayuno tiene sentido? ¿La cena es coherente?

4. Balance de temperaturas (temperature_score): ¿Hay 3 días seguidos de ensaladas frías o todo es sopa?

5. Coherencia comida↔horario (slot_coherence_score):
   - MERIENDAS deben ser SNACKS LIGEROS (yogurt+fruta, batido, casabe+queso, sándwich pequeño, fruta+mani).
     Si una merienda es "Salteado de…", "Locrio de…", "Pechuga al grill con puré", o cualquier mini-almuerzo, BAJA este score a ≤4.
   - CENA NO debe repetir la PROTEÍNA PRINCIPAL ni el CARBOHIDRATO PRINCIPAL del almuerzo del mismo día. Si los repite, BAJA este score a ≤4.
   - Si el bloque 'INCOHERENCIAS POR SLOT' en el HumanMessage lista hallazgos, son hechos: BAJA slot_coherence_score a ≤4 obligatoriamente.

REGLA DE DECISIÓN:
Si DOS O MÁS scores son < 6, o si ALGÚN score es < 4, marca needs_correction=True y da instrucciones CLARAS Y CORTAS de qué cambiar, mencionando explícitamente el día afectado (ej. "Día 2").
""".strip()


# [P1-FLASH-LITE-AUX-NODES · 2026-05-15] Modelos auxiliares de pipeline
# (judge / fact_checker / reviewer) hacen tareas binarias bajo schema strict.
# Gemini Flash-Lite Preview ($0.10/M in, $0.40/M out) rinde igual que Flash
# regular ($0.30/M in, $2.50/M out) en `with_structured_output(...)` cuando
# el output es boolean/enum/schema fijo (vs. generación creativa de planes
# que sí requiere Flash o Pro).
#
# Pre-fix:
#   - judge_llm: usa `_route_model(form_data, force_fast=False)` → ESCALA A
#     PRO en perfiles clínicos complejos (medical conditions, alergias >1).
#     Costoso (~$0.030/judge invocation cuando hay perfil complejo).
#   - fact_checker_llm: ya usa `_route_model(force_fast=True)` → siempre
#     `gemini-3-flash-preview`. Margen de bajar a Lite.
#   - reviewer_llm: `_route_model(form_data, attempt=1)` → mismo escalado a
#     Pro que judge.
#
# Trade-off documentado: este P-fix REMUEVE el escalado automático a Pro
# para estos 3 nodos. Mitigación: cada nodo tiene su propio knob de
# override — si la calidad regresiona en un perfil clínico, el SRE pone
# `MEALFIT_<NODE>_MODEL=gemini-3.1-pro-preview` en EasyPanel sin redeploy.
#
# Sigue convención P3-PREVIEW-MODEL-KNOB (CLAUDE.md): callsites en
# crons/loops productivos leen model ID desde knob — modelos preview pueden
# deprecarse sin aviso (audit 2026-05-11: `gemini-3.1-pro-preview` open=true
# 4.4 días). Knob permite swap sin redeploy.
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] DeepSeek no tiene tier "lite"; el
# modelo barato del stack es V4 Flash. El nombre de la constante se preserva
# (los ~10 callsites/helpers la referencian como "el default barato").
_FLASH_LITE_DEFAULT = DEEPSEEK_FLASH


def _judge_model_name() -> str:
    """[P1-FLASH-LITE-AUX-NODES] Modelo del adversarial judge.
    Default Flash-Lite (tarea binaria con `AdversarialJudgeResult` schema).
    Override knob: `MEALFIT_JUDGE_MODEL` (sin redeploy si calidad degrada)."""
    return _env_str("MEALFIT_JUDGE_MODEL", _FLASH_LITE_DEFAULT) or _FLASH_LITE_DEFAULT


# [P2-ORCH-7 · 2026-05-28] Modelo "risk-tier" para el reviewer médico y el
# fact-checker clínico cuando el perfil declara alergias/condiciones médicas.
# El reviewer es el ÚNICO gate LLM de seguridad clínica.
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default `deepseek-v4-pro` PARA TODOS
# los tiers (incluido gratis): la seguridad clínica de perfiles con
# alergias/condiciones NO se degrada por plan de pago — es 1 call por plan y
# el delta de costo es centavos. Perfiles sin restricciones siguen en el
# modelo barato. Tooltip-anchor: P2-ORCH-7.
_REVIEWER_RISK_TIER_DEFAULT = DEEPSEEK_PRO

_PROFILE_RISK_NEGATIVES = {"", "ninguna", "ninguno", "none", "n/a", "na", "no", "nada"}


def _profile_has_medical_risk(form_data) -> bool:
    """[P2-ORCH-7] True si el perfil declara alergias o condiciones médicas
    no-vacías. Determina el risk-tier del reviewer/fact-checker."""
    if not isinstance(form_data, dict):
        return False
    for key in ("allergies", "medicalConditions"):
        v = form_data.get(key)
        if isinstance(v, (list, tuple, set)):
            if any(str(x).strip() and str(x).strip().lower() not in _PROFILE_RISK_NEGATIVES for x in v):
                return True
        elif v and str(v).strip().lower() not in _PROFILE_RISK_NEGATIVES:
            return True
    return False


def _fact_checker_model_name(form_data=None) -> str:
    """[P1-FLASH-LITE-AUX-NODES · P2-ORCH-7] Modelo del fact-checker clínico.
    Hard-override `MEALFIT_FACT_CHECKER_MODEL` siempre gana. Si el perfil tiene
    alergias/condiciones → risk-tier (`MEALFIT_FACT_CHECKER_RISK_TIER_MODEL`,
    default `gemini-3.5-flash`); de lo contrario Flash-Lite."""
    _override = _env_str("MEALFIT_FACT_CHECKER_MODEL", "")
    if _override:
        return _override
    if _profile_has_medical_risk(form_data):
        return (_env_str("MEALFIT_FACT_CHECKER_RISK_TIER_MODEL", _REVIEWER_RISK_TIER_DEFAULT)
                or _REVIEWER_RISK_TIER_DEFAULT)
    return _FLASH_LITE_DEFAULT


def _reviewer_model_name(form_data=None) -> str:
    """[P1-FLASH-LITE-AUX-NODES · P2-ORCH-7] Modelo del reviewer (pipeline_holistic).
    Hard-override `MEALFIT_REVIEWER_MODEL` siempre gana. Si el perfil tiene
    alergias/condiciones médicas → risk-tier (`MEALFIT_REVIEWER_RISK_TIER_MODEL`,
    default `gemini-3.5-flash`, más capaz que Lite para razonamiento clínico);
    de lo contrario Flash-Lite (`ReviewResult` schema strict, temp=0.1)."""
    _override = _env_str("MEALFIT_REVIEWER_MODEL", "")
    if _override:
        return _override
    if _profile_has_medical_risk(form_data):
        return (_env_str("MEALFIT_REVIEWER_RISK_TIER_MODEL", _REVIEWER_RISK_TIER_DEFAULT)
                or _REVIEWER_RISK_TIER_DEFAULT)
    return _FLASH_LITE_DEFAULT


# [P1-PROMPT-TRIM-FORM-DATA · 2026-05-15] Reduce el dump de `form_data`
# inyectado al prompt de `plan_skeleton_node` y `generate_single_day`.
#
# Pre-fix: el callsite hacía `json.dumps(form_data, indent=2)` que volcaba
# TODAS las claves del form_data — incluyendo metadata pipeline-internal
# con prefijo `_` (e.g. `_chunk_lessons`, `_adherence_hint`, `_emotional_state`,
# `_chunk_prior_meals`, `_pipeline_drift_alert`, `_days_offset`,
# `_plan_start_date`, `_drastic_change_strategy`, `_quality_hint`, ~25
# claves más). Estas claves:
#   1. NO son información que el LLM generador necesite leer directamente.
#   2. YA están "absorbidas" en los `ctx[...]` que se inyectan separadamente
#      (e.g. `ctx['chunk_lessons_context']` viene de `_chunk_lessons`,
#      `ctx['quality_hint_context']` viene de `_adherence_hint`, etc.).
#   3. Las que SÍ son user-facing (chunk_prior_meals, pipeline_drift_alert,
#      auto_simplify) ya se inyectan como instrucciones explícitas en el
#      prompt en sus respectivas ramas (líneas 4003, 3995, 3998).
#
# Resultado pre-fix: ~30-50% del dump JSON eran claves `_*` redundantes.
# Para un form_data típico el dump pesa ~2-3K tokens; ~1-1.5K son ruido.
#
# Estimación de ahorro: planner ~1K tokens × 1 invocación + day_gen ~1K
# tokens × 3 invocaciones paralelas = ~4K input tokens menos por plan. A
# Pro pricing ($1.25/M) = ~$0.005/plan. A Flash ($0.30/M) en day_gen =
# ~$0.001/plan. Sumado en escala: significativo.
#
# Trade-off: si el LLM resulta usar alguna `_*` clave en su "razonamiento
# implícito" (e.g. inferir de `_emotional_state` un tono más cálido sin
# necesitar el `motivation_context` explícito), calidad puede degradar
# sutilmente. Kill switch: `MEALFIT_PROMPT_TRIM_FORM_DATA=False`.
#
# IMPORTANTE: este helper solo afecta el dump al PROMPT. El código backend
# sigue leyendo todas las claves de `form_data` (line 4003 chunk_prior_meals,
# etc.). No es una mutación del state.
PROMPT_TRIM_FORM_DATA = _env_bool("MEALFIT_PROMPT_TRIM_FORM_DATA", True)


def _sanitize_form_data_for_prompt(form_data: dict) -> dict:
    """[P1-PROMPT-TRIM-FORM-DATA · 2026-05-15] Retorna una copia de
    `form_data` SIN las claves pipeline-internal con prefijo `_`.

    Si `PROMPT_TRIM_FORM_DATA=False` (kill switch), retorna `form_data`
    sin cambios (legacy behavior).

    No muta el original. El código backend que consume `form_data["_xxx"]`
    debe seguir leyendo del state, NO del retorno de este helper.
    """
    if not isinstance(form_data, dict):
        return form_data
    if not PROMPT_TRIM_FORM_DATA:
        return form_data
    return {k: v for k, v in form_data.items() if not (isinstance(k, str) and k.startswith("_"))}
# [P3-PLAN-MODEL-KNOBS · 2026-05-20] Modelos del plan-gen pipeline ahora
# via knobs (no hardcoded). Cierre del gap C4 del audit
# `docs/gaps-audit-2026-05.md`: pre-fix estos eran string literals
# inmutables → swap requería redeploy + Nixpacks rebuild (~5min) + bumpear
# `_LAST_KNOWN_PFIX`. Con knobs, SRE setea env var en EasyPanel + restart
# worker = <1 min.
#
# Por qué importa para COSTOS:
#   - `gemini-3-flash-preview` actual: 15% cache hit promedio en últimas 14d
#     (medido via llm_usage_events 2026-05-20). $4.83 de costo del bucket.
#   - `gemini-3.5-flash` (medido en chat_call_model): 34% cache hit con prompts
#     más grandes. ~2x mejor cache → ahorro estimado ~30-50% en input tokens.
#   - Modelos `*-preview` de Google pueden deprecarse sin aviso prolongado
#     (mismo riesgo R2 que vision_agent — CB stale 4.4 días en 2026-05-11).
#
# SOP migración (validable sin redeploy):
#   1. Setear `MEALFIT_FLASH_MODEL=gemini-3.5-flash` en EasyPanel.
#   2. Restart worker.
#   3. Esperar 4-6 horas para acumular eventos.
#   4. Query `SELECT model, ROUND(100.0*SUM(cached_tokens)/SUM(input_tokens),2)
#      AS cache_pct, SUM(cost_usd_micros)/1e6 AS usd FROM llm_usage_events
#      WHERE created_at > NOW() - INTERVAL '6 hours' GROUP BY model`.
#   5. Si cache_pct sube y/o usd baja → mantener. Si calidad de output baja
#      (validable via plan_quality_degraded alerts) → rollback inmediato a
#      `gemini-3-flash-preview`.
#
# Defaults preservan comportamiento actual. Tooltip-anchor: P3-PLAN-MODEL-KNOBS.
def _plan_pro_model_name() -> str:
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] El "modelo PRO" del pipeline es el
    # modelo del TIER PAGADO: DeepSeek V4 Pro ($0.435/M in · $0.87/M out).
    # `_route_model` lo asigna a usuarios basic/plus/ultra.
    # Rollback/swap sin redeploy: `MEALFIT_PRO_MODEL=<model-id>`.
    return _env_str("MEALFIT_PRO_MODEL", DEEPSEEK_PRO)


def _plan_flash_model_name() -> str:
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] El "modelo FLASH" del pipeline es
    # el modelo del TIER GRATIS y de los paths force_fast/aux: DeepSeek V4
    # Flash ($0.14/M in · $0.28/M out). Rollback/swap sin redeploy:
    # `MEALFIT_FLASH_MODEL=<model-id>`.
    return _env_str("MEALFIT_FLASH_MODEL", DEEPSEEK_FLASH)


def _planner_model_name() -> str:
    """[P3-PLANNER-LITE-COST · 2026-05-21] Override del modelo del SKELETON
    PLANNER ÚNICAMENTE. El planner solo asigna nombres + slots (proteína por día,
    carbo por día, vegetales, técnica de cocción) — clasificación-like, sin
    constraints duros de macros (esos se aplican en day generators downstream).
    Flash-lite cubre la tarea + reduce costo ~7% per-plan generation.

    Si vacío (`""`), el callsite respeta el ruteo dinámico de `_route_model`
    (PRO para perfiles clínicos complejos, FLASH para simples). Útil si la
    calidad del skeleton degrada visiblemente — flip a `""` recupera ruteo
    legacy sin redeploy.

    Rollback sin redeploy: `MEALFIT_PLANNER_MODEL=""` (cadena vacía) restaura
    el ruteo dinámico. O `MEALFIT_PLANNER_MODEL=deepseek-v4-pro` fuerza Pro.
    """
    return _env_str("MEALFIT_PLANNER_MODEL", DEEPSEEK_FLASH)


def _self_critique_model_name() -> str:
    """[P3-SELF-CRITIQUE-LITE-COST · 2026-05-22] Override del modelo del
    EVALUATOR de self_critique únicamente. El evaluator solo emite 5 scores
    (1-10) + un bool + un string corto de suggestions sobre un summary
    comprimido del plan (NO el plan completo). Las señales críticas
    (slot_issues + staple_repetitions) están pre-calculadas por código y van
    al prompt — el LLM solo necesita leer + ponderar + emitir structured
    output. Tarea adecuada para flash-lite.

    Safety nets contra false-negatives:
      1. `deterministic_days` FLOOR (línea ~6928): días con slot_incoherence
         se inyectan al floor de corrección AUN si el LLM los omite.
      2. `EVALUATOR_USE_PRO` knob override: si la calidad degrada visiblemente,
         flip `MEALFIT_EVALUATOR_USE_PRO=1` para escalar a Pro (este knob
         tiene precedencia sobre `MEALFIT_SELF_CRITIQUE_MODEL`).
      3. El corrector (que regenera días) sigue usando Flash full vía
         `_route_model(force_fast=True)` — NO se afecta por este knob.

    Default: `_FLASH_MODEL_NAME` (preserva comportamiento pre-fix). Para
    activar el A/B de ahorro: setear `MEALFIT_SELF_CRITIQUE_MODEL=gemini-3.1-flash-lite`
    en EasyPanel/.env y restart del worker. Para revertir: cadena vacía o
    el nombre del modelo Flash original.

    Costo estimado en `self_critique` últimos 7 días (datos prod 2026-05-22):
      - gemini-3.5-flash:           12 eventos, $0.68 → $0.11 con lite (84% off)
      - gemini-3-flash-preview:    116 eventos, $1.96 → $1.20 con lite (39% off)
      - TOTAL 7d: $2.64 → $1.31 → ahorro ~$1.33/7d = ~$5.70/mes.
    (Si el pricing dict está inflado vs Google Cloud Billing real, ajustar
    proporcionalmente — el % de ahorro relativo se mantiene.)
    """
    return _env_str("MEALFIT_SELF_CRITIQUE_MODEL", _plan_flash_model_name())


def _compressor_model_name() -> str:
    """[P3-COST-CUT-AUX · 2026-05-22] Modelo del `context_compression_node`.
    Default `gemini-3.1-flash-lite`. La compresión es síntesis textual sobre
    contexto de historial (>2000 chars) — el LLM recibe el texto + un system
    prompt explícito que dice "no inventes nada, solo comprime + preserva
    alergias/medical/restricciones". Lite cubre la tarea sin riesgo de perder
    señal: el system prompt fuerza preservación literal de los facts críticos.

    Safety nets ya activas: si CB OPEN o timeout → fallback `compressed_context =
    history_context` (sin comprimir, pero no falla). El downstream funciona
    con contexto completo.

    Costo 14d prod: $0.017/14d (1 evento) → $0.003/14d con lite. Centavos
    ahorro; el valor es estructural (limpia el hot-path de aux nodes).

    Rollback sin redeploy: `MEALFIT_COMPRESSOR_MODEL=deepseek-v4-pro`.
    """
    return _env_str("MEALFIT_COMPRESSOR_MODEL", DEEPSEEK_FLASH)


def _meta_learning_model_name() -> str:
    """[P3-COST-CUT-AUX · 2026-05-22] Modelo del `reflection_node` (meta-learning).
    Default `gemini-3.1-flash-lite`. El reflector diagnostica el ciclo previo
    en UNA oración con structured output `ReflectionResult`. Input: 4 strings
    cortos (quality_score + meal_adherence + successful + abandoned + fatigued)
    ~300 tokens. Output: 1 oración corta ~50-100 tokens. Tarea de pattern
    matching simple, sin razonamiento clínico.

    Safety nets: cache_key por hash conductual (TTL del PersistentLLMCache).
    Si CB OPEN o timeout → exception capturada upstream, el `reflection_text`
    queda vacío y el siguiente nodo opera sin él.

    Costo 14d prod: $0.019/14d (4 eventos) → $0.003/14d con lite. Ahorro
    marginal en absoluto pero ~84% relativo del node.

    Rollback sin redeploy: `MEALFIT_META_LEARNING_MODEL=deepseek-v4-pro`.
    """
    return _env_str("MEALFIT_META_LEARNING_MODEL", DEEPSEEK_FLASH)


# Module-level constants preserved para compatibilidad con los 34 callsites
# existentes que comparan strings literales (e.g. `if model == _FLASH_MODEL_NAME`).
# Estos NO se re-leen en runtime — para cambiar el modelo se requiere restart
# del worker (que re-importa el módulo). Aceptable trade-off: env var seteada
# en EasyPanel + restart = <1min, vs PR + merge + Nixpacks rebuild = ~10min.
_PRO_MODEL_NAME = _plan_pro_model_name()
_FLASH_MODEL_NAME = _plan_flash_model_name()

# [P6-EVALUATOR-USE-PRO] Knob para escalar EL EVALUATOR del self-critique a Pro,
# manteniendo Flash en day_generators y corrector. El evaluator es UN solo call
# (vs 3 paralelos en day_gen) y su decisión gate-quea las correcciones — si
# omite un día (como en PDF 2026-05-05 19:13 donde dejó Día 2 sin mencionar),
# todo el downstream pierde la chance de corregirlo.
#
# Trade-off: ~$0.001-0.003 extra/plan, +5-10s en self-critique, mejor cobertura
# de incoherencias. Bajo riesgo, fácil de revertir con OFF.
#
# OFF por default mientras se valida en producción (cualquier change al evaluator
# afecta todos los planes). Activar con `MEALFIT_EVALUATOR_USE_PRO=1`.
EVALUATOR_USE_PRO = _env_bool("MEALFIT_EVALUATOR_USE_PRO", False)


async def _attempt_pro_critique_correction(
    correction_prompt: str,
    day_num: int,
    log_prefix: str = "[CRITIQUE/PRO-FALLBACK]",
):
    """[2026-05-06] Reintenta corrección con Gemini Pro tras fallo de Flash.

    Llamado desde `_correct_single_day` (self_critique_node) y `_re_correct_one`
    (P5 marker regen) cuando el corrector Flash hace timeout o devuelve None.
    Pro es ~2x más lento pero más estable bajo prompts complejos. Solo se
    invoca si `CRITIQUE_PRO_FALLBACK_ENABLED` está activo.

    Returns
    -------
    tuple
        `(corrected_day_dict | None, reason_str)` donde:
          - corrected_day_dict: dict del día corregido (con `day=day_num`
            inyectado) en éxito; None en cualquier fallo.
          - reason_str: una de "fallback_disabled", "pro_cb_open",
            "pro_success", "pro_returned_none", "pro_timeout",
            "pro_error:<ExcName>".
    """
    if not CRITIQUE_PRO_FALLBACK_ENABLED:
        return None, "fallback_disabled"
    pro_cb = _get_circuit_breaker(_PRO_MODEL_NAME)
    if not await pro_cb.acan_proceed():
        logger.warning(
            f"⚠️ {log_prefix} CB OPEN para {_PRO_MODEL_NAME}. "
            f"Día {day_num} sin reintento Pro."
        )
        return None, "pro_cb_open"
    try:
        logger.info(
            f"🔄 {log_prefix} Día {day_num} reintentando con {_PRO_MODEL_NAME} "
            f"(timeout={CRITIQUE_PRO_FALLBACK_TIMEOUT_S:.0f}s)..."
        )
        pro_corrector = ChatDeepSeek(
            model=_PRO_MODEL_NAME,
            temperature=0.3,
            max_retries=0,
            timeout=int(CRITIQUE_PRO_FALLBACK_TIMEOUT_S),
        ).with_structured_output(SingleDayPlanModel)
        result = await _safe_ainvoke(
            pro_corrector,
            correction_prompt,
            timeout=CRITIQUE_PRO_FALLBACK_TIMEOUT_S,
        )
        if result:
            await pro_cb.arecord_success()
            corrected = result.model_dump()
            corrected["day"] = day_num
            logger.info(
                f"✅ {log_prefix} Día {day_num} corregido con {_PRO_MODEL_NAME}."
            )
            return corrected, "pro_success"
        await pro_cb.arecord_failure()
        logger.warning(
            f"⚠️ {log_prefix} {_PRO_MODEL_NAME} retornó None para Día {day_num}. "
            f"Manteniendo original."
        )
        return None, "pro_returned_none"
    except asyncio.TimeoutError:
        logger.info(
            f"⏱️ {log_prefix} {_PRO_MODEL_NAME} también timeout para Día {day_num} "
            f"({CRITIQUE_PRO_FALLBACK_TIMEOUT_S:.0f}s)."
        )
        return None, "pro_timeout"
    except Exception as e:
        try:
            await _record_cb_failure_unless_transient(pro_cb, e)  # P1-ORCH-1/2
        except Exception:
            pass
        logger.warning(
            f"⚠️ {log_prefix} Error con {_PRO_MODEL_NAME} para Día {day_num}: "
            f"{type(e).__name__}: {e}"
        )
        return None, f"pro_error:{type(e).__name__}"


def _is_skeleton_fidelity_rejection(rejection_reasons) -> bool:
    """[P4-MODEL-1 / P4-MODEL-2] Detecta si el rechazo previo es síntoma
    de baja adherencia del LLM al skeleton del planner — caso donde
    escalar a Pro en retry mejora outcomes (~25 puntos de adherencia).

    P4-MODEL-1 (original) — skeleton fidelity literal:
      - "Día N omitió múltiples proteínas clave asignadas: [...]"
      - "skeleton fidelity"

    P4-MODEL-2 (expansión) — síntomas de "model laziness" del mismo origen.
    Cuando Flash no respeta el skeleton suele también:
      - Repetir la misma proteína (toma el camino fácil) → "repetición excesiva"
      - Colapsar variedad en el menú → "falta de variedad"
    Ambos son señales correlacionadas con baja adherencia, no causas
    independientes — en producción aparecen frecuentemente JUNTAS al
    skeleton fidelity violation, y cuando aparecen solos suelen ser el
    primer síntoma del mismo modo de fallo. Tratarlos también como trigger
    de escalación a Pro evita el cascading donde Flash falla 2 retries
    seguidos por la misma raíz (model laziness) sin que el clasificador
    lo detecte por usar palabras distintas en cada review.

    NOTA: rechazos ortogonales (despensa, alergia, gluten, recipe-ingredient
    mismatch) NO entran aquí — esos no se arreglan con un modelo más
    grande; se arreglan con prompt o con datos. Solo señales de adherencia.
    """
    if not rejection_reasons:
        return False
    text = " ".join(str(r).lower() for r in rejection_reasons)
    return bool(
        _re.search(
            r'omiti[oó]\s+m[uú]ltiples\s+prote[ií]nas|'
            r'skeleton\s+fidelity|'
            r'omiti[oó]\s+prote[ií]nas\s+asignadas|'
            r'repetici[oó]n\s+excesiva|'
            r'falta\s+de\s+variedad',
            text,
        )
    )


# [P2-DAYGEN-LITE-EASY · 2026-05-28] Opt-in: bajar el day_generator a un modelo
# lite para perfiles FÁCILES (no clínicos) en attempt 1. day_generator es el paso
# más voluminoso y el mayor costo del pipeline (~17k tokens in × N días/plan, ~$5.75
# acumulado en prod); flash-lite cuesta ~10× menos por token. Default OFF: el lite
# puede degradar instruction-following / tool-use (consultar_nutricion) → más
# correcciones de self_critique que anularían el ahorro. El operador lo activa tras
# validar calidad. NUNCA aplica a perfiles clínicos complejos (siempre full/PRO) ni
# a retries (attempt>1 conserva el modelo de fiabilidad).
DAYGEN_LITE_FOR_EASY = _env_bool("MEALFIT_DAYGEN_LITE_FOR_EASY", False)
DAYGEN_EASY_MODEL = _env_str("MEALFIT_DAYGEN_EASY_MODEL", _FLASH_LITE_DEFAULT) or _FLASH_LITE_DEFAULT


def _route_model_for_day_generator(
    form_data: dict,
    attempt: int,
    prev_rejection_reasons=None,
) -> str:
    """[P4-MODEL-1] Routing especializado para el day generator.

    Reglas:
      1. Attempt #1 o knob deshabilitado → routing default (`_route_model`).
      2. Attempt > 1 + prev rejection con skeleton fidelity → escalar a Pro.
      3. Attempt > 1 sin esa señal → routing default (Flash si easy, Pro
         si clinical complex).

    Esto cierra el cascading de falla observado en producción donde Flash
    fallaba el skeleton 2 intentos seguidos. El swap de P0-PIPE-1
    rescataba el menos malo, pero el usuario aún recibía disclaimer.
    Pro en el retry específico mejora la adherencia ~25 puntos a costo
    de ~$0.05 extra por plan (estimado por intento que escala).
    """
    if attempt <= 1 or not DAY_GEN_RETRY_USE_PRO:
        _base = _route_model(form_data, attempt)
        # [P2-DAYGEN-LITE-EASY · 2026-05-28] Override lite solo en attempt 1 y solo
        # si el router resolvió a FLASH (perfil fácil; NO escaló a PRO por
        # complejidad clínica). Conserva la garantía de que perfiles complejos y
        # retries usan el modelo full/PRO.
        if attempt <= 1 and DAYGEN_LITE_FOR_EASY and _base == _FLASH_MODEL_NAME:
            logger.info(
                f"🔀 [ROUTER/DAYGEN-LITE] Perfil fácil + MEALFIT_DAYGEN_LITE_FOR_EASY=1 "
                f"→ day generator usa {DAYGEN_EASY_MODEL} (en vez de {_FLASH_MODEL_NAME}). "
                f"Recorte de costo en el paso más voluminoso; validar calidad."
            )
            return DAYGEN_EASY_MODEL
        return _base

    if _is_skeleton_fidelity_rejection(prev_rejection_reasons):
        logger.info(
            f"🔀 [ROUTER P4] Day generator escalado a PRO ({_PRO_MODEL_NAME}) "
            f"en retry attempt={attempt} por skeleton fidelity violation previa. "
            f"Trade: ~+40s/día vs Flash, +25pts de adherencia esperada."
        )
        return _PRO_MODEL_NAME

    return _route_model(form_data, attempt)


def _route_model(form_data: dict, attempt: int = 1, force_fast: bool = False) -> str:
    """Mejora 1: Ruteo Dinámico de Modelos (Cost/Latency Routing).

    [P0-DEEPSEEK-MIGRATION · 2026-06-12] El ruteo ahora es POR TIER DE
    SUSCRIPCIÓN (decisión de producto 2026-06-12):
      - `gratis` / guests / tier irresoluble → `_FLASH_MODEL_NAME` (V4 Flash)
      - `basic` / `plus` / `ultra` (pagados) → `_PRO_MODEL_NAME` (V4 Pro)

    El user_id se lee del ContextVar `user_id_var`, que `arun_plan_pipeline`
    setea al entrar (cubre requests síncronos Y chunk workers de fondo).
    Sin user en contexto → FLASH (fail-cheap: un fallo de lookup jamás
    encarece la llamada). `force_fast=True` preserva su semántica histórica:
    paths correctores/aux que eligen el modelo barato deliberadamente.

    El ruteo legacy por complejidad clínica (P1-A) quedó reemplazado: la
    seguridad clínica NO depende de este router — el reviewer médico y el
    fact-checker escalan a PRO via risk-tier (P2-ORCH-7,
    `_REVIEWER_RISK_TIER_DEFAULT`) para CUALQUIER perfil con
    alergias/condiciones, incluido tier gratis. `form_data`/`attempt` se
    conservan en la firma por compat con los callers (y un futuro ruteo
    híbrido tier+complejidad puede reusarlos).
    """
    if force_fast:
        return _FLASH_MODEL_NAME

    _uid = user_id_var.get()
    tier = get_user_tier(_uid)
    if tier in PAID_TIERS:
        logger.info(
            f"🔀 [ROUTER] Tier '{tier}' (pagado) → modelo PRO ({_PRO_MODEL_NAME})."
        )
        return _PRO_MODEL_NAME
    logger.info(
        f"🔀 [ROUTER] Tier '{tier or 'gratis'}' → modelo FLASH ({_FLASH_MODEL_NAME})."
    )
    return _FLASH_MODEL_NAME


# [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] Cap duro del history_context cuando NO
# se pudo comprimir (CB-open / resultado sospechoso / excepción). Pre-fix esos
# 3 fallbacks devolvían el blob crudo sin límite — y justo cuando el provider
# está degradado (CB open) el planner recibía el contexto completo,
# amplificando costo/latencia de tokens durante un outage. El reflection_node
# además concatena al history_context en cada retry, así que crece sin tope.
# Elisión head+tail: preserva las reglas médicas/alergias prepended desde
# memory_context Y las alertas de calidad appended al final; descarta el medio
# (la zona "lost in the middle"). Knob `MEALFIT_HISTORY_CONTEXT_MAX_CHARS`
# (default 16000, floor 2000). Tooltip-anchor: P2-HISTORY-CONTEXT-CAP.
# ============================================================
# [P3-COMPRESSOR-CACHE · 2026-05-29] Cache content-addressed del resultado de
# context_compression_node (optimización de costo LLM).
# ------------------------------------------------------------
# context_compression_node hace una llamada LLM (flash-lite, budget ~30s) en
# CADA pipeline cuando history_context > 2000 chars — el caso común de usuarios
# recurrentes con mucho historial (reglas médicas, alergias, platos recientes,
# despensa, adherencia). El resultado es determinístico para un mismo input
# (temperature=0.0 + prompt "no inventes nada, solo resume"), así que cachearlo
# por hash SHA-256 del history_context es SEGURO: cualquier cambio del historial
# produce otra key → invalidación automática (cero staleness). Un cache hit
# ahorra una llamada LLM completa.
#
# In-process (no Redis): best-effort y por-worker. Un worker frío recomputa una
# vez; no hay corrección que dependa del cache (el fallback siempre existe).
# Acotado por nº de entradas + TTL. Knobs:
#   MEALFIT_COMPRESSOR_CACHE_ENABLED   (True)  — kill switch sin redeploy.
#   MEALFIT_COMPRESSOR_CACHE_TTL_S     (3600)  — vida de cada entrada.
#   MEALFIT_COMPRESSOR_CACHE_MAX_ENTRIES (256) — cota de tamaño (LRU aprox).
# Tooltip-anchor: P3-COMPRESSOR-CACHE.
_COMPRESSOR_CACHE: "dict[str, tuple[str, float]]" = {}
_COMPRESSOR_CACHE_LOCK = threading.Lock()


def _compressor_cache_key(history_context: str) -> str:
    return hashlib.sha256((history_context or "").encode("utf-8")).hexdigest()


def _compressor_cache_get(history_context: str) -> Optional[str]:
    """Devuelve el compressed_text cacheado para este history_context, o None."""
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_ENABLED", True):
        return None
    key = _compressor_cache_key(history_context)
    now = time.time()
    with _COMPRESSOR_CACHE_LOCK:
        entry = _COMPRESSOR_CACHE.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry < now:
            _COMPRESSOR_CACHE.pop(key, None)
            return None
        return value


def _compressor_cache_put(history_context: str, compressed_text: str) -> None:
    """Cachea un compressed_text exitoso, con GC perezoso + cota de tamaño."""
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_ENABLED", True):
        return
    ttl = _env_int("MEALFIT_COMPRESSOR_CACHE_TTL_S", 3600, validator=lambda v: v > 0)
    max_entries = _env_int("MEALFIT_COMPRESSOR_CACHE_MAX_ENTRIES", 256, validator=lambda v: v > 0)
    key = _compressor_cache_key(history_context)
    now = time.time()
    with _COMPRESSOR_CACHE_LOCK:
        if len(_COMPRESSOR_CACHE) >= max_entries and key not in _COMPRESSOR_CACHE:
            # 1) purgar expiradas.
            for k in [k for k, (_, exp) in _COMPRESSOR_CACHE.items() if exp < now]:
                _COMPRESSOR_CACHE.pop(k, None)
            # 2) si sigue lleno, evict por expiry más cercano (≈LRU).
            while len(_COMPRESSOR_CACHE) >= max_entries:
                oldest = min(_COMPRESSOR_CACHE.items(), key=lambda kv: kv[1][1])[0]
                _COMPRESSOR_CACHE.pop(oldest, None)
        _COMPRESSOR_CACHE[key] = (compressed_text, now + ttl)


# --- Telemetría (hit-rate) -------------------------------------------------
# Contadores in-process para medir cuánto ahorra el cache en prod antes de
# invertir más. Expuestos vía GET /api/system/admin/health-snapshot.
_COMPRESSOR_CACHE_STATS = {"hits_memory": 0, "hits_kv": 0, "misses": 0}


def _compressor_cache_record_hit(source: str) -> None:
    with _COMPRESSOR_CACHE_LOCK:
        if source == "kv":
            _COMPRESSOR_CACHE_STATS["hits_kv"] += 1
        else:
            _COMPRESSOR_CACHE_STATS["hits_memory"] += 1


def _compressor_cache_record_miss() -> None:
    with _COMPRESSOR_CACHE_LOCK:
        _COMPRESSOR_CACHE_STATS["misses"] += 1


def get_compressor_cache_stats() -> dict:
    """Snapshot de hits/misses del cache de compresión (telemetría admin).
    [P3-COMPRESSOR-CACHE]"""
    with _COMPRESSOR_CACHE_LOCK:
        s = dict(_COMPRESSOR_CACHE_STATS)
        s["in_process_entries"] = len(_COMPRESSOR_CACHE)
    total = s["hits_memory"] + s["hits_kv"] + s["misses"]
    s["total_lookups"] = total
    s["hit_rate"] = round((s["hits_memory"] + s["hits_kv"]) / total, 4) if total else 0.0
    return s


# --- Capa 2: persistencia en app_kv_store (sobrevive restarts/deploys) ------
# Bajo `--workers 1` el cache in-process se pierde en cada reinicio. El chunk
# worker genera semanas sucesivas del mismo plan a lo largo de horas; si el
# proceso reinicia entre chunks, el cache queda frío. Persistir en app_kv_store
# (mismo patrón que rag_/reflection_) hace que el hit sobreviva. Sigue siendo
# content-addressed (key = hash) → cero staleness. TTL SELECT-side via
# MEALFIT_COMPRESSOR_CACHE_TTL_S; filas stale las GC el cron _sweep_stale_app_kv_store_prefixes
# (prefix `compressor_cache:`). Best-effort: cualquier fallo de DB no rompe el pipeline.
# Knob de kill-switch independiente: MEALFIT_COMPRESSOR_CACHE_PERSIST (True).
_COMPRESSOR_CACHE_KV_PREFIX = "compressor_cache:"


def _compressor_cache_kv_key(history_context: str) -> str:
    return f"{_COMPRESSOR_CACHE_KV_PREFIX}{_compressor_cache_key(history_context)}"


async def _compressor_cache_get_persistent(history_context: str) -> Optional[str]:
    """Lee el compressed_text desde app_kv_store. Best-effort → None si falla."""
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_ENABLED", True):
        return None
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_PERSIST", True):
        return None
    ttl = _env_int("MEALFIT_COMPRESSOR_CACHE_TTL_S", 3600, validator=lambda v: v > 0)
    key = _compressor_cache_kv_key(history_context)
    try:
        row = await aexecute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s "
            "AND updated_at > NOW() - make_interval(secs => %s)",
            (key, ttl), fetch_one=True,
        )
        if not row:
            return None
        value = row.get("value") or {}
        text = value.get("v") if isinstance(value, dict) else None
        return text if (isinstance(text, str) and text) else None
    except Exception as e:
        logger.debug(f"[P3-COMPRESSOR-CACHE] get_persistent best-effort fail: {e}")
        return None


async def _compressor_cache_put_persistent(history_context: str, compressed_text: str) -> None:
    """Persiste el compressed_text en app_kv_store. Best-effort (swallow)."""
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_ENABLED", True):
        return
    if not _env_bool("MEALFIT_COMPRESSOR_CACHE_PERSIST", True):
        return
    key = _compressor_cache_kv_key(history_context)
    try:
        await aexecute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) "
            "VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (key, json.dumps({"v": compressed_text})),
        )
    except Exception as e:
        logger.debug(f"[P3-COMPRESSOR-CACHE] put_persistent best-effort fail: {e}")


def _cap_history_context(text: str) -> str:
    cap = _env_int("MEALFIT_HISTORY_CONTEXT_MAX_CHARS", 16000, validator=lambda v: v >= 2000)
    if not isinstance(text, str) or len(text) <= cap:
        return text
    half = (cap - 80) // 2
    head = text[:half]
    tail = text[-half:]
    elided = len(text) - 2 * half
    logger.warning(
        f"✂️ [COMPRESIÓN] history_context sin comprimir ({len(text)} chars) "
        f"excede cap {cap}; elisión head+tail ({elided} chars al medio) para "
        f"acotar tokens. [P2-HISTORY-CONTEXT-CAP]"
    )
    return f"{head}\n\n[...{elided} chars elididos por cap MEALFIT_HISTORY_CONTEXT_MAX_CHARS...]\n\n{tail}"


# ============================================================
@_node_label("compressor")
async def context_compression_node(state: PlanState) -> dict:
    """Mejora 4: Comprime el exceso de contexto para evitar Lost in the Middle."""
    history_context = state.get("history_context", "")
    
    # Solo comprimir si el contexto es demasiado largo (ej. > 2000 caracteres)
    if len(history_context) < 2000:
        return {"compressed_context": history_context}

    # [P3-COMPRESSOR-CACHE · 2026-05-29] Cache content-addressed → saltar la
    # llamada LLM completa. Seguro porque la key es el hash del history_context.
    # Capa 1: in-process (más rápido). Capa 2: app_kv_store (sobrevive restarts).
    _cached = _compressor_cache_get(history_context)
    if _cached is not None:
        _compressor_cache_record_hit("memory")
        logger.info(
            f"🗜️ [COMPRESIÓN] Cache hit in-process ({len(history_context)} chars → "
            f"{len(_cached)} cacheados). Skip LLM. [P3-COMPRESSOR-CACHE]"
        )
        return {"compressed_context": _cached}

    _cached_kv = await _compressor_cache_get_persistent(history_context)
    if _cached_kv is not None:
        _compressor_cache_put(history_context, _cached_kv)  # calentar in-process
        _compressor_cache_record_hit("kv")
        logger.info(
            f"🗜️ [COMPRESIÓN] Cache hit persistente ({len(history_context)} chars → "
            f"{len(_cached_kv)} cacheados). Skip LLM. [P3-COMPRESSOR-CACHE]"
        )
        return {"compressed_context": _cached_kv}

    _compressor_cache_record_miss()
    logger.info(f"🗜️ [COMPRESIÓN] Contexto masivo detectado ({len(history_context)} caracteres). Comprimiendo...")
    
    # P1-Q3: capturar el modelo para usar el CB per-modelo.
    # [P3-COST-CUT-AUX · 2026-05-22] Usar helper `_compressor_model_name()`
    # (knob `MEALFIT_COMPRESSOR_MODEL`, default DeepSeek V4 Flash) en
    # lugar del ruteo dinámico — la compresión es síntesis textual literal,
    # no requiere razonamiento Pro. Rollback: setear el knob a deepseek-v4-pro.
    # [P0-DEEPSEEK-MIGRATION] Usa el wrapper module-level ChatDeepSeek
    # (backpressure + instrumentación), igual que el resto de nodos.
    _compressor_model = _compressor_model_name()
    _cb = _get_circuit_breaker(_compressor_model)
    compressor_llm = ChatDeepSeek(
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
            logger.warning("⚠️ [COMPRESIÓN] Circuit Breaker OPEN. Saltando compresión.")
            return {"compressed_context": _cap_history_context(history_context)}

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
            logger.warning(f"⚠️ [COMPRESIÓN] Resultado sospechoso ({len(compressed_text)} chars). Usando contexto original.")
            return {"compressed_context": _cap_history_context(history_context)}

        # [P3-COMPRESSOR-CACHE] Cachear el resultado exitoso (content-addressed)
        # en ambas capas: in-process + app_kv_store (sobrevive restarts).
        _compressor_cache_put(history_context, compressed_text)
        await _compressor_cache_put_persistent(history_context, compressed_text)
        logger.info(f"🗜️ [COMPRESIÓN] Contexto reducido a {len(compressed_text)} caracteres.")
        return {"compressed_context": compressed_text}
    except Exception as e:
        await _record_cb_failure_unless_transient(_cb, e)  # P1-Q3 · P1-ORCH-1/2
        logger.warning(f"⚠️ [COMPRESIÓN] Error comprimiendo contexto: {e}. Usando original.")
        return {"compressed_context": _cap_history_context(history_context)}

# ============================================================
# NODO 1: PLANIFICADOR (Fase Map — esqueleto liviano)
# ============================================================
@_node_label("planner")
async def plan_skeleton_node(state: PlanState) -> dict:
    attempt = state.get("attempt", 1)
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    logger.info(f"\n{'='*60}")
    logger.info(f"📋 [PLANIFICADOR] Diseñando estructura del plan (intento #{attempt})...")
    logger.info(f"{'='*60}")

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
        logger.info(f"🔀 [RETRY MUTATION] Técnicas del intento anterior bloqueadas: {previous_techniques}")
        
    # [GAP 1 FIX] Bloquear la última técnica del chunk anterior si es continuación
    last_tech = form_data.get("_last_technique")
    if last_tech and attempt == 1:
        aban_techs = list(set(aban_techs + [last_tech]))
        logger.info(f"🔄 [CHUNK CONTINUATION] Técnica anterior bloqueada para este chunk: {last_tech}")

    # [G9-BLOCKED-TECHNIQUES · P1-CHUNK-LEARN-3 · 2026-05-29] Honrar `_blocked_techniques`.
    # El cron lo acumula (append+dedup a través de ciclos de resume/zero-log) y está
    # whitelisteado, pero NINGÚN consumer lo leía: solo `_last_technique` (la ÚLTIMA) entraba
    # a aban_techs → las técnicas bloqueadas de ciclos anteriores se re-seleccionaban (dead-write
    # clase G10/G11). `_select_techniques` ya down-weightea aban_techs (×0.1), así que mergearlas
    # honra la acumulación sin lógica nueva. attempt==1 (los retries ya bloquean vía el skeleton previo).
    if attempt == 1:
        blocked_techs = form_data.get("_blocked_techniques") or []
        if isinstance(blocked_techs, list) and blocked_techs:
            aban_techs = list(set(aban_techs + [t for t in blocked_techs if t]))
            logger.info(f"🚫 [BLOCKED-TECHNIQUES] Técnicas acumuladas bloqueadas: {blocked_techs}")

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

    # [P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] El prompt se compone en dos
    # tramos. Cuando `PROMPT_CACHE_SYSTEM_MESSAGE=True` (default), el tramo
    # ESTÁTICO (`PLANNER_SYSTEM_PROMPT`) viaja en un `SystemMessage` separado
    # — Gemini lo trata como `system_instruction`, primer target del implicit
    # caching. El tramo DINÁMICO (info usuario + ctx + técnicas) va en el
    # `HumanMessage` que cambia cada llamada y no es cacheable.
    # Cuando el knob está False, ambos se concatenan en un solo string como
    # antes (legacy path, kill switch sin redeploy).
    dynamic_prompt_text = (
        f"Analiza la siguiente información del usuario y diseña el ESQUELETO de un plan de {days_in_chunk} alternativas/días.\n"
        f"Semilla de generación aleatoria: {random_seed}\n\n"
        f"Información del Usuario:\n{json.dumps(_sanitize_form_data_for_prompt(form_data), indent=2)}\n"
        f"{ctx['quality_context']}\n{ctx['quality_hint_context']}\n{ctx['chunk_lessons_context']}\n{ctx['prev_chunk_adherence_context']}\n{ctx['weight_history_context']}\n{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
        f"{ctx['unified_behavioral_profile']}\n{ctx['correction_context']}\n{ctx['pantry_correction_context']}\n{ctx['history_context']}\n"
        f"{ctx['variety_prompt']}\n{ctx['pantry_context']}\n{ctx['pantry_drift_context']}\n{ctx['prices_context']}\n"
        f"{ctx['adherence_context']}\n{ctx['success_patterns_context']}\n"
        f"{ctx['temporal_adherence_context']}\n"
        f"{ctx['motivation_context']}\n"
        f"{ctx['sleep_stress_context']}\n\n"
        f"Técnicas de cocción asignadas (una por día):\n"
        f"{techniques_str}\n\n"
    )
    if PROMPT_CACHE_SYSTEM_MESSAGE:
        prompt_text = dynamic_prompt_text
    else:
        prompt_text = dynamic_prompt_text + PLANNER_SYSTEM_PROMPT

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

    # [G10-FORCE-TECHNIQUE-VARIETY · 2026-05-29] Consumo de `_force_technique_variety`.
    # El worker (cron_tasks.py:_force_variety→_force_technique_variety downgrade) setea este
    # flag cuando la repetición de ingredientes es alta PERO la nevera tiene baja diversidad:
    # forzar variedad de INGREDIENTES sería incoherente con la despensa, así que se pide variar
    # TÉCNICAS de preparación. Antes era dead-write (whitelisteado en _TRUSTED_INTERNAL_FORM_KEYS
    # pero SIN consumer) → la adaptación era no-op silencioso (el usuario seguía recibiendo platos
    # repetitivos justo cuando el sistema lo detectó). Hermano de `_force_variety` (ai_helpers.py).
    if form_data.get("_force_technique_variety"):
        prompt_text += "\n\n🍳 VARIEDAD DE TÉCNICA OBLIGATORIA: Los platos recientes repiten ingredientes y la despensa tiene baja diversidad. NO fuerces ingredientes nuevos (sería incoherente con lo disponible); en su lugar VARÍA DRÁSTICAMENTE las técnicas de preparación entre días y comidas — alterna método de cocción (horneado, salteado, guisado, a la plancha, hervido, al vapor), cortes, marinados, salsas y combinaciones, de modo que los mismos ingredientes base se perciban como platos distintos."

    # [G11-CREATIVE-FREEDOM · 2026-05-29] Consumo de `_creative_freedom`. El preflight
    # meta-learning (graph_orchestrator.py: rama score histórico >0.90) lo setea para PREMIAR
    # con mayor libertad creativa — contraparte positiva de `_auto_simplify` (que SÍ se consumía
    # acá arriba). Antes era dead-write (whitelisteado pero SIN consumer) → el refuerzo positivo
    # del meta-learning estaba muerto: usuarios con planes consistentemente excelentes nunca
    # recibían el boost de variedad/creatividad previsto.
    if form_data.get("_creative_freedom"):
        prompt_text += "\n\n✨ LIBERTAD CREATIVA: Tus planes recientes han sido de alta calidad y buena adherencia. Tienes mayor libertad para proponer combinaciones más creativas y recetas algo más elaboradas (siempre manteniendo coherencia nutricional, despensa y restricciones médicas). Aprovecha para introducir variedad y platos atractivos."

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
    # [P3-PLANNER-LITE-COST · 2026-05-21] Override del planner a lite cuando
    # `MEALFIT_PLANNER_MODEL` está set (default `gemini-3.1-flash-lite`). Si
    # vacío (`""`), respeta el ruteo dinámico. El skeleton es classification-like
    # (nombres + slots) — no requiere razonamiento Pro/Flash. Si calidad degrada:
    # `MEALFIT_PLANNER_MODEL=""` restaura ruteo dinámico.
    _planner_override = _planner_model_name()
    planner_model = _planner_override if _planner_override else _route_model(form_data, attempt)
    if attempt > 1:
        logger.info(f"🔀 [RETRY MUTATION] Modelo '{planner_model}' + temp={base_temp} para intento {attempt}")

    planner_llm = ChatDeepSeek(
        model=planner_model,
        temperature=base_temp,
        max_retries=0,
        timeout=45,
    ).with_structured_output(PlanSkeletonModel)

    # P1-Q3: CB per-modelo. Si pro está saturado, perfiles complejos paran;
    # los simples (que usan flash) no se ven afectados.
    _planner_cb = _get_circuit_breaker(planner_model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        # [P2-ORCH-4 · 2026-05-28] El 429 spending-cap es persistente: reintentar
        # es desperdicio puro. Falla rápido (resto de errores siguen reintentando;
        # parse/Pydantic se auto-corrigen con el bump de temperatura por intento).
        retry=retry_if_exception(lambda e: not _is_plan_spend_cap_error(e)),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"⚠️  [PLANIFICADOR] Reintento #{retry_state.attempt_number}...")
    )
    async def invoke_planner():
        if not await _planner_cb.acan_proceed():
            raise Exception(f"Circuit Breaker OPEN para {planner_model} - LLM cascade failure prevented")
        try:
            logger.info(f"⏳ [PLANIFICADOR] Generando esqueleto del plan...")
            # P0-4: Hard timeout con cancelación graceful. El constructor pone
            # timeout=45 pero el SDK no siempre lo respeta con sockets colgados.
            # tenacity hará 3 retries, con cap explícito de 50s/intento mantenemos
            # el peor caso ~150s. _safe_ainvoke garantiza cleanup del socket.
            # [P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] Cuando el knob está
            # habilitado, enviar SystemMessage + HumanMessage para que Gemini
            # reciba el system prompt como `system_instruction` separado
            # (target canónico del implicit caching).
            if PROMPT_CACHE_SYSTEM_MESSAGE:
                planner_payload = [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_text),
                ]
            else:
                planner_payload = prompt_text
            res = await _safe_ainvoke(planner_llm, planner_payload, timeout=50.0)
            await _planner_cb.arecord_success()
            return res
        except Exception as e:
            # [P1-LLM-TRANSIENT-5XX · 2026-05-21 · P1-ORCH-1/2 · 2026-05-28]
            # Helper centralizado: excluye del CB los 5xx transitorios de Google
            # Y el 429 spending-cap (persistente); además activa el latch del cap.
            await _record_cb_failure_unless_transient(_planner_cb, e)
            raise e

    response = await invoke_planner()

    duration = round(time.time() - start_time, 2)
    logger.info(f"✅ [PLANIFICADOR] Esqueleto generado en {duration}s")

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
                    logger.info(f"🧹 [SKELETON SCRUB] Día {skel_days[idx].get('day')}: "
                          f"eliminadas {removed} (cap '{restricted}' ya asignado al Día {skel_days[keep_idx].get('day')})")

    # 2. No combinar atún + embutido en el mismo día — remover embutidos si coexisten con atún
    for d in skel_days:
        pool = d.get('protein_pool', [])
        pool_lower = ' '.join((p or '').lower() for p in pool)
        has_atun = 'atún' in pool_lower or 'atun' in pool_lower
        embutidos_in_pool = [p for p in pool if any(emb in (p or '').lower() for emb in _EMBUTIDO_KEYS)]
        if has_atun and embutidos_in_pool:
            d['protein_pool'] = [p for p in pool if p not in embutidos_in_pool]
            logger.info(f"🧹 [SKELETON SCRUB] Día {d.get('day')}: eliminados embutidos "
                  f"{embutidos_in_pool} (conflicto con atún presente)")

    # 3. Fallback: si algún pool quedó vacío tras scrub, inyectar Lentejas como proteína segura
    for d in skel_days:
        if not d.get('protein_pool'):
            d['protein_pool'] = ['Lentejas']
            logger.warning(f"⚠️ [SKELETON SCRUB] Día {d.get('day')}: protein_pool vacío tras scrub, "
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


def _sanitize_unauthorized_protein_text(
    day_result: dict,
    unauthorized_keys: set,
    day_num: int,
) -> int:
    """[P2-PROTEIN-VIOLATION-SANITIZE · 2026-05-16] Sanitización defensiva
    del `name` y `recipe` text de meals cuando el bounded regen NO logra
    eliminar las menciones de proteínas prohibidas.

    Bug observado plan_id=fbd014b2 2026-05-16: el LLM persistió en proponer
    "Chuleta al Airfryer con Tostones" después del regen forzado. Sin esta
    sanitización, el meal final tenía name="Chuleta..." + ingredients SIN
    chuleta → reviewer médico rechazaba por "almuerzo sin proteína adecuada".

    Estrategia:
      - Para cada meal con violation: scrub el name reemplazando la palabra
        prohibida por placeholder genérico ("Plato del Almuerzo", "Cena del
        Día"). Si el name queda vacío tras strip, usar fallback por meal_type.
      - Para cada step del recipe array: reemplazar la palabra prohibida por
        "ingrediente alternativo". El usuario verá texto consistente con
        ingredients (que ya está strippeado).
      - Marcar el meal con `_protein_violation_sanitized: list[str]` para
        audit downstream.

    Returns:
        Número de text replacements realizados (suma de name + recipe steps).
    """
    if not isinstance(day_result, dict) or not unauthorized_keys:
        return 0

    _MEAL_TYPE_FALLBACK_NAMES = {
        "desayuno": "Desayuno del Día",
        "almuerzo": "Plato del Almuerzo",
        "merienda": "Merienda del Día",
        "cena": "Cena del Día",
    }
    _total_replacements = 0
    for _meal in day_result.get('meals', []):
        if not isinstance(_meal, dict):
            continue
        _meal_violations_sanitized = []
        # 1. Sanitizar name.
        _name = _meal.get('name', '') or ''
        _name_lower = _name.lower()
        _name_was_modified = False
        for _uk in unauthorized_keys:
            if ' ' in _uk:
                # Multi-palabra: substring replace case-insensitive.
                if _uk in _name_lower:
                    _pattern = _re.compile(_re.escape(_uk), _re.IGNORECASE)
                    _name = _pattern.sub('', _name).strip()
                    _name_lower = _name.lower()
                    _name_was_modified = True
                    _meal_violations_sanitized.append(_uk)
                    _total_replacements += 1
            else:
                # Single word: word boundary.
                _pattern = _re.compile(rf'\b{_re.escape(_uk)}\b', _re.IGNORECASE)
                if _pattern.search(_name):
                    _name = _pattern.sub('', _name).strip()
                    _name_lower = _name.lower()
                    _name_was_modified = True
                    _meal_violations_sanitized.append(_uk)
                    _total_replacements += 1
        # Cleanup: colapsar dobles espacios, strip puntuación huérfana al inicio/fin.
        if _name_was_modified:
            _name = _re.sub(r'\s+', ' ', _name).strip(' .,:;-')
            # Si el name queda muy corto o vacío, usar fallback por meal_type.
            if len(_name) < 4:
                _meal_type = (_meal.get('meal_type') or '').lower().strip()
                _name = _MEAL_TYPE_FALLBACK_NAMES.get(_meal_type, "Plato del Día")
            _meal['name'] = _name

        # 2. Sanitizar recipe steps.
        _recipe = _meal.get('recipe', []) or []
        if isinstance(_recipe, list):
            _new_recipe = []
            for _step in _recipe:
                if not isinstance(_step, str):
                    _new_recipe.append(_step)
                    continue
                _step_modified = False
                for _uk in unauthorized_keys:
                    if ' ' in _uk:
                        if _uk in _step.lower():
                            _pattern = _re.compile(_re.escape(_uk), _re.IGNORECASE)
                            _step = _pattern.sub('ingrediente alternativo', _step)
                            _step_modified = True
                    else:
                        _pattern = _re.compile(rf'\b{_re.escape(_uk)}\b', _re.IGNORECASE)
                        if _pattern.search(_step):
                            _step = _pattern.sub('ingrediente alternativo', _step)
                            _step_modified = True
                if _step_modified:
                    _total_replacements += 1
                _new_recipe.append(_step)
            _meal['recipe'] = _new_recipe

        # 3. Marcar audit trail.
        if _meal_violations_sanitized:
            _meal['_protein_violation_sanitized'] = _meal_violations_sanitized
            logger.info(
                f"🩹 [DÍA {day_num}] Meal '{_meal.get('name')}' sanitizado: "
                f"removidas palabras {_meal_violations_sanitized} de name/recipe."
            )

    return _total_replacements


def _apply_protein_pool_scrub(
    day_result: dict,
    skeleton_day: dict,
    day_num: int,
    context_label: str = "PARALLEL-GEN",
) -> tuple[int, set[str]]:
    """[PROTEIN-POOL-SCRUB 2026-05-07] Helper compartido — cleanup + scan.

    Aplicado en 3 puntos del pipeline donde el LLM produce/corrige un día:
      1. `generate_days_parallel_node` — generación inicial paralela.
      2. `self_critique` correction — reescritura tras crítica.
      3. `_marker_regen_node` (P5-MARKER-REGEN) — surgical post-aprobación.

    Antes solo el path (1) tenía cleanup+scan. Bug observable plan 089e541c:
    pool elegido `[Queso Blanco, Gandules, Atún]`, pero la lista final tenía
    "Pechuga de pollo 1¼ lbs" porque el surgical regen (path 3) reintrodujo
    pollo respondiendo al critique sin pasar por el cleanup. Centralizar la
    lógica garantiza paridad entre los 3 paths.

    Returns:
        (violations_count, unauthorized_keys) — el caller usa keys para
        construir el augmented prompt si decide forzar bounded regen
        (solo PARALLEL-GEN lo hace; los otros paths lo descartan).
    """
    if not isinstance(day_result, dict):
        return 0, set()
    from prompts.day_generator import _RESTRICTED_PROTEIN_KEYS, _POOL_IMPLICATIONS

    pool_lower = ' '.join(skeleton_day.get('protein_pool', [])).lower() if skeleton_day else ''

    def _key_in_text(key: str, text: str) -> bool:
        # Word-boundary para keys de una palabra (evita 'res' en 'fresco');
        # substring para keys multi-palabra ('jamón de pavo').
        if ' ' in key:
            return key in text
        return bool(_re.search(rf'\b{_re.escape(key)}\b', text))

    # [P0-PROTEIN-POOL-IMPLICATIONS · 2026-05-16] Auto-autorizar keys del
    # mismo grupo proteico cuando el pool las trae explícitamente.
    # Ej: pool=['Chuleta'] → 'cerdo' queda auto-autorizado porque "chuleta
    # de cerdo" es la expansión natural del LLM. Sin esto, el matcher
    # eliminaba el ingrediente y la receta quedaba sin proteína (bug del
    # plan aeb25e1c día 2).
    implied_authorized: set[str] = set()
    for pool_key, implied_set in _POOL_IMPLICATIONS.items():
        if _key_in_text(pool_key, pool_lower):
            implied_authorized.update(implied_set)

    unauthorized_keys = {
        k for k in _RESTRICTED_PROTEIN_KEYS
        if not _key_in_text(k, pool_lower) and k not in implied_authorized
    }
    if not unauthorized_keys:
        return 0, set()

    for _meal in day_result.get('meals', []):
        _original = _meal.get('ingredients', [])
        _cleaned, _removed = [], []
        for _ing in _original:
            _ing_lower = _ing.lower()
            if any(_key_in_text(_uk, _ing_lower) for _uk in unauthorized_keys):
                _removed.append(_ing)
            else:
                _cleaned.append(_ing)
        if _removed:
            _meal['ingredients'] = _cleaned
            logger.info(
                f"🚫 [{context_label}/DÍA {day_num}] Proteínas no autorizadas eliminadas de "
                f"'{_meal.get('name')}': {_removed}"
            )

    _violations = []
    _meals_scanned = 0
    for _meal in day_result.get('meals', []):
        _recipe_text = ' '.join(_meal.get('recipe', []) or []).lower()
        _meal_name_lower = (_meal.get('name', '') or '').lower()
        _full_text = f"{_meal_name_lower} {_recipe_text}"
        _meals_scanned += 1
        for _uk in unauthorized_keys:
            if _key_in_text(_uk, _full_text):
                _violations.append((_meal.get('name', '?'), _uk))
                break

    if _violations:
        _summary = ', '.join(f"'{m}' menciona '{k}'" for m, k in _violations[:3])
        logger.warning(
            f"⚠️  [{context_label}/DÍA {day_num}] PROTEIN-RECIPE-VIOLATION detectada: "
            f"{_summary}{'...' if len(_violations) > 3 else ''}"
        )
    else:
        logger.info(
            f"🔍 [{context_label}/DÍA {day_num}] PROTEIN-POOL-SCRUB limpio — "
            f"{_meals_scanned} meals × {len(unauthorized_keys)} keys, 0 violations. "
            f"Pool: {skeleton_day.get('protein_pool', []) if skeleton_day else []}"
        )

    return len(_violations), unauthorized_keys


# ============================================================
# NODO 2: GENERADORES PARALELOS (Fase Reduce — 3 días simultáneos)
# ============================================================
@_node_label("day_generator")
async def generate_days_parallel_node(state: PlanState) -> dict:
    """Genera los 7 días completos en PARALELO usando el esqueleto del planificador."""
    skeleton = state["plan_skeleton"]
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 [GENERADORES PARALELOS] Lanzando 3 workers para generar las opciones...\n")
    logger.info(f"{'='*60}")

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
    affected_days = list(state.get("_affected_days") or [])

    # [P1-SURGICAL-1] Augmentar `affected_days` con días que arrastran un
    # `_critique_unresolved` del intento previo. Helper extraído para
    # testabilidad — ver docstring de `_augment_affected_days_with_critique_markers`.
    if attempt > 1:
        affected_days = _augment_affected_days_with_critique_markers(
            affected_days,
            state.get("plan_result", {}).get("days", []) if isinstance(state.get("plan_result"), dict) else [],
        )

    surgical_mode = attempt > 1 and affected_days and len(affected_days) < len(skeleton_days[:days_in_chunk])

    recycled_days_context = ""
    recycled_days_cache = {} # day_num -> recycled_day

    if surgical_mode:
        logger.info(f"🔪 [SURGICAL FIX] Reciclando días válidos. Regenerando SOLO los días afectados: {affected_days}")
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

        # [P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] Mismo patrón que en
        # plan_skeleton: tramo DINÁMICO en `prompt_text`; tramo ESTÁTICO
        # (DAY_GENERATOR_SYSTEM_PROMPT + schema JSON) viaja en SystemMessage
        # cuando el knob está habilitado. El schema_dict de Pydantic es
        # determinístico — incluirlo en el SystemMessage lo hace cacheable
        # también (~5K tokens adicionales que cobramos solo en el primer
        # request del TTL bucket).
        dynamic_day_prompt = (
            f"Genera las comidas completas para el DÍA {day_num} del plan.\n"
            f"Semilla de generación aleatoria: {random_seed}\n\n"
            f"Información del Usuario:\n{json.dumps(_sanitize_form_data_for_prompt(form_data), indent=2)}\n"
            f"{ctx['quality_context']}\n{ctx['quality_hint_context']}\n{ctx['chunk_lessons_context']}\n{ctx['prev_chunk_adherence_context']}\n"
            f"{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
            f"{ctx['unified_behavioral_profile']}\n{ctx['correction_context']}\n{ctx['pantry_correction_context']}\n"
            f"{ctx['supplements_context']}\n{ctx['budget_context']}\n{ctx['grocery_duration_context']}\n"
            f"{ctx['pantry_context']}\n{ctx['pantry_drift_context']}\n{ctx['adherence_context']}\n{ctx['success_patterns_context']}\n"
            f"{ctx['temporal_adherence_context']}\n"
            f"{ctx['motivation_context']}\n"
            f"{ctx['sleep_stress_context']}\n"
            f"{assignment_context}\n"
            f"{recycled_days_context}\n"
        )
        if PROMPT_CACHE_SYSTEM_MESSAGE:
            prompt_text = dynamic_day_prompt
        else:
            prompt_text = dynamic_day_prompt + DAY_GENERATOR_SYSTEM_PROMPT

        # [P4-MODEL-1] Usar router especializado del day generator que
        # escala a Pro en retry cuando el rechazo previo fue skeleton
        # fidelity. Para attempt #1 o causas no-skeleton, routing default
        # (Flash si easy, Pro si clinical complex).
        day_model = _route_model_for_day_generator(
            form_data,
            attempt,
            prev_rejection_reasons=state.get("rejection_reasons") or [],
        )
        # P1-Q3: CB per-modelo. Cada día puede usar pro o flash según routing;
        # la salud de cada modelo se rastrea independientemente.
        _day_cb = _get_circuit_breaker(day_model)

        _day_llm_kwargs = dict(
            model=day_model,
            temperature=temp_override if temp_override is not None else (base_temp if attempt == 1 else (base_temp + 0.1)),
            max_retries=0,
            timeout=90,
        )
        if DAYGEN_JSON_MODE:
            # [P1-DEEPSEEK-JSON-MODE · 2026-06-13] Fuerza salida JSON válida —
            # sin esto DeepSeek emite prosa-reasoning y el parse falla.
            _day_llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        day_llm = ChatDeepSeek(**_day_llm_kwargs)

        # Enlazar herramientas (Mejora 3)
        from tools_nutrition import NUTRITION_TOOLS, consultar_nutricion
        # [P1-DEEPSEEK-JSON-MODE] JSON mode es incompatible con tool-calling
        # streaming → no bindear tools cuando está activo. La tabla nutricional
        # ya viaja en el SystemMessage cacheado, así que el modelo no la necesita.
        if DAYGEN_BIND_NUTRITION_TOOL and not DAYGEN_JSON_MODE:
            day_llm_with_tools = day_llm.bind_tools(NUTRITION_TOOLS)
        else:
            # [L1-UNBIND-NUTRITION-TOOL] Tool des-enlazada: el modelo no puede
            # invocar consultar_nutricion → cero rondas de tool (el loop rompe en
            # iter 0 vía el `else: break`); la tabla pre-computada del SystemMessage
            # es la fuente autoritativa. Sin tool schema en cada call.
            day_llm_with_tools = day_llm

        # [P1-PROMPT-CACHE-STAGGER · 2026-05-16] System instruction
        # pre-computado a nivel módulo (`_DAY_SYSTEM_INSTRUCTION_CACHED` /
        # `_DAY_SCHEMA_INSTRUCTION`). Eliminamos la generación per-call de
        # `schema_dict` que (a) gastaba ~2-3ms x día y (b) podía romper
        # byte-equivalence del SystemMessage si Pydantic reordenaba keys
        # entre llamadas — cualquier variance = cache miss garantizado.
        # `sort_keys=True` en el dumps módulo-level garantiza determinismo.
        if PROMPT_CACHE_SYSTEM_MESSAGE:
            streaming_prompt = prompt_text
            day_system_instruction = _DAY_SYSTEM_INSTRUCTION_CACHED
        else:
            streaming_prompt = prompt_text + _DAY_SCHEMA_INSTRUCTION
            day_system_instruction = None

        # P1-10: HumanMessage/AIMessage/ToolMessage están a nivel módulo

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            retry=retry_if_exception(lambda e: not _is_plan_spend_cap_error(e)),  # P2-ORCH-4
            reraise=True,
            before_sleep=lambda rs: logger.warning(f"⚠️  [DÍA {day_num}] Reintento #{rs.attempt_number}...")
        )
        async def invoke_day():
            if not await _day_cb.acan_proceed():
                raise Exception(f"Circuit Breaker OPEN para {day_model} - LLM cascade failure prevented")
            try:
                # [P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] SystemMessage al
                # inicio cuando el knob está habilitado — Gemini lo trata
                # como system_instruction separado, target del implicit cache.
                if day_system_instruction is not None:
                    messages = [
                        SystemMessage(content=day_system_instruction),
                        HumanMessage(content=streaming_prompt),
                    ]
                else:
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
                                logger.info(f"🔧 [DÍA {day_num}] Tool Call: consultar_nutricion({tool_args})")
                                try:
                                    # P0-3: Despachar la tool sync a thread pool. Aunque la
                                    # tool actual es CPU-puro (lookup en dict), el agent loop
                                    # corre con N días en paralelo + retries: cualquier
                                    # evolución a I/O (USDA API, DB externa) congelaría el
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
                # [P1-LLM-TRANSIENT-5XX · 2026-05-21] No contamines el CB con
                # errores 5xx transitorios de Google (502/503/504). Esos son
                # infra de Google teniendo problemas, no fallas del modelo.
                # Tenacity ya retry-ea el call; el CB solo debe abrir si el
                # modelo en sí (o nuestra integración) está consistentemente
                # roto. Pre-fix: 3 bursts de 502 abrían el CB → Días caían en
                # cascada al fallback emergency. Ver incidente 2026-05-21 02:58.
                # [P1-ORCH-1/2 · 2026-05-28] Helper centralizado: excluye 5xx
                # transitorios + 429 spending-cap (y activa el latch del cap).
                await _record_cb_failure_unless_transient(_day_cb, e)  # P1-Q3
                raise e

        day_result = await invoke_day()

        # Forzar day number y name correctos
        day_result["day"] = day_num
        day_result["day_name"] = day_name

        # ── Scrub determinista de proteínas restringidas no autorizadas ──
        # [PROTEIN-POOL-SCRUB 2026-05-07] Helper centralizado: limpia
        # `ingredients` y escanea `recipe` text. Devuelve (violations_count,
        # unauthorized_keys) — usamos el set de keys para construir el
        # prompt augmentado del bounded regen abajo.
        try:
            _violations_count, _unauthorized_keys = _apply_protein_pool_scrub(
                day_result, skeleton_day, day_num, context_label="PARALLEL-GEN",
            )
        except Exception as _scrub_err:
            logger.warning(
                f"⚠️ [PARALLEL-GEN/DÍA {day_num}] protein-scrub falló "
                f"(best-effort): {_scrub_err}"
            )
            _violations_count, _unauthorized_keys = 0, set()

        # [PROTEIN-RECIPE-SCAN 2026-05-07] Bounded regen forzado
        # ----------------------------------------------------------------
        # El helper limpió `ingredients` y detectó violaciones en `recipe`
        # text (que cleanup no toca para no romper bidireccionalidad). Este
        # path es ÚNICO de PARALLEL-GEN porque solo aquí tenemos `invoke_day`
        # + `streaming_prompt` para re-llamar el LLM con prompt augmentado.
        # Caso real plan 7ab9a552: critique sugirió "usa jamón de pavo" →
        # LLM nombró el plato "Salteado de Plátano Maduro y Pavo" sin añadir
        # pavo a ingredients → reviewer rechazó por bidireccionalidad rota.
        if (
            _violations_count > 0
            and _unauthorized_keys
            and not getattr(invoke_day, '_already_regen_for_pool', False)
        ):
            logger.warning(
                f"⚠️  [DÍA {day_num}] PROTEIN-RECIPE-VIOLATION → "
                f"forzando regen del día (bounded: 1 retry max)"
            )
            invoke_day._already_regen_for_pool = True

            _violated_in_recipe = sorted(_unauthorized_keys)
            _augment = (
                f"\n\n⛔ ALERTA CRÍTICA — REGEN FORZADO POR VIOLACIÓN DE POOL:\n"
                f"Tu intento anterior mencionó proteínas prohibidas en los pasos de "
                f"recetas (`recipe`) o nombre del plato. Esas proteínas NO ESTÁN en el pool del "
                f"día y son PROHIBIDO ABSOLUTO: {_violated_in_recipe}. Re-genera el día COMPLETO "
                f"eliminando cualquier mención de esas proteínas en TODAS las recetas (nombre, "
                f"pasos, ingredientes, meal_type). Usa SOLO las proteínas del pool asignado: "
                f"{', '.join(skeleton_day.get('protein_pool', []))}. Para diversificar desayuno/"
                f"merienda usa huevos, claras, queso fresco, yogurt, frutos secos o mantequilla "
                f"de maní (estas son OK siempre)."
            )
            _saved_prompt = streaming_prompt
            try:
                streaming_prompt = streaming_prompt + _augment
                day_result = await invoke_day()
                day_result["day"] = day_num
                day_result["day_name"] = day_name

                # Re-aplicar helper sobre el regen — re-cleanup ingredients
                # y re-scan recipe; el log dirá si quedó limpio o no.
                _violations_post, _ = _apply_protein_pool_scrub(
                    day_result, skeleton_day, day_num,
                    context_label="PARALLEL-GEN-REGEN",
                )
                if _violations_post > 0:
                    # [P2-PROTEIN-VIOLATION-SANITIZE · 2026-05-16] Pre-fix:
                    # se aceptaba el meal sin más → reviewer médico rechazaba
                    # el plan porque el name decía "Chuleta..." pero ingredients
                    # NO tenía chuleta → "deficiencia nutricional severa en el
                    # almuerzo del Día 2, donde solo se incluyen tostones sin
                    # una fuente de proteína adecuada" (plan_id=fbd014b2 2026-05-16).
                    # Post-fix: sanitización defensiva del name y recipe text
                    # del meal afectado, eliminando las palabras de la proteína
                    # prohibida. El meal queda con name genérico ("Plato del
                    # Almuerzo") + recipe sin referencias a la proteína prohibida,
                    # consistente con el ingredients ya strippeado.
                    _sanitized_count = _sanitize_unauthorized_protein_text(
                        day_result, _unauthorized_keys, day_num,
                    )
                    logger.warning(
                        f"⚠️  [DÍA {day_num}] PROTEIN-RECIPE-VIOLATION persistió tras regen "
                        f"({_violations_post} meals afectados); sanitizando name/recipe "
                        f"defensivamente ({_sanitized_count} text replacements) para "
                        f"mantener consistencia con ingredients strippeado."
                    )
                else:
                    logger.info(f"✅ [DÍA {day_num}] PROTEIN-RECIPE-VIOLATION resuelta tras regen forzado.")
            except Exception as _regen_e:
                logger.warning(
                    f"⚠️  [DÍA {day_num}] Regen forzado falló ({_regen_e}); "
                    f"manteniendo intento original."
                )
            finally:
                streaming_prompt = _saved_prompt

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
                    logger.info(f"🧹 [DÍA {day_num}] Unidad normalizada: '{_ing}' → '{_scrubbed}'")
                _new_ings.append(_scrubbed)
            _meal['ingredients'] = _new_ings

        day_duration = round(time.time() - day_start, 2)
        logger.info(f"✅ [DÍA {day_num}] Generado en {day_duration}s")
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
    # [P2-HEDGE-LIMITER-RAISE · 2026-05-16] Antes: hardcoded `max(1, max_concurrent // 2)`
    # que daba 2 con LLM_MAX_CONCURRENT=4. Ahora knob explícito con default 3.
    # Clamp superior preserva headroom para primaries (al menos 1 slot libre del
    # semáforo global para que un primary nuevo pueda arrancar mientras hay
    # hedges en flight). Floor 1 para que el limiter no quede en 0 si alguien
    # setea el knob a un valor extremo.
    _hedge_cap_ceiling = max(1, LLM_SEMAPHORE.max_concurrent - 1)
    HEDGE_MAX_CONCURRENT = max(1, min(HEDGE_MAX_CONCURRENT_KNOB, _hedge_cap_ceiling))
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
            logger.warning(f"⚠️ [HEDGE] Día {day_num} lleva >{elapsed}s pero "
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

        # 3. Hay slot disponible → reservar e inyectar hedge especulativo.
        #
        # [P0-7] Increment + try/finally simétricos. ANTES, el increment
        # vivía FUERA del `try` (separado por `print` + `create_task` +
        # init de `racing`/`last_exc`). Si cualquiera de esas tres líneas
        # lanzaba — `print` con codificador roto, `asyncio.create_task`
        # con `RuntimeError: no running event loop` durante teardown,
        # mismo `CancelledError` externa entrando mid-setup — el counter
        # quedaba en +1 permanente porque el `finally` que decrementa
        # nunca ejecutaba. Bajo carga sostenida con N pipelines concurrentes
        # esto saturaba `HEDGE_MAX_CONCURRENT` para SIEMPRE, forzando que
        # todos los días lentos esperaran al primary hasta `HARD_CEILING_S`
        # sin posibilidad de hedge → degradación de p99 invisible.
        # AHORA el increment es la PRIMERA línea protegida por el try, y el
        # decremento se condiciona a que el increment haya completado
        # exitosamente (defensa simétrica si list[0]+=1 fallara — no debe
        # bajo GIL pero el contrato queda explícito).
        hedge_in_flight[0] += 1
        try:
            logger.info(f"🪁 [HEDGE] Día {day_num} lleva >{elapsed}s. Lanzando intento especulativo "
                  f"({hedge_in_flight[0]}/{HEDGE_MAX_CONCURRENT} hedges activos).")
            hedge = asyncio.create_task(generate_single_day(skel_day, day_num, temp_override))

            racing = {primary, hedge}
            last_exc: Optional[Exception] = None

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
                        logger.info(f"🏁 [HEDGE] Día {day_num} ganador: {winner}")
                        return result
                    except Exception as e:
                        last_exc = e
                        name = "primary" if t is primary else "hedge"
                        logger.warning(f"⚠️ [HEDGE] Día {day_num}: {name} falló ({type(e).__name__}). Esperando al otro.")
                racing = pending

            raise last_exc or RuntimeError(f"Día {day_num}: ambos intentos fallaron sin excepción")
        finally:
            # P0-D + P0-7: liberar slot SIEMPRE (éxito, falla o cancelación)
            # para que otros días puedan hedge. El finally se ejecuta antes de
            # que cualquier excepción propague al caller.
            #
            # Sanity log: si el counter quedara en negativo (imposible bajo el
            # contrato actual, pero detectable si alguien introduce un decremento
            # extra en el futuro), avisar al SRE en lugar de silenciarlo.
            hedge_in_flight[0] -= 1
            if hedge_in_flight[0] < 0:
                logger.error(
                    "[HEDGE][INVARIANT] hedge_in_flight=%s tras decrement (esperado >= 0). "
                    "Día %s. Decremento sin increment correspondiente — bug en flow.",
                    hedge_in_flight[0], day_num,
                )

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
            logger.info(f"⚔️ [ADVERSARIAL AUTO-ACTIVATE] Activado autónomamente por: {', '.join(_auto_reasons)}")

    # --- GAP 2: Desactivar adversarial si la calibración es mala ---
    health_profile = form_data.get("health_profile", {})
    judge_calib = health_profile.get("judge_calibration", {})
    if judge_calib.get("total", 0) >= 5 and judge_calib.get("score", 1.0) < 0.5:
        logger.error("🛑 [ADVERSARIAL JUDGE] Desactivado automáticamente. Score de calibración muy bajo (< 0.5).")
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
                        logger.info(f"♻️ [SURGICAL FIX] Día {day_num} reciclado con éxito.")
                    continue
                else:
                    if temp_override is None or temp_override == 0.7:
                        logger.warning(f"⚠️ [SURGICAL FIX] No se encontró el Día {day_num} para reciclar, forzando regeneración.")

            day_coros.append(_safe_gen(skel_day, day_num, temp_override))

        failed_days = []
        if day_coros:
            if len(day_coros) == 1:
                results = [await day_coros[0]]
            else:
                # [P1-PROMPT-CACHE-STAGGER · 2026-05-16] Stagger opt-in: si
                # `DAY_GEN_CACHE_STAGGER_MS > 0`, retrasamos days 2..N para
                # que el day 1 popule el implicit cache de Gemini ANTES de
                # que los demás disparen. Resultado esperado: days 2..N
                # hitan cache → cached_tokens > 0 → input cost -25% para
                # esos N-1 días. Trade: latencia plan += (N-1) * stagger_ms.
                # Default knob = 0 (sin cambio de comportamiento legacy).
                stagger_ms = DAY_GEN_CACHE_STAGGER_MS
                if stagger_ms > 0 and len(day_coros) > 1:
                    async def _staggered(coro, delay_s):
                        if delay_s > 0:
                            await asyncio.sleep(delay_s)
                        return await coro
                    staggered_coros = [
                        _staggered(c, (i * stagger_ms) / 1000.0)
                        for i, c in enumerate(day_coros)
                    ]
                    results = await asyncio.gather(*staggered_coros)
                else:
                    results = await asyncio.gather(*day_coros)
                
            for day_num, result, err in results:
                if result is not None:
                    generated_days.append(result)
                else:
                    # [P2-HEDGE-EXC-DETAIL · 2026-05-16] Logueamos tipo + mensaje +
                    # traceback. Antes solo `type(err).__name__` ("Exception")
                    # → imposible distinguir 503 Gemini vs rate-limit vs CB-OPEN
                    # vs TimeoutError(ceiling). Incidente plan bf6f1383 perdió
                    # 8 min de diagnóstico por esto.
                    if err is not None:
                        err_name = type(err).__name__
                        err_msg = (str(err) or "(empty)")[:300]
                        logger.error(
                            f"❌ [DÍA {day_num}] Falló definitivamente tras hedging: "
                            f"{err_name}: {err_msg}",
                            exc_info=(type(err), err, err.__traceback__),
                        )
                    else:
                        logger.error(
                            f"❌ [DÍA {day_num}] Falló definitivamente tras hedging: "
                            f"unknown (no exception object captured)"
                        )
                    failed_days.append(day_num)

        if failed_days and generated_days:
            # [P2-ORCH-3 · 2026-05-28] Días fallidos → `_build_fallback_day`
            # matemático (balanceado + allergen-aware) en vez de CLONAR verbatim
            # el Día 0. El clon producía días byte-idénticos (mismas recetas/
            # ingredientes) que evadían la anti-repetición y la fidelidad de
            # skeleton; con 2/3 días fallidos el usuario veía 3 días idénticos.
            # El día math respeta las restricciones declaradas y queda marcado
            # (`_day_fallback`) para que review/scoring puedan degradar calidad.
            _restricted = _fallback_restricted_tokens(form_data)
            for f_day in failed_days:
                fb_day = _build_fallback_day(nutrition, f_day, _restricted)
                fb_day["_day_fallback"] = True
                _skel = next((d for d in skeleton_days if d.get("day") == f_day), None)
                if _skel and _skel.get("day_name"):
                    fb_day["day_name"] = _skel["day_name"]
                generated_days.append(fb_day)
                logger.warning(f"⚠️ [FALLBACK EXTREMO] Día {f_day} reemplazado por día de contingencia matemático (worker falló).")
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
        logger.info(f"⚔️ [ADVERSARIAL SELF-PLAY] Activado. Par '{_ab_pair_selected['label']}': A={_ab_temp_a} / B={_ab_temp_b}")
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
            logger.error(f"❌ [ADVERSARIAL] Ambos candidatos fallaron. "
                  f"A={type(candidate_a).__name__}, B={type(candidate_b).__name__}. "
                  "Re-raise para activar fallback global.")
            raise candidate_a  # consistente con _generate_candidate cuando todo falla
        if a_failed:
            logger.warning(f"⚠️ [ADVERSARIAL] Candidato A falló ({type(candidate_a).__name__}); "
                  "promoviendo Candidato B como único.")
            candidate_a = candidate_b
            candidate_b = None
        elif b_failed:
            logger.warning(f"⚠️ [ADVERSARIAL] Candidato B falló ({type(candidate_b).__name__}); "
                  "continuando solo con Candidato A.")
            candidate_b = None
    else:
        candidate_a = await _generate_candidate(temp_override=None)

    parallel_duration = round(time.time() - parallel_start, 2)
    logger.info(f"✅ [PARALELO] Días generados en {parallel_duration}s")

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
        logger.info(f"🔬 [AB-TEMP] Exploración: par '{selected['label']}' ({totals_raw[selected['label']]}/{_AB_MIN_SAMPLES_PER_PAIR} muestras)")
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
    logger.info(f"🎯 [AB-TEMP] Explotación: par '{selected['label']}' "
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
        logger.warning(f"⚠️ [AB-TEMP] Error leyendo historial: {e}")
        rows = []
    return _compute_ab_temp_pair_from_rows(rows)


async def _aselect_ab_temp_pair(user_id: str) -> dict:
    """P0-NEW-1.a: variante async que usa `aexecute_sql_query` para no congelar
    el event loop. Misma semántica que `_select_ab_temp_pair`. Se llama desde
    `generate_days_parallel_node` (async) cuando el adversarial self-play está
    activo. La consulta se ejecuta sobre `pipeline_metrics` (~90 filas) y
    típicamente toma 30-150ms — sync bloqueaba ese tiempo todos los SSE
    callbacks y demás coroutines del worker.

    [P1-BESTEFFORT-DB-CB · 2026-05-21] Si el pool está saturado, fail-fast con
    rows vacíos en lugar de gastar 8-12s en timeout. El AB-TEMP funciona bien
    con `rows=[]` (cae al fallback exploration uniforme).
    """
    _be_cb = _get_be_db_cb("ab_temp_async")
    if _be_cb.is_open():
        return _compute_ab_temp_pair_from_rows([])
    try:
        rows = await aexecute_sql_query(_AB_TEMP_QUERY, (user_id,), fetch_all=True) or []
        _be_cb.record_success()
    except Exception as e:
        if _is_pool_timeout_error(e):
            _be_cb.record_pool_timeout()
        logger.warning(f"⚠️ [AB-TEMP] Error leyendo historial: {e}")
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


@_node_label("judge")
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
            logger.warning("⚠️ [ADVERSARIAL JUDGE] Candidato B inválido (sin días/meals). "
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
            logger.warning("⚠️ [ADVERSARIAL JUDGE] Candidato A inválido (sin días/meals). "
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
        logger.warning("⚠️ [ADVERSARIAL JUDGE] Ambos candidatos inválidos. Devolviendo A; "
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

    logger.info(f"\n{'='*60}")
    logger.info(f"⚖️ [ADVERSARIAL JUDGE] Evaluando Candidato A vs Candidato B...")
    logger.info(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "adversarial_judging", "message": "Seleccionando el mejor plan candidato..."})
    start_time = time.time()

    # P1-Q3: capturar modelo del juez para CB per-modelo
    # [P1-FLASH-LITE-AUX-NODES · 2026-05-15] Default Flash-Lite via knob
    # `MEALFIT_JUDGE_MODEL` — tarea schema-strict (AdversarialJudgeResult).
    # Pre-fix usaba `_route_model(force_fast=False)` que escalaba a Pro en
    # perfiles clínicos; ahora unified Lite default + override por knob
    # si calidad regresiona en un perfil específico.
    _judge_model = _judge_model_name()
    _judge_cb = _get_circuit_breaker(_judge_model)
    judge_llm = ChatDeepSeek(
        model=_judge_model,
        temperature=0.2,
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
        logger.warning(f"⚠️ [ADVERSARIAL JUDGE] P1-X6: prompt {len(prompt)}c > budget "
              f"{_ADVERSARIAL_PROMPT_CHAR_BUDGET}c. Aplicando trim agresivo "
              f"(ing_cap=3, recipe_cap=60).")
        summary_a = _compress_candidate(candidate_a, ing_cap=3, recipe_cap=60)
        summary_b = _compress_candidate(candidate_b, ing_cap=3, recipe_cap=60)
        prompt = _build_judge_prompt(summary_a, summary_b)

    # Skip: si tras trim agresivo aún excede, no invocar al juez. Promover A
    # como default arbitrario (sin señal AB → `include_pair=False`).
    if len(prompt) > _ADVERSARIAL_PROMPT_CHAR_BUDGET:
        logger.error(f"🛑 [ADVERSARIAL JUDGE] P1-X6: prompt aún excede budget tras trim "
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
        logger.info(f"🏆 [ADVERSARIAL JUDGE] Ganador: {winner_key}. Razón: {rationale}")
        
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
        await _record_cb_failure_unless_transient(_judge_cb, e)  # P1-Q3 · P1-ORCH-1/2
        logger.warning(f"⚠️ [ADVERSARIAL JUDGE] Falló la evaluación: {e}. Defaulting to Candidate A.")
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
    # [P6-SLOT-CROSS-PROTEIN] 'pollo' ahora requiere "pollo" o "pechuga
    # de pollo" explícito — antes "pechuga" solo creaba colisión con
    # "pechuga de pavo" (ambos quedaban etiquetados 'pollo'), permitiendo
    # que pavo pasara desapercibido en checks de duplicación.
    "pollo": ["pollo", "pechuga de pollo", "filete de pollo"],
    # [P6-SLOT-CROSS-PROTEIN] 'pavo' añadido. Antes faltaba, lo que dejaba
    # cualquier repetición de pavo invisible al detector de slot incoherence.
    # Caso real (Día Martes): pavo en desayuno + merienda no se flagaba.
    "pavo": ["pavo", "pechuga de pavo", "carne de pavo", "pavo molido"],
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

# [P6-SLOT-CROSS-PROTEIN] Set de proteínas "pesadas" — carnes principales
# que no deberían repetirse en >1 comida del mismo día. Light proteins
# (huevo, claras, yogurt, gandules, habichuelas, lentejas) sí pueden
# aparecer en múltiples slots (ej. huevo en desayuno + huevo duro snack)
# sin ser problema.
_HEAVY_PROTEIN_LABELS = {"pollo", "pavo", "cerdo", "res", "pescado", "atun"}


def _count_cross_day_heavy_protein_repetition(days: list, min_days: int = 3) -> dict:
    """[P2-ORCH-6 · 2026-05-28] Cuenta en cuántos días DISTINTOS aparece cada
    proteína PESADA (pollo/pavo/cerdo/res/pescado/atún). Devuelve las que
    aparecen en >= `min_days` días — monotonía cross-day.

    Cierra el gap del skip-when-clean: `_count_staple_repetitions` OMITE pescado
    y carnes de su mapa de staples, y `_detect_slot_incoherence` solo mira
    overlap INTRA-día. Sin este detector, 'salmón/pescado todos los días'
    escapaba a ambos y se saltaba el evaluador subjetivo. Tooltip-anchor: P2-ORCH-6.
    """
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode("ascii")

    aliases_norm = {
        label: [_norm(a) for a in _MAIN_PROTEIN_ALIASES[label]]
        for label in _HEAVY_PROTEIN_LABELS
    }
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
    return {k: v for k, v in day_counts.items() if v >= min_days}


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

        # [P6-SLOT-CROSS-PROTEIN] 1.5: Detectar HEAVY protein duplicada
        # en >1 slot del día (no solo almuerzo+cena). Caso real (corrida
        # Día Martes): pavo aparecía en desayuno (Revoltillo de Pavo) +
        # merienda (Casabe con Pavo) — el check anterior solo veía
        # almuerzo+cena, dejando esta repetición invisible. Skip si ya
        # cubierto por el check #1 (ahorra mensaje duplicado).
        protein_in_slots: dict = {}
        for slot_name, meal in by_slot.items():
            if not meal:
                continue
            for label in _detect_main_items(meal, _MAIN_PROTEIN_ALIASES):
                if label not in _HEAVY_PROTEIN_LABELS:
                    continue
                protein_in_slots.setdefault(label, set()).add(slot_name)

        for protein_label, slots_set in protein_in_slots.items():
            if len(slots_set) < 2:
                continue
            # Skip duplicado: el check almuerzo+cena ya emitió mensaje
            # específico para ese caso.
            if slots_set == {"almuerzo", "cena"}:
                continue
            slot_list = sorted(slots_set)
            issues.append(
                f"Día {day_num}: la proteína '{protein_label}' aparece en "
                f"{len(slots_set)} comidas ({', '.join(slot_list)}). "
                f"Para mayor variedad de proteínas en el día, sustituye en "
                f"una de ellas (ej. desayuno con huevo/queso/yogurt, "
                f"merienda con yogurt+fruta o casabe+queso)."
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


@_node_label("self_critique")
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
        logger.info(f"⏭️ [SELF-CRITIQUE] Saltado en intento {state.get('attempt')} (budget-aware).")
        return {}

    logger.info(f"\n{'='*60}")
    logger.info(f"🧐 [SELF-CRITIQUE] Evaluando calidad post-generación...")
    logger.info(f"{'='*60}")
    
    _emit_progress(state, "phase", {"phase": "critique", "message": "Evaluando atractivo y coherencia del plan..."})
    start_time = time.time()

    # P1-Q3: capturar modelo del evaluator para CB per-modelo.
    # [P6-EVALUATOR-USE-PRO] Si el knob está ON, el evaluator (decisor de qué
    # días corregir) usa Pro en vez de Flash. Day generators y corrector siguen
    # en Flash — solo escalamos el "juez", no los "ejecutores".
    # [P3-SELF-CRITIQUE-LITE-COST · 2026-05-22] Si `EVALUATOR_USE_PRO` está
    # OFF, el modelo del evaluator sale de `_self_critique_model_name()` (knob
    # `MEALFIT_SELF_CRITIQUE_MODEL`), default `_FLASH_MODEL_NAME` para preservar
    # comportamiento. Pre-fix usaba `_route_model(force_fast=True)` directo —
    # equivalente porque `force_fast=True` resuelve a `_FLASH_MODEL_NAME` para
    # los perfiles no-Pro. La diferencia: ahora el knob permite swap a
    # `gemini-3.1-flash-lite` sin tocar `_route_model` (que afecta day_generators).
    if EVALUATOR_USE_PRO:
        _evaluator_model = _PRO_MODEL_NAME
        logger.info(f"🎓 [SELF-CRITIQUE] Evaluator escalado a PRO ({_PRO_MODEL_NAME}) "
              f"vía MEALFIT_EVALUATOR_USE_PRO=1 (mejor cobertura de incoherencias).")
    else:
        _evaluator_model = _self_critique_model_name()
    _evaluator_cb = _get_circuit_breaker(_evaluator_model)
    evaluator_llm = ChatDeepSeek(
        model=_evaluator_model,
        temperature=0.1,
        max_retries=1,
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
    # [P2-ORCH-6] Monotonía cross-day de proteína pesada (incluye pescado, que
    # el mapa de staples OMITE). Alimenta el gate de skip-when-clean abajo.
    heavy_protein_monotony = _count_cross_day_heavy_protein_repetition(days)
    suggested_day_hint = ""
    if staple_repetitions:
        items_str = ", ".join([f"'{k}' en {v} días" for k, v in staple_repetitions.items()])
        logger.info(f"🔁 [SELF-CRITIQUE] Staples repetidos detectados: {items_str}")
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
        logger.info(f"🍽️ [SELF-CRITIQUE] Incoherencias de slot detectadas:{joined}")
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

    # [P1-SELF-CRITIQUE-SKIP-CLEAN · 2026-05-28] Early-exit: si los detectores
    # determinísticos vinieron limpios (cero staples repetidos, cero incoherencias
    # de slot, cero monotonía cross-day de proteína pesada), saltamos el evaluador
    # LLM (~30s) y, por ende, sus correcciones (las llamadas más caras del
    # pipeline). El plan ya es estructuralmente sano; el evaluador solo aportaría
    # pulido subjetivo. Knob `MEALFIT_SELF_CRITIQUE_SKIP_WHEN_CLEAN` (default True)
    # — flip a False restaura el evaluador en cada plan.
    # [P2-ORCH-6 · 2026-05-28] `heavy_protein_monotony` añadido al gate: el skip
    # NO debe aplicar cuando una proteína pesada (p.ej. pescado/salmón) se repite
    # en >=3 días — gap que `staple_repetitions` (sin pescado) no cubría.
    # NOTA: las dimensiones puramente subjetivas del evaluador LLM (visual /
    # cultural / temperatura) se OMITEN intencionalmente cuando todo está limpio
    # (decisión de costo P1-GEN-EFFICIENCY); los detectores determinísticos son
    # el piso de calidad garantizado.
    if (SELF_CRITIQUE_SKIP_WHEN_CLEAN and not staple_repetitions
            and not slot_issues and not heavy_protein_monotony):
        logger.info(
            "⏭️ [SELF-CRITIQUE] Detectores determinísticos limpios (0 staples "
            "repetidos, 0 incoherencias de slot, 0 monotonía de proteína pesada) "
            "→ skip evaluador+correcciones (MEALFIT_SELF_CRITIQUE_SKIP_WHEN_CLEAN). "
            "Recorte de latencia/costo; plan entregado sin pulido subjetivo."
        )
        return {}

    # Brecha 4: Inyección de Contexto de Usuario
    form_data = state.get("form_data", {})
    adherence = form_data.get("_adherence_hint", "")
    emotional = form_data.get("_emotional_state", "")

    user_context = ""
    if adherence == "low":
        user_context += "\nNOTA CRÍTICA: Este usuario tiene BAJA adherencia. Un plan extremadamente simple, repetitivo y fácil de cocinar es BUENO para él. NO penalices el plan por falta de variedad excesiva o simplicidad."
    if emotional == "needs_comfort":
        user_context += "\nNOTA CRÍTICA: Este usuario reportó necesitar 'comfort food'. Platos calientes, densos y reconfortantes son POSITIVOS. NO los penalices por falta de frescura o ensaladas."

    # [P3-COST-CUT-V2 · 2026-05-21] Split del prompt para habilitar cache hit:
    # parte estática (rol + criterios) en SystemMessage cacheable; parte
    # per-plan (días + bloques determinísticos + contexto user) en HumanMessage.
    # Pre-fix: todo iba concatenado como string a `_safe_ainvoke` → cache miss
    # garantizado aunque `PROMPT_CACHE_SYSTEM_MESSAGE=True`.
    pista_dia = (
        f"Pista: empieza por {suggested_day_hint} si necesitas elegir cuál corregir primero."
        if suggested_day_hint else ""
    )
    human_content = f"""
PLAN A EVALUAR (días generados):
{days_summary_json}
{staples_block}{slot_block}{user_context}
{pista_dia}
""".strip()

    if PROMPT_CACHE_SYSTEM_MESSAGE:
        evaluator_payload = [
            SystemMessage(content=_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION),
            HumanMessage(content=human_content),
        ]
    else:
        # Legacy path: concatenar todo en un string (sin cache eligible).
        evaluator_payload = (
            _CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION + "\n\n" + human_content
        )

    try:
        if not await _evaluator_cb.acan_proceed():  # P1-Q3
            raise Exception(f"Circuit Breaker OPEN para {_evaluator_model} - LLM cascade failure prevented")
        try:
            # P0-4: Hard timeout con cancelación graceful. Self-critique es
            # opcional/quality-of-life; 30s es suficiente para Flash con
            # structured output. [P6-EVALUATOR-USE-PRO] Pro tarda más en
            # structured output (~20-40s) → bumpear a 60s cuando el knob está
            # ON para no perder evaluaciones por timeout.
            _eval_timeout = 60.0 if EVALUATOR_USE_PRO else 30.0
            critique: CritiqueEvaluation = await _safe_ainvoke(
                evaluator_llm, evaluator_payload, timeout=_eval_timeout
            )
            await _evaluator_cb.arecord_success()  # P1-Q3
        except Exception as e:
            await _record_cb_failure_unless_transient(_evaluator_cb, e)  # P1-Q3 · P1-ORCH-1/2
            raise e
        logger.info(f"📊 [SELF-CRITIQUE] Scores -> Visual: {critique.visual_score}, Diversidad: {critique.diversity_score}, Cultural: {critique.cultural_score}, Temp: {critique.temperature_score}, Slot: {critique.slot_coherence_score}")
        
        if critique.needs_correction:
            logger.warning(f"⚠️ [SELF-CRITIQUE] Problemas detectados. Sugerencias: {critique.suggestions}")

            # Parsear qué días necesitan corrección desde el texto de sugerencias.
            # P1-1: regex \d+ (no \d) — antes "Día 10" se parseaba como "Día 1".
            mentioned = list(dict.fromkeys(
                int(d) for d in _re.findall(r'[Dd]ía\s*(\d+)', critique.suggestions)
            ))
            # [P6-CRITIQUE-DAY-FLOOR] Las señales determinísticas (slot_issues,
            # staple_repetitions) se calcularon ANTES del LLM y mencionan los
            # días afectados con certeza. El LLM a veces omite días en su
            # `suggestions` (corrida 2026-05-05 19:13: LLM mencionó solo
            # [1, 3] aunque slot_issues listaba Día 2 con 'res' x3 → Día 2
            # quedó sin corregir aunque cap dinámico=3 tenía capacidad).
            # Fix: usar las señales determinísticas como FLOOR; el LLM puede
            # añadir días, pero no puede dropear días con incoherencia conocida.
            deterministic_days = list(dict.fromkeys(
                int(d) for d in _re.findall(r'[Dd]ía\s*(\d+)', "\n".join(slot_issues or []))
            ))
            if deterministic_days:
                missing = [d for d in deterministic_days if d not in mentioned]
                if missing:
                    logger.info(f"🛟 [SELF-CRITIQUE] Días con incoherencia determinística no mencionados "
                          f"por el LLM: {missing} → añadidos al floor de corrección.")
                # [P2-ORCH-5 · 2026-05-28] Floor-FIRST: los días con incoherencia
                # DETERMINÍSTICA (code-cierta) van al FRENTE para que el slice
                # `mentioned[:critique_max_days]` bajo presión de budget (cap 1-2)
                # NUNCA los descarte en favor de días que el LLM solo "mencionó".
                # Pre-fix: `mentioned + missing` los ponía al final → eran los
                # primeros en caer al recortar, anulando el floor. Tooltip-anchor: P2-ORCH-5.
                mentioned = list(dict.fromkeys(deterministic_days + mentioned))
            if not mentioned:
                mentioned = [1]  # Default: corregir día 1 si no se menciona ninguno
            critique_max_days = _compute_self_critique_max_days(state)
            logger.info(f"🔧 [SELF-CRITIQUE] Corrigiendo días afectados: {mentioned[:critique_max_days]} "
                  f"(cap dinámico={critique_max_days})")

            # P1-Q3: capturar modelo del corrector para CB per-modelo
            _corrector_model = _route_model(form_data, force_fast=True)
            _corrector_cb = _get_circuit_breaker(_corrector_model)
            corrector_llm = ChatDeepSeek(
                model=_corrector_model,
                temperature=0.3,
                max_retries=0,
                timeout=80,
            ).with_structured_output(SingleDayPlanModel)

            ctx = _build_shared_context(state)

            # Extraer asignaciones del skeleton para inyectarlas en cada corrector
            _skeleton = state.get("plan_result", {}).get("_skeleton", {})
            _skeleton_days = _skeleton.get("days", [])

            # [P1-SURGICAL-1] Helper para marcar días cuya corrección de
            # self-critique no se completó (timeout / CB-skip / error / None
            # result). El marcador `_critique_unresolved` viaja con el día en
            # `partial["days"]` → `assemble_plan_node` → `state["plan_result"]`.
            # Si `review_plan_node` no flagea estos días explícitamente (porque
            # medical reviewer usa otro lente), el siguiente retry los REGENERA
            # igualmente vía `plan_skeleton_node` que augmenta `affected_days`
            # con los días marcados. Cierra el modo de fallo del incidente
            # 2026-05-05 donde Día 2 con timeout de corrección fue reciclado y
            # arrastró el problema (pollo en receta sin estar en ingredients).
            _attempt_n_for_marker = state.get("attempt", 1)

            def _mark_critique_unresolved(day_dict, reason: str, issue_text: str):
                if not isinstance(day_dict, dict):
                    return
                day_dict["_critique_unresolved"] = {
                    "reason": reason,
                    "issue": issue_text,
                    "attempt": _attempt_n_for_marker,
                }

            async def _correct_single_day(day_num: int):
                """Devuelve `(day_num, corrected_day_or_None, failure_reason_or_None)`.

                `failure_reason` codifica el modo de fallo para que el caller
                (P4-TIMEOUT-2 circuit breaker) pueda contar timeouts y abortar
                tareas pendientes antes de quemar wall-clock en cascada.
                """
                target_day = next((d for d in days if d.get("day") == day_num), None)
                if not target_day:
                    return day_num, None, "no_target"
                if not await _corrector_cb.acan_proceed():  # P1-Q3
                    logger.warning(f"⚠️ [SELF-CRITIQUE] Circuit Breaker OPEN ({_corrector_model}). Saltando corrección Día {day_num}.")
                    _mark_critique_unresolved(target_day, "cb_open", critique.suggestions or "")
                    return day_num, None, "cb_open"
                try:
                    logger.info(f"🔧 [SELF-CRITIQUE] Corrigiendo Día {day_num}...")

                    # Incluir asignación del skeleton para que el corrector respete proteínas/carbos
                    skeleton_day = next((d for d in _skeleton_days if d.get("day") == day_num), {})
                    skeleton_block = ""
                    if skeleton_day:
                        from prompts.day_generator import build_day_assignment_context
                        skeleton_block = f"\n⚠️ ASIGNACIÓN OBLIGATORIA DEL PLANIFICADOR (no la ignores):\n{build_day_assignment_context(skeleton_day, day_num)}"

                    # [P5-PROMPT-D] Usa `nutrition_context_minimal` en vez del
                    # full context — el corrector solo necesita targets duros,
                    # no kinematics/adaptación. Reduce tokens ~30%, traduce a
                    # ~20-30s menos bajo provider load (caso real Día 3 04:14
                    # tomó 143s contra cap 150s — sin margen).
                    #
                    # [P6-CRITIQUE-VS-SKELETON] Regla de precedencia añadida
                    # explícitamente. Caso real corrida 14:53: critique pidió
                    # "cambiar proteína de cena (pavo) por res/pescado" para
                    # resolver slot violation almuerzo↔cena. Corrector obedeció
                    # → removió pavo → skeleton fidelity validator rechazó HIGH
                    # ("Día 3 omitió pavo asignado"). Loop sin salida: 2/2
                    # attempts fallaron idéntico, plan tolerado entregado con
                    # violación. La regla explícita instruye al LLM: skeleton
                    # GANA, resolver slot por OTRO medio (cambiar carbo,
                    # técnica, vegetal — NUNCA la proteína asignada).
                    correction_prompt = f"""Eres un nutricionista chef. Corrige SOLO el Día {day_num} del plan alimenticio.

PROBLEMA DETECTADO: {critique.suggestions}

RESTRICCIONES NUTRICIONALES (respétalas siempre):
{ctx['nutrition_context_minimal']}
{skeleton_block}
REGLA DE PRECEDENCIA INVIOLABLE (si hay conflicto, gana esta):
- La ASIGNACIÓN DEL PLANIFICADOR es HARD CONSTRAINT — NUNCA la violes aunque el critique pida cambiar una proteína/carbohidrato asignado.
- Si el critique sugiere cambiar la proteína de almuerzo o cena (slot coherence violation), pero esa proteína FUE ASIGNADA por el planificador para este día, MANTÉN la proteína y resuelve la coherencia de slot por OTRO medio:
  • Cambia el CARBOHIDRATO de la cena (yuca→batata, arroz→ñame, papas→casabe).
  • Cambia la TÉCNICA de cocción (a la plancha→guisada→al horno→ceviche).
  • Cambia el VEGETAL/acompañamiento (ensalada→sopa, fresca→cocida).
  • Cambia la PRESENTACIÓN (bowl→wrap, plato→pita).
- Solo si NINGUNA proteína está duplicada en el skeleton (ej. skeleton dice "pavo" Y "queso") puedes alternarlas entre slots.

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
                    # [P2-SKELETON-AB-TELEMETRY · 2026-05-28] Tag distinto en
                    # `llm_usage_events.node` para la corrección (vs el evaluador,
                    # ambos vivían bajo `node='self_critique'`). Permite el A/B del
                    # modelo del skeleton (knob `MEALFIT_PLANNER_MODEL`): correlacionar
                    # `node='planner'` model ↔ tasa de `node='self_critique_correction'`
                    # por plan_id, midiendo si un skeleton más fuerte reduce las
                    # correcciones (el paso post-gen más caro). Set+reset task-local;
                    # el emit ocurre dentro de `_safe_ainvoke` antes del return.
                    _crit_node_token = _current_node_var.set("self_critique_correction")
                    try:
                        corrected_result: SingleDayPlanModel = await _safe_ainvoke(
                            corrector_llm, correction_prompt, timeout=CRITIQUE_FIX_TIMEOUT_S
                        )
                    finally:
                        _current_node_var.reset(_crit_node_token)
                    await _corrector_cb.arecord_success()  # P1-Q3
                    if corrected_result:
                        corrected_day = corrected_result.model_dump()
                        corrected_day["day"] = day_num
                        # [P3-SKELETON-FIDELITY-CRITIQUE-AWARE · 2026-05-16]
                        # Marca el día como modificado por self_critique. El
                        # check de SKELETON FIDELITY en `_run_assembly_validations`
                        # usa este marker para aplicar threshold más permisivo
                        # (>=3 missing proteins en lugar de >=2): self_critique
                        # puede legítimamente reemplazar 1-2 proteínas para
                        # mejorar diversidad/slot coherence; solo si TODAS las
                        # proteínas asignadas se removieron hay bug real.
                        # Incidente 2026-05-16 plan post-reset: critique sugirió
                        # "Cambiar las claras de la cena por queso" → LLM removió
                        # Soya/Tofu Y Queso Mozzarella → SKELETON FIDELITY
                        # rechazó fatal → plan crítico abortado.
                        corrected_day["_critique_applied"] = True
                        logger.info(f"✅ [SELF-CRITIQUE] Día {day_num} corregido exitosamente.")
                        # [PROTEIN-POOL-SCRUB 2026-05-07] Aplicar cleanup +
                        # scan tras corrección — el LLM puede reintroducir
                        # proteínas fuera del pool al "resolver" el slot
                        # collision sugerido por el critique. Mismo helper
                        # que la generación inicial para paridad.
                        try:
                            _apply_protein_pool_scrub(
                                corrected_day, skeleton_day, day_num,
                                context_label="CRITIQUE-FIX",
                            )
                        except Exception as _scrub_err:
                            logger.warning(
                                f"⚠️ [CRITIQUE-FIX/DÍA {day_num}] protein-scrub falló "
                                f"(best-effort): {_scrub_err}"
                            )
                        # [P1-SURGICAL-1] Corrección exitosa → el day NUEVO
                        # (`corrected_day`) reemplaza al anterior en `days[i]`
                        # (line ~5054), así que cualquier marcador previo en
                        # `target_day` queda descartado naturalmente.
                        return day_num, corrected_day, None
                    # `corrected_result` None: LLM retornó respuesta no-parseable
                    # tras `_safe_ainvoke`. No es timeout ni excepción, pero el
                    # día sigue sin corrección — mismo modo de fallo.
                    logger.warning(f"⚠️ [SELF-CRITIQUE] Día {day_num}: corrector LLM retornó None. Manteniendo original.")
                    _mark_critique_unresolved(target_day, "llm_returned_none", critique.suggestions or "")
                    return day_num, None, "llm_returned_none"
                except asyncio.TimeoutError:
                    # [P6-TIMEOUT-DIAG] Log diagnóstico del prompt y target_day
                    # cuando un día timeoutea para detectar patrones (ej. Día 2
                    # consistentemente más lento por densidad de ingredientes,
                    # sugerencias verbosas, etc.). Tamaño en chars: el LLM
                    # latency correlaciona con input tokens.
                    _prompt_chars = len(correction_prompt)
                    _target_chars = len(json.dumps(target_day, ensure_ascii=False))
                    _suggestion_chars = len(critique.suggestions or "")
                    _ingredients_count = sum(
                        len(m.get("ingredients", [])) for m in target_day.get("meals", [])
                    )
                    logger.info(
                        f"⏱️ [SELF-CRITIQUE] Timeout corrigiendo Día {day_num} "
                        f"({CRITIQUE_FIX_TIMEOUT_S:.0f}s) con {_corrector_model}."
                    )
                    logger.info(
                        f"📐 [P6-TIMEOUT-DIAG] Día {day_num} sizes: "
                        f"prompt={_prompt_chars}c, target_day={_target_chars}c, "
                        f"suggestion={_suggestion_chars}c, "
                        f"ingredients={_ingredients_count}"
                    )
                    # [P2-LLM-TIMEOUT-PIPELINE-METRICS · 2026-05-15] Tick a
                    # `pipeline_metrics(node='self_critique_correction_timeout')`
                    # con día + model + dimensiones del prompt para
                    # correlación post-hoc. Best-effort (helper silencia
                    # excepciones internas — un fallo de DB NO bloquea el
                    # PRO fallback que sigue).
                    _emit_llm_timeout_metric(
                        node="self_critique_correction_timeout",
                        timeout_threshold_s=float(CRITIQUE_FIX_TIMEOUT_S),
                        actual_wait_s=float(CRITIQUE_FIX_TIMEOUT_S),
                        llm=corrector_llm,
                        extra_metadata={
                            "day_num": int(day_num),
                            "prompt_chars": int(_prompt_chars),
                            "target_chars": int(_target_chars),
                            "suggestion_chars": int(_suggestion_chars),
                            "ingredients_count": int(_ingredients_count),
                            "model_label": _corrector_model,
                        },
                    )
                    pro_corrected, pro_reason = await _attempt_pro_critique_correction(
                        correction_prompt, day_num,
                        log_prefix="[SELF-CRITIQUE/PRO-FALLBACK]",
                    )
                    if pro_corrected:
                        return day_num, pro_corrected, None
                    _mark_critique_unresolved(
                        target_day, f"timeout+{pro_reason}", critique.suggestions or ""
                    )
                    return day_num, None, "timeout"
                except Exception as e:
                    await _record_cb_failure_unless_transient(_corrector_cb, e)  # P1-Q3 · P1-ORCH-1/2
                    logger.warning(f"⚠️ [SELF-CRITIQUE] Error corrigiendo Día {day_num}: {e}. Manteniendo original.")
                    _mark_critique_unresolved(target_day, f"error:{type(e).__name__}", critique.suggestions or "")
                    return day_num, None, f"error:{type(e).__name__}"

            days_to_fix = mentioned[:critique_max_days]

            # ============================================================
            # [P4-TIMEOUT-2] Circuit breaker en self-critique correction
            # ------------------------------------------------------------
            # Antes: `asyncio.gather(*[_correct_single_day(d) ...])` esperaba
            # a TODAS las correcciones aunque el provider estuviera saturado.
            # En cascade-failure (Gemini overload) los 3 días timeoutean en
            # paralelo a 120s — 0 días corregidos por el costo de 1 timeout
            # de wall-clock.
            #
            # Ahora: scheduling con `asyncio.wait(FIRST_COMPLETED)`. Cuando
            # acumulamos `CRITIQUE_TIMEOUT_ABORT_THRESHOLD` timeouts, cancelamos
            # las tareas pendientes — su día queda con marker `_critique_unresolved`
            # ("cb_aborted_provider_overload") y P1-SURGICAL-1 lo regenera en retry.
            #
            # Sigue costando 1 ciclo de timeout en el peor caso (los timeouts
            # son simultáneos), pero evita el escenario donde 1 día corregido
            # válido quedaría enmascarado por 2 timeouts paralelos. La única
            # asimetría es: si threshold=2 y al primer timeout aún hay 1 task
            # pendiente, esperamos su resultado antes de evaluar (puede ser
            # éxito y entonces no abortamos).
            # ============================================================
            tasks_by_day: dict = {}
            for d in days_to_fix:
                tasks_by_day[asyncio.ensure_future(_correct_single_day(d))] = d

            correction_results: list = []
            timeout_count = 0
            aborted_pending = False
            pending = set(tasks_by_day.keys())

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    try:
                        result = await finished
                    except Exception as e:
                        # Defensa: `_correct_single_day` ya captura todo y
                        # devuelve tuple. Si por bug futuro escapa una excepción,
                        # la tratamos como fallo del día asociado.
                        day_num = tasks_by_day[finished]
                        logger.warning(f"⚠️ [SELF-CRITIQUE] Excepción no esperada en task del Día {day_num}: {e}")
                        result = (day_num, None, f"unhandled:{type(e).__name__}")
                    correction_results.append(result)
                    _, _, fail_reason = result
                    if fail_reason == "timeout":
                        timeout_count += 1

                if (
                    timeout_count >= CRITIQUE_TIMEOUT_ABORT_THRESHOLD
                    and pending
                    and not aborted_pending
                ):
                    aborted_count = len(pending)
                    logger.info(
                        f"⚡ [SELF-CRITIQUE/CB] {timeout_count} timeouts detectados "
                        f"(threshold={CRITIQUE_TIMEOUT_ABORT_THRESHOLD}) → abortando "
                        f"{aborted_count} task(s) pendiente(s) (provider overload pattern)."
                    )
                    aborted_pending = True
                    for p in pending:
                        p.cancel()
                        aborted_day = tasks_by_day[p]
                        target_day = next(
                            (d for d in days if d.get("day") == aborted_day), None
                        )
                        if target_day is not None:
                            _mark_critique_unresolved(
                                target_day,
                                "cb_aborted_provider_overload",
                                critique.suggestions or "",
                            )
                        correction_results.append(
                            (aborted_day, None, "cb_aborted_provider_overload")
                        )
                    # Drenar las tasks canceladas para no dejar warnings de
                    # "Task was destroyed but it is pending".
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    pending = set()

            corrected_any = False
            for day_num, corrected_day, _fail_reason in correction_results:
                if corrected_day is not None:
                    for i, d in enumerate(days):
                        if d.get("day") == day_num:
                            days[i] = corrected_day
                            break
                    corrected_any = True

            if corrected_any:
                partial["days"] = days
        else:
            logger.info("✅ [SELF-CRITIQUE] El plan pasó la evaluación visual y de coherencia.")
            
    except Exception as e:
        # [P1-ORCH-3 · 2026-05-28] NO re-raise. self_critique es un nodo de
        # PULIDO no-esencial (se salta en retries y en planes limpios; sus skip
        # paths hacen `return {}` sin lanzar). En este punto `partial` YA contiene
        # el plan completo del day-generator. Abortar TODO el grafo por un fallo
        # del evaluador/corrector escalaba "saltar el pulido" a la degradación
        # global pesada (logs CRITICAL + riesgo de entregar un fallback matemático
        # cuando ya había un plan válido en mano). Degradamos localmente: se
        # conservan los días ya generados y deciden los gatekeepers reales
        # (review_plan médico + coherence guard de assemble). `CancelledError`
        # NO hereda de Exception en Py3.8+, así que la cancelación nunca se tragó.
        # Tooltip-anchor: P1-ORCH-3.
        logger.warning(
            f"⚠️ [SELF-CRITIQUE] Error en evaluación/corrección (no-fatal, se "
            f"conserva el plan generado): {type(e).__name__}: {e}"
        )
        return {"plan_result": partial}

    duration = round(time.time() - start_time, 2)
    logger.info(f"⏱️ [SELF-CRITIQUE] Completado en {duration}s")
    
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


# [P0-6] Helper compartido para las validaciones críticas post-assembly que
# DEBEN ejecutarse SIEMPRE, independiente de si el plan vino del cache
# semántico (`semantic_cache_hit=True`) o del path LLM normal.
#
# El audit P0-6 identificó el riesgo de divergencia: en cache-hit, `skeleton`
# es `{}` (los días vienen completos del cache, no hay skeleton intermedio),
# y un futuro refactor que agregue una validación entre la rama if/else y
# este punto podría olvidar aplicarla a cache-hit. Centralizar las 3
# validaciones (skeleton fidelity, recipe coherence, schema validation)
# en un único helper hace el contrato explícito y testeable en aislamiento.
#
# Reglas:
#   1. Skeleton Fidelity — solo aplica si hay skeleton (path LLM). Para
#      cache-hit es trivial-pass: el plan cacheado ya pasó la validación
#      cuando se generó originalmente.
#   2. Recipe Coherence — aplica SIEMPRE: detecta inconsistencias receta↔
#      ingredientes en cache-hit (humanización corrompiendo, schema viejo).
#   3. Schema Validation `PlanModel` — aplica SIEMPRE: si el plan cacheado
#      tiene shape vieja (campo renombrado, schema bumped sin invalidar
#      entradas en vuelo, validación nueva añadida después), `_schema_invalid`
#      se setea y `review_plan_node` lo eleva a critical → guardrail entrega
#      fallback matemático.
_SKELETON_PROTEIN_MODIFIER_STOPWORDS = {
    "proteina", "proteína", "principal", "secundaria", "entero", "enteros",
    "entera", "enteras", "molida", "molido", "molidas", "fresco", "fresca",
    "frescos", "frescas", "cocida", "cocido", "cocidas", "cocidos", "asado",
    "asada", "guisado", "guisada", "del", "con", "sin", "las", "los",
}


def _skeleton_protein_present(assigned_label: str, ingredients_text: str) -> bool:
    """[P0-SKELETON-FIDELITY-MATCH · 2026-06-13] ¿La proteína que el skeleton
    asignó al día aparece en los ingredientes? El `protein_pool` del skeleton
    trae etiquetas con descriptores y alternativas ("lentejas (proteína
    principal)", "huevos enteros / claras", "maní / mantequilla de maní") que
    NUNCA aparecen verbatim en los ingredientes ("lentejas", "huevos", "maní").
    El substring directo (`label in text`) daba FALSO POSITIVO de "omitió" aunque
    el día SÍ incluía la proteína → rechazos HIGH espurios + entrega degradada
    (`plan_quality_degraded`). Este matcher quita "(...)", parte alternativas en
    "/", y exige match (con frontera de palabra) del token-núcleo de la proteína,
    ignorando modificadores. Preserva la detección real (día que de verdad omite
    la proteína asignada: ningún token matchea → sigue flageado).

    Anchor: P0-SKELETON-FIDELITY-MATCH.
    """
    import re as _re2
    base = _re2.sub(r"\([^)]*\)", " ", (assigned_label or "").lower())
    for alt in base.split("/"):
        # Token-núcleo de la proteína (≥3 chars, sin modificadores), con frontera
        # de palabra para no matchear substrings ("pollo" NO debe matchear "repollo").
        toks = [
            t for t in _re2.findall(r"[a-záéíóúñ]+", alt)
            if len(t) >= 3 and t not in _SKELETON_PROTEIN_MODIFIER_STOPWORDS
        ]
        if any(_re2.search(r"\b" + _re2.escape(t) + r"\b", ingredients_text) for t in toks):
            return True
    return False


# [P3-MACRO-SOLVER · 2026-06-13] Knob del "cerebro dividido" (lado determinista):
# re-escala porciones para clavar macros desde master_ingredients. Default False
# (rollout gradual; requiere la DB de macros poblada — ver food_db_integration.md).
MACRO_SOLVER_ENABLED = _env_bool("MEALFIT_MACRO_SOLVER_ENABLED", False)

# [P3-PROTEIN-TOPUP · 2026-06-13] Tras el escalado del solver, si una comida sigue corta
# de proteína, añade una porción de la proteína más magra del pool aprobado del día para
# cerrar el gap (el escalado no crea proteína inexistente; esto sí). Solo activo con el
# solver ON. Flip a False revierte a solo-escalado.
MACRO_SOLVER_PROTEIN_TOPUP = _env_bool("MEALFIT_MACRO_SOLVER_PROTEIN_TOPUP", True)

# [P3-CAL-RECONCILE · 2026-06-13] Paso final del cerebro dividido: nivela las calorías
# de cada día EXACTAMENTE al target re-escalando porciones + macros uniformemente (el
# escalado uniforme preserva la consistencia receta↔macro). Cierra `cal_score` del
# holistic (= max(0, 1 − desviación×5)) → 1.0 cuando la desviación es ~0. Solo con el
# solver ON. Clamp de seguridad para no producir porciones absurdas.
MACRO_SOLVER_CAL_RECONCILE = _env_bool("MEALFIT_MACRO_SOLVER_CAL_RECONCILE", True)

# [P3-RECIPE-COHERENCE-AUTOFIX · 2026-06-13] Auto-fix determinista de las violaciones de
# coherencia receta↔ingrediente que HOY rechazan+reintentan (→ retry_penalty < 1.0 en el
# holistic). Caso "forward" (la receta menciona una proteína que no está listada → la
# scrubea/reemplaza por la proteína real del meal) + "completion" (falta paso final → lo
# añade). El caso "reverse" ya lo auto-parchea review_plan_node. Convierte reject-and-retry
# en fix-in-place → el revisor aprueba al 1er intento. Flip a False revierte a rechazar.
RECIPE_COHERENCE_AUTOFIX = _env_bool("MEALFIT_RECIPE_COHERENCE_AUTOFIX", True)

# [C2-ALLERGEN-GUARD · 2026-06-13] Backstop DETERMINISTA de alérgenos en review_plan_node:
# escanea los ingredientes vs las alergias declaradas (+ sinónimos) y rechaza-duro
# (severity critical → regen) si encuentra uno. Defensa-en-profundidad sobre el revisor
# LLM (que puede fallar). Crítico para uso clínico. Flip a False solo para debug.
ALLERGEN_HARD_GUARD = _env_bool("MEALFIT_ALLERGEN_HARD_GUARD", True)

# [P0-ALLERGEN-SUBS · 2026-06-14] Sustitución QUIRÚRGICA de alérgenos IgE declarados en la capa
# clínica (antes del review): para los alérgenos con un reemplazo seguro que RESUELVE al catálogo
# (pescado/mariscos/soya→pollo; gluten→casabe/maíz/arroz), swap del ingrediente ofensor in-place
# (preserva cantidad + recalcula macros por delta) conservando el plan rico del LLM, en vez de
# nukear todo a fallback. El backstop `_scan_allergen_violations` (ALLERGEN_HARD_GUARD) queda como
# red de seguridad post-swap → cero regresión de seguridad. Flip a False revierte al comportamiento
# previo (detección→rechazo crítico→fallback). Lácteos/huevo/maní/frutos secos NO se sustituyen
# (sin target libre del alérgeno que resuelva en el catálogo es-DO) → siguen por el path crítico.
ALLERGEN_SUBSTITUTION_ENABLED = _env_bool("MEALFIT_ALLERGEN_SUBSTITUTION", True)

# [P3-CLOSER-EGG-BUDGET · 2026-06-14] El protein-closer (P3-PROTEIN-FLOOR) elige clara/huevo como la
# proteína MAGRA de las comidas ligeras (están en _DAIRY_EGG_PROTEIN_HINT y tienen alta proteína/kcal)
# → AÑADE huevo de relleno y empuja el conteo sobre el cap del VARIETY_HARD_GATE, que luego rechaza el
# plan por sobreuso → se entrega marcado-degradado (medido en vivo: huevo en 5/12 comidas). Este
# presupuesto hace al closer consciente del cap: una vez que el huevo llega al cap (mismo cálculo que
# el gate), le pasa candidatos SIN huevo → diversifica con yogur/queso/whey. Complementa al closer
# (cantidad de proteína) SIN tocar su lógica de selección/dish-fit; cuenta el huevo del LLM + el que el
# closer añade. NO toca el huevo-protagonista (revoltillo/tortilla — eso lo pone el LLM). Flip a False
# revierte al comportamiento previo.
CLOSER_EGG_BUDGET_ENABLED = _env_bool("MEALFIT_CLOSER_EGG_BUDGET", True)

# [P3-FOOD-SAFETY · 2026-06-13] Guard determinista de seguridad alimentaria en assemble:
# detecta huevo (TCS) crudo/poco cocido (batido licuado o sin paso de cocción) e inyecta una
# nota de seguridad accionable a la receta (macro-preservante, no rompe coherencia receta↔
# lista). Cierra el hallazgo CRÍTICO de la auditoría clínica. Flip a False solo para debug.
FOOD_SAFETY_GUARD = _env_bool("MEALFIT_FOOD_SAFETY_GUARD", True)

# [P3-PORTION-QUANTIZE · 2026-06-13] Redondea las porciones del plan a unidades de cocina
# medibles (¼ taza, ¼ cda, ½ unidad discreta, 5 g) ajustando los macros por el delta exacto.
# Cierra el hallazgo de la auditoría: las fracciones decimales no medibles ('0.66 huevos',
# '0.53 taza', '3.87 papas') matan la adherencia. Corre SIEMPRE (las fracciones vienen del
# LLM y/o del solver). Trade-off: pequeña deriva del target (medibilidad > precisión exacta).
PORTION_QUANTIZE_ENABLED = _env_bool("MEALFIT_PORTION_QUANTIZE", True)

# [P3-SLOT-DISTRIBUTION · 2026-06-13] El solver usa el split FISIOLÓGICO canónico por slot
# (desayuno 20% / almuerzo 35% / merienda 15% / cena 30%) como target de cada comida, en vez
# del cal_share desbalanceado del LLM. Cierra el hallazgo de la auditoría: el desayuno
# concentraba 48% kcal / 62% proteína. Flip a False → vuelve al cal_share del LLM (legacy).
SLOT_DISTRIBUTION_ENABLED = _env_bool("MEALFIT_SLOT_DISTRIBUTION", True)

# [P3-MICRONUTRIENTS · 2026-06-13] Computa el panel de micros del plan (sodio/fibra/azúcar/
# vit D/calcio/hierro/B12/potasio) vs DRI/WHO e inyecta un reporte ADVISORY a `result`
# (con sugerencia de suplemento para los gaps estructurales). NO es un gate duro → cero
# loops de regen. Cierra el hallazgo de la auditoría. Flip a False → no computa el reporte.
MICRONUTRIENT_REPORT_ENABLED = _env_bool("MEALFIT_MICRONUTRIENT_REPORT", True)

# [P3-VARIETY · 2026-06-13] Reporte advisory de variedad/pertinencia cultural (huevo,
# 'cremoso', premium, plato-base repetido intra-día). NO bloquea (calidad blanda). El lever
# duro es el prompt (regla de variedad+fidelidad cultural); esto da observabilidad. Cierra FS5.
VARIETY_REPORT_ENABLED = _env_bool("MEALFIT_VARIETY_REPORT", True)

# [P3-SUPPLEMENT-ADVICE · 2026-06-13] A partir de los gaps del panel de micros (vit D/calcio/
# hierro/B12), genera recomendaciones de suplementación accionables (suplemento+dosis+alimentos
# +precaución). Cierra honestamente lo que los alimentos enteros rara vez alcanzan. Advisory.
SUPPLEMENT_ADVICE_ENABLED = _env_bool("MEALFIT_SUPPLEMENT_ADVICE", True)

# [P3-PRO-REVIEW-FLAG · 2026-06-13] Si el usuario declaró condiciones médicas reales, marca el
# plan con `requires_professional_review` + nota prominente. Honestidad/seguridad clínica.
PRO_REVIEW_FLAG_ENABLED = _env_bool("MEALFIT_PRO_REVIEW_FLAG", True)

# [P3-VARIETY-HARD-GATE · 2026-06-13] Convierte el cap de huevo (FS5 advisory) en restricción
# DURA: review_plan_node rechaza+reintenta (acotado: 1 retry, luego entrega) si el huevo supera
# el cap+threshold. El lever upstream es el prompt; esto da presión de retry. Flip a False → advisory.
VARIETY_HARD_GATE_ENABLED = _env_bool("MEALFIT_VARIETY_HARD_GATE", True)
VARIETY_HARD_GATE_EGG_SLACK = _env_int("MEALFIT_VARIETY_HARD_GATE_EGG_SLACK", 1)  # rechaza si egg > cap+slack

# [P3-PROTEIN-FLOOR · 2026-06-13] Cierre DURO del déficit de proteína (la re-auditoría del
# plan fresco encontró que se entregaba 68% del target). El closer rellena cada comida a
# `fill_pct` del target de proteína del slot con proteína de alta densidad allergen-safe
# integrada; el reconcile protein-preserving nivela kcal escalando SOLO carbos/grasas. El
# validador duro (review_plan_node) rechaza/reintenta si el día queda < floor_pct del target.
PROTEIN_FLOOR_ENABLED = _env_bool("MEALFIT_PROTEIN_FLOOR", True)
PROTEIN_FLOOR_FILL_PCT = _env_float("MEALFIT_PROTEIN_FLOOR_FILL_PCT", 0.92)  # closer determinista
PROTEIN_FLOOR_HARD_PCT = _env_float("MEALFIT_PROTEIN_FLOOR_HARD_PCT", 0.90)  # gate de retry
PROTEIN_FLOOR_HARD_GATE = _env_bool("MEALFIT_PROTEIN_FLOOR_HARD_GATE", True)
# [P3-PROTEIN-CEILING-GOAL-AWARE · 2026-06-13] Techo de proteína ENTREGADA en g/kg, dependiente
# del objetivo: ≤2.2 g/kg para volumen/mantenimiento (más proteína desplaza carbos útiles), pero
# hasta ~2.6 g/kg en DÉFICIT (la evidencia respalda proteína alta para preservar músculo al perder
# grasa). El trim por-día usa este techo absoluto en vez de un % fijo del target.
PROTEIN_GKG_CEILING_DEFAULT = _env_float("MEALFIT_PROTEIN_GKG_CEILING_DEFAULT", 2.2)
PROTEIN_GKG_CEILING_CUT = _env_float("MEALFIT_PROTEIN_GKG_CEILING_CUT", 2.6)

# [P3-CONDITION-RULES · 2026-06-14] Reglas clínicas deterministas para el set Pareto cardio-
# metabólico DR. Empieza con las DOS condiciones que el piso regulatorio más estricto (Medicare
# MNT reembolsable) cubre: diabetes T2 y ERC. Cada una se modela como CONSTRAINT determinista
# (no como confianza en el LLM), anclada a guía citable:
#   • DM2  → ADA 2025/2026 ABANDONÓ el % fijo de carbos y el índice glucémico → "calidad del
#            carbohidrato": fibra ≥14 g/1000 kcal + granos integrales + sin bebidas azucaradas.
#   • ERC  → KDIGO 2024: proteína TECHO ~0.8 g/kg (G3-G5 no-diálisis) — lo OPUESTO al piso alto
#            del producto. Generar un plan renal "correcto" requiere estadio/diálisis/K/P que NO
#            modelamos → se trata como cap de SEGURIDAD + gate de derivación (FS9), no prescripción.
CONDITION_RULES_ENABLED = _env_bool("MEALFIT_CONDITION_RULES", True)
RENAL_PROTEIN_GKG_CEILING = _env_float("MEALFIT_RENAL_PROTEIN_GKG", 0.8)   # KDIGO 2024 G3-G5 no-diálisis
DM2_FIBER_G_PER_1000KCAL = _env_float("MEALFIT_DM2_FIBER_PER_1000KCAL", 14.0)  # ADA 2026 calidad de carbo
# [P3-CONDITION-RULES · 2026-06-14] DM2: el revisor LLM escala preocupaciones glucémicas (miel,
# plátano grande) a 'critical' → fallback matemático que pierde el plan real + la fibra ADA. Para
# diabéticos sin violación de alérgeno/schema, la preocupación glucémica es de CALIDAD (no peligro
# agudo): se degrada a 'high' (entrega el plan real con advertencia + retry, no fallback). Flip a
# False revierte al fallback duro. Allergen/schema/renal criticals NUNCA se degradan.
DM2_GLYCEMIC_SOFT_REJECT = _env_bool("MEALFIT_DM2_GLYCEMIC_SOFT_REJECT", True)
# [P3-CONDITION-RULES · 2026-06-14] Guard determinista anti-azúcar-añadido para diabéticos (patrón
# food-safety del huevo crudo): sustituye miel/sirope/azúcar/bebidas azucaradas por stevia/agua en
# el plato ANTES del review → el plan es clínicamente mejor (sin pico glucémico) Y deja de gatillar
# el rechazo del revisor. Macro-preservante (conservador: el diabético recibe ≤ carbos que la
# etiqueta) + shopping-consistente (reemplaza el token en ingredients e ingredients_raw). Flip a False desactiva.
DM2_SUGAR_GUARD = _env_bool("MEALFIT_DM2_SUGAR_GUARD", True)

# [P3-DATA-PROVENANCE · 2026-06-14] (Roadmap M1, quick-win) Anclaje de proveniencia: computa qué
# fracción de los ingredientes del plan está trazada a USDA FoodData Central (columna `fdc_id`, ya
# poblada) y expone un sample de IDs en `result["data_provenance"]` → el PDF/UI muestra "Datos: USDA
# FDC #XXXXX". Convierte la banda de precisión AUTOAFIRMADA en trazabilidad verificable de terceros.
DATA_PROVENANCE_ENABLED = _env_bool("MEALFIT_DATA_PROVENANCE", True)

# [P3-FALLBACK-CLINICAL-LAYER · 2026-06-14] Aplica la capa clínica determinista (FS1-FS9) al plan de
# fallback matemático, que bypassa assemble y por tanto la perdía. Default True (el fallback debe
# heredar food-safety + sustitución de sodio/azúcar + micros + gate FS9). Knob de rollback sin redeploy
# si correr la capa sobre meals-plantilla matemáticas alguna vez se porta mal en prod.
FALLBACK_CLINICAL_LAYER_ENABLED = _env_bool("MEALFIT_FALLBACK_CLINICAL_LAYER", True)

# [P4-SCOREBOARD · 2026-06-14] Banda de tolerancia del `clinical_band_score` (precisión por-plan medida,
# no autoafirmada). Macros en [lower, upper] × target; kcal usa [0.95, 1.05] fijo (el reconcile las clava).
# Default [0.90, 1.12] = la banda clínica documentada (piso de proteína 90%, techo ~112%).
BAND_SCORE_LOWER = _env_float("MEALFIT_BAND_SCORE_LOWER", 0.90)
BAND_SCORE_UPPER = _env_float("MEALFIT_BAND_SCORE_UPPER", 1.12)


def _protein_gkg_ceiling(goal) -> float:
    """[P3-PROTEIN-CEILING-GOAL-AWARE] Techo de proteína entregada (g/kg) por objetivo: déficit/
    pérdida de grasa → más alto (preservación muscular); volumen/mantenimiento → 2.2."""
    g = str(goal or "").strip().lower()
    is_cut = any(t in g for t in ("lose_fat", "lose_weight", "fat_loss", "deficit", "déficit",
                                  "perdida", "pérdida", "cut", "definir", "adelgaz"))
    return PROTEIN_GKG_CEILING_CUT if is_cut else PROTEIN_GKG_CEILING_DEFAULT


def _weight_kg_from_form(form_data: dict) -> float:
    """Peso en kg desde form_data (convierte lb→kg). 0.0 si ausente/no parseable."""
    try:
        w = float(form_data.get("weight") or 0)
        unit = str(form_data.get("weightUnit") or "kg").strip().lower()
        return w / 2.20462 if unit in ("lb", "lbs") else w
    except (TypeError, ValueError):
        return 0.0


def _goal_aware_trim_ceiling_pct(form_data: dict, target_protein_day: float) -> float:
    """[P3-PROTEIN-CEILING-GOAL-AWARE] `ceiling_pct` para el trim = techo_g/kg × peso / target.
    Robusto a peso ausente: fallback por objetivo (déficit más laxo, volumen estricto).
    [P3-CONDITION-RULES] ERC: el techo g/kg es 0.8 (KDIGO 2024), NO el alto goal-aware — invierte
    la regla para que el trim baje la proteína a nivel renal en vez de protegerla alta."""
    renal = CONDITION_RULES_ENABLED and isinstance(form_data, dict) and _is_renal_condition(form_data)
    goal = form_data.get("mainGoal") if isinstance(form_data, dict) else None
    gkg = RENAL_PROTEIN_GKG_CEILING if renal else _protein_gkg_ceiling(goal)
    wkg = _weight_kg_from_form(form_data) if isinstance(form_data, dict) else 0.0
    if wkg > 0 and target_protein_day and target_protein_day > 0:
        return max(1.0, min(1.30, (gkg * wkg) / target_protein_day))
    if renal:
        return 1.0  # ERC sin peso parseable → techo estricto al target (ya capeado aguas arriba)
    return 1.18 if gkg >= 2.5 else 1.08  # sin peso → fallback por objetivo

# [P2-ANTI-REPETITION-TOLERANCE · 2026-06-13] Tolerancia de platos repetidos vs los
# últimos 3 planes ANTES de rechazar+reintentar. Pre-fix era tolerancia CERO: 1 solo
# plato repetido (de ~12) forzaba 3 reintentos → entrega degradada + alerta, aunque el
# plan fuera médicamente sano y con buenos macros. Algo de solape entre planes es
# natural (la gente tiene comidas favoritas). Default 2: hasta 2 repetidos = OK; 3+ =
# rechazo legítimo por falta de variedad. Subir a 0 revierte al comportamiento estricto.
ANTI_REPETITION_TOLERANCE = _env_int("MEALFIT_ANTI_REPETITION_TOLERANCE", 2)


def _meal_macro_num(x) -> float:
    """Parsea un valor de macro/caloría a float, tolerando "154g", "464 kcal", None.
    [P2-PERSIST-NAN-GUARD · 2026-06-13] Coacciona NaN/Inf → 0.0 (sin import math:
    `v == v` detecta NaN, comparación detecta Inf) para que un no-finito no se propague
    al solver/balancing ni al INSERT — Postgres jsonb rechaza NaN/Infinity y perdería el plan."""
    try:
        s = str(x).strip().lower().replace("g", "").replace("kcal", "").strip()
        v = float(s) if s else 0.0
        return v if (v == v and v not in (float("inf"), float("-inf"))) else 0.0
    except Exception:
        return 0.0


_MEDICAL_NONE_SENTINELS = {
    "", "ninguna", "ninguno", "ningunas", "ningunos", "none", "n/a", "na",
    "no", "nada", "sin alergias", "sin condiciones", "ninguna alergia",
}


def _has_real_medical_flags(items) -> bool:
    """[P2-MEDICAL-FACTCHECK-GATE · 2026-06-13] True si `items` (lista o string del
    form) contiene alguna alergia/condición REAL, no el sentinel "Ninguna".

    El frontend manda `allergies=["Ninguna"]` cuando el usuario no tiene alergias →
    lista no-vacía → truthy → el gate `if allergies or medical_conditions` corría el
    fact-checking médico por-ingrediente (~40s de tool calls) inútilmente para usuarios
    sanos. Este helper filtra los sentinels. NO reduce rigor para quien SÍ tiene
    restricciones reales — solo evita investigar interacciones que no aplican. Anchor:
    P2-MEDICAL-FACTCHECK-GATE."""
    if not items:
        return False
    if isinstance(items, str):
        items = [items]
    try:
        return any(str(x).strip().lower() not in _MEDICAL_NONE_SENTINELS for x in items)
    except Exception:
        return bool(items)


# [P3-CONDITION-RULES · 2026-06-14] Detección determinista de las condiciones del set Pareto DR.
# Términos SIN acentos (se normalizan con strip_accents) y sobre-inclusivos: en seguridad clínica,
# un falso positivo (aplicar un cap conservador a quien no lo necesita) es preferible a un falso
# negativo (no detectar la condición). Anclaje regulatorio: DM2 + ERC = las 2 condiciones que
# Medicare MNT reembolsable cubre.
# Términos SSOT en constants.py — compartidos con prompts/plan_generator para que el detector del
# cap determinista y el de la directiva de prompt NUNCA driften (cierra P3-CONDITION-RULES review #3).
from constants import (RENAL_CONDITION_TERMS as _RENAL_CONDITION_TERMS,
                       DIABETES_CONDITION_TERMS as _DIABETES_CONDITION_TERMS)


def _condition_strings(form_data) -> list:
    """Lista normalizada (lower + sin acentos, sin sentinel 'Ninguna') de las condiciones
    médicas declaradas en el form. Base de toda regla determinista por condición."""
    if not isinstance(form_data, dict):
        return []
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731 — degradación: sin normalizar acentos (los términos cubren variantes)
    raw = form_data.get("medicalConditions") or form_data.get("medical_conditions") or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for c in raw:
        s = str(c).strip().lower()
        if not s or s in _MEDICAL_NONE_SENTINELS:
            continue
        try:
            s = _sa(s)
        except Exception:
            pass
        out.append(s)
    return out


def _is_renal_condition(form_data) -> bool:
    """[P3-CONDITION-RULES] True si el perfil declara ERC/enfermedad renal (gate KDIGO)."""
    return any(any(t in c for t in _RENAL_CONDITION_TERMS) for c in _condition_strings(form_data))


def _is_diabetes_condition(form_data) -> bool:
    """[P3-CONDITION-RULES] True si el perfil declara diabetes/prediabetes (regla ADA 2026)."""
    return any(any(t in c for t in _DIABETES_CONDITION_TERMS) for c in _condition_strings(form_data))


def _cap_macros_dict_renal(md, cap_g: float, reassign_to: str):
    """[P3-CONDITION-RULES] Capea la proteína de un dict de macros (`{protein_g/_str, ...}`) al
    cap renal in place y reasigna las kcal liberadas a grasa (diabético) o carbo (iso-calórico).
    Retorna la proteína original (g) si capeó, None si ya estaba ≤ cap. Mantiene `_g` y `_str`."""
    if not isinstance(md, dict):
        return None
    p = _meal_macro_num(md.get("protein_g") if md.get("protein_g") is not None else md.get("protein_str"))
    if p <= cap_g:
        return None
    freed = (p - cap_g) * 4.0  # proteína = 4 kcal/g
    icap = int(round(cap_g))
    md["protein_g"] = icap
    md["protein_str"] = f"{icap}g"
    if reassign_to == "fat":
        f = _meal_macro_num(md.get("fats_g") if md.get("fats_g") is not None else md.get("fats_str"))
        nf = int(round(f + freed / 9.0))
        md["fats_g"] = nf; md["fats_str"] = f"{nf}g"
    else:
        c = _meal_macro_num(md.get("carbs_g") if md.get("carbs_g") is not None else md.get("carbs_str"))
        nc = int(round(c + freed / 4.0))
        md["carbs_g"] = nc; md["carbs_str"] = f"{nc}g"
    return int(round(p))


def _apply_renal_cap_to_nutrition(nutrition: dict, form_data: dict) -> None:
    """[P3-CONDITION-RULES · 2026-06-14] CAP RENAL EN LA FUENTE. Capea el target de proteína de
    `nutrition` (KDIGO ~0.8 g/kg) ANTES de que fluya al skeleton/generación/review/fallback. Es el
    fix raíz del modo de fallo hallado en la prueba en vivo: con el target a 2.2 g/kg, el LLM
    generaba un plan alto en proteína, el revisor médico renal-aware lo RECHAZABA críticamente
    ('carga proteica excesiva para ERC') y caía a un fallback matemático SIN capear. Capeando el
    target aquí, el LLM genera un plan renal-apropiado que el revisor aprueba, y el fallback (que
    se construye desde `nutrition.macros.protein_g`) también queda capeado. Mutates `nutrition`.
    Diabético-nefropatía: reasigna a grasa (no carbo). Fail-safe — no aplica si no hay peso."""
    if not (CONDITION_RULES_ENABLED and isinstance(nutrition, dict) and _is_renal_condition(form_data)):
        return
    try:
        wkg = _weight_kg_from_form(form_data)
        if wkg <= 0:
            return
        cap_g = round(RENAL_PROTEIN_GKG_CEILING * wkg)
        if cap_g <= 0:
            return
        diabetic = _is_diabetes_condition(form_data)
        reassign = "fat" if diabetic else "carb"
        was = None
        for key in ("macros", "total_daily_macros"):
            w = _cap_macros_dict_renal(nutrition.get(key), cap_g, reassign)
            if w:
                was = w
        if was:
            nutrition["renal_protein_cap"] = {
                "applied": True, "gkg": RENAL_PROTEIN_GKG_CEILING, "protein_g": cap_g,
                "was_g": was, "weight_kg": round(wkg, 1),
                "guideline": "KDIGO 2024 (G3-G5 no-diálisis)", "reassigned_to": reassign,
                "comorbid_diabetes": bool(diabetic), "source": "nutrition_target",
            }
            logger.warning(
                f"🫘 [P3-CONDITION-RULES] ERC: target de proteína capeado en la FUENTE {was}g→{cap_g}g "
                f"(~{RENAL_PROTEIN_GKG_CEILING} g/kg × {wkg:.1f}kg), kcal a {reassign}"
                f"{' (DM2)' if diabetic else ''}. Fluye a generación/review/fallback.")
    except Exception as _rcn_e:
        logger.warning(f"[P3-CONDITION-RULES] cap renal en la fuente falló: "
                       f"{type(_rcn_e).__name__}: {_rcn_e}")


# [P3-CONDITION-ENGINE] Captura el prefijo de cantidad de un ingrediente ("100g de ", "2 lonjas de ",
# "0.5 taza de ") para preservarlo al sustituir → la lista de compras conserva el peso y el coherence
# guard sigue cuadrando. Sin prefijo (condimentos "al gusto") el sustituto queda sin cantidad (OK).
_COND_SUB_QTY_PREFIX_RE = _re.compile(
    r"^\s*(\d[\d.,/]*\s*[a-záéíóúñ]*\.?\s*(?:de\s+)?)", _re.IGNORECASE)


def _apply_substitutions_core(plan: dict, subs: list, note_builder, note_sentinel: str,
                              on_meal_fixed=None) -> int:
    """[P0-ALLERGEN-SUBS · 2026-06-14] Motor de swap determinista COMPARTIDO por las sustituciones
    por condición (DM2/HTA/dislipidemia) y por alérgeno IgE. `subs`: lista de dicts
    {tokens, replacement, label, negatives, condition, preserve_qty} (de `collect_substitutions` o
    `collect_allergen_substitutions`). `note_builder(uniq_labels, conds) -> str`: la nota es-DO
    inyectada a la receta; `note_sentinel`: substring para la guarda de idempotencia de la nota.
    `on_meal_fixed(meal, uniq_labels, conds)`: callback opcional para flags por-llamador.

    PRESERVA el prefijo de cantidad (staples) → la lista de compras conserva el peso comprable y el
    coherence guard sigue cuadrando. RECOMPUTA los macros del plato por DELTA quirúrgico (fail-safe:
    si la DB no resuelve AMBOS strings, deja los macros previos). Idempotente. Mutates `plan`.
    Retorna #comidas afectadas. Extraído verbatim de `_apply_condition_substitutions` (P3-CONDITION-
    ENGINE) para reusar el motor sin drift entre los dos llamadores."""
    if not subs:
        return 0
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    try:
        from nutrition_db import IngredientNutritionDB
        _db = IngredientNutritionDB()
    except Exception:
        _db = None

    def _match(ing_norm):
        for s in subs:
            if any(neg in ing_norm for neg in s["negatives"]):
                continue
            if any(t in ing_norm for t in s["tokens"]):
                return s
        return None

    def _sub_string(orig, replacement, preserve_qty):
        # Staples (preserve_qty): preserva el prefijo de cantidad ("100g de", "2 lonjas de") + el
        # reemplazo → la lista de compras conserva el peso comprable. Condimentos/azúcares: deja el
        # reemplazo bare (queda "al gusto"); preservar el prefijo dejaría la palabra ofensora ("cubito").
        if preserve_qty:
            m = _COND_SUB_QTY_PREFIX_RE.match(str(orig)) if _COND_SUB_QTY_PREFIX_RE else None
            if m and m.group(1).strip():
                return m.group(1) + replacement
        return replacement

    fixed = 0
    for day in plan.get("days", []) or []:
        for meal in day.get("meals", []) or []:
            labels, conds, changed = [], set(), False
            swaps = []  # (string_viejo, string_nuevo) — solo de la lista canónica, para el delta de macros
            for key in ("ingredients", "ingredients_raw"):
                ings = meal.get(key)
                if not isinstance(ings, list):
                    continue
                out = []
                for ing in ings:
                    s = _match(_sa(str(ing).lower()))
                    if s:
                        new = _sub_string(ing, s["replacement"], s.get("preserve_qty"))
                        if new not in out:  # evita duplicar el sustituto
                            out.append(new)
                        if key == "ingredients":
                            swaps.append((str(ing), new))
                        labels.append(s["label"]); conds.add(s["condition"]); changed = True
                    else:
                        out.append(ing)
                meal[key] = out
            if changed:
                # Ajusta los macros del plato por DELTA quirúrgico: macros(nuevo) - macros(viejo) SOLO de
                # los ingredientes sustituidos → el header deja de describir el ingrediente viejo sin
                # tocar (ni perder por 0-silencioso) la contribución del resto del plato. Fail-safe: si
                # un string no resuelve, ese sumando es 0 (conservador). Mismo patrón que el quantize.
                if _db is not None and swaps:
                    try:
                        _d = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
                        _touched = False
                        for _old, _new in swaps:
                            _om = _db.macros_from_ingredient_string(_old) or {}
                            _nm = _db.macros_from_ingredient_string(_new) or {}
                            # [review adversaria P4] AMBOS deben resolver para aplicar el delta. Si solo
                            # uno resuelve, el delta sería asimétrico: reemplazo no-resuelve → resta el
                            # viejo y suma 0 (pierde proteína, re-introduce el "0 silencioso"); viejo
                            # no-resuelve → suma el nuevo sin restar (infla). En ambos casos dejamos los
                            # macros intactos (conservador) — la sustitución textual + cantidad ya pasó.
                            if _om and _nm:
                                _touched = True
                                for _k in _d:
                                    _d[_k] += (_nm.get(_k) or 0.0) - (_om.get(_k) or 0.0)
                        if _touched and any(_d.values()):
                            meal["protein"] = max(0, round(_meal_macro_num(meal.get("protein")) + _d["protein"]))
                            meal["carbs"] = max(0, round(_meal_macro_num(meal.get("carbs")) + _d["carbs"]))
                            meal["fats"] = max(0, round(_meal_macro_num(meal.get("fats")) + _d["fats"]))
                            meal["cals"] = max(0, round(_meal_macro_num(meal.get("cals")) + _d["kcal"]))
                            meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
                    except Exception:
                        pass
                uniq = sorted(set(labels))
                note = note_builder(uniq, conds)
                rec = meal.get("recipe")
                if not isinstance(rec, list):
                    rec = [] if rec is None else [str(rec)]
                if note and not any(note_sentinel in str(s) for s in rec):
                    meal["recipe"] = rec + [note]
                if on_meal_fixed:
                    try:
                        on_meal_fixed(meal, uniq, conds)
                    except Exception:
                        pass
                fixed += 1
    return fixed


def _apply_condition_substitutions(plan: dict, form_data: dict) -> int:
    """[P3-CONDITION-ENGINE · 2026-06-14] Guard determinista de SUSTITUCIÓN de ingredientes por
    condición (patrón food-safety GENERALIZADO, dirigido por el registro `condition_rules`). Para
    CADA condición activa sustituye los ingredientes contraindicados por una alternativa segura en
    `ingredients`+`ingredients_raw` ANTES del review+shopping: DM2 → azúcar añadida→stevia/agua; HTA
    → embutidos/cubitos/bacalao salado→fresco. El plan queda clínicamente mejor Y deja de gatillar
    el rechazo del revisor. Reusa `_apply_substitutions_core` (motor compartido con los alérgenos).

    PRESERVA el prefijo de cantidad del ingrediente original ("100g de longaniza" → "100g de Pechuga
    de pollo") → la lista conserva el peso comprable. RECOMPUTA macros por delta (fail-safe).
    Idempotente. Mutates `plan`. Retorna #comidas."""
    if not CONDITION_RULES_ENABLED:
        return 0
    try:
        from condition_rules import collect_substitutions
        subs = collect_substitutions(form_data)
    except Exception:
        subs = []

    def _note(uniq, conds):
        return (f"⚕️ Ajuste clínico (condición médica): se sustituyó {', '.join(uniq)} por una "
                f"alternativa segura (sin azúcar añadida / baja en sodio) para tu condición.")

    def _flags(meal, uniq, conds):
        meal["_condition_subs_fixed"] = uniq
        if "dm2" in conds:
            meal["_dm2_sugar_fixed"] = uniq  # flag de compatibilidad hacia atrás

    return _apply_substitutions_core(plan, subs, _note, "Ajuste clínico", _flags)


def _apply_allergen_substitutions(plan: dict, form_data: dict) -> int:
    """[P0-ALLERGEN-SUBS · 2026-06-14] Sustitución QUIRÚRGICA de alérgenos IgE DECLARADOS
    (`form_data['allergies']`) por una alternativa segura que RESUELVE al catálogo, ANTES del review.
    Cierra el gap del audit clínico: las alergias dependían del prompt + el backstop romo
    `_scan_allergen_violations` (que nukea el plan entero a fallback). Aquí el swap es in-place
    (preserva cantidad + macros por delta, vía el motor compartido) y conserva el plan rico del LLM;
    `_scan_allergen_violations` (review) sigue escalando cualquier RESIDUAL a crítico → cero
    regresión de seguridad. Solo fish/shellfish/soy/gluten (lácteos/huevo/maní/frutos secos NO tienen
    target libre del alérgeno que resuelva en el catálogo es-DO → siguen por el path crítico→fallback).
    Mutates `plan`. Retorna #comidas afectadas."""
    if not ALLERGEN_SUBSTITUTION_ENABLED:
        return 0
    try:
        from condition_rules import collect_allergen_substitutions
        subs = collect_allergen_substitutions(form_data)
    except Exception:
        subs = []

    def _note(uniq, conds):
        return (f"🛡️ Sustitución por alergia declarada: se reemplazó {', '.join(uniq)} por una "
                f"alternativa segura libre del alérgeno. Verifica igualmente las etiquetas de lo que compras.")

    def _flags(meal, uniq, conds):
        meal["_allergen_subs_fixed"] = uniq

    return _apply_substitutions_core(plan, subs, _note, "Sustitución por alergia", _flags)


# [C2-ALLERGEN-GUARD · 2026-06-13] Mapa de sinónimos de alérgenos comunes (es-DO) → términos
# que pueden aparecer en los ingredientes. Sesgo a SOBRE-detectar: para alérgenos, un falso
# positivo (rechazar un plan seguro) es MUCHO mejor que un falso negativo (servir el alérgeno).
# [P1-ALLERGEN-DERIVATIVES · 2026-06-14] Extensión con DERIVADOS / ingredientes compuestos por
# categoría (whey/caseína/helado para lácteos; mayonesa/merengue/holandesa para huevo;
# miso/tempeh/teriyaki/lecitina para soya; cuscús/seitán/malta/sémola para gluten; anchoa/surimi/
# salsa inglesa para pescado; mazapán/nutella/praliné para frutos secos). Cierra el punto ciego
# del audit: el alérgeno escondido en un compuesto (whey en un batido, mayonesa en un aderezo)
# antes solo dependía del revisor LLM (falible). Es DETECCIÓN (backstop _scan_allergen_violations
# + filtro de candidatos del closer), no sustitución → no requiere filas de catálogo nuevas y
# preserva el sesgo de sobre-detección intencional. Anchor: P1-ALLERGEN-DERIVATIVES.
_ALLERGEN_SYNONYMS = {
    "mani": ["mani", "cacahuate", "peanut", "mantequilla de mani", "crema de mani",
             "salsa de mani"],
    "frutos secos": ["almendra", "almendras", "nuez", "nueces", "maranon", "pistacho",
                     "avellana", "merey", "maranon", "anacardo", "marzipan", "mazapan",
                     "nutella", "praline", "turron", "pesto", "crema de avellana"],
    "mariscos": ["camaron", "camarones", "langosta", "cangrejo", "langostino", "gambas",
                 "marisco", "mariscos", "pulpo", "calamar", "almeja", "ostra", "lambi",
                 "surimi"],
    "pescado": ["pescado", "bacalao", "atun", "salmon", "tilapia", "mero", "chillo",
                "dorado", "sardina", "merluza", "carite", "anchoa", "anchoas",
                "salsa de pescado", "surimi", "caviar", "salsa inglesa", "worcestershire"],
    "lacteos": ["leche", "queso", "yogurt", "mantequilla", "crema", "lacteo", "ricotta",
                "mozzarella", "parmesano", "cottage", "whey", "suero de leche", "caseina",
                "caseinato", "proteina de suero", "proteina de leche", "helado", "mantecado",
                "dulce de leche", "queso crema", "requeson", "kefir", "natilla", "flan",
                "leche condensada", "leche evaporada", "nata", "ghee"],
    "lactosa": ["leche", "queso", "yogurt", "mantequilla", "crema", "ricotta", "mozzarella",
                "whey", "suero de leche", "helado", "mantecado", "dulce de leche", "queso crema",
                "requeson", "kefir", "natilla", "flan", "leche condensada", "leche evaporada"],
    "gluten": ["trigo", "pan", "pasta", "harina de trigo", "galleta", "galletas", "cebada",
               "centeno", "gluten", "tortilla integral", "pan integral", "cuscus", "couscous",
               "seitan", "bulgur", "malta", "cerveza", "semola", "espagueti", "macarrones",
               "lasana", "lasagna", "empanada", "bizcocho", "wheat"],
    "huevo": ["huevo", "huevos", "clara", "claras", "yema", "yemas", "mayonesa", "merengue",
              "aioli", "alioli", "holandesa", "ponche", "mousse"],
    "huevos": ["huevo", "huevos", "clara", "claras", "yema", "yemas", "mayonesa", "merengue",
               "aioli", "alioli", "holandesa", "ponche", "mousse"],
    "soya": ["soya", "soja", "tofu", "salsa de soya", "edamame", "miso", "tempeh",
             "salsa teriyaki", "teriyaki", "natto", "lecitina de soya", "proteina de soya",
             "proteina vegetal texturizada", "tvp"],
}


def _scan_allergen_violations(plan: dict, allergies) -> list:
    """[C2-ALLERGEN-GUARD · 2026-06-13] Backstop DETERMINISTA de seguridad de alérgenos
    (encima del revisor LLM, que puede fallar). Escanea cada ingrediente del plan contra
    las alergias declaradas + sinónimos comunes DD; retorna lista de violaciones
    (meal_name, ingrediente, término_alérgeno). Para alimentos sin sinónimo conocido,
    matchea el texto literal de la alergia. Sesgo a sobre-detectar (seguridad > comodidad).
    Anchor: C2-ALLERGEN-GUARD."""
    try:
        from constants import strip_accents
    except Exception:
        def strip_accents(s):
            import unicodedata
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    import re as _re
    if isinstance(allergies, str):
        allergies = [allergies]
    forbidden = set()
    for a in (allergies or []):
        a_low = strip_accents(str(a).strip().lower())
        if not a_low or a_low in _MEDICAL_NONE_SENTINELS:
            continue
        matched = False
        for cat, syns in _ALLERGEN_SYNONYMS.items():
            cat_n = strip_accents(cat)
            if a_low == cat_n or cat_n in a_low or a_low in cat_n or \
               any(a_low in strip_accents(s) or strip_accents(s) in a_low for s in syns):
                forbidden.update(strip_accents(s) for s in syns)
                matched = True
        if not matched:
            forbidden.add(a_low)  # alergia free-text → match literal
    if not forbidden:
        return []
    violations = []
    for day in plan.get("days", []):
        for meal in day.get("meals", []):
            for ing in meal.get("ingredients", []):
                ing_low = strip_accents(str(ing).lower())
                for f in forbidden:
                    # `(?:s|es)?` captura el plural español (fresa→fresas, pan→panes,
                    # camaron→camarones) SIN falsos positivos de prefijo (leche≠lechosa).
                    if f and _re.search(r"\b" + _re.escape(f) + r"(?:s|es)?\b", ing_low):
                        violations.append((meal.get("name", "?"), str(ing), f))
                        break
    return violations


# [P3-FOOD-SAFETY · 2026-06-13] Seguridad alimentaria determinista: el huevo es un alimento
# TCS (Time/Temperature Control for Safety); crudo/poco cocido = vector directo de Salmonella
# enteritidis (CDC/FDA Food Code lo desaconsejan sin pasteurizar). Hallazgo CRÍTICO de la
# auditoría clínica multi-agente (plan 11d17452: ½ huevo crudo LICUADO en un batido +
# 1¼ huevos sin paso de cocción en un wrap). Simétrico al allergen guard: el revisor LLM
# puede dejar pasar huevo crudo, y el solver/top-up podría AÑADIRLO a una preparación fría;
# este guard determinista nunca sirve huevo crudo sin una mitigación explícita.
_RAW_EGG_TERMS = ("huevo", "huevos", "clara", "claras", "yema", "yemas")
# Preparaciones LICUADAS/FRÍAS donde el huevo queda inequívocamente CRUDO (no hay cocción
# posible aguas abajo): un batido no se cocina. Aquí el riesgo es máximo y el fix es fuerte.
_NO_COOK_BLENDED = ("batido", "smoothie", "licuado", "licuada", "malteada", "jugo")
# Indicadores de que el huevo SÍ se cuece (≥71°C) → seguro. Sesgo a SEGURIDAD: un falso
# "crudo" solo añade una nota inocua; un falso "cocido" sería peligroso (omitiría la nota),
# así que se exigen términos ESPECÍFICOS de huevo cocido, NO genéricos ambiguos. Nota:
# "tortilla" a secas es el pan plano (wrap) en es-DO, NO la tortilla de huevo → se exige
# "tortilla de huevo" explícito para no dar por cocido un huevo crudo dentro de un wrap.
_EGG_COOK_INDICATORS = (
    "tortilla de huevo", "tortilla espanola", "revoltillo", "revuelto", "omelet", "omelette",
    "huevo frito", "huevos fritos", "huevo cocid", "huevos cocid", "huevo hervi", "huevos hervi",
    "huevo duro", "huevos duros", "escalfad", "poche", "huevo estrellad", "huevos estrellad",
    "pasado por agua", "cocina el huevo", "cocina los huevo", "cuece el huevo", "cuece los huevo",
    "frie el huevo", "frie los huevo", "fríe el huevo", "fríe los huevo",
)


def _meal_has_egg(meal: dict, strip_accents) -> bool:
    """True si algún ingrediente del meal contiene huevo/clara/yema (token aislado)."""
    import re as _re
    for ing in meal.get("ingredients", []) or []:
        ing_low = strip_accents(str(ing).lower())
        for t in _RAW_EGG_TERMS:
            if _re.search(r"\b" + t + r"\b", ing_low):
                return True
    return False


def _meal_egg_is_cooked(meal: dict, strip_accents) -> bool:
    """True si el nombre o la receta del meal indica que el huevo se cocina (≥71°C)."""
    parts = [str(meal.get("name", ""))]
    rec = meal.get("recipe")
    if isinstance(rec, list):
        parts.extend(str(s) for s in rec)
    text = strip_accents(" ".join(parts).lower())
    return any(c in text for c in _EGG_COOK_INDICATORS)


def _scan_raw_egg_violations(plan: dict) -> list:
    """[P3-FOOD-SAFETY · 2026-06-13] Detector determinista de huevo crudo/poco cocido.
    Retorna lista de (day_idx, meal_idx, meal_name, kind) con kind ∈ {'blended','no_cook'}:
    - 'blended': huevo en una preparación licuada (batido/jugo) → CRUDO inequívoco.
    - 'no_cook': huevo presente y ningún indicador de cocción en nombre/receta → riesgo.
    Sesgo a sobre-detectar (seguridad > comodidad); el auto-fix es macro-preservante y no
    rompe la coherencia receta↔lista (no muta el token canónico del ingrediente)."""
    try:
        from constants import strip_accents
    except Exception:
        def strip_accents(s):
            import unicodedata
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    violations = []
    for di, day in enumerate(plan.get("days", []) or []):
        for mi, meal in enumerate(day.get("meals", []) or []):
            if not _meal_has_egg(meal, strip_accents):
                continue
            name_low = strip_accents(str(meal.get("name", "")).lower())
            if any(b in name_low for b in _NO_COOK_BLENDED):
                violations.append((di, mi, meal.get("name", "?"), "blended"))
            elif not _meal_egg_is_cooked(meal, strip_accents):
                violations.append((di, mi, meal.get("name", "?"), "no_cook"))
    return violations


# Notas de seguridad (prominentes, accionables) que el auto-fix inyecta a la receta.
_FOOD_SAFETY_NOTE_BLENDED = (
    "⚠️ Seguridad alimentaria: NO uses huevo crudo en un batido (riesgo de Salmonella). "
    "Usa huevo PASTEURIZADO, o sustitúyelo por 1 medida de proteína en polvo o 2-3 "
    "cucharadas de yogur griego para el mismo aporte proteico sin riesgo."
)
_FOOD_SAFETY_NOTE_NOCOOK = (
    "⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, "
    "sin partes líquidas) antes de servir; evita el huevo crudo o poco cocido."
)


def _apply_food_safety_fixes(plan: dict) -> int:
    """[P3-FOOD-SAFETY · 2026-06-13] Aplica mitigación determinista a las violaciones de
    huevo crudo detectadas por `_scan_raw_egg_violations`. Macro-PRESERVANTE (no toca
    cantidades ni macros) y shopping-SAFE (no muta el token canónico del ingrediente, solo
    añade una nota a la receta) → corre con seguridad antes de la agregación de compras sin
    introducir divergencias receta↔lista. Idempotente: no duplica una nota ya presente.
    Retorna el número de meals mitigados. Anchor: P3-FOOD-SAFETY."""
    fixed = 0
    for di, mi, _name, kind in _scan_raw_egg_violations(plan):
        try:
            meal = plan["days"][di]["meals"][mi]
        except (KeyError, IndexError, TypeError):
            continue
        note = _FOOD_SAFETY_NOTE_BLENDED if kind == "blended" else _FOOD_SAFETY_NOTE_NOCOOK
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            rec = [] if rec is None else [str(rec)]
        if any("Seguridad alimentaria" in str(s) for s in rec):
            continue  # ya mitigado → idempotente
        meal["recipe"] = rec + [note]
        meal["_food_safety_fixed"] = kind
        fixed += 1
    return fixed


def _recover_meal_macros(meal: dict, ratio_p: float, ratio_c: float, ratio_f: float) -> None:
    """[P0-MEAL-MACRO-RECOVERY · 2026-06-13] Garantiza que un meal tenga breakdown
    de macros. Si protein/carbs/fats vienen todos en 0/ausentes (gap del day-gen o
    self-critique fallido que dejó el día sin macros → shippeaba protein=0 +
    placeholder "Plan Matemático" y el usuario veía 0g de proteína), los ESTIMA
    desde las cals del meal con el split objetivo del plan (ratio_p/c/f = fracción
    de cals por macro). Determinístico y muy superior a 0. Si además faltan las
    cals, deja el placeholder. Si los numéricos existen pero falta la lista de
    display `macros`, la reconstruye. Mutates `meal` in-place.

    Anchor: P0-MEAL-MACRO-RECOVERY.
    """
    has_real = not all(_meal_macro_num(meal.get(k)) == 0 for k in ("protein", "carbs", "fats"))
    if not has_real:
        mcals = _meal_macro_num(meal.get("cals"))
        if mcals > 0:
            pm = round(mcals * ratio_p / 4)
            cm = round(mcals * ratio_c / 4)
            fm = round(mcals * ratio_f / 9)
            meal["protein"], meal["carbs"], meal["fats"] = pm, cm, fm
            meal["macros"] = [f"P:{pm}g", f"C:{cm}g", f"G:{fm}g"]
        elif not meal.get("macros"):
            meal["macros"] = ["Plan Matemático"]
    elif not meal.get("macros"):
        meal["macros"] = [
            f"P:{round(_meal_macro_num(meal.get('protein')))}g",
            f"C:{round(_meal_macro_num(meal.get('carbs')))}g",
            f"G:{round(_meal_macro_num(meal.get('fats')))}g",
        ]


def _apply_macro_solver_to_meal(meal: dict, slot_target: dict, db) -> bool:
    """[P3-MACRO-SOLVER · 2026-06-13] Cerebro dividido — lado determinista.
    Re-escala las porciones de los ingredientes del meal para clavar `slot_target`
    {kcal,protein,carbs,fats} usando macros reales de `master_ingredients` (vía
    `portion_solver.solve_meal_macros`), en vez de confiar en el porcionado a-ojo
    del LLM. Re-escribe `ingredients` + `ingredients_raw` (cantidad líder + hint de
    gramos, formato preservado → coherence guard/shopping/frontend lo parsean igual)
    y actualiza los campos numéricos protein/carbs/fats/cals + display `macros`.

    Fail-safe TOTAL: cualquier excepción o 0 ingredientes resueltos → deja el meal
    INTACTO y retorna False (nunca rompe el assembly). Gated por knob upstream.
    Mutates `meal` in-place. Retorna True si aplicó cambios. Anchor: P3-MACRO-SOLVER.
    """
    try:
        ings = meal.get("ingredients")
        if not isinstance(ings, list) or not ings:
            return False
        from portion_solver import solve_meal_macros
        from nutrition_db import rescale_ingredient_string
        res = solve_meal_macros([str(x) for x in ings], slot_target, db=db)
        if res.get("resolved_count", 0) == 0:
            return False  # nada resoluble → no tocar (degradación grácil)
        meal["ingredients"] = res["ingredients"]
        factors = res.get("factors_applied") or []
        raw = meal.get("ingredients_raw")
        if isinstance(raw, list) and len(raw) == len(factors):
            meal["ingredients_raw"] = [
                rescale_ingredient_string(str(r), f) for r, f in zip(raw, factors)
            ]
        ach = res["achieved"]
        meal["protein"] = round(ach["protein"])
        meal["carbs"] = round(ach["carbs"])
        meal["fats"] = round(ach["fats"])
        meal["cals"] = round(ach["kcal"])
        meal["macros"] = [f"P:{round(ach['protein'])}g",
                          f"C:{round(ach['carbs'])}g",
                          f"G:{round(ach['fats'])}g"]
        return True
    except Exception as e:
        logger.warning(f"[P3-MACRO-SOLVER] solver falló para meal "
                       f"{str(meal.get('name'))[:40]!r}: {type(e).__name__}: {e} — meal intacto")
        return False


def _protein_topup_meal(meal: dict, slot_cal_target: float, db, approved_proteins,
                        *, floor_g: float = 12.0, fill_to_g: float = 18.0,
                        max_add_g: int = 120) -> int:
    """[P3-PROTEIN-TOPUP · 2026-06-13] RESCATE de comidas egregiamente pobres en proteína.
    Si tras el solver una comida queda por DEBAJO de un piso absoluto (`floor_g`, 12g),
    añade una porción MODESTA de la proteína más MAGRA del pool APROBADO del día para
    llevarla a `fill_to_g` (~18g) — NO al target completo del slot (eso sobre-disparaba:
    el target diario es muy alto, ~2.8g/kg, casi toda comida quedaba "corta" → top-up
    masivo → calorías explotaban). El objetivo es eliminar las meriendas/desayunos de 0-6g
    que el escalado no puede arreglar (no hay proteína que escalar), sin reventar las kcal.

    Calorie-aware: nunca añade tanto que el meal supere su `slot_cal_target`. Fail-secure:
    pool vacío o sin proteína resoluble (≥5g/100g) → no añade nada (jamás mete un alimento
    fuera del pool aprobado → cero riesgo de alérgeno). La más MAGRA (mayor proteína/kcal)
    minimiza el costo calórico. Mutates `meal`. Retorna gramos añadidos. Anchor: P3-PROTEIN-TOPUP."""
    try:
        if meal.get("_protein_closed"):
            return 0  # [P3-PROTEIN-IDEMPOTENT] ya se cerró en una pasada previa — no re-añadir
        cur_p = _meal_macro_num(meal.get("protein"))
        if cur_p >= floor_g:
            return 0  # ya tiene proteína suficiente — no tocar
        # [P3-FOOD-SAFETY] No añadir huevo crudo a preparaciones licuadas/frías (batido,
        # jugo): no hay cocción posible aguas abajo → sería un vector de Salmonella. En esos
        # slots se prefiere una proteína no-cocción-dependiente (yogur/proteína en polvo).
        try:
            from constants import strip_accents as _sa
        except Exception:
            def _sa(s):
                import unicodedata
                return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
        _name_low = _sa(str(meal.get("name", "")).lower())
        _no_cook = any(b in _name_low for b in _NO_COOK_BLENDED)
        best = None  # (leanness, info)
        for p in (approved_proteins or []):
            info = db.lookup(p)
            if info and info.protein >= 5:
                if _no_cook and any(t in _sa(str(info.name).lower()) for t in _RAW_EGG_TERMS):
                    continue  # huevo crudo en batido prohibido → probar otra proteína del pool
                leanness = info.protein / max(info.kcal, 1.0)  # g proteína por kcal
                if best is None or leanness > best[0]:
                    best = (leanness, info)
        if best is None:
            return 0  # nada con proteína real en el pool → fail-safe, no inventar
        info = best[1]
        gap = max(0.0, fill_to_g - cur_p)
        grams = int(round(gap / (info.protein / 100.0)))
        # Calorie-aware: no exceder el target calórico del slot.
        cur_cal = _meal_macro_num(meal.get("cals"))
        if slot_cal_target and slot_cal_target > cur_cal and info.kcal > 0:
            grams_cal_cap = int((slot_cal_target - cur_cal) / (info.kcal / 100.0))
            grams = min(grams, grams_cal_cap)
        grams = min(max_add_g, grams)
        if grams < 15:
            return 0
        f = grams / 100.0
        name_disp = str(info.name).lower()
        line = f"{grams}g de {name_disp} ({grams}g)"
        meal.setdefault("ingredients", []).append(line)
        if isinstance(meal.get("ingredients_raw"), list):
            meal["ingredients_raw"].append(line)
        meal["_protein_closed"] = True  # [P3-PROTEIN-IDEMPOTENT] no re-cerrar en re-assemble
        meal["protein"] = round(cur_p + info.protein * f)
        meal["carbs"] = round(_meal_macro_num(meal.get("carbs")) + info.carbs * f)
        meal["fats"] = round(_meal_macro_num(meal.get("fats")) + info.fats * f)
        meal["cals"] = round(_meal_macro_num(meal.get("cals")) + info.kcal * f)
        meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
        rec = meal.get("recipe")
        if isinstance(rec, list):
            meal["recipe"] = rec + [
                f"💪 Nota del Nutricionista AI: añade {grams}g de {name_disp} "
                f"para completar tu objetivo de proteína."]
        return grams
    except Exception as e:
        logger.warning(f"[P3-PROTEIN-TOPUP] falló para meal "
                       f"{str(meal.get('name'))[:40]!r}: {type(e).__name__}: {e}")
        return 0


# [P3-PROTEIN-FLOOR · 2026-06-13] Cierre del DÉFICIT de proteína (la re-auditoría del plan
# fresco encontró que se entregaba 68% del target — los fixes eran advisory, no restricción
# dura). El closer rellena cada comida a su target de proteína del slot con una proteína de
# ALTA DENSIDAD allergen-safe, integrada como ingrediente real (gramos), y el reconcile
# protein-preserving nivela las kcal escalando SOLO carbos/grasas (la proteína queda fija).
# Proteínas no-cocción-safe (para batidos/ensaladas frías): yogur/queso/whey, NUNCA huevo crudo.
_NO_COOK_SAFE_PROTEIN_HINT = ("yogur", "yogurt", "queso", "whey", "proteina", "proteína", "ricotta")
# [P3-PROTEIN-FLOOR] Dish-fit del closer: comidas ligeras (desayuno/merienda) prefieren
# proteína de huevo/lácteo; las principales prefieren carne — evita combos incongruentes
# (camarón en un revoltillo de desayuno). Congruencia (proteína ya en el plato) gana primero.
_LIGHT_MEAL_HINT = ("desayuno", "merienda", "avena", "batido", "smoothie", "licuado", "jugo",
                    "tostada", "yogur", "fruta", "panqueque", "omelet", "tortilla", "revoltillo",
                    "cereal", "granola", "bowl", "snack")
_DAIRY_EGG_PROTEIN_HINT = ("huevo", "clara", "yogur", "yogurt", "queso", "ricotta", "whey", "proteina", "proteína")
_MEAT_PROTEIN_HINT = ("pollo", "pavo", "cerdo", "res", "carne", "pescado", "atun", "atún",
                      "sardina", "camaron", "camarón", "tilapia", "lomo", "chuleta", "longaniza")


def _safe_high_density_proteins(allergies, db, min_protein: float = 18.0) -> list:
    """[P3-PROTEIN-FLOOR] Proteínas de ALTA densidad (≥min_protein g/100g) del catálogo
    dominicano que son allergen-SAFE para el usuario y resuelven en el catálogo nutricional.
    Ordenadas por magrez (proteína/kcal) desc → el closer prefiere la de menor costo calórico.
    Retorna [(leanness, name, info), …]. Cero alérgeno (reusa el mapa de sinónimos C2)."""
    try:
        from constants import DOMINICAN_PROTEINS, strip_accents
    except Exception:
        return []
    import re as _re
    forbidden = set()
    if _has_real_medical_flags(allergies):
        al = allergies if isinstance(allergies, list) else [allergies]
        for a in al:
            a_low = strip_accents(str(a).strip().lower())
            if not a_low or a_low in _MEDICAL_NONE_SENTINELS:
                continue
            matched = False
            for cat, syns in _ALLERGEN_SYNONYMS.items():
                cat_n = strip_accents(cat)
                if a_low == cat_n or cat_n in a_low or a_low in cat_n or \
                   any(a_low in strip_accents(s) or strip_accents(s) in a_low for s in syns):
                    forbidden.update(strip_accents(s) for s in syns)
                    matched = True
            if not matched:
                forbidden.add(a_low)
    out = []
    for name in DOMINICAN_PROTEINS:
        nlow = strip_accents(str(name).lower())
        if forbidden and any(f and (f in nlow or nlow in f) for f in forbidden):
            continue  # alérgeno → excluir (fail-secure)
        info = db.lookup(name)
        if info and info.protein >= min_protein and info.kcal > 0:
            out.append((info.protein / info.kcal, name, info))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _close_protein_gap_for_meal(meal: dict, slot_protein_target: float, db, candidates,
                                *, fill_pct: float = 0.92, max_add_g: int = 300) -> int:
    """[P3-PROTEIN-FLOOR · 2026-06-13] Rellena el meal hasta ~fill_pct del target de proteína
    del slot con una proteína de ALTA DENSIDAD allergen-safe (de `candidates`), integrada como
    INGREDIENTE real en gramos (no como nota). Cierra el déficit que el escalado no puede (no
    hay proteína que escalar en bases vegetales/almidón). Para preparaciones frías/licuadas
    solo usa proteínas no-cocción-safe (yogur/queso/whey), NUNCA huevo crudo (FS1). Las kcal
    extra las nivela aguas abajo el reconcile protein-preserving. Mutates meal. Retorna gramos."""
    try:
        if meal.get("_protein_closed"):
            return 0  # [P3-PROTEIN-IDEMPOTENT] ya se cerró en una pasada previa de assemble →
            # no re-añadir (re-assemble en retries/surgical regen acumulaba proteína + duplicaba
            # ingredientes, inflando el día por encima del techo g/kg). Fix del gap post-review.
        cur_p = _meal_macro_num(meal.get("protein"))
        target = slot_protein_target * fill_pct
        if target <= 0 or cur_p >= target:
            return 0
        try:
            from constants import strip_accents as _sa
        except Exception:
            def _sa(s):
                import unicodedata
                return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
        name_low = _sa(str(meal.get("name", "")).lower())
        no_cook = any(b in name_low for b in _NO_COOK_BLENDED)
        light = any(h in name_low for h in _LIGHT_MEAL_HINT)
        meal_text = _sa((str(meal.get("name", "")) + " "
                         + " ".join(str(i) for i in meal.get("ingredients", []) or [])).lower())
        # Pool de candidatos válidos para el CONTEXTO (no-cook → solo no-cocción-safe).
        _pool = []
        for _leanness, _name, info in (candidates or []):
            nlow = _sa(str(info.name).lower())
            if no_cook:
                if not any(h in nlow for h in _NO_COOK_SAFE_PROTEIN_HINT):
                    continue  # batido/frío → solo yogur/queso/whey
                if any(t in nlow for t in _RAW_EGG_TERMS):
                    continue
            _pool.append((info, nlow))
        if not _pool:
            return 0  # no-cook sin candidato seguro → no forzar carne cruda en un batido
        # Dish-fit: 1) congruencia (proteína ya mencionada en el plato) → escala el tema;
        # 2) categoría (ligera→huevo/lácteo, principal→carne); 3) fallback la más magra.
        chosen = None
        for info, nlow in _pool:
            # congruencia solo con proteína de alta densidad (≥18): no "escala" lentejas
            # (baja densidad → gramos absurdos); para esas cae a categoría. Matchea por TOKEN
            # (1ª palabra del nombre, ej. "queso" de "queso mozzarella") para que "con queso"
            # en el plato escale el queso en vez de meter una proteína ajena.
            _tok = nlow.split()[0] if nlow else ""
            if _tok and len(_tok) >= 4 and _tok in meal_text and (info.protein or 0) >= 18:
                chosen = info
                break
        if chosen is None:
            _pref = _DAIRY_EGG_PROTEIN_HINT if light else _MEAT_PROTEIN_HINT
            for info, nlow in _pool:
                if any(h in nlow for h in _pref):
                    chosen = info
                    break
        if chosen is None:
            chosen = _pool[0][0]  # la más magra (candidates ya viene ordenado)
        gap = target - cur_p
        grams = int(round(gap / (chosen.protein / 100.0)))
        grams = max(10, min(grams, max_add_g))
        f = grams / 100.0
        nm = str(chosen.name).lower()
        cook = "" if no_cook else " cocido"
        line = f"{grams}g de {nm}{cook} ({grams}g)"
        meal.setdefault("ingredients", []).append(line)
        if isinstance(meal.get("ingredients_raw"), list):
            meal["ingredients_raw"].append(line)
        meal["_protein_closed"] = True  # [P3-PROTEIN-IDEMPOTENT] no re-cerrar en re-assemble
        meal["protein"] = round(cur_p + chosen.protein * f)
        meal["carbs"] = round(_meal_macro_num(meal.get("carbs")) + chosen.carbs * f)
        meal["fats"] = round(_meal_macro_num(meal.get("fats")) + chosen.fats * f)
        meal["cals"] = round(_meal_macro_num(meal.get("cals")) + chosen.kcal * f)
        meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
        rec = meal.get("recipe")
        if isinstance(rec, list):
            # Nota SIN gramaje hardcodeado: el ingrediente (que puede re-escalarse en el trim/
            # cuantización) es la fuente de verdad → la nota nunca se desfasa.
            verb = "Añade" if no_cook else "Cocina e incorpora"
            meal["recipe"] = rec + [
                f"💪 {verb} el {nm} indicado en los ingredientes como fuente principal de "
                f"proteína de esta comida."]
        return grams
    except Exception as e:
        logger.warning(f"[P3-PROTEIN-FLOOR] closer falló para meal "
                       f"{str(meal.get('name'))[:40]!r}: {type(e).__name__}: {e}")
        return 0


def _ingredient_is_protein_dominant(s: str, db) -> bool:
    """True si la macro dominante (por kcal) del ingrediente es proteína → el reconcile NO lo
    escala (protege la proteína). Ingredientes no resueltos → False (se escalan: suelen ser
    carbo/veg/condimento de macro despreciable)."""
    mc = db.macros_from_ingredient_string(str(s))
    if not mc:
        return False
    pc = 4 * (mc.get("protein") or 0.0)
    return pc >= 4 * (mc.get("carbs") or 0.0) and pc >= 9 * (mc.get("fats") or 0.0)


def _protein_preserving_day_reconcile(meals: list, daily_cals: float, db) -> bool:
    """[P3-PROTEIN-FLOOR · 2026-06-13] Nivela las kcal del día al target PRESERVANDO la
    proteína: la proteína queda FIJA y solo se escalan carbos+grasas (macros e ingredientes
    NO proteína-dominantes) por un factor isocalórico. Reemplaza el cal-reconcile uniforme
    (que escalaba TODO, reduciendo la proteína que el closer acababa de añadir). Mantiene la
    consistencia receta↔macro (escala porción + macro juntos). Mutates meals. Retorna True
    si aplicó. Anchor: P3-PROTEIN-FLOOR."""
    try:
        from nutrition_db import rescale_ingredient_string as _resc
        P = sum(_meal_macro_num(m.get("protein")) for m in meals)
        day_cals = sum(_meal_macro_num(m.get("cals")) for m in meals)
        if day_cals <= 0:
            return False
        protein_cals = 4.0 * P
        cur_np = day_cals - protein_cals
        tgt_np = daily_cals - protein_cals
        if cur_np <= 0 or tgt_np <= 0:
            return False  # la proteína sola ≈ o excede el target → no tocar (raro)
        factor = max(0.4, min(1.8, tgt_np / cur_np))
        if abs(factor - 1.0) < 0.02:
            return False
        for m in meals:
            ings = m.get("ingredients")
            if isinstance(ings, list):
                m["ingredients"] = [
                    str(s) if _ingredient_is_protein_dominant(s, db) else _resc(str(s), factor)
                    for s in ings]
            raw = m.get("ingredients_raw")
            if isinstance(raw, list):
                m["ingredients_raw"] = [
                    str(s) if _ingredient_is_protein_dominant(s, db) else _resc(str(s), factor)
                    for s in raw]
            m["carbs"] = round(_meal_macro_num(m.get("carbs")) * factor)
            m["fats"] = round(_meal_macro_num(m.get("fats")) * factor)
            mp = round(_meal_macro_num(m.get("protein")))
            m["protein"] = mp
            m["cals"] = round(4 * mp + 4 * m["carbs"] + 9 * m["fats"])
            m["macros"] = [f"P:{mp}g", f"C:{m['carbs']}g", f"G:{m['fats']}g"]
        return True
    except Exception as e:
        logger.warning(f"[P3-PROTEIN-FLOOR] reconcile protein-preserving falló: "
                       f"{type(e).__name__}: {e}")
        return False


def _trim_day_protein_to_ceiling(meals: list, target_protein_day: float, db,
                                 *, ceiling_pct: float = 1.12) -> bool:
    """[P3-PROTEIN-FLOOR · 2026-06-13] Techo simétrico al piso: si el día entrega
    > ceiling_pct × target de proteína (el LLM/solver sobre-produjo, o el closer infló),
    escala las porciones e ingredientes PROTEÍNA-dominantes hacia abajo para traer el día
    AL target. Cierra el techo C1 sobre la proteína ENTREGADA (no solo la target) → evita
    g/kg por encima de 2.2. Las kcal las rebalancea el reconcile protein-preserving después.
    Mantiene consistencia receta↔macro (escala porción + macro juntos). Retorna True si trimó."""
    try:
        from nutrition_db import rescale_ingredient_string as _resc
        P = sum(_meal_macro_num(m.get("protein")) for m in meals)
        if target_protein_day <= 0 or P <= target_protein_day * ceiling_pct:
            return False
        factor = target_protein_day / P  # < 1
        for m in meals:
            ings = m.get("ingredients")
            if isinstance(ings, list):
                m["ingredients"] = [
                    _resc(str(s), factor) if _ingredient_is_protein_dominant(s, db) else str(s)
                    for s in ings]
            raw = m.get("ingredients_raw")
            if isinstance(raw, list):
                m["ingredients_raw"] = [
                    _resc(str(s), factor) if _ingredient_is_protein_dominant(s, db) else str(s)
                    for s in raw]
            mp = round(_meal_macro_num(m.get("protein")) * factor)
            mc = round(_meal_macro_num(m.get("carbs")))
            mf = round(_meal_macro_num(m.get("fats")))
            m["protein"] = mp
            m["cals"] = round(4 * mp + 4 * mc + 9 * mf)
            m["macros"] = [f"P:{mp}g", f"C:{mc}g", f"G:{mf}g"]
        return True
    except Exception as e:
        logger.warning(f"[P3-PROTEIN-FLOOR] trim de techo falló: {type(e).__name__}: {e}")
        return False


# [P3-VARIETY · 2026-06-13] Tokens de plato-base (para detectar repetición intra-día) +
# ingredientes premium (cap de apariciones) + descriptor sobre-usado. Advisory, no gate.
_VARIETY_BASE_DISHES = ("revoltillo", "revuelto", "tortilla", "batido", "smoothie",
                        "licuado", "wrap", "ensalada", "sopa", "sancocho", "guiso",
                        "guisad", "salteado", "horneado", "plancha", "pure")
# Sinónimos del mismo plato-base → token canónico (revuelto/revoltillo = huevo revuelto;
# smoothie/licuado = batido) para que la detección intra-día no los trate como distintos.
_VARIETY_BASE_CANON = {"revuelto": "revoltillo", "smoothie": "batido",
                       "licuado": "batido", "guisad": "guiso"}
_PREMIUM_INGREDIENTS = ("ricotta", "yogur griego", "yogurt griego", "queso parmesano",
                        "queso mozzarella", "salmon", "salmón")


def build_variety_report(plan: dict) -> dict:
    """[P3-VARIETY · 2026-06-13] Reporte ADVISORY de variedad/pertinencia cultural (FS5):
    cuenta apariciones de huevo, descriptor 'cremoso', ingredientes premium, y platos-base
    repetidos el mismo día. NO bloquea (variedad es calidad blanda → un gate causaría loops
    de regen); surface telemetría + se inyecta a `result` para observabilidad. Cierra el
    hallazgo de la auditoría (huevo 6/12, cremoso 4/12, ricotta×3). Anchor: P3-VARIETY."""
    try:
        from constants import strip_accents
    except Exception:
        def strip_accents(s):
            import unicodedata
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    total_meals = egg_meals = cremoso = premium = same_day_repeats = 0
    issues = []
    for day in plan.get("days", []) or []:
        meals = day.get("meals", []) or []
        day_tokens = {}
        for meal in meals:
            total_meals += 1
            name_low = strip_accents(str(meal.get("name", "")).lower())
            if _meal_has_egg(meal, strip_accents):
                egg_meals += 1
            if "cremos" in name_low:
                cremoso += 1
            ings_low = strip_accents(" ".join(str(i) for i in meal.get("ingredients", []) or []).lower())
            if any(p in name_low or p in ings_low for p in _PREMIUM_INGREDIENTS):
                premium += 1
            tok = next((t for t in _VARIETY_BASE_DISHES if t in name_low), None)
            if tok:
                tok = _VARIETY_BASE_CANON.get(tok, tok)  # colapsa sinónimos del mismo plato
                day_tokens[tok] = day_tokens.get(tok, 0) + 1
        for tok, n in day_tokens.items():
            if n >= 2:
                same_day_repeats += 1
                issues.append(f"Día {day.get('day', '?')}: '{tok}' repetido {n}x el mismo día")
    egg_cap = max(3, round(total_meals * 0.25))  # ~2-3 en 12 comidas
    if egg_meals > egg_cap:
        issues.append(f"Huevo en {egg_meals}/{total_meals} comidas (cap sugerido {egg_cap})")
    if cremoso > 1:
        issues.append(f"Descriptor 'cremoso' en {cremoso} platos (cap sugerido 1)")
    if premium > 2:
        issues.append(f"Ingredientes premium en {premium} comidas (cap sugerido 2)")
    return {"total_meals": total_meals, "egg_meals": egg_meals, "cremoso": cremoso,
            "premium": premium, "same_day_repeats": same_day_repeats, "issues": issues,
            "ok": not issues}


# [P3-SLOT-DISTRIBUTION · 2026-06-13] Mapa nombre-de-slot (es-DO) → key del split canónico.
_SLOT_KEY_MAP = {
    "desayuno": "desayuno", "breakfast": "desayuno",
    "almuerzo": "almuerzo", "comida": "almuerzo", "lunch": "almuerzo",
    "cena": "cena", "dinner": "cena",
    "merienda": "merienda", "snack": "merienda", "merienda am": "merienda",
    "merienda pm": "merienda", "media manana": "merienda", "media tarde": "merienda",
    "merienda matutina": "merienda", "merienda vespertina": "merienda",
}


def _canonical_slot_fractions(meals: list) -> list:
    """[P3-SLOT-DISTRIBUTION · 2026-06-13] Fracción de macros/kcal por meal según el split
    FISIOLÓGICO canónico (`MEAL_SLOT_SPLITS`: desayuno 20% / almuerzo 35% / merienda 15% /
    cena 30% para 4 comidas), NO según la distribución (a menudo desbalanceada) que emite el
    LLM. Cierra el hallazgo de la auditoría: el desayuno concentraba 48% de las kcal y 62%
    de la proteína del día; usar el `cal_share` del LLM como target del solver propagaba ese
    desbalance (3 comidas bajo el umbral leucínico ~22g). Mapea cada slot por nombre; los
    no-mapeados reciben parte igual del remanente; el vector se normaliza a sumar 1.0 →
    preserva el total diario. Retorna lista de fracciones alineada con `meals`. Anchor:
    P3-SLOT-DISTRIBUTION."""
    try:
        from constants import strip_accents
    except Exception:
        def strip_accents(s):
            import unicodedata
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    from nutrition_calculator import MEAL_SLOT_SPLITS
    n = len(meals)
    if n == 0:
        return []
    split = MEAL_SLOT_SPLITS.get(n, MEAL_SLOT_SPLITS[4])
    mer_keys = [k for k in split if k.startswith("merienda")]
    fracs, mer_i = [], 0
    for m in meals:
        key = _SLOT_KEY_MAP.get(strip_accents(str(m.get("slot", "")).lower().strip()))
        f = None
        if key in split:
            f = split[key]
        elif key == "merienda" and mer_keys:
            f = split[mer_keys[min(mer_i, len(mer_keys) - 1)]]
            mer_i += 1
        fracs.append(f)
    assigned = sum(f for f in fracs if f is not None)
    n_un = sum(1 for f in fracs if f is None)
    if n_un:
        rem = max(0.0, 1.0 - assigned) / n_un
        fracs = [rem if f is None else f for f in fracs]
    total = sum(fracs) or 1.0
    return [f / total for f in fracs]


def _apply_portion_quantization(plan: dict, db) -> int:
    """[P3-PORTION-QUANTIZE · 2026-06-13] Redondea las porciones de cada meal a unidades
    de cocina medibles (¼ taza, ¼ cda, ½ unidad discreta, 5 g) vía
    `nutrition_db.quantize_ingredient_string`, y AJUSTA los macros del meal por el delta
    EXACTO del aporte de cada ingrediente reescalado (vía `db.macros_from_ingredient_string`)
    → receta↔macro↔lista quedan consistentes. Reescribe `ingredients` (+ `ingredients_raw`
    en lockstep, con el MISMO factor por índice). Cierra el hallazgo de la auditoría: las
    fracciones decimales no medibles ('0.66 huevos', '0.53 taza', '3.87 papas') matan la
    adherencia. Trade-off aceptado: el redondeo introduce una pequeña deriva del target
    diario (la auditoría prioriza medibilidad sobre precisión exacta). Corre tras el
    cal-reconcile y ANTES de la agregación de compras → los gramos redondeados fluyen a la
    lista. Mutates `plan` in-place. Retorna nº de meals modificados. Anchor: P3-PORTION-QUANTIZE."""
    from nutrition_db import quantize_ingredient_string, rescale_ingredient_string
    changed_meals = 0
    for day in plan.get("days", []) or []:
        for meal in day.get("meals", []) or []:
            ings = meal.get("ingredients")
            if not isinstance(ings, list) or not ings:
                continue
            new_ings, factors = [], []
            dp = dc = df = dk = 0.0
            any_change = False
            for ing in ings:
                s = str(ing)
                new_s, fac = quantize_ingredient_string(s)
                factors.append(fac)
                if abs(fac - 1.0) > 1e-6:
                    mc = db.macros_from_ingredient_string(s)  # aporte ORIGINAL del ingrediente
                    if mc:
                        dp += mc["protein"] * (fac - 1.0)
                        dc += mc["carbs"] * (fac - 1.0)
                        df += mc["fats"] * (fac - 1.0)
                        dk += mc["kcal"] * (fac - 1.0)
                if new_s != s:
                    any_change = True
                new_ings.append(new_s)
            if not any_change:
                continue
            meal["ingredients"] = new_ings
            raw = meal.get("ingredients_raw")
            if isinstance(raw, list) and len(raw) == len(factors):
                meal["ingredients_raw"] = [
                    rescale_ingredient_string(str(r), f) if abs(f - 1.0) > 1e-6 else str(r)
                    for r, f in zip(raw, factors)
                ]
            meal["protein"] = max(0, round(_meal_macro_num(meal.get("protein")) + dp))
            meal["carbs"] = max(0, round(_meal_macro_num(meal.get("carbs")) + dc))
            meal["fats"] = max(0, round(_meal_macro_num(meal.get("fats")) + df))
            meal["cals"] = max(0, round(_meal_macro_num(meal.get("cals")) + dk))
            meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
            changed_meals += 1
    return changed_meals


def _run_assembly_validations(
    result: dict,
    skeleton: dict,
    affected_days_set: set,
) -> None:
    """Ejecuta las validaciones post-assembly y mutará `result` con las
    keys `_skeleton_fidelity_errors`, `_recipe_coherence_errors`,
    `_schema_invalid`, `_schema_errors` cuando aplique.

    Diseñado para ejecutarse SIEMPRE — tanto en cache-hit como en path LLM.
    """
    import re as _re

    # 1. Skeleton Fidelity (Brecha 1) — trivial-pass cuando skeleton={}.
    skeleton_fidelity_errors = []
    skeleton_days = (skeleton or {}).get("days", [])
    for i, day in enumerate(result.get("days", [])):
        day_num = day.get("day", i + 1)
        if affected_days_set and day_num not in affected_days_set:
            continue
        skeleton_day = next((s for s in skeleton_days if s.get("day") == day_num), {})
        assigned_proteins = [_flatten_ingredient(p).lower() for p in skeleton_day.get("protein_pool", [])]
        if not assigned_proteins:
            continue  # cache-hit u skeleton vacío → nada que validar
        day_ingredients_text = " ".join(
            _flatten_ingredient(ing).lower() for meal in day.get("meals", [])
            for ing in meal.get("ingredients", [])
        )
        # [P0-SKELETON-FIDELITY-MATCH · 2026-06-13] Matcher tolerante a los
        # descriptores/alternativas del protein_pool (ver _skeleton_protein_present).
        # El `p not in text` directo daba falsos positivos de "omitió".
        missing_proteins = [
            p for p in assigned_proteins
            if not _skeleton_protein_present(p, day_ingredients_text)
        ]
        # [P3-SKELETON-FIDELITY-CRITIQUE-AWARE · 2026-05-16] Threshold dinámico
        # según si self_critique modificó el día. Pre-fix: threshold hardcoded
        # >=2 missing → rechazo crítico cuando self_critique legítimamente
        # reemplazaba 1-2 proteínas para mejorar diversidad/slot coherence.
        # Incidente 2026-05-16 plan post-reset: skeleton asignó
        # ['soya/tofu', 'queso mozzarella'] a Día 1; critique sugirió
        # "Cambiar las claras de la cena por queso"; LLM removió ambas →
        # fidelity check rechazó fatal → plan crítico abortado pese a que
        # el día tenía proteínas válidas (solo no las DEL SKELETON ORIGINAL).
        # Post-fix: días con `_critique_applied=True` toleran missing 2/3
        # del pool (solo flagean si TODAS las proteínas asignadas se
        # removieron). Días sin self_critique mantienen threshold estricto
        # >=2 (preserva detección de bugs del day_generator que ignora
        # skeleton — caso original que motivó la check).
        _critique_applied_for_day = bool(day.get("_critique_applied"))
        _missing_threshold = 3 if _critique_applied_for_day else 2
        if len(missing_proteins) >= _missing_threshold:
            msg = f"Día {day_num} omitió múltiples proteínas clave asignadas: {missing_proteins}"
            logger.warning(
                f"⚠️ [SKELETON FIDELITY] {msg} "
                f"(threshold={_missing_threshold}, critique_applied={_critique_applied_for_day})"
            )
            skeleton_fidelity_errors.append(msg)
    if skeleton_fidelity_errors:
        result["_skeleton_fidelity_errors"] = skeleton_fidelity_errors

    # 2. Recipe Coherence (Brechas 3 y 4) — aplica SIEMPRE.
    # ------------------------------------------------------------
    # Estructura: `{ KEY_GENERICA: [SINONIMOS_PERMITIDOS_EN_INGREDIENTES] }`
    # - KEY = palabra que el LLM puede escribir GENÉRICAMENTE en el texto
    #   de la receta ("La receta indica `pescado`...").
    # - VALUES = todos los nombres específicos que pueden aparecer en
    #   `ingredients` y SATISFACEN el match (incluyendo el KEY mismo).
    #
    # Si recipe.text contiene el KEY pero ingredients NO contienen ningún
    # synonym → flag como `_recipe_coherence_errors` → review_plan_node
    # eleva a HIGH severity → retry forzado (o entrega del best previo
    # vía P0-PIPE-1).
    #
    # [P2-COH-1] Fish list expandida: merluza, róbalo, pargo, corvina,
    # mahi-mahi, lubina, carite, jurel — todos comunes en plan dominicano.
    # Sin estos, recetas como "Merluza Estofada Boca Chica" cuyo texto
    # decía "el pescado" disparaban falso positivo HIGH y rechazaban planes
    # válidos (incidente 2026-05-05).
    #
    # Nota: constants.PROTEIN_SYNONYMS tiene el mapa canónico para
    # categorización de ingredients (consumido por fact_extractor,
    # ai_helpers, agent). Este mapa LOCAL es complementario — incluye
    # singulares ("huevo", "camarón") porque el LLM frecuentemente los
    # escribe en singular en el texto de receta, mientras el mapa canónico
    # solo tiene plurales para evitar pollution con el reverse-map global.
    # Mantener AMBOS sincronizados al añadir nuevos sinónimos comunes.
    recipe_coherence_errors = []
    protein_synonyms = {
        "pescado": [
            "pescado", "pescados",
            # Fish específicos comunes en RD/Caribbean
            "chillo", "dorado", "mero", "salmón", "salmon", "tilapia",
            "bacalao", "atún", "atun", "sardina", "sardinas",
            # [P2-COH-1] Adiciones del incidente 2026-05-05
            "merluza", "róbalo", "robalo", "pargo", "corvina",
            "mahi-mahi", "mahi mahi", "mahimahi", "lubina",
            "carite", "jurel", "lambí", "lambi",
            # Filete genérico + composiciones
            "filete de pescado", "filete de mero", "filete de tilapia",
            "filete de chillo", "filete de dorado", "filete de bacalao",
            "filete de merluza", "filete de salmón", "filete de salmon",
        ],
        "pollo": [
            "pollo", "pechuga", "muslo", "muslos", "alitas", "alas",
            # Cortes / preparaciones comunes
            "encuentros", "cuartos", "pernil de pollo", "filete de pollo",
            "chicharrón de pollo", "chicharron de pollo",
        ],
        "res": [
            "res", "carne", "filete", "sirloin", "churrasco", "molida",
            # Cortes dominicanos comunes
            "carne de res", "carne molida", "lomo", "vacío", "vacio",
            "puyazo", "asado", "picadillo", "ropa vieja", "falda",
            "bistec", "bisteck",
        ],
        "cerdo": [
            "cerdo", "chuleta", "chuletas", "lomo", "masita",
            # [P2-COH-1] Cortes faltantes
            "costilla", "costillas", "pernil", "pernil de cerdo",
            "tocino", "chicharrón", "chicharron",
        ],
        "huevo": ["huevo", "huevos", "clara", "claras", "yema", "yemas",
                  "tortilla", "revoltillo"],
        "huevos": ["huevo", "huevos", "clara", "claras", "yema", "yemas",
                   "tortilla", "revoltillo"],
        "pavo": [
            "pavo", "pechuga de pavo", "jamón de pavo", "jamon de pavo",
            # [P2-COH-1] LLM también escribe estas variantes
            "pavo molido", "carne de pavo", "pavo asado", "pavo desmenuzado",
        ],
        "camarón": [
            "camarón", "camarones", "camaron",
            # [P2-COH-1] Variantes comerciales
            "gambas", "langostinos",
        ],
        "camarones": [
            "camarón", "camarones", "camaron",
            "gambas", "langostinos",
        ],
    }
    stopwords = RECIPE_INGREDIENT_STOPWORDS
    for day in result.get("days", []):
        for meal in day.get("meals", []):
            ingredients = [_flatten_ingredient(i).lower() for i in meal.get("ingredients", [])]
            _recipe_raw = meal.get("recipe", "")
            _recipe_steps = list(_recipe_raw) if isinstance(_recipe_raw, list) else ([_recipe_raw] if _recipe_raw else [])
            recipe = " ".join(str(s) for s in _recipe_steps).lower()

            # Proteína REAL del meal (primer sinónimo presente en ingredients) — para el auto-fix.
            _actual_protein = None
            for _ing in ingredients:
                for _syns in protein_synonyms.values():
                    _m = next((_s for _s in _syns if _re.search(r'\b' + _re.escape(_s) + r'\b', _ing)), None)
                    if _m:
                        _actual_protein = _m
                        break
                if _actual_protein:
                    break

            # FORWARD (Brecha 4): la receta menciona una proteína KEY sin sinónimo en ingredients.
            _orphan_keys = []
            for cp, synonyms in protein_synonyms.items():
                if _re.search(r'\b' + _re.escape(cp) + r'\b', recipe):
                    if not any(any(_re.search(r'\b' + _re.escape(syn) + r'\b', ing) for syn in synonyms) for ing in ingredients):
                        _orphan_keys.append(cp)
            if _orphan_keys:
                if RECIPE_COHERENCE_AUTOFIX and _recipe_steps:
                    # [P3-RECIPE-COHERENCE-AUTOFIX] Reemplaza la mención huérfana por la
                    # proteína real del meal (o "proteína") → coherente, sin retry.
                    _repl = _actual_protein or "proteína"
                    _new_steps = []
                    for _step in _recipe_steps:
                        _s = str(_step)
                        for _cp in _orphan_keys:
                            _s = _re.sub(r'\b' + _re.escape(_cp) + r'\b', _repl, _s, flags=_re.IGNORECASE)
                        _new_steps.append(_re.sub(r'\s{2,}', ' ', _s).strip())
                    _recipe_steps = _new_steps
                    meal["recipe"] = _new_steps
                    recipe = " ".join(_new_steps).lower()
                    logger.info(f"🩹 [RECIPE-COHERENCE-AUTOFIX] Día {day.get('day')} "
                                f"{str(meal.get('name'))[:30]!r}: mención(es) huérfana(s) "
                                f"{_orphan_keys} → {_repl!r} (evita retry, retry_penalty=1.0)")
                else:
                    for _cp in _orphan_keys:
                        _syns3 = protein_synonyms.get(_cp, [])[:3]
                        recipe_coherence_errors.append(
                            f"Día {day.get('day')}, {meal.get('name')}: La receta indica "
                            f"'{_cp}' pero no hay ningún ingrediente equivalente (ej. {', '.join(_syns3)}) listado.")

            # COMPLETION (Brecha 3): falta un paso final de servido.
            completion_pattern = r'\b(sirve|servir|montaje|monta|emplata|emplatar|empaca|empacar|disfruta|disfrutar|agrega)\b'
            if not _re.search(completion_pattern, recipe):
                if RECIPE_COHERENCE_AUTOFIX:
                    _recipe_steps = _recipe_steps + ["Montaje: Sirve y disfruta tu comida."]
                    meal["recipe"] = _recipe_steps
                    recipe = " ".join(str(s) for s in _recipe_steps).lower()
                    logger.info(f"🩹 [RECIPE-COHERENCE-AUTOFIX] Día {day.get('day')} "
                                f"{str(meal.get('name'))[:30]!r}: añadido paso final faltante")
                else:
                    recipe_coherence_errors.append(
                        f"Día {day.get('day')}, {meal.get('name')}: La receta parece incompleta, "
                        f"falta un paso final (ej: 'Servir', 'Montaje' o 'Empacar').")

            for ing in ingredients:
                clean_ing = _re.sub(r'[\d\.,\(\)/\-]', ' ', ing)
                words = [w.strip() for w in clean_ing.split() if w.strip() and len(w.strip()) > 2]
                core_nouns = [w for w in words if w not in stopwords]
                if not core_nouns:
                    continue
                # [P6-AUTO-PATCH-1] Antes: solo `core_nouns[0]` se chequeaba
                # contra la receta. Para ingredientes multi-palabra como
                # "lomo de cerdo", "queso mozzarella fresco", "ensalada de
                # tomate y cebolla", esto producía falso positivo cuando la
                # receta usaba el segundo o tercer núcleo (ej. recipe dice
                # "el cerdo" pero no "el lomo").
                #
                # El falso positivo cascadeaba: emitía error "ingrediente
                # principal 'lomo' está listado pero no se menciona" →
                # `_auto_patch_ingredient_coherence` removía CUALQUIER
                # ingrediente con substring 'lomo' → "lomo de cerdo"
                # eliminado → re-review forward-check fallaba "recipe dice
                # 'cerdo' pero no hay ingrediente equivalente".
                # Caso real corrida 2026-05-05 13:12 — Día 1 Cerdo en
                # Salsa Criolla terminó rechazado por minor en Día 1
                # (ortogonal al surgical regen en Días 2/3) → P0-PIPE-1
                # rolled back todo el surgical fix de P5.
                #
                # Ahora chequeamos TODOS los core_nouns. Si AL MENOS UNO
                # aparece en la receta, el ingrediente está "usado". Solo
                # flageamos si NINGUNO se menciona.
                any_mentioned = False
                for cn in core_nouns:
                    if len(cn) < 4:
                        continue
                    prefix = cn[:min(5, len(cn))]
                    permissive_pattern = r'\b' + _re.escape(prefix) + r'[a-z]*\b'
                    if _re.search(permissive_pattern, recipe):
                        any_mentioned = True
                        break
                if not any_mentioned:
                    # Mensaje conserva el primer core_noun de tamaño ≥4
                    # (más identificativo); fallback a core_nouns[0] si
                    # ninguno alcanza el threshold.
                    err_noun = next(
                        (cn for cn in core_nouns if len(cn) >= 4),
                        core_nouns[0],
                    )
                    msg = f"Día {day.get('day')}, {meal.get('name')}: El ingrediente principal '{err_noun}' está listado pero no se menciona en las instrucciones de la receta."
                    recipe_coherence_errors.append(msg)
    if recipe_coherence_errors:
        result["_recipe_coherence_errors"] = list(set(recipe_coherence_errors))

    # 3. Schema Validation contra PlanModel — aplica SIEMPRE.
    # Un plan cacheado con shape vieja (campo renombrado, validación nueva
    # añadida después) cae aquí. `review_plan_node` luego eleva a critical
    # vía la cláusula `if plan.get("_schema_invalid")`.
    try:
        PlanModel(**result)
    except Exception as e:
        err_msg = str(e)[:500]
        logger.error(f"🚨 [ASSEMBLY VALIDATION] Plan corrupto post-assembly detectado: {err_msg}")
        result["_schema_invalid"] = True
        result["_schema_errors"] = err_msg


# [P4-CONSTRAINT-ABC · 2026-06-14] Cuerpos VERBATIM extraídos del cap renal para que el motor de
# constraints (`clinical_constraints.RenalProteinCapConstraint`) DELEGUE a ellos sin reimplementar la
# matemática validada. `_enforce_renal_per_meal` = cuerpo del Guard 1 de la capa clínica;
# `_renal_exit_safety_net` = bloque renal de la red de salida (`_apply_final_defense_guardrails`). El
# refactor es behavior-preserving: estos son copia textual, los call sites llaman al engine que llama
# a estos. Tooltip-anchor: P4-CONSTRAINT-ABC.
def _enforce_renal_per_meal(plan: dict, pg: float, daily_cals: float, db) -> None:
    """Cuerpo verbatim del Guard 1 (ERC enforcement determinista per-comida). El gate (CONDITION_RULES_
    ENABLED + PROTEIN_FLOOR_ENABLED + db + renal_protein_cap.applied + pg>0) lo mantiene el call site."""
    try:
        _enf_days = 0
        for _d in plan.get("days", []) or []:
            _rmeals = _d.get("meals", []) or []
            if _trim_day_protein_to_ceiling(_rmeals, pg, db, ceiling_pct=1.0):
                _enf_days += 1
            if daily_cals > 0:
                _protein_preserving_day_reconcile(_rmeals, daily_cals, db)
        plan["renal_protein_cap"]["meals_enforced"] = True
        plan["renal_protein_cap"]["enforced_days"] = _enf_days
    except Exception as _renf_e:
        plan["renal_protein_cap"]["meals_enforced"] = False
        logger.warning(f"[P3-FALLBACK-CLINICAL-LAYER] enforcement renal per-comida falló: "
                       f"{type(_renf_e).__name__}: {_renf_e}")


def _renal_exit_safety_net(plan: dict, nutrition: dict, form_data: dict) -> None:
    """Cuerpo verbatim del bloque RENAL de la red de seguridad en el punto único de salida. Garantiza
    que un paciente renal NUNCA reciba un plan (incl. fallback) sin la advertencia de nefrólogo + la meta
    del cap. No-op en el happy path (assemble ya seteó renal_gate). El gate genérico (cualquier condición)
    queda inline en el call site. Fail-safe lo maneja el call site."""
    _rcap_src = nutrition.get("renal_protein_cap") if isinstance(nutrition, dict) else None
    if (PRO_REVIEW_FLAG_ENABLED and isinstance(plan, dict)
            and isinstance(_rcap_src, dict) and _rcap_src.get("applied")):
        if not plan.get("renal_protein_cap"):
            plan["renal_protein_cap"] = {**_rcap_src, "meals_enforced": True}
        _rpr_existing = plan.get("requires_professional_review")
        if not (isinstance(_rpr_existing, dict) and _rpr_existing.get("renal_gate")):
            _cap_txt = (f" Se aplicó un límite conservador de proteína a ~{_rcap_src.get('protein_g')}g/día "
                        f"(≈{RENAL_PROTEIN_GKG_CEILING} g/kg) como medida de seguridad.")
            _comorbid_txt = (" Tienes diabetes y enfermedad renal a la vez: el balance fibra/leguminosas "
                             "vs potasio/fósforo/proteína SOLO lo define tu nefrólogo/nutricionista."
                             if _rcap_src.get("comorbid_diabetes") else "")
            _conds = [str(c) for c in (form_data.get("medicalConditions") or [])
                      if str(c).strip().lower() not in _MEDICAL_NONE_SENTINELS]
            plan["requires_professional_review"] = {
                "flag": True, "renal_gate": True, "conditions": _conds,
                "note": ("🫘 CONDICIÓN RENAL DETECTADA — IMPORTANTE: la nutrición en enfermedad renal "
                         "depende de tu estadio (G1–G5) y de si estás en diálisis, y DEBE ser supervisada "
                         "por tu nefrólogo o nutricionista renal." + _cap_txt + " Este plan NO es una "
                         "prescripción renal: el potasio y el fósforo (críticos en ERC) no se ajustan aquí." +
                         _comorbid_txt + " NO sigas este plan sin la validación de tu profesional de salud."),
            }
            logger.warning("🫘 [P3-CONDITION-RULES] Red de seguridad renal aplicó gate de derivación "
                           "profesional al plan entregado (cubre fallback/paths sin assemble).")


# [P3-FALLBACK-CLINICAL-LAYER · 2026-06-14] SSOT de la CAPA CLÍNICA DETERMINISTA (solver-independiente):
# los 8 guards FS1-FS9 que viven inline en assemble_plan_node (renal per-comida → food-safety →
# condition-subs → quantize → micros/suplementos → variedad → proveniencia → gate FS9). Extraída para
# que el FALLBACK matemático (`_get_extreme_fallback_plan`/`_repair_partial_plan`), que BYPASSA assemble,
# la HEREDE en vez de entregar un plan sin food-safety, sin sustitución de sodio/azúcar, sin reporte de
# micros y sin gate de derivación profesional. Self-contained: re-deriva `active_macros` (COPIA — jamás
# muta `nutrition` del caller), re-corre el cap renal de la fuente, y recomputa `_pg`/`_daily_cals` desde
# (plan, nutrition). Idempotente vía marker `_clinical_layer_applied`. Cada guard fail-safe individual.
# Comparte UNA instancia de IngredientNutritionDB (antes el bloque inline instanciaba 4). Mutates `plan`.
# Retorna `plan`. [Fase A: SSOT nueva consumida por el fallback; assemble sigue con su bloque inline.]
def _apply_deterministic_clinical_layer(plan: dict, form_data: dict, nutrition: dict = None) -> dict:
    if not isinstance(plan, dict):
        return plan
    if plan.get("_clinical_layer_applied"):
        return plan  # ya corrió (happy path en assemble o doble-llamada en delivery) → no re-aplicar
    if not isinstance(form_data, dict):
        form_data = {}
    nutrition = nutrition if isinstance(nutrition, dict) else {}

    # ── Re-derivación de los locals que assemble computa antes del bloque (espejo del post-proceso de
    # macros + cap renal de la fuente [P3-CONDITION-RULES] + _daily_cals/_pg). ──
    # active_macros: COPIA del macros de la fuente (jamás mutar el `nutrition` del caller — bug si fuese ref).
    # `or` (truthiness, NO .get(key, default)): endurecimiento intencional sobre el inline — un
    # total_daily_macros vacío {} cae a `macros` en vez de quedar como {} sin protein_str.
    _src_macros = nutrition.get("total_daily_macros") or nutrition.get("macros") or {}
    active_macros = dict(_src_macros) if isinstance(_src_macros, dict) else {}

    # Re-corre el cap renal de la fuente (espejo del cap [P3-CONDITION-RULES] de assemble): el fallback
    # (que bypassa assemble) queda capeado + con `renal_protein_cap` seteado → habilita el enforcement
    # per-comida + el gate FS9. En el happy path active_macros ya viene capeado → no-op. Fail-safe.
    try:
        if CONDITION_RULES_ENABLED and _is_renal_condition(form_data) and active_macros:
            _wkg = _weight_kg_from_form(form_data)
            _old_p = _meal_macro_num(active_macros.get("protein_str"))
            _cap = round(RENAL_PROTEIN_GKG_CEILING * _wkg)
            # Copia PRIVADA del header del plan antes de mutarlo → si un caller (p.ej. Fase B) pasara un
            # plan cuyo `macros` ES el dict de `nutrition`, jamás lo corromperíamos (defensa anti-aliasing).
            if isinstance(plan.get("macros"), dict):
                plan["macros"] = dict(plan["macros"])
                _pm = plan["macros"]
            else:
                _pm = None
            if _wkg > 0 and _cap > 0 and _cap < _old_p:
                _freed = (_old_p - _cap) * 4.0
                _diab = _is_diabetes_condition(form_data)
                active_macros["protein_str"] = f"{_cap}g"
                if _pm is not None:
                    _pm["protein"] = f"{_cap}g"
                if _diab:
                    _old_f = _meal_macro_num(active_macros.get("fats_str"))
                    _new_f = round(_old_f + _freed / 9.0)
                    active_macros["fats_str"] = f"{_new_f}g"
                    if _pm is not None:
                        _pm["fats"] = f"{_new_f}g"
                    _reassigned = "fat"
                else:
                    _old_c = _meal_macro_num(active_macros.get("carbs_str"))
                    _new_c = round(_old_c + (_old_p - _cap))
                    active_macros["carbs_str"] = f"{_new_c}g"
                    if _pm is not None:
                        _pm["carbs"] = f"{_new_c}g"
                    _reassigned = "carb"
                if not (plan.get("renal_protein_cap") or {}).get("applied"):
                    plan["renal_protein_cap"] = {
                        "applied": True, "gkg": RENAL_PROTEIN_GKG_CEILING,
                        "protein_g": _cap, "was_g": round(_old_p),
                        "weight_kg": round(_wkg, 1), "guideline": "KDIGO 2024 (G3-G5 no-diálisis)",
                        "reassigned_to": _reassigned, "comorbid_diabetes": bool(_diab),
                        "meals_enforced": False,
                    }
        # Propaga el cap si vino ya aplicado desde la fuente (espejo de la propagación inline).
        if (CONDITION_RULES_ENABLED and (nutrition.get("renal_protein_cap") or {}).get("applied")
                and not (plan.get("renal_protein_cap") or {}).get("applied")):
            plan["renal_protein_cap"] = {**nutrition["renal_protein_cap"], "meals_enforced": False}
    except Exception as _rc_e:
        logger.warning(f"[P3-FALLBACK-CLINICAL-LAYER] re-derivación del cap renal falló: "
                       f"{type(_rc_e).__name__}: {_rc_e}")

    _daily_cals = _meal_macro_num(plan.get("calories")) or _meal_macro_num(
        nutrition.get("total_daily_calories") or nutrition.get("target_calories"))
    _pg = _meal_macro_num(active_macros.get("protein_str"))
    if not _pg:  # fallback path: el header del plan trae macros={protein:"Ng"}
        _pm2 = plan.get("macros")
        if isinstance(_pm2, dict):
            _pg = _meal_macro_num(_pm2.get("protein"))

    # Una sola instancia de DB para todos los guards (food-safety no la usa). Si falla la construcción
    # (import roto), los guards quantize/micros/proveniencia se omiten — emitimos UN warning para no
    # perder la señal (el bloque inline la emitía per-guard al instanciar 4 DBs). Anchor: P3-FALLBACK-CLINICAL-LAYER.
    try:
        from nutrition_db import IngredientNutritionDB
        _db = IngredientNutritionDB()
    except Exception as _db_e:
        _db = None
        logger.warning(f"[P3-FALLBACK-CLINICAL-LAYER] IngredientNutritionDB no disponible — guards "
                       f"quantize/micros/proveniencia se omiten: {type(_db_e).__name__}: {_db_e}")

    # [P4-CONSTRAINT-ABC · 2026-06-14] Motor de constraints declarativo (posición-preservante): despacha
    # los guards condition-específicos (Guard 1 renal, Guard 3 sustituciones) vía constraints en orden de
    # precedencia, SIN reordenar (food-safety sigue físicamente entre ambos). El cap renal NO se
    # reimplementa — los constraints delegan a las funciones validadas. Fail-safe: si el engine falla, los
    # guards inline-equivalentes (renal/subs) simplemente no corren (mismo efecto que su gate en False).
    try:
        from clinical_constraints import ClinicalConstraintEngine, ClinicalContext
        _eng = ClinicalConstraintEngine(form_data)
        _ctx = ClinicalContext(db=_db, daily_cals=_daily_cals, protein_g=_pg, active_macros=active_macros)
    except Exception as _eng_e:
        _eng = None
        logger.warning(f"[P4-CONSTRAINT-ABC] engine no disponible: {type(_eng_e).__name__}: {_eng_e}")

    # ── Guard 1 (FS6/ERC): enforcement determinista per-comida del cap renal (vía RenalProteinCapConstraint) ──
    # [P4-CONSTRAINT-ABC review] Fallback directo si el engine no cargó (simétrico a Guard 3): el trim
    # renal per-comida es SEGURIDAD iatrogénica — nunca debe saltarse silenciosamente por un import roto.
    if (CONDITION_RULES_ENABLED and PROTEIN_FLOOR_ENABLED and _db is not None
            and isinstance(plan.get("renal_protein_cap"), dict)
            and plan["renal_protein_cap"].get("applied") and _pg > 0):
        if _eng is not None:
            _eng.enforce_one("renal", plan, nutrition, _ctx)   # delega a _enforce_renal_per_meal
        else:
            _enforce_renal_per_meal(plan, _pg, _daily_cals, _db)   # fallback inline (engine no disponible)

    # ── Guard 2 (FS1): food-safety / huevo crudo (espejo [P3-FOOD-SAFETY]) ──
    if FOOD_SAFETY_GUARD:
        try:
            _fs_n = _apply_food_safety_fixes(plan)
            if _fs_n:
                logger.warning(f"🥚 [P3-FOOD-SAFETY] Mitigó huevo crudo/poco cocido en {_fs_n} comida(s)")
        except Exception as _fs_e:
            logger.warning(f"[P3-FOOD-SAFETY] error: {type(_fs_e).__name__}: {_fs_e}")

    # ── Guard 2.5 (FS-IgE): sustitución QUIRÚRGICA de alérgenos IgE declarados (espejo [P0-ALLERGEN-SUBS]) ──
    # Corre ANTES de las sustituciones por condición (Guard 3): la seguridad del alérgeno tiene la mayor
    # precedencia. Swap del ingrediente ofensor por una alternativa segura que resuelve al catálogo,
    # conservando el plan rico. El backstop `_scan_allergen_violations` (review) sigue escalando cualquier
    # residual a rechazo crítico → cero regresión de seguridad. fish/shellfish/soy/gluten únicamente.
    if ALLERGEN_SUBSTITUTION_ENABLED:
        try:
            _al_n = _apply_allergen_substitutions(plan, form_data)
            if _al_n:
                logger.warning(f"🛡️ [P0-ALLERGEN-SUBS] Sustituyó alérgeno(s) declarado(s) por alternativa "
                               f"segura en {_al_n} comida(s)")
        except Exception as _al_e:
            logger.warning(f"[P0-ALLERGEN-SUBS] error: {type(_al_e).__name__}: {_al_e}")

    # ── Guard 3: sustitución de ingredientes por condición DM2/HTA/dislipidemia (vía SubstitutionEngineConstraint) ──
    if DM2_SUGAR_GUARD:
        try:
            # delega al pase único `_apply_condition_substitutions`; fallback inline si el engine no cargó.
            _sg_n = (_eng.enforce_one("substitutions", plan, nutrition, _ctx) if _eng is not None
                     else _apply_condition_substitutions(plan, form_data))
            if _sg_n:
                logger.warning(f"⚕️ [P3-CONDITION-ENGINE] Sustituyó ingredientes contraindicados "
                               f"(azúcar/sodio/satfat) en {_sg_n} comida(s)")
        except Exception as _sg_e:
            logger.warning(f"[P3-CONDITION-ENGINE] error: {type(_sg_e).__name__}: {_sg_e}")

    # ── Guard 4 (FS2): cuantización de porciones a unidades medibles (espejo [P3-PORTION-QUANTIZE]) ──
    if PORTION_QUANTIZE_ENABLED and _db is not None:
        try:
            _q_n = _apply_portion_quantization(plan, _db)
            if _q_n:
                logger.info(f"📏 [P3-PORTION-QUANTIZE] Redondeó porciones en {_q_n} comida(s)")
        except Exception as _q_e:
            logger.warning(f"[P3-PORTION-QUANTIZE] error: {type(_q_e).__name__}: {_q_e}")

    # ── Guard 5 (FS4/FS8): panel de micros + suplementación accionable (espejo [P3-MICRONUTRIENTS]) ──
    if MICRONUTRIENT_REPORT_ENABLED and _db is not None:
        try:
            from micronutrients import build_micronutrient_report
            _sex = form_data.get("gender", "female")
            _mn = build_micronutrient_report(
                plan, _db, sex=_sex,
                conditions=_condition_strings(form_data), daily_kcal=_daily_cals,
                fiber_per_1000kcal=DM2_FIBER_G_PER_1000KCAL)
            plan["micronutrient_report"] = _mn
            _ngaps = len(_mn.get("gaps", []))
            logger.info(f"🧪 [P3-MICRONUTRIENTS] Panel de micros computado "
                        f"(cobertura {int(_mn.get('coverage', 0)*100)}%, {_ngaps} gap(s) advisory)")
            if SUPPLEMENT_ADVICE_ENABLED:
                from micronutrients import build_supplement_recommendations
                _supp = build_supplement_recommendations(_mn, sex=_sex)
                if _supp.get("count"):
                    plan["micronutrient_supplement_advice"] = _supp
                    logger.info(f"💊 [P3-SUPPLEMENT-ADVICE] {_supp['count']} recomendación(es) de "
                                f"suplementación para cerrar gaps de micros")
        except Exception as _mn_e:
            logger.warning(f"[P3-MICRONUTRIENTS] error: {type(_mn_e).__name__}: {_mn_e}")

    # ── Guard 6 (FS5): reporte advisory de variedad/pertinencia cultural (espejo [P3-VARIETY]) ──
    if VARIETY_REPORT_ENABLED:
        try:
            plan["variety_report"] = build_variety_report(plan)
        except Exception as _vr_e:
            logger.warning(f"[P3-VARIETY] error: {type(_vr_e).__name__}: {_vr_e}")

    # ── Guard 7 (M1): trazabilidad de proveniencia USDA FDC (espejo [P3-DATA-PROVENANCE]) ──
    # NO en fallbacks: el plan matemático usa ingredientes-plantilla genéricos ("arroz blanco", sin
    # gramos) que rara vez resuelven a un fdc_id → la nota "X de Y trazables a USDA" sería engañosa
    # (afirmaría anclaje de datos que ese plan no tiene). El happy path (assemble) sí la computa.
    if DATA_PROVENANCE_ENABLED and _db is not None and not plan.get("_is_fallback"):
        try:
            _seen_prov = {}
            _tot_ings = _res_ings = 0  # [P4-UNIFIED-RESOLVER] cobertura de resolución (mismo loop, 0 costo extra)
            for _d in plan.get("days", []) or []:
                for _m in _d.get("meals", []) or []:
                    for _ing in _m.get("ingredients", []) or []:
                        _tot_ings += 1
                        _info = _db.lookup(str(_ing))
                        if not _info:
                            continue
                        _res_ings += 1
                        _key = (_info.name or str(_ing)).lower()
                        _seen_prov.setdefault(_key, _info)
            _total_u = len(_seen_prov)
            _usda = [(i.name, i.fdc_id) for i in _seen_prov.values()
                     if i.fdc_id and str(i.source or "").lower() == "usda"]
            plan["data_provenance"] = {
                "primary_source": "USDA FoodData Central",
                "secondary_sources": ["INCAP/LATINFOODS", "manual (curado es-DO)"],
                "ingredients_resolved": _total_u,
                "usda_traced": len(_usda),
                "fdc_sample": [{"name": n, "fdc_id": int(f)} for n, f in _usda[:8]],
                "note": (f"Datos nutricionales anclados a fuentes verificables: {len(_usda)} de "
                         f"{_total_u} ingredientes resueltos trazables a USDA FoodData Central "
                         "(IDs públicos en fdc.nal.usda.gov); el resto, INCAP/curado es-DO."),
            }
            # [P4-UNIFIED-RESOLVER · 2026-06-14] KPI de cobertura de resolución: qué fracción de los
            # ingredientes del plan resuelve al catálogo (los NO resueltos son el "0 silencioso" — aportan
            # 0 macros al solver). Sube con el resolver unificado (fuzzy + Cohere). Telemetría de flota.
            if _tot_ings:
                plan["resolution_coverage"] = {
                    "resolved": _res_ings, "total": _tot_ings, "pct": round(_res_ings / _tot_ings, 3)}
                logger.info(f"🔎 [P4-UNIFIED-RESOLVER] Cobertura de resolución: {_res_ings}/{_tot_ings} "
                            f"({round(100 * _res_ings / _tot_ings)}%) ingredientes resueltos al catálogo")
        except Exception as _pv_e:
            logger.warning(f"[P3-DATA-PROVENANCE] error: {type(_pv_e).__name__}: {_pv_e}")

    # ── Guard 8 (FS9): gate de revisión profesional + nota ERC reforzada (espejo [P3-PRO-REVIEW-FLAG]) ──
    if PRO_REVIEW_FLAG_ENABLED:
        try:
            _conds = form_data.get("medicalConditions") or form_data.get("medical_conditions")
            if _has_real_medical_flags(_conds):
                _cond_list = [str(c) for c in (_conds if isinstance(_conds, list) else [_conds])
                              if str(c).strip().lower() not in _MEDICAL_NONE_SENTINELS]
                _note = ("⚕️ Declaraste condición(es) de salud (" + ", ".join(_cond_list) + "). "
                         "Este plan las considera de forma general pero NO sustituye la evaluación "
                         "de tu médico o nutricionista. Consúltalo antes de seguir este plan, "
                         "especialmente para ajustar porciones, sodio, azúcares o restricciones específicas.")
                _renal_flag = CONDITION_RULES_ENABLED and _is_renal_condition(form_data)
                if _renal_flag:
                    _cap_info = plan.get("renal_protein_cap") or {}
                    _cap_txt = (f" Se aplicó un límite conservador de proteína a ~{_cap_info.get('protein_g')}g/día "
                                f"(≈{RENAL_PROTEIN_GKG_CEILING} g/kg) como medida de seguridad."
                                if _cap_info.get("meals_enforced") else "")
                    _comorbid_txt = (" Tienes diabetes y enfermedad renal a la vez: las recomendaciones de "
                                     "fibra/leguminosas (diabetes) y de potasio/fósforo/proteína (renal) deben "
                                     "balancearse caso por caso — esto SOLO lo define tu nefrólogo/nutricionista."
                                     if _cap_info.get("comorbid_diabetes") else "")
                    _note = ("🫘 CONDICIÓN RENAL DETECTADA — IMPORTANTE: la nutrición en enfermedad renal "
                             "depende de tu estadio (G1–G5) y de si estás en diálisis, y DEBE ser supervisada "
                             "por tu nefrólogo o nutricionista renal." + _cap_txt + " Este plan NO es una "
                             "prescripción renal: el potasio y el fósforo (críticos en ERC) no se ajustan aquí." +
                             _comorbid_txt + " NO sigas este plan sin la validación de tu profesional de salud. ") + _note
                plan["requires_professional_review"] = {
                    "flag": True,
                    "conditions": _cond_list,
                    "renal_gate": bool(_renal_flag),
                    "note": _note,
                }
        except Exception as _pr_e:
            logger.warning(f"[P3-PRO-REVIEW-FLAG] error: {type(_pr_e).__name__}: {_pr_e}")

    plan["_clinical_layer_applied"] = True
    return plan


@_node_label("assembler")
async def assemble_plan_node(state: PlanState) -> dict:
    """Ensambla el plan final combinando skeleton + días paralelos + datos del calculador, o re-ensambla un plan en caché."""
    nutrition = state["nutrition"]
    form_data = state["form_data"]

    logger.info(f"\n{'='*60}")
    logger.info(f"🔧 [ENSAMBLADOR] Combinando plan final...")
    logger.info(f"{'='*60}")

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
            logger.error(
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
                logger.error(
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
            logger.info(
                f"💊 [SUPPLEMENTS] Eliminados {_stripped_supps} en campo supplements + "
                f"{_stripped_ings} colados en ingredients (includeSupplements=false)."
            )

    # [EGG-WHITE-CAP] Cap programático de claras de huevo por meal y por día.
    # El planner las usa como proteína fácil sin límite (visto 2026-05-06: 16
    # claras en una cena → revisor médico rechazó CRÍTICO por avidina/estrés
    # renal, usuario quedaba en bucle). Aquí parseamos cada ingredient string
    # tipo "11 claras de huevo" / "5 claras", recortamos a MAX_PER_MEAL y
    # luego sumamos por día y recortamos a MAX_PER_DAY (raspando el último
    # meal con claras del día).
    # NOTA: usamos `_re` (alias módulo, línea 1013), no `re`. Esta función
    # tiene `import re` local más abajo (línea ~6329 en consolidación), lo
    # que convierte `re` en variable local en toda la función — referenciar
    # `re` aquí, antes de ese import, lanza UnboundLocalError.
    _egg_white_pattern = _re.compile(r'^\s*(\d+(?:[.,]\d+)?)\s+(claras?\b.*)', _re.IGNORECASE)

    def _parse_eggw_count(_ing_str):
        m = _egg_white_pattern.match(str(_ing_str))
        if not m:
            return None, None
        try:
            n = float(m.group(1).replace(',', '.'))
        except (TypeError, ValueError):
            return None, None
        return n, m.group(2)

    _capped_per_meal = 0
    _capped_per_day = 0
    for _d in result.get("days") or []:
        _day_total = 0.0
        _day_meals_with_eggw = []
        for _m in _d.get("meals") or []:
            _ings = _m.get("ingredients") or []
            for _idx, _ing in enumerate(_ings):
                _count, _rest = _parse_eggw_count(_ing)
                if _count is None:
                    continue
                _new_count = _count
                if _new_count > MAX_EGG_WHITES_PER_MEAL:
                    _new_count = float(MAX_EGG_WHITES_PER_MEAL)
                    _capped_per_meal += 1
                if _new_count != _count:
                    _ings[_idx] = f"{int(_new_count) if _new_count.is_integer() else _new_count} {_rest}"
                _day_total += _new_count
                _day_meals_with_eggw.append((_m, _idx, _new_count, _rest))
            _m["ingredients"] = _ings
        # Cap por día: raspar de los últimos meals con claras hasta cumplir el cap.
        while _day_total > MAX_EGG_WHITES_PER_DAY and _day_meals_with_eggw:
            _meal_ref, _ing_idx, _cur_count, _rest = _day_meals_with_eggw.pop()
            _excess = _day_total - MAX_EGG_WHITES_PER_DAY
            _reduced = max(0.0, _cur_count - _excess)
            if _reduced <= 0:
                # Eliminar el ingrediente entero
                _meal_ref["ingredients"][_ing_idx] = ""
            else:
                _meal_ref["ingredients"][_ing_idx] = (
                    f"{int(_reduced) if _reduced.is_integer() else _reduced} {_rest}"
                )
            _day_total -= (_cur_count - _reduced)
            _capped_per_day += 1
        # Limpiar ingredients vacíos que pudimos haber dejado al recortar.
        for _m in _d.get("meals") or []:
            _m["ingredients"] = [i for i in (_m.get("ingredients") or []) if str(i).strip()]

    # [P2-EGG-WHITE-MEALS-CAP · 2026-05-16] Tercer pass: limitar # meals/día
    # con claras como proteína base. Pre-fix los caps anteriores no limitaban
    # FRECUENCIA → reviewer médico rechazaba por "claras en múltiples comidas".
    _capped_per_meal_count = 0
    for _d in result.get("days") or []:
        # Encontrar los meals que tienen claras (cualquier cantidad).
        _meals_with_eggw_list = []  # list of (meal, ing_idx, count, rest)
        for _m in _d.get("meals") or []:
            _ings = _m.get("ingredients") or []
            for _idx, _ing in enumerate(_ings):
                _count, _rest = _parse_eggw_count(_ing)
                if _count is not None and _count > 0:
                    _meals_with_eggw_list.append((_m, _idx, _count, _rest))
                    break  # 1 entry de claras por meal es suficiente para contarlo
        # Si el día tiene más de MAX_MEALS_WITH_EGG_WHITES meals con claras,
        # los meals EN EXCESO (los últimos en orden) ven sus claras recortadas
        # a 1 simbólica (como aglutinante, no como proteína base).
        if len(_meals_with_eggw_list) > MAX_MEALS_WITH_EGG_WHITES:
            _meals_excess = _meals_with_eggw_list[MAX_MEALS_WITH_EGG_WHITES:]
            for (_m_ref, _ing_idx, _cur_count, _rest) in _meals_excess:
                if _cur_count > 1:
                    _m_ref["ingredients"][_ing_idx] = f"1 {_rest}"
                    _capped_per_meal_count += 1

    if _capped_per_meal or _capped_per_day or _capped_per_meal_count:
        logger.info(
            f"🥚 [EGG-WHITE-CAP] {_capped_per_meal} meal(s) con >{MAX_EGG_WHITES_PER_MEAL} "
            f"claras recortado(s); {_capped_per_day} ajuste(s) por cap diario "
            f"({MAX_EGG_WHITES_PER_DAY}/día); {_capped_per_meal_count} meal(s) recortados "
            f"a 1 clara simbólica por exceder cap de frecuencia "
            f"({MAX_MEALS_WITH_EGG_WHITES} meals/día)."
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
    logger.info(f"📅 [DAY NAMES] Inyectados: {injected_names} (start={start_dt.isoformat()}, tzOffset={tz_offset_minutes})")

    # Post-proceso: forzar valores exactos del calculador
    result["calories"] = nutrition.get("total_daily_calories", nutrition["target_calories"])
    active_macros = nutrition.get("total_daily_macros", nutrition["macros"])
    result["macros"] = {
        "protein": active_macros["protein_str"],
        "carbs": active_macros["carbs_str"],
        "fats": active_macros["fats_str"],
    }
    result["main_goal"] = nutrition["goal_label"]

    # [P3-CONDITION-RULES · 2026-06-14] ERC — CAP DE SEGURIDAD de proteína (KDIGO 2024). En
    # enfermedad renal G3-G5 no-diálisis la proteína se LIMITA a ~0.8 g/kg — lo OPUESTO al piso
    # alto del producto. Sin esto, el solver/closer empujarían la proteína a 2.2-2.6 g/kg, que es
    # IATROGÉNICO en ERC (acelera la progresión). Capeamos el TARGET en la fuente (`active_macros`
    # + `result["macros"]`) para que TODO aguas abajo (solver, closer, trim, validador de piso,
    # surgical regen, macros mostrados) use el target renal y quede consistente. Reasignamos las
    # kcal liberadas a carbohidrato (swap 1:1 g, iso-calórico — proteína y carbo = 4 kcal/g). NO es
    # una prescripción renal (estadio/diálisis/K/P no se modelan) → el gate FS9 abajo fuerza
    # derivación profesional. Fail-safe total. Anchor: P3-CONDITION-RULES.
    if CONDITION_RULES_ENABLED and _is_renal_condition(form_data):
        try:
            _wkg_renal = _weight_kg_from_form(form_data)
            _old_p_renal = _meal_macro_num(active_macros.get("protein_str"))
            _renal_cap = round(RENAL_PROTEIN_GKG_CEILING * _wkg_renal)
            if _wkg_renal > 0 and _renal_cap > 0 and _renal_cap < _old_p_renal:
                _freed_kcal_renal = (_old_p_renal - _renal_cap) * 4.0   # proteína = 4 kcal/g
                _renal_diabetic = _is_diabetes_condition(form_data)
                active_macros["protein_str"] = f"{_renal_cap}g"
                result["macros"]["protein"] = f"{_renal_cap}g"
                if _renal_diabetic:
                    # [P3-CONDITION-RULES] Comorbilidad diabético-nefropatía (causa #1 de ERC): NO
                    # volcar las kcal liberadas a carbohidrato (sube la carga glucémica, choca con la
                    # regla DM2). Van a GRASA saludable (9 kcal/g), que ni DM2 ni ERC restringen tan
                    # fuerte. El reparto fino lo valida el nefrólogo (gate FS9 reforzado abajo).
                    _old_f_renal = _meal_macro_num(active_macros.get("fats_str"))
                    _new_f_renal = round(_old_f_renal + _freed_kcal_renal / 9.0)
                    active_macros["fats_str"] = f"{_new_f_renal}g"
                    result["macros"]["fats"] = f"{_new_f_renal}g"
                    _reassigned = "fat"
                else:
                    _old_c_renal = _meal_macro_num(active_macros.get("carbs_str"))
                    _new_c_renal = round(_old_c_renal + (_old_p_renal - _renal_cap))  # swap proteína→carbo iso-kcal
                    active_macros["carbs_str"] = f"{_new_c_renal}g"
                    result["macros"]["carbs"] = f"{_new_c_renal}g"
                    _reassigned = "carb"
                result["renal_protein_cap"] = {
                    "applied": True, "gkg": RENAL_PROTEIN_GKG_CEILING,
                    "protein_g": _renal_cap, "was_g": round(_old_p_renal),
                    "weight_kg": round(_wkg_renal, 1), "guideline": "KDIGO 2024 (G3-G5 no-diálisis)",
                    "reassigned_to": _reassigned, "comorbid_diabetes": bool(_renal_diabetic),
                    "meals_enforced": False,  # lo setea el enforcement determinista per-comida abajo
                }
                logger.warning(
                    f"🫘 [P3-CONDITION-RULES] ERC detectada → cap de proteína {round(_old_p_renal)}g→"
                    f"{_renal_cap}g (~{RENAL_PROTEIN_GKG_CEILING} g/kg × {_wkg_renal:.1f}kg); kcal "
                    f"reasignadas a {_reassigned}{' (comorbilidad DM2)' if _renal_diabetic else ''}. "
                    f"Gate de revisión profesional forzado (FS9).")
        except Exception as _renal_e:
            logger.warning(f"[P3-CONDITION-RULES] cap renal de proteína falló: "
                           f"{type(_renal_e).__name__}: {_renal_e}")

    # [P3-CONDITION-RULES · 2026-06-14] Si el cap renal se aplicó EN LA FUENTE (nutrition_target),
    # el bloque de arriba no-opeó (active_macros ya venían capeados) y `result["renal_protein_cap"]`
    # quedó sin setear. Propagamos la meta de la fuente aquí → habilita el enforcement per-comida
    # (abajo) y el gate FS9. Caso normal en prod (el cap vive en la fuente desde el fix raíz).
    if (CONDITION_RULES_ENABLED and isinstance(nutrition, dict)
            and (nutrition.get("renal_protein_cap") or {}).get("applied")
            and not (result.get("renal_protein_cap") or {}).get("applied")):
        result["renal_protein_cap"] = {**nutrition["renal_protein_cap"], "meals_enforced": False}

    # =========================================================
    # OPTIMIZACIÓN DETERMINISTA POST-ASSEMBLY (GAP 4)
    # =========================================================
    days = result.get("days", [])
    
    # [P0-MEAL-MACRO-RECOVERY · 2026-06-13] Ratios del split objetivo del plan
    # para recuperar el breakdown de macros de meals que el day-gen (gap del LLM)
    # o un self-critique fallido (cb_open/timeout/None → `_critique_unresolved` +
    # surgical-regen que falla) dejó SIN protein/carbs/fats → shippeaban con
    # protein=0 + placeholder "Plan Matemático" y el usuario veía 0g de proteína
    # ese día. Estimar desde las cals del meal con el split del plan es
    # determinístico y muy superior a 0 (caso transitorio, no el path feliz).
    _daily_cals = _meal_macro_num(result.get("calories"))
    _pg = _meal_macro_num(active_macros.get("protein_str"))
    _cg = _meal_macro_num(active_macros.get("carbs_str"))
    _fg = _meal_macro_num(active_macros.get("fats_str"))
    if _daily_cals > 0 and (_pg or _cg or _fg):
        _ratio_p = (_pg * 4) / _daily_cals
        _ratio_c = (_cg * 4) / _daily_cals
        _ratio_f = (_fg * 9) / _daily_cals
    else:
        _ratio_p, _ratio_c, _ratio_f = 0.30, 0.45, 0.25  # split estándar fallback

    # Normalizar claves para compatibilidad con cachés antiguos o fallbacks
    for d in days:
        for m in d.get("meals", []):
            if "calories" in m and "cals" not in m: m["cals"] = m.pop("calories")
            if "description" in m and "desc" not in m: m["desc"] = m.pop("description")
            if "instructions" in m and "recipe" not in m: m["recipe"] = m.pop("instructions")
            
            # Auto-fill missing required keys to pass Pydantic Validation & Frontend
            if "meal" not in m: m["meal"] = m.get("name", "Comida").split(" ")[0] if " " in m.get("name", "") else m.get("name", "Comida")
            if not m.get("time"): m["time"] = "Flexible"  # [Z3] Optional emite None → guard .get()
            if "prep_time" not in m: m["prep_time"] = "15 min"
            # [P0-MEAL-MACRO-RECOVERY · 2026-06-13] Recupera el breakdown de macros
            # del meal (estima desde cals + split si vienen en 0) — antes shippeaba
            # protein=0 + placeholder "Plan Matemático" y el usuario veía 0g.
            _recover_meal_macros(m, _ratio_p, _ratio_c, _ratio_f)
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
            # [P2-RECIPE-DISCLAIMER-LIST · 2026-05-30] Append el disclaimer como
            # NUEVO elemento de la lista `recipe`, NO stringificar. Pre-fix:
            # `" ".join(recipe_text) + disclaimer` convertía `recipe` de
            # List[str] a `str` → (a) `PlanModel(**result)` rechazaba el tipo
            # (`list_type`) marcando `_schema_invalid=True` → `review_plan_node`
            # escalaba a crítico y el plan real del LLM se descartaba por el
            # `_get_extreme_fallback_plan` genérico; (b) si llegaba al cliente,
            # `Recipes.jsx` (`recipe.map(...)`) crasheaba (caught por
            # GlobalErrorBoundary). Mantener List[str] preserva los 3 pilares
            # (Mise en place / Fuego / Montaje) y el contrato del schema.
            # Tooltip-anchor: P2-RECIPE-DISCLAIMER-LIST.
            recipe_list = largest_meal.get("recipe", [])
            if isinstance(recipe_list, str):
                recipe_list = [recipe_list] if recipe_list.strip() else []
            elif not isinstance(recipe_list, list):
                recipe_list = []
            disclaimer = f"⚠️ Nota del Nutricionista AI: Las cantidades de los ingredientes fueron ajustadas matemáticamente para corregir un desvío de {abs(adjustment)} kcal."
            if not any("Nota del Nutricionista AI" in str(s) for s in recipe_list):
                largest_meal["recipe"] = recipe_list + [disclaimer]
                
            logger.info(f"⚖️ [MACRO BALANCING] Día {day.get('day')}: Desviación {diff}kcal -> Ajustado {adjustment}kcal en '{largest_meal.get('meal')}'")

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
                    # [P2-RECIPE-DISCLAIMER-LIST · 2026-05-30] Append como
                    # elemento de lista, NO stringificar (ver nota en la sección
                    # de macro balancing arriba — mismo modo de fallo schema/crash).
                    recipe_list = meal.get("recipe", [])
                    if isinstance(recipe_list, str):
                        recipe_list = [recipe_list] if recipe_list.strip() else []
                    elif not isinstance(recipe_list, list):
                        recipe_list = []
                    disclaimer = f"⚠️ Nota del Nutricionista AI: Las porciones fueron escaladas un {abs(1 - scale_factor)*100:.0f}% matemáticamente para cumplir tu meta estricta."
                    if not any("Nota del Nutricionista AI" in str(s) for s in recipe_list):
                        meal["recipe"] = recipe_list + [disclaimer]
                        
            new_total = sum(m.get("cals", 0) for m in day_meals)
            logger.info(f"🔒 [STRICT NUTRITION] Día {day.get('day')}: Redistribución forzada ({deviation_pct*100:.0f}% desviación, {day_cals}→{new_total} kcal)")

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
                        logger.info(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy bajo ({current_cals}kcal). Transferidos {transfer}kcal desde '{largest_meal.get('meal')}'")

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
                    logger.info(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy alto ({current_cals}kcal). Transferidos {excess}kcal hacia '{smallest_meal.get('meal')}'")

                    scale_down = meal["cals"] / max(current_cals, 1)
                    meal["protein"] = max(0, int(meal.get("protein", 0) * scale_down))
                    meal["carbs"] = max(0, int(meal.get("carbs", 0) * scale_down))
                    meal["fats"] = max(0, int(meal.get("fats", 0) * scale_down))
                    
                    scale_up = smallest_meal["cals"] / max(smallest_cals_before, 1)
                    # P1-C: clamps defensivos simétricos al ramal anterior.
                    smallest_meal["protein"] = max(0, int(smallest_meal.get("protein", 0) * scale_up))
                    smallest_meal["carbs"] = max(0, int(smallest_meal.get("carbs", 0) * scale_up))
                    smallest_meal["fats"] = max(0, int(smallest_meal.get("fats", 0) * scale_up))


    # [P1-30] Final macro coherence pass: recomputar `meal["cals"]` desde
    # canonical `4*P + 4*C + 9*F` cuando deriva del valor displayado tras
    # las 3 fases de balancing previas.
    #
    # Por qué: las secciones 1, 1.5 y 1.6 modifican `cals` y los macros
    # individuales con operaciones que NO preservan la invariante
    # `cals == 4*P + 4*C + 9*F`:
    #   - Sección 1 ajusta `cals += adjustment` (full kcal) pero solo
    #     splitea el `adjustment` en carbs (60%) + fats (40%) — protein
    #     queda intacto. Más: el `int()` en `carb_delta = int(adj/4)` y
    #     `fat_delta = int(adj/9)` introduce drift por truncación. Bajo
    #     condiciones de borde (carbs/fats bajos), `max(0, ...)` clampa el
    #     decremento real a menos del esperado, dejando `cals` aún más
    #     desincrónico.
    #   - Sección 1.5 escala uniformemente cals + macros por scale_factor
    #     pero usa `int()` por field, introduciendo drift incremental.
    #   - Sección 1.6 transfiere kcal entre meals con scale_up/down per-
    #     macro, también con `int()` rounding.
    #
    # Resultado pre-fix: la UI mostraba "Almuerzo: 500 kcal | 40g P / 35g
    # C / 20g F" — pero 4*40 + 4*35 + 9*20 = 480, no 500. Una desviación
    # de N kcal repetida 3-4 veces/día acumula 60-80 kcal/día → ~600
    # kcal/semana de desfase entre macros y kcal mostrados, alimentando
    # quejas de usuarios sobre adherencia y disparando flags "macros no
    # coinciden con cals" en revisor médico.
    #
    # Fix: tras los 3 balancing phases, recomputar cals desde macros
    # SI la deriva supera 5% (tolerancia para variabilidad genuina del
    # LLM — agua de cocción, mezclas, etc.). Si está dentro del 5%, el
    # valor original del LLM se preserva (puede capturar nutrientes
    # sutiles fuera del 4-4-9). Si excede, sustituimos por el valor
    # canónico — los macros mostrados son la fuente de verdad.
    if _balancing_safe:
        _p130_recomputed = 0
        for day in days:
            for meal in day.get("meals", []):
                try:
                    p = max(0, int(meal.get("protein", 0) or 0))
                    c = max(0, int(meal.get("carbs", 0) or 0))
                    f = max(0, int(meal.get("fats", 0) or 0))
                except (TypeError, ValueError):
                    # Macros corruptos (strings no numéricos, etc.):
                    # saltar, no podemos derivar coherencia.
                    continue
                macro_kcal = 4 * p + 4 * c + 9 * f
                if macro_kcal == 0:
                    # Sin señal de macros (todos 0): conservar cals
                    # original del LLM, no podemos sobre-restar a 0.
                    continue
                current_cals = meal.get("cals", 0)
                if not isinstance(current_cals, (int, float)):
                    # cals corrupto: forzar al canonical.
                    meal["cals"] = macro_kcal
                    _p130_recomputed += 1
                    continue
                # Tolerancia 5%: drift natural del LLM (agua cocción,
                # mezclas, redondeo de ingredientes) NO requiere ajuste.
                deviation = abs(macro_kcal - current_cals)
                if current_cals > 0 and deviation / current_cals > 0.05:
                    meal["cals"] = macro_kcal
                    _p130_recomputed += 1
        if _p130_recomputed > 0:
            logger.info(
                f"⚖️ [P1-30] Macro coherence: {_p130_recomputed} meal(s) "
                f"recomputaron cals desde 4P+4C+9F (deriva > 5% post-balancing)."
            )

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
        logger.info(f"📦 [CONSOLIDATION] Parseando {len(all_raw_ingredients)} ingredientes únicos (Regex Fast-Path)...")
        parse_start = time.time()
        
        import re
        # [P6-FRACTION-MIXED-FIX] Fracción mixta pegada al entero ("6¼") debe
        # convertirse a "6.25", NO a "60.25". El loop literal `replace('¼','0.25')`
        # concatenaba string-wise → qty inflada 10× en consolidación →
        # rechazo médico falso por "120 lonjas pavo / 160 fresas" (PDF 18:48).
        # [P6-FRACTION-MIXED-FIX-2] Tolerar whitespace opcional entre dígito y
        # fracción ("1 ¼" → "1.25"). PDF 20:14 ([825c94ef]) mostró:
        #   '1 ¼ lonjas de queso' → '1 0.25 lonjas de queso' (BUG)
        # → aggregator interpretó "0.25" como NOMBRE del ingrediente
        # → output incluyó '0.25 lonjas de queso' como item huérfano.
        # Pre-fix regex `(\d)([½⅓⅔¼¾])` requería pegado; ahora `\s*` tolera
        # uno o más espacios. Ambas formas (pegada y separada) son comunes:
        # LLM emite "1¼" cuando concatena, "1 ¼" cuando formatea.
        _MIXED_FRAC = {'½': 0.5, '⅓': 1/3, '⅔': 2/3, '¼': 0.25, '¾': 0.75}
        for raw in all_raw_ingredients:
            raw_lower = raw.lower().strip()
            raw_lower = re.sub(
                r'(\d)\s*([½⅓⅔¼¾])',
                lambda m: f"{int(m.group(1)) + _MIXED_FRAC[m.group(2)]:.4g}",
                raw_lower,
            )
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
                    logger.info(f"📦 [CONSOLIDATION] Unificado '{old_ing}' -> '{new_ing}'")
    # =========================================================

    # [P3-MACRO-SOLVER · 2026-06-13] Cerebro dividido — lado determinista (gated por
    # knob MEALFIT_MACRO_SOLVER_ENABLED, default False). Tras el balancing legacy
    # (que escala los NÚMEROS de macros sin tocar los ingredientes y tapa la
    # inconsistencia con un disclaimer) y la consolidación, re-escala las PORCIONES
    # reales de cada meal para clavar su target de macros (= macro diario × cal_share)
    # usando los macros reales de master_ingredients. Corre ANTES de la agregación de
    # la lista de compras Y de la humanización → los gramos re-escritos fluyen a
    # recipe + shopping + coherence guard CONSISTENTES (si corriera después, la lista
    # de compras quedaría con cantidades pre-solver → divergencias receta↔lista).
    # Fail-safe TOTAL: cualquier error deja el plan como lo dejó el balancing legacy.
    # Anchor: P3-MACRO-SOLVER.
    if MACRO_SOLVER_ENABLED and _daily_cals > 0 and (_pg or _cg or _fg):
        try:
            from nutrition_db import IngredientNutritionDB
            _nut_db = IngredientNutritionDB()
            _skel_days = (skeleton or {}).get("days", []) if isinstance(skeleton, dict) else []
            _solver_n = 0
            _topup_g = 0
            # [P3-PROTEIN-FLOOR] Proteínas de alta densidad allergen-safe para el closer
            # (cierra el déficit que el escalado no puede). Se computan una vez por plan.
            # min_protein=9 incluye yogur (blend-friendly para batidos) + el dish-fit del
            # closer prefiere carne (≥18) para principales y lácteo/yogur para licuados/ligeras.
            _hd_candidates = (_safe_high_density_proteins(form_data.get("allergies"), _nut_db, min_protein=9.0)
                              if PROTEIN_FLOOR_ENABLED else [])
            # [P3-CLOSER-EGG-BUDGET · 2026-06-14] Presupuesto de huevo del closer: una vez que el huevo
            # aparece en > cap comidas (mismo cap que VARIETY_HARD_GATE), pasa candidatos SIN huevo →
            # diversifica con yogur/queso/whey en vez de empujar el huevo sobre el cap. Cuenta el huevo
            # del LLM + el que el closer añade. `_c[1]` es el nombre (tupla (leanness, name, info)).
            try:
                from constants import strip_accents as _sa_egg
            except Exception:
                _sa_egg = lambda _s: _s  # noqa: E731
            _egg_total_meals = sum(len(_dd.get("meals") or []) for _dd in days)
            _egg_cap = max(3, round(_egg_total_meals * 0.25))
            _egg_count = (sum(1 for _dd in days for _mm in (_dd.get("meals") or [])
                              if _meal_has_egg(_mm, _sa_egg)) if CLOSER_EGG_BUDGET_ENABLED else 0)
            _hd_candidates_no_egg = [_c for _c in _hd_candidates
                                     if not any(_t in _sa_egg(str(_c[1]).lower())
                                                for _t in ("huevo", "clara", "yema"))]
            for _di, _d in enumerate(days):
                _ms = _d.get("meals", []) or []
                _day_c = sum(_meal_macro_num(_mm.get("cals")) for _mm in _ms)
                # Pool de proteínas APROBADO del día (allergen-safe) para el top-up.
                _day_num = _d.get("day", _di + 1)
                _sk = next((s for s in _skel_days if s.get("day") == _day_num), {})
                _approved = _sk.get("protein_pool", []) if isinstance(_sk, dict) else []
                # [P3-SLOT-DISTRIBUTION] Fracción por slot: split fisiológico canónico
                # (redistribuye kcal+proteína equitativamente) o cal_share del LLM (legacy).
                _slot_fracs = _canonical_slot_fractions(_ms) if SLOT_DISTRIBUTION_ENABLED else None
                for _mi, _m in enumerate(_ms):
                    if _slot_fracs:
                        _share = _slot_fracs[_mi]
                    else:
                        _share = (_meal_macro_num(_m.get("cals")) / _day_c) if _day_c > 0 \
                            else (1.0 / max(1, len(_ms)))
                    _slot_target = {
                        "kcal": _daily_cals * _share,
                        "protein": _pg * _share,
                        "carbs": _cg * _share,
                        "fats": _fg * _share,
                    }
                    if _apply_macro_solver_to_meal(_m, _slot_target, _nut_db):
                        _solver_n += 1
                    # [P3-PROTEIN-FLOOR] Cierre del déficit: rellena al TARGET de proteína del
                    # slot (no solo a un piso) con proteína de alta densidad allergen-safe
                    # integrada como ingrediente real. Las kcal extra las nivela el reconcile
                    # protein-preserving aguas abajo. Fallback al top-up legacy si está off.
                    if PROTEIN_FLOOR_ENABLED and _hd_candidates:
                        # [P3-CLOSER-EGG-BUDGET] sobre el cap → candidatos sin huevo (yogur/queso/whey).
                        _egg_cands = (_hd_candidates_no_egg
                                      if (CLOSER_EGG_BUDGET_ENABLED and _egg_count >= _egg_cap)
                                      else _hd_candidates)
                        _had_egg_pre = _meal_has_egg(_m, _sa_egg) if CLOSER_EGG_BUDGET_ENABLED else True
                        _topup_g += _close_protein_gap_for_meal(
                            _m, _slot_target["protein"], _nut_db, _egg_cands,
                            fill_pct=PROTEIN_FLOOR_FILL_PCT)
                        if CLOSER_EGG_BUDGET_ENABLED and not _had_egg_pre and _meal_has_egg(_m, _sa_egg):
                            _egg_count += 1  # el closer añadió huevo a esta comida → consume presupuesto
                    elif MACRO_SOLVER_PROTEIN_TOPUP:
                        _topup_g += _protein_topup_meal(
                            _m, _slot_target["kcal"], _nut_db, _approved)
                # [P3-PROTEIN-CEILING-GOAL-AWARE] Techo simétrico GOAL-AWARE: trima la proteína
                # del día si excede el techo en g/kg del objetivo (2.2 volumen/mant; 2.6 déficit
                # — proteína alta protege músculo al perder grasa). Usa el techo ABSOLUTO en g/kg
                # (no un % fijo del target) → ningún día de volumen pasa de 2.2 g/kg, y los de
                # déficit pueden subir hasta 2.6 g/kg de forma clínicamente correcta.
                if PROTEIN_FLOOR_ENABLED and _pg > 0:
                    _trim_day_protein_to_ceiling(
                        _ms, _pg, _nut_db,
                        ceiling_pct=_goal_aware_trim_ceiling_pct(form_data, _pg))
            logger.info(f"🧮 [P3-MACRO-SOLVER] Re-escaló porciones de {_solver_n} meals "
                        f"a su target de macros real (cerebro dividido)"
                        + (f" + top-up de proteína {_topup_g}g total" if _topup_g else ""))
        except Exception as _solver_e:
            logger.warning(f"[P3-MACRO-SOLVER] bloque deshabilitado por error: "
                           f"{type(_solver_e).__name__}: {_solver_e}")

    # [P3-CAL-RECONCILE · 2026-06-13] Paso FINAL del cerebro dividido: nivelar las
    # calorías de cada día EXACTAMENTE al target. Tras el solver (macros) + top-up
    # (proteína), las kcal/día derivan ±. El holistic `cal_score = max(0, 1 − desv×5)`
    # → 1.0 solo con desviación ~0. Escala uniforme por-día (porciones + macros por el
    # mismo factor) → nivela kcal SIN romper la consistencia receta↔macro. Corre ANTES
    # del shopping → los gramos finales fluyen a la lista. Fail-safe total.
    if MACRO_SOLVER_ENABLED and MACRO_SOLVER_CAL_RECONCILE and _daily_cals > 0:
        try:
            from nutrition_db import rescale_ingredient_string as _rescale, IngredientNutritionDB as _RCDB
            _rec_n = 0
            if PROTEIN_FLOOR_ENABLED:
                # [P3-PROTEIN-FLOOR] Reconcile PROTEIN-PRESERVING: la proteína (que el closer
                # acaba de llevar al target) queda FIJA; solo se escalan carbos+grasas para
                # nivelar las kcal. Reemplaza el escalado uniforme (que reducía la proteína).
                _rcdb = _RCDB()
                for _d in days:
                    if _protein_preserving_day_reconcile(_d.get("meals", []) or [], _daily_cals, _rcdb):
                        _rec_n += 1
                if _rec_n:
                    logger.info(f"🎯 [P3-PROTEIN-FLOOR] Reconcile protein-preserving niveló "
                                f"{_rec_n} día(s) al target ({_daily_cals:.0f} kcal) sin tocar la proteína")
            else:
                for _d in days:
                    _ms = _d.get("meals", []) or []
                    _dc = sum(_meal_macro_num(_mm.get("cals")) for _mm in _ms)
                    if _dc <= 0:
                        continue
                    _f = max(0.6, min(1.6, _daily_cals / _dc))  # clamp anti-porciones-absurdas
                    if abs(_f - 1.0) < 0.02:
                        continue  # ya dentro del 2% → no tocar
                    for _m in _ms:
                        _ings = _m.get("ingredients")
                        if isinstance(_ings, list):
                            _m["ingredients"] = [_rescale(str(s), _f) for s in _ings]
                        _raw = _m.get("ingredients_raw")
                        if isinstance(_raw, list):
                            _m["ingredients_raw"] = [_rescale(str(s), _f) for s in _raw]
                        _m["protein"] = round(_meal_macro_num(_m.get("protein")) * _f)
                        _m["carbs"] = round(_meal_macro_num(_m.get("carbs")) * _f)
                        _m["fats"] = round(_meal_macro_num(_m.get("fats")) * _f)
                        _m["cals"] = round(_meal_macro_num(_m.get("cals")) * _f)
                        _m["macros"] = [f"P:{_m['protein']}g", f"C:{_m['carbs']}g", f"G:{_m['fats']}g"]
                    _rec_n += 1
                if _rec_n:
                    logger.info(f"🎯 [P3-CAL-RECONCILE] Niveló calorías de {_rec_n} día(s) al "
                                f"target exacto ({_daily_cals:.0f} kcal) — cierra cal_score del holistic")
        except Exception as _rec_e:
            logger.warning(f"[P3-CAL-RECONCILE] deshabilitado por error: "
                           f"{type(_rec_e).__name__}: {_rec_e}")

    # [P3-FALLBACK-CLINICAL-LAYER · 2026-06-14 · Fase B] Capa clínica determinista (FS1-FS9) — SSOT
    # ÚNICA, ahora compartida con el path de fallback. El cap renal de la fuente (arriba) ya seteó
    # result["renal_protein_cap"] + active_macros; la función re-deriva idempotentemente (no-op aquí en
    # el happy path) y corre los 8 guards en orden: enforcement renal per-comida → food-safety →
    # sustitución por condición → quantize → micros/suplementos → variedad → proveniencia → gate FS9.
    # Antes este bloque vivía inline DUPLICADO; ahora assemble y el fallback consumen la MISMA función
    # (cero drift posible). El marker `_clinical_layer_applied` lo deja idempotente cross-path.
    _apply_deterministic_clinical_layer(result, form_data, nutrition)

    # Calcular shopping lists
    # Solo usar user_id real (autenticado); session_id no tiene inventory en DB
    _uid = form_data.get("user_id")
    if not _uid or _uid == "guest": _uid = None

    from shopping_calculator import get_shopping_list_delta, fetch_inventory_and_consumed_for_plan
    from constants import compute_household_multiplier
    try:
        # [P1-3] householdComposition (adults/children) — fallback a householdSize legacy.
        household = compute_household_multiplier(form_data)
        if household > 1.0:
            _hc = form_data.get("householdComposition") or {}
            _label = (
                f"{_hc.get('adults', 0)}A+{_hc.get('children', 0)}N (×{household:.2f})"
                if isinstance(_hc, dict) and (_hc.get("adults") or _hc.get("children"))
                else f"×{household:.2f}"
            )
            logger.info(f"👨‍👩‍👧‍👦 [HOUSEHOLD] Escalando lista de compras {_label}")

        # P0-NEW-1.b: paralelizar las 3 multiplicidades en `_DB_EXECUTOR`. Antes
        # corrían secuenciales bloqueando el event loop ~150-600ms total
        # (cada `get_shopping_list_delta` hace queries a inventory + pricing).
        # Ahora: 3 calls en paralelo → latencia ≈ max(latencias) y el loop libre
        # para SSE callbacks y otras coroutines del worker.
        #
        # [P1-5] Fetch inventario + consumidos UNA vez antes de las 3 calls.
        # ANTES, cada `get_shopping_list_delta` re-consultaba `user_inventory`
        # y `consumed_meals` independientemente — si entre llamadas un
        # cron, restock o Realtime channel mutaba el inventario, las 3 listas
        # weekly/biweekly/monthly quedaban inconsistentes (basadas en
        # snapshots distintos). El usuario veía cantidades dispares al
        # cambiar `groceryDuration` para el mismo plan. Ahora pasamos el
        # snapshot atómico vía `inventory_override` + `consumed_override`.
        if _uid:
            inv_snapshot, consumed_snapshot = await _adb(
                fetch_inventory_and_consumed_for_plan, _uid, result, True
            )
            aggr_list_7, aggr_list_15, aggr_list_30 = await asyncio.gather(
                _adb(get_shopping_list_delta, _uid, result, True, False, True, 1.0 * household,
                     inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
                _adb(get_shopping_list_delta, _uid, result, True, False, True, 2.0 * household,
                     inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
                _adb(get_shopping_list_delta, _uid, result, True, False, True, 4.0 * household,
                     inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
            )
        else:
            aggr_list_7, aggr_list_15, aggr_list_30 = [], [], []

        grocery_duration = form_data.get("groceryDuration", "weekly")

        # [VISIÓN-C / HYBRID-SHOPPING-LIST] Para periodos > 1 semana,
        # construir versión híbrida: staples (despensa, granos, conservas, etc.)
        # se compran al periodo completo; perishables (carnes frescas, frutas,
        # lácteos perecederos, vegetales) se compran semanalmente. Resuelve
        # la sobreestimación de perecederos en planes mensuales que asumen
        # menú repetido, mientras que en realidad chunks 2-N traen variedad.
        try:
            from shopping_calculator import _build_hybrid_shopping_list as _build_hybrid
            aggr_list_15_hybrid = _build_hybrid(aggr_list_7, aggr_list_15) if aggr_list_15 else aggr_list_15
            aggr_list_30_hybrid = _build_hybrid(aggr_list_7, aggr_list_30) if aggr_list_30 else aggr_list_30
        except Exception as e_hybrid:
            logger.warning(f"⚠️ [HYBRID-SHOPPING-LIST] Error construyendo híbrida: {e_hybrid}")
            aggr_list_15_hybrid = aggr_list_15
            aggr_list_30_hybrid = aggr_list_30

        if grocery_duration == "biweekly":
            aggr_list = aggr_list_15_hybrid
        elif grocery_duration == "monthly":
            aggr_list = aggr_list_30_hybrid
        else:
            aggr_list = aggr_list_7

        result["aggregated_shopping_list"] = aggr_list
        result["aggregated_shopping_list_weekly"] = aggr_list_7
        result["aggregated_shopping_list_biweekly"] = aggr_list_15_hybrid
        result["aggregated_shopping_list_monthly"] = aggr_list_30_hybrid
        # [P1-C 2026-05-07] Persistir el multiplier para que el guard de coherencia
        # (capa magnitudes) pueda escalar `expected_sum_from_recipes` simétricamente
        # al aggregated. Espejo de la convención cron_tasks/routers/plans.
        result["calc_household_multiplier"] = float(household)
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15] Mantenemos `WARNING` (no
        # `ERROR`) porque el bloque siguiente degrada graceful con listas
        # vacías — el plan continúa. `exc_info=True` adjunta el stack al
        # mismo record sin elevarlo a ERROR.
        logger.warning(f"⚠️ [SHOPPING MATH] Error agregando lista delta: {e}", exc_info=True)
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
        logger.warning(f"⚠️ [HUMANIZE] Error al humanizar ingredientes: {e}")

    # Guardar técnicas seleccionadas para persistencia en DB
    result["_selected_techniques"] = skeleton.get("_selected_techniques", [])

    logger.info(f"✅ [ENSAMBLADOR] Plan final ensamblado")

    # [P0-6] Validaciones críticas post-assembly extraídas a un helper compartido.
    # Antes estas tres validaciones (Skeleton Fidelity, Recipe Coherence, Schema
    # Validation contra PlanModel) vivían inline aquí. Aunque el flujo SÍ las
    # ejecutaba para cache-hit, su entremezcla con código no relacionado hacía
    # que un futuro refactor pudiera olvidar aplicarlas o moverlas a una rama
    # exclusiva del path LLM. El helper centralizado documenta el contrato:
    # se ejecuta SIEMPRE para ambas ramas (cache-hit y LLM), garantizando que
    # un plan cacheado con shape vieja, incoherencia receta↔ingredientes, o
    # schema bumped sin invalidar entradas en vuelo, sea detectado y elevado
    # a critical por `review_plan_node`.
    affected_days_set = set(state.get("_affected_days") or [])
    _run_assembly_validations(result, skeleton, affected_days_set)

    # [P1-shop-coh-1 · 2026-05-07 / P1-C v2 · 2026-05-07] Guard recetas↔lista.
    # v1: presence/absence (cap_swallowed_modifier, fantasmas).
    # v2: magnitudes — escala expected por household multiplier y compara
    # ratios con `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` (default 10%).
    # Modo via env var MEALFIT_SHOPPING_COHERENCE_GUARD (off|warn|block, default warn).
    # El guard lee el multiplier de result["calc_household_multiplier"]; pasamos
    # también explícito como defensa contra dicts mutados antes del hook.
    # Si el guard explota por bug interno NO debe abortar el assembly.
    try:
        from shopping_calculator import run_shopping_coherence_guard
        coh_divergences = run_shopping_coherence_guard(
            result, multiplier=result.get("calc_household_multiplier")
        ) or []
        # [P3-NEW-C · 2026-05-08] Telemetría histórica: una entrada por
        # cada invocación del guard que detecte divergencias. Persiste en
        # plan_data → meal_plans para análisis post-mortem ("¿qué % de
        # planes tropezaron con coh-block antes de pasar review?, ¿qué
        # hipótesis dominaron?, ¿cuántos retries tomó resolver?"). Cap a 20
        # entries para evitar bloat. La acción tomada por review_plan_node
        # (reject_minor/reject_high/degrade) se rellena más tarde en el
        # campo `action_taken` de la última entry. Sin cambio de comportamiento.
        if coh_divergences:
            try:
                from datetime import datetime as _coh_dt, timezone as _coh_tz
                from collections import Counter as _coh_Counter
                # Preservar history a través de retries leyendo del plan_result
                # del attempt previo (LangGraph mantiene state entre nodos).
                prior_result = state.get("plan_result") or {}
                prior_history = prior_result.get("_shopping_coherence_block_history") or []
                if not isinstance(prior_history, list):
                    prior_history = []
                attempt_n = state.get("attempt", 1)
                try:
                    attempt_n = int(attempt_n)
                except (TypeError, ValueError):
                    attempt_n = 1
                hyp_counter = _coh_Counter(
                    str(d.get("hypothesis") or "unknown") for d in coh_divergences
                )
                _block_set = bool(result.get("_shopping_coherence_block"))
                entry = {
                    "ts": _coh_dt.now(_coh_tz.utc).isoformat(),
                    "attempt": attempt_n,
                    "divergence_count": len(coh_divergences),
                    "presence_count": sum(
                        1 for d in coh_divergences if not d.get("magnitude")
                    ),
                    "magnitude_count": sum(
                        1 for d in coh_divergences if d.get("magnitude")
                    ),
                    "hypotheses": dict(hyp_counter),
                    "block_set": _block_set,
                    # [P2-2 · 2026-05-08] Si `block_set=False`, NO entrará al
                    # branch en review_plan_node → `action_taken` queda como
                    # `not_applicable` para distinguir post-mortem el flujo
                    # normal (warn-only) de un bug de hidratación. Si block
                    # está set, se queda `None` y review_plan_node lo hidrata
                    # con el knob resuelto. La combinación (block_set=True,
                    # action_taken=None) es invariante reservado para errores.
                    "action_taken": None if _block_set else "not_applicable",
                }
                # [P2-HIST-AUDIT-4 · 2026-05-09] Cap aplicado vía
                # helper SSOT con knob `MEALFIT_COHERENCE_BLOCK_HISTORY_CAP`
                # (default 20). Antes era literal hardcoded en 2 sites
                # — ahora cualquier flujo nuevo que apenda al history
                # hereda la política sin coordinación.
                new_history = _apply_coherence_history_cap(
                    prior_history,
                    entry,
                    plan_id_hint=(result.get("id") or result.get("plan_id")),
                )
                result["_shopping_coherence_block_history"] = new_history
            except Exception as _coh_hist_e:
                logging.debug(
                    f"[COH-GUARD/HISTORY] no-op (telemetría): {_coh_hist_e}"
                )
    except Exception as _coh_e:
        logging.warning(f"[COH-GUARD] excepción inesperada (no aborta): {_coh_e}")

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

    [P6-AUTO-PATCH-1] Defensa-en-profundidad contra remover proteínas
    principales por substring match accidental. Si el ingrediente a
    remover contiene un synonym de proteína (cerdo, pollo, res, etc.)
    Y ese synonym aparece en la receta, NO removemos — la presencia del
    synonym en recipe satisface el bidirectional check, así que el
    "huérfano" reportado por el reverse-check fue un falso positivo
    (probablemente el primer fix P6 ya lo eliminó del error stream,
    pero esta defensa cierra el modo de fallo desde dos lados).
    """
    # Map de protein synonyms para defensa secundaria. Mantenido en sync
    # manualmente con el del recipe-coherence forward check (línea ~5478).
    _PROTEIN_KEYS_FOR_PATCH = (
        "cerdo", "pollo", "res", "carne", "pescado",
        "huevo", "huevos", "pavo", "camarón", "camarones",
    )
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
                # Pre-extraer recipe text para el guard de proteínas
                recipe_txt = meal.get("recipe", "")
                if isinstance(recipe_txt, list):
                    recipe_txt = " ".join(recipe_txt)
                recipe_lower = recipe_txt.lower()

                ings = meal.get("ingredients", [])
                new_ings = []
                for i in ings:
                    i_lower = i.lower() if isinstance(i, str) else str(i).lower()
                    if core_noun not in i_lower:
                        new_ings.append(i)
                        continue
                    # Match candidate. Aplicar defensa: si el ingrediente
                    # contiene un protein synonym Y la receta también lo
                    # menciona, preservar. Ejemplo: core_noun='lomo',
                    # ingrediente='lomo de cerdo', recipe contiene 'cerdo'
                    # → preservar.
                    protected = any(
                        p in i_lower and _re.search(
                            r'\b' + _re.escape(p) + r'[a-z]*\b', recipe_lower
                        )
                        for p in _PROTEIN_KEYS_FOR_PATCH
                    )
                    if protected:
                        new_ings.append(i)
                        continue
                    # No proteína presente o no en receta — proceder a
                    # eliminar (caso original del auto-patch).
                meal["ingredients"] = new_ings
                if len(new_ings) < len(ings):
                    patched += 1
                break
    return patched


# [P1-AUTO-PATCH-FORWARD-SYNC-TOOLTIP · 2026-05-21] Sinónimos seguros para
# substitución en `_auto_patch_recipe_forward_coherence`. Solo single-words o
# multi-word inequívocos por categoría — sin "filete"/"lomo"/"carne" sueltos
# que producirían false-replaces en dishes no-relacionados. Mantener en sync
# con el `protein_synonyms` LOCAL del forward check (línea ~6942 en
# `_run_assembly_and_validation`); si añades un pescado/corte nuevo allí Y
# es inequívoco, replícalo aquí. Si NO es inequívoco (sólo distingue por
# contexto), déjalo fuera de aquí para que el revisor fuerce retry en lugar
# de que el patch produzca texto incoherente.
_FORWARD_PATCH_SYNONYMS = {
    "pescado": [
        "pescado", "pescados",
        "chillo", "dorado", "mero", "salmón", "salmon", "tilapia",
        "bacalao", "atún", "atun", "sardina", "sardinas",
        "merluza", "róbalo", "robalo", "pargo", "corvina",
        "mahi-mahi", "mahi mahi", "mahimahi", "lubina",
        "carite", "jurel", "lambí", "lambi",
        "filete de pescado", "filete de mero", "filete de tilapia",
        "filete de chillo", "filete de dorado", "filete de bacalao",
        "filete de merluza", "filete de salmón", "filete de salmon",
    ],
    "pollo": [
        "pollo", "pechuga de pollo", "muslo de pollo",
        "pernil de pollo", "filete de pollo",
        "chicharrón de pollo", "chicharron de pollo",
    ],
    "res": [
        "res", "carne de res", "carne molida de res",
        "ropa vieja", "bistec", "bisteck",
    ],
    "cerdo": [
        "cerdo", "chuleta de cerdo", "pernil de cerdo",
        "tocino", "chicharrón", "chicharron",
    ],
    "huevo": [
        "huevo", "huevos", "clara", "claras", "yema", "yemas",
        "tortilla", "revoltillo",
    ],
    "pavo": [
        "pavo", "pechuga de pavo", "jamón de pavo", "jamon de pavo",
        "pavo molido", "carne de pavo", "pavo asado", "pavo desmenuzado",
    ],
    "camarón": [
        "camarón", "camarones", "camaron",
        "gambas", "langostinos",
    ],
    # Proteínas vegetales / lácteas — solo aparecen como TARGET de substitución
    # cuando el orphan es una proteína animal y los ingredientes son veggie.
    "garbanzos": ["garbanzo", "garbanzos"],
    "yogurt": ["yogurt", "yogur"],
    "ricotta": ["ricotta"],
    "almendras": ["almendra", "almendras"],
}


def _auto_patch_recipe_forward_coherence(plan: dict, errors: list) -> tuple[int, list]:
    """[P1-AUTO-PATCH-FORWARD · 2026-05-21] Resuelve la dirección 'forward' de
    la incoherencia receta↔ingrediente: la receta (o el nombre del plato)
    menciona una proteína (e.g. 'pescado') que NO tiene sinónimo en
    `meal.ingredients`. Antes este error escalaba a `structural_coherence_errors`
    → severity='minor' → retry del plan completo (~10 llamadas LLM adicionales).

    Estrategia quirúrgica (cero LLM):
      1. Parsear `Día N, MEAL_NAME: La receta indica 'KEY' pero no hay ningún
         ingrediente equivalente`.
      2. Identificar la proteína REAL en `meal.ingredients` (qué categoría de
         `_FORWARD_PATCH_SYNONYMS` matchea algún ingrediente, excluyendo el
         propio orphan).
      3. Reescribir `meal.name` y `meal.recipe` reemplazando el orphan KEY y
         sus sinónimos seguros con la categoría real (case-preserving).
      4. Si no podemos identificar una proteína real (ingredients = solo
         vegetales/cereales) preservar el error en `unpatched_errors` y dejar
         que el flujo legacy escale a retry — preferimos NO inventar.

    Retorna: `(count_patched, unpatched_errors)`.

    Asimetría histórica que esto cierra: `_auto_patch_ingredient_coherence`
    (P6-AUTO-PATCH-1) cubría ya la dirección reverse (ingrediente en lista
    pero no en receta). El forward quedó descubierto durante ~6 meses y forzó
    retries innecesarios cada vez que el LLM nombraba un plato con proteína
    no-listada (caso real chunk 713ff43a 2026-05-21 00:08).
    """
    if not errors:
        return 0, []

    unpatched = []
    patched_count = 0

    def _preserve_case_sub(match, target_word: str) -> str:
        """Reemplaza preservando el case del primer carácter del match."""
        orig = match.group(0)
        if orig and orig[0].isupper():
            return target_word[:1].upper() + target_word[1:]
        return target_word

    for error in errors:
        m = _re.search(
            r"D[ií]a (\d+), (.+?): La receta indica '(.+?)' pero no hay ning[úu]n ingrediente equivalente",
            error,
        )
        if not m:
            unpatched.append(error)
            continue
        day_num = int(m.group(1))
        meal_name = m.group(2).strip()
        orphan_key = m.group(3).strip().lower()

        if orphan_key not in _FORWARD_PATCH_SYNONYMS:
            # Orphan fuera del mapa de substitución seguro — no podemos
            # garantizar reescritura limpia. Escalar a retry.
            unpatched.append(error)
            continue

        # Localizar el meal exacto.
        target_meal = None
        for day in plan.get("days", []):
            if day.get("day") != day_num:
                continue
            for meal in day.get("meals", []):
                if meal.get("name", "").strip() == meal_name:
                    target_meal = meal
                    break
            if target_meal:
                break

        if target_meal is None:
            unpatched.append(error)
            continue

        ings_lower = [
            (i.lower() if isinstance(i, str) else str(i).lower())
            for i in target_meal.get("ingredients", [])
        ]

        # Buscar la proteína "actual" en ingredients (categoría != orphan_key).
        actual_protein = None
        for cat_key, cat_syns in _FORWARD_PATCH_SYNONYMS.items():
            if cat_key == orphan_key:
                continue
            for syn in cat_syns:
                pat = r'\b' + _re.escape(syn) + r'\b'
                if any(_re.search(pat, ing) for ing in ings_lower):
                    actual_protein = cat_key
                    break
            if actual_protein:
                break

        if not actual_protein:
            # Solo vegetales/cereales en ingredients → reescribir sería inventar.
            # Mejor que el retry intente proponer un plato vegetariano coherente.
            unpatched.append(error)
            continue

        # Reescribir: longest-first para que multi-word ("filete de chillo")
        # se substituya antes que single-word ("chillo") y no quede texto
        # residual desincronizado tras la primera pasada.
        syns_sorted = sorted(_FORWARD_PATCH_SYNONYMS[orphan_key], key=len, reverse=True)

        def _apply_subs(text: str) -> str:
            new = text
            for syn in syns_sorted:
                pat = r'\b' + _re.escape(syn) + r'\b'
                new = _re.sub(
                    pat,
                    lambda mm, _ap=actual_protein: _preserve_case_sub(mm, _ap),
                    new,
                    flags=_re.IGNORECASE,
                )
            return new

        orig_name = target_meal.get("name", "")
        if isinstance(orig_name, str) and orig_name:
            target_meal["name"] = _apply_subs(orig_name)

        recipe = target_meal.get("recipe", "")
        if isinstance(recipe, list):
            target_meal["recipe"] = [
                _apply_subs(step) if isinstance(step, str) else step for step in recipe
            ]
        elif isinstance(recipe, str) and recipe:
            target_meal["recipe"] = _apply_subs(recipe)

        # Marker para debug / evitar re-patch en doble pasada.
        prev = target_meal.get("_forward_patched_proteins") or []
        target_meal["_forward_patched_proteins"] = prev + [orphan_key]

        patched_count += 1

    return patched_count, unpatched


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


def _augment_affected_days_with_critique_markers(
    affected_days: list,
    previous_days: list,
) -> list:
    """[P1-SURGICAL-1] Augmenta `affected_days` con días del intento previo
    cuyo self-critique no logró corregir su problema.

    Por qué importa:
      `plan_skeleton_node` decide qué días RECICLAR vs REGENERAR en el retry
      basándose en `affected_days` (escrito por `review_plan_node` parsing
      "Día N" desde las razones de rechazo médico). El revisor médico usa un
      lente específico (alergias, sodio, repetición proteína, etc.) y puede
      pasar por alto problemas que el self-critique YA detectó pero NO logró
      corregir por timeout / CB-open / excepción / LLM-returned-None. Esos
      días tienen `_critique_unresolved` set por `_correct_single_day`.

      Sin esta augmentación, el surgical fix recicla esos días como "válidos"
      y el bug latente entra al intento N+1 — caso real del incidente
      2026-05-05 (Día 2 con timeout de corrección reciclado en attempt #2,
      review_plan rechazó con HIGH "Día 2: receta indica 'pollo' pero no hay
      ingrediente equivalente listado").

    Política: forzamos regen aunque medical no lo flagee. Costo: ~30-40s
    por día regenerado. Beneficio: el usuario no recibe planes con
    problemas conocidos sin resolver.

    Args:
        affected_days: lista actual de días a regenerar (de `state["_affected_days"]`).
        previous_days: lista de días del intento previo (de `state["plan_result"]["days"]`),
                       potencialmente con marker `_critique_unresolved` por día.

    Returns:
        Lista ordenada (orden ascendente) de día_num únicos. Si ninguno tiene
        marker, devuelve `affected_days` sin cambios (mismo objeto reordenado
        es OK — el caller no depende de identidad).
    """
    affected_set = set(affected_days or [])
    forced_by_marker = []
    for d in (previous_days or []):
        if not isinstance(d, dict):
            continue
        day_num = d.get("day")
        if not isinstance(day_num, int):
            continue
        marker = d.get("_critique_unresolved")
        if isinstance(marker, dict) and day_num not in affected_set:
            affected_set.add(day_num)
            forced_by_marker.append((day_num, marker.get("reason", "?")))

    if forced_by_marker:
        for d_n, reason in forced_by_marker:
            logger.info(
                f"📌 [P1-SURGICAL-1] Día {d_n} añadido a regen forzada por "
                f"warning previo no resuelto (reason={reason!r}); medical no lo "
                f"flageó pero el self-critique lo dejó sin corregir."
            )
        return sorted(affected_set)

    return list(affected_days or [])


def _collect_unresolved_marker_days(plan_result) -> list[int]:
    """[P5-MARKER-APPROVED-1] Devuelve los `day` numbers que tienen un
    marker `_critique_unresolved` pendiente.

    Helper compartido entre `should_retry` (decide ruta) y
    `surgical_marker_regen_node` (ejecuta la regen). Mantenerlo en un
    solo sitio evita drift en la heurística de detección.

    Solo incluye días con marker dict no-vacío. Markers con valor None,
    {}, o con shape inesperado se ignoran (defensa ante state corrupto)."""
    if not isinstance(plan_result, dict):
        return []
    days = plan_result.get("days") or []
    if not isinstance(days, list):
        return []
    out = []
    for d in days:
        if not isinstance(d, dict):
            continue
        marker = d.get("_critique_unresolved")
        if isinstance(marker, dict) and marker:
            day_num = d.get("day")
            if isinstance(day_num, int):
                out.append(day_num)
    return sorted(out)


@_node_label("surgical_marker")
async def surgical_marker_regen_node(state: PlanState) -> dict:
    """[P5-MARKER-APPROVED-1] Re-corrige días con `_critique_unresolved`
    después de que el reviewer médico aprobó.

    Caso real (corrida 2026-05-05 03:54):
      Self-critique detectó repetición almuerzo↔cena en Días 1, 2, 3.
      Día 1 corrigió OK (43s), Día 3 corrigió OK (90s), Día 2 timeoutó
      al cap individual de 150s (P4-TIMEOUT-3) → marker `_critique_unresolved`.
      Reviewer médico aprobó porque su lente (alergias, sodio, despensa,
      skeleton fidelity) es ortogonal a slot coherence.
      Plan llegó al usuario con Día 2 mostrando la repetición intacta.

    Este nodo cierra ese gap: cuando review_passed=True PERO hay markers,
    re-corremos el corrector LLM con budget fresco solo para esos días.
    El provider típicamente se ha recuperado del overload momentáneo
    (~5-10 min después). Si la re-corrección también falla, dejamos el
    marker en su lugar y aceptamos el plan parcial — mejor que loop
    infinito.

    Diseño:
      - Una sola pasada por gate (`_marker_regen_attempted` lo previene).
      - Siguiente nodo: `assemble_plan` (re-aggregate shopping list con
        los días corregidos), luego `review_plan` re-evalúa.
      - Si el segundo review aprueba → END.
      - Si rechaza (la regen introdujo otro issue, raro) → cae al retry
        path normal.
    """
    plan_result = state.get("plan_result") or {}
    if not isinstance(plan_result, dict):
        return {"_marker_regen_attempted": True}

    marker_day_nums = _collect_unresolved_marker_days(plan_result)
    if not marker_day_nums:
        # Defensa: should_retry ya filtra este caso pero el nodo es resiliente.
        return {"_marker_regen_attempted": True}

    days = list(plan_result.get("days") or [])
    form_data = state.get("form_data") or {}
    # [P1-ORCH-4 · 2026-05-28] Fallback a `state['plan_skeleton']` cuando
    # `plan_result` no trae `_skeleton`. Este nodo corre DESPUÉS de assemble,
    # que retorna un dict nuevo SIN `_skeleton` (sobrescribe state['plan_result']).
    # Sin este fallback, `skeleton` quedaba {} → (1) el prompt del corrector
    # perdía la asignación de proteína obligatoria del planner, y (2)
    # `_apply_protein_pool_scrub(corrected_day, {})` veía pool vacío y eliminaba
    # TODAS las proteínas de los ingredientes del día regenerado (incoherencia
    # receta↔ingredientes — justo lo que el guard combate). Mismo patrón de
    # fallback que assemble_plan_node. Tooltip-anchor: P1-ORCH-4.
    skeleton = plan_result.get("_skeleton") or state.get("plan_skeleton") or {}
    skeleton_days = skeleton.get("days", []) if isinstance(skeleton, dict) else []

    logger.info(
        f"🩹 [P5-MARKER-REGEN] Re-corrigiendo {len(marker_day_nums)} día(s) "
        f"con `_critique_unresolved` tras approved: {marker_day_nums}"
    )
    start_time = time.time()

    # Setup del corrector — paridad con self_critique_node line ~5124.
    _corrector_model = _route_model(form_data, force_fast=True)
    _corrector_cb = _get_circuit_breaker(_corrector_model)
    corrector_llm = ChatDeepSeek(
        model=_corrector_model,
        temperature=0.3,
        max_retries=0,
        timeout=80,
    ).with_structured_output(SingleDayPlanModel)

    ctx = _build_shared_context(state)

    async def _re_correct_one(day_num: int):
        target_day = next((d for d in days if d.get("day") == day_num), None)
        if not target_day:
            return day_num, None
        marker = target_day.get("_critique_unresolved") or {}
        original_issue = marker.get("issue") or ""

        if not await _corrector_cb.acan_proceed():
            logger.warning(
                f"⚠️ [P5-MARKER-REGEN] CB OPEN ({_corrector_model}). "
                f"Saltando Día {day_num}, marker preservado."
            )
            return day_num, None

        skeleton_day = next(
            (d for d in skeleton_days if d.get("day") == day_num), {}
        )
        skeleton_block = ""
        if skeleton_day:
            skeleton_block = (
                f"\n⚠️ ASIGNACIÓN OBLIGATORIA DEL PLANIFICADOR (no la ignores):\n"
                f"{build_day_assignment_context(skeleton_day, day_num)}"
            )

        # [P5-PROMPT-D] Mismo prompt mínimo que self_critique correction.
        # [P6-CRITIQUE-VS-SKELETON] Misma regla de precedencia inviolable
        # que el corrector — el surgical regen también puede recibir
        # "issue" del critique original que pidió cambiar protein asignada.
        correction_prompt = f"""Eres un nutricionista chef. Corrige SOLO el Día {day_num} del plan alimenticio.

PROBLEMA DETECTADO (sin resolver en pasada anterior): {original_issue}

RESTRICCIONES NUTRICIONALES (respétalas siempre):
{ctx['nutrition_context_minimal']}
{skeleton_block}
REGLA DE PRECEDENCIA INVIOLABLE (si hay conflicto, gana esta):
- La ASIGNACIÓN DEL PLANIFICADOR es HARD CONSTRAINT — NUNCA la violes aunque el problema previo pida cambiar una proteína/carbohidrato asignado.
- Si el problema sugiere cambiar proteína de almuerzo o cena para resolver slot coherence, pero esa proteína FUE ASIGNADA por el planificador, MANTÉN la proteína y resuelve por OTRO medio: cambiar carbohidrato, técnica, vegetal o presentación.

REGLA BIDIRECCIONAL CRÍTICA:
- Todo ingrediente en `ingredients` DEBE aparecer nombrado en la receta (Mise en place, El Toque de Fuego o Montaje).
- Todo alimento nombrado en la receta DEBE estar en `ingredients`.
- Si un ingrediente no se usa en la receta, elimínalo de `ingredients`.

Día {day_num} actual (JSON):
{json.dumps({k: v for k, v in target_day.items() if k != "_critique_unresolved"}, ensure_ascii=False)}

Devuelve el Día {day_num} corregido con EXACTAMENTE la misma estructura JSON y los mismos targets calóricos."""

        try:
            logger.info(f"🩹 [P5-MARKER-REGEN] Re-corrigiendo Día {day_num}...")
            corrected_result: SingleDayPlanModel = await _safe_ainvoke(
                corrector_llm, correction_prompt, timeout=CRITIQUE_FIX_TIMEOUT_S
            )
            await _corrector_cb.arecord_success()
            if corrected_result:
                corrected_day = corrected_result.model_dump()
                corrected_day["day"] = day_num
                # IMPORTANTE: NO copiar `_critique_unresolved` — la corrección
                # es el evento que limpia el marker.
                logger.info(f"✅ [P5-MARKER-REGEN] Día {day_num} re-corregido exitosamente.")
                # [PROTEIN-POOL-SCRUB 2026-05-07] Aplicar cleanup + scan tras
                # surgical regen — caso plan 089e541c: surgical metió "Pechuga
                # de pollo" aunque pool era [Queso Blanco, Gandules, Atún].
                # El cleanup remueve pollo de ingredients; el scan loguea si
                # quedó en recipe text para visibilidad downstream.
                try:
                    _apply_protein_pool_scrub(
                        corrected_day, skeleton_day, day_num,
                        context_label="SURGICAL-REGEN",
                    )
                except Exception as _scrub_err:
                    logger.warning(
                        f"⚠️ [SURGICAL-REGEN/DÍA {day_num}] protein-scrub falló "
                        f"(best-effort): {_scrub_err}"
                    )
                return day_num, corrected_day
            logger.warning(
                f"⚠️ [P5-MARKER-REGEN] Día {day_num}: corrector LLM retornó None. "
                f"Marker preservado, plan sigue al usuario sin la corrección."
            )
        except asyncio.TimeoutError:
            # [P6-TIMEOUT-DIAG] Mismo logging diagnóstico que en self_critique_node
            # — captura el patrón "Día N timeoutea dos veces" (corrida 19:58 [c6eaf808]
            # mostró Día 2 fallando en self-critique Y en marker-regen).
            _prompt_chars = len(correction_prompt)
            _target_chars = len(json.dumps(target_day, ensure_ascii=False)) if target_day else 0
            _ingredients_count = sum(
                len(m.get("ingredients", [])) for m in (target_day or {}).get("meals", [])
            )
            logger.info(
                f"⏱️ [P5-MARKER-REGEN] Día {day_num} timeout otra vez "
                f"({CRITIQUE_FIX_TIMEOUT_S:.0f}s) con {_corrector_model}."
            )
            logger.info(
                f"📐 [P6-TIMEOUT-DIAG] Día {day_num} (regen) sizes: "
                f"prompt={_prompt_chars}c, target_day={_target_chars}c, "
                f"ingredients={_ingredients_count}"
            )
            # [P2-LLM-TIMEOUT-PIPELINE-METRICS · 2026-05-15] Tick distinto
            # del self_critique_correction_timeout — surgical-regen ya es
            # un retry del corrector tras un primer timeout, importante
            # discriminar en dashboards. `attempt=2` indica que estamos
            # en la segunda pasada del corrector sobre el mismo día.
            _emit_llm_timeout_metric(
                node="surgical_regen_timeout",
                timeout_threshold_s=float(CRITIQUE_FIX_TIMEOUT_S),
                actual_wait_s=float(CRITIQUE_FIX_TIMEOUT_S),
                llm=corrector_llm,
                extra_metadata={
                    "day_num": int(day_num),
                    "prompt_chars": int(_prompt_chars),
                    "target_chars": int(_target_chars),
                    "ingredients_count": int(_ingredients_count),
                    "model_label": _corrector_model,
                    "attempt": 2,
                },
            )
            pro_corrected, _pro_reason = await _attempt_pro_critique_correction(
                correction_prompt, day_num,
                log_prefix="[P5-MARKER-REGEN/PRO-FALLBACK]",
            )
            if pro_corrected:
                return day_num, pro_corrected
        except Exception as e:
            await _record_cb_failure_unless_transient(_corrector_cb, e)  # P1-ORCH-1/2
            logger.warning(
                f"⚠️ [P5-MARKER-REGEN] Error re-corrigiendo Día {day_num}: {e}. "
                f"Marker preservado."
            )
        return day_num, None

    results = await asyncio.gather(
        *[_re_correct_one(d) for d in marker_day_nums]
    )

    fixed_count = 0
    for day_num, corrected_day in results:
        if corrected_day is not None:
            for i, d in enumerate(days):
                if d.get("day") == day_num:
                    days[i] = corrected_day
                    break
            fixed_count += 1

    duration = round(time.time() - start_time, 2)
    logger.info(
        f"🩹 [P5-MARKER-REGEN] Completado en {duration}s — "
        f"{fixed_count}/{len(marker_day_nums)} días re-corregidos."
    )

    # [P3-PROTEIN-CEILING-GOAL-AWARE · 2026-06-14] Re-aplica el techo de proteína a los días
    # RE-GENERADOS por el corrector LLM: el surgical regen escapa el trim de assemble (corre
    # después), así que un día de volumen podía entregar >2.2 g/kg. Determinista, fail-safe →
    # cierra el gap "las mutaciones post-review escapan los guards de assemble".
    if PROTEIN_FLOOR_ENABLED:
        try:
            import re as _re_sr
            from nutrition_db import IngredientNutritionDB as _SRDB
            _tgt_p = None
            _macros_sr = plan_result.get("macros")
            if isinstance(_macros_sr, dict):
                _m = _re_sr.search(r"(\d+)", str(_macros_sr.get("protein", "")))
                if _m:
                    _tgt_p = float(_m.group(1))
            if _tgt_p and _tgt_p > 0:
                _srdb = _SRDB()
                _cp = _goal_aware_trim_ceiling_pct(form_data, _tgt_p)
                _trim_n = 0
                for _d in days:
                    if _trim_day_protein_to_ceiling(_d.get("meals", []) or [], _tgt_p, _srdb, ceiling_pct=_cp):
                        _trim_n += 1
                if _trim_n:
                    logger.info(f"🩹 [P3-PROTEIN-CEILING-GOAL-AWARE] Re-trim de techo aplicado a "
                                f"{_trim_n} día(s) re-generado(s) por surgical regen")
        except Exception as _srt_e:
            logger.warning(f"[P3-PROTEIN-CEILING-GOAL-AWARE] re-trim surgical falló: "
                           f"{type(_srt_e).__name__}: {_srt_e}")

    new_plan_result = {**plan_result, "days": days}

    # ============================================================
    # [P6-SURGICAL-PROMOTE] Promover snapshot tras surgical regen exitoso
    # ------------------------------------------------------------
    # Sin esto: el flujo después de surgical regen es:
    #   surgical_marker_regen → assemble_plan → review_plan (re-review)
    # Si re-review demote el nuevo plan (ej. de 'approved' a 'minor' por
    # un issue ortogonal en un día NO regenerado), `should_retry` enruta
    # a "end" (review_passed=False + no budget para retry). Ahí
    # `_swap_to_best_attempt_if_better` compara:
    #   - best_snapshot = pre-surgical (approved con markers, rank=0)
    #   - current = post-surgical (minor, rank=1)
    # Y restaura el snapshot pre-surgical → el usuario recibe el plan
    # con markers SIN RESOLVER, tirando el trabajo del surgical regen.
    #
    # Caso real corrida 2026-05-05 13:11-13:12: surgical regen fixed
    # exitosamente Días 2 y 3 (slot violations resueltas), pero re-review
    # rechazó por mismatch ingredient↔recipe en Día 1 (no relacionado al
    # surgical regen). P0-PIPE-1 restauró el plan pre-surgical → usuario
    # recibió plan CON las violaciones de slot que P5 había resuelto.
    #
    # Política: si AL MENOS 1 marker se resolvió, el plan post-surgical
    # es por construcción ≥ al pre-surgical (mismos otros días, ≥ markers
    # resueltos). Promoverlo a best garantiza que el usuario reciba la
    # mejora aunque re-review demote por issues ortogonales. Si re-review
    # APRUEBA, el branch `if approved` en review_plan_node sobrescribe
    # best con el plan re-aprobado — no hay regresión.
    # ============================================================
    state_update = {
        "plan_result": new_plan_result,
        "_marker_regen_attempted": True,
    }
    if fixed_count > 0:
        state_update["_best_attempt_plan"] = copy.deepcopy(new_plan_result)
        state_update["_best_attempt_severity"] = "approved"
        state_update["_best_attempt_reasons"] = []
        state_update["_best_attempt_review_passed"] = True
        state_update["_best_attempt_number"] = state.get("attempt", 1)
        logger.info(
            f"📌 [P6-SURGICAL-PROMOTE] Promoviendo plan post-surgical "
            f"({fixed_count}/{len(marker_day_nums)} markers resueltos) a "
            f"best_attempt — protege contra rollback si re-review demote "
            f"por issue ortogonal."
        )
    return state_update


_HIGH_SEVERITY_CONTEXTUAL_KEYWORDS = (
    # Restricciones del contexto del usuario que NO cambian entre intentos:
    # despensa, alergias, condiciones médicas, intolerancias.
    "despensa estricta", "despensa solo tiene", "no está en la despensa",
    "ingrediente no disponible", "no se encuentra en", "no existe en",
    "no tiene en la nevera",
    # Variantes de "fuera de inventario" que el LLM/reviewer escribe
    # con/sin contracción "del".
    "fuera de inventario", "fuera del inventario",
    "fuera de la despensa",
    "violación de pantry", "violacion de pantry",
    "violación de despensa", "violacion de despensa",
    # Estos disparan severity=critical normalmente, pero por defensa también
    # los listamos aquí — si por alguna razón un rechazo de alergia/condición
    # se clasifica como HIGH (no critical), tampoco debemos retry.
    "alergia", "alérgeno", "alergeno",
    "condición médica", "condicion medica", "intolerancia",
    "celíaco", "celiaco", "diabetes", "hipertensión", "hipertension",
)


def _classify_high_severity(rejection_reasons: list) -> str:
    """[P1-RETRY-CLASSIFY] Clasifica un rechazo HIGH como `contextual`
    (regenerar NO ayuda) vs `regenerable` (otro intento puede arreglarlo).

    Heurística por keywords en las razones del rechazo:
      - **Contextual** (no retry): la causa es una restricción del usuario
        que no cambia entre intentos — despensa estricta, alergias,
        condiciones médicas, intolerancias declaradas. Regenerar produciría
        el mismo error o un loop.
      - **Regenerable** (retry permitido): la causa es un fallo de calidad
        del LLM que un segundo intento puede corregir — skeleton fidelity
        violation, repetición excesiva, falta de variedad, recipe-ingredient
        coherence false positives, falta de proteína asignada.

    Default: `regenerable`. La política original (antes de este fix) era
    "todo HIGH = no retry" por miedo a loops. La realidad de producción
    (incidente 2026-05-05 + corrida 2026-05-05 01:49) muestra que la mayoría
    de HIGH son fallos de adherencia del LLM al prompt, no restricciones
    contextuales — y el retry funciona en la mayoría de los casos. Combinado
    con el cap de `MAX_ATTEMPTS=2` y el snapshot P0-PIPE-1, no hay riesgo
    de loop ni de entregar plan peor.

    Args:
        rejection_reasons: lista de strings de razones del rechazo médico.

    Returns:
        "contextual" o "regenerable".
    """
    if not rejection_reasons:
        return "regenerable"
    joined = " ".join(str(r) for r in rejection_reasons).lower()
    for kw in _HIGH_SEVERITY_CONTEXTUAL_KEYWORDS:
        if kw in joined:
            return "contextual"
    return "regenerable"


def _attempt_quality_rank(review_passed: bool, severity: Optional[str]) -> int:
    """[P0-PIPE-1] Rank de calidad de un intento: menor = mejor.

    Approved siempre gana (rank 0). Para rechazados, ordena por severidad
    según `_SEVERITY_RANK` (minor=1, high=2, critical=3). Usado por
    `review_plan_node` para decidir si el intento actual debe promoverse a
    `_best_attempt_*` y por `_swap_to_best_attempt_if_better` para decidir
    si restaurar el best al final del pipeline.
    """
    if review_passed:
        return 0
    return _SEVERITY_RANK.get(severity or "minor", 1)


def _swap_to_best_attempt_if_better(final_state: dict) -> bool:
    """[P0-PIPE-1] Restaura el snapshot del MEJOR intento como `plan_result`
    si el actual fue rechazado con peor severidad.

    Mutación in-place sobre `final_state`. Devuelve True si hubo swap.

    Lógica:
      1. Si no hay `_best_attempt_plan` snapshot → no-op (primera generación,
         o nada que rescatar).
      2. Si el intento actual fue aprobado → no-op (ya es el mejor posible).
      3. Si rank(best) < rank(current) → swap completo (plan, severity,
         reasons, review_passed) y marca `_best_attempt_swapped=True` en el
         plan resultante para que el frontend pueda mostrar telemetría.
      4. Si rank(best) >= rank(current) → no-op (current ya es el mejor o
         igual; preservamos coherencia con `attempt` actual).

    Por qué ANTES de `_apply_critical_review_guardrails`:
      Si el best era `approved`/`minor` y el actual es `critical`, el
      guardrail dispararía un fallback matemático innecesario. Restaurar el
      best primero permite que el guardrail vea el plan correcto y solo
      marque transparency en lugar de regenerar.
    """
    best_plan = final_state.get("_best_attempt_plan")
    if not isinstance(best_plan, dict) or not best_plan:
        return False

    current_passed = bool(final_state.get("review_passed"))
    if current_passed:
        # Current ya está aprobado — el snapshot best (si existe) puede ser
        # otro aprobado igual de bueno o uno rechazado peor; nunca peor que
        # el actual. No-op.
        return False

    current_severity = final_state.get("_rejection_severity") or "minor"
    best_passed = bool(final_state.get("_best_attempt_review_passed"))
    best_severity = final_state.get("_best_attempt_severity") or "minor"

    current_rank = _attempt_quality_rank(current_passed, current_severity)
    best_rank = _attempt_quality_rank(best_passed, best_severity)

    if best_rank >= current_rank:
        return False  # Current es igual o mejor que el best.

    # Best gana — restauramos.
    best_attempt_n = final_state.get("_best_attempt_number")
    current_attempt_n = final_state.get("attempt", 1)
    best_reasons = list(final_state.get("_best_attempt_reasons") or [])

    logger.warning(
        f"🔄 [P0-PIPE-1] Plan del intento #{current_attempt_n} rechazado "
        f"con severity={current_severity!r} (rank={current_rank}); "
        f"restaurando intento #{best_attempt_n} con "
        f"severity={best_severity!r} review_passed={best_passed} "
        f"(rank={best_rank}). Razones del intento descartado: "
        f"{(final_state.get('rejection_reasons') or [])[:3]}"
    )

    final_state["plan_result"] = best_plan
    final_state["review_passed"] = best_passed
    final_state["_rejection_severity"] = best_severity if not best_passed else "minor"
    final_state["rejection_reasons"] = best_reasons
    # Marcador en el plan para telemetría / debugging downstream. NO
    # afecta a guardrails (que leen `_review_failed_but_delivered` /
    # `_critical_rejection`).
    if isinstance(final_state["plan_result"], dict):
        final_state["plan_result"]["_best_attempt_swapped_from"] = current_attempt_n
        final_state["plan_result"]["_best_attempt_swapped_severity"] = current_severity
    return True


# [P2-HIST-AUDIT-4 · 2026-05-09] Helper SSOT para aplicar el cap del
# `_shopping_coherence_block_history` con knob configurable y telemetría
# al truncar.
#
# Bug original (audit historial 2026-05-08):
#   El cap estaba hardcoded `if len(new_history) > 20` en DOS sites:
#     - graph_orchestrator.py:6941 (assemble_plan_node, P3-NEW-C).
#     - graph_orchestrator.py:7855 (_recompute_aggregates_after_swap,
#       P2-B post-swap revalidation).
#   Si un plan superaba 20 entries, las viejas se descartaban
#   silenciosamente — el chip "X ajustes" en la card subnumeraba
#   sin señal al operador. Trade-off documentado en P3-NEW-C, pero
#   sin telemetría para detectar cuándo el cap empieza a doler.
#
# Fix:
#   - Knob `MEALFIT_COHERENCE_BLOCK_HISTORY_CAP` (default 20)
#     permite bumpear sin redeploy si el dato real lo justifica.
#     Validator >= 1 evita cap=0 patológico (truncaría toda la lista).
#   - Logger.warning cuando truncamos: incluye plan_id (si disponible)
#     y `truncated_count` para que SRE pueda agregar via grep o cron.
#   - SSOT: ambos call sites importan el helper; cualquier flujo
#     futuro que apenda al history hereda la política sin coordinación.
_COHERENCE_BLOCK_HISTORY_CAP_DEFAULT = 20


def _coherence_block_history_cap() -> int:
    """Resuelve el knob runtime con default 20 y guard >= 1.

    Función dedicada (no constante module-level) para que el override
    via env var aplique sin reiniciar el proceso — coherente con
    `_env_int` que lee `os.environ` en cada llamada.

    `_env_int` no soporta validator (solo `_env_float` lo hace post-P1-3
    2026-05-08); aplicamos guard explícito acá. Cap < 1 sería patológico
    (descartaría toda la lista en cada append) — fallback al default
    para protegernos de un knob mal configurado en runtime.
    """
    cap = _env_int(
        "MEALFIT_COHERENCE_BLOCK_HISTORY_CAP",
        _COHERENCE_BLOCK_HISTORY_CAP_DEFAULT,
    )
    if cap < 1:
        try:
            logger.warning(
                "[P2-HIST-AUDIT-4] MEALFIT_COHERENCE_BLOCK_HISTORY_CAP=%d "
                "es inválido (debe ser >= 1). Fallback al default %d.",
                cap, _COHERENCE_BLOCK_HISTORY_CAP_DEFAULT,
            )
        except Exception:
            pass
        return _COHERENCE_BLOCK_HISTORY_CAP_DEFAULT
    return cap


def _apply_coherence_history_cap(
    prior_history,
    new_entry,
    *,
    plan_id_hint=None,
):
    """Anexa `new_entry` a `prior_history` y aplica el cap configurable.

    Returns:
        list: nueva lista con `new_entry` al final, truncada al cap.
        Si el truncamiento ocurrió, emite logger.warning con
        `truncated_count` para diagnóstico SRE.

    Args:
        prior_history: lista actual (puede ser None / no-list → se
            normaliza a []).
        new_entry: dict de la nueva entrada a anexar.
        plan_id_hint: opcional, plan_id para incluir en el log
            cuando se trunca. Sin esto, el log sigue siendo útil pero
            el operador necesita más grep para localizar el plan.
    """
    if not isinstance(prior_history, list):
        prior_history = []
    new_history = list(prior_history) + [new_entry]
    cap = _coherence_block_history_cap()
    if len(new_history) > cap:
        truncated_count = len(new_history) - cap
        new_history = new_history[-cap:]
        try:
            logger.warning(
                "[P2-HIST-AUDIT-4/COH-HISTORY-TRUNCATED] plan=%s "
                "truncated=%d cap=%d. Considera bumpear "
                "MEALFIT_COHERENCE_BLOCK_HISTORY_CAP si el caso es "
                "habitual.",
                plan_id_hint or "unknown",
                truncated_count,
                cap,
            )
        except Exception:
            # Logging es best-effort; no debe romper el append.
            pass
    return new_history


def _emit_post_swap_coherence_alert(
    *,
    user_id,
    plan_id,
    divergences: list,
    hyp_counter: dict,
    critical_total: int,
    household: float,
) -> bool:
    """[P2-2 · 2026-05-08] Emite a `system_alerts` cuando un swap deja
    divergencias críticas tras `_recompute_aggregates_after_swap`.

    Por qué existe:
      P2-B (2026-05-08) cerró el gap de telemetría — toda divergencia
      post-swap se persiste en `_shopping_coherence_block_history` con
      `action_taken="post_swap_revalidation"`. Pero el cron P3-B trata
      ese bucket como observabilidad pura (NO anomalous), así que un
      plan entregado con `cap_swallowed_modifier` (receta dice X, lista
      no lo tiene) o magnitudes >30% off no levantaba alerta.
      Resultado: el usuario recibía un PDF de compras inconsistente con
      sus recetas y operadores sólo lo veían en el log de info.

    Este helper escala las divergencias *críticas* (no las benignas como
    typos canónicos) a `system_alerts` con cooldown per-user, severity=
    warning. Cooldown evita que un usuario con racha de planes problemáticos
    inunde la tabla; el operador inspecciona `metadata.divergences_sample`
    para diagnosticar.

    Definición de "crítica":
      - hypothesis="cap_swallowed_modifier" — el modo de fallo más
        actionable (el aggregator perdió un ingrediente que SÍ está en
        la receta).
      - magnitude=True con `|delta_pct| > 0.30` — error de cantidad
        severo (>30% off vs receta escalada por household).
      Otras hipótesis (`yield_uncovered`, `pantry_overdeduct`, `unknown`,
      `unit_mismatch` con delta <30%) son ruido benigno o ya capturadas
      por otros guards y NO disparan alerta.

    Knobs (registrados en `_KNOBS_REGISTRY`):
      MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED (bool, default True)
        kill switch — si False, no emite (rollback rápido sin redeploy).
      MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD (int, default 3)
        N divergencias críticas en un swap para escalar a alerta.
      MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS (int, default 6)
        ventana de dedupe per-user.

    Idempotente vía `ON CONFLICT (alert_key) DO UPDATE` — la misma
    alert_key se re-trigger con `triggered_at=NOW()` si vuelve a
    aparecer, en vez de duplicar fila.

    Returns:
        True si la alerta se emitió/upserted; False si saltó por
        cooldown, kill switch off, threshold no alcanzado, o cualquier
        error best-effort (no aborta el caller bajo ninguna circunstancia).
    """
    if not _env_bool("MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED", True):
        return False
    threshold = _env_int("MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD", 3)
    if critical_total < threshold:
        return False
    cooldown_h = _env_int("MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS", 6)

    # Dedupe per-user: un usuario con racha mala no inunda la tabla.
    # Si user_id es None (guest), usamos prefix del plan_id como discriminator;
    # caso patológico (ambos None) cae a "unknown" y dedupe global por 6h.
    key_id = user_id or (str(plan_id)[:8] if plan_id else None) or "unknown"
    alert_key = f"post_swap_critical_divergence_{key_id}"

    # Cooldown query: si ya hay alerta abierta dentro del window, skip.
    # Si la query falla, mejor sobre-alertar que silenciar — no abortamos.
    try:
        existing = execute_sql_query(
            """
            SELECT 1 FROM system_alerts
            WHERE alert_key = %s
              AND resolved_at IS NULL
              AND triggered_at > NOW() - make_interval(hours => %s)
            LIMIT 1
            """,
            (alert_key, cooldown_h),
            fetch_one=True,
        )
        if existing:
            return False
    except Exception:
        pass

    # Sample top 5 divergencias para diagnóstico operacional.
    sample = []
    for d in (divergences or [])[:5]:
        if not isinstance(d, dict):
            continue
        sample.append({
            "food": d.get("food"),
            "side": d.get("side"),
            "hypothesis": d.get("hypothesis"),
            "magnitude": bool(d.get("magnitude")),
            "delta_pct": d.get("delta_pct"),
        })

    metadata = {
        "user_id": str(user_id) if user_id else None,
        "plan_id": str(plan_id) if plan_id else None,
        "divergence_count": len(divergences or []),
        "critical_count": critical_total,
        "hypotheses": dict(hyp_counter or {}),
        "household_multiplier": float(household) if household else None,
        "divergences_sample": sample,
        "cooldown_hours": cooldown_h,
        "threshold": threshold,
    }
    title = "Coherencia recetas↔lista degradada post-swap"
    message = (
        f"Tras swap a best_attempt, la re-validación detectó "
        f"{critical_total} divergencia(s) crítica(s) "
        f"(cap_swallowed_modifier o magnitud |Δ|>30%). "
        f"Total divergencias: {len(divergences or [])}. "
        f"Hipótesis: {dict(hyp_counter or {})}. user_id={user_id}. "
        f"Investigar: ¿el best_attempt fue capturado con ingredientes "
        f"que el aggregator no mapeó al ítem normalizado, o hubo drift "
        f"de canonicalización? Ver `canonicalize_pavo` (P3-4) y "
        f"`test_p2_2_post_swap_critical_divergence_alert`."
    )
    affected = [str(user_id)] if user_id else []

    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message,
                 metadata, affected_user_ids)
            VALUES (%s, 'post_swap_coherence', 'warning', %s, %s,
                    %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
                SET triggered_at = NOW(),
                    severity = EXCLUDED.severity,
                    title = EXCLUDED.title,
                    message = EXCLUDED.message,
                    metadata = EXCLUDED.metadata,
                    affected_user_ids = EXCLUDED.affected_user_ids,
                    resolved_at = NULL
            """,
            (
                alert_key, title, message,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(affected, ensure_ascii=False),
            ),
        )
        return True
    except Exception as _ins_err:
        logger.warning(
            f"[P2-2/POST-SWAP-ALERT] INSERT system_alerts falló (best-effort): "
            f"{type(_ins_err).__name__}: {_ins_err}"
        )
        return False


async def _recompute_aggregates_after_swap(final_state: dict) -> None:
    """[ROLLBACK-AGGREGATE-FIX 2026-05-07] Recomputa aggregated_shopping_list_*
    desde los `days` del plan restaurado tras swap.

    Llamado SOLO después de que `_swap_to_best_attempt_if_better` retornó True.
    Reusa la misma lógica de `assemble_plan_node` (get_shopping_list_delta ×3
    + _build_hybrid) para garantizar que weekly/biweekly/monthly reflejen el
    mismo plan días, sin residuos de aggregates capturados en otro punto del
    flow.

    Best-effort: si falla, los aggregates del snapshot persisten (riesgo de
    inconsistencia entre ciclos pero plan se entrega). El caller atrapa.
    """
    plan_result = final_state.get("plan_result") or {}
    if not isinstance(plan_result, dict) or not plan_result.get("days"):
        return  # Nada que aggregar

    form_data = final_state.get("form_data") or {}
    _uid = form_data.get("user_id")
    if not _uid or _uid == "guest":
        _uid = None

    from shopping_calculator import (
        get_shopping_list_delta,
        fetch_inventory_and_consumed_for_plan,
        _build_hybrid_shopping_list as _build_hybrid,
    )
    from constants import compute_household_multiplier

    household = compute_household_multiplier(form_data)

    if _uid:
        inv_snapshot, consumed_snapshot = await _adb(
            fetch_inventory_and_consumed_for_plan, _uid, plan_result, True
        )
        aggr_list_7, aggr_list_15, aggr_list_30 = await asyncio.gather(
            _adb(get_shopping_list_delta, _uid, plan_result, True, False, True, 1.0 * household,
                 inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
            _adb(get_shopping_list_delta, _uid, plan_result, True, False, True, 2.0 * household,
                 inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
            _adb(get_shopping_list_delta, _uid, plan_result, True, False, True, 4.0 * household,
                 inventory_override=inv_snapshot, consumed_override=consumed_snapshot),
        )
    else:
        aggr_list_7, aggr_list_15, aggr_list_30 = [], [], []

    # Hybrid para biweekly/monthly (mismo patrón que assemble_plan_node)
    try:
        aggr_list_15_hybrid = _build_hybrid(aggr_list_7, aggr_list_15) if aggr_list_15 else aggr_list_15
        aggr_list_30_hybrid = _build_hybrid(aggr_list_7, aggr_list_30) if aggr_list_30 else aggr_list_30
    except Exception as e_hybrid:
        logger.warning(f"[ROLLBACK-AGGREGATE-FIX] Hybrid construction falló: {e_hybrid}")
        aggr_list_15_hybrid = aggr_list_15
        aggr_list_30_hybrid = aggr_list_30

    grocery_duration = form_data.get("groceryDuration", "weekly")
    if grocery_duration == "biweekly":
        aggr_list = aggr_list_15_hybrid
    elif grocery_duration == "monthly":
        aggr_list = aggr_list_30_hybrid
    else:
        aggr_list = aggr_list_7

    plan_result["aggregated_shopping_list"] = aggr_list
    plan_result["aggregated_shopping_list_weekly"] = aggr_list_7
    plan_result["aggregated_shopping_list_biweekly"] = aggr_list_15_hybrid
    plan_result["aggregated_shopping_list_monthly"] = aggr_list_30_hybrid

    logger.info(
        f"🔄 [ROLLBACK-AGGREGATE-FIX] Aggregates recomputadas tras swap: "
        f"weekly={len(aggr_list_7)} items, biweekly={len(aggr_list_15_hybrid)}, "
        f"monthly={len(aggr_list_30_hybrid)}. Coherencia entre ciclos garantizada."
    )

    # [P2-B · 2026-05-08] Re-validar coherencia recetas↔lista tras el swap.
    # ------------------------------------------------------------------
    # `assemble_plan_node` ya invocó `run_shopping_coherence_guard` para el
    # plan_result actual (el "peor"); pero al hacer swap a `best_attempt` y
    # RECOMPUTAR aggregates aquí, las divergencias del best_attempt original
    # quedaron stale y las nuevas (sobre los aggregates frescos) jamás se
    # registraron. Sin esta capa, el cron P3-B subestimaba la tasa real de
    # divergencias en planes entregados (best_attempt podía tener un block
    # silencioso que no figuraba en `_shopping_coherence_block_history`).
    #
    # Diseño:
    #   - mode_override="warn" → telemetría pura, NO mutamos
    #     `_shopping_coherence_block` (review ya pasó, no podemos rechazar).
    #   - action_taken="post_swap_revalidation" → bucket dedicado en P3-B
    #     cron, distingue divergencias detectadas post-review de las del
    #     flujo normal. NO es anomalous: es observabilidad.
    #   - swap=True flag → para queries SQL que necesiten filtrar.
    #   - Best-effort: cualquier error → log warning, no aborta el caller.
    try:
        from shopping_calculator import run_shopping_coherence_guard
        from datetime import datetime as _coh_dt, timezone as _coh_tz
        from collections import Counter as _coh_Counter

        coh_div = run_shopping_coherence_guard(
            plan_result, mode_override="warn", multiplier=household
        ) or []
        if coh_div:
            hyp_counter = _coh_Counter(
                str(d.get("hypothesis") or "unknown") for d in coh_div
            )
            # [P2-2 · 2026-05-08] Critical-divergence count: subset accionable
            # del ruido total. Define qué cuenta como "crítica" para escalar
            # a `system_alerts` (ver `_emit_post_swap_coherence_alert` docstring).
            #   - cap_swallowed_modifier: el aggregator perdió un ingrediente
            #     que la receta SÍ menciona (modo de fallo más actionable).
            #   - magnitude con |delta_pct| > 0.30: error de cantidad severo.
            critical_cap = int(hyp_counter.get("cap_swallowed_modifier", 0))
            critical_magnitudes = sum(
                1 for d in coh_div
                if d.get("magnitude")
                and abs(float(d.get("delta_pct") or 0.0)) > 0.30
            )
            critical_total = critical_cap + critical_magnitudes

            entry = {
                "ts": _coh_dt.now(_coh_tz.utc).isoformat(),
                "attempt": None,
                "swap": True,
                "divergence_count": len(coh_div),
                "presence_count": sum(1 for d in coh_div if not d.get("magnitude")),
                "magnitude_count": sum(1 for d in coh_div if d.get("magnitude")),
                "hypotheses": dict(hyp_counter),
                # [P2-2] Critical count separado de divergence_count para
                # que el cron P3-B y dashboards puedan distinguir señal real
                # de ruido benigno (typos canónicos, fantasmas leves).
                "critical_count": critical_total,
                # Telemetría pura: no mutamos el flag aunque mode_override
                # fuera "block". Reflejamos el estado actual (que `assemble`
                # pudo haber seteado pre-swap si el best_attempt lo traía).
                "block_set": bool(plan_result.get("_shopping_coherence_block")),
                "action_taken": "post_swap_revalidation",
            }

            # [P1-SWAP-COHERENCE-ESCALATE · 2026-05-22] Cuando hay
            # divergencias críticas Y el knob está activo (default True,
            # mirror del patrón P2-COHERENCE-1 en chunk_worker T2),
            # exponemos un campo legible para el frontend. Pre-fix la
            # única vía de notificación era el cron diario P3-B que
            # corre 04:00 UTC (delay 6-24h al user). Post-fix: el plan
            # entregado lleva inline las warnings, el Dashboard las puede
            # renderear como banner amber (mismo lenguaje visual que el
            # banner `_quality_degraded` del P1-LOW-SIGNAL-FALLBACK).
            #
            # NO mutamos `_shopping_coherence_block` porque review ya
            # pasó — bloquear la entrega del plan no es viable acá. Es
            # señal user-visible (transparency), no kill-switch.
            #
            # Knob: MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY (bool,
            # default True). Flip a False revierte al telemetry-only.
            try:
                _escalate_enabled = _env_bool(
                    "MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY", True
                )
            except Exception:
                _escalate_enabled = True
            if _escalate_enabled and critical_total > 0:
                # Resumen compacto para inyectar al plan_result (cap a 5
                # divergencias para no inflar plan_data). Frontend lo
                # renderea inline.
                _severe_summary = []
                for _d in coh_div[:5]:
                    if (
                        _d.get("hypothesis") == "cap_swallowed_modifier"
                        or (_d.get("magnitude") and abs(float(_d.get("delta_pct") or 0.0)) > 0.30)
                    ):
                        _severe_summary.append({
                            "ingredient": str(_d.get("ingredient") or _d.get("canonical_name") or "?")[:80],
                            "hypothesis": str(_d.get("hypothesis") or "unknown"),
                            "delta_pct": round(float(_d.get("delta_pct") or 0.0), 3),
                        })
                plan_result["_swap_coherence_warnings"] = {
                    "critical_count": critical_total,
                    "summary": _severe_summary,
                    "detected_at": entry["ts"],
                }
                # Telemetría adicional para que el cron P3-B pueda
                # distinguir escalaciones inline-user-visible de las
                # warn-only históricas. NO altera contrato de buckets.
                entry["escalated_user_visible"] = True
                logger.warning(
                    f"⚠️ [P1-SWAP-COHERENCE-ESCALATE] critical_total={critical_total} "
                    f"→ _swap_coherence_warnings inyectado al plan (severe_n="
                    f"{len(_severe_summary)})."
                )

            # [P2-2 · 2026-05-08] Escalación inline a system_alerts cuando
            # critical_total >= threshold. Best-effort: cualquier fallo
            # del helper se loguea pero NO interrumpe el history append.
            entry["alerted"] = False
            try:
                _alerted = _emit_post_swap_coherence_alert(
                    user_id=_uid,
                    plan_id=plan_result.get("id") or plan_result.get("plan_id"),
                    divergences=coh_div,
                    hyp_counter=dict(hyp_counter),
                    critical_total=critical_total,
                    household=household,
                )
                entry["alerted"] = bool(_alerted)
            except Exception as _alert_err:
                logger.warning(
                    f"[P2-2/POST-SWAP-ALERT] no-op (best-effort): "
                    f"{type(_alert_err).__name__}: {_alert_err}"
                )

            prior_history = plan_result.get("_shopping_coherence_block_history") or []
            # [P2-HIST-AUDIT-4 · 2026-05-09] Helper SSOT (knob
            # `MEALFIT_COHERENCE_BLOCK_HISTORY_CAP`) reemplaza el cap
            # hardcoded del sitio espejo en assemble_plan_node.
            new_history = _apply_coherence_history_cap(
                prior_history,
                entry,
                plan_id_hint=(plan_result.get("id") or plan_result.get("plan_id")),
            )
            plan_result["_shopping_coherence_block_history"] = new_history
            logger.info(
                f"🔄 [ROLLBACK-COH-REVALIDATE] {len(coh_div)} divergencia(s) "
                f"detectada(s) post-swap (critical={critical_total}, "
                f"alerted={entry['alerted']}). history_len={len(new_history)}."
            )
    except Exception as _coh_err:
        logger.warning(
            f"[ROLLBACK-COH-REVALIDATE] no-op (best-effort): "
            f"{type(_coh_err).__name__}: {_coh_err}"
        )


@_node_label("reviewer")
async def review_plan_node(state: PlanState) -> dict:
    """Revisa el plan generado para verificar seguridad médica."""
    plan = state["plan_result"]
    form_data = state["form_data"]
    taste_profile = state.get("taste_profile", "")
    attempt = state.get("attempt", 1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🩺 [AGENTE REVISOR MÉDICO] Verificando plan (intento #{attempt})...")
    logger.info(f"{'='*60}")
    
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
    # [P2-MEDICAL-FACTCHECK-GATE] `_has_real_medical_flags` filtra el sentinel "Ninguna"
    # (el form manda allergies=["Ninguna"], que era truthy) — antes el bypass nunca
    # disparaba para usuarios sin restricciones reales.
    if not _has_real_medical_flags(allergies) and not _has_real_medical_flags(medical_conditions) and diet_type == "balanced" and not _has_real_medical_flags(dislikes) and not taste_profile:
        logger.info("✅ [REVISOR] Sin restricciones declaradas → Bypassing LLM Reviewer, procediendo a validaciones deterministas.")
        approved = True
        issues = []
        severity = "none"
        fact_check_report = "N/A"
    else:
        # ============================================================
        # FASE 1: AGENTE DE FACT-CHECKING (INVESTIGACIÓN CLÍNICA)
        # ============================================================
        fact_check_report = "Sin hallazgos adicionales."
        # [P2-MEDICAL-FACTCHECK-GATE · 2026-06-13] Solo investigar interacciones si hay
        # alergias/condiciones REALES (no el sentinel "Ninguna"). Antes corría ~40s de
        # tool calls por-ingrediente para usuarios sanos. Skip seguro: sin condiciones,
        # no hay interacciones que evaluar.
        if _has_real_medical_flags(allergies) or _has_real_medical_flags(medical_conditions):
            from tools_medical import consultar_base_datos_medica
            # P1-10: SystemMessage/HumanMessage/ToolMessage a nivel módulo

            logger.info("🔬 [FACT-CHECKING] Iniciando investigación de alergias/condiciones...")
            # P1-Q3: capturar modelo del fact-checker para CB per-modelo
            # [P1-FLASH-LITE-AUX-NODES · 2026-05-15] Default Flash-Lite via knob
            # `MEALFIT_FACT_CHECKER_MODEL`. Pre-fix: `_route_model(force_fast=True)`
            # hardcodeaba `gemini-3-flash-preview` (Flash regular).
            _fact_checker_model = _fact_checker_model_name(form_data)  # P2-ORCH-7 risk-tier
            _fact_checker_cb = _get_circuit_breaker(_fact_checker_model)
            fact_checker_llm = ChatDeepSeek(
                model=_fact_checker_model,
                temperature=0.0,
            ).bind_tools([consultar_base_datos_medica])
            
            fc_sys_prompt = "Eres un investigador clínico. Revisa las alergias y condiciones médicas frente a los ingredientes presentados. Usa tu herramienta para investigar posibles reacciones cruzadas, contraindicaciones o interacciones peligrosas. Cuando termines o si no ves riesgo, responde con un REPORTE CLÍNICO CONCISO con tus hallazgos."
            
            # [P3-COST-FACTCHECK-INGREDIENTS-DEDUP · 2026-06-01] Dedup EXACTO
            # (set sobre strings con cantidad) solo para el prompt del
            # fact-checker, que re-envía esta HumanMessage en cada una de hasta
            # 4 iteraciones del loop (10676). `all_ingredients` crudo (con
            # duplicados) se PRESERVA intacto para validate_ingredients_against_pantry
            # downstream. NO normaliza ni quita cantidades (eso arriesgaría
            # reorder/pérdida clínica) — solo colapsa repeticiones byte-idénticas
            # acumuladas entre días. Cero costo LLM. tooltip-anchor:
            # P3-COST-FACTCHECK-INGREDIENTS-DEDUP
            _fc_ingredients = sorted(set(all_ingredients))
            fc_messages = [
                SystemMessage(content=fc_sys_prompt),
                HumanMessage(content=f"Alergias: {allergies}\nCondiciones: {medical_conditions}\nDieta: {diet_type}\nIngredientes a evaluar: {_fc_ingredients}")
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
                            logger.warning(f"⚠️ [FACT-CHECK] Reporte sospechosamente corto "
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
                                logger.info(f"🔍 [FACT-CHECK] Consulta DB: {tool_call['args']} -> {str(tool_res)[:80]}...")
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
                                logger.info(f"⏱️ [FACT-CHECK] Tool timeout (>{_FACT_CHECK_TOOL_TIMEOUT:.0f}s) "
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
                    # [P1-ORCH-1/2 · 2026-05-28] Helper centralizado (excluye 5xx
                    # transitorio + spend-cap). El timeout de la TOOL clínica
                    # (arriba) NO usa el helper: ahí fail-closed es lo correcto.
                    await _record_cb_failure_unless_transient(_fact_checker_cb, fc_e)
                    logger.warning(f"⚠️ [FACT-CHECK] Error durante la investigación: {fc_e}")
                    fact_check_report = f"Error en la investigación: {str(fc_e)}. Asumir precaución máxima."
                    break
            else:
                # P1-7: El loop terminó sin break — el fact-checker no convergió en
                # 4 iteraciones (siguió pidiendo tool calls). Mantenemos el reporte
                # inicial pero lo logueamos para detectar bucles patológicos.
                logger.warning(f"⚠️ [FACT-CHECK] No convergió en 4 iteraciones (loop de tool calls). "
                      f"Usando reporte conservador inicial.")

        # ============================================================
        # FASE 2: REVISIÓN DETERMINISTA FINAL
        # ============================================================
        # [P3-COST-REVIEWER-CACHE · 2026-06-01] Split del REVIEWER_SYSTEM_PROMPT
        # estático (~350-450 tok) a un SystemMessage separado para que Gemini lo
        # trate como system_instruction cacheable (mismo patrón ya vivo en
        # planner/day-gen/evaluator bajo PROMPT_CACHE_SYSTEM_MESSAGE; el reviewer
        # era el último nodo del pipeline en string plano). El cache-hit lo dan
        # los retries tenacity del mismo plan + reviews consecutivas de los crons
        # nightly/chunk dentro del TTL. Mismo texto, cero cambio semántico. Rama
        # else = string plano legacy (rollback sin redeploy vía el knob existente
        # MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE). tooltip-anchor: P3-COST-REVIEWER-CACHE
        review_human_content = f"""--- RESTRICCIONES DEL PACIENTE ---
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

Responde ÚNICAMENTE con el JSON de revisión.
"""
        if PROMPT_CACHE_SYSTEM_MESSAGE:
            review_prompt = [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=review_human_content),
            ]
        else:
            review_prompt = f"{REVIEWER_SYSTEM_PROMPT}\n{review_human_content}"
        # [P3-GENCHUNK-SPEED · 2026-06-01] Eliminado el bloque
        # `--- TODOS LOS INGREDIENTES DEL PLAN --- {json.dumps(all_ingredients)}`
        # que duplicaba textualmente cada ingrediente ya presente (agrupado
        # por comida/día, vista más rica que el reviewer realmente escanea —
        # ver REVIEWER_SYSTEM_PROMPT punto 1) en `all_meals_summary` arriba.
        # `all_ingredients` SIGUE construido: lo consumen los validadores
        # deterministas downstream (validate_ingredients_against_pantry) y la
        # FASE 1 fact-checker — solo se quitó la duplicación en el prompt del LLM.
        # tooltip-anchor: P3-GENCHUNK-SPEED-REVIEWER-DEDUP
        
        # P1-Q3: capturar modelo del reviewer para CB per-modelo
        # [P1-FLASH-LITE-AUX-NODES · 2026-05-15] Default Flash-Lite via knob
        # `MEALFIT_REVIEWER_MODEL` — tarea schema-strict (ReviewResult) con
        # temp=0.1. Pre-fix: `_route_model(form_data, attempt=1)` escalaba a
        # Pro en perfiles clínicos. Override por knob si regresión.
        _reviewer_model = _reviewer_model_name(form_data)  # P2-ORCH-7 risk-tier
        _reviewer_cb = _get_circuit_breaker(_reviewer_model)
        reviewer_llm = ChatDeepSeek(
            model=_reviewer_model,
            temperature=0.1,  # Temperatura muy baja para ser preciso
            max_retries=0,
            timeout=60
        ).with_structured_output(ReviewResult)

        # Invocar con reintentos automáticos
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            retry=retry_if_exception(lambda e: not _is_plan_spend_cap_error(e)),  # P2-ORCH-4
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(f"⚠️  [REVISOR] Reintento #{retry_state.attempt_number}...")
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
                await _record_cb_failure_unless_transient(_reviewer_cb, e)  # P1-Q3 · P1-ORCH-1/2
                raise e
        
        try:
            result: ReviewResult = await invoke_with_retry()
            approved = result.approved
            issues = result.issues
            severity = result.severity
            llm_affected_days = result.affected_days
        except Exception as e:
            logger.warning(f"⚠️  [REVISOR] Error en structured output, RECHAZANDO por defecto (Fail-Closed): {e}")
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
        logger.error(f"🚨 [REVISOR] Schema inválido detectado por assemble_plan: {schema_errors}")
        approved = False
        issues.append(
            f"SCHEMA INVÁLIDO: el plan no cumple la estructura esperada y no es renderizable. "
            f"Detalles: {schema_errors}"
        )
        severity = _severity_max(severity, "critical")

    # [C2-ALLERGEN-GUARD · 2026-06-13] Backstop DETERMINISTA de alérgenos sobre el revisor
    # LLM. Si CUALQUIER ingrediente matchea una alergia declarada (+ sinónimos), rechazo
    # CRÍTICO inmediato → regen con directiva. Para alérgenos, jamás servir > comodidad.
    _had_allergen_critical = False  # [P3-CONDITION-RULES] marca criticals que NO deben degradarse
    if ALLERGEN_HARD_GUARD and _has_real_medical_flags(allergies):
        _allergen_viol = _scan_allergen_violations(plan, allergies)
        if _allergen_viol:
            _had_allergen_critical = True
            _viol_str = "; ".join(f"'{_ing}' (alérgeno '{_term}') en {_mn}"
                                  for _mn, _ing, _term in _allergen_viol[:6])
            logger.error(f"🚨 [C2-ALLERGEN-GUARD] {len(_allergen_viol)} violación(es) de alérgeno "
                         f"declarado: {_viol_str}")
            approved = False
            issues.append(
                f"ALÉRGENO DETECTADO (rechazo de seguridad clínica): el plan contiene "
                f"ingrediente(s) que el usuario declaró como alergia. DEBES eliminarlos y "
                f"reemplazarlos por alternativas seguras. Violaciones: {_viol_str}"
            )
            severity = _severity_max(severity, "critical")

    # [P3-PROTEIN-FLOOR · 2026-06-13] Validador DURO de piso de proteína: si algún día entrega
    # < HARD_PCT del target diario → rechazo → regen con directiva. Cierra el déficit sistémico
    # (la re-auditoría del plan fresco halló 68% del target). Simétrico al techo C1: un techo
    # sin piso es media solución para hipertrofia. Es un BACKSTOP — el closer determinista
    # (assemble_plan_node) ya rellena al target, así que rara vez dispara (solo si no hubo
    # proteína de alta densidad disponible). Severity high → retry si hay budget, no falla duro.
    # [P3-CONDITION-RULES · 2026-06-14] Exención renal: si se aplicó el cap renal de proteína, NO
    # correr el piso duro — su directiva de rechazo ordena "fuente animal de alta densidad para
    # ganancia muscular", justo lo CONTRARIO de lo clínicamente correcto en ERC (KDIGO baja la
    # proteína). El cap ya garantiza que la proteína no es excesiva; un "déficit" vs el target
    # capeado es aceptable en renal. Evita un retry que empuje proteína arriba en un paciente renal.
    _renal_capped_plan = bool((plan.get("renal_protein_cap") or {}).get("applied")) if isinstance(plan, dict) else False
    if PROTEIN_FLOOR_HARD_GATE and not _renal_capped_plan:
        try:
            import re as _re_pf
            _tgt_p = None
            _macros = plan.get("macros")
            if isinstance(_macros, dict):
                _mp = _re_pf.search(r"(\d+)", str(_macros.get("protein", "")))
                if _mp:
                    _tgt_p = float(_mp.group(1))
            if _tgt_p and _tgt_p > 0:
                _short_days = []
                for _di_pf, _day_pf in enumerate(plan.get("days", []) or [], 1):
                    _dp = sum(_meal_macro_num(_mm.get("protein")) for _mm in (_day_pf.get("meals", []) or []))
                    if _dp < PROTEIN_FLOOR_HARD_PCT * _tgt_p:
                        _short_days.append((_day_pf.get("day", _di_pf), round(_dp), round(_tgt_p)))
                if _short_days:
                    _sd_str = "; ".join(f"Día {d}: {p}g de {t}g" for d, p, t in _short_days)
                    logger.warning(f"🛡 [P3-PROTEIN-FLOOR] {len(_short_days)} día(s) bajo el piso de "
                                   f"proteína ({int(PROTEIN_FLOOR_HARD_PCT*100)}% de {int(_tgt_p)}g): {_sd_str}")
                    approved = False
                    issues.append(
                        f"DÉFICIT DE PROTEÍNA (rechazo clínico para ganancia muscular): el plan no "
                        f"alcanza el piso de proteína en {_sd_str}. Cada comida PRINCIPAL (almuerzo y "
                        f"cena) DEBE incluir una fuente animal de alta densidad (pollo, pescado, cerdo, "
                        f"res, huevos, queso) dimensionada en gramos para que cada día sume al menos "
                        f"{int(PROTEIN_FLOOR_HARD_PCT*100)}% del target ({int(_tgt_p)}g). NO dependas solo "
                        f"de leguminosas/almidón en las comidas principales."
                    )
                    severity = _severity_max(severity, "high")
        except Exception as _pf_e:
            logger.warning(f"[P3-PROTEIN-FLOOR] validador falló: {type(_pf_e).__name__}: {_pf_e}")

    # [P3-VARIETY-HARD-GATE · 2026-06-13] Cap de huevo como restricción DURA (era advisory FS5).
    # Si el huevo aparece en > cap + slack comidas, rechaza → retry con directiva. ACOTADO por
    # `should_retry` (entrega en el attempt final, no loop infinito). El lever upstream es el
    # prompt (rotar proteínas ligeras); esto añade presión de retry sobre el sobreuso egregio.
    if VARIETY_HARD_GATE_ENABLED:
        try:
            _vr = plan.get("variety_report") if isinstance(plan, dict) else None
            if isinstance(_vr, dict):
                _egg = int(_vr.get("egg_meals", 0))
                _tot = int(_vr.get("total_meals", 0)) or 12
                _cap = max(3, round(_tot * 0.25))
                if _egg > _cap + VARIETY_HARD_GATE_EGG_SLACK:
                    logger.warning(f"🥚 [P3-VARIETY-HARD-GATE] Huevo en {_egg}/{_tot} comidas "
                                   f"(cap {_cap}+{VARIETY_HARD_GATE_EGG_SLACK}) → rechazo para diversificar")
                    approved = False
                    issues.append(
                        f"SOBREUSO DE HUEVO (rechazo de variedad): el huevo aparece en {_egg} de {_tot} "
                        f"comidas (máximo {_cap}). Reemplaza el huevo en al menos {_egg - _cap} comida(s) "
                        f"por otras proteínas dominicanas (pollo guisado, pescado, atún, sardina, res molida "
                        f"magra, queso de freír, yogur griego, habichuelas) — NO uses huevo como relleno "
                        f"por defecto. Mantén el huevo solo en desayunos/platos donde es el protagonista."
                    )
                    severity = _severity_max(severity, "high")
        except Exception as _vg_e:
            logger.warning(f"[P3-VARIETY-HARD-GATE] validador falló: {type(_vg_e).__name__}: {_vg_e}")

    # [P2-A · 2026-05-07] Coherencia recetas↔lista en mode `block`.
    # `assemble_plan_node` invoca `run_shopping_coherence_guard`; cuando
    # `MEALFIT_SHOPPING_COHERENCE_GUARD=block` detecta divergencias críticas
    # (foods de receta ausentes en la lista o magnitudes con delta > tolerance)
    # setea `plan["_shopping_coherence_block"]` con la lista. Sin este consumer
    # el contrato de "block" era no-op silencioso — el flag se persistía pero
    # nada lo accionaba. Cierre del gap detectado en el re-audit 2026-05-07.
    #
    # Acción modulada por `MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION`
    # (env var, default `reject_minor`). El knob es env-var por preferencia
    # operacional (revertible sin redeploy):
    #   - `reject_minor`  : rechaza como minor → `should_retry` permite retry
    #                       si hay budget. Comportamiento default seguro:
    #                       opera el "block" sin escalar agresivamente.
    #   - `reject_high`   : rechaza como high (regenerable: ningún keyword
    #                       de `_HIGH_SEVERITY_CONTEXTUAL_KEYWORDS` matchea
    #                       "COHERENCIA RECETAS LISTA") → retry forzado por
    #                       `_classify_high_severity`.
    #   - `degrade`       : kill switch — limpia el flag, no-op. Util para
    #                       rollback rápido si `reject_*` produce loops sobre
    #                       planes de calidad límite. Restaura el comportamiento
    #                       previo al fix.
    # Cualquier otro valor → `reject_minor` + warning de knob inválido.
    coherence_block = plan.get("_shopping_coherence_block") or []
    if coherence_block:
        # [P1-2 · 2026-05-08] Knob ahora pasa por `_env_str` para auto-registro
        # en `_KNOBS_REGISTRY`. La validación de choices vive en el helper:
        # un valor inválido → WARNING + fallback a `reject_minor`.
        _block_action = _env_str(
            "MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION",
            "reject_minor",
            choices={"degrade", "reject_minor", "reject_high"},
        )

        # [P3-NEW-C · 2026-05-08] Marcar la última entry de history con la
        # acción que estamos tomando. assemble_plan_node ya creó la entry
        # con `action_taken=None`; aquí la hidratamos con el knob resuelto
        # (reject_minor/reject_high/degrade). Defensivo: si la lista está
        # vacía o el último item no es dict, no fallamos.
        # [P2-2 · 2026-05-08] Si la hidratación falla por estado inesperado
        # (history vacío, último item no-dict, excepción), seteamos
        # `action_taken="hydration_error"` para que el post-mortem
        # distinga bug de flujo normal. La invariante post-condición es:
        # tras review_plan_node, ninguna entry tiene `action_taken=None`.
        try:
            _coh_hist = plan.get("_shopping_coherence_block_history")
            if isinstance(_coh_hist, list) and _coh_hist and isinstance(_coh_hist[-1], dict):
                _coh_hist[-1]["action_taken"] = _block_action
            else:
                logging.warning(
                    f"[COH-BLOCK/HISTORY] estado inesperado al hidratar action_taken: "
                    f"history={type(_coh_hist).__name__} len="
                    f"{len(_coh_hist) if isinstance(_coh_hist, list) else 'n/a'}. "
                    f"block_action={_block_action!r} no se pudo registrar; "
                    f"un nuevo entry de error se añade para preservar invariante."
                )
                # Crear/extender history con un entry de error sintético para
                # mantener la invariante "siempre hay action_taken no-None
                # cuando _shopping_coherence_block estuvo set".
                if not isinstance(_coh_hist, list):
                    _coh_hist = []
                from datetime import datetime as _hydr_dt, timezone as _hydr_tz
                _coh_hist.append({
                    "ts": _hydr_dt.now(_hydr_tz.utc).isoformat(),
                    "attempt": None,
                    "divergence_count": len(coherence_block),
                    "block_set": True,
                    "action_taken": "hydration_error",
                    "hydration_error_reason": "history_missing_or_corrupt",
                })
                plan["_shopping_coherence_block_history"] = _coh_hist[-20:]
        except Exception as _coh_mark_e:
            logging.warning(
                f"[COH-BLOCK/HISTORY] excepción al hidratar action_taken: {_coh_mark_e}. "
                f"Marcando entry sintético con hydration_error."
            )
            try:
                _coh_hist = plan.get("_shopping_coherence_block_history") or []
                if isinstance(_coh_hist, list) and _coh_hist and isinstance(_coh_hist[-1], dict):
                    _coh_hist[-1]["action_taken"] = "hydration_error"
                    _coh_hist[-1]["hydration_error_reason"] = str(_coh_mark_e)[:200]
            except Exception:
                pass

        if _block_action == "degrade":
            plan.pop("_shopping_coherence_block", None)
            logger.info(
                f"🛒 [REVISOR/COH-BLOCK degrade] {len(coherence_block)} divergencia(s) "
                f"toleradas por knob; flag limpiado, plan se entrega como-is."
            )
        else:
            sample = ", ".join(str(d.get("food", "?")) for d in coherence_block[:5])
            msg = (
                f"COHERENCIA RECETAS LISTA: {len(coherence_block)} divergencia(s) "
                f"críticas (foods: {sample}). action={_block_action}."
            )
            approved = False
            issues.append(msg)
            if _block_action == "reject_high":
                severity = _severity_max(severity, "high")
                logger.info(f"🛒 [REVISOR/COH-BLOCK reject_high] {msg} → retry forzado.")
            else:
                severity = _severity_max(severity, "minor")
                logger.info(f"🛒 [REVISOR/COH-BLOCK reject_minor] {msg} → retry si budget permite.")

    # Brechas 1 y 4: Errores deterministas del ensamblador
    skeleton_fidelity_errors = plan.get("_skeleton_fidelity_errors", [])
    coherence_errors = plan.get("_recipe_coherence_errors", [])

    # Separar errores cosméticos (ingrediente listado sin aparecer en instrucciones) de errores estructurales.
    # Los cosméticos se parchean en-place; los estructurales siguen requiriendo regeneración.
    COHERENCE_PATCHABLE_MARKER = "está listado pero no se menciona en las instrucciones"
    # [P1-AUTO-PATCH-FORWARD · 2026-05-21] Marker de la dirección forward del
    # bidirectional check: receta menciona una proteína KEY pero ningún
    # ingrediente es sinónimo. Hasta P1-AUTO-PATCH-FORWARD escalaba a retry; ahora
    # `_auto_patch_recipe_forward_coherence` lo resuelve quirúrgicamente cuando la
    # proteína real está identificable en `ingredients` (cero LLM).
    COHERENCE_FORWARD_PATCHABLE_MARKER = "pero no hay ningún ingrediente equivalente"
    patchable_errors = [e for e in coherence_errors if COHERENCE_PATCHABLE_MARKER in e]
    forward_patchable_errors = [e for e in coherence_errors if COHERENCE_FORWARD_PATCHABLE_MARKER in e]
    structural_coherence_errors = [
        e for e in coherence_errors
        if COHERENCE_PATCHABLE_MARKER not in e and COHERENCE_FORWARD_PATCHABLE_MARKER not in e
    ]

    if patchable_errors:
        n_patched = _auto_patch_ingredient_coherence(plan, patchable_errors)
        logger.info(f"🩹 [AUTO-PATCH] {n_patched}/{len(patchable_errors)} ingredientes huérfanos eliminados de listas (no aparecían en instrucciones).")

    if forward_patchable_errors:
        n_fwd_patched, fwd_unpatched = _auto_patch_recipe_forward_coherence(plan, forward_patchable_errors)
        logger.info(
            f"🩹 [AUTO-PATCH-FORWARD] {n_fwd_patched}/{len(forward_patchable_errors)} "
            f"proteínas huérfanas en receta reescritas a la proteína real de ingredients."
        )
        # Errores que no pudimos patchear (ingredients sin proteína identificable,
        # orphan fuera de mapa seguro) deben seguir el flujo legacy: escalar a
        # structural → severity='minor' → retry. Mejor retry que invención.
        if fwd_unpatched:
            structural_coherence_errors.extend(fwd_unpatched)

    assembly_errors = skeleton_fidelity_errors + structural_coherence_errors
    if assembly_errors:
        logger.error(f"❌ [REVISOR] Errores deterministas de ensamblaje detectados: {assembly_errors}")
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
                    logger.info(f"🔄 [PANTRY GUARD] Rotación implícita detectada (pantry + previous_meals). Validando estricto.")

                val_result = validate_ingredients_against_pantry(all_ingredients, clean_pantry, strict_quantities=False)
                if val_result is not True:
                    approved = False
                    issues.append(val_result)  # val_result es el string de error generado por constants.py
                    # P1-6: max — preservar critical si ya estaba marcado
                    severity = _severity_max(severity, "high")
                    logger.error(f"🚨 [PANTRY GUARD] Validación fallida en Revisor Médico.")
                else:
                    logger.info(f"✅ [PANTRY GUARD] Todos los ingredientes cumplen con la despensa.")

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
                    
                    # [P2-ANTI-REPETITION-TOLERANCE · 2026-06-13] Rechazar SOLO si los
                    # repetidos superan la tolerancia (default 2). 1-2 repetidos vs los
                    # últimos 3 planes es solape natural; no degrada un plan sano. Anchor:
                    # P2-ANTI-REPETITION-TOLERANCE.
                    if len(filtered_repeated) > ANTI_REPETITION_TOLERANCE:
                        approved = False
                        issues.append(
                            f"REPETICIÓN DETECTADA: Los siguientes platos principales ya aparecieron en planes recientes y deben ser reemplazados por alternativas completamente diferentes: {', '.join(repeated_meals)}."
                        )
                        # P1-6: max — preservar critical/high si ya estaba marcado
                        severity = _severity_max(severity, "minor")
                        logger.info(f"🔄 [ANTI-REPETICIÓN] {len(repeated_meals)} platos repetidos detectados (>{ANTI_REPETITION_TOLERANCE} tolerancia): {repeated_meals}")
                    elif filtered_repeated:
                        logger.info(f"✅ [ANTI-REPETICIÓN] {len(filtered_repeated)} repetido(s) tolerado(s) (≤{ANTI_REPETITION_TOLERANCE}): {filtered_repeated}")
                    else:
                        logger.info(f"✅ [ANTI-REPETICIÓN] Sin repeticiones detectadas contra {len(recent_meal_names)} platos recientes.")
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
                            logger.info(f"🔄 [ANTI-REPETICIÓN GUEST] {len(filtered_guest_repeated)} platos repetidos detectados")
        except Exception as e:
            logger.warning(f"⚠️ [ANTI-REPETICIÓN] Error en validación (no bloqueante): {e}")
            
    # 3. Validación de Complejidad (Feedback Loop Enforcement)
    if approved:
        try:
            adherence_hint = form_data.get("_adherence_hint", "")
            if adherence_hint == "low":
                complexity_score = _calculate_complexity_score(plan)
                logger.info(f"📉 [COMPLEXITY GUARD] Adherencia baja. Score de complejidad del plan generado: {complexity_score}/10")
                if complexity_score > 4.5:
                    approved = False
                    issues.append(
                        f"COMPLEJIDAD EXCESIVA: El usuario tiene baja adherencia. El plan actual es demasiado complejo (Score: {complexity_score}/10). Simplifica radicalmente las recetas usando menos ingredientes y menos pasos (máx 3 pasos simples). Evita el horno."
                    )
                    # P1-6: max — preservar critical/high si ya estaba marcado
                    severity = _severity_max(severity, "minor")
                    logger.error(f"🚨 [COMPLEXITY GUARD] Plan rechazado por ser muy complejo para el nivel actual del usuario.")
                else:
                    logger.info(f"✅ [COMPLEXITY GUARD] Plan validado. Score: {complexity_score}/10 (Adecuado para baja adherencia).")
        except Exception as e:
            logger.warning(f"⚠️ [COMPLEXITY GUARD] Error en validación (no bloqueante): {e}")

    _emit_progress(state, "metric", {
        "node": "review_plan",
        "duration_ms": int(duration * 1000) if 'duration' in locals() else 0,
        "retries": state.get("attempt", 1) - 1,
        "tokens_estimated": 0,
        "confidence": 1.0 if approved else 0.0,
        "metadata": {"issues": len(issues)}
    })
    
    if approved:
        logger.info(f"✅ [REVISOR MÉDICO] Plan APROBADO en {duration}s ✅")
        # [P0-PIPE-1] Approved siempre es el mejor posible (rank 0). Snapshot
        # incondicional. `copy.deepcopy` ya importado a nivel módulo.
        _approved_attempt = state.get("attempt", 1)
        _approved_plan_snap = (
            copy.deepcopy(state.get("plan_result"))
            if isinstance(state.get("plan_result"), dict) else None
        )
        return {
            "review_passed": True,
            "review_feedback": "",
            "rejection_reasons": [],
            "_best_attempt_plan": _approved_plan_snap,
            "_best_attempt_severity": "approved",
            "_best_attempt_reasons": [],
            "_best_attempt_review_passed": True,
            "_best_attempt_number": _approved_attempt,
        }
    else:
        feedback = "\n".join([f"• {issue}" for issue in issues])
        logger.error(f"❌ [REVISOR MÉDICO] Plan RECHAZADO en {duration}s (Severidad: {severity})")
        logger.info(f"   Problemas encontrados:")
        for issue in issues:
            logger.error(f"   ❌ {issue}")
            
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
                    logger.info(f"💾 [META-LEARNING] Patrones de rechazo persistidos en DB.")
            except Exception as e:
                logger.warning(f"⚠️ [META-LEARNING] Error persistiendo patrones de rechazo: {e}")
                
        # Brecha 2: Determinar días afectados para Surgical Fix
        days_in_chunk = int(form_data.get("_days_to_generate", 3))
        final_affected_days = set(llm_affected_days) if 'llm_affected_days' in locals() else set()
        for issue in issues:
            # Capturar explícitamente menciones a días (soporta chunks > 3 días)
            for day_num_str in _re.findall(r'(?i)D[íi]a[s]?\s*(\d+)', issue):
                day_int = int(day_num_str)
                if 1 <= day_int <= days_in_chunk:
                    final_affected_days.add(day_int)
                
        # [P3-CONDITION-RULES · 2026-06-14] DM2: NO degradar a fallback matemático por preocupación
        # glucémica. El revisor LLM escala miel/plátano-grande a 'critical' → `_apply_critical_review_
        # guardrails` dispara un fallback matemático que PIERDE el plan real + la fibra ADA (validado
        # en vivo). Para diabéticos, salvo allergen/schema/renal (criticals deterministas de seguridad
        # que NO se tocan), la preocupación glucémica es de CALIDAD: la bajamos a 'high' → `should_retry`
        # da un retry (con la directiva glucémica) y entrega el plan REAL con banner ámbar + el gate
        # profesional, en vez de un plan de contingencia genérico. Knob MEALFIT_DM2_GLYCEMIC_SOFT_REJECT.
        if (DM2_GLYCEMIC_SOFT_REJECT and CONDITION_RULES_ENABLED and severity == "critical"
                and not plan.get("_schema_invalid") and not _had_allergen_critical
                and _is_diabetes_condition(form_data)):
            logger.warning("🩸 [P3-CONDITION-RULES] DM2: rechazo glucémico CRÍTICO degradado a 'high' "
                           "→ entrega el plan real (con fibra ADA) + advertencia, no fallback matemático.")
            severity = "high"

        result = {
            "review_passed": False,
            "review_feedback": feedback,
            "rejection_reasons": issues,
            "_rejection_severity": severity,
            "_affected_days": list(final_affected_days)
        }

        # [P0-PIPE-1] Promover el intento actual a `_best_attempt_*` SOLO si:
        #   (a) no hay best previo (primer rechazo registrado), O
        #   (b) el current tiene MENOR rank de severidad que el best previo
        #       (rank: approved=0, minor=1, high=2, critical=3 — menor = mejor).
        # Approved nunca pasa por aquí; se promueve incondicional en el branch
        # `if approved` arriba.
        prior_best_passed = bool(state.get("_best_attempt_review_passed"))
        prior_best_severity = state.get("_best_attempt_severity")
        prior_has_best = isinstance(state.get("_best_attempt_plan"), dict) and \
            bool(state.get("_best_attempt_plan"))

        current_rank = _attempt_quality_rank(False, severity)
        prior_rank = _attempt_quality_rank(prior_best_passed, prior_best_severity) \
            if prior_has_best else 99  # 99 = "peor que cualquier real" → siempre promover

        if current_rank < prior_rank:
            _attempt_n = state.get("attempt", 1)
            _plan_snap = (
                copy.deepcopy(state.get("plan_result"))
                if isinstance(state.get("plan_result"), dict) else None
            )
            if _plan_snap:
                result["_best_attempt_plan"] = _plan_snap
                result["_best_attempt_severity"] = severity
                result["_best_attempt_reasons"] = list(issues)
                result["_best_attempt_review_passed"] = False
                result["_best_attempt_number"] = _attempt_n
                logger.info(
                    f"📌 [P0-PIPE-1] Snapshot intento #{_attempt_n} promovido "
                    f"a best (severity={severity!r}, rank={current_rank}; "
                    f"best previo: "
                    f"{'ninguno' if not prior_has_best else f'severity={prior_best_severity!r}, rank={prior_rank}'})."
                )

        return result


# ============================================================
# DECISIÓN CONDICIONAL: ¿Repetir o finalizar?
# ============================================================
def _persist_gemini_spend_cap_alert(user_id: Optional[str]) -> None:
    """[P1-SPEND-CAP-ALERT · 2026-05-28] Emite `system_alerts.gemini_spend_cap_exceeded`
    cuando el pipeline detecta el 429 "spending cap" de Gemini (cap mensual de
    AI Studio agotado). A diferencia de un 429 transitorio (rate-limit, se libera
    en segundos), el spending cap queda activo hasta que el operador suba/quite el
    cap (https://ai.studio/spend) O ruede el ciclo de billing — reintentar NO ayuda.

    Por qué existe (incidente 2026-05-28, plan user bf6f1383): el pipeline cayó
    con `429 RESOURCE_EXHAUSTED: monthly spending cap` pero el único rastro quedó
    en logs; el usuario veía "IA temporalmente saturada, intenta en 1-2 min"
    (falso) y un operador solo se entera leyendo logs. Esta alert escala la
    condición a `system_alerts` para que sea visible en dashboards SRE.

    Best-effort (no propaga; el pipeline ya está degradando). Idempotente:
    `alert_key` GLOBAL (la condición es project-wide, no per-user) → ON CONFLICT
    bumpea `triggered_at`. Modelo de resolution: Manual (operador cierra tras
    subir el cap).

    tooltip-anchor: P1-SPEND-CAP-ALERT — row `gemini_spend_cap_exceeded` en
    backend/docs/system_alerts_resolution_table.md.
    """
    try:
        from db_core import execute_sql_write
        import json as _json
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'gemini_spend_cap', 'critical', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                resolved_at = NULL
            """,
            (
                "gemini_spend_cap_exceeded",
                "Gemini spending cap agotado — generacion de planes caida",
                (
                    "El proyecto de Gemini supero su spending cap mensual (429 "
                    "RESOURCE_EXHAUSTED). TODAS las generaciones de plan fallaran "
                    "hasta subir/quitar el cap en https://ai.studio/spend o hasta "
                    "el reset del ciclo de billing. Reintentar NO ayuda. Resolver "
                    "esta alerta tras subir el cap."
                ),
                _json.dumps(
                    {"provider": "gemini", "remediation_url": "https://ai.studio/spend"},
                    ensure_ascii=False,
                ),
                _json.dumps([str(user_id)] if user_id else [], ensure_ascii=False),
            ),
        )
        logger.warning(
            "🛑 [P1-SPEND-CAP-ALERT] system_alert `gemini_spend_cap_exceeded` "
            "emitido — generacion caida hasta subir el cap en ai.studio/spend."
        )
    except Exception as _e:
        logger.warning(
            f"[P1-SPEND-CAP-ALERT] No se pudo persistir gemini_spend_cap_exceeded: {_e!r}"
        )


def _emit_plan_quality_degraded_alert(
    state: "PlanState",
    exit_reason: str,
    severity: Optional[str] = None,
) -> None:
    """[P1-NEW-3 · 2026-05-11] Emite `system_alerts.plan_quality_degraded`
    cuando `should_retry` decide entregar un plan SIN `review_passed=True`.

    Por qué existe (audit 2026-05-11):
      Antes, cuando `should_retry` retornaba "end" con `review_passed=False`
      (rechazo crítico, high-contextual, attempts agotados, budget agotado),
      el plan se entregaba al usuario con disclaimer pero NO había señal
      a SRE para detectar el patrón. El `_shopping_coherence_block_history`
      registra el último `action_taken` (incluyendo `reject_minor`/
      `reject_high`) pero solo dentro del plan_data — no escala como alert.

      Resultado: usuarios podían recibir N planes degradados consecutivos
      sin que nadie supiera, hasta que abrieran un ticket. Esta alert
      cierra ese gap: cada plan degradado emite UN row idempotente
      (upsert por alert_key=`plan_quality_degraded:<user_id>:<plan_id>`),
      visible en dashboards SRE.

    Conservador:
      - Best-effort: cualquier fallo del emit NO debe abortar el routing
        de `should_retry`. La alert es observacional, no crítica.
      - Idempotente: ON CONFLICT (alert_key) bumpea `triggered_at` +
        metadata sin duplicar rows.
      - severity DB = 'warning' (no 'critical' — el plan se entrega,
        solo degradado).
      - Modelo de resolution: Auto (implicit) — re-emite si patrón
        persiste; un plan regenerado exitosamente NO cierra la alert
        automáticamente (el operador debe verificar manualmente).
    """
    try:
        form_data = state.get("form_data") or {}
        user_id = form_data.get("user_id") or form_data.get("session_id") or "unknown"
        plan_result = state.get("plan_result") or {}
        # [P1-NEW-9 · 2026-05-11] Caller-injected fallback ANTES del sentinel
        # final. Cuando el caller es JIT week-2 (proactive_agent) u otro
        # flujo que opera sobre un meal_plan PRE-EXISTENTE (no inserta plan
        # nuevo), plan_result no trae `id`/`plan_id` y antes el alert
        # colapsaba al sentinel — útil solo para el path /generate-plan
        # inicial, no para extensiones. Ahora el caller inyecta el plan_id
        # real en form_data y SRE puede correlacionar al plan extendido.
        plan_id = (
            plan_result.get("id")
            or plan_result.get("plan_id")
            or form_data.get("_caller_target_plan_id")
            or "no_plan_id"
        )
        alert_key = f"plan_quality_degraded:{user_id}:{plan_id}"
        rejection_reasons = state.get("rejection_reasons") or []
        attempts = state.get("attempt", 1)
        # [P1-NEW-9] Caller context para que SRE filtre alerts por origen
        # (initial generate vs jit_week2 vs futuros flows). Default
        # 'initial_generate' preserva el contrato histórico.
        caller_context = form_data.get("_caller_context") or "initial_generate"

        metadata = {
            "exit_reason": exit_reason,  # 'critical' | 'high_contextual' | 'max_attempts' | 'invalid_pipeline_start' | 'budget_exhausted'
            "rejection_severity": severity,
            "attempts": attempts,
            "review_passed": bool(state.get("review_passed")),
            "plan_id": plan_id,
            "user_id": user_id,
            "top_rejection_reasons": rejection_reasons[:5],
            "caller_context": caller_context,  # P1-NEW-9
        }
        message = (
            f"Plan entregado al usuario {user_id} con calidad degradada "
            f"(exit_reason={exit_reason}, severity={severity}, "
            f"attempts={attempts}). Revisar `rejection_reasons` en metadata "
            f"para diagnosticar patrón."
        )

        from db_core import execute_sql_write, execute_sql_query
        import json as _json

        # [P3-NEW-9 · 2026-05-11] Coalesce inter-thread vía advisory lock
        # en `app_kv_store`. Pre-fix: dos requests concurrentes del MISMO
        # user con ambas terminando en `should_retry "end"` en el mismo
        # segundo emitían 2 INSERTs paralelos. La DB deduplicaba el row
        # (ON CONFLICT) pero webhooks downstream (Sentry/Slack si configurados)
        # podían disparar N veces — observability noise.
        #
        # Mecanismo: UPSERT condicional con WHERE updated_at < NOW() - 60s.
        # Si la fila no existe → INSERT → RETURNING → procede.
        # Si la fila existe y está stale (>60s) → UPDATE → RETURNING → procede.
        # Si la fila existe y está fresh (<60s) → WHERE falla → no RETURNING → skip.
        # 60s cubre el racing window de should_retry sin perder señal legítima
        # de regeneraciones espaciadas. NO requiere cron de cleanup — el
        # natural churn de keys (millones de plans, una key por evento) cabe
        # en KV; un sweep separado puede añadirse si la tabla crece.
        emit_lock_key = f"plan_quality_emit_lock:{user_id}:{plan_id}"
        try:
            lock_row = execute_sql_query(
                """
                INSERT INTO app_kv_store (key, value, updated_at)
                VALUES (%s, '{}'::jsonb, NOW())
                ON CONFLICT (key) DO UPDATE
                SET updated_at = NOW(), value = EXCLUDED.value
                WHERE app_kv_store.updated_at < NOW() - INTERVAL '60 seconds'
                RETURNING key
                """,
                (emit_lock_key,),
                fetch_one=True,
            )
        except Exception as _lock_err:
            # Best-effort: si el KV falla (DB down), caemos al comportamiento
            # pre-P3-NEW-9 (sin coalesce). Mejor emit duplicado que perder
            # señal por bug del lock.
            logger.debug(
                f"[P3-NEW-9] dedup lock falló (best-effort, sigo emit): {_lock_err}"
            )
            lock_row = {"key": emit_lock_key}

        if not lock_row:
            # Otro thread ya emitió en los últimos 60s — skip silencioso.
            # El emit duplicado a DB es idempotente (ON CONFLICT bumpea
            # triggered_at), pero saltar evita webhooks downstream N-firing.
            logger.debug(
                f"[P3-NEW-9] plan_quality_degraded emit coalesced "
                f"(lock fresh) user={user_id} plan={plan_id} exit={exit_reason}"
            )
            return

        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'plan_quality', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                message = EXCLUDED.message,
                resolved_at = NULL
            """,
            (
                alert_key,
                f"Plan degradado entregado: {exit_reason}",
                message,
                _json.dumps(metadata, ensure_ascii=False),
                _json.dumps([str(user_id)] if user_id != "unknown" else []),
            ),
        )
        logger.warning(
            f"[P1-NEW-3] plan_quality_degraded alert emitida user={user_id} "
            f"plan={plan_id} exit={exit_reason} severity={severity}"
        )
    except Exception as e:
        # Best-effort: NO escalar fallo del alert al routing del pipeline.
        logger.debug(
            f"[P1-NEW-3] plan_quality_degraded alert emit falló (best-effort): {e}"
        )


def _mark_plan_result_quality_degraded(state: PlanState, reason: str, severity: str) -> None:
    """[P1-PROD-AUDIT-BUNDLE · 2026-05-28] Setea el flag user-visible
    `plan_data._quality_degraded` en `state['plan_result']` para que el
    frontend muestre el banner "la IA no pudo generar un plan óptimo".

    Pre-fix: solo la rama `max_attempts` de `should_retry` seteaba este flag
    (P1-LOW-SIGNAL-FALLBACK). Las otras ramas "end" que entregaban un plan con
    `review_passed=False` (`high_contextual`, `invalid_pipeline_start`,
    `budget_exhausted`) emitían SOLO el alert SRE → el usuario recibía un plan
    con fallo creyéndolo normal (misma clase que P0-DEAD-LETTER-USER-NOTIFY).
    Este helper centraliza el flag para que TODAS esas ramas notifiquen.

    Best-effort: un fallo aquí no debe abortar la entrega del plan.
    Tooltip-anchor: P1-QUALITY-DEGRADED-ALL-BRANCHES.
    """
    try:
        _pr = state.get("plan_result")
        if isinstance(_pr, dict):
            _pr["_quality_degraded"] = True
            _pr["_quality_degraded_reason"] = reason
            _pr["_quality_degraded_severity"] = severity or "minor"
            _pr["_quality_degraded_attempts"] = int(state.get("attempt", 1))
    except Exception as _flag_e:
        logger.warning(
            f"[P1-QUALITY-DEGRADED-ALL-BRANCHES] No pude setear flag "
            f"_quality_degraded en plan_result (reason={reason}): {_flag_e}"
        )


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
        # [P5-MARKER-APPROVED-1] Antes de despachar al usuario, verificar si
        # algún día tiene `_critique_unresolved` (self-critique falló para
        # ese día por timeout / CB / error). El reviewer médico aprueba
        # porque su lente es médico, pero markers de slot coherence quedan
        # silenciados. Si los hay y NO hemos intentado el surgical regen
        # todavía, enrutamos al gate. Una sola pasada por gate (flag).
        if not state.get("_marker_regen_attempted", False):
            marker_days = _collect_unresolved_marker_days(state.get("plan_result"))
            if marker_days:
                logger.info(
                    f"🩹 [ORQUESTADOR] Revisión aprobada PERO {len(marker_days)} "
                    f"día(s) con `_critique_unresolved` (días {marker_days}) → "
                    f"surgical regen antes de enviar al usuario."
                )
                return "marker_regen"
        logger.info("✅ [ORQUESTADOR] Revisión aprobada → Enviando al usuario.")
        return "end"

    # [P0-5] `dict.get(k, default)` devuelve `None` cuando la key existe con
    # valor None (NO el default). El `initial_state` ahora setea
    # `_rejection_severity="minor"` por defecto (P0-5), pero un nodo
    # intermedio podría escribir `None` explícitamente (excepción no
    # capturada en `review_plan_node`, schema_invalid silencioso, refactor
    # futuro). Si eso ocurre, `severity == "critical"` y `severity == "high"`
    # ambos fallan, cae a la rama "retry" sin honrar la severidad → loop
    # silencioso. La normalización con `or "minor"` garantiza que cualquier
    # falsy (None, "", 0) se trate como "minor" (recuperable, retry permitido).
    raw_severity = state.get("_rejection_severity")
    if raw_severity is None:
        # Invariant: si llegamos aquí sin severity poblada, hay un bug
        # upstream — loggear para que SRE detecte el estado inconsistente.
        logger.error(
            "[ORQUESTADOR][INVARIANT] _rejection_severity=None pero "
            "review_passed=False — review_plan_node lanzó excepción sin "
            "setear severidad? attempt=%s, rejection_reasons=%s",
            state.get("attempt", 1),
            (state.get("rejection_reasons") or [])[:3],
        )
    severity = raw_severity or "minor"

    # Brecha 6 + P0-2: Retry por Severidad
    # 'critical' = peligro médico (alergia/condición). Abortar y delegar a P0-1 guardrail
    # para entregar fallback matemático con disclaimer.
    if severity == "critical":
        logger.error("🚨 [ORQUESTADOR] Rechazo CRÍTICO → No tiene sentido reintentar con el mismo contexto. Abortando temprano.")
        _emit_plan_quality_degraded_alert(state, exit_reason="critical", severity=severity)
        return "end"

    # [P1-RETRY-CLASSIFY] Clasificación fina de HIGH:
    #   - 'contextual' (despensa, alergia, condición médica) → no retry, las
    #     restricciones del usuario no cambian entre intentos.
    #   - 'regenerable' (skeleton fidelity, repetición, variedad, coherence)
    #     → SÍ retry, el LLM puede hacerlo mejor en intento #2.
    #
    # Pre-fix: TODO HIGH abortaba sin retry, conflando ambos casos. Caso real
    # 2026-05-05 01:49: planner asignó ['Atún, Lentejas, Huevos'] pero el LLM
    # ignoró y puso pavo procesado en casi todas las comidas → HIGH "skeleton
    # fidelity + repetición pavo + falta variedad" → entregado roto al usuario
    # con disclaimer (holistic 0.850), cuando un retry probablemente lo habría
    # arreglado.
    if severity == "high":
        _retry_class = _classify_high_severity(state.get("rejection_reasons") or [])
        if _retry_class == "contextual":
            logger.error(
                f"🛑 [ORQUESTADOR] Rechazo HIGH (contextual: despensa/alergia/"
                f"condición — no-recuperable por retry) → Abortando y entregando "
                f"plan marcado. Razones: "
                f"{(state.get('rejection_reasons') or [])[:2]}"
            )
            _emit_plan_quality_degraded_alert(state, exit_reason="high_contextual", severity=severity)
            _mark_plan_result_quality_degraded(state, reason="high_contextual", severity=severity)
            return "end"
        # 'regenerable' → cae al check de attempts/budget para decidir retry.
        # Si pasa los gates, ejecutará el retry como cualquier minor.
        logger.info(
            f"🔁 [ORQUESTADOR] Rechazo HIGH (regenerable: fallo de calidad del LLM, "
            f"no contextual) → Evaluando retry. Razones: "
            f"{(state.get('rejection_reasons') or [])[:2]}"
        )

    # P1-X4: default `attempt=1` para alinear con el resto del archivo
    # (initial_state lo setea a 1; otros nodos leen `state.get("attempt", 1)`).
    # En el path normal no cambia el comportamiento — `initial_state` siempre
    # incluye attempt=1 — pero cierra una trampa de mantenimiento (tests
    # directos del nodo con state vacío y refactors futuros).
    if state.get("attempt", 1) >= MAX_ATTEMPTS:
        if not state.get("review_passed", False):
            logger.error(f"🚨 [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada → Tolerando y enviando mejor versión disponible.")
            _emit_plan_quality_degraded_alert(state, exit_reason="max_attempts", severity=severity)
            # [P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Flag user-visible para que el
            # frontend muestre un banner explícito ("La IA no pudo generar un plan
            # óptimo tras 3 intentos. Revisalo y, si algo no cuadra, usa Cambiar
            # Plato"). Pre-fix: el alert sólo iba a `system_alerts` (SRE-visible)
            # — el usuario recibía el plan degradado SIN saber que el sistema
            # se había "rendido". Ahora `plan_data._quality_degraded=True` +
            # `_quality_degraded_reason` viajan hasta el cliente vía SSE/REST.
            try:
                _pr = state.get("plan_result")
                if isinstance(_pr, dict):
                    _pr["_quality_degraded"] = True
                    _pr["_quality_degraded_reason"] = "max_attempts"
                    _pr["_quality_degraded_severity"] = severity or "minor"
                    _pr["_quality_degraded_attempts"] = int(state.get("attempt", 1))
            except Exception as _flag_e:
                # Best-effort — un fallo aquí no debe abortar la entrega del plan.
                logger.warning(f"[P1-LOW-SIGNAL-FALLBACK] No pude setear flag _quality_degraded en plan_result: {_flag_e}")
            return "end"
        logger.warning(f"⚠️  [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado → Enviando mejor versión disponible.")
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
        logger.warning(
            f"⚠️  [ORQUESTADOR] pipeline_start={start!r} (tipo {type(start).__name__}) "
            f"inválido — no puedo calcular budget de retry. Tratando como "
            f"agotado para evitar bypass del guard de timeout. Preservando "
            f"mejor versión disponible."
        )
        if not state.get("review_passed", False):
            _emit_plan_quality_degraded_alert(state, exit_reason="invalid_pipeline_start", severity=severity)
            _mark_plan_result_quality_degraded(state, reason="invalid_pipeline_start", severity=severity)
        return "end"

    elapsed = time.time() - start
    remaining = GLOBAL_TIMEOUT - elapsed
    budget_threshold = MIN_RETRY_BUDGET_SECONDS + SAFETY_MARGIN + HEDGE_DELTA
    if remaining < budget_threshold:
        logger.info(
            f"⏰ [ORQUESTADOR] Sin presupuesto para retry "
            f"({remaining:.0f}s restantes < {budget_threshold:.0f}s mínimo: "
            f"{MIN_RETRY_BUDGET_SECONDS}s generación + {SAFETY_MARGIN}s "
            f"margen post-retry + {HEDGE_DELTA:.0f}s cobertura hedging). "
            f"Preservando mejor versión disponible."
        )
        if not state.get("review_passed", False):
            _emit_plan_quality_degraded_alert(state, exit_reason="budget_exhausted", severity=severity)
            _mark_plan_result_quality_degraded(state, reason="budget_exhausted", severity=severity)
        return "end"

    logger.info("🔄 [ORQUESTADOR] Revisión fallida → Regenerando plan con correcciones...")
    return "retry"


# ============================================================
# NODO 0: REFLEXIÓN META-LEARNING
# ============================================================
# P1-10: `pydantic.BaseModel` y `Field` ya importados al inicio del módulo
# (línea ~13); no reimportar.

class ReflectionResult(BaseModel):
    reflection: str = Field(description="Diagnóstico de la causa raíz en una oración.")

@_node_label("meta_learning")
async def reflection_node(state: PlanState) -> dict:
    """Analiza POR QUÉ el ciclo anterior tuvo bajo rendimiento (o alto rendimiento)."""
    form_data = state["form_data"]
    user_id = form_data.get("user_id") or form_data.get("session_id", "guest")
    
    # Solo reflexionamos si hay métricas de un ciclo previo
    quality_score = form_data.get("_previous_plan_quality")
    if quality_score is None:
        return {} # Skip, es usuario nuevo o no hay datos recientes
        
    # [P1-CHUNK-3] La clave que el feedback-loop ESCRIBE es `_meal_level_adherence`
    # (cron_tasks.py:14815 + 15445). El read legacy `_meal_adherence` (singular) era
    # una clave muerta: jamás se escribía, así que `reflection_node` siempre veía el
    # default y el meta-aprendizaje nunca incorporaba la adherencia real por comida.
    meal_adherence = form_data.get("_meal_level_adherence", "Sin datos granulares")
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
        logger.info(f"⚡ [CACHE HIT] Reutilizando reflexión anterior para el score ({quality_score}).")
        return cached_val

    logger.info(f"🤔 [META-LEARNING] Reflexionando sobre el plan anterior (Quality: {quality_score})...")
    start_time = time.time()
    
    try:
        # P1-Q3: capturar modelo del reflector para CB per-modelo.
        # [P3-COST-CUT-AUX · 2026-05-22] Usar helper `_meta_learning_model_name()`
        # (knob `MEALFIT_META_LEARNING_MODEL`, default `gemini-3.1-flash-lite`).
        # El reflector emite UNA oración con structured output — tarea simple,
        # lite cubre. Rollback: setear el knob a flash full.
        _reflector_model = _meta_learning_model_name()
        _reflector_cb = _get_circuit_breaker(_reflector_model)
        reflector_llm = ChatDeepSeek(
            model=_reflector_model,
            temperature=0.2,
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
            await _record_cb_failure_unless_transient(_reflector_cb, e)  # P1-Q3 · P1-ORCH-1/2
            raise e
        reflection_text = result.reflection
        logger.info(f"💡 [META-LEARNING] Diagnóstico: {reflection_text}")
        
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
                logger.info(f"💾 [META-LEARNING] Reflexión guardada en el perfil de {user_id}")
        
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
        logger.warning(f"⚠️ [META-LEARNING] Error en nodo de reflexión: {e}")
        return {}

# ============================================================
# NODO 0: PRE-FLIGHT OPTIMIZATION (GAP D)
# ============================================================
@_node_label("preflight")
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
        # nodos si la DB tenía réplicas de lectura.
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
                logger.info(f"🧊 [PREFLIGHT] {len(score_history)} entradas en pipeline_score_history "
                      f"pero todas con antigüedad >{PIPELINE_SCORE_FRESHNESS_DAYS}d. Ignorando para auto-ajuste.")
            else:
                last_score = hp.get("last_pipeline_score")  # legacy
                if last_score is not None:
                    score_source = "last_pipeline_score (legacy, sin timestamp)"

            if last_score is not None:
                logger.info(f"🧠 [META-LEARNING] score detectado: {last_score} (fuente: {score_source})")
                if last_score < 0.65:
                    logger.info(f"🔧 [PREFLIGHT] Score histórico bajo ({last_score}). Activando auto_simplify y drift_alert para maximizar estabilidad.")
                    new_form_data["_auto_simplify"] = True
                    new_form_data["_pipeline_drift_alert"] = True
                    auto_adjusted = True
                elif last_score > 0.90:
                    logger.info(f"✨ [PREFLIGHT] Score histórico alto ({last_score}). Permitiendo mayor libertad creativa.")
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
                        logger.info(f"📈 [PREFLIGHT] Auto-ajuste por métricas: {int(avg_rejections*100)}% rechazos, {int(avg_critique_corrections*100)}% correcciones.")
                        new_form_data["_pipeline_drift_alert"] = True
                        if avg_rejections > 0.6:
                            new_form_data["_auto_simplify"] = True
                        auto_adjusted = True

    except Exception as e:
        # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info y wording
        # ajustado: meta-learning preflight es BEST-EFFORT (si no se puede
        # leer historial, el pipeline usa defaults seguros). Pre-fix el
        # mensaje empezaba con "Error" lo cual confundía al user que veía
        # los logs durante la generación pensando que el plan fallaba.
        logger.info(f"[PREFLIGHT] meta-learning sin datos previos, usando defaults (best-effort): {e}")
        
    return {"form_data": new_form_data} if auto_adjusted else {}

# ============================================================
# CONSTRUCTOR DEL GRAFO
# ============================================================
@_node_label("retry_reflection")
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
        # [P5-MARKER-APPROVED-1] Reset del flag para que el nuevo attempt
        # tenga su propia oportunidad de surgical regen post-approval si
        # el self-critique vuelve a dejar markers. Sin reset, attempt #2
        # aprobado-con-markers iría directo a "end" porque arrastraría
        # el flag del attempt #1.
        "_marker_regen_attempted": False,
    }
    if reasons:
        directive = f"El plan anterior fue RECHAZADO por: {'; '.join(reasons)}. MUTA DRÁSTICAMENTE la estrategia."
        logger.info(f"🔄 [RETRY REFLECTION] Intento {attempt}. Directiva inyectada: {directive}")
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


# ============================================================
# [P2-ORCH-2 · 2026-05-28] Threshold del semantic cache como knob + telemetría.
# ------------------------------------------------------------
# El threshold de similitud coseno estaba hardcoded a 0.98 en el callsite — el
# parámetro de COSTE más impactante del pipeline (un hit ahorra 30-90s + dólares
# de generación) NO era tuneable sin redeploy y sin telemetría no se podía saber
# si 0.98 estaba matando el hit-rate. Ahora es un knob clamp + stats por outcome
# (hit / anti_repetition_reject / miss) expuestos vía
# `get_semantic_cache_stats_snapshot()` para medir y bajar el umbral empíricamente.
# Tooltip-anchor: P2-ORCH-2.
# ============================================================
SEMANTIC_CACHE_COSINE_THRESHOLD = _env_float(
    "MEALFIT_SEMANTIC_CACHE_COSINE_THRESHOLD", 0.98,
    validator=lambda v: 0.5 <= v <= 0.999,
)
_SEMANTIC_CACHE_STATS = {"hit": 0, "miss": 0, "anti_repetition_reject": 0}
_SEMANTIC_CACHE_STATS_LOCK = threading.Lock()


def _inc_semantic_cache_stat(kind: str, n: int = 1) -> None:
    """[P2-ORCH-2] Incremento best-effort de los contadores del semantic cache."""
    try:
        with _SEMANTIC_CACHE_STATS_LOCK:
            _SEMANTIC_CACHE_STATS[kind] = _SEMANTIC_CACHE_STATS.get(kind, 0) + n
    except Exception:
        pass


def get_semantic_cache_stats_snapshot() -> dict:
    """[P2-ORCH-2] Snapshot de los contadores del semantic cache (hit-rate).
    Consumible por endpoints de observabilidad para tunear el threshold."""
    with _SEMANTIC_CACHE_STATS_LOCK:
        snap = dict(_SEMANTIC_CACHE_STATS)
    total = snap.get("hit", 0) + snap.get("miss", 0) + snap.get("anti_repetition_reject", 0)
    snap["total"] = total
    snap["hit_rate"] = round(snap.get("hit", 0) / total, 4) if total else 0.0
    snap["threshold"] = SEMANTIC_CACHE_COSINE_THRESHOLD
    return snap


@_node_label("semantic_cache_check")
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

    # [P1-25] Attempt-guard: si el form_data carga señales de que ESTA
    # invocación es un reintento posterior a un rechazo, bypasear el cache.
    # Sin esto, el wrapper externo `_validate_pantry_and_retry_pipeline`
    # (cron_tasks.py:4979/5042) re-invoca `run_plan_pipeline` con
    # `_pantry_correction` no-vacío tras un fallo de pantry, y este nodo
    # puede devolver el MISMO plan (cualquier candidato compatible por
    # similitud de embedding del perfil — el cache no sabe que el plan ya
    # fue rechazado por pantry hace segundos). Resultado: retry no-op +
    # rápida degradación a fallback matemático.
    #
    # Señales reconocidas (cualquiera truthy → bypass):
    #   - `_pantry_correction`: string con la violación que provocó el retry.
    #     `_validate_pantry_and_retry_pipeline` lo escribe antes de re-invocar.
    #   - `_drift_retries`: contador de reintentos por pantry drift
    #     (`cron_tasks.py:17280`). >0 indica al menos un fallo previo.
    #
    # Telemetría: log warning per-bypass para que `cache_hit_rate` no caiga
    # silenciosamente en producción si los retries proliferan (señal de
    # problema upstream — pantry validation flaky, drift detector mal
    # calibrado, etc.).
    _p125_bypass_reasons: list = []
    if actual_form_data.get("_pantry_correction"):
        _p125_bypass_reasons.append("_pantry_correction set")
    try:
        _drift_n = int(actual_form_data.get("_drift_retries") or 0)
    except (TypeError, ValueError):
        _drift_n = 0
    if _drift_n > 0:
        _p125_bypass_reasons.append(f"_drift_retries={_drift_n}")
    if _p125_bypass_reasons:
        _uid = actual_form_data.get("user_id") or "unknown"
        logger.warning(
            f"🟠 [P1-25] Cache bypass para user_id={_uid} por retry "
            f"signals: {', '.join(_p125_bypass_reasons)}. El intento previo "
            f"fue rechazado; servir un cache hit re-introduciría el mismo "
            f"plan rechazado."
        )
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
        # [P2-ORCH-2] Threshold via knob (era 0.98 hardcoded). Loguear la mejor
        # similitud candidata permite tunear el umbral empíricamente: si vemos
        # candidatos consistentemente en ~0.95 rechazados por 0.98, bajar el knob.
        similar_plans = await _adb(search_similar_plan, profile_embedding,
                                   SEMANTIC_CACHE_COSINE_THRESHOLD, 10)
        if similar_plans:
            try:
                _best_sim = max((p.get("similarity", 0) or 0) for p in similar_plans)
                logger.info(
                    f"🔎 [SEMANTIC CACHE] {len(similar_plans)} candidato(s) "
                    f"≥ threshold {SEMANTIC_CACHE_COSINE_THRESHOLD:.3f} "
                    f"(mejor similitud: {_best_sim:.4f})."
                )
            except Exception:
                pass

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
                    logger.info(f"🗑️ [SEMANTIC CACHE] P0-A3: plan descartado por schema incompatible "
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
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por tamaño de chunk "
                          f"({cached_days_count}d ≠ {expected_days}d esperado).")
                    continue

                _cached_form_for_offset = cand_data.get("form_data") or \
                    cand_data.get("metadata", {}).get("form_data", {}) or {}
                try:
                    cached_offset = int(_cached_form_for_offset.get("_days_offset") or 0)
                except (TypeError, ValueError):
                    cached_offset = 0
                if cached_offset != expected_offset:
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por offset "
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
                            logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por antigüedad ({days_old} días > 30).")
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
                        logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de target "
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
                        logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de "
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
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por form_data ausente o inválido "
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
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de alergias post-hoc.")
                    continue
                if curr_medical != cache_medical:
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de condiciones médicas post-hoc.")
                    continue
                if curr_diet != cache_diet:
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de tipo de dieta.")
                    continue
                if curr_dislikes != cache_dislikes:
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de rechazos/dislikes.")
                    continue
                if curr_cooking != cache_cooking:
                    logger.info(f"🗑️ [SEMANTIC CACHE] P1-ORQ-7: Plan descartado por "
                          f"cambio de cookingTime (cached={cache_cooking!r}, "
                          f"current={curr_cooking!r}). Recetas en caché "
                          f"probablemente no encajan con la nueva capacidad de cocina.")
                    continue
                if curr_budget != cache_budget:
                    logger.info(f"🗑️ [SEMANTIC CACHE] P1-ORQ-7: Plan descartado por "
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
                    logger.info(f"🗑️ [SEMANTIC CACHE] Plan descartado por {_pantry_discard} "
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
                            logger.warning(f"⚠️ [SEMANTIC CACHE] Plan rechazado por anti-repetición ({len(filtered_repeated)}/{len(cached_meal_names)} platos repetidos). Forzando IA.")
                            _inc_semantic_cache_stat("anti_repetition_reject")  # P2-ORCH-2
                            return {"semantic_cache_hit": False, "cached_plan_data": None}
                except Exception as e:
                    logger.warning(f"⚠️ [SEMANTIC CACHE] Error validando anti-repetición del caché: {e}")

            logger.info(f"🚀 [SEMANTIC CACHE HIT] Plan idéntico encontrado (Similitud: {similar_plan.get('similarity', 0):.3f}). Saltando LLM.")
            _inc_semantic_cache_stat("hit")  # P2-ORCH-2
            return {
                "semantic_cache_hit": True,
                "cached_plan_data": plan_data
            }

    # [P2-ORCH-2] Miss real: hubo candidatos evaluados pero ninguno fue aceptado
    # (las salidas tempranas por sin-embedding/cache-disabled/error NO cuentan
    # como miss — no medirían el hit-rate del threshold honestamente).
    _inc_semantic_cache_stat("miss")
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
    # [P5-MARKER-APPROVED-1] Gate post-approval para re-corregir días que
    # quedaron con `_critique_unresolved` cuando el reviewer aprobó.
    graph.add_node("surgical_marker_regen", surgical_marker_regen_node)

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

    # Edge condicional: revisor decide si regenerar, hacer surgical regen
    # de markers post-approval, o terminar.
    graph.add_conditional_edges(
        "review_plan",
        should_retry,
        {
            "retry": "retry_reflection",  # Vuelve a reflexión ligera en caso de rechazo (GAP 1)
            "marker_regen": "surgical_marker_regen",  # [P5-MARKER-APPROVED-1]
            "end": END,
        }
    )

    # [P5-MARKER-APPROVED-1] Tras re-corregir markers, re-aggregar shopping
    # list (assemble) y re-evaluar (review). El flag `_marker_regen_attempted`
    # previene loop: la 2da pasada por `should_retry` enrutará a "end" o
    # "retry" pero NUNCA a "marker_regen" otra vez.
    graph.add_edge("surgical_marker_regen", "assemble_plan")

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

# [P3-READY-REASON · 2026-05-12] Snapshot del último error de build del grafo,
# expuesto via `is_plan_graph_ready_with_reason()` → `/ready`.
# Pre-fix `/ready` retornaba `not_ready` sin razón; orquestadores (EasyPanel,
# k8s, load balancer) sólo veían "503 not_ready" sin pista de QUÉ falló.
# Ahora si el build crashea por timeout DB / missing API key / langgraph
# version mismatch, el operador ve el tipo de excepción + mensaje truncado
# en el body del 503 sin necesitar acceso a logs.
# Formato: dict con type/message/timestamp; None si nunca falló o si tras un
# fallo ya hubo build exitoso (reset al éxito).
_PLAN_GRAPH_LAST_BUILD_ERROR: dict | None = None


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
    global _PLAN_GRAPH, _PLAN_GRAPH_BUILD_FAILURES, _PLAN_GRAPH_LAST_BUILD_ERROR
    if _PLAN_GRAPH is not None:
        return _PLAN_GRAPH
    with _PLAN_GRAPH_LOCK:
        if _PLAN_GRAPH is not None:
            return _PLAN_GRAPH
        try:
            graph = build_plan_graph()
        except Exception as e:
            _PLAN_GRAPH_BUILD_FAILURES += 1
            # [P3-READY-REASON · 2026-05-12] Snapshot del error para /ready.
            # message truncado a 240 chars para evitar leak de paths / SQL
            # / stack traces en body del 503 que orquestadores pueden loguear.
            _PLAN_GRAPH_LAST_BUILD_ERROR = {
                "type": type(e).__name__,
                "message": str(e)[:240],
                "timestamp": time.time(),
                "failures_total": _PLAN_GRAPH_BUILD_FAILURES,
            }
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
        # [P3-READY-REASON · 2026-05-12] Reset el snapshot del último error
        # tras un build exitoso: el operador no debe ver "ready=true + reason
        # apunta a un error viejo ya resuelto".
        _PLAN_GRAPH_LAST_BUILD_ERROR = None
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

    [P3-READY-REASON · 2026-05-12] Wrapper compat con call-sites legacy. La
    versión con razón vive en `is_plan_graph_ready_with_reason()`.
    """
    return _PLAN_GRAPH is not None


def is_plan_graph_ready_with_reason() -> tuple[bool, str | None]:
    """[P3-READY-REASON · 2026-05-12] Variante del readiness probe que
    devuelve `(ready, reason)`.

    Pre-fix: `/ready` retornaba solo `{status: not_ready}` cuando el grafo
    no estaba compilado, sin pista de QUÉ falló — el operador debía abrir
    logs separadamente. Ahora `reason` indica el último modo de fallo
    conocido: tipo de excepción + mensaje truncado + count de intentos.

    Estados posibles:
      - `(True, None)`: grafo compilado y operativo.
      - `(False, "uninitialized")`: nunca se intentó (raro — `warm_plan_graph`
        debería correrse en `lifespan` al startup).
      - `(False, "build_failed:<ExcType>:<msg>:<n>")`: build crasheó.
        ExcType = `TimeoutError` / `ImportError` / `KeyError` / etc.
        msg = primeros 240 chars del str(exception).
        n = `_PLAN_GRAPH_BUILD_FAILURES` actual.

    El formato es estructurado pero parseable a ojo en el body del 503.
    Si en el futuro un orquestador quiere dispatchear por tipo de error,
    puede splitear por `:` (los primeros 2 segmentos son fijos).
    """
    if _PLAN_GRAPH is not None:
        return (True, None)
    err = _PLAN_GRAPH_LAST_BUILD_ERROR
    if err is None:
        return (False, "uninitialized")
    return (
        False,
        f"build_failed:{err.get('type','Unknown')}:"
        f"{err.get('message','')}:{err.get('failures_total',0)}",
    )


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

# [P1-33] State tracking adicional para el endpoint admin. Permite a SRE
# inspeccionar (a) cuándo se evaluó el flag por última vez, (b) si el último
# evento fue probe automático o forzado por admin, (c) el error original si
# el probe falló. Esto convierte una decisión startup-only en algo
# observable y operable sin redeploy.
_GUEST_METRICS_LAST_PROBE_AT: Optional[float] = None
_GUEST_METRICS_LAST_PROBE_RESULT: Optional[bool] = None
_GUEST_METRICS_LAST_ERROR: Optional[str] = None
_GUEST_METRICS_LAST_SOURCE: str = "default"  # "default" | "probe" | "admin_force"
_GUEST_METRICS_LAST_REASON: Optional[str] = None
_GUEST_METRICS_STATE_LOCK = threading.Lock()


def _is_guest_metrics_enabled() -> bool:
    """P1-Q10: snapshot del flag para emitters. Lectura atómica (bool en GIL)."""
    return _GUEST_METRICS_ENABLED


def get_guest_metrics_status() -> dict:
    """[P1-33] Snapshot read-only del estado de `_GUEST_METRICS_ENABLED`
    + metadata útil para el endpoint admin.

    Devuelve:
      - `enabled`: bool actual del flag.
      - `last_probe_at`: epoch seconds del último probe (None si nunca).
      - `last_probe_result`: bool del último probe (None si nunca).
      - `last_error`: string del último fallo (None si OK o nunca probado).
      - `last_source`: "default" | "probe" | "admin_force".
      - `last_reason`: string opcional explicando el último cambio (admin
        override) o None.
    """
    with _GUEST_METRICS_STATE_LOCK:
        return {
            "enabled": _GUEST_METRICS_ENABLED,
            "last_probe_at": _GUEST_METRICS_LAST_PROBE_AT,
            "last_probe_result": _GUEST_METRICS_LAST_PROBE_RESULT,
            "last_error": _GUEST_METRICS_LAST_ERROR,
            "last_source": _GUEST_METRICS_LAST_SOURCE,
            "last_reason": _GUEST_METRICS_LAST_REASON,
        }


def force_set_guest_metrics_enabled(enabled: bool, reason: Optional[str] = None) -> dict:
    """[P1-33] Override administrativo del flag `_GUEST_METRICS_ENABLED`.

    Permite a SRE forzar el flag sin esperar a un re-probe (útil cuando el
    operador acaba de aplicar la migración manual y quiere habilitar
    inmediatamente, o cuando detecta inserts fallidos en logs y quiere
    desactivar antes de que la siguiente verificación auto corra).

    Args:
        enabled: True para habilitar, False para deshabilitar.
        reason: string opcional explicando el motivo (queda en
            `last_reason` para auditoría posterior).

    Returns:
        Snapshot del estado tras el cambio (mismo formato que
        `get_guest_metrics_status`).
    """
    global _GUEST_METRICS_ENABLED, _GUEST_METRICS_LAST_SOURCE, _GUEST_METRICS_LAST_REASON
    with _GUEST_METRICS_STATE_LOCK:
        previous = _GUEST_METRICS_ENABLED
        _GUEST_METRICS_ENABLED = bool(enabled)
        _GUEST_METRICS_LAST_SOURCE = "admin_force"
        _GUEST_METRICS_LAST_REASON = reason
    logger.warning(
        f"[P1-33] _GUEST_METRICS_ENABLED forced via admin: "
        f"{previous} → {_GUEST_METRICS_ENABLED!r}. Reason: {reason!r}"
    )
    return get_guest_metrics_status()


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
    global _GUEST_METRICS_ENABLED, _GUEST_METRICS_LAST_PROBE_AT
    global _GUEST_METRICS_LAST_PROBE_RESULT, _GUEST_METRICS_LAST_ERROR
    global _GUEST_METRICS_LAST_SOURCE, _GUEST_METRICS_LAST_REASON
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
        # [P1-33] Tracking del resultado del probe.
        with _GUEST_METRICS_STATE_LOCK:
            _GUEST_METRICS_ENABLED = True
            _GUEST_METRICS_LAST_PROBE_AT = time.time()
            _GUEST_METRICS_LAST_PROBE_RESULT = True
            _GUEST_METRICS_LAST_ERROR = None
            _GUEST_METRICS_LAST_SOURCE = "probe"
            _GUEST_METRICS_LAST_REASON = None
        logger.info(
            "[STARTUP] P1-Q10: pipeline_metrics acepta inserts con user_id=NULL "
            "(guest metrics habilitadas)."
        )
        return True
    except Exception as probe_err:
        # [P1-33] Tracking del fallo del probe.
        with _GUEST_METRICS_STATE_LOCK:
            _GUEST_METRICS_ENABLED = False
            _GUEST_METRICS_LAST_PROBE_AT = time.time()
            _GUEST_METRICS_LAST_PROBE_RESULT = False
            _GUEST_METRICS_LAST_ERROR = (
                f"{type(probe_err).__name__}: {probe_err!s}"[:500]
            )
            _GUEST_METRICS_LAST_SOURCE = "probe"
            _GUEST_METRICS_LAST_REASON = None
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
        logger.info(f"💾 [HOLISTIC SCORE] Guardado en health_profile "
              f"(historial: {len(new_hp.get('pipeline_score_history', []))} entradas).")
    except Exception as db_err:
        logger.warning(f"⚠️ [HOLISTIC SCORE] Error guardando en DB: {db_err}")


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
    # [G13-EMERGENCY-FLAG-DEAD · P1-CHUNK-LEARN-3 · 2026-05-29] Removida `_is_emergency_generation`:
    # era dead-write sin consumer (el comportamiento de emergencia va vía el string `emergency_memory`
    # en cron_tasks `_refill_emergency_backup`, no vía este flag). Si en el futuro se cablea un consumer
    # real (e.g. un build_emergency_context que ramifique el prompt), re-añadir aquí + documentar el caller.
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
    # [P1-A] Removida `_strict_post_gen_required`: era un dead-write en
    # `cron_tasks.py` sin consumer. El gate post-gen estricto que la habría
    # justificado ya corre inline vía `_is_flex` recomputado. Si en el futuro
    # se cablea un consumer cross-pipeline (e.g. propagar al snapshot del
    # siguiente chunk para forzar strict tras live-recovery), volver a añadirla
    # acá Y documentar el caller que la consume.
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
    # [A1-KEYDRIFT · 2026-05-29] Dos señales conductuales backend-only derivadas
    # de `nudge_outcomes` que `_inject_advanced_learning_signals`/`inject_learning_signals_from_profile`
    # ESCRIBEN (cron_tasks.py:15355/15370 + 15764/15770) y que `build_adherence_context`
    # LEE (graph_orchestrator.py:4129/4135 → prompts/plan_generator.py:603-620) para emitir
    # instrucciones obligatorias al LLM ("CONVERSIÓN DE NUDGES CRÍTICA" + "TONO COMPROBADO").
    # Faltaban acá → `_strip_untrusted_internal_keys` las borraba en silencio (misma clase
    # que P1-CHUNK-3, trasladada a la capa del strip). Son derivadas server-side, NO claves
    # del cliente, así que whitelistearlas no debilita la defensa anti-injection P0-A2.
    "_nudge_conversion_rates",
    "_successful_tone_strategies",

    # graph_orchestrator.py — flags internos (preflight + reroll)
    "_auto_simplify",
    "_pipeline_drift_alert",
    "_creative_freedom",
    "_is_rotation_reroll",
    "_is_same_day_reroll",

    # [P1-NEW-9 · 2026-05-11] Atribución del caller para el helper
    # `_emit_plan_quality_degraded_alert`. Sin estos kwargs, el alert
    # emitido por should_retry usa plan_id="no_plan_id" para flujos
    # que NO insertan plan nuevo (ej. JIT week-2 extension del
    # proactive_agent). El plan target ya existe en `meal_plans`; con
    # estas keys el alert correlaciona al plan correcto y SRE puede
    # filtrar por `metadata.caller_context`.
    "_caller_target_plan_id",
    "_caller_context",
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
        # [P2-ORCH-9 · 2026-05-28] 'stable' añadido: el productor
        # (cron_tasks.inject_learning_signals_from_profile) emite 'stable' en el
        # caso neutral (el más común). Sin esta entrada, `_validate_form_hint_enums`
        # lo trataba como out-of-enum → lo limpiaba a None + WARNING falso de
        # "drift de schema" en CADA generación de adherencia neutral (fatiga de
        # alerta que entrena a ignorar la alarma P1-A8). El consumidor
        # `_ema_hint_instructions` no tiene clave 'stable' → no-op reconocido
        # (sin instrucción extra), que es el comportamiento correcto.
        "temporary_dip", "drastic_change", "improving", "stable",
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


# ============================================================
# [P0-ORCH-1 · 2026-05-28] Fallback matemático con filtrado de alérgenos.
# ------------------------------------------------------------
# El fallback determinista hardcodeaba un menú huevo+pollo+pescado SIN mirar
# las restricciones declaradas. Peor: la rama crítica de
# `_apply_critical_review_guardrails` se dispara JUSTO cuando un plan fue
# rechazado por violar una alergia/condición médica — y entregaba un plan de
# contingencia con el MISMO alérgeno (riesgo clase anafilaxia) bajo un
# disclaimer de "plan seguro". Ahora cada slot elige la primera plantilla
# cuyos tokens de alérgeno NO intersectan las restricciones del usuario; la
# ÚLTIMA plantilla de cada pool es neutral (cero tokens) → siempre hay un meal
# seguro por slot. Over-detección = dirección SEGURA (peor caso: se evita una
# proteína innecesariamente y cae al template neutral; jamás se sirve el
# alérgeno declarado).
#
# Knob `MEALFIT_FALLBACK_ALLERGEN_FILTER` (default True) revierte sin redeploy
# al comportamiento previo (restricted_tokens vacío → pool[0] de cada slot =
# el menú histórico idéntico).
# Tooltip-anchor: P0-ORCH-1.
# ============================================================
FALLBACK_ALLERGEN_FILTER = _env_bool("MEALFIT_FALLBACK_ALLERGEN_FILTER", True)

# token canónico -> keywords (lowercase, sin acentos) que lo delatan en las
# restricciones declaradas. Las plantillas neutrales (tokens vacíos) cierran
# la garantía de que SIEMPRE hay un meal seguro por slot.
_FALLBACK_ALLERGEN_KEYWORDS = {
    "egg":       ("huevo", "huevos", "egg", "clara de huevo"),
    "chicken":   ("pollo", "chicken", "pechuga de pollo", "gallina"),
    "fish":      ("pescado", "pescados", "fish", "atun", "salmon", "tilapia",
                  "bacalao", "sardina", "mero"),
    "shellfish": ("marisco", "mariscos", "camaron", "camarones", "langosta",
                  "cangrejo", "shellfish", "shrimp", "ostra", "calamar", "pulpo"),
    "beef":      ("carne de res", "ternera", "vacuno", "beef"),
    "pork":      ("cerdo", "puerco", "pork", "tocino", "jamon", "chorizo",
                  "salchicha", "embutido"),
    "dairy":     ("leche", "lacteo", "lacteos", "lactosa", "dairy", "queso",
                  "yogur", "yogurt", "mantequilla"),
    "peanut":    ("mani", "peanut", "cacahuate", "cacahuete"),
    "soy":       ("soya", "soja", "tofu", "edamame"),
    "gluten":    ("gluten", "trigo", "wheat", "celiaco", "celiaca"),
    "oats":      ("avena", "oat"),
    "legume":    ("lenteja", "lentejas", "garbanzo", "garbanzos", "frijol",
                  "frijoles", "habichuela", "habichuelas", "legumbre", "legumbres"),
    "nuts":      ("nuez", "nueces", "almendra", "almendras", "frutos secos",
                  "tree nut", "anacardo", "merey", "pistacho"),
}

# Des-acentuado mínimo (es-DO) para normalizar el texto de restricciones.
_FALLBACK_ACCENT_MAP = str.maketrans("áéíóúüñ", "aeiouun")

# Pools ordenados por slot: la PRIMERA entrada reproduce el menú histórico
# (restricted vacío → comportamiento idéntico); la ÚLTIMA es neutral (sin
# tokens). Cada entrada: (name, frozenset(tokens), desc, [ingredients]).
_FALLBACK_MEAL_POOLS = {
    "Desayuno": [
        ("Huevos y Avena", frozenset({"egg", "oats", "gluten"}),
         "Huevos revueltos con avena cocida y fruta.",
         ["2 huevos", "1/2 taza de avena cocida", "1 fruta de temporada"]),
        ("Avena con Frutas y Semillas", frozenset({"oats", "gluten"}),
         "Avena cocida con frutas frescas y semillas.",
         ["1/2 taza de avena", "frutas variadas", "1 cda de semillas de chía"]),
        ("Frutas Frescas con Semillas", frozenset(),
         "Bowl de frutas frescas de temporada con semillas.",
         ["frutas variadas de temporada", "semillas de girasol o chía", "agua"]),
    ],
    "Almuerzo": [
        ("Pollo y Arroz", frozenset({"chicken"}),
         "Pechuga a la plancha con arroz blanco y vegetales.",
         ["pechuga de pollo a la plancha", "arroz blanco", "vegetales al gusto"]),
        ("Carne de Res y Arroz", frozenset({"beef"}),
         "Carne de res magra con arroz y vegetales.",
         ["carne de res magra", "arroz blanco", "vegetales al gusto"]),
        ("Pescado con Arroz", frozenset({"fish"}),
         "Filete de pescado con arroz y ensalada.",
         ["filete de pescado", "arroz blanco", "ensalada verde"]),
        ("Lentejas con Arroz", frozenset({"legume"}),
         "Lentejas guisadas con arroz y vegetales.",
         ["lentejas guisadas", "arroz blanco", "vegetales al gusto"]),
        ("Arroz con Vegetales y Aguacate", frozenset(),
         "Arroz con vegetales salteados y aguacate.",
         ["arroz blanco", "vegetales salteados", "1/2 aguacate"]),
    ],
    "Cena": [
        ("Pescado y Batata", frozenset({"fish"}),
         "Filete de pescado al horno con batata asada.",
         ["filete de pescado al horno", "batata asada", "vegetales al vapor"]),
        ("Pollo con Vegetales", frozenset({"chicken"}),
         "Pechuga de pollo con vegetales al vapor y batata.",
         ["pechuga de pollo", "vegetales al vapor", "batata asada"]),
        ("Carne de Res con Vegetales", frozenset({"beef"}),
         "Carne de res magra con vegetales y batata.",
         ["carne de res magra", "vegetales al vapor", "batata asada"]),
        ("Garbanzos con Vegetales", frozenset({"legume"}),
         "Garbanzos guisados con vegetales y batata.",
         ["garbanzos guisados", "vegetales al vapor", "batata asada"]),
        ("Ensalada de Vegetales con Aguacate", frozenset(),
         "Ensalada abundante de vegetales con aguacate y aceite de oliva.",
         ["vegetales variados", "1/2 aguacate", "aceite de oliva"]),
    ],
}


def _detect_restricted_tokens(form_data: dict) -> frozenset:
    """[P0-ORCH-1] Extrae los tokens de alérgeno declarados en el formulario.

    Lee allergies / medicalConditions / dislikes / restrictions / intolerances
    (listas o strings), normaliza (lowercase + sin acentos) y matchea contra
    `_FALLBACK_ALLERGEN_KEYWORDS`. Over-detección es la dirección SEGURA: un
    falso positivo solo evita una proteína de más (cae al template neutral).
    """
    if not isinstance(form_data, dict):
        return frozenset()
    parts = []
    for key in ("allergies", "medicalConditions", "dislikes",
                "restrictions", "intolerances", "foodRestrictions"):
        v = form_data.get(key)
        if isinstance(v, (list, tuple, set)):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    if not parts:
        return frozenset()
    text = " ".join(parts).lower().translate(_FALLBACK_ACCENT_MAP)
    found = set()
    for token, keywords in _FALLBACK_ALLERGEN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.add(token)
                break
    return frozenset(found)


def _fallback_restricted_tokens(form_data: dict) -> frozenset:
    """[P0-ORCH-1] Wrapper gateado por `MEALFIT_FALLBACK_ALLERGEN_FILTER`.
    Con el knob OFF retorna vacío → fallback idéntico al histórico."""
    if not FALLBACK_ALLERGEN_FILTER:
        return frozenset()
    return _detect_restricted_tokens(form_data)


def _select_safe_fallback_meal(pool: list, restricted_tokens: frozenset):
    """[P0-ORCH-1] Primera plantilla del pool cuyos tokens NO intersectan las
    restricciones. El último elemento de cada pool es neutral (tokens vacíos),
    así que SIEMPRE hay un retorno seguro."""
    for tmpl in pool:
        if not (tmpl[1] & restricted_tokens):
            return tmpl
    # Inalcanzable en la práctica (pool[-1] es neutral); defensa por si un
    # refactor futuro elimina la entrada neutral.
    return pool[-1] if pool else None


def _build_fallback_day(nutr: dict, day_number: int,
                        restricted_tokens: frozenset = frozenset()) -> dict:
    """P1-A5: extraída del closure local de `arun_plan_pipeline`.

    Construye un día fallback determinista (3 comidas balanceadas).

    [P0-ORCH-1 · 2026-05-28] `restricted_tokens` filtra alérgenos: cada slot
    elige la primera plantilla segura. Vacío → menú histórico idéntico.
    """
    target_cal = nutr.get('target_calories', 2000)
    macros_dict = nutr.get('macros', {})
    target_pro = macros_dict.get('protein_g', 150)
    target_car = macros_dict.get('carbs_g', 200)
    target_fat = macros_dict.get('fats_g', 60)

    def create_meal(tmpl, cal_ratio, p_ratio, c_ratio, f_ratio, meal_type):
        name, _tokens, desc, ingredients = tmpl
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
            "ingredients": list(ingredients),
            "recipe": ["Mise en place: Preparar todo", "El Toque de Fuego: Cocinar la proteína", "Montaje: Servir"]
        }

    # (meal_type, ratio_cal, ratio_pro, ratio_car, ratio_fat) — ratios idénticos al menú histórico.
    _slots = (
        ("Desayuno", 0.3, 0.3, 0.3, 0.3),
        ("Almuerzo", 0.4, 0.4, 0.4, 0.4),
        ("Cena",     0.3, 0.3, 0.3, 0.3),
    )
    meals = []
    for meal_type, r_cal, r_pro, r_car, r_fat in _slots:
        tmpl = _select_safe_fallback_meal(_FALLBACK_MEAL_POOLS[meal_type], restricted_tokens)
        meals.append(create_meal(tmpl, r_cal, r_pro, r_car, r_fat, meal_type))

    return {
        "day": day_number,
        "daily_summary": "Plan de Contingencia de Emergencia (Generado Matemáticamente)",
        "total_calories": target_cal,
        "total_protein": target_pro,
        "total_carbs": target_car,
        "total_fats": target_fat,
        "meals": meals,
    }


def _get_extreme_fallback_plan(nutr: dict, goal: str, num_days: int = 3, day_offset: int = 0,
                               restricted_tokens: frozenset = frozenset()) -> dict:
    """P1-A5: extraída del closure local de `arun_plan_pipeline`.

    Fallback matemático determinista para evitar caídas del sistema.

    P0-1: `num_days` ahora se respeta — antes estaba hardcodeado a 3 y entregaba
    plans truncados al frontend cuando el usuario pidió 7. `day_offset` permite
    continuar la numeración cuando se rellenan días faltantes a un plan parcial.

    [P0-ORCH-1 · 2026-05-28] `restricted_tokens` (de `_fallback_restricted_tokens`)
    filtra alérgenos en cada slot. Vacío → menú histórico idéntico.
    """
    target_cal = nutr.get('target_calories', 2000)
    macros_dict = nutr.get('macros', {})
    target_pro = macros_dict.get('protein_g', 150)
    target_car = macros_dict.get('carbs_g', 200)
    target_fat = macros_dict.get('fats_g', 60)

    safe_num_days = max(1, int(num_days or 1))
    days = [_build_fallback_day(nutr, day_offset + i + 1, restricted_tokens) for i in range(safe_num_days)]

    plan = {
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
    # [P0-ORCH-1] Observabilidad: marcar cuando el fallback evitó alérgenos
    # declarados (útil para Grafana / debugging — el plan sigue siendo válido).
    if restricted_tokens:
        plan["_allergen_filtered"] = True
        plan["_allergen_tokens"] = sorted(restricted_tokens)
    return plan


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


def _repair_partial_plan(plan: dict, *, nutrition: dict, requested_days: int,
                         restricted_tokens: frozenset = frozenset()) -> bool:
    """P1-A5: extraída del closure local. `nutrition` y `requested_days`
    ahora kwargs explícitos.

    [P0-ORCH-1 · 2026-05-28] `restricted_tokens` se propaga a los
    `_build_fallback_day` que reemplazan/rellenan días, para que un plan
    parcial reparado tampoco reintroduzca alérgenos declarados.

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
        logger.warning("🛡️ [P0-2] Macros agregados faltantes inyectados.")

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
            new_days.append(_build_fallback_day(nutrition, idx + 1, restricted_tokens))
            replaced_count += 1
            repaired = True

    # Rellenar faltantes
    filled_count = 0
    while len(new_days) < requested_days:
        new_days.append(_build_fallback_day(nutrition, len(new_days) + 1, restricted_tokens))
        filled_count += 1
        repaired = True

    plan["days"] = new_days

    if replaced_count or filled_count:
        logger.warning(f"🛡️ [P0-2] Días reparados: {replaced_count} reemplazados (vacíos/inválidos), "
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
        logger.warning(f"🛡️ [P0-C] Shopping lists normalizadas: {shopping_repaired}/4 claves "
              f"ausentes o no-list reparadas a `[]`.")

    if repaired:
        plan["_is_fallback"] = True
        plan.setdefault(
            "_review_disclaimer",
            "El plan se completó con valores de contingencia matemáticos por "
            "indisponibilidad temporal de la IA. Por favor regenera más tarde."
        )

    return repaired


def compute_clinical_band_score(plan: dict, nutrition: dict, *,
                                lower: float = None, upper: float = None) -> dict:
    """[P4-SCOREBOARD · 2026-06-14] Score DETERMINISTA de precisión por-plan: la fracción de celdas
    (día × macro) en que el macro ENTREGADO (suma de las comidas del día) cae en la banda
    [lower, upper] × target diario. Mide la precisión REAL del plan entregado → la precisión deja de ser
    AUTOAFIRMADA (antes el "90-112%" era un claim sin medición por-plan). kcal usa una banda más estrecha
    ([0.95, 1.05]) porque el reconcile clava las calorías y deja que los macros absorban el residual.
    Cero LLM. Retorna {score, cells_in_band, cells_total, per_macro, band_*}. Fail-safe → score=None."""
    try:
        lo_m = BAND_SCORE_LOWER if lower is None else lower
        hi_m = BAND_SCORE_UPPER if upper is None else upper
        tm = (nutrition.get("total_daily_macros") if isinstance(nutrition, dict) else None) or \
             (nutrition.get("macros") if isinstance(nutrition, dict) else None) or {}
        pm = plan.get("macros") if isinstance(plan.get("macros"), dict) else {}
        targets = {
            "protein": _meal_macro_num(tm.get("protein_str")) or _meal_macro_num(pm.get("protein")),
            "carbs": _meal_macro_num(tm.get("carbs_str")) or _meal_macro_num(pm.get("carbs")),
            "fats": _meal_macro_num(tm.get("fats_str")) or _meal_macro_num(pm.get("fats")),
            "kcal": _meal_macro_num((nutrition or {}).get("total_daily_calories")
                                    or (nutrition or {}).get("target_calories")) or _meal_macro_num(plan.get("calories")),
        }
        bands = {"protein": (lo_m, hi_m), "carbs": (lo_m, hi_m), "fats": (lo_m, hi_m), "kcal": (0.95, 1.05)}
        per = {k: {"in": 0, "total": 0} for k in targets}
        for day in plan.get("days", []) or []:
            meals = day.get("meals", []) or []
            delivered = {
                "protein": sum(_meal_macro_num(m.get("protein")) for m in meals),
                "carbs": sum(_meal_macro_num(m.get("carbs")) for m in meals),
                "fats": sum(_meal_macro_num(m.get("fats")) for m in meals),
                "kcal": sum(_meal_macro_num(m.get("cals")) for m in meals),
            }
            for k, t in targets.items():
                if t and t > 0:
                    lo, hi = bands[k]
                    per[k]["total"] += 1
                    if lo * t <= delivered[k] <= hi * t:
                        per[k]["in"] += 1
        cin = sum(v["in"] for v in per.values())
        ctot = sum(v["total"] for v in per.values())
        return {
            "score": round(cin / ctot, 3) if ctot else None,
            "cells_in_band": cin, "cells_total": ctot,
            "band_macros": [lo_m, hi_m], "band_kcal": [0.95, 1.05],
            "per_macro": {k: (round(v["in"] / v["total"], 3) if v["total"] else None) for k, v in per.items()},
        }
    except Exception as _bs_e:
        logger.warning(f"[P4-SCOREBOARD] compute_clinical_band_score falló: {type(_bs_e).__name__}: {_bs_e}")
        return {"score": None, "cells_in_band": 0, "cells_total": 0, "error": True}


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
            logger.info(
                f"📊 [HOLISTIC SCORE] Pipeline Quality: {holistic_score:.3f} "
                f"(retry={retry_penalty:.2f}, review={review_bonus:.2f}, "
                f"cal={cal_score:.2f}, zero_cal_days={days_with_zero_cals}, "
                f"fallback={delivered_was_fallback})"
            )

            # [P4-SCOREBOARD · 2026-06-14] Precisión MEDIDA por-plan (no autoafirmada): qué fracción de
            # celdas (día × macro) del plan ENTREGADO cae en la banda clínica vs el target. Inyecta
            # `plan["clinical_band_score"]` (→ persistido + visible para PDF/UI) y emite la métrica
            # `clinical_band` a `pipeline_metrics` para el scoreboard de flota + el cron de drift. El
            # fallback se marca en metadata para que la agregación lo excluya (sus macros son ~target
            # por construcción → score engañosamente alto). Fail-safe: no afecta al plan entregado.
            try:
                _band = compute_clinical_band_score(plan, nutrition)
                plan["clinical_band_score"] = _band
                _band_val = _band.get("score")
                if _band_val is not None:
                    _emit_progress(initial_state, "metric", {
                        "node": "clinical_band",
                        "duration_ms": int(pipeline_duration * 1000),
                        "retries": final_state.get("attempt", 1) - 1,
                        "tokens_estimated": 0,
                        "confidence": _band_val,   # = el band score [0,1]
                        "metadata": {
                            "cells_in_band": _band.get("cells_in_band"),
                            "cells_total": _band.get("cells_total"),
                            "per_macro": _band.get("per_macro"),
                            "band_macros": _band.get("band_macros"),
                            "delivered_was_fallback": delivered_was_fallback,
                            "review_passed": final_state.get("review_passed"),
                        },
                    })
                    logger.info(
                        f"🎯 [CLINICAL BAND SCORE] Precisión medida: {_band_val:.2f} "
                        f"({_band.get('cells_in_band')}/{_band.get('cells_total')} celdas en banda; "
                        f"por-macro {_band.get('per_macro')}; fallback={delivered_was_fallback})")
            except Exception as _cbs_e:
                logger.warning(f"[P4-SCOREBOARD] emit del band score falló: {type(_cbs_e).__name__}: {_cbs_e}")

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
                    logger.warning(
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
        logger.warning(f"⚠️ [HOLISTIC SCORE] Error calculando score: {e}")


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

    # [P1-32] Caso anómalo: el plan YA es `_is_fallback=True` PERO
    # presenta `_schema_invalid=True` o un rechazo médico crítico.
    # `_get_extreme_fallback_plan()` produce un plan válido y, tras P0-ORCH-1
    # (2026-05-28), allergen-aware cuando se le pasa `restricted_tokens` (la
    # regeneración de abajo lo hace), así que esta combinación NUNCA debería
    # ocurrir naturalmente. Si llegamos aquí
    # con ese estado, indica:
    #   (a) Mutación downstream corrompió el plan post-fallback (race,
    #       bug aislado en `_repair_partial_plan`, etc.).
    #   (b) Doble graceful degradation (e.g. el except del global timeout
    #       generó fallback, luego un nodo posterior lo marcó schema_invalid).
    #   (c) `_get_extreme_fallback_plan()` rompió su contrato de "siempre
    #       válido" por bug — alerta de regression.
    #
    # Pre-fix, el guard `not already_fallback` short-circuiteaba toda la
    # rama crítica, entregando un plan `_is_fallback + _schema_invalid`
    # al frontend que no podía renderizarlo (hojas en blanco). Y para
    # `_is_fallback + critical_rejection`, el plan se entregaba pese a
    # violar alergias/condiciones médicas — riesgo de salud.
    #
    # Fix: detectar la anomalía explícitamente, emitir error logging para
    # que SRE pueda alertar (la frecuencia del log mide la salud del
    # `_get_extreme_fallback_plan`), y forzar regeneración de un fallback
    # fresco (`already_fallback=False` reentra al path estándar).
    if already_fallback and (
        schema_invalid
        or (not review_passed and rejection_severity == "critical")
    ):
        _p132_cause = (
            "_schema_invalid=True" if schema_invalid
            else f"rechazo médico critical (severity={rejection_severity!r})"
        )
        logger.error(
            f"🚨 [P1-32] Plan ya `_is_fallback=True` PERO también "
            f"{_p132_cause}. Anómalo — `_get_extreme_fallback_plan()` "
            f"debería producir planes válidos y médicamente seguros. "
            f"Forzando regeneración. razones={rejection_reasons}"
        )
        # Reset del flag para forzar entrada a la rama de needs_critical_
        # fallback (regeneración fresca + disclaimer + preservación del
        # embedding del intento original).
        already_fallback = False

    needs_critical_fallback = isinstance(plan_result, dict) and not already_fallback and (
        schema_invalid
        or (not review_passed and rejection_severity == "critical")
    )

    if needs_critical_fallback:
        cause = (
            "SCHEMA INVÁLIDO (plan no renderizable por el frontend)"
            if schema_invalid else "rechazo médico CRÍTICO"
        )
        logger.error(f"🚨 [P0-1/P1-8 GUARDRAIL] Fallback forzado por: {cause}. "
              f"Generando plan matemático ({requested_days} días). "
              f"review_passed={review_passed}, severity={rejection_severity}, "
              f"razones={rejection_reasons}")
        fallback_plan = _get_extreme_fallback_plan(
            nutrition,
            actual_form_data.get("mainGoal", "Salud General"),
            num_days=requested_days,
            # [P0-ORCH-1] CRÍTICO: esta rama se dispara por rechazo médico/alergia.
            # El fallback DEBE excluir el alérgeno declarado, nunca reintroducirlo.
            restricted_tokens=_fallback_restricted_tokens(actual_form_data),
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
        logger.warning(f"⚠️ [P0-1 TRANSPARENCY] Plan entregado tras rechazo no-crítico. "
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
        logger.warning(f"🛡️ [P0-2 GUARDRAIL] Plan ausente tras pipeline exitoso. "
              f"Generando fallback completo ({requested_days} días).")
        # P1-9: `_get_extreme_fallback_plan` ya setea `_is_fallback=True` en el plan.
        final_state["plan_result"] = _get_extreme_fallback_plan(
            nutrition,
            actual_form_data.get("mainGoal", "Salud General"),
            num_days=requested_days,
            restricted_tokens=_fallback_restricted_tokens(actual_form_data),  # [P0-ORCH-1]
        )
    elif not _is_plan_complete(plan_final, requested_days):
        days_count = len(plan_final.get("days") or [])
        invalid_count = sum(1 for d in (plan_final.get("days") or []) if not _is_day_valid(d))
        logger.warning(f"🛡️ [P0-2 GUARDRAIL] Plan terminó incompleto pese a graph success. "
              f"Días: {days_count}/{requested_days}, inválidos: {invalid_count}. Reparando.")
        # P1-9: `_repair_partial_plan` ya setea el flag en plan_final si repara.
        _repair_partial_plan(plan_final, nutrition=nutrition, requested_days=requested_days,
                             restricted_tokens=_fallback_restricted_tokens(actual_form_data))  # [P0-ORCH-1]

    # Inyectar profile_embedding para que el caller lo guarde en la BD
    if final_state.get("profile_embedding") and final_state.get("plan_result"):
        final_state["plan_result"]["_profile_embedding"] = final_state["profile_embedding"]

    # [P3-FALLBACK-CLINICAL-LAYER · 2026-06-14] El fallback matemático (`_get_extreme_fallback_plan`) y el
    # parcial reparado (`_repair_partial_plan`) BYPASSAN assemble_plan_node → no heredan FS1-FS8. Aquí, en
    # el punto único de salida, les aplicamos la capa clínica determinista COMPLETA (food-safety +
    # sustitución de sodio/azúcar + quantize + micros + variedad + proveniencia + gate FS9). Solo corre
    # sobre planes `_is_fallback` → no-op en el happy path (que ya pasó por el bloque inline de assemble);
    # idempotente vía marker. Gated por knob para rollback. La red renal de abajo queda redundante-pero-
    # inofensiva (idempotente) y sigue cubriendo el caso `FALLBACK_CLINICAL_LAYER_ENABLED=False`. Fail-safe.
    if FALLBACK_CLINICAL_LAYER_ENABLED:
        _fcl_plan = final_state.get("plan_result")
        if (isinstance(_fcl_plan, dict) and _fcl_plan.get("_is_fallback")
                and not _fcl_plan.get("_clinical_layer_applied")):
            try:
                _apply_deterministic_clinical_layer(_fcl_plan, actual_form_data, nutrition)
                logger.warning("🛡️ [P3-FALLBACK-CLINICAL-LAYER] Capa clínica determinista aplicada al "
                               "plan de fallback (food-safety + condition-subs + quantize + micros + gate FS9).")
            except Exception as _fcl_e:
                logger.warning(f"[P3-FALLBACK-CLINICAL-LAYER] error aplicando la capa al fallback: "
                               f"{type(_fcl_e).__name__}: {_fcl_e}")

    # [P3-CONDITION-RULES · 2026-06-14] RED DE SEGURIDAD RENAL en el punto único de salida →
    # cubre TODO plan entregado, incluido el fallback matemático (que bypassa assemble y su gate
    # FS9). El protein del fallback YA viene capeado (se construye desde `nutrition` capeado en la
    # fuente); aquí solo garantizamos el gate de derivación profesional + la meta del cap, para que
    # un paciente renal NUNCA reciba un plan sin la advertencia de nefrólogo. No-op en el happy path
    # (assemble ya seteó renal_gate). Fail-safe.
    try:
        _pf_renal = final_state.get("plan_result")
        # [P4-CONSTRAINT-ABC · 2026-06-14] Capa 3 (red de salida) RENAL vía el engine → delega a
        # `_renal_exit_safety_net` (cuerpo verbatim del bloque renal). El engine solo corre el constraint
        # renal si el perfil es renal; el bloque interno re-chequea `renal_protein_cap.applied`. Efecto
        # idéntico al bloque inline previo.
        if isinstance(_pf_renal, dict):
            try:
                from clinical_constraints import ClinicalConstraintEngine as _CCE3
                _CCE3(actual_form_data).run_safety_net(_pf_renal, nutrition)
            except Exception:
                _renal_exit_safety_net(_pf_renal, nutrition, actual_form_data)  # fallback (engine no disponible)
        # [P3-CONDITION-RULES] Gate GENÉRICO en el punto de salida: si el usuario declaró CUALQUIER
        # condición médica real y el plan entregado (incl. el fallback matemático, que bypassa el
        # gate FS9 de assemble) no lo lleva, lo añadimos. Honestidad/seguridad: un plan para alguien
        # con condición declarada SIEMPRE debe llevar la advertencia profesional, en cualquier path.
        if (PRO_REVIEW_FLAG_ENABLED and isinstance(_pf_renal, dict)
                and not isinstance(_pf_renal.get("requires_professional_review"), dict)
                and _has_real_medical_flags(actual_form_data.get("medicalConditions"))):
            _gc = [str(c) for c in (actual_form_data.get("medicalConditions") or [])
                   if str(c).strip().lower() not in _MEDICAL_NONE_SENTINELS]
            _pf_renal["requires_professional_review"] = {
                "flag": True, "conditions": _gc,
                "note": ("⚕️ Declaraste condición(es) de salud (" + ", ".join(_gc) + "). Este plan las "
                         "considera de forma general pero NO sustituye la evaluación de tu médico o "
                         "nutricionista. Consúltalo antes de seguir este plan, especialmente para "
                         "ajustar porciones, sodio, azúcares o restricciones específicas."),
            }
            logger.warning(f"⚕️ [P3-CONDITION-RULES] Gate profesional genérico aplicado al plan "
                           f"entregado (condiciones: {_gc}) — cubre fallback/paths sin assemble.")
    except Exception as _rsafe_e:
        logger.warning(f"[P3-CONDITION-RULES] red de seguridad de condiciones falló: "
                       f"{type(_rsafe_e).__name__}: {_rsafe_e}")

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
    logger.info("\n" + "🔗" * 30)
    logger.info("🔗 [LANGGRAPH] Iniciando Pipeline Multi-Agente")
    logger.info("🔗" * 30)
    
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
    # [P3-CONDITION-RULES · 2026-06-14] ERC: capear el target de proteína EN LA FUENTE (antes de
    # generación/review/fallback). Sin esto, el LLM generaba un plan alto en proteína que el revisor
    # renal-aware rechazaba → fallback matemático SIN capear (hallado en prueba en vivo). Fail-safe.
    # [P4-CONSTRAINT-ABC · 2026-06-14] Capa 1 (ajuste-en-fuente) vía el engine → delega a
    # `_apply_renal_cap_to_nutrition` (renal es hoy el único constraint con adjust_targets). Efecto idéntico.
    try:
        from clinical_constraints import ClinicalConstraintEngine as _CCE
        _CCE(actual_form_data).run_adjust_targets(nutrition)
    except Exception as _cct_e:
        logger.warning(f"[P4-CONSTRAINT-ABC] adjust_targets vía engine falló, fallback directo: "
                       f"{type(_cct_e).__name__}: {_cct_e}")
        _apply_renal_cap_to_nutrition(nutrition, actual_form_data)

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
            logger.warning(f"⚠️ Error recuperando comidas recientes desde db: {e}")
            
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
            logger.warning(f"⚠️ Error recuperando feedback evolutivo de 7 días: {e}")
            
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
            logger.info(f"✅ [GAP 1] Señal 1: Adherencia EMA granular inyectada → {_meal_level_adherence}")

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
            logger.info(f"✅ [GAP 1] Señal 1b: EMA largo plazo inyectado → low={list(low_long)}, high={list(high_long)}")

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
        logger.info(f"✅ [GAP 1] Señal 2: Razones de abandono inyectadas → {_abandoned_reasons}")

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
        logger.info("✅ [GAP 1] Señal 3: Estado emocional 'needs_comfort' inyectado")
    elif _emotional_state == 'ready_for_challenge':
        history_context += (
            "\n\n--- 🔥 [CRON] ESTADO EMOCIONAL: USUARIO MOTIVADO Y LISTO PARA RETOS ---\n"
            "El usuario está altamente motivado y positivo (detectado vía nudge_outcomes).\n"
            "INSTRUCCIÓN: Introduce recetas más desafiantes, nuevos perfiles de sabor y técnicas culinarias avanzadas. "
            "Diseña comidas enfocadas en máximo rendimiento y variedad para mantener el impulso.\n"
            "----------------------------------------------------------------------\n"
        )
        logger.info("✅ [GAP 1] Señal 3: Estado emocional 'ready_for_challenge' inyectado")

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
        logger.info(f"✅ [GAP 1] Señal 4: Tipos de comida ignorados inyectados → {_ignored_meal_types}")

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
            logger.info(f"✅ [GAP 1] Señal 5: Días de baja adherencia inyectados → {low_days}")

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
        logger.info(f"✅ [GAP 1] Señal 6: Técnicas éxito/abandono inyectadas → succ={_succ_techs}, aban={_aban_techs}")

    # [P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Confidence gate para Señales 7-9.
    # Razón (user request 2026-05-21): "no quiero que forze gustos que no
    # existen — si el usuario no agregó suficiente información de preferencias,
    # el chunk debe crear platos en base a macros + nevera/lista de compras".
    #
    # Las Señales 1-6 son self-gating (solo inyectan si data subyacente
    # existe — EMA con N≥3 muestras, abandono con razones explícitas, etc.).
    # Las Señales 7-9, sin embargo, inyectan basándose en `_previous_plan_quality`
    # que se computa SIEMPRE (vía meta-learning reflection sobre consumo escaso).
    # En usuarios nuevos / con baja adherencia, eso produce señales DÉBILES
    # presentadas al LLM como instrucciones FUERTES → forzaba gustos que el
    # usuario nunca expresó.
    #
    # Gate: si `_previous_plan_quality < MIN_LEARNING_CONFIDENCE` (default 0.40)
    # → SKIP Señales 7-9. El LLM recibe sólo macros + pantry + alergias +
    # diet/goal — fallback limpio sin invención.
    #
    # Rollback sin redeploy: `MEALFIT_MIN_LEARNING_CONFIDENCE=0.0` desactiva
    # el gate (vuelta al comportamiento pre-fix).
    _prev_quality = actual_form_data.get("_previous_plan_quality")
    # [A4-KEYDRIFT · 2026-05-29] La clave que el pipeline de chunks ESCRIBE es
    # `quality_history_chunks` (cron_tasks.py:15281, lista de floats 0-1). La clave
    # legacy `quality_history` (sin sufijo) NO tiene ningún writer en backend ni
    # frontend → el fallback de confianza estaba MUERTO justo en el caso para el que
    # fue diseñado (P2-CHUNK-4 omite `_previous_plan_quality` con consumed<3, y este
    # promedio del historial era la red de seguridad). Fallback al nombre viejo por compat.
    _quality_hist = actual_form_data.get("quality_history_chunks") or actual_form_data.get("quality_history", [])
    _adherence_hint = actual_form_data.get("_adherence_hint")
    _adherence_ema_hint = actual_form_data.get("_adherence_ema_hint")

    # Confidence inferida: prev_quality numérica O promedio del historial
    # (si tiene ≥3 ciclos, el promedio es una estimación razonable).
    _inferred_confidence = None
    if isinstance(_prev_quality, (int, float)):
        _inferred_confidence = float(_prev_quality)
    elif isinstance(_quality_hist, list) and len(_quality_hist) >= 3:
        try:
            _inferred_confidence = float(sum(_quality_hist) / len(_quality_hist))
        except (TypeError, ZeroDivisionError):
            _inferred_confidence = None

    _skip_pref_signals = (
        _inferred_confidence is None or _inferred_confidence < MIN_LEARNING_CONFIDENCE
    )

    # Excepción: si la EMA hint es explícitamente "drastic_change"/"improving"
    # o adherence_hint=high/low, esas son señales OBSERVADAS (no estimadas
    # vía meta-learning) y SÍ se respetan independiente de confidence —
    # significan que el usuario tiene patrón claro de adherencia que merece
    # inyectarse incluso con prev_quality bajo.
    _has_explicit_adherence_signal = bool(_adherence_hint) or bool(_adherence_ema_hint)

    if _skip_pref_signals and not _has_explicit_adherence_signal:
        logger.info(
            f"🔧 [GAP 1 GATE/P1-LOW-SIGNAL] Confidence={_inferred_confidence} "
            f"< MIN_LEARNING_CONFIDENCE={MIN_LEARNING_CONFIDENCE} y sin adherencia "
            f"observada → SKIP Señales 7-9 (fallback macros+pantry+alergias)."
        )

    # Señal 7: Snapshot de calidad global — adherencia escalar + score previo + tendencia
    # (_adherence_hint ya se usa en complexity_guard pero no se narraba aqui;
    #  _previous_plan_quality + quality_history solo iban al reflection_node lateral.)
    if (not _skip_pref_signals or _has_explicit_adherence_signal) and (
        _adherence_hint or isinstance(_prev_quality, (int, float)) or (isinstance(_quality_hist, list) and len(_quality_hist) >= 2)
    ):
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
        logger.info(f"✅ [GAP 1] Señal 7: Snapshot calidad/adherencia inyectado (hint={_adherence_hint}, ema_hint={_adherence_ema_hint}, prev={_prev_quality})")

    # Señal 8: Platos recurrentes de alta adherencia (variaciones bienvenidas)
    # [P1-LOW-SIGNAL-FALLBACK] Gateada por confidence (ver Señal 7). Si el usuario
    # no tiene historial robusto, los "platos recurrentes" son comidas episódicas
    # (1-2 entradas) NO patrones reales de preferencia → forzarlos al LLM crea
    # gustos artificiales.
    _frequent_meals = actual_form_data.get("frequent_meals", [])
    if (not _skip_pref_signals) and isinstance(_frequent_meals, list) and _frequent_meals:
        freq_fmt = ", ".join(_frequent_meals[:5])
        history_context += (
            "\n\n--- 🔁 [CRON] PLATOS RECURRENTES DE ALTA ADHERENCIA ---\n"
            f"El usuario consume estos platos repetidamente con éxito: {freq_fmt}.\n"
            "INSTRUCCIÓN: Incluye al menos 1 VARIACIÓN (no copia literal) de estos platos en la semana. "
            "Son anclas de adherencia probadas; respetar su estructura base reduce el riesgo de abandono.\n"
            "----------------------------------------------------------------------\n"
        )
        logger.info(f"✅ [GAP 1] Señal 8: Platos recurrentes inyectados → {_frequent_meals[:5]}")
    elif _skip_pref_signals and isinstance(_frequent_meals, list) and _frequent_meals:
        logger.info(
            f"⏭️ [GAP 1 GATE/P1-LOW-SIGNAL] Señal 8 skipped (confidence baja): "
            f"{_frequent_meals[:5]} no inyectados como preferencia forzada."
        )

    # Señal 9: Tipos de comida que generan frustración emocional (más fuerte que "ignorados")
    # [P1-LOW-SIGNAL-FALLBACK] Gateada por confidence. Frustración emocional requiere
    # señal observable (nudge_outcomes con `frustrated=True`) — si el sistema la
    # inferió vía meta-learning sobre adherencia baja, no es señal genuina.
    _frustrated = actual_form_data.get("_frustrated_meal_types", [])
    if (not _skip_pref_signals) and isinstance(_frustrated, list) and _frustrated:
        fru_fmt = ", ".join([m.capitalize() for m in _frustrated])
        history_context += (
            "\n\n--- 😤 [CRON] COMIDAS QUE GENERAN FRUSTRACIÓN ---\n"
            f"El usuario expresó frustración/agobio asociados a estas comidas: {fru_fmt}.\n"
            "INSTRUCCIÓN OBLIGATORIA: Rediseña estas comidas con placer explícito (sabor, textura cremosa, calor). "
            "Evita presentaciones 'fit/dieta tradicional'. Si no logras reencantarlas, elimínalas del plan.\n"
            "----------------------------------------------------------------------\n"
        )
        logger.info(f"✅ [GAP 1] Señal 9: Comidas frustrantes inyectadas → {_frustrated}")



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
                    logger.info("🔄 [ROTACIÓN DE PLATOS] Generando nuevas recetas ESTRICTAMENTE con la misma despensa.")
                    
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
                    logger.info("🔄 [REGENERACIÓN] Usuario solicitó 'Generar Nueva Opción' el mismo día = RECHAZO del menú actual.")
                    
                    history_context += (
                        f"\n\n🚨 INSTRUCCIÓN DE VARIEDAD (RE-ROLL) 🚨\n"
                        f"El usuario quiere cambiar completamente las opciones de hoy:\n{', '.join(previous_meals)}\n"
                        f"REGLA CREATIVA: Inventa preparaciones inéditas. Cambia el método de cocción, la combinación o el corte para sorprender al usuario con algo nuevo.\n"
                        f"----------------------------------------------------------------------\n"
                    )
                    actual_form_data["_is_same_day_reroll"] = True
            else:
                logger.info("🌅 [NUEVO DÍA] Generación para un nuevo día iniciada.")
        except Exception as e:
            logger.warning(f"⚠️ Error validando regeneración del mismo día: {e}")


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
            # la DB (~50-200ms). Antes corrían secuenciales bloqueando el
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
            logger.info(f"🔍 [RAG] Query dinámica: {dynamic_query}")
            
            # Brecha 5: Validación de Caché RAG
            # P1-3: aget (no get) — el call site está dentro de arun_plan_pipeline
            # async; .get síncrono bloqueaba el event loop esperando Redis (típicamente
            # 5-50ms, pero acumulativo si Redis está lento o caído con DB fallback).
            rag_cache_key = f"rag_{user_id}_{dynamic_query}"
            cached_rag = await _LLM_CACHE.aget(rag_cache_key)
            if cached_rag is not None:
                logger.info(f"⚡ [CACHE HIT] Reutilizando contexto RAG para la misma query dinámica.")
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
                    logger.info(f"🧠 [RAG] Hechos textuales recuperados: {len(facts_data)} (ordenados por prioridad de categoría)")

                if visual_data:
                    visual_list = [f"• {item['description']}" for item in visual_data]
                    logger.info(f"📸 [VISUAL RAG] Entradas visuales recuperadas: {len(visual_data)}")

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
            logger.warning(f"⚠️ [RAG] Error recuperando memoria: {e}")
            
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
                logger.info(f"✂️ [PRUNING] {skipped_count} hechos de baja prioridad descartados completos (límite tokens)")
    
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
        logger.info(f"✅ [PRUNING] Contexto final: {estimate_tokens(full_rag_context)} tokens aprox (fact-by-fact, sin cortes)")
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
    _p134_req_token = request_id_var.set(req_id)
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
    _p134_uid_token = user_id_var.set(_rate_limit_uid)

    # P1-NEW-4: dict mutable per-pipeline para trackear pérdidas de eventos
    # SSE. Las tasks descendientes (asyncio.Task, run_in_executor) heredan
    # el contexto y mutan este mismo objeto. Al final del pipeline, los
    # totales se emiten como métrica `progress_cb`.
    _p134_cb_token = _pipeline_cb_stats_var.set({
        "dropped_cap": 0,
        "timed_out": 0,
        "failed_async": 0,
        "failed_sync": 0,
    })

    # [P1-34] Try/finally que garantiza el reset de ContextVars al
    # exit de la función (normal return o exception path).
    try:

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
            # [P0-5] _rejection_severity: escrito por `review_node` cuando
            # rechaza con clasificación crítica/high/minor. Consumer en
            # `should_retry` lo lee y aborta retry si crítico/high.
            # Default explícito "minor" en lugar de None: `dict.get(k, default)`
            # devuelve `None` cuando la key existe con valor None (NO el
            # default), por lo que el comentario antiguo afirmando equivalencia
            # entre None y "minor" era falso ante cualquier refactor futuro
            # que añadiera ramas (`severity in {...}`). Hardcodear "minor" desde
            # el inicio elimina la trampa. `should_retry` además normaliza con
            # `or "minor"` por si un nodo intermedio escribe None explícito.
            "_rejection_severity": "minor",
            # [P0-PIPE-1] Snapshot del best attempt (se llena por
            # `review_plan_node` en cada iteración). Inicializado a None/
            # vacíos: si la pipeline aborta antes de un solo review (e.g.
            # cancellation, exception muy temprana), el swap helper detecta
            # snapshot ausente y deja `plan_result` actual sin cambios.
            "_best_attempt_plan": None,
            "_best_attempt_severity": None,
            "_best_attempt_reasons": [],
            "_best_attempt_review_passed": False,
            "_best_attempt_number": None,
            # [P5-MARKER-APPROVED-1] Inicializa el flag de re-entrada del
            # surgical regen post-approval. False permite que la 1ra pasada
            # por `should_retry` con review aprobado + markers pendientes
            # enrute a `surgical_marker_regen`. Reset a False también en
            # `retry_reflection_node` para nuevos attempts.
            "_marker_regen_attempted": False,
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
    
        # [P6-CANCEL-PIPELINE-CHECK] Defense-in-depth: chequear cancel
        # registry directo dentro del loop de LangGraph (cada nodo del
        # grafo). Cubre el caso donde `asyncio.cancel()` del task externo
        # es atrapado por `asyncio.shield` interno de LLM calls — el
        # registry check es independiente de asyncio cancellation.
        # Bug observable corrida 2026-05-06 00:31: cancel POST llega al
        # backend, _pipeline_task.cancel() programa cancel asyncio, pero
        # el pipeline sigue avanzando porque shields atrapan el
        # CancelledError. Este check self-aborta el pipeline entre nodos.
        _pipeline_session_id = actual_form_data.get("session_id")

        async def run_graph():
            # Usamos astream con stream_mode="values".
            # P1-Q2: `_get_plan_graph()` construye el grafo lazy en la primera
            # request por proceso. Si el build falla, propaga la excepción al
            # `except` de afuera que dispara el fallback matemático (P0-1) en
            # lugar de devolver 5xx al cliente.
            plan_graph = _get_plan_graph()
            # Import lazy para evitar circular import (routers.plans importa
            # graph_orchestrator).
            try:
                from routers.plans import is_session_cancelled as _is_cancelled
            except Exception:
                _is_cancelled = lambda _sid: False
            # [P2-ORCH-12] recursion_limit explícito: red de seguridad si un
            # refactor futuro del retry deja un ciclo sin incrementar `attempt`
            # (GraphRecursionError rápido vs colgar hasta el timeout de 720s).
            async for event in plan_graph.astream(
                initial_state, stream_mode="values",
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            ):
                latest_state[0] = event
                # [P6-CANCEL-PIPELINE-CHECK] Self-abort entre nodos
                if _pipeline_session_id and _is_cancelled(_pipeline_session_id):
                    logger.warning(
                        f"🛑 [P6-CANCEL-PIPELINE-CHECK] Cancel detectado para "
                        f"session={_pipeline_session_id} entre nodos LangGraph. "
                        f"Abortando pipeline."
                    )
                    raise asyncio.CancelledError("user_cancelled_via_registry")
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
            logger.error(f"🚨 [EXTREME GRACEFUL DEGRADATION] Error crítico en pipeline ({type(e).__name__}): {e}")
            # [P1-SPEND-CAP-ALERT · 2026-05-28 · P1-ORCH-2 · 2026-05-28]
            # Distinguir el 429 "spending cap" de Gemini (persistente hasta subir
            # el cap) de un fallo transitorio. Reusa el detector canónico
            # (`_is_plan_spend_cap_error`, lazy import sin ciclo) sobre la
            # excepción que escapó, PERO también consulta el latch global
            # (`_plan_spend_cap_active`): durante un cap real, lo que escapa al
            # handler suele ser un SÍNTOMA downstream ("Circuit Breaker OPEN",
            # "Todos los workers fallaron") que NO contiene el string del cap —
            # sin el latch perdíamos la señal y mostrábamos el mensaje falso
            # "IA saturada, intenta en 1-2 min". El latch lo activó el primer
            # nodo que vio el 429 (vía `_record_cb_failure_unless_transient`).
            _spend_cap_hit = _is_plan_spend_cap_error(e) or _plan_spend_cap_active()
            final_state = latest_state[0] if latest_state else {}
            # [P1-26] Flush de TODOS los token buffers pendientes ANTES de
            # entregar el fallback. Sin esto, tokens acumulados en buffers
            # (especialmente `_default` para eventos no-day-keyed) se pierden
            # cuando `asyncio.wait_for(GLOBAL_PIPELINE_TIMEOUT_S)` cancela el
            # grafo a mitad de stream. La UI ya recibió un comienzo de
            # respuesta vía SSE; sin este flush, ve texto truncado y debe
            # esperar al evento de fallback completo. Best-effort: si el
            # callback explota (cliente desconectó, queue lleno), se logea y
            # seguimos al fallback — el flush no debe bloquear la entrega del
            # plan.
            try:
                _p126_flushed = _flush_all_token_buffers(final_state)
                if _p126_flushed:
                    logger.info(
                        f"[P1-26] Flush en degradación global: "
                        f"{_p126_flushed} buffer(s) drenados antes del fallback."
                    )
            except Exception as _p126_err:
                logger.debug(
                    f"[P1-26] Flush en degradación global falló (best-effort): "
                    f"{_p126_err!r}"
                )
            plan_partial = final_state.get("plan_result")
            # P0-1/P0-2: si no hay plan parcial usable, fallback total con la cantidad
            # de días solicitada. Si hay plan parcial (aunque sea con días vacíos o
            # incompletos), repararlo en lugar de descartar el trabajo del LLM.
            if not isinstance(plan_partial, dict) or not plan_partial:
                logger.warning(f"🛡️ [FALLBACK] Generando plan de emergencia matemático ({requested_days} días, by-pass de LLM)...")
                # P1-9: `_get_extreme_fallback_plan` ya setea `_is_fallback=True` dentro
                # del plan retornado. No duplicar en final_state — `arun_plan_pipeline`
                # retorna `final_state["plan_result"]` al caller, así que el flag en
                # final_state nunca se lee externamente y solo agregaba ambigüedad.
                final_state["plan_result"] = _get_extreme_fallback_plan(
                    nutrition,
                    actual_form_data.get("mainGoal", "Salud General"),
                    num_days=requested_days,
                    restricted_tokens=_fallback_restricted_tokens(actual_form_data),  # [P0-ORCH-1]
                )
                final_state["attempt"] = 1
                final_state["review_passed"] = True
            else:
                # P1-9: `_repair_partial_plan` ya setea `plan_partial["_is_fallback"]=True`
                # cuando hace cualquier reparación. Nada más que hacer aquí.
                _repair_partial_plan(plan_partial, nutrition=nutrition, requested_days=requested_days,
                                     restricted_tokens=_fallback_restricted_tokens(actual_form_data))  # [P0-ORCH-1]

            # [P1-SPEND-CAP-ALERT · 2026-05-28] Si el pipeline cayó por el spending
            # cap de Gemini: (1) marcar plan_result para que routers/plans.py emita
            # un mensaje honesto al usuario (reintentar NO ayuda hasta subir el cap),
            # (2) emitir system_alert al operador (idempotente, dedupe por alert_key
            # global). Ambos best-effort — no abortan la entrega del fallback.
            if _spend_cap_hit:
                _pr = final_state.get("plan_result")
                if isinstance(_pr, dict):
                    _pr["_llm_spend_cap"] = True
                _persist_gemini_spend_cap_alert(actual_form_data.get("user_id"))

        pipeline_duration = round(time.time() - pipeline_start, 2)

        # [P0-PIPE-1] Si el último intento empeoró respecto al mejor previo
        # (e.g. retry pasó de `minor` a `high` por skeleton fidelity roto),
        # restaurar el mejor antes de aplicar guardrails y antes del print
        # final — para que el banner de "Aprobado:" refleje el plan que el
        # usuario realmente recibirá.
        try:
            _swapped = _swap_to_best_attempt_if_better(final_state)
            if _swapped:
                logger.info(
                    "🔄 [P0-PIPE-1] Plan final restaurado a snapshot del mejor "
                    "intento previo (telemetría: `_best_attempt_swapped_from`)."
                )
                # [ROLLBACK-AGGREGATE-FIX 2026-05-07] Tras swap, los
                # `aggregated_shopping_list_*` del snapshot pueden estar
                # desincronizados entre ciclos (capturados en distintos
                # puntos del flow: pre-surgical, post-surgical pero
                # pre-rollback, etc.). Caso real plan 7ab9a552: weekly+
                # biweekly mostraban "Jamón de pavo" pero monthly no — el
                # mismo plan, misma comida = inconsistencia visible al
                # usuario al cambiar groceryDuration.
                # Fix: recomputar las 3 aggregates desde los `days`
                # restaurados para garantizar consistencia.
                try:
                    await _recompute_aggregates_after_swap(final_state)
                except Exception as _agg_err:
                    logger.warning(
                        f"[ROLLBACK-AGGREGATE-FIX] Re-aggregation tras swap falló "
                        f"(best-effort): {type(_agg_err).__name__}: {_agg_err}. "
                        f"Manteniendo aggregates del snapshot (puede haber "
                        f"inconsistencia entre ciclos)."
                    )
        except Exception as _swap_err:
            # Best-effort: si el helper revienta por estado corrupto, NO
            # tumbamos el pipeline — preservamos comportamiento previo
            # (entregar plan_result actual aunque sea peor) y logueamos.
            logger.error(
                f"⚠️ [P0-PIPE-1] _swap_to_best_attempt_if_better falló "
                f"(best-effort): {type(_swap_err).__name__}: {_swap_err}. "
                f"Continuando con plan_result actual sin swap."
            )

        logger.info(f"\n{'🔗' * 30}")
        logger.info(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
        logger.info(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
        logger.info("🔗" * 30 + "\n")

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
                restricted_tokens=_fallback_restricted_tokens(actual_form_data),  # [P0-ORCH-1]
            )
            # Marcar para que el caller (router/cron) NO lo persista como plan
            # real. `_is_fallback` ya lo setea `_get_extreme_fallback_plan`, pero
            # añadimos una marca específica de este path para que Grafana pueda
            # distinguir el origen del fallback en alerts.
            plan_to_return["_p1_5_emergency_return"] = True
            # [P3-FALLBACK-CLINICAL-LAYER · 2026-06-14] Este fallback de emergencia se construye DESPUÉS
            # de `_apply_final_defense_guardrails` → escapa la costura de allá. Aplícale la capa clínica
            # determinista aquí también (idempotente vía marker). Fail-safe, no debe tumbar el return.
            if FALLBACK_CLINICAL_LAYER_ENABLED:
                try:
                    _apply_deterministic_clinical_layer(plan_to_return, actual_form_data, nutrition)
                except Exception as _fcl5_e:
                    logger.warning(f"[P3-FALLBACK-CLINICAL-LAYER] capa clínica sobre fallback P1-5 "
                                   f"falló (no bloquea el return): {type(_fcl5_e).__name__}: {_fcl5_e}")

        # [P3-NEW-8 · 2026-05-11] Validador runtime de tipos/rangos del
        # contrato. Best-effort log-only — NUNCA raise (el caller espera
        # un dict, no una excepción).
        try:
            _ensure_plan_result_contract(plan_to_return, source="arun_plan_pipeline_return")
        except Exception as _contract_err:
            logger.debug(
                f"[P3-NEW-8/CONTRACT] validador falló (best-effort): {_contract_err}"
            )

        return plan_to_return
    finally:
        # [P1-34] Reset de ContextVars del pipeline para prevenir leakage
        # entre invocaciones consecutivas en el mismo contexto. Sin estos
        # `reset()`, los `set()` arriba persistían en el contexto del caller
        # tras el return: el siguiente pipeline (mismo task / sync wrapper /
        # test runner) heredaba el req_id, user_id y cb_stats del anterior,
        # contaminando logs de tracing y métricas SSE. asyncio.Task aísla
        # contextos en producción HTTP, pero los paths sync (cron / batch /
        # tests) NO tienen esa garantía. Cubre normal return + exception
        # path (try/finally garantiza el reset incluso si el pipeline falla).
        # Cada reset envuelto en su propio try porque ContextVar.reset puede
        # lanzar LookupError/ValueError si el token expiró por algún reason
        # exótico (e.g. evento de cancelación que ya hizo cleanup parcial).
        try:
            _pipeline_cb_stats_var.reset(_p134_cb_token)
        except (LookupError, ValueError):
            pass
        try:
            user_id_var.reset(_p134_uid_token)
        except (LookupError, ValueError):
            pass
        try:
            request_id_var.reset(_p134_req_token)
        except (LookupError, ValueError):
            pass

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


# [P3-KNOBS-INVENTORY-LATE-EMIT · 2026-05-15] Emit del inventario AL FINAL
# del módulo — captura TODOS los knobs declarados a nivel módulo vía
# `_env_*`, incluyendo los que viven downstream del cuerpo principal
# (e.g. `PROMPT_CACHE_SYSTEM_MESSAGE`, `PROMPT_TRIM_FORM_DATA`,
# `DAY_GEN_RETRY_USE_PRO`). Ver comentario en línea ~489 para contexto.
_log_active_knobs()
