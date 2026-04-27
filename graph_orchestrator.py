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
import uuid
import time
import asyncio
from contextlib import contextmanager, asynccontextmanager

class DistributedLLMSemaphore:
    """Semáforo distribuido usando Redis Sorted Sets para backpressure global.
    Aplica límite de concurrencia a través de múltiples workers Gunicorn/Uvicorn.
    Fallback a threading.Semaphore local si Redis no está disponible."""
    def __init__(self, max_concurrent=4, timeout_seconds=120):
        self.max_concurrent = max_concurrent
        self.timeout = timeout_seconds
        self.key = "semaphore:llm_global"
        self._local_semaphore = threading.Semaphore(max_concurrent)
        self._local_async_semaphore = None  # Inicializado lazymente en el loop

    def _get_async_semaphore(self):
        if self._local_async_semaphore is None:
            self._local_async_semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._local_async_semaphore

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
                        time.sleep(1.0)
                except Exception as e:
                    logger = logging.getLogger(__name__)
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
            async with self._get_async_semaphore():
                yield
            return

        req_id = str(uuid.uuid4())
        acquired = False
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
                        await asyncio.sleep(1.0)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Redis semaphore aasync error: {e}. Fallback to local async semaphore.")
                    async with self._get_async_semaphore():
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


LLM_SEMAPHORE = DistributedLLMSemaphore(max_concurrent=4, timeout_seconds=120)

class ChatGoogleGenerativeAI(_ChatGoogleGenerativeAI):
    """Wrapper para aplicar backpressure transparente a TODAS las llamadas LLM en LangGraph (Síncrono y Asíncrono)"""
    def invoke(self, *args, **kwargs):
        with LLM_SEMAPHORE.acquire():
            return super().invoke(*args, **kwargs)
            
    def stream(self, *args, **kwargs):
        with LLM_SEMAPHORE.acquire():
            yield from super().stream(*args, **kwargs)
            
    def generate(self, *args, **kwargs):
        with LLM_SEMAPHORE.acquire():
            return super().generate(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        async with LLM_SEMAPHORE.aacquire():
            return await super().ainvoke(*args, **kwargs)
            
    async def astream(self, *args, **kwargs):
        async with LLM_SEMAPHORE.aacquire():
            async for chunk in super().astream(*args, **kwargs):
                yield chunk
            
    async def agenerate(self, *args, **kwargs):
        async with LLM_SEMAPHORE.aacquire():
            return await super().agenerate(*args, **kwargs)

import concurrent.futures
from datetime import datetime, timezone
import random
import re as _re
import contextvars
import builtins
import uuid

# Mejora 8: ContextVar para Distributed Tracing
request_id_var = contextvars.ContextVar("request_id", default="SYS")

def custom_print(*args, **kwargs):
    req_id = request_id_var.get()
    msg = " ".join(str(a) for a in args)
    if not msg.startswith("["):
        msg = f"[{req_id}] {msg}"
    else:
        msg = f"[{req_id}] {msg}"
    builtins.print(msg, **kwargs)

print = custom_print
# NOTA: NO importar 'from agent import ...' a nivel de módulo → causa import circular
# (app → agent → tools → graph_orchestrator → agent). Se usa lazy import donde se necesite.
from cpu_tasks import _validar_repeticiones_cpu_bound, _normalize_meal_name
from constants import normalize_ingredient_for_tracking, strip_accents, TECHNIQUE_FAMILIES, ALL_TECHNIQUES, TECH_TO_FAMILY, SUPPLEMENT_NAMES
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from db import get_recent_techniques, get_recent_meals_from_plans, check_meal_plan_generated_today, search_user_facts, search_visual_diary, get_user_facts_by_metadata
from nutrition_calculator import get_nutrition_targets

import threading
logger = logging.getLogger(__name__)

from cache_manager import redis_client, redis_async_client
from db_core import execute_sql_query, execute_sql_write, aexecute_sql_query, aexecute_sql_write

class LLMCircuitBreaker:
    """Circuit breaker distribuido usando Redis INCR atómico.
    Seguro para multi-worker (Gunicorn/uvicorn --workers N).
    Fallback a DB si Redis no está disponible."""
    def __init__(self, failure_threshold=3, reset_timeout=30):
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures_key = "cb:llm:failures"
        self._open_key = "cb:llm:open"
        self._lock = threading.Lock()  # Fallback para modo sin-Redis
        self._alock = None   # Fallback async
        self._local_state_lock = threading.Lock() # P0-3: Lock unificado seguro (sync/async) para variables locales
        self._local_healthy = True     # Optimización para no golpear la DB si está sano
        self._last_db_check = 0        # TTL para el estado DB local

    @property
    def alock(self):
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    def record_failure(self):
        with self._local_state_lock:
            self._local_healthy = False
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
        if is_healthy and (time.time() - last_check) < 10:
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
        if is_healthy and (time.time() - last_check) < 10:
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
            res = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", ("llm_circuit_breaker",), fetch_one=True)
            if res:
                return res["value"] if isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception:
            pass
        return {"failures": 0, "last_failure": 0, "is_open": False}

    def _save_db_state(self, state):
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            ("llm_circuit_breaker", json.dumps(state))
        )

    async def arecord_failure(self):
        with self._local_state_lock:
            self._local_healthy = False
        if redis_async_client:
            try:
                failures = await redis_async_client.incr(self._failures_key)
                await redis_async_client.expire(self._failures_key, self.reset_timeout)
                if failures >= self.threshold:
                    await redis_async_client.set(self._open_key, "1", ex=self.reset_timeout)
                return
            except Exception as e:
                logger.warning(f"Redis async CB write error: {e}")
        async with self.alock:
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
        if is_healthy and (time.time() - last_check) < 10:
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
        async with self.alock:
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
        if is_healthy and (time.time() - last_check) < 10:
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
        async with self.alock:
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
            res = await aexecute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", ("llm_circuit_breaker",), fetch_one=True)
            if res:
                return res["value"] if isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception:
            pass
        return {"failures": 0, "last_failure": 0, "is_open": False}

    async def _asave_db_state(self, state):
        await aexecute_sql_write(
            "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            ("llm_circuit_breaker", json.dumps(state))
        )

_circuit_breaker = LLMCircuitBreaker()

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
_LLM_CACHE = PersistentLLMCache(ttl_seconds=300)
CACHE_TTL_SECONDS = 300

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
    Factores: cantidad de ingredientes, longitud de receta, técnicas complejas."""
    total_score = 0
    meal_count = 0
    days = plan.get("days", [])
    
    complex_techniques = ["horno", "guiso", "horneado", "lento", "marinado", "relleno", "empanizado"]
    
    for day in days:
        for meal in day.get("meals", []):
            meal_count += 1
            score = 1.0 # Base score
            
            # 1. Por cantidad de ingredientes
            ing_count = len(meal.get("ingredients", []))
            if ing_count > 8: score += 3.0
            elif ing_count > 5: score += 1.5
            elif ing_count <= 3: score -= 0.5
            
            # 2. Por pasos/longitud de receta
            recipe = meal.get("recipe", "")
            steps = len([s for s in recipe.split('\n') if s.strip()])
            if steps > 5: score += 3.0
            elif steps > 3: score += 1.5
            elif steps <= 2: score -= 0.5
            
            # 3. Técnicas complejas
            recipe_lower = recipe.lower()
            if any(tech in recipe_lower for tech in complex_techniques):
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
            metrics_data = {
                "user_id": uid if uid != "guest" else None,
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
                    logger.error(f"Failed to insert metric: {e}")
            
            bg_tasks = state.get("background_tasks")
            if bg_tasks and hasattr(bg_tasks, "add_task"):
                bg_tasks.add_task(_save)
            else:
                import threading
                threading.Thread(target=_save, daemon=True).start()
        except Exception as e:
            logger.error(f"⚠️ Error preparing metric insertion: {e}")

    cb = state.get("progress_callback")
    if cb:
        try:
            cb({"event": event, "data": data})
        except Exception:
            pass


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
    disliked = form_data.get("dislikedIngredients", [])
    
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
    
    compressor_llm = ChatGoogleGenerativeAI(
        model=_route_model(state.get("form_data", {}), force_fast=True),
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
        if not await _circuit_breaker.acan_proceed():
            print("⚠️ [COMPRESIÓN] Circuit Breaker OPEN. Saltando compresión.")
            return {"compressed_context": history_context}
            
        result = await compressor_llm.ainvoke(prompt)
        await _circuit_breaker.arecord_success()

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
        await _circuit_breaker.arecord_failure()
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
    from datetime import datetime, timezone, timedelta
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
    from constants import PLAN_CHUNK_SIZE
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
        f"{ctx['temporal_adherence_context']}\n\n"
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
    base_temp = 0.95 if is_re_roll else (0.7 if attempt == 1 else 0.9)

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [PLANIFICADOR] Reintento #{retry_state.attempt_number}...")
    )
    async def invoke_planner():
        if not await _circuit_breaker.acan_proceed():
            raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
        try:
            print(f"⏳ [PLANIFICADOR] Generando esqueleto del plan...")
            res = await planner_llm.ainvoke(prompt_text)
            await _circuit_breaker.arecord_success()
            return res
        except Exception as e:
            await _circuit_breaker.arecord_failure()
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
    base_temp = 0.95 if is_re_roll else 0.7
    attempt = state.get("attempt", 1)

    # --- INICIO MEJORA 2: Extraer días reciclados ANTES del worker para inyectar contexto ---
    from constants import PLAN_CHUNK_SIZE
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
        
        from datetime import datetime, timezone, timedelta
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
            f"{assignment_context}\n"
            f"{recycled_days_context}\n"
            f"{DAY_GENERATOR_SYSTEM_PROMPT}"
        )

        day_model = _route_model(form_data, attempt)
        
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

        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
            before_sleep=lambda rs: print(f"⚠️  [DÍA {day_num}] Reintento #{rs.attempt_number}...")
        )
        async def invoke_day():
            if not await _circuit_breaker.acan_proceed():
                raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
            try:
                messages = [HumanMessage(content=streaming_prompt)]
                accumulated_json = ""

                # Agent loop (hasta 4 interacciones de herramientas permitidas)
                for _ in range(4):
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
                                    result = consultar_nutricion.invoke(tool_args)
                                except Exception as e:
                                    result = f"Error ejecutando herramienta: {str(e)}"
                                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

                        _emit_progress(state, "token_reset", {"day": day_num})
                        accumulated_json = ""
                        continue
                    else:
                        # Generación finalizada sin más tools
                        break
                        
                await _circuit_breaker.arecord_success()
                
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
                await _circuit_breaker.arecord_failure()
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
    # Si un día tarda >HEDGE_AFTER segundos, se lanza un retry especulativo en paralelo.
    # Gana el primero que termine. Ceiling duro por día: HARD_CEILING segundos.
    # Esto elimina el patrón anterior "120s timeout → cancelar → retry sequential" que
    # acumulaba ~90s extra cuando un día era lento.
    HEDGE_AFTER = 45.0
    HARD_CEILING = 170.0

    async def _generate_day_hedged(skel_day: dict, day_num: int, temp_override: float = None) -> dict:
        day_start_time = time.time()
        primary = asyncio.create_task(generate_single_day(skel_day, day_num, temp_override))

        # 1. Esperar primary hasta el soft timeout
        done, _ = await asyncio.wait({primary}, timeout=HEDGE_AFTER)
        if primary in done:
            return primary.result()  # propaga excepciones normales

        # 2. Primary sigue corriendo → lanzar hedge especulativo
        elapsed = round(time.time() - day_start_time, 1)
        print(f"🪁 [HEDGE] Día {day_num} lleva >{elapsed}s. Lanzando intento especulativo en paralelo.")
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
                    print(f"🏁 [HEDGE] Día {day_num} ganador: {winner}")
                    return result
                except Exception as e:
                    last_exc = e
                    name = "primary" if t is primary else "hedge"
                    print(f"⚠️ [HEDGE] Día {day_num}: {name} falló ({type(e).__name__}). Esperando al otro.")
            racing = pending

        raise last_exc or RuntimeError(f"Día {day_num}: ambos intentos fallaron sin excepción")

    # Lanzar generaciones en paralelo (con hedging per-day)
    parallel_start = time.time()
    generated_days = []

    async def _safe_gen(skel_day, day_num, temp_override=None):
        try:
            result = await _generate_day_hedged(skel_day, day_num, temp_override)
            return day_num, result, None
        except Exception as e:
            return day_num, None, e

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
            import copy
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
        _ab_pair_selected = _select_ab_temp_pair(_ab_user_id) if _ab_user_id != "guest" else _AB_TEMP_PAIRS[1]
        _ab_temp_a = _ab_pair_selected["temp_a"]
        _ab_temp_b = _ab_pair_selected["temp_b"]
        print(f"⚔️ [ADVERSARIAL SELF-PLAY] Activado. Par '{_ab_pair_selected['label']}': A={_ab_temp_a} / B={_ab_temp_b}")
        cand_a_coro = _generate_candidate(temp_override=_ab_temp_a)
        cand_b_coro = _generate_candidate(temp_override=_ab_temp_b)
        candidate_a, candidate_b = await asyncio.gather(cand_a_coro, cand_b_coro)
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

def _select_ab_temp_pair(user_id: str) -> dict:
    """
    Thompson Sampling sobre pares de temperatura para el adversarial self-play.

    Fase exploración: round-robin hasta MIN_SAMPLES_PER_PAIR muestras por par.
    Fase explotación: infiere preferencia conservador/creativo del usuario y
    selecciona el par más alineado vía Beta(wins+1, losses+1).
    """
    try:
        from db_core import execute_sql_query
        rows = execute_sql_query(
            """SELECT metadata->>'pair_label' AS pair_label,
                      metadata->>'winner' AS winner
               FROM pipeline_metrics
               WHERE user_id = %s
                 AND node = 'adversarial_judge'
                 AND metadata->>'pair_label' IS NOT NULL
               ORDER BY created_at DESC LIMIT 90""",
            (user_id,)
        ) or []
    except Exception as e:
        print(f"⚠️ [AB-TEMP] Error leyendo historial: {e}")
        rows = []

    stats = {p["label"]: {"wins_a": 0, "wins_b": 0} for p in _AB_TEMP_PAIRS}
    for row in rows:
        lbl = row.get("pair_label")
        if lbl in stats:
            key = "wins_a" if row.get("winner") == "candidate_a" else "wins_b"
            stats[lbl][key] += 1

    totals = {lbl: s["wins_a"] + s["wins_b"] for lbl, s in stats.items()}
    min_total = min(totals.values())

    # Exploración: round-robin hasta cubrir MIN_SAMPLES_PER_PAIR por par
    if min_total < _AB_MIN_SAMPLES_PER_PAIR:
        under_explored = [p for p in _AB_TEMP_PAIRS if totals[p["label"]] == min_total]
        selected = random.choice(under_explored)
        print(f"🔬 [AB-TEMP] Exploración: par '{selected['label']}' ({totals[selected['label']]}/{_AB_MIN_SAMPLES_PER_PAIR} muestras)")
        return selected

    # Explotación: inferir preferencia conservador/creativo del usuario
    total_a_wins = sum(s["wins_a"] for s in stats.values())
    total_b_wins = sum(s["wins_b"] for s in stats.values())
    total_all = total_a_wins + total_b_wins
    conservative_ratio = total_a_wins / total_all if total_all > 0 else 0.5

    samples = {}
    for p in _AB_TEMP_PAIRS:
        lbl = p["label"]
        s = stats[lbl]
        total = totals[lbl]
        # Wins del candidato alineado con la preferencia global del usuario
        if conservative_ratio >= 0.6:
            aligned_wins = s["wins_a"]   # prefiere conservador → candidate_a es el valioso
        elif conservative_ratio <= 0.4:
            aligned_wins = s["wins_b"]   # prefiere creativo → candidate_b es el valioso
        else:
            aligned_wins = (s["wins_a"] + s["wins_b"]) // 2  # equilibrado → ambos aportan
        samples[lbl] = random.betavariate(aligned_wins + 1, max(1, total - aligned_wins) + 1)

    best_label = max(samples, key=lambda k: samples[k])
    selected = next(p for p in _AB_TEMP_PAIRS if p["label"] == best_label)
    print(f"🎯 [AB-TEMP] Explotación: par '{selected['label']}' (ratio_conservador={conservative_ratio:.2f})")
    return selected


class AdversarialJudgeResult(BaseModel):
    winner: Literal["candidate_a", "candidate_b"] = Field(description="El candidato seleccionado como ganador.")
    rationale: str = Field(description="Explicación breve de por qué este candidato maximiza la adherencia.")

async def adversarial_judge_node(state: PlanState) -> dict:
    """Evalúa dos candidatos de plan y selecciona el que mejor se adapte al perfil conductual del usuario."""
    candidate_b = state.get("candidate_b")
    if not candidate_b:
        # Adversarial mode was not triggered, fast-path
        return {}

    candidate_a = state.get("candidate_a")
    form_data = state["form_data"]

    print(f"\n{'='*60}")
    print(f"⚖️ [ADVERSARIAL JUDGE] Evaluando Candidato A vs Candidato B...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "adversarial_judging", "message": "Seleccionando el mejor plan candidato..."})
    start_time = time.time()

    judge_llm = ChatGoogleGenerativeAI(
        model=_route_model(form_data, force_fast=False), # Necesitamos buen razonamiento
        temperature=0.2,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=1
    ).with_structured_output(AdversarialJudgeResult)

    ctx = state.get("_cached_context", {})
    behavioral_profile = ctx.get("unified_behavioral_profile", "")
    adherence_context = ctx.get("adherence_context", "")
    fatigue_context = ctx.get("fatigue_context", "")

    # Solo pasamos la estructura básica de los días para ahorrar tokens
    def _compress_candidate(cand):
        summary = []
        for d in cand.get("days", []):
            day_str = f"Día {d.get('day')}:"
            for m in d.get("meals", []):
                day_str += f"\n  - {m.get('meal')}: {m.get('name')} (Ingredientes: {', '.join(m.get('ingredients', []))})"
            summary.append(day_str)
        return "\n".join(summary)

    prompt = f"""
Eres un juez clínico experto en nutrición y adherencia conductual. 
Tu objetivo es seleccionar el MEJOR de dos planes candidatos para un usuario, maximizando la probabilidad de que NO abandone el plan.

--- PERFIL CONDUCTUAL Y RESTRICCIONES ---
{behavioral_profile}
{adherence_context}
{fatigue_context}

--- CANDIDATO A (Conservador) ---
{_compress_candidate(candidate_a)}

--- CANDIDATO B (Creativo) ---
{_compress_candidate(candidate_b)}

EVALÚA:
1. ¿Cuál candidato respeta mejor las reglas estrictas (alergias, disgustos, fatiga)?
2. ¿Cuál se adapta mejor al estado emocional y adherencia histórica?
3. ¿Cuál ofrece mejor variedad sin volverse demasiado complejo si el usuario tiene baja adherencia?

Selecciona el ganador ("candidate_a" o "candidate_b") y da una breve justificación.
"""

    try:
        if not await _circuit_breaker.acan_proceed():
            raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
        
        result: AdversarialJudgeResult = await judge_llm.ainvoke(prompt)
        await _circuit_breaker.arecord_success()
        
        winner_key = result.winner
        rationale = result.rationale
        print(f"🏆 [ADVERSARIAL JUDGE] Ganador: {winner_key}. Razón: {rationale}")
        
        # Persistir el resultado para RLHF (logging por ahora)
        user_id = form_data.get("user_id", "guest")
        if user_id != "guest":
            try:
                from db_core import execute_sql_write
                loser_key = "candidate_b" if winner_key == "candidate_a" else "candidate_a"
                loser_cand = candidate_b if loser_key == "candidate_b" else candidate_a
                # En un futuro, guardar en db_profiles o metrics el perdedor para calibración
                # Recuperar metadatos del par AB usado (propagados por day_generator_node en el estado)
                _ab_meta = state.get("_ab_temp_meta") or {}
                execute_sql_write(
                    "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, tokens_estimated, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (user_id, state.get("request_id"), "adversarial_judge", int((time.time() - start_time) * 1000), len(prompt)//4, 1.0, json.dumps({
                        "winner": winner_key,
                        "rationale": rationale,
                        "loser_snapshot": _compress_candidate(loser_cand),
                        "pair_label": _ab_meta.get("label", "balanced"),
                        "temp_a": _ab_meta.get("temp_a", 0.7),
                        "temp_b": _ab_meta.get("temp_b", 0.95),
                        "winning_temperature": _ab_meta.get("temp_a") if winner_key == "candidate_a" else _ab_meta.get("temp_b"),
                    }))
                )
            except Exception as e:
                print(f"⚠️ [ADVERSARIAL JUDGE] Error guardando métrica: {e}")

        # Update the plan_result to be the winner's payload
        winner_cand = state.get(winner_key)
        if isinstance(winner_cand, dict):
            winner_cand["_adversarial_winner"] = winner_key
        return {
            "plan_result": winner_cand,
            "adversarial_rationale": rationale
        }
        
    except Exception as e:
        await _circuit_breaker.arecord_failure()
        print(f"⚠️ [ADVERSARIAL JUDGE] Falló la evaluación: {e}. Defaulting to Candidate A.")
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
    needs_correction: bool = Field(description="True si >=2 scores son < 6, o si algún score es < 4")
    suggestions: str = Field(description="Si needs_correction es True, especifica exactamente qué cambiar.")

class CorrectedDays(BaseModel):
    days: list[SingleDayPlanModel] = Field(description="Lista de los 3 días con las correcciones aplicadas.")

# Número máximo de días que el self-critique corregirá en un solo run
_SELF_CRITIQUE_MAX_DAYS = 2

# Staples de desayuno/merienda que tienden a repetirse silenciosamente entre días.
# El ANTI MODE-COLLAPSE solo rota proteína/carbo principal, no estos.
# Map: etiqueta canónica → aliases buscados (case-insensitive, sin acentos).
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
}


def _count_staple_repetitions(days: list) -> dict:
    """Cuenta en cuántos días distintos aparece cada staple. Devuelve solo staples
    que aparecen en >=2 días (señal de mode-collapse a nivel de staples).
    """
    import unicodedata as _ud

    def _norm(s: str) -> str:
        s = _ud.normalize("NFD", s.lower()).encode("ascii", "ignore").decode("ascii")
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

    evaluator_llm = ChatGoogleGenerativeAI(
        model=_route_model(state.get("form_data", {}), force_fast=True),
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
        suggested_day_hint = ""

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
    {staples_block}{user_context}

    Evalúa del 1 al 10:
    1. Atractivo visual (¿Se lee apetitoso o son combinaciones raras?)
    2. Diversidad real de sabores. Penaliza con score <=4 si:
       - Se repite la misma proteína o guarnición principal con nombres distintos.
       - Un staple (avena, claras, pan, yogurt, queso, lechosa, guineo, plátano maduro, aguacate, tortilla)
         aparece en 2+ días (ver bloque 'STAPLES REPETIDOS' arriba si está presente).
    3. Coherencia cultural Dominicana (¿El desayuno tiene sentido? ¿La cena es coherente?)
    4. Balance de temperaturas (¿Hay 3 días seguidos de ensaladas frías o todo es sopa?)

    Si DOS O MÁS scores son < 6, o si ALGÚN score es < 4, marca needs_correction=True y da instrucciones CLARAS Y CORTAS de qué cambiar, mencionando explícitamente el día (ej. "Día 2").
    {f"Pista: empieza por {suggested_day_hint} si necesitas elegir cuál corregir primero." if suggested_day_hint else ""}
    """
    
    try:
        if not await _circuit_breaker.acan_proceed():
            raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
        try:
            critique: CritiqueEvaluation = await evaluator_llm.ainvoke(prompt)
            await _circuit_breaker.arecord_success()
        except Exception as e:
            await _circuit_breaker.arecord_failure()
            raise e
        print(f"📊 [SELF-CRITIQUE] Scores -> Visual: {critique.visual_score}, Diversidad: {critique.diversity_score}, Cultural: {critique.cultural_score}, Temp: {critique.temperature_score}")
        
        if critique.needs_correction:
            print(f"⚠️ [SELF-CRITIQUE] Problemas detectados. Sugerencias: {critique.suggestions}")

            # Parsear qué días necesitan corrección desde el texto de sugerencias
            mentioned = list(dict.fromkeys(
                int(d) for d in _re.findall(r'[Dd]ía\s*(\d)', critique.suggestions)
            ))
            if not mentioned:
                mentioned = [1]  # Default: corregir día 1 si no se menciona ninguno
            print(f"🔧 [SELF-CRITIQUE] Corrigiendo días afectados: {mentioned[:_SELF_CRITIQUE_MAX_DAYS]}")

            corrector_llm = ChatGoogleGenerativeAI(
                model=_route_model(form_data, force_fast=True),
                temperature=0.3,
                google_api_key=os.environ.get("GEMINI_API_KEY"),
                max_retries=0,
                timeout=55,
            ).with_structured_output(SingleDayPlanModel)

            ctx = _build_shared_context(state)

            # Extraer asignaciones del skeleton para inyectarlas en cada corrector
            _skeleton = state.get("plan_result", {}).get("_skeleton", {})
            _skeleton_days = _skeleton.get("days", [])

            async def _correct_single_day(day_num: int):
                target_day = next((d for d in days if d.get("day") == day_num), None)
                if not target_day:
                    return day_num, None
                if not await _circuit_breaker.acan_proceed():
                    print(f"⚠️ [SELF-CRITIQUE] Circuit Breaker OPEN. Saltando corrección Día {day_num}.")
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

                    corrected_result: SingleDayPlanModel = await asyncio.wait_for(
                        corrector_llm.ainvoke(correction_prompt),
                        timeout=45.0
                    )
                    await _circuit_breaker.arecord_success()
                    if corrected_result:
                        corrected_day = corrected_result.model_dump()
                        corrected_day["day"] = day_num
                        print(f"✅ [SELF-CRITIQUE] Día {day_num} corregido exitosamente.")
                        return day_num, corrected_day
                except asyncio.TimeoutError:
                    print(f"⏱️ [SELF-CRITIQUE] Timeout corrigiendo Día {day_num} (45s). Manteniendo original.")
                except Exception as e:
                    await _circuit_breaker.arecord_failure()
                    print(f"⚠️ [SELF-CRITIQUE] Error corrigiendo Día {day_num}: {e}. Manteniendo original.")
                return day_num, None

            days_to_fix = mentioned[:_SELF_CRITIQUE_MAX_DAYS]
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

    if state.get("semantic_cache_hit"):
        cached_plan = state["cached_plan_data"] or {}
        skeleton = {}
        
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
                    
        result = {
            "main_goal": nutrition.get("goal_label", ""),
            "insights": cached_plan.get("insights") or [],
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

    # Renumerar días y asignar day_name obligatoriamente (para parchar planes del semantic_cache)
    days_offset = form_data.get("_days_offset", 0)
    
    from datetime import datetime, timezone, timedelta
    start_date_str = form_data.get("_plan_start_date")
    if start_date_str:
        try:
            start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        except Exception:
            start_dt = datetime.now(timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc)
    
    # Ajustar al timezone local del usuario (tzOffset viene en minutos, ej. 240 = UTC-4)
    tz_offset_minutes = form_data.get("tzOffset", 0)
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
    
    # 1. Macro Balancing Post-Assembly
    # Ajusta porciones si un día se desvió del target calórico por más de 100 kcal.
    for day in days:
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
            
            largest_meal["cals"] += adjustment
            
            # Ajustar macros (60% del ajuste vía carbs @4kcal/g, 40% vía fats @9kcal/g)
            carb_adj_kcal = abs(adjustment) * 0.6
            fat_adj_kcal = abs(adjustment) * 0.4
            carb_delta = int(carb_adj_kcal / 4)  # 1g carb = 4 kcal
            fat_delta = int(fat_adj_kcal / 9)     # 1g fat = 9 kcal
            if adjustment < 0:
                largest_meal["carbs"] = max(0, largest_meal.get("carbs", 0) - carb_delta)
                largest_meal["fats"] = max(0, largest_meal.get("fats", 0) - fat_delta)
            else:
                largest_meal["carbs"] = largest_meal.get("carbs", 0) + carb_delta
                largest_meal["fats"] = largest_meal.get("fats", 0) + fat_delta
                
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
                meal["cals"] = int(meal.get("cals", 0) * scale_factor)
                meal["protein"] = int(meal.get("protein", 0) * scale_factor)
                meal["carbs"] = int(meal.get("carbs", 0) * scale_factor)
                meal["fats"] = int(meal.get("fats", 0) * scale_factor)
                
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
                        
                        largest_meal["cals"] -= transfer
                        meal["cals"] += transfer
                        print(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy bajo ({current_cals}kcal). Transferidos {transfer}kcal desde '{largest_meal.get('meal')}'")
                        
                        scale_up = meal["cals"] / max(current_cals, 1)
                        meal["protein"] = int(meal.get("protein", 0) * scale_up)
                        meal["carbs"] = int(meal.get("carbs", 0) * scale_up)
                        meal["fats"] = int(meal.get("fats", 0) * scale_up)
                        
                        scale_down = largest_meal["cals"] / max(largest_cals_before, 1)
                        largest_meal["protein"] = int(largest_meal.get("protein", 0) * scale_down)
                        largest_meal["carbs"] = int(largest_meal.get("carbs", 0) * scale_down)
                        largest_meal["fats"] = int(largest_meal.get("fats", 0) * scale_down)

            elif current_cals > max_cal:
                excess = current_cals - max_cal
                smallest_meal = min([m for m in day_meals if m != meal], key=lambda m: m.get("cals", float('inf')))
                if smallest_meal:
                    smallest_cals_before = smallest_meal.get("cals", 0)
                    
                    meal["cals"] -= excess
                    smallest_meal["cals"] += excess
                    print(f"⚖️ [MEAL COHERENCE] Día {day.get('day')}: '{meal.get('meal')}' muy alto ({current_cals}kcal). Transferidos {excess}kcal hacia '{smallest_meal.get('meal')}'")
                    
                    scale_down = meal["cals"] / max(current_cals, 1)
                    meal["protein"] = int(meal.get("protein", 0) * scale_down)
                    meal["carbs"] = int(meal.get("carbs", 0) * scale_down)
                    meal["fats"] = int(meal.get("fats", 0) * scale_down)
                    
                    scale_up = smallest_meal["cals"] / max(smallest_cals_before, 1)
                    smallest_meal["protein"] = int(smallest_meal.get("protein", 0) * scale_up)
                    smallest_meal["carbs"] = int(smallest_meal.get("carbs", 0) * scale_up)
                    smallest_meal["fats"] = int(smallest_meal.get("fats", 0) * scale_up)


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

        aggr_list_7 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=1.0 * household) if _uid else []
        aggr_list_15 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=2.0 * household) if _uid else []
        aggr_list_30 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=4.0 * household) if _uid else []

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
    try:
        from humanize_ingredients import humanize_plan_ingredients
        result = humanize_plan_ingredients(result)
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
    
    # Brecha 3: Stopwords para extraer sustantivos núcleo de los ingredientes
    stopwords = {"cdta", "semillas", "semilla", "guineo", "guineítos", "guineito", "guineitos", "esencia", "extracto", "polvo", "jugo", "zumo", "salsa", "pasta", "concentrado", "caldo", "gotas", "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "taza", "tazas", "cucharada", "cucharadas", "cucharadita", "cucharaditas", "cdita", "cditas", "g", "ml", "oz", "libra", "libras", "kg", "litro", "litros", "pizca", "al", "gusto", "para", "con", "y", "o", "fresco", "fresca", "frescos", "frescas", "picado", "picada", "molido", "molida", "rallado", "rallada", "cocido", "cocida", "crudo", "cruda", "mediano", "grande", "pequeño", "rebanada", "rebanadas", "diente", "dientes", "filete", "filetes", "porción", "porcion", "sobre", "proteína", "proteina", "carbohidratos", "carbohidrato", "vegetales", "vegetal", "grasas", "grasa", "macronutriente", "macronutrientes", "opcional", "acompañamiento", "acompañante", "unidad", "unidades", "lonja", "lonjas", "pote", "potes", "lata", "latas", "puñado", "manojo", "hoja", "hojas", "rama", "ramas", "vaso", "vasos", "botella", "botellas", "paquete", "paquetes", "bolsa", "bolsas", "gramos", "mililitros", "onzas", "pedazo", "pedazos", "trozo", "trozos", "mitad", "cuarto", "tercio", "entero", "entera",
                # Adjetivos de madurez y preparación — si aparecen solos (sin sustantivo) no son ingredientes
                "maduro", "madura", "maduros", "maduras",
                "verde", "verdes",  # excepción: "plátano verde" → "plátano" será el core noun
                "hervido", "hervida", "hervidos", "hervidas",
                "asado", "asada", "asados", "asadas",
                "frito", "frita", "fritos", "fritas",
                "desalado", "desalada", "remojado", "remojada",
                "fileteado", "fileteada", "entero", "entera",
                "natural", "naturales"}


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
    try:
        from schemas import PlanModel
        PlanModel(**result)
    except Exception as e:
        print(f"🚨 [ASSEMBLY VALIDATION] Plan corrupto post-assembly detectado: {e}")

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
    severity: Literal["none", "minor", "critical"] = Field(default="none")
    affected_days: list[int] = Field(default_factory=list, description="Lista de números de día (1, 2, o 3) afectados por los problemas. Vacío si aplica a todos o a ninguno.")

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
    
    # Extraer restricciones del usuario
    allergies = form_data.get("allergies", [])
    # Combinar alergias del array con las escritas en texto libre (otherAllergies)
    other_allergies_text = form_data.get("otherAllergies", "")
    if other_allergies_text:
        # Separar por comas si el usuario escribió varias
        extra_allergies = [a.strip() for a in other_allergies_text.replace(",", ",").split(",") if a.strip()]
        allergies = list(allergies) + extra_allergies
    
    medical_conditions = form_data.get("medicalConditions", [])
    # Combinar condiciones del array con las escritas en texto libre (otherConditions)
    other_conditions_text = form_data.get("otherConditions", "")
    if other_conditions_text:
        extra_conditions = [c.strip() for c in other_conditions_text.replace(",", ",").split(",") if c.strip()]
        medical_conditions = list(medical_conditions) + extra_conditions
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
            from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
            
            print("🔬 [FACT-CHECKING] Iniciando investigación de alergias/condiciones...")
            fact_checker_llm = ChatGoogleGenerativeAI(
                model=_route_model(form_data, force_fast=True),
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
                    fc_response = await fact_checker_llm.ainvoke(fc_messages)
                    fc_messages.append(fc_response)
                    
                    if not fc_response.tool_calls:
                        fact_check_report = fc_response.content
                        break
                        
                    for tool_call in fc_response.tool_calls:
                        if tool_call["name"] == "consultar_base_datos_medica":
                            tool_res = consultar_base_datos_medica.invoke(tool_call["args"])
                            print(f"🔍 [FACT-CHECK] Consulta DB: {tool_call['args']} -> {str(tool_res)[:80]}...")
                            fc_messages.append(ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=str(tool_res)
                            ))
                except Exception as fc_e:
                    print(f"⚠️ [FACT-CHECK] Error durante la investigación: {fc_e}")
                    fact_check_report = f"Error en la investigación: {str(fc_e)}. Asumir precaución máxima."
                    break

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
        
        reviewer_llm = ChatGoogleGenerativeAI(
            model=_route_model(state.get("form_data", {}), attempt=1),
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
            if not await _circuit_breaker.acan_proceed():
                raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
            try:
                res = await reviewer_llm.ainvoke(review_prompt)
                await _circuit_breaker.arecord_success()
                return res
            except Exception as e:
                await _circuit_breaker.arecord_failure()
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
        severity = "minor"
    
    # ============================================================
    # VALIDACIÓN DETERMINISTA DE DESPENSA Y ANTI-REPETICIÓN (Post-LLM)
    # Verifica que el plan cumpla restricciones de inventario y no repita platos.
    # ============================================================
    if approved:
        # 1. Validación Estricta de Despensa (Pantry Guardrail)
        is_rotation = form_data.get("_is_rotation_reroll", False)
        if is_rotation:
            from constants import validate_ingredients_against_pantry
            current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
            
            clean_pantry = []
            if current_pantry and isinstance(current_pantry, list):
                clean_pantry = [item.strip() for item in current_pantry if item and isinstance(item, str) and len(item) > 2]
                
            if clean_pantry:
                val_result = validate_ingredients_against_pantry(all_ingredients, clean_pantry, strict_quantities=False)
                if val_result is not True:
                    approved = False
                    issues.append(val_result)  # val_result es el string de error generado por constants.py
                    severity = "high"
                    print(f"🚨 [PANTRY GUARD] Validación fallida en Revisor Médico.")
                else:
                    print(f"✅ [PANTRY GUARD] Todos los ingredientes cumplen con la despensa.")

    # 2. Validación Anti-Repetición
    if approved:
        try:
            user_id = form_data.get("user_id") or form_data.get("session_id")
            if user_id and user_id != "guest":

                recent_meal_names = get_recent_meals_from_plans(user_id, days=3)
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
                        severity = "minor"
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
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            future = executor.submit(
                                _validar_repeticiones_cpu_bound,
                                guest_recent,
                                days
                            )
                            repeated_meals = future.result()
                        filtered_guest_repeated = [rm for rm in repeated_meals if not any(g in rm for g in ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa'])]
                        if len(filtered_guest_repeated) > 0:
                            approved = False
                            issues.append(
                                f"REPETICIÓN DETECTADA (Guest): {', '.join(filtered_guest_repeated)}. Regenerar con variantes diferentes."
                            )
                            severity = "minor"
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
                    severity = "minor"
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
        user_id = form_data.get("user_id")
        if user_id and user_id != "guest":
            try:
                from db_profiles import get_user_profile, update_user_health_profile
                profile = get_user_profile(user_id)
                if profile:
                    hp = profile.get("health_profile") or {}
                    rejection_patterns = hp.get("rejection_patterns", [])
                    # Evitar duplicados exactos y mantener historial manejable
                    for issue in issues:
                        if issue not in rejection_patterns:
                            rejection_patterns.append(issue)
                    # Mantener solo los últimos 10 patrones
                    hp["rejection_patterns"] = rejection_patterns[-10:]
                    update_user_health_profile(user_id, hp)
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
    """Decide si regenerar el plan o enviarlo al usuario."""
    MAX_ATTEMPTS = 2
    # Mínimo de tiempo necesario para un retry completo: reflexión + planificador + 3 días paralelos
    MIN_RETRY_BUDGET_SECONDS = 180
    GLOBAL_TIMEOUT = 600

    if state.get("review_passed", False):
        print("✅ [ORQUESTADOR] Revisión aprobada → Enviando al usuario.")
        return "end"

    severity = state.get("_rejection_severity", "minor")

    # Brecha 6: Retry por Severidad
    if severity == "critical":
        print("🚨 [ORQUESTADOR] Rechazo CRÍTICO → No tiene sentido reintentar con el mismo contexto. Abortando temprano.")
        return "end"

    if state.get("attempt", 0) >= MAX_ATTEMPTS:
        if not state.get("review_passed", False):
            print(f"🚨 [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada → Tolerando y enviando mejor versión disponible.")
            return "end"
        print(f"⚠️  [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado → Enviando mejor versión disponible.")
        return "end"

    # Guard de presupuesto: si no hay tiempo suficiente para un retry completo, no intentarlo
    start = state.get("pipeline_start")
    if start:
        elapsed = time.time() - start
        remaining = GLOBAL_TIMEOUT - elapsed
        if remaining < MIN_RETRY_BUDGET_SECONDS:
            print(f"⏰ [ORQUESTADOR] Sin presupuesto para retry ({remaining:.0f}s restantes < {MIN_RETRY_BUDGET_SECONDS}s mínimo). Enviando mejor versión disponible.")
            return "end"

    print("🔄 [ORQUESTADOR] Revisión fallida → Regenerando plan con correcciones...")
    return "retry"


# ============================================================
# NODO 0: REFLEXIÓN META-LEARNING
# ============================================================
from pydantic import BaseModel, Field

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
    import hashlib
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
        reflector_llm = ChatGoogleGenerativeAI(
            model=_route_model(form_data, force_fast=True), # Modelo rápido para reflexiones
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
        
        if not await _circuit_breaker.acan_proceed():
            raise Exception("Circuit Breaker OPEN - LLM cascade failure prevented")
        try:
            result = await reflector_llm.ainvoke(prompt)
            await _circuit_breaker.arecord_success()
        except Exception as e:
            await _circuit_breaker.arecord_failure()
            raise e
        reflection_text = result.reflection
        print(f"💡 [META-LEARNING] Diagnóstico: {reflection_text}")
        
        # --- PERSISTENCIA Y CARGA HISTÓRICA (GAP 1) ---
        user_id = form_data.get("user_id") or form_data.get("session_id")
        historical_reflections_text = ""
        
        if user_id and user_id != "guest":
            from db_profiles import get_user_profile, update_user_health_profile
            from datetime import datetime, timezone
            
            profile = get_user_profile(user_id)
            if profile:
                health_profile = profile.get("health_profile") or {}
                reflection_history = health_profile.get("reflection_history", [])
                
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
                    
                health_profile["reflection_history"] = reflection_history
                update_user_health_profile(user_id, health_profile)
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
    
    import copy
    new_form_data = copy.deepcopy(form_data)
    auto_adjusted = False
    
    try:
        from db_core import execute_sql_query
        import json
        
        # Mejora 7: Leer el last_pipeline_score (Holistic Score) directo del perfil
        profile_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if profile_res and profile_res.get("health_profile"):
            hp = profile_res["health_profile"] if isinstance(profile_res["health_profile"], dict) else json.loads(profile_res["health_profile"])
            last_score = hp.get("last_pipeline_score")
            if last_score is not None:
                print(f"🧠 [META-LEARNING] last_pipeline_score detectado: {last_score}")
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
        if not auto_adjusted:
            recent_data = execute_sql_query(
                "SELECT node, duration_ms, confidence, metadata FROM pipeline_metrics WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
                (user_id,), fetch_all=True
            )
            
            if recent_data:
                critique_metrics = [m for m in recent_data if m["node"] == "self_critique"]
                avg_critique_corrections = 0
                if critique_metrics:
                    avg_critique_corrections = sum(
                        1 for m in critique_metrics 
                        if (json.loads(m["metadata"]) if isinstance(m.get("metadata"), str) else m.get("metadata", {})).get("needs_correction")
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
    """Inyecta contexto de rechazo como directiva para el retry (GAP 1)."""
    reasons = state.get("rejection_reasons", [])
    attempt = state.get("attempt", 1) + 1
    
    update_data = {"attempt": attempt}
    if reasons:
        directive = f"El plan anterior fue RECHAZADO por: {'; '.join(reasons)}. MUTA DRÁSTICAMENTE la estrategia."
        print(f"🔄 [RETRY REFLECTION] Intento {attempt}. Directiva inyectada: {directive}")
        update_data["reflection_directive"] = directive
    return update_data

def semantic_cache_check_node(state: PlanState) -> dict:
    """Busca un plan similar en la base de datos usando similitud de coseno para saltar la generación LLM."""
    profile_embedding = state.get("profile_embedding")
    actual_form_data = state.get("form_data", {})
    
    # 1. No usar caché si el usuario pidió un re-roll forzado hoy (quiere algo distinto explícitamente)
    is_reroll = actual_form_data.get("_is_same_day_reroll") or actual_form_data.get("_is_rotation_reroll")
    if is_reroll:
        return {"semantic_cache_hit": False, "cached_plan_data": None}
        
    if profile_embedding:
        # P1-2: Búsqueda semántica ampliada (límite 5) para poder aplicar filtros de frescura y médicos
        similar_plans = search_similar_plan(profile_embedding, threshold=0.98, limit=5)
        
        valid_plan = None
        plan_data = None
        
        if similar_plans:
            from datetime import datetime, timezone
            from constants import safe_fromisoformat
            now_utc = datetime.now(timezone.utc)
            
            for plan_candidate in similar_plans:
                cand_data = plan_candidate.get("plan_data")
                if not cand_data or not isinstance(cand_data, dict):
                    continue
                    
                # 🛡️ Evitar que la caché semántica sirva planes de Fallback Matemático
                is_fallback = False
                for day in cand_data.get("days", []):
                    if "Contingencia de Emergencia" in day.get("daily_summary", "") or "Matemáticamente" in day.get("daily_summary", ""):
                        is_fallback = True
                        break
                        
                if is_fallback:
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
                        
                # P1-2 (b): Compatibilidad Médica Estricta
                cached_form = cand_data.get("form_data") or cand_data.get("metadata", {}).get("form_data", {})
                if cached_form:
                    def _normalize(val):
                        if not val: return ""
                        if isinstance(val, str): return ",".join(sorted([v.strip().lower() for v in val.split(",")]))
                        if isinstance(val, list): return ",".join(sorted([str(v).strip().lower() for v in val]))
                        return ""
                    
                    curr_allergies = _normalize(actual_form_data.get("allergies"))
                    cache_allergies = _normalize(cached_form.get("allergies"))
                    curr_medical = _normalize(actual_form_data.get("medicalConditions"))
                    cache_medical = _normalize(cached_form.get("medicalConditions"))
                    
                    if curr_allergies != cache_allergies:
                        print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de alergias post-hoc.")
                        continue
                    if curr_medical != cache_medical:
                        print(f"🗑️ [SEMANTIC CACHE] Plan descartado por cambio de condiciones médicas post-hoc.")
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
                    recent_meal_names = get_recent_meals_from_plans(user_id, days=3)
                    if recent_meal_names:
                        # Extraer nombres de platos del plan cacheado
                        cached_meal_names = []
                        for day in plan_data.get("days", []):
                            for meal in day.get("meals", []):
                                if isinstance(meal, dict) and meal.get("name"):
                                    cached_meal_names.append(meal.get("name"))
                        
                        # Validar — la función espera days_plan (lista de días con 'meals'), no nombres planos
                        repeated_meals = _validar_repeticiones_cpu_bound(recent_meal_names, plan_data.get("days", []))
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

# Module-level singleton: el grafo compilado es stateless y reutilizable entre requests.
_PLAN_GRAPH = build_plan_graph()


# ============================================================
# FUNCIÓN PÚBLICA: Ejecutar el pipeline completo
# ============================================================
async def arun_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None, previous_ai_error: str = None, background_tasks=None) -> dict:
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
    import copy
    actual_form_data = copy.deepcopy(form_data)
    
    # 0.1 Sanitización de campos de texto libre del usuario (anti-prompt-injection)
    _PROMPT_INJECTION_PATTERNS = [
        "ignore previous", "ignore above", "ignore all", "disregard",
        "system prompt", "you are now", "act as", "new instructions",
        "override", "forget everything", "reveal your", "output all",
        # Patrones en Español
        "ignora", "olvida", "nuevas instrucciones", "actua como",
        "eres ahora", "revela tu", "imprime tu",
    ]
    _SANITIZE_FIELDS = ["mainGoal", "otherAllergies", "otherConditions", "struggles", "notes"]
    
    import unicodedata
    def _normalize_text(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
        
    for field in _SANITIZE_FIELDS:
        val = actual_form_data.get(field)
        if isinstance(val, str):
            val_norm = _normalize_text(val)
            if any(p in val_norm for p in _PROMPT_INJECTION_PATTERNS):
                logger.warning(f"🛡️ [SANITIZE] Prompt injection pattern detected in '{field}': {val[:80]}")
                actual_form_data[field] = ""  # Neutralizar campo sospechoso
    
    # 1. Pre-calcular nutrición
    nutrition = get_nutrition_targets(actual_form_data)
    
    # 2. Preparar contexto del historial (memoria inteligente y platos recientes)
    history_context = ""
    if memory_context:
        if len(memory_context) > 10000:
            logger.warning(f"⚠️ [ORQUESTADOR] memory_context excede los 10k caracteres ({len(memory_context)}). Truncando para evitar exceder tokens.")
            memory_context = memory_context[:10000] + "\n...[TRUNCADO POR LIMITE DE TOKENS]..."
        history_context = memory_context + "\n"
        
    if previous_ai_error:
        history_context += (
            "\n\n--- ⚠️ AUTO-CORRECCIÓN (INTENTO ANTERIOR FALLIDO) ⚠️ ---\n"
            f"El intento anterior de generar el plan falló debido a este error:\n"
            f"'{previous_ai_error}'\n"
            "INSTRUCCIÓN OBLIGATORIA: Por favor, corrige este error en esta nueva generación. Asegúrate de devolver una estructura válida, con el array 'days' y al menos 2 comidas por día.\n"
            "----------------------------------------------------------------------\n"
        )

    user_id = form_data.get("user_id") or form_data.get("session_id")
    if user_id == "guest":
        user_id = None
        
    # Nuevo motor anti-repetición robusto: Query directo a la base de datos
    if user_id:
        try:
            recent_meals = get_recent_meals_from_plans(user_id, days=5)
            if recent_meals:
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
            from datetime import datetime, timezone, timedelta
            
            since_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            consumed = get_consumed_meals_since(user_id, since_date)
            
            if consumed:
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
            if check_meal_plan_generated_today(user_id):
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
    user_facts_text = ""
    visual_facts_text = ""
    facts_data_sorted = []
    visual_list = []
    if user_id:
        try:
            
            # 1. Recuperación estricta (Metadata JSONB) - ALERGIAS, CONDICIONES, RECHAZOS
            strict_facts_text = ""
            alergias = get_user_facts_by_metadata(user_id, 'category', 'alergia')
            if alergias:
                strict_facts_text += "🔴 ALERGIAS ESTRICTAS (PROHIBIDO USAR):\n" + "\n".join([f"  - {a['fact']}" for a in alergias]) + "\n"
                
            rechazos = get_user_facts_by_metadata(user_id, 'category', 'rechazo')
            if rechazos:
                strict_facts_text += "🔴 RECHAZOS (NO USAR):\n" + "\n".join([f"  - {r['fact']}" for r in rechazos]) + "\n"
                
            condiciones = get_user_facts_by_metadata(user_id, 'category', 'condicion_medica')
            if condiciones:
                strict_facts_text += "⚠️ CONDICIONES MÉDICAS (ADAPTAR PLAN):\n" + "\n".join([f"  - {c['fact']}" for c in condiciones]) + "\n"

            if strict_facts_text:
                user_facts_text += "=== REGLAS MÉDICAS Y DE GUSTO ABSOLUTAS (Extraídas de Base de Datos Estructurada) ===\n"
                user_facts_text += strict_facts_text + "=================================================================================\n\n"

            # 2. Buscar hechos textuales — QUERY DINÁMICA (Vectorial) para contexto general
            dynamic_parts = []
            if form_data.get("mainGoal"):
                dynamic_parts.append(f"Objetivo: {form_data['mainGoal']}")
            if form_data.get("allergies"):
                allergies = form_data["allergies"] if isinstance(form_data["allergies"], list) else [form_data["allergies"]]
                dynamic_parts.append(f"Alergias: {', '.join(allergies)}")
            if form_data.get("medicalConditions"):
                conditions = form_data["medicalConditions"] if isinstance(form_data["medicalConditions"], list) else [form_data["medicalConditions"]]
                dynamic_parts.append(f"Condiciones: {', '.join(conditions)}")
            if form_data.get("dietType"):
                dynamic_parts.append(f"Dieta: {form_data['dietType']}")
            if form_data.get("dislikes"):
                dislikes = form_data["dislikes"] if isinstance(form_data["dislikes"], list) else [form_data["dislikes"]]
                dynamic_parts.append(f"No le gusta: {', '.join(dislikes)}")
            if form_data.get("struggles"):
                struggles = form_data["struggles"] if isinstance(form_data["struggles"], list) else [form_data["struggles"]]
                dynamic_parts.append(f"Obstáculos: {', '.join(struggles)}")
            
            dynamic_query = ". ".join(dynamic_parts) if dynamic_parts else "Preferencias de comida, restricciones médicas, gustos y síntomas digestivos del usuario"
            print(f"🔍 [RAG] Query dinámica: {dynamic_query}")
            
            # Brecha 5: Validación de Caché RAG
            rag_cache_key = f"rag_{user_id}_{dynamic_query}"
            cached_rag = _LLM_CACHE.get(rag_cache_key)
            query_emb = None
            if cached_rag is not None:
                print(f"⚡ [CACHE HIT] Reutilizando contexto RAG para la misma query dinámica.")
                if len(cached_rag) == 3:
                    facts_data_sorted, visual_list, query_emb = cached_rag
                else:
                    facts_data_sorted, visual_list = cached_rag
                    query_emb = get_embedding(dynamic_query)
            else:
                query_emb = get_embedding(dynamic_query)
                if query_emb:
                    # Usar búsqueda híbrida pasando el texto
                    facts_data = search_user_facts(user_id, query_emb, query_text=dynamic_query, threshold=0.5, limit=10)
                    if facts_data:
                        # === PRIORIZACIÓN POR CATEGORÍA (Anti-Poda Bruta) ===
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
                        
                        facts_data_sorted = sorted(facts_data, key=get_fact_weight)
                        print(f"🧠 [RAG] Hechos textuales recuperados: {len(facts_data)} (ordenados por prioridad de categoría)")
                        
                # Buscar memoria visual 
                visual_query_emb = get_multimodal_embedding(dynamic_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        visual_list = [f"• {item['description']}" for item in visual_data]
                        print(f"📸 [VISUAL RAG] Entradas visuales recuperadas: {len(visual_data)}")
                        
                # Guardar en caché síncrono
                _LLM_CACHE[rag_cache_key] = (facts_data_sorted, visual_list, query_emb)
                    
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
    
    # 3. Estado inicial del grafo
    req_id = str(uuid.uuid4())[:8]
    request_id_var.set(req_id)
    
    initial_state: PlanState = {
        "request_id": req_id,
        "form_data": actual_form_data,
        "taste_profile": taste_profile,
        "nutrition": nutrition,
        "history_context": history_context,
        "user_facts": full_rag_context,
        "semantic_cache_hit": False,
        "cached_plan_data": None,
        "profile_embedding": query_emb if 'query_emb' in locals() else None,
        "plan_result": None,
        "plan_skeleton": None,
        "review_passed": False,
        "review_feedback": "",
        "attempt": 1,
        "rejection_reasons": [],
        "progress_callback": progress_callback,
        "background_tasks": background_tasks,
        "pipeline_start": pipeline_start,
    }
    
    # 4. Ejecutar el grafo con Timeout Global y Graceful Degradation (Mejora 5)
    import asyncio
    latest_state = [initial_state]
    
    async def run_graph():
        # Usamos astream con stream_mode="values"
        async for event in _PLAN_GRAPH.astream(initial_state, stream_mode="values"):
            latest_state[0] = event
        return latest_state[0]

    def _get_extreme_fallback_plan(nutr: dict, goal: str) -> dict:
        """Fallback matemático determinista para evitar caídas del sistema."""
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
        
        day_plan = {
            "daily_summary": "Plan de Contingencia de Emergencia (Generado Matemáticamente)",
            "total_calories": target_cal,
            "total_protein": target_pro,
            "total_carbs": target_car,
            "total_fats": target_fat,
            "meals": [meal1, meal2, meal3]
        }
        
        return {
            "main_goal": goal,
            "insights": ["Este es un plan de contingencia generado matemáticamente debido a indisponibilidad temporal de la IA."],
            "calories": target_cal,
            "macros": {
                "protein": f"{target_pro}g",
                "carbs": f"{target_car}g",
                "fats": f"{target_fat}g"
            },
            "days": [
                {"day": 1, **day_plan},
                {"day": 2, **day_plan},
                {"day": 3, **day_plan}
            ]
        }

    try:
        # Ejecutar asíncronamente con un timeout global (sin saltos de hilo)
        final_state = await asyncio.wait_for(run_graph(), timeout=600)
    except Exception as e:
        print(f"🚨 [EXTREME GRACEFUL DEGRADATION] Error crítico en pipeline ({type(e).__name__}): {e}")
        final_state = latest_state[0] if latest_state else {}
        if not final_state.get("plan_result"):
            print("🛡️ [FALLBACK] Generando plan de emergencia matemático (By-pass de LLM)...")
            final_state["plan_result"] = _get_extreme_fallback_plan(nutrition, actual_form_data.get("mainGoal", "Salud General"))
            final_state["attempt"] = 1
            final_state["review_passed"] = True
            final_state["_is_fallback"] = True
        else:
            # GAP 3: Asegurar que los macros estén presentes incluso si falló el ensamblador
            if "calories" not in final_state["plan_result"] or not final_state["plan_result"].get("calories"):
                target_cal = nutrition.get('target_calories', 2000)
                macros_dict = nutrition.get('macros', {})
                target_pro = macros_dict.get('protein_g', 150)
                target_car = macros_dict.get('carbs_g', 200)
                target_fat = macros_dict.get('fats_g', 60)
                
                final_state["plan_result"]["calories"] = target_cal
                final_state["plan_result"]["macros"] = {
                    "protein": f"{target_pro}g",
                    "carbs": f"{target_car}g",
                    "fats": f"{target_fat}g"
                }
                print("🛡️ [FALLBACK] Inyectados calorías y macros faltantes al plan_result parcial.")
    
    pipeline_duration = round(time.time() - pipeline_start, 2)
    
    # GAP 4: Score holístico End-to-End para auto-mejora continua
    try:
        plan = final_state.get("plan_result")
        if plan and isinstance(plan, dict):
            # 1. Penalizar retries (cada retry = -25% en este componente)
            retry_penalty = max(0.0, 1.0 - (final_state.get("attempt", 1) - 1) * 0.25)
            # 2. Bonus por aprobación médica
            review_bonus = 1.0 if final_state.get("review_passed") else 0.5
            # 3. Desviación calórica promedio real vs target
            cal_deviations = []
            for day in plan.get("days", []):
                day_cals = sum(m.get("cals", 0) for m in day.get("meals", []))
                if day_cals > 0:
                    cal_deviations.append(abs(day_cals - nutrition["target_calories"]) / nutrition["target_calories"])
            avg_deviation = sum(cal_deviations) / len(cal_deviations) if cal_deviations else 0
            cal_score = max(0.0, 1.0 - avg_deviation * 5)  # 20% desviación = score 0
            
            holistic_score = round(retry_penalty * 0.3 + review_bonus * 0.3 + cal_score * 0.4, 3)
            
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
                    "pipeline_duration_s": pipeline_duration
                }
            })
            print(f"📊 [HOLISTIC SCORE] Pipeline Quality: {holistic_score:.3f} (retry={retry_penalty:.2f}, review={review_bonus:.2f}, cal={cal_score:.2f})")
            
            # GAP 3: Persistir el holistic score en la DB
            user_id = actual_form_data.get("user_id") or actual_form_data.get("session_id")
            if user_id and user_id != "guest":
                try:
                    from db_profiles import get_user_profile, update_user_health_profile
                    profile = get_user_profile(user_id)
                    if profile:
                        hp = profile.get("health_profile") or {}
                        hp["last_pipeline_score"] = holistic_score
                        hp["last_pipeline_attempts"] = final_state.get("attempt", 1)
                        update_user_health_profile(user_id, hp)
                        print("💾 [HOLISTIC SCORE] Guardado en health_profile exitosamente.")
                except Exception as db_err:
                    print(f"⚠️ [HOLISTIC SCORE] Error guardando en DB: {db_err}")
                    
    except Exception as e:
        print(f"⚠️ [HOLISTIC SCORE] Error calculando score: {e}")
    
    print(f"\n{'🔗' * 30}")
    print(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
    print(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
    print("🔗" * 30 + "\n")
    
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
        
    return final_state["plan_result"]

def run_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None, previous_ai_error: str = None, background_tasks=None) -> dict:
    """Wrapper síncrono para mantener compatibilidad con cron/callers no-async."""
    import asyncio
    import threading
    import contextvars
    
    coro = arun_plan_pipeline(form_data, history, taste_profile, memory_context, progress_callback, previous_ai_error, background_tasks)
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        ctx = contextvars.copy_context()
        res = [None]
        err = [None]
        
        def _runner():
            try:
                res[0] = asyncio.run(coro)
            except Exception as e:
                err[0] = e
                
        t = threading.Thread(target=ctx.run, args=(_runner,))
        t.start()
        t.join()
        if err[0] is not None:
            raise err[0]
        return res[0]
    else:
        return asyncio.run(coro)
