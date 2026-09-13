# backend/llm_circuit_breaker.py
"""[P1-PLAN-LOTE-32 · 2026-09-13] El circuit breaker distribuido del LLM (`LLMCircuitBreaker`: Redis `cb:llm:*` + fila
`app_kv_store.llm_circuit_breaker[:<modelo>]` con fallback atómico), el breaker LOCAL de las escrituras best-effort
(`_BestEffortDBCircuitBreaker`, `_get_be_db_cb`, `_is_pool_timeout_error`) y `LLMCircuitOpenError`.

Vivían en `graph_orchestrator.py` (E7 / ARQ30-P2-01) y se movieron TAL CUAL; el grafo los re-exporta. Se quedan allí
los knobs `MEALFIT_CB_*`, la instancia global `_circuit_breaker`, el registro per-modelo `_get_circuit_breaker` y
`_record_cb_failure_unless_transient` (la política: qué fallo cuenta y con qué umbral); aquí vive el mecanismo.

El ciclo de vida del KV (tres vías de reset y el sweep de filas stale) está en CLAUDE.md → «Ciclo de vida del KV
`llm_circuit_breaker:*`» y en el runbook que enlaza.

El logger conserva el nombre `graph_orchestrator` (ver `llm_concurrency.py`). Para cambiar lo que este código ve en un
test (`redis_client`, `get_redis_async`, `execute_sql_write`…), parchea `llm_circuit_breaker.<nombre>`.
"""
import asyncio
import json
import logging
import os
import threading
import time

from cache_manager import get_redis_async, redis_client
from contextlib import asynccontextmanager
from db_core import aexecute_sql_query, aexecute_sql_write, execute_sql_query, execute_sql_write

# Mismo nombre que el logger del grafo: ver el docstring.
logger = logging.getLogger("graph_orchestrator")


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


class LLMCircuitOpenError(Exception):
    """[P2-REVIEWER-CB-TRANSIENT · 2026-07-02] (test clínico gemini, batch P2-ENGINE-CLINICAL-SAVERS)
    Excepción DEDICADA para "circuit breaker abierto" en los guards de invocación LLM (antes
    `Exception` genérica con mensaje). Razón: `_is_reviewer_transient_error` clasifica por TIPO
    (nunca por substring, para no enmascarar rechazos clínicos) → un breaker abierto en el REVISOR
    caía a la rama fail-closed "Error en la estructura del revisor médico" con severity=critical y
    abort inmediato del plan entero, cuando un breaker abierto es transitorio POR DEFINICIÓN
    (existe para shed load). Subclase de Exception → los `except Exception` existentes la siguen
    atrapando sin cambios. tooltip-anchor: P2-REVIEWER-CB-TRANSIENT"""


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
        # [P1-DREAMING-CB-KWARG · 2026-07-24] Auto-corrección de un error de llamada que
        # rompía el breaker EN SILENCIO. `LLMCircuitBreaker("glm-5.3-flash")` (posicional)
        # dejaba `threshold` con un str: `failures >= self.threshold` lanzaba TypeError, que
        # los `except Exception: pass` de los callers se tragaban → el breaker no abría NUNCA,
        # y además las keys quedaban sin sufijo de modelo (pisando el breaker global legacy).
        # En prod eso se veía como 17 errores de escritura del CB sin un solo breaker abierto.
        # Se corrige la intención evidente y se grita: un breaker roto no protege de nada, así
        # que degradar a "no protege" en silencio es el peor resultado posible.
        if isinstance(failure_threshold, str):
            logger.error(
                f"🚨 [P1-DREAMING-CB-KWARG] LLMCircuitBreaker recibió el modelo "
                f"{failure_threshold!r} como PRIMER posicional (= failure_threshold). "
                f"Interpretado como model_name; usa `LLMCircuitBreaker(model_name=...)`."
            )
            if model_name is None:
                model_name = failure_threshold
            failure_threshold = 3
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
        # [P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] Cliente Redis PER-LOOP (get_redis_async),
        # NO el module-global. La generación corre en un loop fresco (asyncio.run); reusar
        # el cliente global atado al loop de import lanzaba "Future attached to a different
        # loop" en cada CB write (~25/30min en prod). Completa la migración P2-REDIS-ASYNC-PERLOOP
        # que solo cubrió los semáforos. Fail-soft: get_redis_async()→None cae al path DB.
        _rc = get_redis_async()
        if _rc:
            try:
                failures = await _rc.incr(self._failures_key)
                await _rc.expire(self._failures_key, self.reset_timeout)
                if failures >= self.threshold:
                    await _rc.set(self._open_key, "1", ex=self.reset_timeout)
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

        # [P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] cliente per-loop (ver arecord_failure).
        _rc = get_redis_async()
        if _rc:
            try:
                await _rc.delete(self._failures_key, self._open_key)
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

        # [P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] cliente per-loop (ver arecord_failure).
        _rc = get_redis_async()
        if _rc:
            try:
                is_open = await _rc.get(self._open_key)
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
