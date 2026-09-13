# backend/llm_concurrency.py
"""[P1-PLAN-LOTE-32 · 2026-09-13] Semáforos distribuidos del LLM — el global (`DistributedLLMSemaphore`) y el per-user
(`DistributedPerUserSemaphore`) —, el contador de presupuesto que ambos alimentan (`_inc_budget_stat`,
`get_llm_budget_stats_snapshot`) y los dos knobs que sólo ellos leen (`MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX`,
`MEALFIT_LLM_LOCAL_MAX_WAIT_S`).

Vivían en `graph_orchestrator.py` (E7 / ARQ30-P2-01: el god-file llegó a 53.099 líneas con el tope en 53.100). Se movieron
TAL CUAL — mismo texto, mismos comentarios; sólo cambian los imports — y el grafo los re-exporta, así que
`graph_orchestrator.DistributedLLMSemaphore`, `graph_orchestrator._inc_budget_stat`, etc. siguen valiendo. Las INSTANCIAS
(`LLM_SEMAPHORE`, `PER_USER_LLM_SEMAPHORE`) y la composición `acquire_user_and_global` se quedan en el grafo: leen los
knobs `MEALFIT_LLM_*` que él define, y los tests que parchean esas instancias siguen apuntando donde deben.

El logger conserva el nombre `graph_orchestrator` a propósito: el formato de producción imprime `%(name)s`, y los greps del
operador y los `caplog` de la suite filtran por ese nombre. Mover el código no mueve sus logs.

Para cambiar lo que ESTE código ve en un test (p. ej. `redis_client`), parchea `llm_concurrency.<nombre>`: parchear
`graph_orchestrator.<nombre>` ya no lo alcanza.
"""
import asyncio
import logging
import threading
import time
import uuid

from contextlib import asynccontextmanager, contextmanager
from knobs import _env_int

# Mismo nombre que el logger del grafo: ver el docstring.
logger = logging.getLogger("graph_orchestrator")


# [P3-PROD-AUDIT-3 · 2026-05-30] Cota del dict in-process `_local_sync` de
# DistributedPerUserSemaphore. En modo Redis-DOWN (REDIS_URL ausente o Redis
# caído — degradación soportada por cache_manager) el dict acumula un
# threading.Semaphore por user_id DISTINTO sin eviction → fuga lenta pero
# ilimitada en un proceso long-lived (--workers 1). Al exceder el cap se purgan
# entradas IDLE (todos sus permits disponibles). Clamp generoso.
LLM_PER_USER_LOCAL_CACHE_MAX = _env_int(
    "MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX", 4096, validator=lambda v: 64 <= v <= 1_000_000
)


# [P2-ORCH-11 · 2026-05-28] Cota del busy-poll LOCAL (fallback cuando Redis está
# caído). Los paths Redis ya degradan vía LLM_MAX_WAIT_S/LLM_USER_MAX_WAIT_S, pero
# el busy-poll local giraba a 20Hz SIN cota → bajo Redis-down + saturación
# sostenida, N corrutinas quemaban CPU del event loop hasta el techo de 720s del
# pipeline. Al expirar: stat + warning + backpressure error (fail-fast → el handler
# global entrega fallback). Default 120s, clamp [1, 3600]. Tooltip-anchor: P2-ORCH-11.
LLM_LOCAL_MAX_WAIT_S        = _env_int  ("MEALFIT_LLM_LOCAL_MAX_WAIT_S",        120,
                                         validator=lambda v: 1 <= v <= 3600)


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
        # [P2-REDIS-ASYNC-PERLOOP · 2026-06-17] Cliente per-loop: el global
        # `redis_async_client` (creado al import) rompe con "Event loop is closed"
        # cuando esta corutina corre en el loop de generación. `get_redis_async()`
        # devuelve uno ligado al loop ACTUAL (o None → fallback local).
        from cache_manager import get_redis_async
        _rc = get_redis_async()
        if not _rc:
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
                    await _rc.zremrangebyscore(self.key, "-inf", now - self.timeout)

                    # 2. Intentar adquirir agregando a la cola
                    await _rc.zadd(self.key, {req_id: now})

                    # 3. Verificar posición en la cola
                    rank = await _rc.zrank(self.key, req_id)

                    if rank is not None and rank < self.max_concurrent:
                        acquired = True
                        break
                    else:
                        # Cola llena, retirarse y esperar
                        await _rc.zrem(self.key, req_id)
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
                try:
                    await _rc.zrem(self.key, req_id)
                except Exception:
                    pass


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

        # [P2-REDIS-ASYNC-PERLOOP · 2026-06-17] Cliente per-loop (ver get_redis_async).
        # Antes usaba el global `redis_async_client` → "Event loop is closed" en el
        # loop de generación → caía a fallback local en CADA plan. Ahora el path Redis
        # funciona; el fallback local queda solo para Redis-down real.
        from cache_manager import get_redis_async
        _rc = get_redis_async()
        if not _rc:
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
                    await _rc.zremrangebyscore(key, "-inf", now - self.timeout)
                    await _rc.zadd(key, {req_id: now})
                    rank = await _rc.zrank(key, req_id)
                    if rank is not None and rank < self.max_per_user:
                        acquired = True
                        break
                    await _rc.zrem(key, req_id)
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
                    await _rc.zrem(key, req_id)
                except Exception:
                    pass


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
