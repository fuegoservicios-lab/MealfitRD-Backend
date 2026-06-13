from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from datetime import datetime, timedelta, timezone
import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from supabase import create_client, Client
from dotenv import load_dotenv
# [P1-DB-STMT-TIMEOUT · 2026-05-27] Helper registrado en `_KNOBS_REGISTRY`
# (visible en /health/version y /api/system/knobs). `knobs` no tiene deps
# internas → import a top-level seguro, sin riesgo de ciclo.
from knobs import _env_int as _knob_env_int



logger = logging.getLogger(__name__)
# Suprimir warnings cosméticos de psycopg.pool sobre conexiones ACTIVE retornadas al pool (ej. por LangGraph PostgresSaver)
logging.getLogger("psycopg.pool").setLevel(logging.ERROR)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

# Knobs MEALFIT_DB_POOL_* para sintonizar el psycopg ConnectionPool sin redeploy.
# Existen porque bajo picos (RAG + cron chunk_queue + cache writes en paralelo)
# se observó saturación con `couldn't get a connection after 30.00 sec`. Permite:
#   - Subir MAX_SIZE para absorber concurrencia mayor.
#   - Bajar TIMEOUT_S para fallar rápido y diagnosticar antes de que se acumule cola.
#   - Ajustar MAX_IDLE_S por debajo del idle timeout de Supavisor si éste cambia.
def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, str(default)))))
    except (TypeError, ValueError):
        return default

def _float_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.environ.get(name, str(default)))))
    except (TypeError, ValueError):
        return default

# [P0-3] Defaults subidos tras incidente 2026-05-06 (`couldn't get a connection 30s` +
# APScheduler skips bajo 2 pipelines paralelos). Trade-off:
#   - MAX_SIZE 30→60: absorbe picos de RAG + cron chunk_queue + cache writes en paralelo.
#     Supabase Transaction Pooler (6543) tolera muchas conexiones (multiplexa via pgBouncer).
#   - TIMEOUT 30s→15s: fail-fast para diagnosticar saturación antes de que se acumule cola
#     de waiters. APScheduler ya skipea jobs cuando este timeout vence; bajarlo evita que
#     un solo waiter bloquee el slot del job 15s extra.
#   - MIN_SIZE sin cambio (2): no consumir conexiones idle en entornos low-traffic.
# Override cualquiera con `MEALFIT_DB_POOL_*` si el comportamiento previo era preferido
# o si el pooler reporta saturación al alza.
DB_POOL_MIN_SIZE = _int_env("MEALFIT_DB_POOL_MIN_SIZE", 2, 0, 50)
DB_POOL_MAX_SIZE = _int_env("MEALFIT_DB_POOL_MAX_SIZE", 60, 1, 200)
DB_POOL_TIMEOUT_S = _float_env("MEALFIT_DB_POOL_TIMEOUT_S", 15.0, 1.0, 120.0)
DB_POOL_MAX_IDLE_S = _float_env("MEALFIT_DB_POOL_MAX_IDLE_S", 300.0, 30.0, 1800.0)

# [P1-POOL-ASYNC-SPLIT · 2026-05-16] Pool ASYNC con knobs SEPARADOS del SYNC.
# Pre-fix: sync y async compartían `MEALFIT_DB_POOL_MAX_SIZE` → cada uno reservaba
# hasta `max_size` conexiones independientes → total real = 2× max contra el cap
# de pgBouncer Transaction Mode (~15-30 client conns free tier). Con max=25
# compartido eso eran 50 conexiones contra un cap de 15-30 → el async era
# stretched-thin permanentemente y emitía "couldn't get a connection after Xs"
# 5-15× por plan (cache async, CB resets, meta-learning preflight).
#
# Post-fix: SYNC sigue con `MEALFIT_DB_POOL_MAX_SIZE` (queries críticas: INSERTs
# de planes, transacciones de inventario, etc.); ASYNC tiene su propio knob con
# default más conservador (8). Total = 25 + 8 = 33, MUY cercano al cap pgBouncer
# pero dejando headroom: el sync sigue siendo el path principal y el async solo
# absorbe bursts de cache/CB-reset/observabilidad que toleran fail-best-effort.
#
# Si migras a Supabase Pro / dedicated, subir ambos: SYNC max=60, ASYNC max=20.
DB_ASYNC_POOL_MIN_SIZE = _int_env("MEALFIT_DB_ASYNC_POOL_MIN_SIZE", 2, 0, 20)
# [P1-POOL-ASYNC-SPLIT-TUNE · 2026-05-16] Subido 8→12 tras observar que post-split
# inicial los warnings "couldn't get a connection after 20.00 sec" seguían ~12
# por plan (espaciados, no en burst). Causa probable: con max=8, los crons
# concurrent (drain_pending_facts, worker_metrics, etc.) + CB resets de 3 day_gen
# paralelos no caben en 8 slots. Subir a 12 da margen sin acercar peligrosamente
# al cap pgBouncer (sync 25 + async 12 = 37, todavía manejable con keepalive
# recycling rápido). Si los warnings persisten >5/plan tras este tune, investigar
# connection leak o subir clamp del knob.
DB_ASYNC_POOL_MAX_SIZE = _int_env("MEALFIT_DB_ASYNC_POOL_MAX_SIZE", 12, 1, 100)
DB_ASYNC_POOL_TIMEOUT_S = _float_env("MEALFIT_DB_ASYNC_POOL_TIMEOUT_S", 20.0, 1.0, 120.0)

# [P1-DB-STMT-TIMEOUT · 2026-05-27] Timeout de sentencia + idle-in-transaction
# a NIVEL DE SESIÓN sobre cada conexión de los pools sync/async principales.
# Cierra el vector de agotamiento del pool donde una query atascada (scan JSONB
# lento sobre `meal_plans`, espera de `pg_advisory_lock`, stall de red) retiene
# su slot indefinidamente. Con handlers FastAPI + crons APScheduler +
# `_chunk_worker` compartiendo el pool sync (max 60), unas pocas queries
# colgadas agotan el pool → waiters caen en `DB_POOL_TIMEOUT_S` → cascada
# `couldn't get a connection`. Único net previo: el `statement_timeout`
# server-side de Supavisor (~60s), demasiado alto bajo contención.
#
# Convierte "query atascada retiene slot para siempre" en "query falla tras N
# ms, el slot vuelve al pool". El `idle_in_transaction_session_timeout` sólo
# mata conexiones idle DENTRO de una transacción abierta (siempre un
# bug/leak), nunca una query corriendo — por eso su default es más agresivo.
#
# Rollback sin redeploy: `MEALFIT_DB_STATEMENT_TIMEOUT_MS=0` desactiva el
# statement_timeout (vuelve a depender sólo del backstop de Supavisor);
# `MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS=0` desactiva el idle-in-txn.
# Los `SET LOCAL` per-transacción existentes (`set_meal_plan_for_update_timeouts`,
# `execute_sql_write(lock_timeout_ms=...)`) overridean estos defaults por-tx.
# Tooltip-anchor: P1-DB-STMT-TIMEOUT.
DB_STATEMENT_TIMEOUT_MS = _knob_env_int(
    "MEALFIT_DB_STATEMENT_TIMEOUT_MS", 30000, validator=lambda v: 0 <= v <= 600000
)
DB_IDLE_IN_TXN_TIMEOUT_MS = _knob_env_int(
    "MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS", 15000, validator=lambda v: 0 <= v <= 600000
)


def _session_timeout_statements() -> List[str]:
    """[P1-DB-STMT-TIMEOUT · 2026-05-27] Lista de sentencias `SET ...` a aplicar
    en el `configure` de cada conexión nueva de los pools sync/async. Valores
    son ints clampeados por el knob (0..600000) → seguro inlinearlos (no es
    user input; `SET` no acepta placeholders `%s` para estos parámetros).
    Helper aislado para testabilidad (los tests monkeypatchean los módulo-level
    `DB_*_TIMEOUT_MS` y verifican el SQL generado sin abrir conexión real)."""
    stmts: List[str] = []
    if DB_STATEMENT_TIMEOUT_MS > 0:
        stmts.append(f"SET statement_timeout = {int(DB_STATEMENT_TIMEOUT_MS)}")
    if DB_IDLE_IN_TXN_TIMEOUT_MS > 0:
        stmts.append(
            f"SET idle_in_transaction_session_timeout = {int(DB_IDLE_IN_TXN_TIMEOUT_MS)}"
        )
    return stmts

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# Configuración del ConnectionPool para psycopg
# [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Hay DOS pools:
#   - `connection_pool` (sync) + `async_connection_pool`: tráfico genérico de la
#     app (queries del frontend, crons, RAG, embeddings). Usa **Transaction
#     Pooler** (6543) para multiplexar conexiones via pgBouncer/Supavisor.
#   - `chat_checkpoint_pool`: SOLO para `langgraph.checkpoint.postgres.PostgresSaver`.
#     Usa el URL ORIGINAL (port 5432, session mode). Razón documentada en el
#     bloque donde se construye.
connection_pool = None
async_connection_pool = None  # Siempre declarado a nivel de módulo para garantizar importabilidad
chat_checkpoint_pool = None  # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20]
# [P1-NEON-DB-MIGRATION · 2026-06-12] URL session-mode resuelta según backend
# activo (Neon direct o Supabase :5432). Consumida por el leader lock del
# scheduler (app.py::_build_session_mode_db_url) — advisory locks de sesión
# requieren session mode; un transaction pooler los liberaría por sentencia.
DB_SESSION_MODE_URL = None

# [P1-NEON-DB-MIGRATION · 2026-06-12] Selección del backend de DATOS Postgres.
# Knob `MEALFIT_DB_BACKEND`: "supabase" (default histórico) | "neon".
#   - neon: pools principales ← NEON_DATABASE_URL_POOLED (PgBouncer transaction
#     mode de Neon, análogo de Supavisor :6543); chat_checkpoint_pool ←
#     NEON_DATABASE_URL (endpoint directo, session mode — mismo contrato que
#     P1-CHECKPOINT-POOL-SPLIT). Supabase queda SOLO para Auth+Storage (el
#     cliente `supabase` de arriba NO se toca).
#   - supabase: comportamiento histórico con rewrites :5432↔:6543.
# Cutover/rollback sin redeploy: flip del env var + restart. Si backend=neon
# pero faltan los URLs, se aborta la config de pools (fail-loud → /ready 503)
# en vez de degradar silenciosamente a Supabase: un fallback silencioso
# escribiría en la DB equivocada (split-brain).
MEALFIT_DB_BACKEND = (os.environ.get("MEALFIT_DB_BACKEND") or "supabase").strip().lower()
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL")
NEON_DATABASE_URL_POOLED = os.environ.get("NEON_DATABASE_URL_POOLED")

if SUPABASE_DB_URL or MEALFIT_DB_BACKEND == "neon":
    try:
        from psycopg_pool import ConnectionPool, AsyncConnectionPool

        if MEALFIT_DB_BACKEND == "neon":
            if not (NEON_DATABASE_URL and NEON_DATABASE_URL_POOLED):
                raise RuntimeError(
                    "[P1-NEON-DB-MIGRATION] MEALFIT_DB_BACKEND=neon requiere "
                    "NEON_DATABASE_URL (direct) y NEON_DATABASE_URL_POOLED "
                    "(pooler). Sin fallback silencioso a Supabase: split-brain."
                )
            clean_url = NEON_DATABASE_URL_POOLED.strip().strip("'").strip('"')
            original_session_url = NEON_DATABASE_URL.strip().strip("'").strip('"')
            logger.info(
                "🔌 [P1-NEON-DB-MIGRATION] Backend de datos: NEON "
                "(pooled→pools principales, direct→chat_checkpoint_pool). "
                "Supabase queda solo para Auth+Storage."
            )
        else:
            # Limpiar comillas basura por si acaso
            clean_url = SUPABASE_DB_URL.strip().strip("'").strip('"')

            # [P1-CHAT-CHECKPOINT-FIX · 2026-05-20] `original_session_url` debe
            # SIEMPRE apuntar al puerto session-mode (5432), independientemente
            # de cómo viene el env var. Pre-fix: `original_session_url = clean_url`
            # capturaba el URL crudo antes del rewrite — pero si el operator
            # copió "Transaction pooler" del dashboard de Supabase (que viene
            # con :6543 hardcoded), el "rescate" no aplicaba y el
            # `chat_checkpoint_pool` terminaba apuntando también a Supavisor
            # transaction mode. Resultado: el bug SSL bad length / EOF que
            # P1-CHECKPOINT-POOL-SPLIT pretendía cerrar se perpetuaba, y SOLO
            # se duplicaban los pools sin beneficio.
            #
            # Fix: forzar el rewrite a :5432 SIEMPRE para `original_session_url`,
            # tanto si el env var viene con :5432 (no-op) como con :6543 (revert
            # explícito). El pool principal `connection_pool` sigue rewriteando
            # a :6543 (lógica existente, transaction mode para concurrencia).
            # Tooltip-anchor: P1-CHAT-CHECKPOINT-FIX.
            if ".supabase." in clean_url and ":6543" in clean_url:
                original_session_url = clean_url.replace(":6543", ":5432")
                logger.info(
                    "🔧 [P1-CHAT-CHECKPOINT-FIX] chat_checkpoint_pool URL forzada "
                    "a :5432 (session mode) — env var SUPABASE_DB_URL venía con "
                    ":6543. Sin este rewrite, chat_checkpoint_pool reusaría "
                    "Supavisor transaction mode → SSL bad length / EOF al "
                    "`put_writes` del checkpointer al final de cada chat stream."
                )
            else:
                original_session_url = clean_url

            # MEJORA 3: Connection Pooling - Forzar el uso del Transaction Pooler de Supabase (6543)
            # Esto previene el agotamiento de conexiones directas (Connection Exhaustion) si el volumen crece.
            if ".supabase." in clean_url and ":5432" in clean_url:
                clean_url = clean_url.replace(":5432", ":6543")
                logger.info("🔧 [psycopg] DB URL automatically rewritten to use Supabase Transaction Pooler (port 6543).")

        # [P1-NEON-DB-MIGRATION] Export para consumidores que necesitan
        # session mode (leader lock del scheduler). Válido en ambos backends.
        DB_SESSION_MODE_URL = original_session_url

        def get_client_kwargs():
            return {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "autocommit": True,
            }

        def configure_sync_conn(conn):
            conn.autocommit = True
            # pgBouncer Transaction Mode no soporta server-side prepared statements.
            # Deshabilitar auto-prepare para evitar errores "_pg3_N requires N params".
            conn.prepare_threshold = None
            # [P1-DB-STMT-TIMEOUT · 2026-05-27] Session-level statement_timeout +
            # idle_in_transaction_session_timeout. Best-effort: un fallo del SET
            # NO debe impedir checkout de la conexión.
            _stmts = _session_timeout_statements()
            if _stmts:
                try:
                    with conn.cursor() as _c:
                        for _stmt in _stmts:
                            _c.execute(_stmt)
                except Exception as _to_err:
                    logger.debug(f"[P1-DB-STMT-TIMEOUT] SET sync falló (best-effort): {_to_err}")

        async def configure_async_conn(conn):
            await conn.set_autocommit(True)
            conn.prepare_threshold = None
            # [P1-DB-STMT-TIMEOUT · 2026-05-27] idem async.
            _stmts = _session_timeout_statements()
            if _stmts:
                try:
                    async with conn.cursor() as _c:
                        for _stmt in _stmts:
                            await _c.execute(_stmt)
                except Exception as _to_err:
                    logger.debug(f"[P1-DB-STMT-TIMEOUT] SET async falló (best-effort): {_to_err}")

        def configure_checkpoint_conn(conn):
            # [P1-DB-STMT-TIMEOUT · 2026-05-27] Pool del LangGraph checkpointer
            # SIN session timeouts — preserva el comportamiento EXACTO del pool
            # sensible (P1-CHAT-CHECKPOINT-*, bug SSL bad length / EOF detected).
            # El PostgresSaver mantiene conexiones checked-out ~5-30s durante LLM
            # calls; en autocommit NO están idle-in-transaction (el
            # idle_in_transaction_session_timeout NO las mataría), pero por la
            # fragilidad documentada de este pool (max=4, low-concurrency) no le
            # aplicamos timeouts — el riesgo de exhaustion es despreciable.
            conn.autocommit = True
            conn.prepare_threshold = None

        connection_pool = ConnectionPool(
            conninfo=clean_url,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            timeout=DB_POOL_TIMEOUT_S,
            max_idle=DB_POOL_MAX_IDLE_S,
            reconnect_timeout=5,  # Wait up to 5s for reconnection
            kwargs=get_client_kwargs(),
            configure=configure_sync_conn,
            check=ConnectionPool.check_connection,  # Health check on each checkout
            open=False
        )

        # [P1-POOL-ASYNC-SPLIT · 2026-05-16] Pool ASYNC con knobs propios. Ver
        # bloque de docstring arriba donde se definen los knobs DB_ASYNC_POOL_*.
        async_connection_pool = AsyncConnectionPool(
            conninfo=clean_url,
            min_size=DB_ASYNC_POOL_MIN_SIZE,
            max_size=DB_ASYNC_POOL_MAX_SIZE,
            timeout=DB_ASYNC_POOL_TIMEOUT_S,
            max_idle=DB_POOL_MAX_IDLE_S,
            reconnect_timeout=5,
            kwargs=get_client_kwargs(),
            configure=configure_async_conn,
            check=AsyncConnectionPool.check_connection,
            open=False
        )
        logger.info(
            "🔌 [psycopg] ConnectionPool SYNC configurado: "
            f"min={DB_POOL_MIN_SIZE}, max={DB_POOL_MAX_SIZE}, "
            f"timeout={DB_POOL_TIMEOUT_S}s, max_idle={DB_POOL_MAX_IDLE_S}s "
            f"[P1-DB-STMT-TIMEOUT: statement_timeout={DB_STATEMENT_TIMEOUT_MS}ms, "
            f"idle_in_txn={DB_IDLE_IN_TXN_TIMEOUT_MS}ms (0=off)]."
        )
        logger.info(
            "🔌 [psycopg] ConnectionPool ASYNC configurado: "
            f"min={DB_ASYNC_POOL_MIN_SIZE}, max={DB_ASYNC_POOL_MAX_SIZE}, "
            f"timeout={DB_ASYNC_POOL_TIMEOUT_S}s "
            "[P1-POOL-ASYNC-SPLIT: knobs separados del sync para no saturar pgBouncer]."
        )

        # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Pool separado para el
        # `langgraph.checkpoint.postgres.PostgresSaver`. Bug observado el
        # 2026-05-20: SSE chat completa response al user, pero al hacer
        # `put_writes` final del checkpoint → SSL error "bad length"
        # / "SSL SYSCALL error: EOF detected" → `outcome=error` + banner
        # frontend "El asistente tuvo un problema" + pérdida del state del
        # LangGraph para el siguiente turn.
        #
        # Root cause: el pool principal (`connection_pool`) usa Transaction
        # Pooler (Supavisor port 6543), que mata conexiones idle agresivamente
        # (~10-30s). Durante el chat flow, el PostgresSaver mantiene una
        # connection abierta del pool ~5-15s mientras espera la LLM call;
        # Supavisor la cierra mid-stream → al `put_writes` final, la conexión
        # ya está muerta. Defensivamente psycopg keepalives no ayudan porque
        # Supavisor corta a nivel aplicación, no TCP.
        #
        # Fix: el checkpointer usa `original_session_url` (sin rewrite a
        # 6543) → conexión directa session-mode (5432). Session mode no
        # mata conexiones idle de forma agresiva. Pool pequeño (1-4) porque
        # el chat-flow es low-concurrency (1 stream por user activo).
        #
        # [P1-CHAT-CHECKPOINT-FIX · 2026-05-20] El WARN legacy
        # "SUPABASE_DB_URL ya contiene :6543" fue removido — la nueva
        # lógica de force-rewrite arriba garantiza que `original_session_url`
        # SIEMPRE apunte a :5432 (session mode) cuando el host es Supabase,
        # haciendo el WARN inalcanzable. Mantenemos el try/except por si
        # `ConnectionPool(...)` falla por otras razones (DNS, auth, etc).
        # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20] `min_size=0` + `max_idle=30`
        # son recycling agresivo contra Supavisor session-pooler idle-kill.
        # Pre-fix (`min_size=1, max_idle=300`): el pool pre-warmaba 1 conn al
        # startup que envejecía ~44s antes del primer chat, y LangGraph la
        # mantenía checkout durante el LLM call (10-30s). Al `put_writes`
        # final, Supavisor ya la había matado → SSL bad length / EOF detected.
        # `min_size=0` evita warming de conns viejas. `max_idle=30` recicla
        # conns idle antes del kill threshold ~60-70s. Combinación: la conn
        # siempre nace fresh y vive corto. Tooltip-anchor: P1-CHAT-CHECKPOINT-DEGRADE.
        try:
            chat_checkpoint_pool = ConnectionPool(
                conninfo=original_session_url,
                min_size=0,
                max_size=4,
                timeout=15.0,
                max_idle=30.0,
                reconnect_timeout=5,
                kwargs=get_client_kwargs(),
                configure=configure_checkpoint_conn,  # [P1-DB-STMT-TIMEOUT] sin session timeouts
                check=ConnectionPool.check_connection,
                open=False,
            )
            logger.info(
                "🔌 [psycopg] chat_checkpoint_pool configurado (session "
                "mode, port 5432): min=0, max=4, max_idle=30s "
                "[P1-CHAT-CHECKPOINT-DEGRADE: recycle agresivo evita conns "
                "rancias que Supavisor mata mid-pipeline → SSL bad length]."
            )
        except Exception as checkpoint_pool_err:
            logger.error(
                f"⚠️ [P1-CHECKPOINT-POOL-SPLIT] Error configurando "
                f"chat_checkpoint_pool: {checkpoint_pool_err}. Caller usará "
                f"connection_pool fallback (puede reaparecer SSL bad length)."
            )
            chat_checkpoint_pool = None
    except Exception as pool_err:
        logger.error(f"⚠️ [psycopg] Error configurando ConnectionPool: {pool_err}")

def close_connection_pool():
    if connection_pool:
        connection_pool.close()
        logger.info("Connection pool cerrado.")
    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Cerrar también el pool del
    # LangGraph checkpointer en shutdown — same convention que connection_pool.
    if chat_checkpoint_pool:
        chat_checkpoint_pool.close()
        logger.info("chat_checkpoint_pool cerrado.")

async def aclose_connection_pool():
    if 'async_connection_pool' in globals() and async_connection_pool:
        await async_connection_pool.close()
        logger.info("Async Connection pool cerrado.")

def execute_sql_query(query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
    """Ejecuta una consulta SQL directa usando el pool de psycopg.

    [P0-bonus] Antes, si el caller olvidaba pasar `fetch_one=True` o `fetch_all=True`,
    este helper devolvía silenciosamente `[]` aunque la query produjera filas. Esto
    enmascaró bugs serios — `_recover_pantry_paused_chunks` y otros 8 callsites en
    `cron_tasks.py` y `graph_orchestrator.py` venían recibiendo `[]` en producción
    sin que ningún test lo detectara (los tests usan mocks que esquivan el helper
    real). El comportamiento ahora:

      - Si `fetch_one`: devuelve `cursor.fetchone()` (un dict o None).
      - Si `fetch_all`: devuelve `cursor.fetchall()` (lista de dicts).
      - Si ninguno y la query produjo filas: emite un WARNING con el query truncado
        y devuelve `cursor.fetchall()` (default seguro). El caller debería pasar el
        flag explícito; el WARNING permite detectar callsites pendientes.
      - Si ninguno y la query NO produjo filas (UPDATE/DELETE sin RETURNING, etc.):
        devuelve `[]` silenciosamente — comportamiento histórico preservado para no
        romper callers que usan este helper como variante de execute_sql_write.
    """
    if not connection_pool:
        # Fallback de seguridad si el pool no estuviera disponible (aunque no debería)
        raise RuntimeError("db connection_pool is not available.")

    import psycopg
    from psycopg.rows import dict_row

    with connection_pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            # [P0-bonus] Default seguro: si la query devolvió filas pero el caller
            # no pidió ninguna, devolvemos las filas + WARNING en lugar de tirarlas
            # silenciosamente.
            try:
                if cursor.description is not None:
                    rows = cursor.fetchall()
                    if rows:
                        _query_preview = " ".join((query or "").split())[:120]
                        logger.warning(
                            f"[db_core/execute_sql_query] Caller no pasó "
                            f"fetch_one/fetch_all pero la query devolvió "
                            f"{len(rows)} filas. Retornando filas como default seguro; "
                            f"el caller debería marcar fetch_all=True explícito. "
                            f"Query: {_query_preview!r}"
                        )
                        return rows
            except Exception:
                pass
            return []

def execute_sql_write(query: str, params: tuple = None, returning: bool = False, lock_timeout_ms: int = None):
    """Ejecuta una transacción INSERT/UPDATE/DELETE.

    [P1-3 · 2026-05-10] `lock_timeout_ms` opcional: cuando se provee, la
    sentencia se ejecuta dentro de una transacción explícita con
    `SET LOCAL lock_timeout = '<N>ms'` aplicado ANTES del cursor.execute
    del query principal. Si el lock no se adquiere en ese tiempo,
    psycopg propaga `psycopg.errors.LockNotAvailable` (SQLSTATE 55P03) y
    el caller decide retry/skip — vs. el comportamiento default (sin
    `lock_timeout` local) que espera indefinidamente hasta que el
    `statement_timeout` de la sesión actúe.

    Mismo helper SSOT que el heartbeat de chunks usa para no colgarse
    si otro worker tiene la fila bloqueada. Sin esto, el thread daemon
    del heartbeat podía esperar minutos por un row lock y morirse,
    dejando el chunk como zombie hasta que el rescue lo matase tras
    CHUNK_LOCK_STALE_MINUTES.

    Backward-compatible: callers existentes que pasen `lock_timeout_ms=None`
    obtienen exactamente el mismo comportamiento de antes (sin transacción
    explícita, sin SET LOCAL). El nuevo path solo se activa cuando el
    caller opta in.
    """
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available.")

    import psycopg
    from psycopg.rows import dict_row

    with connection_pool.connection() as conn:
        if lock_timeout_ms is not None and int(lock_timeout_ms) > 0:
            # [P1-3] Wrap explícito en transacción para que `SET LOCAL`
            # tenga scope. Sin `conn.transaction()`, autocommit aplicaría
            # el SET LOCAL inmediatamente y luego lo "perdería" en el
            # siguiente comando.
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    try:
                        cursor.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'")
                    except Exception as set_err:
                        # Best-effort: si el SET LOCAL falla, seguimos sin él
                        # (comportamiento de antes — mejor que abortar el write).
                        logger.debug(f"[P1-3] SET LOCAL lock_timeout falló: {set_err}")
                    cursor.execute(query, params)
                    if returning:
                        return cursor.fetchall()
                    try:
                        cursor.fetchall()
                    except psycopg.ProgrammingError:
                        pass
                    return True
        else:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, params)
                if returning:
                    return cursor.fetchall()

                try:
                    cursor.fetchall()
                except psycopg.ProgrammingError:
                    pass
                return True

def execute_sql_transaction(queries_with_params: List[Tuple[str, tuple]]):
    """Ejecuta múltiples consultas SQL dentro de una única transacción (BEGIN ... COMMIT)."""
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available.")
    
    import psycopg
    from psycopg.rows import dict_row
    
    with connection_pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cursor:
                for query, params in queries_with_params:
                    cursor.execute(query, params)
                    try:
                        cursor.fetchall()
                    except psycopg.ProgrammingError:
                        pass
        return True

async def aexecute_sql_query(query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
    """Ejecuta una consulta SQL asíncrona usando el pool de psycopg async."""
    if 'async_connection_pool' not in globals() or not async_connection_pool:
        raise RuntimeError("db async_connection_pool is not available.")
    
    from psycopg.rows import dict_row
    import psycopg
    
    async with async_connection_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            if fetch_one:
                res = await cursor.fetchone()
                try:
                    await cursor.fetchall()  # Drenar resultados restantes para liberar la conexión
                except psycopg.ProgrammingError:
                    pass
                return res
            if fetch_all:
                return await cursor.fetchall()
            
            try:
                await cursor.fetchall() # Drenar siempre
            except psycopg.ProgrammingError:
                pass
            return []

async def aexecute_sql_write(query: str, params: tuple = None, returning: bool = False):
    """Ejecuta una transacción INSERT/UPDATE/DELETE asíncrona."""
    if 'async_connection_pool' not in globals() or not async_connection_pool:
        raise RuntimeError("db async_connection_pool is not available.")
    
    from psycopg.rows import dict_row
    import psycopg
    
    async with async_connection_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            if returning:
                return await cursor.fetchall()
                
            # Aunque no retorne nada (e.g. INSERT sin RETURNING), debemos asegurar
            # que el cursor consume cualquier mensaje final (CommandComplete)
            try:
                await cursor.fetchall()
            except psycopg.ProgrammingError:
                pass # The cursor has no results (expected for some queries)
            return True
