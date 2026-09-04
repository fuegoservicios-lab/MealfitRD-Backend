from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from datetime import datetime, timedelta, timezone
import os
import re
import logging
from typing import Optional, List, Dict, Any, Tuple, Union, overload, Literal
from dotenv import load_dotenv
# [P1-DB-STMT-TIMEOUT · 2026-05-27] Helper registrado en `_KNOBS_REGISTRY`
# (visible en /health/version y /api/system/knobs). `knobs` no tiene deps
# internas → import a top-level seguro, sin riesgo de ciclo.
from knobs import _env_int as _knob_env_int
# [P0-TEST-DB-ISOLATION-3 · 2026-07-29] Import top-level de `psycopg` (best
# effort, `ImportError` → None) para poder subclasificar `psycopg.Cursor` /
# `psycopg.AsyncCursor` más abajo (`_GuardedSyncCursor`/`_GuardedAsyncCursor`).
# Ya es dependencia dura de `psycopg_pool` (importado condicional en el
# bloque `MEALFIT_DB_BACKEND == "neon"` más abajo) — este import no añade
# ninguna dependencia nueva, solo la hace explícita donde se necesita.
try:
    import psycopg as _psycopg
except ImportError:  # pragma: no cover - psycopg siempre instalado en runtime real
    _psycopg = None



logger = logging.getLogger(__name__)
# Suprimir warnings cosméticos de psycopg.pool sobre conexiones ACTIVE retornadas al pool (p.ej. por LangGraph PostgresSaver)
logging.getLogger("psycopg.pool").setLevel(logging.ERROR)

load_dotenv()

# Knobs MEALFIT_DB_POOL_* para sintonizar el psycopg ConnectionPool sin redeploy.
# Existen porque bajo picos (RAG + cron chunk_queue + cache writes en paralelo)
# se observó saturación con `couldn't get a connection after 30.00 sec`. Permite:
#   - Subir MAX_SIZE para absorber concurrencia mayor.
#   - Bajar TIMEOUT_S para fallar rápido y diagnosticar antes de que se acumule cola.
#   - Ajustar MAX_IDLE_S por debajo del idle timeout del pooler si éste cambia.
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
#     El Transaction Pooler de Neon (6543) tolera muchas conexiones (multiplexa via pgBouncer).
#   - TIMEOUT 30s→15s: fail-fast para diagnosticar saturación antes de que se acumule cola
#     de waiters. APScheduler ya skipea jobs cuando este timeout vence; bajarlo evita que
#     un solo waiter bloquee el slot del job 15s extra.
#   - MIN_SIZE sin cambio (2): no consumir conexiones idle en entornos low-traffic.
# Override cualquiera con `MEALFIT_DB_POOL_*` si el comportamiento previo era preferido
# o si el pooler reporta saturación al alza.
# [P1-POOL-DEFAULTS-SSOT · 2026-09-04] Defaults de CÓDIGO = valores afinados. Antes el
# default era max=60 (y async max=12): la configuración que SATURÓ el pooler en mayo
# (P0-DB-POOL-PGBOUNCER-SATURATION / P0-POOL-FREETIER-RETUNE) y que sólo el `.env` local
# del dueño corregía. Un `.env` reseteado o un deploy nuevo volvía al estado saturado sin
# que ningún test lo viera (los guards leían ese mismo `.env`). Cierra la regresión
# silenciosa con el mismo criterio que P1-VERIFIED-ONLY-DEFAULT-ON: el default seguro
# vive en el código; el env var del operador sigue ganando.
DB_POOL_MIN_SIZE = _int_env("MEALFIT_DB_POOL_MIN_SIZE", 2, 0, 50)
DB_POOL_MAX_SIZE = _int_env("MEALFIT_DB_POOL_MAX_SIZE", 12, 1, 200)
DB_POOL_TIMEOUT_S = _float_env("MEALFIT_DB_POOL_TIMEOUT_S", 20.0, 1.0, 120.0)
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
# Si migras a un plan dedicado/Pro, subir ambos: SYNC max=60, ASYNC max=20.
DB_ASYNC_POOL_MIN_SIZE = _int_env("MEALFIT_DB_ASYNC_POOL_MIN_SIZE", 2, 0, 20)
# [P1-POOL-ASYNC-SPLIT-TUNE · 2026-05-16] Subido 8→12 tras observar que post-split
# inicial los warnings "couldn't get a connection after 20.00 sec" seguían ~12
# por plan (espaciados, no en burst). Causa probable: con max=8, los crons
# concurrent (drain_pending_facts, worker_metrics, etc.) + CB resets de 3 day_gen
# paralelos no caben en 8 slots. Subir a 12 da margen sin acercar peligrosamente
# al cap pgBouncer (sync 25 + async 12 = 37, todavía manejable con keepalive
# recycling rápido). Si los warnings persisten >5/plan tras este tune, investigar
# connection leak o subir clamp del knob.
# [P1-POOL-DEFAULTS-SSOT] 12→10 y 20→12 s (P1-BESTEFFORT-DB-CB): sync 12 + async 10 = 22 ≤ 30.
DB_ASYNC_POOL_MAX_SIZE = _int_env("MEALFIT_DB_ASYNC_POOL_MAX_SIZE", 10, 1, 100)
DB_ASYNC_POOL_TIMEOUT_S = _float_env("MEALFIT_DB_ASYNC_POOL_TIMEOUT_S", 12.0, 1.0, 120.0)

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
# statement_timeout (vuelve a depender sólo del backstop del pooler);
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
    # [P1-NEON-DB-MIGRATION · 2026-06-13] search_path con `extensions`. Neon crea
    # la extensión pgvector en el schema `extensions` (el layout del proveedor
    # anterior los tenía en el path por defecto), pero en Neon el search_path
    # por default es solo `"$user", public` — sin esto, un `::vector` o un
    # operador `<=>` app-side falla con `type "vector" does not exist`. Lo
    # replicamos explícito para paridad. `extensions` solo tiene objetos de
    # extensión (cero colisión con `public`). Las funciones match_* tienen su
    # propio `SET search_path` y no dependen de esto.
    stmts.append("SET search_path TO public, extensions")
    if DB_STATEMENT_TIMEOUT_MS > 0:
        stmts.append(f"SET statement_timeout = {int(DB_STATEMENT_TIMEOUT_MS)}")
    if DB_IDLE_IN_TXN_TIMEOUT_MS > 0:
        stmts.append(
            f"SET idle_in_transaction_session_timeout = {int(DB_IDLE_IN_TXN_TIMEOUT_MS)}"
        )
    return stmts

# [P1-NEON-AUTH-MIGRATION · 2026-06-13] Cliente de object storage ELIMINADO.
# Auth → Neon Auth (neon_auth.verify_neon_jwt). Datos → Neon (execute_sql_*).
# `_storage_client = None` es un placeholder para el object storage de
# visual_diary (pendiente de migrar a un provider de object storage nuevo;
# vision está disabled). Los callsites en db_chat/db_profiles/shopping_calculator/
# diary usan guards `if _storage_client:` que quedan no-op mientras sea None.
# NO se importa ningún paquete de storage (removido de requirements).
_storage_client = None

# Configuración del ConnectionPool para psycopg
# [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Hay DOS pools:
#   - `connection_pool` (sync) + `async_connection_pool`: tráfico genérico de la
#     app (queries del frontend, crons, RAG, embeddings). Usa **Transaction
#     Pooler** (6543) para multiplexar conexiones via pgBouncer/Supavisor (Neon).
#   - `chat_checkpoint_pool`: SOLO para `langgraph.checkpoint.postgres.PostgresSaver`.
#     Usa el URL ORIGINAL (port 5432, session mode). Razón documentada en el
#     bloque donde se construye.
connection_pool = None
async_connection_pool = None  # Siempre declarado a nivel de módulo para garantizar importabilidad
chat_checkpoint_pool = None  # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20]
# [P1-NEON-DB-MIGRATION · 2026-06-12] URL session-mode resuelta según backend
# activo (Neon direct, puerto 5432). Consumida por el leader lock del
# scheduler (app.py::_build_session_mode_db_url) — advisory locks de sesión
# requieren session mode; un transaction pooler los liberaría por sentencia.
DB_SESSION_MODE_URL = None

# [P1-NEON-DB-MIGRATION · 2026-06-12] Selección del backend de DATOS Postgres.
# Knob `MEALFIT_DB_BACKEND`: solo "neon" soportado (único backend de datos tras
# la migración 2026-06-13). Cualquier otro valor → fail-loud (la rama legacy fue
# eliminada).
#   - neon: pools principales ← NEON_DATABASE_URL_POOLED (PgBouncer transaction
#     mode de Neon); chat_checkpoint_pool ← NEON_DATABASE_URL (endpoint directo,
#     session mode — mismo contrato que P1-CHECKPOINT-POOL-SPLIT). Auth y object
#     storage son servicios separados (el placeholder `_storage_client = None`
#     se conserva por compat de imports hasta cablear object storage).
# Cutover/rollback sin redeploy: flip del env var + restart. Si backend=neon
# pero faltan los URLs, se aborta la config de pools (fail-loud → /ready 503)
# en vez de degradar silenciosamente: un fallback silencioso escribiría en la
# DB equivocada (split-brain).
# [P1-SUPABASE-CLEANUP · 2026-06-13] Default a "neon": el proyecto legacy fue
# eliminado, así que un env var ausente cae en Neon (el único backend real).
# La rama legacy se eliminó por completo; cualquier valor distinto de "neon"
# hace fail-loud (ver el raise más abajo).
MEALFIT_DB_BACKEND = (os.environ.get("MEALFIT_DB_BACKEND") or "neon").strip().lower()
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL")
NEON_DATABASE_URL_POOLED = os.environ.get("NEON_DATABASE_URL_POOLED")

# [P1-DBCORE-CONFIGURE-HOIST · 2026-09-04] Los configuradores de conexión viven a nivel de
# módulo, NO dentro de la rama `if MEALFIT_DB_BACKEND == "neon": try: …` que construye los
# pools. Ahí dentro sólo existían cuando había NEON_DATABASE_URL*: en un checkout sin DB
# (CI, un clon limpio) `db_core.configure_sync_conn` ni siquiera era un atributo y los
# tres tests que cablean la guarda P0-TEST-DB-ISOLATION contra el pool REAL fallaban con
# AttributeError — es decir, la única prueba de que la guarda está enganchada al pool de
# producción no corría precisamente donde importa. No dependen de nada de la rama:
# `_install_write_guard_on_*` y `_session_timeout_statements()` son de módulo.
def configure_sync_conn(conn):
    conn.autocommit = True
    # pgBouncer Transaction Mode no soporta server-side prepared statements.
    # Deshabilitar auto-prepare para evitar errores "_pg3_N requires N params".
    conn.prepare_threshold = None
    # [P0-TEST-DB-ISOLATION-2 · 2026-07-29] Envuelve `conn.cursor` para que
    # CUALQUIER escritura cruda (no solo la que pasa por `execute_sql_write`)
    # quede sujeta a la guarda test-vs-prod. Ver comentario extenso arriba de
    # `_install_write_guard_on_connection`.
    _install_write_guard_on_connection(conn)
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
    # [P0-TEST-DB-ISOLATION-2 · 2026-07-29] Variante async del wrapper sync
    # de arriba — mismo razonamiento, ver `_install_write_guard_on_connection`.
    _install_write_guard_on_async_connection(conn)
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
    # [P0-TEST-DB-ISOLATION-2 · 2026-07-29] El checkpointer de LangGraph
    # también persiste (`checkpoints`/`checkpoint_writes`) via SQL crudo del
    # driver `PostgresSaver` — mismo riesgo si un test no-e2e llega a correr
    # el grafo contra este pool real.
    _install_write_guard_on_connection(conn)

if MEALFIT_DB_BACKEND == "neon":
    try:
        from psycopg_pool import ConnectionPool, AsyncConnectionPool

        # [P1-NEON-DB-MIGRATION] Neon es el ÚNICO backend de datos activo.
        # Fail-loud si faltan los URLs → la config de pools aborta (/ready 503)
        # en vez de dejar un pool a medio configurar.
        if not (NEON_DATABASE_URL and NEON_DATABASE_URL_POOLED):
            raise RuntimeError(
                "[P1-NEON-DB-MIGRATION] Requiere NEON_DATABASE_URL (direct) y "
                "NEON_DATABASE_URL_POOLED (pooler)."
            )
        clean_url = NEON_DATABASE_URL_POOLED.strip().strip("'").strip('"')
        original_session_url = NEON_DATABASE_URL.strip().strip("'").strip('"')
        logger.info(
            "🔌 [NEON] Backend de datos: NEON "
            "(pooled→pools principales, direct→chat_checkpoint_pool)."
        )

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
        # Pooler (Neon/Supavisor port 6543), que mata conexiones idle
        # agresivamente (~10-30s). Durante el chat flow, el PostgresSaver
        # mantiene una connection abierta del pool ~5-15s mientras espera la
        # LLM call; el pooler la cierra mid-stream → al `put_writes` final, la
        # conexión ya está muerta. Defensivamente psycopg keepalives no ayudan
        # porque el pooler corta a nivel aplicación, no TCP.
        #
        # Fix: el checkpointer usa `original_session_url` (sin rewrite a
        # 6543) → conexión directa session-mode (5432). Session mode no
        # mata conexiones idle de forma agresiva. Pool pequeño (1-4) porque
        # el chat-flow es low-concurrency (1 stream por user activo).
        #
        # [P1-CHAT-CHECKPOINT-FIX · 2026-05-20] El WARN legacy sobre rewrite
        # de puerto fue removido — la nueva lógica de force-rewrite arriba
        # garantiza que `original_session_url` SIEMPRE apunte a :5432
        # (session mode), haciendo el WARN inalcanzable. Mantenemos el
        # try/except por si
        # `ConnectionPool(...)` falla por otras razones (DNS, auth, etc).
        # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20] `min_size=0` + `max_idle=30`
        # son recycling agresivo contra el idle-kill del session-pooler.
        # Pre-fix (`min_size=1, max_idle=300`): el pool pre-warmaba 1 conn al
        # startup que envejecía ~44s antes del primer chat, y LangGraph la
        # mantenía checkout durante el LLM call (10-30s). Al `put_writes`
        # final, el pooler ya la había matado → SSL bad length / EOF detected.
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
                "rancias que el pooler mata mid-pipeline → SSL bad length]."
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

# [P-TYPING-1] Overloads para resolución de tipos por flag: `fetch_one=True`
# devuelve `dict | None`; `fetch_all=True` (o sin flag, default seguro) devuelve
# `list[dict]`. Sin esto, callsites que hacen `res.get(...)` / `res[0]` veían una
# unión sucia y pyright emitía cientos de reportAttributeAccessIssue. Type-only.
@overload
def execute_sql_query(query: str, params: Optional[tuple] = ..., *, fetch_one: Literal[True], fetch_all: bool = ...) -> Optional[Dict[str, Any]]: ...
@overload
def execute_sql_query(query: str, params: Optional[tuple] = ..., *, fetch_all: Literal[True], fetch_one: Literal[False] = ...) -> List[Dict[str, Any]]: ...
@overload
def execute_sql_query(query: str, params: Optional[tuple] = ..., fetch_one: Literal[False] = ..., fetch_all: Literal[False] = ...) -> List[Dict[str, Any]]: ...
def execute_sql_query(query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
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
            cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
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

# [P-TYPING-1] Overloads: `returning=True` devuelve `list[dict]` (filas RETURNING);
# sin `returning` devuelve `bool`. Type-only, runtime intacto.
@overload
def execute_sql_write(query: str, params: Optional[tuple] = ..., *, returning: Literal[True], lock_timeout_ms: Optional[int] = ...) -> List[Dict[str, Any]]: ...
@overload
def execute_sql_write(query: str, params: Optional[tuple] = ..., returning: Literal[False] = ..., lock_timeout_ms: Optional[int] = ...) -> bool: ...
# [P1-TEST-ALERT-POLLUTION · 2026-07-25] La suite de tests corre contra la MISMA base Neon que
# producción (no hay staging), y ~9 módulos escriben `system_alerts` con SQL directo. Resultado
# medido: una noche de corridas dejó 3 incidentes FALSOS abiertos en el tablero —
#
#     plan_persist_failed:user-abc          meta: 'invalid input syntax for type uuid: "user-abc"'
#     plan_data_corrupted:plan-corrupt:_last_chunk_learning
#     plan_data_corrupted:plan-corrupt:_recent_chunk_lessons
#
# `user-abc` y `plan-corrupt` son fixtures literales (test_chunk_recovery_7day_plan.py,
# test_chunked_learning_propagation.py). Así es como se acaba ignorando el canal de alertas: se
# llena de ruido y el operador deja de mirarlo.
#
# La guarda vive en el ÚNICO cuello de botella (este writer) en vez de en los 9 call sites, y usa
# `PYTEST_CURRENT_TEST` — la variable que pytest pone durante la ejecución — porque el señal
# preciso es "esto es un test", NO el entorno: un dev corriendo la app a mano apunta a la misma
# base y SÍ quiere sus alertas reales.
#
# Cubre INSERT y también UPDATE/DELETE: un test que "resuelve" alertas podría cerrar una REAL,
# que es peor que crear una falsa. Los tests que afirman sobre la inserción parchean
# `execute_sql_write` a nivel de módulo (`patch("cron_tasks.execute_sql_write", ...)`), así que
# reemplazan esta función entera y no ven la guarda — sus aserciones siguen intactas.
_SYSTEM_ALERTS_WRITE_RE = re.compile(r"\bsystem_alerts\b", re.IGNORECASE)


def _suppress_test_alert_write(query: Optional[str]) -> bool:
    """True si hay que descartar esta escritura a `system_alerts` por venir de un test."""
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("MEALFIT_ALLOW_TEST_SYSTEM_ALERTS", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not query or not _SYSTEM_ALERTS_WRITE_RE.search(query):
        return False
    logging.debug("[P1-TEST-ALERT-POLLUTION] escritura a system_alerts descartada (corrida de tests)")
    return True


# [P0-TEST-DB-ISOLATION · 2026-07-29] Incidente: un fixture de test (`fresh_plan` en
# `test_chunk_corrupted_plan_data_pauses.py`, y 3 copias hermanas) elegía su usuario con
# `SELECT id FROM user_profiles LIMIT 1` — sin ORDER BY, un accidente de orden físico de
# almacenamiento, no intención — y ese SELECT devolvía determinísticamente al OWNER real.
# El INSERT en `meal_plans` que seguía escribía un plan corrupto (sin `plan_data.name`) en la
# cuenta REAL, y sobrevivió porque el proceso de test murió antes del `yield`'s teardown.
#
# Censo (ver runbook / CLAUDE.md P0-TEST-DB-ISOLATION) midió: no existe base de test separada
# — `backend/.env` define un único NEON_DATABASE_URL con ENVIRONMENT=production — y solo
# 12/1544 archivos de test escriben de verdad contra esa base, todos legítimos (E2E contra
# Neon real con fixtures que crean SU PROPIO usuario sintético). Bloquear TODA escritura de
# test por defecto rompería esos 12 archivos; en cambio, esta guarda es table-agnostic pero
# marker-aware vía `@pytest.mark.e2e`: solo los tests marcados e2e (los que YA declaran
# "necesito Neon real") pueden escribir bajo pytest. `conftest.py::_guard_test_writes_to_prod`
# (autouse) fija `_CURRENT_TEST_IS_E2E` según el marker del test en curso, ANTES de que
# cualquier fixture de setup corra.
#
# Por qué NO reusar `_suppress_test_alert_write` (silent no-op): system_alerts es ruido
# tolerable de silenciar; un INSERT/UPDATE/DELETE en `meal_plans`/`user_profiles`/etc que
# debería fallar sonoramente NO debe degradar en silencio — un test que asume que su propio
# INSERT ocurrió y luego lee `[]` de vuelta falla de forma confusa. `RuntimeError` con mensaje
# accionable es la superficie correcta ("loud es el punto", brief P0-TEST-DB-ISOLATION).
_CURRENT_TEST_IS_E2E = False


def _looks_like_test_db_url(url: Optional[str]) -> bool:
    """True si la URL de conexión ya apunta a algo que NO es la Neon de producción.

    Hoy (2026-07-29) esto siempre es False — censo confirmó una única
    NEON_DATABASE_URL sin staging/test separado — pero deja el guard listo para
    cuando exista una base de test real (recomendación censo P0-TEST-DB-ISOLATION
    item 4): en ese caso las escrituras de test dejan de necesitar el escape
    hatch por completo.
    """
    if not url:
        return False
    _low = url.lower()
    return any(tok in _low for tok in ("test", "staging", "localhost", "127.0.0.1"))


def _db_target_is_nonprod() -> tuple[bool, Optional[str]]:
    """[P0-TEST-DB-DUAL-URL · 2026-07-31] ¿Apuntan a algo que NO es producción **las dos**
    URLs? Devuelve `(es_nonprod, motivo_de_desacuerdo)`.

    POR QUÉ NO BASTA MIRAR UNA
    El pool se conecta con `NEON_DATABASE_URL_POOLED` (ver `clean_url` en la
    inicialización del pool, arriba), pero este guard evaluaba SOLO
    `NEON_DATABASE_URL`, la directa. Así que una migración a medias —apuntar la directa
    al branch de test y olvidar la pooled, que es lo natural porque la directa es la que
    se ve primero— hacía que el guard concluyera "esto es base de test, las escrituras
    son seguras" mientras **todas** las escrituras salían a PRODUCCIÓN por la pooled.
    Sin marker `e2e`, sin escape hatch y sin aviso: el guard pasaba de freno a acelerador
    justo al dar el paso que su propio docstring recomienda dar.

    FAIL-SECURE ANTE EL DESACUERDO
    Si una parece test y la otra producción, NO se asume test. Una configuración a medias
    es peor que cualquiera de los dos extremos, porque el operador CREE que está aislado.
    Se devuelve `False` (trátalo como producción) más un motivo, para que el mensaje del
    bloqueo pueda explicar la causa real en vez de repetir el genérico.
    """
    _directa = os.environ.get("NEON_DATABASE_URL")
    _pooled = os.environ.get("NEON_DATABASE_URL_POOLED")
    _d_test = _looks_like_test_db_url(_directa)
    _p_test = _looks_like_test_db_url(_pooled)
    if _d_test and _p_test:
        return True, None
    if _d_test != _p_test:
        return False, (
            "NEON_DATABASE_URL "
            f"{'parece de test' if _d_test else 'apunta a producción'} pero "
            "NEON_DATABASE_URL_POOLED "
            f"{'parece de test' if _p_test else 'apunta a producción'}. "
            "El pool usa la POOLED, así que las escrituras irían a producción aunque la "
            "directa apunte al branch de test. Mueve las DOS."
        )
    return False, None


def _guard_test_write_to_prod(query: Optional[str]) -> None:
    """Bloquea (RuntimeError) escrituras reales a Neon prod desde tests NO marcados e2e.

    No-op fuera de pytest (PYTEST_CURRENT_TEST ausente) — nunca afecta runtime normal.
    """
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if os.environ.get("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", "").strip().lower() in ("1", "true", "yes"):
        return
    if _CURRENT_TEST_IS_E2E:
        return
    # [P0-TEST-DB-DUAL-URL · 2026-07-31] Exige que las DOS URLs sean no-producción; el
    # desacuerdo se trata como producción. Ver `_db_target_is_nonprod`.
    _es_nonprod, _desacuerdo = _db_target_is_nonprod()
    if _es_nonprod:
        return
    _preview = " ".join((query or "").split())[:160]
    if _desacuerdo:
        raise RuntimeError(
            "[P0-TEST-DB-DUAL-URL] BLOQUEADO: configuración de base de test A MEDIAS. "
            f"{_desacuerdo}\n"
            "Se trata como producción a propósito: creer que estás aislado sin estarlo es "
            "peor que no estarlo.\n"
            f"Query bloqueada: {_preview!r}"
        )
    raise RuntimeError(
        "[P0-TEST-DB-ISOLATION] BLOQUEADO: un test SIN @pytest.mark.e2e intentó escribir en "
        "Neon PRODUCCIÓN (no existe base de test separada — ver CLAUDE.md 'DB + Auth: 100% "
        "Neon'). Qué hacer AHORA:\n"
        "  1. Si este test necesita escribir de verdad (E2E contra Neon real): añade "
        "`pytestmark = pytest.mark.e2e` al tope del archivo y usa la fixture "
        "`seeded_user_profile` de conftest.py (o tu propio `str(uuid.uuid4())`) para el "
        "usuario — JAMÁS `SELECT ... FROM user_profiles LIMIT 1` sin ORDER BY: eso adopta "
        "la fila de un usuario REAL (ver incidente P0-TEST-DB-ISOLATION 2026-07-29).\n"
        "  2. Si esta escritura es un accidente (el test debería estar mockeado): parchea "
        "`execute_sql_write` a nivel de módulo en el test en vez de dejar pasar la escritura "
        "real.\n"
        "  3. Escape hatch de emergencia, NO usar en CI: MEALFIT_ALLOW_TEST_WRITES_TO_PROD=1.\n"
        f"Query bloqueada: {_preview!r}"
    )


# [P0-TEST-DB-ISOLATION-2 · 2026-07-29 · code review] El comentario original (arriba)
# afirmaba que `execute_sql_write` es "el ÚNICO cuello de botella de escritura" —
# FALSO. Censo del review: 27+ call sites (`update_plan_data_atomic` en db_plans.py —
# el patrón PREFERIDO de la invariante I7, CLAUDE.md —, el endpoint `/name` de
# rename en routers/plans.py, más cron_tasks.py/db_profiles.py/dreaming.py/
# routers/notifications.py) escriben abriendo `connection_pool.connection()` CRUDO y
# llamando `cursor.execute(...)` directo, sin pasar NUNCA por `execute_sql_write` — la
# guarda de arriba nunca los ve. Un futuro fixture que reintroduzca el incidente
# (`SELECT ... LIMIT 1` sin ORDER BY) y persista via CUALQUIERA de esos call sites
# escribiría en Neon prod con CERO interferencia.
#
# Perseguir 27+ call sites uno por uno queda desactualizado con el próximo call site
# nuevo. En su lugar, la guarda se instala UNA VEZ por CONEXIÓN FÍSICA dentro de
# `configure_sync_conn`/`configure_async_conn` (los callbacks `configure=` de los
# pools, más abajo): se envuelve `conn.cursor` para que CUALQUIER cursor que esa
# conexión produzca intercepte `execute()` cuando la sentencia empiece por
# INSERT/UPDATE/DELETE/TRUNCATE. psycopg_pool invoca `configure` solo al CREAR la
# conexión (no en cada checkout) — el wrapper vive mientras la conexión viva, así que
# cubre TODOS los checkouts futuros sin importar qué función la tome del pool. Esto
# hace de `connection_pool`/`async_connection_pool` el cuello de botella real que el
# comentario original afirmaba (incorrectamente) que ya era `execute_sql_write`.
#
# Tests que patchean `db_core.connection_pool` a nivel de módulo con un Mock (p.ej.
# `test_p1_hist_5_rename_atomic.py`) NO pasan por `configure_sync_conn` — nunca crean
# una conexión real, así que no ven esta guarda, pero tampoco tocan Neon real: sin gap.
_WRITE_STATEMENT_RE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|TRUNCATE)\b", re.IGNORECASE)


def _install_write_guard_on_cursor(cursor):
    """Envuelve `cursor.execute` (sync) para invocar `_guard_test_write_to_prod` en
    sentencias de escritura. Los SELECT pasan intactos — la guarda original solo
    protegía escrituras; bloquear lecturas de tests no-e2e rompería call sites
    legítimos que leen contra el pool real (p.ej. mocks parciales).

    [P0-TEST-DB-ISOLATION-3 · 2026-07-29] Este path (asignación de INSTANCIA
    `cursor.execute = ...`) solo es seguro sobre objetos plain con `__dict__`
    (los test doubles `_FakeCursor`/`_FakeAsyncCursor` de
    `test_p0_test_db_isolation.py`). Se PRESERVA sin cambios exclusivamente
    para esos fakes — `_install_write_guard_on_connection`/`_install_write_guard_on_async_connection`
    de abajo ya NO lo usan contra una conexión psycopg real; ver
    `_GuardedSyncCursor` para la razón (`Cursor`/`AsyncCursor` son
    `__slots__`-only, esta asignación lanza `AttributeError` incondicional
    sobre ellos)."""
    _orig_execute = cursor.execute

    def _guarded_execute(query, *args, **kwargs):
        if isinstance(query, str) and _WRITE_STATEMENT_RE.match(query):
            _guard_test_write_to_prod(query)
        return _orig_execute(query, *args, **kwargs)

    cursor.execute = _guarded_execute
    return cursor


def _install_write_guard_on_async_cursor(cursor):
    """Variante async de `_install_write_guard_on_cursor` — `execute` es awaitable.
    Mismo alcance reducido a test doubles, ver docstring de la variante sync."""
    _orig_execute = cursor.execute

    async def _guarded_execute(query, *args, **kwargs):
        if isinstance(query, str) and _WRITE_STATEMENT_RE.match(query):
            _guard_test_write_to_prod(query)
        return await _orig_execute(query, *args, **kwargs)

    cursor.execute = _guarded_execute
    return cursor


# [P0-TEST-DB-ISOLATION-3 · 2026-07-29] Incidente causado por P0-TEST-DB-ISOLATION-2:
# `_install_write_guard_on_connection`/`_install_write_guard_on_async_connection`
# envolvían `conn.cursor` para devolver un cursor con `_install_write_guard_on_cursor`
# aplicado — es decir, `cursor.execute = _guarded_execute` como ATRIBUTO DE
# INSTANCIA. Eso funciona sobre objetos plain (los `_FakeCursor` de los tests)
# pero es SIEMPRE ilegal sobre un `psycopg.Cursor`/`AsyncCursor` REAL: ambas
# clases (y toda su MRO, `BaseCursor` incluido) son `__slots__`-only — sin
# `__dict__` de instancia — así que `cursor.execute = ...` lanza
# `AttributeError: 'Cursor' object attribute 'execute' is read-only`
# INCONDICIONALMENTE, sin importar la query, sin importar pytest.
#
# Cadena de fallo medida (2026-07-29, `PYTEST_CURRENT_TEST` completamente
# AUSENTE — esto rompía producción, no solo tests):
#   1. `configure_sync_conn` llama a `_install_write_guard_on_connection(conn)`.
#      Esa llamada en sí NO lanza (solo reemplaza `conn.cursor`).
#   2. Dos líneas después, el bloque best-effort `SET statement_timeout`
#      hace `with conn.cursor() as _c:` → invoca el factory recién
#      reemplazado → `_install_write_guard_on_cursor` → `AttributeError` ANTES
#      de que exista un cursor (antes de `__enter__`). Atrapado por el
#      `try/except Exception` preexistente de ese bloque → `configure_sync_conn`
#      retorna limpio, el pool cree que la conexión quedó configurada.
#   3. Pero `conn.cursor` quedó PERMANENTEMENTE apuntando al factory
#      guardado → CUALQUIER `.cursor()` futuro sobre esa conexión física
#      vuelve a lanzar, para siempre.
#   4. `ConnectionPool.check_connection` (el `check=` del pool — hace
#      `conn.execute("")`, que internamente es `self.cursor().execute(...)`)
#      la encuentra "rota" en cada checkout → la descarta y crea una
#      conexión de reemplazo.
#   5. La conexión de reemplazo pasa por `configure_sync_conn` de nuevo →
#      vuelve a envenenarse → vuelve a descartarse. Bucle hasta agotar
#      `MEALFIT_DB_POOL_TIMEOUT_S` (20s en `.env`) →
#      `psycopg_pool.PoolTimeout: couldn't get a connection after 20.00 sec`.
#      CERO conexiones podían escapar el bucle — el pool quedaba 100%
#      incapaz de servir NINGUNA query (lectura o escritura), sync Y async
#      Y el pool del checkpointer de LangGraph, en tests Y EN PRODUCCIÓN.
#
# Fix: en vez de asignar sobre la INSTANCIA del cursor, subclasificar
# `psycopg.Cursor`/`AsyncCursor` con `execute` como MÉTODO DE CLASE real
# (nunca toca `__slots__`) y cablearla vía `conn.cursor_factory` — el hook
# que psycopg YA usa internamente (`Connection.__init__` hace
# `self.cursor_factory = Cursor`; `Connection.cursor()` construye
# `self.cursor_factory(self, row_factory=row_factory)` cuando no hay
# `name=`). `cursor_factory` vive en `Connection`/`AsyncConnection`, que NO
# son `__slots__`-only (tienen `__dict__` real, `self.cursor_factory = Cursor`
# ya lo demuestra en el propio `__init__` de psycopg) — asignarlo está
# soportado. Cubre exactamente lo que el wrap legacy pretendía: CUALQUIER
# cursor que esa conexión física produzca a lo largo de su vida, para TODOS
# los checkouts futuros, sin tocar los 27+ call sites (I7/CLAUDE.md).
if _psycopg is not None:

    class _GuardedSyncCursor(_psycopg.Cursor):
        """`psycopg.Cursor` con `execute` guardado como MÉTODO REAL (no
        atributo de instancia) — ver comentario extenso arriba. Cableada vía
        `conn.cursor_factory`, nunca instanciada directamente."""

        def execute(self, query, *args, **kwargs):
            if isinstance(query, str) and _WRITE_STATEMENT_RE.match(query):
                _guard_test_write_to_prod(query)
            return super().execute(query, *args, **kwargs)

    class _GuardedAsyncCursor(_psycopg.AsyncCursor):
        """Variante async de `_GuardedSyncCursor` — `execute` es awaitable."""

        async def execute(self, query, *args, **kwargs):
            if isinstance(query, str) and _WRITE_STATEMENT_RE.match(query):
                _guard_test_write_to_prod(query)
            return await super().execute(query, *args, **kwargs)

else:  # pragma: no cover - psycopg siempre instalado en runtime real
    _GuardedSyncCursor = None
    _GuardedAsyncCursor = None


def _install_write_guard_on_connection(conn) -> None:
    """Cablea la guarda test-vs-prod para TODO cursor que `conn` produzca a
    lo largo de su vida. Cablear desde `configure_sync_conn`/
    `configure_checkpoint_conn` — corre UNA VEZ por conexión física.

    [P0-TEST-DB-ISOLATION-3 · 2026-07-29] Dos caminos:
      - Conexión psycopg REAL (`hasattr(conn, "cursor_factory")`, ver
        comentario extenso arriba de `_GuardedSyncCursor`): swap de
        `conn.cursor_factory` a la subclase con `execute` como método real.
        Nunca toca `__slots__` — el bug que causó el `PoolTimeout` universal.
      - Test double sin `cursor_factory` (los `_FakeConn` de
        `test_p0_test_db_isolation.py`, que no imitan ese atributo): legacy
        fallback, wrap de `conn.cursor` + `_install_write_guard_on_cursor`
        vía asignación de instancia — funciona porque esos fakes son
        objetos plain con `__dict__`.
    """
    if _GuardedSyncCursor is not None and hasattr(conn, "cursor_factory"):
        conn.cursor_factory = _GuardedSyncCursor
        return
    _orig_cursor = conn.cursor

    def _guarded_cursor(*args, **kwargs):
        return _install_write_guard_on_cursor(_orig_cursor(*args, **kwargs))

    conn.cursor = _guarded_cursor


def _install_write_guard_on_async_connection(conn) -> None:
    """Variante async de `_install_write_guard_on_connection` — cablear desde
    `configure_async_conn`. Mismos dos caminos, ver docstring de la variante
    sync."""
    if _GuardedAsyncCursor is not None and hasattr(conn, "cursor_factory"):
        conn.cursor_factory = _GuardedAsyncCursor
        return
    _orig_cursor = conn.cursor

    def _guarded_cursor(*args, **kwargs):
        return _install_write_guard_on_async_cursor(_orig_cursor(*args, **kwargs))

    conn.cursor = _guarded_cursor


def execute_sql_write(query: str, params: Optional[tuple] = None, returning: bool = False, lock_timeout_ms: Optional[int] = None) -> Union[List[Dict[str, Any]], bool]:
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
    # [P1-TEST-ALERT-POLLUTION · 2026-07-25] Ver `_suppress_test_alert_write`.
    if _suppress_test_alert_write(query):
        return [] if returning else True

    # [P0-TEST-DB-ISOLATION · 2026-07-29] Ver `_guard_test_write_to_prod`. DESPUÉS de la
    # suppression de system_alerts (ese ruido se sigue silenciando, no bloqueando) y ANTES
    # de tocar el pool real — ningún byte llega a Neon si un test no-e2e cae aquí.
    _guard_test_write_to_prod(query)

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
                        cursor.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'")  # pyright: ignore[reportArgumentType, reportCallIssue]  # psycopg LiteralString FP
                    except Exception as set_err:
                        # Best-effort: si el SET LOCAL falla, seguimos sin él
                        # (comportamiento de antes — mejor que abortar el write).
                        logger.debug(f"[P1-3] SET LOCAL lock_timeout falló: {set_err}")
                    cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
                    if returning:
                        return cursor.fetchall()
                    try:
                        cursor.fetchall()
                    except psycopg.ProgrammingError:
                        pass
                    return True
        else:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
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
                    cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
                    try:
                        cursor.fetchall()
                    except psycopg.ProgrammingError:
                        pass
        return True

async def aexecute_sql_query(query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
    """Ejecuta una consulta SQL asíncrona usando el pool de psycopg async."""
    if 'async_connection_pool' not in globals() or not async_connection_pool:
        raise RuntimeError("db async_connection_pool is not available.")
    
    from psycopg.rows import dict_row
    import psycopg
    
    async with async_connection_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
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

async def aexecute_sql_write(query: str, params: Optional[tuple] = None, returning: bool = False):
    """Ejecuta una transacción INSERT/UPDATE/DELETE asíncrona."""
    if 'async_connection_pool' not in globals() or not async_connection_pool:
        raise RuntimeError("db async_connection_pool is not available.")
    
    from psycopg.rows import dict_row
    import psycopg
    
    async with async_connection_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)  # pyright: ignore[reportArgumentType]  # psycopg exige LiteralString; query dinámico es el propósito del helper
            if returning:
                return await cursor.fetchall()
                
            # Aunque no retorne nada (e.g. INSERT sin RETURNING), debemos asegurar
            # que el cursor consume cualquier mensaje final (CommandComplete)
            try:
                await cursor.fetchall()
            except psycopg.ProgrammingError:
                pass # The cursor has no results (expected for some queries)
            return True
