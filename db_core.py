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



logger = logging.getLogger(__name__)
# Suprimir warnings cosméticos de psycopg.pool sobre conexiones ACTIVE retornadas al pool (ej. por LangGraph PostgresSaver)
logging.getLogger("psycopg.pool").setLevel(logging.ERROR)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# Configuración del ConnectionPool para psycopg (Usado por LangGraph PostgresSaver)
connection_pool = None
async_connection_pool = None  # Siempre declarado a nivel de módulo para garantizar importabilidad
if SUPABASE_DB_URL:
    try:
        from psycopg_pool import ConnectionPool, AsyncConnectionPool
        # Limpiar comillas basura por si acaso
        clean_url = SUPABASE_DB_URL.strip().strip("'").strip('"')
        
        # MEJORA 3: Connection Pooling - Forzar el uso del Transaction Pooler de Supabase (6543)
        # Esto previene el agotamiento de conexiones directas (Connection Exhaustion) si el volumen crece.
        if ".supabase." in clean_url and ":5432" in clean_url:
            clean_url = clean_url.replace(":5432", ":6543")
            logger.info("🔧 [psycopg] DB URL automatically rewritten to use Supabase Transaction Pooler (port 6543).")

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

        async def configure_async_conn(conn):
            await conn.set_autocommit(True)
            conn.prepare_threshold = None

        connection_pool = ConnectionPool(
            conninfo=clean_url, 
            min_size=2,
            max_size=30, 
            max_idle=300,  # Kill idle connections after 5 minutes
            reconnect_timeout=5,  # Wait up to 5s for reconnection
            kwargs=get_client_kwargs(),
            configure=configure_sync_conn,
            check=ConnectionPool.check_connection,  # Health check on each checkout
            open=False
        )
        
        async_connection_pool = AsyncConnectionPool(
            conninfo=clean_url, 
            min_size=2,
            max_size=30, 
            max_idle=300,
            reconnect_timeout=5,
            kwargs=get_client_kwargs(),
            configure=configure_async_conn,
            check=AsyncConnectionPool.check_connection,
            open=False
        )
        logger.info("🔌 [psycopg] ConnectionPool (Sync y Async) de Postgres configurado con autocommit=True, keepalives + health checks.")
    except Exception as pool_err:
        logger.error(f"⚠️ [psycopg] Error configurando ConnectionPool: {pool_err}")

def close_connection_pool():
    if connection_pool:
        connection_pool.close()
        logger.info("Connection pool cerrado.")

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

def execute_sql_write(query: str, params: tuple = None, returning: bool = False):
    """Ejecuta una transacción INSERT/UPDATE/DELETE."""
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available.")
    
    import psycopg
    from psycopg.rows import dict_row
    
    with connection_pool.connection() as conn:
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
