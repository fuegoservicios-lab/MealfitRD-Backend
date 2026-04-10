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
if SUPABASE_DB_URL:
    try:
        from psycopg_pool import ConnectionPool
        # Limpiar comillas basura por si acaso
        clean_url = SUPABASE_DB_URL.strip().strip("'").strip('"')
        def get_client_kwargs():
            return {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }

        connection_pool = ConnectionPool(
            conninfo=clean_url, 
            min_size=2,
            max_size=20, 
            max_idle=300,  # Kill idle connections after 5 minutes
            reconnect_timeout=5,  # Wait up to 5s for reconnection
            kwargs=get_client_kwargs(),
            check=ConnectionPool.check_connection,  # Health check on each checkout
            open=False
        )
        logger.info("🔌 [psycopg] ConnectionPool de Postgres configurado con keepalives + health checks.")
    except Exception as pool_err:
        logger.error(f"⚠️ [psycopg] Error configurando ConnectionPool: {pool_err}")

def close_connection_pool():
    if connection_pool:
        connection_pool.close()
        logger.info("Connection pool cerrado.")

def execute_sql_query(query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
    """Ejecuta una consulta SQL directa usando el pool de psycopg."""
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
            return True

