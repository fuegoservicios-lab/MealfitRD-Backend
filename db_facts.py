from functools import lru_cache as _lru_cache
import json
import time as _time
import threading as _threading
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
# [P1-NEON-DB-MIGRATION · 2026-06-12] Módulo migrado a SQL directo (psycopg
# pool, Neon). El cliente PostgREST `supabase` ya no se usa aquí — los datos
# Postgres viven en Neon; Supabase conserva solo Auth+Storage (fuera de db_*).
from db_core import connection_pool, execute_sql_query, execute_sql_write
from psycopg.types.json import Jsonb
from cache_manager import redis_client


# [P2-PROD-AUDIT-BUNDLE · 2026-05-27] Debounce in-process para
# `delete_expired_temporal_facts`. Pre-fix: 3 callsites side-effect en search
# (`get_user_facts_by_metadata`, `search_user_facts`, `search_user_facts_hybrid`)
# emitían 3 DELETEs consecutivos al mismo `user_id` en milisegundos durante
# un mismo pipeline run — observado en API logs 2026-05-28 (sintoma_temporal
# cleanup repetido 3× en 4ms). El cleanup real solo necesita correr ~1x/min
# por user; el resto son no-ops costosos (round-trip Supabase + WAL write
# por el `delete` que mata 0 rows si ya fue limpiado).
#
# Knob `MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S` (default 60, clamp [5, 3600]):
# ventana de skip per-user. Cuando se llama dentro de la ventana, el delete
# se omite silenciosamente. Cuando se llama tras la ventana, ejecuta y
# refresca el timestamp.
#
# Implementación: dict in-process `{user_id: last_cleanup_monotonic}` + lock.
# In-process porque (a) el cleanup es idempotente — perder el debounce tras
# restart solo hace 1 DELETE extra, no es bug; (b) cross-process sería overkill
# para algo que solo evita N+1 dentro del mismo request/pipeline.
_TEMPORAL_CLEANUP_LAST_RUN: Dict[str, float] = {}
_TEMPORAL_CLEANUP_LOCK = _threading.Lock()
# [P2-TEMPORAL-CLEANUP-GC · 2026-05-27] Umbral a partir del cual el record-path
# de `_should_skip_temporal_cleanup` barre entries stale. Evita escanear el
# dict cuando es pequeño; con muchos users distintos dispara la eviction.
_TEMPORAL_CLEANUP_GC_THRESHOLD = 512


def _temporal_cleanup_debounce_seconds() -> int:
    """Lee la ventana de debounce del env. Clamp defensivo [5, 3600]."""
    try:
        raw = int(os.environ.get(
            "MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S", "60"
        ) or 60)
    except (TypeError, ValueError):
        return 60
    if raw < 5:
        return 5
    if raw > 3600:
        return 3600
    return raw


def _should_skip_temporal_cleanup(user_id: Optional[str]) -> bool:
    """True si ya se hizo cleanup para este user dentro de la ventana de
    debounce. False (y registra el timestamp) si toca correr. Si `user_id`
    es None (cleanup global), nunca debouncea — esos son crons explícitos,
    no side-effects de search."""
    if not user_id:
        return False
    window_s = _temporal_cleanup_debounce_seconds()
    now = _time.monotonic()
    with _TEMPORAL_CLEANUP_LOCK:
        last = _TEMPORAL_CLEANUP_LAST_RUN.get(user_id, 0.0)
        if last and (now - last) < window_s:
            return True
        _TEMPORAL_CLEANUP_LAST_RUN[user_id] = now
        # [P2-TEMPORAL-CLEANUP-GC · 2026-05-27] Eviction oportunista bajo el
        # lock ya tomado. Pre-fix el dict crecía +1 por cada `user_id` distinto
        # para siempre (leak lento ~10MB/100k users). Barre entries con edad
        # > 2× ventana (un user fuera de esa ventana ya pasó el debounce de
        # todos modos → re-añadir su key en su próxima búsqueda es correcto).
        # Solo corre en el record-path (≈1×/ventana/user), no en cada skip.
        if len(_TEMPORAL_CLEANUP_LAST_RUN) > _TEMPORAL_CLEANUP_GC_THRESHOLD:
            _stale_cutoff = now - (2 * window_s)
            _stale_keys = [
                k for k, v in _TEMPORAL_CLEANUP_LAST_RUN.items()
                if v < _stale_cutoff
            ]
            for _k in _stale_keys:
                _TEMPORAL_CLEANUP_LAST_RUN.pop(_k, None)
        return False


# Errores transitorios típicos del pooler de Supabase / red. Reintentar con
# backoff corto. Otros errores (sintaxis SQL, permisos) NO se reintentan.
# [P3-RECALC-503-CLASSIFICATION · 2026-05-16] Añadido `couldn't get a connection`
# — el error específico de pool exhaustion del psycopg connection pool cuando
# se excede el cap pgBouncer del free tier. Aparece en logs con frecuencia
# durante visibilitychange/recalc bursts; clasificarlo como transient permite
# al caller escalar a 503 (cliente reintenta) en lugar de 500 (sugiere bug).
_TRANSIENT_DB_ERROR_FRAGMENTS = (
    "server disconnected",
    "connection reset",
    "connection refused",
    "connection closed",
    "consuming input failed",
    "ssl connection has been closed",
    "broken pipe",
    "timeout",
    "couldn't get a connection",  # psycopg_pool exhaustion (free tier cap)
    "remoteprotocolerror",  # httpx upstream supabase-py (idle conn died)
)


def _is_transient_db_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(frag in msg for frag in _TRANSIENT_DB_ERROR_FRAGMENTS)


def _with_db_retry(fn, *args, _label: str = "db_op", _attempts: int = 2, **kwargs):
    """Llama `fn(*args, **kwargs)` reintentando errores transitorios de DB.

    Crítico para queries que alimentan decisiones médicas (alergias, condiciones):
    devolver `[]` por un disconnect transitorio podría dejar pasar un alérgeno
    al plan generado. Reintento corto (2s + 4s) cubre la reconexión típica del
    pooler de Supabase sin bloquear el pipeline.
    """
    last_exc = None
    delays = (2.0, 4.0)
    for attempt in range(_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_db_error(exc) or attempt == _attempts:
                raise
            delay = delays[min(attempt, len(delays) - 1)]
            logger.warning(
                f"[DB-RETRY] {_label} falló por error transitorio ({type(exc).__name__}: {exc}); "
                f"reintentando en {delay:.1f}s ({attempt + 1}/{_attempts})."
            )
            _time.sleep(delay)
    raise last_exc

def _invalidate_rag_cache(user_id: str):
    """Invalida la caché RAG del usuario para evitar servir datos médicos obsoletos."""
    prefix = f"rag_{user_id}_"
    if redis_client:
        try:
            for key in redis_client.scan_iter(f"{prefix}*"):
                redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Redis cache invalidate error: {e}")
    try:
        execute_sql_write("DELETE FROM app_kv_store WHERE key LIKE %s", (prefix + '%',))
    except Exception as e:
        logger.warning(f"DB cache invalidate error: {e}")

def check_fact_ownership(fact_id: str, user_id: str) -> bool:
    """Verifica si un hecho corresponde a un determinado usuario."""
    if not connection_pool: return False
    try:
        query = "SELECT user_id FROM user_facts WHERE id = %s"
        res = execute_sql_query(query, (fact_id,), fetch_one=True)
        if res:
            return str(res.get("user_id")) == str(user_id)
    except Exception as e:
        logger.error(f"Error en check_fact_ownership: {e}")
    return False

def get_avg_meal_hour(user_id: str, meal_type: str, days_back: int = 14) -> Optional[float]:
    """Calcula la hora promedio en la que el usuario registra un tipo de comida (ej: 10.5 para 10:30 AM)."""
    if not connection_pool: return None
    try:
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        
        # Ajustamos a AST (-4) sumando/restando horas o simplemente extrayendo la hora local
        query = """
            SELECT EXTRACT(HOUR FROM (consumed_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santo_Domingo')) as hr,
                   EXTRACT(MINUTE FROM (consumed_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santo_Domingo')) as mn
            FROM consumed_meals
            WHERE user_id = %s AND meal_type ILIKE %s AND consumed_at >= %s
        """
        res = execute_sql_query(query, (user_id, f"%{meal_type}%", cutoff), fetch_all=True)
        if not res:
            return None
            
        total_hours = sum([float(r['hr']) + float(r['mn'])/60.0 for r in res])
        return round(total_hours / len(res), 2)
    except Exception as e:
        logger.error(f"Error calculando avg_meal_hour para {meal_type}: {e}")
        return None

def acquire_fact_lock(user_id: str) -> bool:
    """Intenta adquirir el bloqueo para extracción de hechos. Retorna True si lo logra, False si ya está bloqueado."""
    if not connection_pool: return True
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)

        # Verificar estado actual en user_profiles.
        # [P1-NEON-DB-MIGRATION · 2026-06-12] psycopg devuelve timestamptz como
        # datetime tz-aware — desaparece el parse manual del string ISO de PostgREST.
        res = execute_sql_query(
            "SELECT fact_locked_at FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        if res:
            locked_at = res.get("fact_locked_at")
            if locked_at and (now - locked_at) < timedelta(minutes=5):
                return False

        # Intentar establecer el timestamp (RETURNING preserva el contrato
        # "True solo si una fila fue actualizada" del `.data` de PostgREST).
        updated = execute_sql_write(
            "UPDATE user_profiles SET fact_locked_at = %s WHERE id = %s RETURNING id",
            (now, user_id),
            returning=True,
        )
        return bool(updated)
    except Exception as e:
        # [P3-PROD-AUDIT-3 · 2026-05-30] Fail-CLOSED (return False), no fail-open.
        # ANTES un error de DB hacía `return True` ("lock adquirido") → dos
        # invocaciones concurrentes durante un blip de DB podían proceder ambas →
        # facts duplicados + doble costo LLM. Fail-closed es estrictamente más
        # seguro SIN riesgo de pérdida: los callers tratan un skip como
        # recuperable (fact_extractor reintenta + enqueue_pending_fact persistente
        # drenado por cron/webhook).
        logger.error(f"Error acquiring fact lock: {e}")
        return False

def release_fact_lock(user_id: str):
    """Libera el bloqueo de extracción de hechos."""
    if not connection_pool: return
    try:
        execute_sql_write(
            "UPDATE user_profiles SET fact_locked_at = NULL WHERE id = %s",
            (user_id,),
        )
    except Exception as e:
        logger.error(f"Error releasing fact lock: {e}")

def save_user_fact(user_id: str, fact: str, embedding: list, metadata: dict = None):
    """Guarda un hecho, su embedding y sus metadatos en la base de datos."""
    if not connection_pool: return None
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] embedding list → literal pgvector
        # '[0.1,0.2,...]' con cast explícito ::vector. Metadata omitido cae al
        # default '{}'::jsonb de la columna (paridad con el INSERT PostgREST).
        emb_str = f"[{','.join(map(str, embedding))}]"
        res = execute_sql_write(
            "INSERT INTO user_facts (user_id, fact, embedding, metadata) "
            "VALUES (%s, %s, %s::extensions.vector, %s) RETURNING id",
            (user_id, fact, emb_str, Jsonb(metadata or {})),
            returning=True,
        )
        _invalidate_rag_cache(user_id)
        return res
    except Exception as e:
        logger.error(f"Error guardando user_fact: {e}")
        return None

def delete_expired_temporal_facts(user_id: str = None, hours: int = 48):
    """Elimina los hechos con categoría 'sintoma_temporal' que son más antiguos que 'hours'.

    [P2-PROD-AUDIT-BUNDLE · 2026-05-27] Debounce in-process per-user via
    `_should_skip_temporal_cleanup` — los 3 callsites side-effect en search
    pueden invocarse en milisegundos durante el mismo pipeline run; el debounce
    convierte los 3 DELETEs idénticos en 1. Cleanup global (`user_id=None`) NO
    debouncea — esos son crons explícitos.
    """
    if not connection_pool: return None
    if _should_skip_temporal_cleanup(user_id):
        return None
    from datetime import datetime, timedelta, timezone

    threshold_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    def _do_delete():
        # [P1-NEON-DB-MIGRATION · 2026-06-12] `.contains()` PostgREST → operador
        # JSONB `@>`; RETURNING preserva la forma list[dict] del `.data` previo.
        query = "DELETE FROM user_facts WHERE metadata @> %s AND created_at < %s"
        params = [Jsonb({"category": "sintoma_temporal"}), threshold_time]
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        return execute_sql_write(query + " RETURNING id", tuple(params), returning=True)

    try:
        return _with_db_retry(_do_delete, _label="delete_expired_temporal_facts")
    except Exception as e:
        logger.error(f"Error borrando temporal facts expirados (tras retries): {e}")
        return None

def get_user_facts_by_metadata(user_id: str, key: str, value: str):
    """Busca hechos exactos filtrando dentro del JSONB de metadata.

    CRÍTICO para alergias/condiciones — devolver `[]` por un disconnect transitorio
    podría dejar pasar un alérgeno al plan generado. Reintenta errores transitorios
    antes de degradar.

    Ejemplo: get_user_facts_by_metadata(user_id, 'category', 'alergia')
    """
    if not connection_pool: return []

    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)

    def _do_select():
        # Los consumidores (graph_orchestrator strict-RAG, cron_tasks alergias)
        # solo leen `fact`/`metadata`; SELECT * preserva la forma del `.data`.
        return execute_sql_query(
            "SELECT * FROM user_facts "
            "WHERE user_id = %s AND is_active = TRUE AND metadata @> %s",
            (user_id, Jsonb({key: value})),
            fetch_all=True,
        )

    try:
        return _with_db_retry(_do_select, _label=f"get_user_facts_by_metadata({key}={value})")
    except Exception as e:
        logger.error(f"Error buscando facts por metadata (tras retries): {e}")
        return []

def delete_user_facts_by_metadata(user_id: str, filter_dict: dict):
    """Soft delete filtrando dentro del JSONB de metadata."""
    if not connection_pool: return None
    try:
        res = execute_sql_write(
            "UPDATE user_facts SET is_active = FALSE "
            "WHERE user_id = %s AND metadata @> %s RETURNING id",
            (user_id, Jsonb(filter_dict)),
            returning=True,
        )
        _invalidate_rag_cache(user_id)
        return res
    except Exception as e:
        logger.error(f"Error haciendo soft delete a facts por metadata: {e}")
        return None

def search_user_facts(user_id: str, query_embedding: list, query_text: str = None, threshold: float = 0.5, limit: int = 5):
    """Busca hechos similares usando búsqueda híbrida (si hay texto) o vectorial pura en Supabase."""
    if not connection_pool: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        # Array Python a string pgvector '[1.2, 0.4, ...]'
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        # [P1-NEON-DB-MIGRATION] id::text — las funciones RETURNS TABLE(id uuid,...)
        # devolverían uuid.UUID nativo y el cache LLM (graph_orchestrator hace
        # json.dumps de estas rows) lanzaría TypeError. Paridad con PostgREST.
        if query_text:
            # Búsqueda híbrida (vector + full-text search)
            query = (
                "SELECT id::text AS id, fact, metadata, similarity "
                "FROM hybrid_search_user_facts(query_text => %s, query_embedding => %s, match_count => %s, p_user_id => %s)"
            )
            res = execute_sql_query(query, (query_text, emb_str, limit, user_id), fetch_all=True)
            return res
        else:
            # Búsqueda vectorial pura
            query = (
                "SELECT id::text AS id, fact, metadata, similarity "
                "FROM match_user_facts(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s)"
            )
            res = execute_sql_query(query, (emb_str, threshold, limit, user_id), fetch_all=True)
            return res
    except Exception as e:
        logger.error(f"Error buscando facts: {e}")
        return []

def search_user_facts_hybrid(user_id: str, query_embedding: list, filter_metadata: dict = None, threshold: float = 0.5, limit: int = 5):
    """Búsqueda Híbrida Vectorial: similitud de vectores pre-filtrada por metadatos (JSONB) usando pgvector RPC."""
    if not connection_pool: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        from psycopg.types.json import Jsonb
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        # [P1-NEON-DB-MIGRATION] casts ::text por la misma razón que
        # search_user_facts (esta función retorna la fila completa).
        query = (
            "SELECT id::text AS id, user_id::text AS user_id, fact, metadata, "
            "created_at::text AS created_at, is_active, similarity "
            "FROM match_user_facts_hybrid_metadata(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s, p_metadata => %s)"
        )
        meta_param = Jsonb(filter_metadata) if filter_metadata else None
        
        res = execute_sql_query(query, (emb_str, threshold, limit, user_id, meta_param), fetch_all=True)
        return res
    except Exception as e:
        logger.error(f"Error en búsqueda híbrida vectorial (metadatos): {e}")
        return []

def delete_user_fact(fact_id: str):
    """Hace un soft delete cambiando is_active a False"""
    if not connection_pool: return None
    try:
        # Extraer user_id antes de borrar para invalidar su caché
        # (::text — el prefijo `rag_<user_id>_` de la caché espera string)
        res_user = execute_sql_query(
            "SELECT user_id::text AS user_id FROM user_facts WHERE id = %s",
            (fact_id,),
            fetch_one=True,
        )
        user_id = res_user["user_id"] if res_user else None

        # En lugar de DELETE, soft delete via UPDATE
        res = execute_sql_write(
            "UPDATE user_facts SET is_active = FALSE WHERE id = %s RETURNING id",
            (fact_id,),
            returning=True,
        )

        if user_id:
            _invalidate_rag_cache(user_id)

        return res
    except Exception as e:
        logger.error(f"Error haciendo soft delete a user_fact: {e}")
        return None

def enqueue_pending_fact(user_id: str, message: str, recent_history: str = ""):
    """Encola un mensaje pendiente de extracción de hechos en la DB."""
    if not connection_pool: return None
    try:
        res = execute_sql_write(
            "INSERT INTO pending_facts_queue (user_id, message, recent_history) "
            "VALUES (%s, %s, %s) RETURNING id",
            (user_id, message, recent_history or ""),
            returning=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error encolando hecho pendiente: {e}")
        return None

def dequeue_pending_facts(user_id: str):
    """Obtiene todos los hechos pendientes de un usuario, ordenados cronológicamente."""
    if not connection_pool: return []
    try:
        # id/user_id ::text — el caller (process_pending_queue_sync) acumula
        # `pending["id"]` y los pasa a delete_pending_facts como strings.
        res = execute_sql_query(
            "SELECT id::text AS id, user_id::text AS user_id, message, "
            "recent_history, created_at "
            "FROM pending_facts_queue WHERE user_id = %s ORDER BY created_at ASC",
            (user_id,),
            fetch_all=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error obteniendo hechos pendientes: {e}")
        return []

def delete_pending_facts(fact_ids: list):
    """Elimina los registros procesados de la cola de pendientes."""
    if not connection_pool or not fact_ids: return None
    try:
        # `.in_()` → `= ANY(%s::uuid[])`; normaliza a str por si llegan uuid.UUID.
        ids = [str(fid) for fid in fact_ids]
        res = execute_sql_write(
            "DELETE FROM pending_facts_queue WHERE id = ANY(%s::uuid[]) RETURNING id",
            (ids,),
            returning=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error eliminando hechos pendientes procesados: {e}")
        return None

def get_all_user_facts(user_id: str):
    """Obtiene todos los hechos (facts) de un usuario para mostrarlos en la UI de Ajustes."""
    if not connection_pool: return []
    try:
        # id::text — el frontend usa el id como string para DELETE
        # /api/user-facts/{id}. `created_at` queda datetime: FastAPI lo
        # serializa a ISO-8601 con 'T', mismo wire-format que daba PostgREST.
        res = execute_sql_query(
            "SELECT id::text AS id, fact, metadata, created_at "
            "FROM user_facts WHERE user_id = %s AND is_active = TRUE "
            "ORDER BY created_at DESC",
            (user_id,),
            fetch_all=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error obteniendo ALL user facts: {e}")
        return []

def save_visual_entry(user_id: str, image_url: str, description: str, embedding: list):
    """
    Guarda una entrada del diario visual con deduplicación semántica.
    Antes de insertar, busca si ya existe un vector con >0.95 de similitud.
    Si existe, actualiza frequency += 1 y last_seen = NOW() en lugar de crear un duplicado.
    Esto previene que el RAG visual se sature con 15 fotos idénticas de mangú.
    """
    if not connection_pool: return None
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] RPC PostgREST → llamada SQL directa
        # a la función; el embedding viaja como literal pgvector con cast ::vector.
        emb_str = f"[{','.join(map(str, embedding))}]"

        # === DEDUPLICACIÓN SEMÁNTICA: Buscar duplicados cercanos ===
        try:
            similar = execute_sql_query(
                "SELECT * FROM match_visual_diary(query_embedding => %s::extensions.vector, "
                "match_threshold => %s, match_count => %s, p_user_id => %s)",
                (emb_str, 0.95, 1, user_id),
                fetch_all=True,
            )

            if similar:
                existing = similar[0]
                existing_id = existing.get("id")
                old_freq = existing.get("frequency", 1) or 1
                logger.info(f"🔄 [DEDUP VISUAL] Entrada similar detectada (sim>{0.95}). "
                      f"Actualizando frequency {old_freq} → {old_freq + 1} en lugar de insertar duplicado.")
                
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                
                # [P3-PROD-AUDIT-2 · 2026-05-30] Incremento ATÓMICO de frequency
                # (`frequency = frequency + 1` server-side) en vez del RMW
                # `old_freq + 1`. Dos fotos casi-simultáneas del mismo plato (vía
                # background tasks no sincronizados) ambas leían `frequency=N` y
                # escribían `N+1` → un incremento perdido. `old_freq` sigue usado
                # solo para el log informativo de arriba.
                execute_sql_write(
                    "UPDATE visual_diary SET frequency = frequency + 1, "
                    "last_seen = %s, image_url = %s, description = %s WHERE id = %s",
                    (now, image_url, description, existing_id),
                )
                
                logger.debug(f"✅ [DEDUP VISUAL] Registro {str(existing_id)[:8]}... actualizado.")
                return similar
        except Exception as dedup_err:
            # Si la deduplicación falla (ej. columnas aún no existen), insertar normalmente
            logger.error(f"⚠️ [DEDUP VISUAL] Error en deduplicación, insertando normalmente: {dedup_err}")

        # === INSERCIÓN NORMAL (no hay duplicado) ===
        res = execute_sql_write(
            "INSERT INTO visual_diary (user_id, image_url, description, embedding, frequency) "
            "VALUES (%s, %s, %s, %s::extensions.vector, 1) RETURNING id",
            (user_id, image_url, description, emb_str),
            returning=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error guardando visual_entry: {e}")
        return None

def search_visual_diary(user_id: str, query_embedding: list, threshold: float = 0.5, limit: int = 5):
    """Busca fotos/entradas visuales similares usando la función SQL match_visual_diary."""
    if not connection_pool: return []

    def _do_search():
        # id::text — los consumidores solo leen `description`, pero la fila
        # puede acabar en JSON; string preserva la paridad con PostgREST.
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        return execute_sql_query(
            "SELECT id::text AS id, description, image_url, similarity "
            "FROM match_visual_diary(query_embedding => %s::extensions.vector, "
            "match_threshold => %s, match_count => %s, p_user_id => %s)",
            (emb_str, threshold, limit, user_id),
            fetch_all=True,
        )

    try:
        return _with_db_retry(_do_search, _label="search_visual_diary")
    except Exception as e:
        logger.error(f"Error buscando visual_diary (tras retries): {e}")
        return []

def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0, ingredients: list = None, meal_type: str = "snack", mark_inventory_synced: bool = False):
    """Guarda una comida consumida en la tabla consumed_meals.

    [P0.1] Si el caller acaba de descontar los ingredientes del inventario,
    debe pasar `mark_inventory_synced=True` para que la reconciliación al
    cierre del chunk no vuelva a descontarlos.
    """
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        synced_at = now if mark_inventory_synced else None

        # [P2-CONSUMED-DEDUP · 2026-05-28] Dedup anti doble-tap. log_consumed_meal
        # se invoca desde la tool del chat-agent (re-emisión posible del LLM) y
        # desde POST /api/diary (doble-tap móvil / retry de red del SSE). El INSERT
        # plano duplica el conteo calórico del día. Dedup por VENTANA CORTA: si ya
        # hay una fila idéntica (user_id, meal_name, meal_type) en los últimos N s,
        # es un doble-submit → skip idempotente. NO bloquea repeticiones legítimas
        # (un 2º café horas después cae fuera de la ventana). Best-effort: si el
        # check falla, se procede con el INSERT (no perder un registro legítimo).
        # Knob MEALFIT_CONSUMED_MEAL_DEDUP_WINDOW_S (default 60, 0=off).
        try:
            from knobs import _env_int as _dedup_env_int
            _dedup_win = _dedup_env_int("MEALFIT_CONSUMED_MEAL_DEDUP_WINDOW_S", 60)
        except Exception:
            _dedup_win = 60
        if _dedup_win > 0:
            try:
                _dup = execute_sql_query(
                    "SELECT id FROM consumed_meals WHERE user_id = %s AND meal_name = %s "
                    "AND meal_type = %s AND consumed_at > NOW() - make_interval(secs => %s) LIMIT 1",
                    (user_id, meal_name, meal_type, _dedup_win),
                    fetch_one=True,
                )
                if _dup:
                    logger.info(
                        f"[P2-CONSUMED-DEDUP] skip doble-tap: '{meal_name}' ({meal_type}) "
                        f"ya registrado en los últimos {_dedup_win}s."
                    )
                    # [P2-CONSUMED-DEDUP-INVENTORY · 2026-05-30] Retornar un
                    # sentinel DISTINGUIBLE (no `True`) para que el caller
                    # (tools.log_consumed_meal) NO vuelva a deducir el inventario.
                    # Pre-fix el dedup retornaba True == path de éxito real → el
                    # caller no podía distinguir dedup-skip de INSERT, así que
                    # deduct_consumed_meal_from_inventory corría de nuevo en la
                    # re-emisión → la nevera se descontaba AL DOBLE del consumo
                    # real (el diario calórico quedaba correcto vía dedup, pero
                    # el inventario no). "deduped" es truthy y no-None, así que
                    # los callers que chequean `is not None` / `if not result`
                    # (tools.py:494, diary.py:478) lo siguen tratando como éxito.
                    return "deduped"
            except Exception as _dedup_err:
                logger.warning(
                    f"[P2-CONSUMED-DEDUP] check falló (procediendo con INSERT): {_dedup_err}"
                )

        # [P1-NEON-DB-MIGRATION · 2026-06-12] Eliminado el fallback PostgREST:
        # los datos viven en Neon; sin pool no hay path válido de escritura.
        if not connection_pool:
            return None
        # [P1-CONSUMED-MEALS-JSONB · 2026-05-20] `consumed_meals.ingredients`
        # es jsonb (verified via information_schema). psycopg3 type
        # adaption convierte `list[str]` a Postgres ARRAY literal
        # `{a,b,c}` por default, NO a JSON. Eso hace que el INSERT
        # falle con `invalid input syntax for type json` ("Expected
        # ':', but found ','" — Postgres trata de parsear `{...}`
        # como JSON object). Pre-fix: el `Jsonb` se importaba pero
        # NO se aplicaba al parámetro → bug silencioso desde la
        # migración a connection_pool. `consumed_meals` quedó vacía
        # en prod (0 rows) — el diario de comidas del agente
        # nunca persistió nada. Tooltip-anchor: P1-CONSUMED-MEALS-JSONB.
        execute_sql_write(
            "INSERT INTO consumed_meals (user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients, consumed_at, meal_type, inventory_synced_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, meal_name, calories, protein, carbs, healthy_fats, Jsonb(ingredients if ingredients is not None else []), now, meal_type, synced_at)
        )
        return True
    except Exception as e:
        logger.error(f"Error guardando comida consumida: {e}")
        return None

def get_consumed_meals_today(user_id: str, date_str: str = None, tz_offset_mins: int = None):
    """Obtiene las comidas consumidas del día especificado en base a la zona horaria del usuario."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            from datetime import datetime, timezone, timedelta
            
            if date_str and tz_offset_mins is not None:
                # El frontend envía tz_offset en minutos (JS: getTimezoneOffset() - diferencia a UTC)
                # Parsear el inicio del día local
                try:
                    local_start = datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S")
                    local_end = datetime.strptime(f"{date_str} 23:59:59", "%Y-%m-%d %H:%M:%S")
                    
                    # Convertir local a UTC sumando tz_offset_mins (para UTC-4 el offset es 240)
                    utc_start = local_start + timedelta(minutes=tz_offset_mins)
                    utc_end = local_end + timedelta(minutes=tz_offset_mins)
                    
                    start_str = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                    end_str = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception as e:
                    logger.error(f"⚠️ Error procesando la zona horaria, usando fallback a UTC: {e}")
                    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    start_str = f"{today_date}T00:00:00Z"
                    end_str = f"{today_date}T23:59:59Z"
            else:
                today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                start_str = f"{today_date}T00:00:00Z"
                end_str = f"{today_date}T23:59:59Z"
            
            # [P2-SELECT-STAR-CONSUMED-MEALS · 2026-05-15] Columnas explícitas
            # (era `SELECT *`). Caller (`/api/diary/consumed/{user_id}` y
            # `agent.py:904`) solo necesita meal_name, calories, protein,
            # carbs, healthy_fats, consumed_at, meal_type. Trim de I/O +
            # JSON parse + payload size; cierra convención post-P1-NEW-3
            # (explicit columns).
            _COLUMNS = (
                "meal_name, calories, protein, carbs, healthy_fats, "
                "consumed_at, meal_type"
            )
            # [P1-NEON-DB-MIGRATION · 2026-06-12] Eliminado el fallback PostgREST.
            if not connection_pool:
                return []
            query = (
                f"SELECT {_COLUMNS} FROM consumed_meals "
                "WHERE user_id = %s AND consumed_at >= %s AND consumed_at < %s"
            )
            res = execute_sql_query(query, (user_id, start_str, end_str), fetch_all=True)
            return res
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            logger.error(f"Error obteniendo comidas consumidas de hoy: {e}")
            return []

def get_consumed_meals_since(user_id: str, since_iso_date: str, include_ingredients: bool = False):
    """Obtiene todas las comidas consumidas por el usuario desde una fecha específica.

    [GAP-1 · 2026-05-29] `include_ingredients=True` añade la columna jsonb
    `ingredients` para callers que hacen análisis ingredient-level
    (`cron_tasks.calculate_ingredient_fatigue`). El default permanece lean per
    P2-SELECT-STAR-CONSUMED-MEALS. **Key-drift histórico**: esta función SIEMPRE
    proyectó `consumed_at` (no `created_at`) y nunca `ingredients`;
    `calculate_ingredient_fatigue`/`calculate_day_of_week_adherence` leían
    `created_at`/`ingredients` → resultado SIEMPRE vacío en prod (la fatiga de
    ingredientes y la EMA day-of-week estaban muertas). Tests usaban fixtures con
    las claves erróneas, enmascarando el bug. Anclado por
    `test_gap_1_consumed_meals_fetcher_contract.py`.
    """
    # [P2-SELECT-STAR-CONSUMED-MEALS · 2026-05-15] Columnas explícitas — caller
    # (`agent.py::log_consumed_meal` y stats agregadas) solo necesita los
    # campos básicos. Reduce I/O sobre tabla append-only que puede acumular
    # miles de filas por usuario.
    _COLUMNS = (
        "meal_name, calories, protein, carbs, healthy_fats, "
        "consumed_at, meal_type"
    )
    if include_ingredients:
        _COLUMNS = _COLUMNS + ", ingredients"
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Eliminado el fallback PostgREST.
        if not connection_pool:
            return []
        query = (
            f"SELECT {_COLUMNS} FROM consumed_meals "
            "WHERE user_id = %s AND consumed_at >= %s"
        )
        res = execute_sql_query(query, (user_id, since_iso_date), fetch_all=True)
        return res
    except Exception as e:
        logger.error(f"Error obteniendo comidas consumidas desde {since_iso_date}: {e}")
        return []
