from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write
from cache_manager import redis_client

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
    if not supabase: return True
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        
        # Verificar estado actual en user_profiles
        res = supabase.table("user_profiles").select("fact_locked_at").eq("id", user_id).execute()
        if res.data and len(res.data) > 0:
            locked_at_str = res.data[0].get("fact_locked_at")
            if locked_at_str:
                try:
                    if locked_at_str.endswith("Z"):
                        locked_at_str = locked_at_str[:-1] + "+00:00"
                    locked_at = datetime.fromisoformat(locked_at_str)
                    if now - locked_at < timedelta(minutes=5):
                        return False
                except ValueError:
                    pass
                
        # Intentar establecer el timestamp
        update_res = supabase.table("user_profiles").update({"fact_locked_at": now.isoformat()}).eq("id", user_id).execute()
            
        return len(update_res.data) > 0
    except Exception as e:
        # Si falla (ej. la columna no existe), imprimimos el error pero no bloqueamos
        error_msg = str(e)
        if "PGRST204" not in error_msg:
            logger.error(f"Error acquiring fact lock: {e}")
        return True

def release_fact_lock(user_id: str):
    """Libera el bloqueo de extracción de hechos."""
    if not supabase: return
    try:
        supabase.table("user_profiles").update({"fact_locked_at": None}).eq("id", user_id).execute()
    except Exception as e:
        error_msg = str(e)
        if "PGRST204" not in error_msg:
            logger.error(f"Error releasing fact lock: {e}")

def save_user_fact(user_id: str, fact: str, embedding: list, metadata: dict = None):
    """Guarda un hecho, su embedding y sus metadatos en la base de datos."""
    if not supabase: return None
    try:
        data_to_insert = {
            "user_id": user_id,
            "fact": fact,
            "embedding": embedding
        }
        if metadata:
            data_to_insert["metadata"] = metadata
            
        res = supabase.table("user_facts").insert(data_to_insert).execute()
        _invalidate_rag_cache(user_id)
        return res.data
    except Exception as e:
        logger.error(f"Error guardando user_fact: {e}")
        return None

def delete_expired_temporal_facts(user_id: str = None, hours: int = 48):
    """Elimina los hechos con categoría 'sintoma_temporal' que son más antiguos que 'hours'."""
    if not supabase: return None
    from datetime import datetime, timedelta, timezone
    
    threshold_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    
    try:
        query = supabase.table("user_facts").delete().contains("metadata", {"category": "sintoma_temporal"}).lt("created_at", threshold_time)
        if user_id:
            query = query.eq("user_id", user_id)
        
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando temporal facts expirados: {e}")
        return None

def get_user_facts_by_metadata(user_id: str, key: str, value: str):
    """Busca hechos Exactos filtrando dentro del JSONB de metadata.
    Ejemplo: get_user_facts_by_metadata(user_id, 'category', 'alergia')
    """
    if not supabase: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        # Supabase Python client filter for JSONB: metadata->>key = value
        # we can use eq() if we query a specific path, but simpler is using contains
        filter_dict = {key: value}
        # Añadimos el filtro is_active
        res = supabase.table("user_facts").select("*").eq("user_id", user_id).eq("is_active", True).contains("metadata", filter_dict).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error buscando facts por metadata: {e}")
        return []

def delete_user_facts_by_metadata(user_id: str, filter_dict: dict):
    """Soft delete filtrando dentro del JSONB de metadata."""
    if not supabase: return None
    try:
        res = supabase.table("user_facts").update({"is_active": False}).eq("user_id", user_id).contains("metadata", filter_dict).execute()
        _invalidate_rag_cache(user_id)
        return res.data
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
        
        if query_text:
            # Búsqueda híbrida (vector + full-text search)
            query = "SELECT * FROM hybrid_search_user_facts(query_text => %s, query_embedding => %s, match_count => %s, p_user_id => %s)"
            res = execute_sql_query(query, (query_text, emb_str, limit, user_id), fetch_all=True)
            return res
        else:
            # Búsqueda vectorial pura
            query = "SELECT * FROM match_user_facts(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s)"
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
        
        query = "SELECT * FROM match_user_facts_hybrid_metadata(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s, p_metadata => %s)"
        meta_param = Jsonb(filter_metadata) if filter_metadata else None
        
        res = execute_sql_query(query, (emb_str, threshold, limit, user_id, meta_param), fetch_all=True)
        return res
    except Exception as e:
        logger.error(f"Error en búsqueda híbrida vectorial (metadatos): {e}")
        return []

def delete_user_fact(fact_id: str):
    """Hace un soft delete cambiando is_active a False"""
    if not supabase: return None
    try:
        # Extraer user_id antes de borrar para invalidar su caché
        res_user = supabase.table("user_facts").select("user_id").eq("id", fact_id).execute()
        user_id = res_user.data[0]["user_id"] if res_user.data else None

        # En lugar de .delete(), usamos .update()
        res = supabase.table("user_facts").update({"is_active": False}).eq("id", fact_id).execute()
        
        if user_id:
            _invalidate_rag_cache(user_id)
            
        return res.data
    except Exception as e:
        logger.error(f"Error haciendo soft delete a user_fact: {e}")
        return None

def enqueue_pending_fact(user_id: str, message: str, recent_history: str = ""):
    """Encola un mensaje pendiente de extracción de hechos en Supabase."""
    if not supabase: return None
    try:
        res = supabase.table("pending_facts_queue").insert({
            "user_id": user_id,
            "message": message,
            "recent_history": recent_history or ""
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error encolando hecho pendiente: {e}")
        return None

def dequeue_pending_facts(user_id: str):
    """Obtiene todos los hechos pendientes de un usuario, ordenados cronológicamente."""
    if not supabase: return []
    try:
        res = supabase.table("pending_facts_queue").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error obteniendo hechos pendientes: {e}")
        return []

def delete_pending_facts(fact_ids: list):
    """Elimina los registros procesados de la cola de pendientes."""
    if not supabase or not fact_ids: return None
    try:
        res = supabase.table("pending_facts_queue").delete().in_("id", fact_ids).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error eliminando hechos pendientes procesados: {e}")
        return None

def get_all_user_facts(user_id: str):
    """Obtiene todos los hechos (facts) de un usuario para mostrarlos en la UI de Ajustes."""
    if not supabase: return []
    try:
        # Añadimos el filtro is_active
        res = supabase.table("user_facts").select("id, fact, metadata, created_at").eq("user_id", user_id).eq("is_active", True).order("created_at", desc=True).execute()
        return res.data
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
    if not supabase: return None
    try:
        # === DEDUPLICACIÓN SEMÁNTICA: Buscar duplicados cercanos ===
        try:
            similar = supabase.rpc("match_visual_diary", {
                "query_embedding": embedding,
                "match_threshold": 0.95,
                "match_count": 1,
                "p_user_id": user_id
            }).execute()
            
            if similar.data and len(similar.data) > 0:
                existing = similar.data[0]
                existing_id = existing.get("id")
                old_freq = existing.get("frequency", 1) or 1
                print(f"🔄 [DEDUP VISUAL] Entrada similar detectada (sim>{0.95}). "
                      f"Actualizando frequency {old_freq} → {old_freq + 1} en lugar de insertar duplicado.")
                
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                
                supabase.table("visual_diary").update({
                    "frequency": old_freq + 1,
                    "last_seen": now,
                    "image_url": image_url,      # Actualizar con la foto más reciente
                    "description": description   # Actualizar con la descripción más reciente
                }).eq("id", existing_id).execute()
                
                logger.debug(f"✅ [DEDUP VISUAL] Registro {str(existing_id)[:8]}... actualizado.")
                return similar.data
        except Exception as dedup_err:
            # Si la deduplicación falla (ej. columnas aún no existen), insertar normalmente
            logger.error(f"⚠️ [DEDUP VISUAL] Error en deduplicación, insertando normalmente: {dedup_err}")
        
        # === INSERCIÓN NORMAL (no hay duplicado) ===
        res = supabase.table("visual_diary").insert({
            "user_id": user_id,
            "image_url": image_url,
            "description": description,
            "embedding": embedding,
            "frequency": 1
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error guardando visual_entry: {e}")
        return None

def search_visual_diary(user_id: str, query_embedding: list, threshold: float = 0.5, limit: int = 5):
    """Busca fotos/entradas visuales similares usando la función RPC match_visual_diary."""
    if not supabase: return []
    try:
        res = supabase.rpc("match_visual_diary", {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": limit,
            "p_user_id": user_id
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error buscando visual_diary: {e}")
        return []

def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0, ingredients: list = None, meal_type: str = "snack"):
    """Guarda una comida consumida en la tabla consumed_meals de Supabase."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        if connection_pool:
            from psycopg.types.json import Jsonb
            execute_sql_write(
                "INSERT INTO consumed_meals (user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients, consumed_at, meal_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients if ingredients is not None else [], now, meal_type)
            )
            return True
        else:
            if not supabase: return None
            res = supabase.table("consumed_meals").insert({
                "user_id": user_id,
                "meal_name": meal_name,
                "meal_type": meal_type,
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "healthy_fats": healthy_fats,
                "ingredients": ingredients if ingredients is not None else [],
                "consumed_at": now
            }).execute()
            return res.data
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
            
            if connection_pool:
                query = "SELECT * FROM consumed_meals WHERE user_id = %s AND consumed_at >= %s AND consumed_at < %s"
                res = execute_sql_query(query, (user_id, start_str, end_str), fetch_all=True)
                return res
            else:
                if not supabase: return []
                res = supabase.table("consumed_meals")\
                    .select("*")\
                    .eq("user_id", user_id)\
                    .gte("consumed_at", start_str)\
                    .lt("consumed_at", end_str)\
                    .execute()
                return res.data
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            logger.error(f"Error obteniendo comidas consumidas de hoy: {e}")
            return []

def get_consumed_meals_since(user_id: str, since_iso_date: str):
    """Obtiene todas las comidas consumidas por el usuario desde una fecha específica."""
    try:
        if connection_pool:
            query = "SELECT * FROM consumed_meals WHERE user_id = %s AND consumed_at >= %s"
            res = execute_sql_query(query, (user_id, since_iso_date), fetch_all=True)
            return res
        else:
            if not supabase: return []
            res = supabase.table("consumed_meals")\
                .select("*")\
                .eq("user_id", user_id)\
                .gte("consumed_at", since_iso_date)\
                .execute()
            return res.data
    except Exception as e:
        logger.error(f"Error obteniendo comidas consumidas desde {since_iso_date}: {e}")
        return []
