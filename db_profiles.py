from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write

def upsert_user_profile(user_id: str, health_profile: dict) -> bool:
    """Hace upsert del perfil de usuario y health_profile en user_profiles."""
    if not supabase: return False
    try:
        supabase.table("user_profiles").upsert({
            "id": user_id,
            "health_profile": health_profile
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error en upsert_user_profile: {e}")
        return False

def get_user_profile(user_id: str):
    """Obtiene el perfil completo del usuario, incluyendo el health_profile."""
    if not supabase: return None
    try:
        from datetime import datetime, timezone
        res = supabase.table("user_profiles").select("*").eq("id", user_id).execute()
        if not res.data:
            return None
            
        profile = res.data[0]
        
        # --- Graceful Degradation Middleware ---
        if profile.get("subscription_status") == "CANCELLED" and profile.get("subscription_end_date"):
            end_date_str = profile.get("subscription_end_date")
            try:
                from constants import safe_fromisoformat
                end_date = safe_fromisoformat(end_date_str)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                
                # Si ya cruzamos la hora exacta de terminación, lo degradamos
                if now_utc > end_date:
                    logger.info(f"⬇️ Degradando perfil de {user_id} a 'gratis'. El tiempo de su cancelación (Graceful) terminó.")
                    supabase.table("user_profiles").update({
                        "plan_tier": "gratis",
                        "subscription_status": "INACTIVE"
                    }).eq("id", user_id).execute()
                    
                    profile["plan_tier"] = "gratis"
                    profile["subscription_status"] = "INACTIVE"
            except Exception as d_e:
                logger.error(f"Error parseando fechas en graceful degradation para {user_id}: {d_e}")
        # ---------------------------------------
        
        return profile
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {e}")
        return None

def _invalidate_stale_chunks(user_id: str, reason: str):
    """Marca chunks pendientes como 'stale' para que el worker los re-genere con datos frescos."""
    from db_core import execute_sql_write
    result = execute_sql_write("""
        UPDATE plan_chunk_queue 
        SET status = 'stale', 
            updated_at = NOW()
        WHERE user_id = %s 
        AND status = 'pending'
        RETURNING id, week_number
    """, (user_id,), returning=True)
    
    if result:
        logger.info(f"♻️ [CHUNK INVALIDATION] {len(result)} chunks marcados como 'stale' para {user_id} (razón: {reason})")
        return True
    return False

def update_user_health_profile(user_id: str, health_profile: dict):
    """Sobreescribe el JSONB de health_profile en la base de datos."""
    if not supabase: return None
    try:
        # --- GAP 4: Chunk Invalidation Detector (Conservative) ---
        try:
            old_profile_data = get_user_profile(user_id)
            if old_profile_data and old_profile_data.get('health_profile'):
                old_hp = old_profile_data['health_profile']
                
                invalidation_reasons = []
                if old_hp.get('goal') != health_profile.get('goal'):
                    invalidation_reasons.append("goal_changed")
                if old_hp.get('budget') != health_profile.get('budget'):
                    invalidation_reasons.append("budget_changed")
                if set(old_hp.get('allergies', [])) != set(health_profile.get('allergies', [])):
                    invalidation_reasons.append("allergies_changed")
                    
                old_w = float(old_hp.get('weight', 0) or 0)
                new_w = float(health_profile.get('weight', 0) or 0)
                if old_w and new_w and abs(old_w - new_w) >= 5:
                    invalidation_reasons.append("significant_weight_change")
                    
                if invalidation_reasons:
                    _invalidate_stale_chunks(user_id, ", ".join(invalidation_reasons))
        except Exception as check_e:
            logger.warning(f"⚠️ [CHUNK INVALIDATION] Error checking for critical profile changes: {check_e}")
        # ---------------------------------------------------------
            
        res = supabase.table("user_profiles").update({
            "health_profile": health_profile
        }).eq("id", user_id).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error actualizando health_profile: {e}")
        return None

def log_api_usage(user_id: str, endpoint: str = "gemini"):
    """Guarda un registro de uso de la API (consume 1 crédito)."""
    if not user_id or user_id == "guest": return None
    try:
        from db_core import connection_pool
        if connection_pool:
            res = execute_sql_write("INSERT INTO api_usage (user_id, endpoint) VALUES (%s, %s)", (user_id, endpoint))
            return res
        else:
            if not supabase: return None
            res = supabase.table("api_usage").insert({
                "user_id": user_id,
                "endpoint": endpoint
            }).execute()
            return res.data
    except Exception as e:
        logger.error(f"Error registrando api_usage: {e}")
        return None

def get_monthly_api_usage(user_id: str) -> int:
    """Cuenta cuántas llamadas a la API ha hecho el usuario este mes."""
    if not user_id or user_id == "guest": return 0
    from datetime import datetime
    
    try:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).isoformat()
        
        from db_core import connection_pool
        if connection_pool:
            res = execute_sql_query("SELECT count(*) as total FROM api_usage WHERE user_id = %s AND created_at >= %s", (user_id, start_date), fetch_one=True)
            if res and 'total' in res:
                return int(res['total'])
            return 0
        else:
            if not supabase: return 0
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    res = supabase.table("api_usage").select("*", count="exact").eq("user_id", user_id).gte("created_at", start_date).execute()
                    return res.count if hasattr(res, 'count') and res.count is not None else 0
                except Exception as inner_e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5)
                        continue
                    logger.error(f"Error supabase get_monthly_api_usage: {inner_e}")
                    return 0
    except Exception as e:
        logger.error(f"Error obteniendo api_usage mensual: {e}")
        return 0

def migrate_guest_data(session_ids: list, new_user_id: str):
    """
    Migra todos los datos asociados a uno o varios session_ids temporales
    (creados cuando el usuario era invitado) hacia el nuevo UUID del usuario registrado.
    """
    if not supabase or not session_ids or not new_user_id: 
        return False
    
    try:
        # 1. Actualizar agent_sessions para vincular historiales de chat
        supabase.table("agent_sessions").update({"user_id": new_user_id}).in_("id", session_ids).execute()
        
        # 2. Actualizar visual_diary (Vectores/Diario Visual)
        supabase.table("visual_diary").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 3. Actualizar user_facts (Vectores/Memoria a largo plazo)
        supabase.table("user_facts").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 4. Actualizar meal_plans (Planes guardados como guest, si los hubiera)
        supabase.table("meal_plans").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 5. Actualizar consumed_meals (Comidas registradas)
        supabase.table("consumed_meals").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 6. Actualizar pending_facts_queue
        supabase.table("pending_facts_queue").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 7. Actualizar meal_rejections (Ojo: estas usan session_id o user_id)
        # Primero intentamos por session_id (como se guardan usualmente los rechazos guest)
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("session_id", session_ids).execute()
        # Y también por user_id por si acaso
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 8. Actualizar meal_likes
        supabase.table("meal_likes").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 9. Actualizar Inventario Físico (Despensa)
        supabase.table("user_inventory").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        # 10. Recalcular frecuencias de ingredientes a partir de los planes migrados
        # Sin esto, el usuario registrado parte con freq=0 y pierde el historial de variedad.
        try:
            from constants import normalize_ingredient_for_tracking
            from db_plans import get_ingredient_frequencies_from_plans, increment_ingredient_frequencies
            migrated_ings = get_ingredient_frequencies_from_plans(new_user_id, limit=10)
            if migrated_ings:
                normalized = [normalize_ingredient_for_tracking(i) for i in migrated_ings if i]
                normalized = [n for n in normalized if n]  # Filtrar vacíos
                if normalized:
                    increment_ingredient_frequencies(new_user_id, normalized)
                    logger.info(f"✅ [MIGRACIÓN] Frecuencias recalculadas para {new_user_id} ({len(normalized)} ingredientes)")
        except Exception as freq_e:
            # No bloquear la migración si falla el recálculo de frecuencias
            logger.error(f"⚠️ [MIGRACIÓN] Error recalculando frecuencias (no crítico): {freq_e}")
        
        logger.info(f"✅ Migración exitosa de {session_ids} a UUID {new_user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error migrando datos de invitado: {e}")
        return False

def reset_user_account_preferences(user_id: str) -> bool:
    """Borra preferencias, rechazos, inventario, planes históricos y limpia el health_profile para un verdadero inicio desde cero."""
    if not supabase or not user_id or user_id == "guest": 
        return False
        
    try:
        from db_core import execute_sql_write
        # 1. Borrar likes
        execute_sql_write("DELETE FROM meal_likes WHERE user_id = %s", (user_id,))
        # 2. Borrar rejections
        execute_sql_write("DELETE FROM meal_rejections WHERE user_id = %s", (user_id,))
        # 3. Borrar inventario
        execute_sql_write("DELETE FROM user_inventory WHERE user_id = %s", (user_id,))
        # 4. Borrar knowledge graph/facts (preferencias aprendidas)
        execute_sql_write("DELETE FROM user_facts WHERE user_id = %s", (user_id,))
        # 5. Borrar frecuencias de ingredientes (para que no rote basado en el pasado)
        execute_sql_write("DELETE FROM ingredient_frequencies WHERE user_id = %s", (user_id,))
        # 6. Borrar planes históricos (esto borra en cascada la plan_chunk_queue)
        execute_sql_write("DELETE FROM meal_plans WHERE user_id = %s", (user_id,))
        # 7. Limpiar health_profile en user_profiles (dejándolo vacío)
        execute_sql_write("UPDATE user_profiles SET health_profile = '{}'::jsonb WHERE id = %s", (user_id,))
        
        logger.info(f"♻️ Preferencias y planes reseteados DESDE CERO con éxito para UUID {user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error reseteando preferencias para {user_id}: {e}")
        return False
