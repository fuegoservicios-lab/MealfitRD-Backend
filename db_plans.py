from functools import lru_cache as _lru_cache
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write
from constants import strip_accents, GLOBAL_REVERSE_MAP
from db_chat import insert_rejection, save_message
from db_profiles import get_user_profile, update_user_health_profile

def check_recent_meal_plan_exists(user_id: str, max_seconds: int = 30) -> bool:
    """Verifica si ya se ha guardado un plan para este usuario recientemente."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                from constants import safe_fromisoformat
                last_saved = safe_fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
                
            now_utc = datetime.now(timezone.utc)
            if (now_utc - last_saved).total_seconds() < max_seconds:
                return True
    except Exception as e:
        logger.warning(f"⚠️ [DEDUP] Error en check_recent_meal_plan_exists: {e}")
    return False

def check_meal_plan_generated_today(user_id: str) -> bool:
    """Valida si el último plan generado por el usuario se realizó el día actual."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                from constants import safe_fromisoformat
                last_saved = safe_fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
            
            now_utc = datetime.now(timezone.utc)
            if last_saved.date() == now_utc.date():
                return True
    except Exception as e:
        logger.error(f"Error comprobando si plan fue hoy: {e}")
    return False

def save_new_meal_plan_robust(insert_data: dict) -> bool:
    """Guarda un nuevo plan nutricional con fallback por si faltan columnas optimizadas."""
    if not connection_pool: return False
    try:
        import copy
        from psycopg.types.json import Jsonb
        safe_data = copy.deepcopy(insert_data)
        
        def dict_to_insert(d):
            cols = list(d.keys())
            vals = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in d.values()]
            placeholders = ", ".join(["%s"] * len(cols))
            col_str = ", ".join(cols)
            return f"INSERT INTO meal_plans ({col_str}) VALUES ({placeholders})", vals
            
        query, vals = dict_to_insert(safe_data)
        execute_sql_write(query, tuple(vals))
        return True
    except Exception as try_db_e:
        err_msg = str(try_db_e)
        if "column" in err_msg and ("meal_names" in err_msg or "techniques" in err_msg):
            logger.warning("⚠️ [DB] Faltan columnas optimizadas (meal_names/techniques). Guardando sin ellas.")
            safe_data.pop("meal_names", None)
            safe_data.pop("ingredients", None)
            safe_data.pop("techniques", None)
            query, vals = dict_to_insert(safe_data)
            try:
                execute_sql_write(query, tuple(vals))
                return True
            except Exception as e2:
                raise e2
        else:
            raise try_db_e

def get_latest_meal_plan(user_id: str):
    """Obtiene el JSON del plan de comidas más reciente del usuario."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("plan_data")
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan actual: {e}")
        return None

def get_recent_meals_from_plans(user_id: str, days: int = 5):
    """Obtiene una lista de nombres de comidas de los planes recientes para evitar repeticiones."""
    if not supabase: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, meal_names").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
        meals = set() # 👈 Usar un Set evita enviar nombres duplicados al LLM y ahorra tokens
        if res.data:
            for row in res.data:
                meal_names_sql = row.get("meal_names")
                if meal_names_sql:
                    # 🚀 Fast Path O(1)
                    for n in meal_names_sql:
                        meals.add(n)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                         for day in plan_data.get("days", []):
                             for meal in day.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
                         if "meals" in plan_data:
                             for meal in plan_data.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
        return list(meals)
    except Exception as e:
        error_msg = str(e)
        if "meal_names" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna meal_names ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
                meals_fb = set()
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                             for day in plan_data.get("days", []):
                                 for meal in day.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                             if "meals" in plan_data:
                                 for meal in plan_data.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                return list(meals_fb)
            except Exception as e2:
                logger.error(f"Error obteniendo comidas recientes (fallback): {e2}")
                return []
                
        logger.error(f"Error obteniendo comidas recientes: {e}")
        return []

def get_recent_techniques(user_id: str, limit: int = 5) -> list:
    """Obtiene las técnicas de cocción usadas en planes recientes desde la columna `techniques` (text[]).
    Retorna una lista de tuplas (technique, created_at) para que el caller pueda aplicar decaimiento temporal.
    Ejemplo: [('Horneado Saludable', '2026-03-18T...'), ('Al Vapor', '2026-03-15T...')]
    """
    if not supabase or not user_id or user_id == "guest": return []
    try:
        res = supabase.table("meal_plans").select("techniques, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        # Retornar lista de tuplas CON duplicados y timestamps para decaimiento temporal.
        techniques = []
        if res.data:
            for row in res.data:
                techs = row.get("techniques")
                created_at = row.get("created_at", "")
                if techs and isinstance(techs, list):
                    for t in techs:
                        if t:
                            techniques.append((t, created_at))
        return techniques
    except Exception as e:
        error_msg = str(e)
        if "techniques" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # La columna aún no existe en la DB → retornar vacío silenciosamente
            return []
        logger.error(f"Error obteniendo técnicas recientes: {e}")
        return []

def get_ingredient_frequencies_from_plans(user_id: str, limit: int = 5) -> list:
    """Extrae los ingredientes crudos directamente del JSON o de la columna optimizada si existe.
    Retorna una lista plana de strings de ingredientes."""
    if not supabase or not user_id: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, ingredients").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        all_ingredients = []
        if res.data:
            for row in res.data:
                ingredients_sql = row.get("ingredients")
                if ingredients_sql:
                    # 🚀 Fast Path O(1)
                    all_ingredients.extend(ingredients_sql)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                        for day in plan_data.get("days", []):
                            for meal in day.get("meals", []):
                                ingredients = meal.get("ingredients", [])
                                if isinstance(ingredients, list):
                                    all_ingredients.extend(ingredients)
        return all_ingredients
    except Exception as e:
        error_msg = str(e)
        if "ingredients" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna ingredients ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
                all_ings_fb = []
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                            for day in plan_data.get("days", []):
                                for meal in day.get("meals", []):
                                    ings = meal.get("ingredients", [])
                                    if isinstance(ings, list):
                                        all_ings_fb.extend(ings)
                return all_ings_fb
            except Exception as e2:
                logger.error(f"Error extrayendo ingredientes de planes (fallback): {e2}")
                return []
                
        logger.error(f"Error extrayendo ingredientes de planes: {e}")
        return []

def get_latest_meal_plan_with_id(user_id: str):
    """Obtiene el plan más reciente del usuario incluyendo su ID para poder actualizarlo."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").select("id, plan_data, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan con ID: {e}")
        return None

def update_meal_plan_data(plan_id: str, new_plan_data: dict):
    """Actualiza el plan_data JSONB de un plan existente por su ID."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").update({"plan_data": new_plan_data}).eq("id", plan_id).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error actualizando plan_data: {e}")
        return None

def track_meal_friction(user_id: str, session_id: str, rejected_meal: str):
    """
    Memoria Conductual: Trackea cuántas veces el usuario rechaza platos con la misma proteína base.
    Al tercer rechazo (strike 3), inserta el ingrediente en rechazos temporales y notifica proactivamente.
    """
    if not user_id or user_id == "guest" or not rejected_meal: return False
    
    from constants import DOMINICAN_PROTEINS
    # GLOBAL_REVERSE_MAP is imported globally
    
    # Usar el mapa pre-computado a nivel de módulo (O(1)) en vez de reconstruirlo por llamada.
    # Crear versión sin acentos para matching robusto
    # (el LLM no siempre preserva tildes: "platano" vs "plátano")
    accent_safe_map = GLOBAL_REVERSE_MAP
    
    base_ingredient = None
    lower_meal = strip_accents(rejected_meal.lower())
    
    # Resolver sinónimos con n-gramas (trigrams → bigrams → unigrams)
    # para detectar multipalabra como "carne molida" → "res", "queso de freír" → "queso de freír"
    words = [w.strip(".,;:!?()\"'") for w in lower_meal.split()]
    for n in range(min(3, len(words)), 0, -1):  # 3-grams primero, luego 2, luego 1
        if base_ingredient:
            break
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in accent_safe_map:
                base_ingredient = accent_safe_map[ngram].capitalize()
                break
    
    # Fallback: búsqueda directa por nombre de proteína base
    if not base_ingredient:
        for p in DOMINICAN_PROTEINS:
            if p.lower() in lower_meal:
                base_ingredient = p
                break
            
    if not base_ingredient: return False
    
    # --- MÉTODO ATÓMICO (RPC) para evitar Race Conditions ---
    # Si dos requests de swap llegan simultáneos, el FOR UPDATE en PostgreSQL
    # serializa las escrituras garantizando que ambos incrementos se registren.
    try:
        rpc_result = supabase.rpc("increment_friction_rpc", {
            "p_user_id": user_id,
            "p_ingredient": base_ingredient
        }).execute()
        
        # El RPC retorna el conteo PRE-RESET (ej: 3 si alcanzó el umbral)
        new_count = rpc_result.data if isinstance(rpc_result.data, int) else 0
        
        if new_count >= 3:
            logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (vía RPC atómico).")
            
            rejection_record = {
                "meal_name": base_ingredient,
                "meal_type": "Ingrediente Fricción",
                "user_id": user_id,
                "session_id": session_id if session_id else None
            }
            insert_rejection(rejection_record)
            
            if session_id:
                msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
                save_message(session_id, "model", msg)
            return True
        
        return False
        
    except Exception as rpc_e:
        error_msg = str(rpc_e)
        if "Could not find the function" in error_msg or "PGRST202" in error_msg:
            # RPC aún no desplegado → fallback al método clásico (read-modify-write)
            pass
        else:
            logger.error(f"⚠️ [FRICCIÓN] Error en RPC atómico, usando fallback: {rpc_e}")
    
    # --- FALLBACK CLÁSICO (read-modify-write, vulnerable a race condition) ---
    # ⚠️ En producción, desplegar rpc_increment_friction.sql en Supabase para eliminar esto.
    profile = get_user_profile(user_id)
    if not profile: return False
    
    hp = profile.get("health_profile") or {}
    frictions = hp.get("frictions", {})
    
    current_count = frictions.get(base_ingredient, 0) + 1
    
    if current_count >= 3:
        logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (fallback).")
        
        rejection_record = {
            "meal_name": base_ingredient,
            "meal_type": "Ingrediente Fricción",
            "user_id": user_id,
            "session_id": session_id if session_id else None
        }
        insert_rejection(rejection_record)
        
        frictions[base_ingredient] = 0
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        
        if session_id:
            msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
            save_message(session_id, "model", msg)
        return True
    else:
        frictions[base_ingredient] = current_count
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        return False

def log_unknown_ingredients(user_id: str, unknown_ings: list, raw_map: dict = None):
    """Loguea ingredientes que el LLM genera pero que el sistema de sinónimos no reconoce.
    Se guardan en la tabla `unknown_ingredients` para revisión periódica y expansión del catálogo.
    Usa RPC atómico con fallback a upsert clásico.
    """
    if not supabase or not user_id or user_id == "guest" or not unknown_ings:
        return
    
    try:
        for ing in unknown_ings[:20]:  # Cap a 20 por plan para no saturar
            raw_text = raw_map.get(ing, "") if raw_map else ""
            try:
                # Intentar RPC atómico
                supabase.rpc("log_unknown_ingredient_rpc", {
                    "p_user_id": user_id,
                    "p_ingredient": ing,
                    "p_raw_text": raw_text or None
                }).execute()
            except Exception as rpc_e:
                err = str(rpc_e)
                if "Could not find the function" in err or "PGRST202" in err or "unknown_ingredients" in err:
                    # RPC o tabla no desplegados aún → silenciar
                    return
                # Fallback: upsert directo
                try:
                    from datetime import datetime, timezone
                    supabase.table("unknown_ingredients").upsert({
                        "user_id": user_id,
                        "ingredient": ing,
                        "raw_text": raw_text or None,
                        "occurrences": 1,
                        "last_seen": datetime.now(timezone.utc).isoformat()
                    }, on_conflict="user_id,ingredient").execute()
                except Exception as fb_e:
                    if "unknown_ingredients" in str(fb_e) or "PGRST205" in str(fb_e):
                        return  # Tabla no existe aún → silenciar
                    logger.error(f"⚠️ [UNKNOWN ING] Error en fallback: {fb_e}")
                    return
        
        logger.info(f"📝 [UNKNOWN ING] {len(unknown_ings)} ingredientes no reconocidos logueados para revisión")
    except Exception as e:
        logger.error(f"⚠️ [UNKNOWN ING] Error logueando ingredientes desconocidos: {e}")

def increment_ingredient_frequencies(user_id: str, ingredients: list[str]):
    """Incrementa la frecuencia histórica de los ingredientes consumidos por un usuario.
    Intenta usar un RPC atómico O(1) robusto ante Race Conditions,
    con fallback al viejo método Select+Upsert si la función SQL no se ha creado.
    """
    if not supabase or not user_id or user_id == "guest": return
    
    try:
        from collections import Counter
        from datetime import datetime, timezone
        
        # strip_accents is imported globally
            
        normalized_ings = [strip_accents(i.lower()).strip() for i in ingredients if i]
        if not normalized_ings: return
        
        incoming_counts = Counter(normalized_ings)
        ingredients_list = list(incoming_counts.keys())
        counts_list = list(incoming_counts.values())
        
        # 1. Intentar método atómico (RPC) para evitar Race Conditions
        try:
            supabase.rpc("increment_ingredient_frequencies_rpc", {
                "p_user_id": user_id,
                "p_ingredients": ingredients_list,
                "p_counts": counts_list
            }).execute()
            logger.info(f"✅ [DB] Frecuencia atómica (RPC) incrementada para {user_id} ({len(ingredients_list)} items)")
            return
        except Exception as rpc_e:
            error_msg = str(rpc_e)
            if "Could not find the function" in error_msg or "PGRST202" in error_msg:
                # El usuario aún no corre el código SQL en Supabase, pasamos al fallback silenciosamente
                pass
            else:
                logger.warning(f"⚠️ [DB] Aviso de RPC, recurriendo a fallback... Detalles: {rpc_e}")

        # 2. Fallback clásico: Leer estado actual y luego hacer upsert
        # ⚠️ RACE CONDITION: Si dos requests concurrentes leen el mismo count antes de que
        # cualquiera escriba, un incremento se pierde (lost update).
        # En producción, desplegar el RPC `increment_ingredient_frequencies_rpc` en Supabase
        # para garantizar atomicidad. Este fallback solo existe para desarrollo local.
        res = supabase.table("ingredient_frequencies").select("ingredient, count").eq("user_id", user_id).execute()
        current_map = {row["ingredient"]: row["count"] for row in res.data} if res.data else {}
        
        upsert_rows = []
        now_str = datetime.now(timezone.utc).isoformat()
        
        for ing, inc_val in incoming_counts.items():
            new_val = current_map.get(ing, 0) + inc_val
            upsert_rows.append({
                "user_id": user_id,
                "ingredient": ing,
                "count": new_val,
                "last_used": now_str
            })
            
        if upsert_rows:
            supabase.table("ingredient_frequencies").upsert(upsert_rows).execute()
            logger.info(f"✅ [DB] Frecuencia (Fallback Clásico) incrementada para {user_id} ({len(upsert_rows)} items)")
            
    except Exception as e:
        logger.error(f"⚠️ [DB] Error incrementando frecuecia de ingredientes: {e}")

def get_user_ingredient_frequencies(user_id: str, days_limit: int = 60) -> dict:
    """Retorna un diccionario {ingrediente_normalizado: conteo_decaimiento} de la DB.
    Implementa Decaimiento Temporal Continuo Matemático: count * (0.9 ^ dias_transcurridos).
    Se amplía days_limit a 60 días por defecto para dar margen al decaimiento suave.
    """
    if not supabase or not user_id or user_id == "guest": return {}
    try:
        from datetime import datetime, timedelta, timezone
        
        now = datetime.now(timezone.utc)
        cutoff_date = (now - timedelta(days=days_limit)).isoformat()
        
        res = supabase.table("ingredient_frequencies").select("ingredient, count, last_used").eq("user_id", user_id).gte("last_used", cutoff_date).execute()
        
        if not res.data:
            return {}
            
        freq_dict = {}
        decay_factor = 0.9  # Retiene 90% del peso por cada día que pasa sin uso
        
        for row in res.data:
            ingredient = row["ingredient"]
            count = row["count"]
            last_used_str = row.get("last_used")
            
            if not last_used_str:
                freq_dict[ingredient] = count
                continue
                
            try:
                # Parse robusto para last_used asumiendo formato de Supabase
                from constants import safe_fromisoformat
                last_used_dt = safe_fromisoformat(last_used_str)
                if last_used_dt.tzinfo is None:
                    last_used_dt = last_used_dt.replace(tzinfo=timezone.utc)
                        
                days_elapsed = max(0, (now - last_used_dt).days)
                
                # Fórmula decaimiento matemático: count * (decay_factor ^ days_elapsed)
                decayed_count = count * (decay_factor ** days_elapsed)
                
                freq_dict[ingredient] = round(decayed_count, 2)
            except Exception as parse_e:
                logger.error(f"⚠️ [DB] Error parseando fecha {last_used_str}: {parse_e}")
                freq_dict[ingredient] = count
                
        return freq_dict
    except Exception as e:
        logger.error(f"⚠️ [DB] Error obteniendo diccionario de frecuencias: {e}")
        return {}


