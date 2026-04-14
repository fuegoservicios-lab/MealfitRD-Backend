import logging
from typing import Optional
from datetime import datetime
import json
import unicodedata
import re
import hashlib

# Imports globales locales, movidos al tope para evitar el code smell "Lazy Loading"
from db import (
    supabase, 
    increment_ingredient_frequencies, 
    check_recent_meal_plan_exists, 
    save_new_meal_plan_robust, 
    log_unknown_ingredients, 
    get_user_profile, 
    get_or_create_session, 
    save_message, 
    insert_rejection, 
    track_meal_friction,
    update_user_health_profile,
    upsert_user_profile
)

from constants import normalize_ingredient_for_tracking, GLOBAL_REVERSE_MAP

# ⚠️ RESTRICCIÓN ARQUITECTÓNICA: services.py importa agent.py → agent.py NUNCA debe importar services.py.
# Si agent.py necesita lógica de services.py en el futuro, usar lazy import dentro de la función.
from ai_helpers import generate_plan_title

logger = logging.getLogger(__name__)


def compute_plan_hash(plan_data: dict) -> str:
    """Calcula un hash SHA-256 truncado del plan basado en ingredientes y suplementos.
    Fuente única de verdad para detectar si un plan cambió."""
    all_ingredients = []
    all_supplements = []
    for d in plan_data.get("days", []):
        for m in d.get("meals", []):
            ing = m.get("ingredients", [])
            if ing:
                all_ingredients.extend(ing)
        for s in d.get("supplements") or []:
            all_supplements.append(s.get("name", ""))
    return hashlib.sha256(
        json.dumps(
            {"ingredients": all_ingredients, "supplements": sorted(set(all_supplements)), "version": "v7_deterministic_math"},
            sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]





def merge_form_data_with_profile(user_id: str, form_data: Optional[dict]) -> dict:
    """
    Merges frontend form_data with the stored health_profile from DB.
    Extracted to avoid DRY violation between /api/chat/stream and /api/chat.
    Returns the merged form_data dict.
    """
    merged = form_data or {}
    if not user_id or user_id == "guest" or user_id == "":
        return merged
    try:
        profile = get_user_profile(user_id)
        if profile:
            existing_hp = profile.get("health_profile") or {}
            if existing_hp:
                non_empty_form = {k: v for k, v in merged.items() if v not in [None, "", [], {}]}
                existing_hp.update(non_empty_form)
                merged = existing_hp
            if form_data and not existing_hp:
                logger.debug(f"🔄 [SYNC] health_profile vacío, sincronizando desde formData del frontend...")
                update_user_health_profile(user_id, merged)
        else:
            if form_data:
                logger.warning(f"⚠️ [SYNC] No existe user_profile para {user_id}, intentando crear...")
                try:
                    upsert_user_profile(user_id, merged)
                    logger.info(f"✅ [SYNC] Perfil creado con health_profile")
                except Exception as e:
                    logger.error(f"❌ [SYNC] Error creando perfil: {e}")
    except Exception as e:
        logger.error(f"⚠️ Error cargando health profile en chat: {e}")
    return merged



def _save_plan_and_track_background(user_id: str, plan_data: dict, selected_techniques: list = None):
    """Background task para guardar plan y actualizar frecuencias de ingredientes."""
    try:
        # 1. Guardar Plan O(1) Arrays
        if supabase:
            calories = plan_data.get("calories", 0)
            macros = plan_data.get("macros", {})
            
            meal_names = []
            ingredients = []
            raw_ingredients = []
            for d in plan_data.get("days", []):
                for m in d.get("meals", []):
                    if m.get("name"):
                        meal_names.append(m.get("name"))
                    if m.get("ingredients"):
                        ingredients.extend(m.get("ingredients"))
                        raw_ingredients.extend(m.get("ingredients"))
                        
            # Nombre creativo generado por IA (Gemini Flash-Lite)
            plan_name = generate_plan_title(plan_data)
                
            insert_data = {
                "user_id": user_id,
                "plan_data": plan_data,
                "name": plan_name,
                "calories": int(calories) if calories else 0,
                "macros": macros,
                "meal_names": meal_names,
                "ingredients": ingredients
            }
            
            # Añadir técnicas de cocción si están disponibles
            if selected_techniques:
                insert_data["techniques"] = selected_techniques
            
            # 🔄 Heredar is_restocked del plan anterior si sigue en el mismo ciclo de compras
            # Esto evita que "Registrar Compras" reaparezca en otros dispositivos
            # cuando el usuario solo hizo "Actualizar Platos" (rotación de recetas).
            try:
                prev_plan = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                if prev_plan.data and len(prev_plan.data) > 0:
                    prev_data = prev_plan.data[0].get("plan_data", {})
                    if prev_data.get("is_restocked") is True:
                        # Verificar que el ciclo de compras no expiró
                        prev_start = prev_data.get("grocery_start_date")
                        if prev_start:
                            from datetime import datetime as dt_check
                            try:
                                start_dt = dt_check.fromisoformat(prev_start.replace("Z", "+00:00"))
                                days_elapsed = (datetime.now(start_dt.tzinfo) - start_dt).days if start_dt.tzinfo else (datetime.now() - start_dt).days
                                # Si el ciclo no ha expirado (30 días max), heredar el flag
                                if days_elapsed <= 30:
                                    plan_data["is_restocked"] = True
                                    plan_data["grocery_start_date"] = prev_start
                                    logger.info(f"🔄 [RESTOCK INHERIT] Plan hereda is_restocked=True del ciclo anterior ({days_elapsed}d elapsed)")
                            except Exception as date_err:
                                logger.warning(f"⚠️ [RESTOCK INHERIT] Error parseando fecha: {date_err}")
                                # Safe fallback: heredar de todos modos si la fecha no se puede parsear
                                plan_data["is_restocked"] = True
                                if prev_start:
                                    plan_data["grocery_start_date"] = prev_start
                        else:
                            # Sin fecha de inicio, heredar de todos modos (seguridad)
                            plan_data["is_restocked"] = True
                            logger.info(f"🔄 [RESTOCK INHERIT] Plan hereda is_restocked=True (sin grocery_start_date)")
            except Exception as restock_err:
                logger.warning(f"⚠️ [RESTOCK INHERIT] Error consultando plan anterior: {restock_err}")

            # 🛡️ Dedup guard: evitar duplicados si otro código path ya guardó el plan
            if check_recent_meal_plan_exists(user_id, max_seconds=30):
                logger.info(f"🛡️ [DEDUP] Plan ya guardado recientemente para {user_id}. Omitiendo duplicado.")
                return
                
            save_new_meal_plan_robust(insert_data)
            logger.debug(f"💾 [DB BACKGROUND] Plan guardado exitosamente en meal_plans para {user_id}")
            
        # 2. Track Frequencies (solo ingredientes canónicos que existan en los catálogos de variedad)
        if raw_ingredients:
            # Conjunto de términos base canónicos (ej: "pollo", "platano verde", "aguacate")
            canonical_bases = set(GLOBAL_REVERSE_MAP.values())
            
            normalized = [normalize_ingredient_for_tracking(ing) for ing in raw_ingredients]
            # Filtrar: solo trackear ingredientes que resolvieron a un término base conocido.
            # Esto evita que condimentos/hierbas (cilantro, orégano, ajo) polucionen la tabla.
            canonical = [n for n in normalized if n and n in canonical_bases]
            non_canonical = [n for n in normalized if n and n not in canonical_bases]
            
            if canonical:
                increment_ingredient_frequencies(user_id, canonical)
            
            # 2b. Loguear ingredientes no reconocidos para revisión y expansión del catálogo
            if non_canonical:
                raw_map = {normalize_ingredient_for_tracking(r): r for r in raw_ingredients if r}
                log_unknown_ingredients(user_id, non_canonical, raw_map)
                logger.info(f"🧹 [FREQ TRACKING] {len(non_canonical)} ingredientes no-canónicos logueados para revisión")
                
            logger.info(f"📈 [FREQ TRACKING] Frecuencias actualizadas en background para {user_id} ({len(canonical)} ingredientes canónicos trackeados)")
            

            
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error asíncrono guardando plan: {e}")


def _process_swap_rejection_background(session_id: str, user_id: str, rejected_meal: str, meal_type: str):
    """Background task: Loguea mensajes y rechazos que expiran en 7 días, asíncronamente."""
    try:
        if session_id and rejected_meal:
            get_or_create_session(session_id)
            save_message(session_id, "user", f"Rechacé explícitamente: {rejected_meal}")
        
        # Guardar rechazo TEMPORAL (expira en 7 días)
        if rejected_meal:
            rejection_record = {
                "meal_name": rejected_meal,
                "meal_type": meal_type,
            }
            if user_id and user_id != "guest":
                rejection_record["user_id"] = user_id
            if session_id:
                rejection_record["session_id"] = session_id
            
            insert_rejection(rejection_record)
            logger.debug(f"💾 [DB BACKGROUND] Rechazo temporal guardado para {rejected_meal}")
            
            # Fricción Silenciosa: Validar si la base ya se rechazó 3 veces
            track_meal_friction(user_id, session_id, rejected_meal)
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error procesando swap rejection: {e}")
