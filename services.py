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
from db_plans import save_new_meal_plan_atomic

from constants import normalize_ingredient_for_tracking, GLOBAL_REVERSE_MAP, IGNORED_TRACKING_TERMS
from db_inventory import release_meal_reservation

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



def save_partial_plan_get_id(user_id: str, plan_data: dict, selected_techniques: list = None, total_days_requested: int = 7) -> str:
    """Guarda la Semana 1 de un plan chunked de forma sincrónica y retorna el plan_id UUID.
    Usado exclusivamente por el flujo de Background Chunking para encolar las semanas restantes.
    """
    from db_plans import save_new_meal_plan_robust
    try:
        # [GAP 3] Limpieza de días huérfanos al regenerar
        if "days" in plan_data and len(plan_data["days"]) > total_days_requested:
            import logging
            logger.warning(f"🧹 [GAP 3] Recortando días huérfanos en partial plan. De {len(plan_data['days'])} a {total_days_requested}")
            plan_data["days"] = plan_data["days"][:total_days_requested]

        calories = plan_data.get("calories", 0)
        macros = plan_data.get("macros", {})

        meal_names = []
        ingredients = []
        for d in plan_data.get("days", []):
            for m in d.get("meals", []):
                if m.get("name"):
                    meal_names.append(m["name"])
                if m.get("ingredients"):
                    ingredients.extend(m["ingredients"])

        plan_name = generate_plan_title(plan_data)
        profile_embedding = plan_data.pop("_profile_embedding", None)

        insert_data = {
            "user_id": user_id,
            "plan_data": {**plan_data, "generation_status": "partial", "total_days_requested": total_days_requested},
            "name": plan_name,
            "calories": int(calories) if calories else 0,
            "macros": macros,
            "meal_names": meal_names,
            "ingredients": ingredients,
        }
        if profile_embedding:
            insert_data["profile_embedding"] = profile_embedding
        if selected_techniques:
            insert_data["techniques"] = selected_techniques

        # [P0-2/ATOMIC] INSERT del plan + cancelación de chunks en una sola transacción.
        # Elimina la ventana TOCTOU entre guardar el plan y cancelar los chunks viejos.
        plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)

        logger.info(f"💾 [CHUNK] Plan parcial (semana 1) guardado para {user_id}, plan_id={plan_id}")
        return plan_id
    except Exception as e:
        logger.error(f"❌ [CHUNK] Error guardando plan parcial para {user_id}: {e}")
        return None


def _save_plan_and_track_background(user_id: str, plan_data: dict, selected_techniques: list = None, additional_db_queries: list = None):
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
                
            # Extraer _profile_embedding si fue inyectado por la caché semántica
            profile_embedding = plan_data.pop("_profile_embedding", None)

            insert_data = {
                "user_id": user_id,
                "plan_data": plan_data,
                "name": plan_name,
                "calories": int(calories) if calories else 0,
                "macros": macros,
                "meal_names": meal_names,
                "ingredients": ingredients
            }
            if profile_embedding:
                insert_data["profile_embedding"] = profile_embedding
            
            # Añadir técnicas de cocción si están disponibles
            if selected_techniques:
                insert_data["techniques"] = selected_techniques
            

            # 🛡️ Dedup guard: evitar duplicados si otro código path ya guardó el plan
            if check_recent_meal_plan_exists(user_id, max_seconds=30):
                logger.info(f"🛡️ [DEDUP] Plan ya guardado recientemente para {user_id}. Omitiendo duplicado.")
                return
                
            if additional_db_queries:
                # Si hay queries adicionales, cancelar chunks primero y luego ejecutar
                # todo en una transacción via save_new_meal_plan_robust con additional_queries.
                # [P0-2] Cancelar chunks justo antes del INSERT (misma conexión, no la misma transacción
                # porque execute_sql_transaction no soporta RETURNING). Riesgo residual es mínimo
                # dado que el P0-1/TOCTOU guard en el worker actúa como red de seguridad.
                try:
                    from db_core import execute_sql_write
                    cancelled = execute_sql_write(
                        "UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW() "
                        "WHERE user_id = %s AND status IN ('pending', 'processing') RETURNING id",
                        (user_id,), returning=True
                    )
                    n_cancelled = len(cancelled) if cancelled else 0
                    if n_cancelled > 0:
                        logger.info(f"✅ [P0-2] {n_cancelled} chunks cancelados (pre-additional_queries) para {user_id}")
                except Exception as ce:
                    logger.warning(f"⚠️ [P0-2] Error cancelando chunks pre-additional_queries: {ce}")
                save_new_meal_plan_robust(insert_data, additional_queries=additional_db_queries)
            else:
                # [P0-2/ATOMIC] Cancelar chunks + INSERT en una sola transacción.
                save_new_meal_plan_atomic(user_id, insert_data, return_id=False)
            logger.debug(f"💾 [DB BACKGROUND] Plan guardado exitosamente en meal_plans para {user_id}")
            
        # 2. Track Frequencies (solo ingredientes canónicos que existan en los catálogos de variedad)
        if raw_ingredients:
            # Conjunto de términos base canónicos (ej: "pollo", "platano verde", "aguacate")
            canonical_bases = set(GLOBAL_REVERSE_MAP.values())
            
            normalized = [normalize_ingredient_for_tracking(ing) for ing in raw_ingredients]
            # Filtrar: solo trackear ingredientes que resolvieron a un término base conocido.
            # Esto evita que condimentos/hierbas (cilantro, orégano, ajo) polucionen la tabla.
            canonical = [n for n in normalized if n and n in canonical_bases]

            def _is_ignored(term: str) -> bool:
                """Ignora el término si es exacto o si alguna de sus palabras está en IGNORED_TRACKING_TERMS.
                Cubre compuestos como 'pimienta negra', 'canela en polvo', 'oregano dominicano'.
                """
                if term in IGNORED_TRACKING_TERMS:
                    return True
                return bool(set(term.split()) & IGNORED_TRACKING_TERMS)

            non_canonical = [n for n in normalized if n and n not in canonical_bases and not _is_ignored(n)]
            
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
            if user_id and user_id != "guest":
                release_meal_reservation(user_id, rejected_meal)
            
            # Fricción Silenciosa: Validar si la base ya se rechazó 3 veces
            track_meal_friction(user_id, session_id, rejected_meal)
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error procesando swap rejection: {e}")
