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
    get_shopping_plan_hash, 
    save_shopping_plan_hash, 
    add_custom_shopping_items, 
    delete_auto_generated_shopping_items, 
    get_custom_shopping_items, 
    get_user_profile, 
    get_user_shopping_lock, 
    purge_old_shopping_items, 
    deduplicate_shopping_items, 
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
from agent import generate_auto_shopping_list
from ai_helpers import generate_plan_title

logger = logging.getLogger(__name__)


def compute_plan_hash(plan_data: dict) -> str:
    """Calcula un hash SHA-256 truncado del plan basado en ingredientes y suplementos.
    Fuente única de verdad para detectar si un plan cambió (usado por cache de shopping list)."""
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
            {"ingredients": all_ingredients, "supplements": sorted(set(all_supplements)), "version": "v4_multiday"},
            sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]


def regenerate_shopping_list_safe(
    user_id: str,
    items: list,
    existing_items: list,
    plan_hash: str
):
    """Ejecuta la operación atómica INSERT-FIRST / DELETE-OLD de la lista de compras.
    Incluye lock per-user, preservación de checkmarks, purga y deduplicación.
    Centralizado para evitar DRY violation entre app.py y _save_plan_and_track_background."""
    shopping_lock = get_user_shopping_lock(user_id)
    with shopping_lock:
        _preserve_shopping_checkmarks(existing_items, items)
        result = add_custom_shopping_items(user_id, items, source="auto")
        new_ids = [r.get("id") for r in result if r.get("id")] if result and isinstance(result, list) else []
        delete_auto_generated_shopping_items(user_id, exclude_ids=new_ids)
        purge_old_shopping_items(user_id)
        deduplicate_shopping_items(user_id)
        save_shopping_plan_hash(user_id, plan_hash)
    return result


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

def _preserve_shopping_checkmarks(existing_items: list, new_items: list):
    """Mantiene activa la casilla de ingredientes ya comprados (is_checked=True) al regenerar el plan."""
    if not existing_items or not new_items:
        return
    
    def _normalize(text: str) -> str:
        if not text: return ""
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))

    checked_names = set()
    for old in existing_items:
        if old.get("is_checked"):
            name = old.get("display_name")
            if not name:
                raw = old.get("item_name", "")
                if raw.startswith("{"):
                    try:
                        parsed = json.loads(raw)
                        name = parsed.get("name", raw)
                    except Exception:
                        name = raw
                else:
                    name = raw
            norm = _normalize(name)
            if norm:
                checked_names.add(norm)
                
    if not checked_names:
        return
        
    for new_item in new_items:
        if isinstance(new_item, dict):
            n_name = _normalize(new_item.get("name", ""))
            if n_name in checked_names:
                new_item["is_checked"] = True

def _save_plan_and_track_background(user_id: str, plan_data: dict, selected_techniques: list = None):
    """Background task: Guarda el plan JSON en supabase y trackea frecuencias, O(1)."""
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
            
        # 3. Auto-generar lista de compras (background, con cache por hash)
        try:
            plan_hash = compute_plan_hash(plan_data)
            if not plan_hash:
                raise ValueError("Plan sin ingredientes para hashear")
                
            stored_hash = get_shopping_plan_hash(user_id)
            existing = get_custom_shopping_items(user_id)
            existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
            
            # 🔒 REGLA DE TIEMPO PARA BACKGROUND (Evita mutación sutil si ya hay lista vigente)
            locked = False
            if existing_items:
                try:
                    from datetime import timezone
                    hp = get_user_profile(user_id)
                    hp_data = hp.get("health_profile", {}) if hp else {}
                    cycle_days = hp_data.get("shopping_cycle", {}).get("duration_days", 7)
                    
                    created_at_strs = [i.get("created_at") for i in existing_items if i.get("created_at")]
                    if created_at_strs:
                        oldest_str = min(created_at_strs)
                        if oldest_str.endswith("Z"):
                            oldest_str = oldest_str[:-1] + "+00:00"
                        created_dt = datetime.fromisoformat(oldest_str)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                            
                        days_elapsed = (datetime.now(timezone.utc) - created_dt).days
                        if days_elapsed < cycle_days:
                            locked = True
                            logger.info(f"🔒 [BACKGROUND BLOCKED] Lista auto-generada está vigente: {days_elapsed}/{cycle_days} días. Cancelando regeneración destructiva en background.")
                except Exception as e:
                    logger.error(f"Error parseando límite de compras en background para {user_id}: {e}")

            if locked:
                logger.debug(f"✅ [BACKGROUND SKIP] Plan ignorado por la lista de compras actual (Aún válida).")
            elif stored_hash != plan_hash:
                items = generate_auto_shopping_list(plan_data)
                if items:
                    regenerate_shopping_list_safe(user_id, items, existing_items, plan_hash)
                    logger.debug(f"🛒 [BACKGROUND] Lista de compras auto-generada ({len(items)} items) para {user_id}")
                else:
                    logger.warning(f"⚠️ [BACKGROUND] No se pudieron consolidar ingredientes para {user_id}")
            else:
                logger.debug(f"✅ [BACKGROUND CACHE HIT] Plan sin cambios, lista de compras ya actualizada para {user_id}")
        except Exception as shop_e:
            logger.error(f"⚠️ [BACKGROUND] Error auto-generando lista de compras: {shop_e}")
            
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
