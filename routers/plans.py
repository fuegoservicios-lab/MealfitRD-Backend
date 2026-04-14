from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks
from typing import Optional
import logging
import traceback
import os
import threading

# Importaciones relativas del entorno
from auth import get_verified_user_id, verify_api_quota
from db import (
    supabase, get_user_likes, get_active_rejections, get_or_create_session, 
    save_message, update_user_health_profile, log_api_usage, get_latest_meal_plan,
    get_latest_meal_plan_with_id, update_meal_plan_data, insert_like
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import analyze_preferences_agent, swap_meal
from graph_orchestrator import run_plan_pipeline
from ai_helpers import expand_recipe_agent
from services import _save_plan_and_track_background, _process_swap_rejection_background
from db_inventory import restock_inventory, consume_inventory_items_completely

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["plans"],
)

# ─── TEMPORARY DEBUG ENDPOINT (REMOVE AFTER DIAGNOSIS) ───
@router.get("/debug-scaling/{user_id}")
def debug_scaling(user_id: str):
    """Temporary: compare shopping list output for household sizes 1-6."""
    from shopping_calculator import get_shopping_list_delta
    from db_plans import get_latest_meal_plan_with_id as _get_plan
    
    plan_record = _get_plan(user_id)
    
    # Fallback: try to find ANY recent plan if user_id yields nothing
    if not plan_record:
        try:
            from db_core import execute_sql_query
            row = execute_sql_query(
                "SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at DESC LIMIT 1",
                fetch_one=True
            )
            if row:
                plan_record = row
                user_id = row.get("user_id", user_id)
            else:
                return {"error": f"No plans exist in database at all"}
        except Exception as e:
            return {"error": f"No plan found for {user_id} and fallback failed: {e}"}
    
    if not plan_record:
        return {"error": f"No plan found for {user_id}"}
    
    plan_data = plan_record["plan_data"]
    days = plan_data.get("days", [])
    num_days = len(days)
    
    KEYWORDS = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz', 'pollo', 'cebolla', 'tomate', 'melón', 'melon']
    
    comparison = {}
    for h in [1, 2, 3, 4, 5, 6]:
        scaled = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(h))
        row = {}
        for item in scaled:
            name = item.get("name", "")
            if any(kw in name.lower() for kw in KEYWORDS):
                row[name] = {
                    "display_qty": item.get("display_qty"),
                    "market_qty": item.get("market_qty"),
                    "market_unit": item.get("market_unit"),
                }
        comparison[f"{h}_personas"] = row
    
    return {
        "found_user_id": user_id,
        "plan_id": plan_record.get("id"),
        "num_days_in_plan": num_days,
        "base_duration_scale": round(7.0 / max(1, num_days), 4),
        "comparison": comparison
    }
# ─── END TEMPORARY DEBUG ENDPOINT ───

@router.post("/analyze")
def api_analyze(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id") 
        
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        history = []
        likes = []
        taste_profile = ""
        
        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]
            
        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
            
        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)
            
        result = run_plan_pipeline(data, history, taste_profile, 
                                   memory_context=memory.get("full_context_str", "") if session_id else "")
        
        if actual_user_id:
            hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
            if hp_data:
                update_user_health_profile(actual_user_id, hp_data)
                logger.info(f"💾 [SYNC] health_profile guardado para user {actual_user_id}")
        
        if session_id:
            goal = data.get('mainGoal', 'Desconocido')
            save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
            save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada generada analíticamente!")
            
            background_tasks.add_task(summarize_and_prune, session_id)
            
        if actual_user_id:
            log_api_usage(actual_user_id, "gemini_analyze")
            
        selected_techniques = result.pop("_selected_techniques", None)
        if actual_user_id:
            background_tasks.add_task(_save_plan_and_track_background, actual_user_id, result, selected_techniques)

        return result
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipe/expand")
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        if not data.get("recipe") or not data.get("name"):
            raise HTTPException(status_code=400, detail="Faltan datos de la receta para expandir.")
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_recipe_expand")
            
        expanded_steps = expand_recipe_agent(data)
        
        if user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            if current_plan and "days" in current_plan:
                updated = False
                for day in current_plan.get("days", []):
                    for m in day.get("meals", []):
                        if m.get("name") == data.get("name"):
                            m["recipe"] = expanded_steps
                            m["isExpanded"] = True
                            updated = True
                            break
                    if updated: break
                
                if updated:
                    plan_with_id = get_latest_meal_plan_with_id(user_id)
                    if plan_with_id and "id" in plan_with_id:
                        update_meal_plan_data(plan_with_id["id"], current_plan)

        return {"success": True, "expanded_recipe": expanded_steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recipe/expand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/swap-meal")
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        rejected_meal = data.get("rejected_meal")
        meal_type = data.get("meal_type", "")
        
        if rejected_meal:
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_swap_meal")
            
        result = swap_meal(data)
        return result
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/like")
def api_like(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        insert_like(data)
        return {"success": True, "message": "Tu like/dislike ha sido guardado exitosamente."}
    except Exception as e:
        return {"error": str(e)}

@router.post("/restock")
def api_restock(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        plan_id = data.get("plan_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión para usar la nevera virtual."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}

        # Validación MURO: Comprobar si el plan ya fue registrado para evitar duplicados
        plan_res = None
        if plan_id and supabase:
            plan_res = supabase.table("meal_plans").select("plan_data").eq("id", plan_id).execute()
            if plan_res.data and len(plan_res.data) > 0:
                plan_data = plan_res.data[0].get("plan_data", {})
                if plan_data.get("is_restocked") is True:
                    return {"success": False, "message": "El plan ya ha sido registrado en la despensa previamente."}
            
        success = restock_inventory(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "restock_inventory")

            # Marcar el plan como "restocked" en BD para proteger futuras peticiones
            if plan_id and supabase and plan_res and plan_res.data and len(plan_res.data) > 0:
                plan_data = plan_res.data[0].get("plan_data", {})
                plan_data["is_restocked"] = True
                supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id", plan_id).execute()

            return {"success": True, "message": "¡Despensa actualizada exitosamente!"}
        else:
            return {"success": False, "message": "Hubo un problema actualizando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/restock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inventory/consume")
def api_consume_inventory(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}
            
        success = consume_inventory_items_completely(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "consume_inventory")
            return {"success": True, "message": "Inventario actualizado exitosamente."}
        else:
            return {"success": False, "message": "Hubo un problema vaciando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/inventory/consume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recalculate-shopping-list")
def api_recalculate_shopping_list(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """
    Recalcula la lista de compras escalando las recetas por el householdSize 
    y LUEGO deduciendo el inventario físico (is_new_plan=False).
    Este acercamiento garantiza exactitud matemática.
    """
    try:
        user_id = data.get("user_id")
        household_size = max(1, int(data.get("householdSize", 1) or 1))
        grocery_duration = data.get("groceryDuration", "weekly")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
        
        plan_record = get_latest_meal_plan_with_id(user_id)
        if not plan_record:
            return {"success": False, "message": "No hay plan activo."}
        
        plan_id = plan_record["id"]
        plan_data = plan_record["plan_data"]
        
        if not plan_data or not isinstance(plan_data, dict):
            return {"success": False, "message": "Plan corrupto."}
        
        logger.info(f"🔄 [RECALC] Escalando lista de compras ×{household_size} personas para user {user_id}")
        
        from shopping_calculator import get_shopping_list_delta
        from db_plans import update_meal_plan_data
        
        # Generar las listas estructuradas con el multiplier
        # is_new_plan=True para obtener la lista COMPLETA sin deducción de inventario
        # La deducción del inventario no tiene sentido aquí porque los ingredientes
        # ya fueron escalados para N personas — el inventario es fijo.
        scaled_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(household_size))
        scaled_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(household_size) * 2.0)
        scaled_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(household_size) * 4.0)
        
        # Debug: Log DETAILED per-item comparison to diagnose scaling
        if scaled_7:
            sample = [f"{it.get('display_string','?')}" for it in scaled_7[:3]]
            has_days = bool(plan_data.get("days"))
            len_days = len(plan_data.get("days", []))
            has_perfectDay = bool(plan_data.get("perfectDay"))
            logger.info(f"🔍 [RECALC DEBUG] ×{household_size} sample (7d): {sample} | has_days={has_days} len={len_days} has_perf={has_perfectDay}")
            
            # DEBUG GRANULAR: rastear proteínas y frutas específicas
            debug_keywords = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'melón', 'melon', 'aguacate', 'arroz', 'pollo']
            for it in scaled_7:
                name_lower = it.get('name', '').lower()
                if any(kw in name_lower for kw in debug_keywords):
                    logger.info(f"  📊 [{household_size}p] {it.get('name')}: display_qty={it.get('display_qty')} | market_qty={it.get('market_qty')} {it.get('market_unit')} | display_string={it.get('display_string')}")
        
        # Seleccionar lista activa para el frontend legacy
        if grocery_duration == "biweekly":
            active_list = scaled_15
        elif grocery_duration == "monthly":
            active_list = scaled_30
        else:
            active_list = scaled_7
        
        # Actualizar plan en BD
        plan_data["aggregated_shopping_list"] = active_list
        plan_data["aggregated_shopping_list_weekly"] = scaled_7
        plan_data["aggregated_shopping_list_biweekly"] = scaled_15
        plan_data["aggregated_shopping_list_monthly"] = scaled_30
        
        # 🔄 Limpiar is_restocked: las cantidades cambiaron, el usuario necesita
        # re-registrar las compras con las nuevas cantidades en la despensa
        if plan_data.get("is_restocked"):
            plan_data.pop("is_restocked", None)
            logger.info(f"🔄 [RECALC] is_restocked limpiado — cantidades cambiaron, requiere re-registro")
        
        # DEBUG fingerprint: allows frontend to verify it received fresh data
        import time
        plan_data["_debug_recalc"] = {
            "household_size": household_size,
            "timestamp": time.time(),
            "weekly_items_count": len(scaled_7),
            "sample_item": scaled_7[0].get("display_string", "?") if scaled_7 else "empty"
        }
        
        update_meal_plan_data(plan_id, plan_data)
        
        logger.info(f"✅ [RECALC] Listas recalculadas exitosamente ×{household_size} personas")
        
        # Devolver el plan_data actualizado directamente para evitar race conditions
        # (el frontend no necesita re-fetch de Supabase)
        return {"success": True, "plan_data": plan_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recalculate-shopping-list: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
