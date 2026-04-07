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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["plans"],
)

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
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
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
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
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
