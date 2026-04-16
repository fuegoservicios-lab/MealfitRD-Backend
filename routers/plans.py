from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
import traceback
import os
import threading
import asyncio
import json as _json
import time as _time

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

@router.post("/analyze/stream")
async def api_analyze_stream(request: Request, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """
    Streaming SSE endpoint para generación de planes con progreso en tiempo real.
    Emite eventos:
      - phase: cambio de fase del pipeline (skeleton, parallel_generation, assembly, review)
      - day_started: un worker paralelo inició la generación de un día
      - day_complete: un worker paralelo terminó un día
      - complete: plan final listo (contiene el plan JSON completo)
      - error: hubo un error
    """
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

        # Cola async para comunicar progreso entre el thread del pipeline y el generador SSE
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(event_data: dict):
            """Callback thread-safe que pone eventos en la cola async."""
            try:
                loop.call_soon_threadsafe(progress_queue.put_nowait, event_data)
            except Exception:
                pass

        # Ejecutar el pipeline en un thread separado para no bloquear el event loop
        pipeline_result = {"result": None, "error": None}

        def run_pipeline():
            try:
                result = run_plan_pipeline(
                    data, history, taste_profile,
                    memory_context=memory.get("full_context_str", "") if session_id else "",
                    progress_callback=progress_callback,
                )
                pipeline_result["result"] = result
            except Exception as e:
                pipeline_result["error"] = str(e)
                logger.error(f"❌ [SSE PIPELINE ERROR]: {e}")
                traceback.print_exc()
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()

        async def event_generator():
            """Generador SSE que consume la cola de progreso."""
            try:
                while True:
                    # Esperar eventos con timeout para detectar desconexión del cliente
                    try:
                        event_data = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Heartbeat para mantener la conexión viva
                        yield f"data: {_json.dumps({'event': 'heartbeat'})}\n\n"

                        # Verificar si el cliente cerró la conexión
                        if await request.is_disconnected():
                            logger.info("🔌 [SSE] Cliente desconectado, abortando stream.")
                            return
                        continue

                    # Señal de fin del pipeline
                    if event_data.get("event") == "_done":
                        # Enviar resultado final o error
                        if pipeline_result["error"]:
                            yield f"data: {_json.dumps({'event': 'error', 'data': {'message': pipeline_result['error']}})}\n\n"
                        elif pipeline_result["result"]:
                            result = pipeline_result["result"]

                            # Post-procesamiento idéntico al endpoint síncrono
                            if actual_user_id:
                                hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
                                if hp_data:
                                    try:
                                        update_user_health_profile(actual_user_id, hp_data)
                                    except Exception:
                                        pass

                            if session_id:
                                goal = data.get('mainGoal', 'Desconocido')
                                try:
                                    save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
                                    save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada!")
                                    # summarize_and_prune se ejecuta en background
                                    threading.Thread(target=summarize_and_prune, args=(session_id,), daemon=True).start()
                                except Exception:
                                    pass

                            if actual_user_id:
                                try:
                                    log_api_usage(actual_user_id, "gemini_analyze")
                                except Exception:
                                    pass

                            selected_techniques = result.pop("_selected_techniques", None)
                            if actual_user_id:
                                threading.Thread(
                                    target=_save_plan_and_track_background,
                                    args=(actual_user_id, result, selected_techniques),
                                    daemon=True
                                ).start()

                            yield f"data: {_json.dumps({'event': 'complete', 'data': result}, ensure_ascii=False, default=str)}\n\n"
                        return

                    # Evento de progreso normal
                    yield f"data: {_json.dumps(event_data, ensure_ascii=False)}\n\n"

            except asyncio.CancelledError:
                logger.info("🔌 [SSE] Stream cancelado por el cliente.")
            except Exception as e:
                logger.error(f"❌ [SSE] Error en generador: {e}")
                yield f"data: {_json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze/stream: {str(e)}")
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
        swap_reason = data.get("swap_reason", "dislike")  # variety | time | dislike | similar
        
        # Solo registrar rechazo cuando el usuario explícitamente dice "No me gusta"
        if rejected_meal and swap_reason == "dislike":
            logger.info(f"👎 [SWAP] Rechazo real registrado: '{rejected_meal}' (razón: {swap_reason})")
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)
        else:
            logger.info(f"🔄 [SWAP] Cambio sin rechazo: '{rejected_meal or 'N/A'}' (razón: {swap_reason})")
            
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

        # Validación MURO Omitida: Ahora confiamos en el Delta Shopping del frontend.
        # El frontend solo envía los ingredientes que no están en la Nevera.
        success = restock_inventory(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "restock_inventory")

            # Marcar el plan como "restocked" en BD para futuras peticiones
            if supabase:
                try:
                    if plan_id:
                        plan_res = supabase.table("meal_plans").select("id, plan_data").eq("id", plan_id).execute()
                    else:
                        plan_res = supabase.table("meal_plans").select("id, plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                        
                    if plan_res and plan_res.data and len(plan_res.data) > 0:
                        real_plan_id = plan_res.data[0].get("id")
                        plan_data = plan_res.data[0].get("plan_data", {})
                        plan_data["is_restocked"] = True
                        supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id", real_plan_id).execute()
                        logger.info(f"✅ [RESTOCK] plan_data 'is_restocked' guardado en DB para plan ID {real_plan_id}")
                except Exception as mark_err:
                    logger.warning(f"⚠️ No se pudo marcar plan como restocked: {mark_err}")

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
        
        # Solo limpiar `is_restocked` si los parámetros cambiaron realmente
        prev_hh = plan_data.get("calc_household_size")
        prev_dur = plan_data.get("calc_grocery_duration")
        has_changed = (prev_hh != household_size) or (prev_dur != grocery_duration)
        
        plan_data["calc_household_size"] = household_size
        plan_data["calc_grocery_duration"] = grocery_duration
        
        if has_changed and plan_data.get("is_restocked"):
            plan_data.pop("is_restocked", None)
            logger.info(f"🔄 [RECALC] is_restocked limpiado — cantidades cambiaron de {prev_hh}p/{prev_dur} a {household_size}p/{grocery_duration}, requiere re-registro")
        
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
