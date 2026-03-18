from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File, Form, Body, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from agent import swap_meal, chat_with_agent, analyze_preferences_agent
from graph_orchestrator import run_plan_pipeline
from db import get_or_create_session, save_message, insert_like, get_user_likes, insert_rejection, get_active_rejections, get_latest_meal_plan, get_user_profile, update_user_health_profile, get_all_user_facts, delete_user_fact, get_custom_shopping_items, delete_custom_shopping_item, log_api_usage, get_monthly_api_usage, connection_pool
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts
from agent import generate_chat_title_background
from langgraph.checkpoint.postgres import PostgresSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    if connection_pool:
        connection_pool.open()
        try:
            import psycopg
            db_uri = os.environ.get("SUPABASE_DB_URL")
            # Setup requires a direct connection with autocommit=True because CREATE INDEX CONCURRENTLY cannot run inside a transaction
            with psycopg.connect(db_uri, autocommit=True) as conn:
                PostgresSaver(conn).setup()
            print("🚀 [Postgres] Tablas de LangGraph Checkpointer verificadas/creadas.")
        except Exception as e:
            print(f"⚠️ [Postgres] Error configurando checkpointer: {e}")
            
    print("🚀 [FastAPI] Servidor de MealfitRD IA iniciado con éxito en el puerto 3001.")
    yield
    
    if connection_pool:
        connection_pool.close()
        print("🔌 [psycopg] Pool de conexiones cerrado.")


# Asegurarnos de que el directorio de uploads exista antes de montar recursos estáticos
os.makedirs("uploads", exist_ok=True)

app = FastAPI(lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Dependencia de seguridad para validar token JWT de Supabase
def get_verified_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extrae el user_id del token JWT en el header Authorization."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    from db import supabase
    if not supabase:
        return None
    try:
        # User auth verification with supabase
        user_res = supabase.auth.get_user(token)
        if user_res and user_res.user:
            return user_res.user.id
    except Exception as e:
        print(f"⚠️ [AUTH] Error validando token: {e}")
    return None

# Setup CORS para el frontend React local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://localhost:5174", 
        "http://127.0.0.1:5174"
    ], # Especificamos las URLs exactas en lugar de "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
def api_analyze(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id") # Para buscar likes (si está logueado)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        history = []
        likes = []
        taste_profile = ""
        
        if session_id:
            get_or_create_session(session_id)
            # Usar sistema de memoria inteligente (resúmenes + mensajes recientes)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]
            
        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)
            
        # --- LÍMITE DE USO (PAYWALL) ---
        if actual_user_id:
            from db import get_monthly_api_usage
            credits_used = get_monthly_api_usage(actual_user_id)
            plan_tier = "gratis"
            
            profile = get_user_profile(actual_user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                
            limit = 15 if plan_tier == "gratis" else (100 if plan_tier == "plus" else 999999)
            
            if credits_used >= limit:
                raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")
        # -------------------------------

        # 1. Obtener rechazos activos (últimos 7 días solamente)
        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
            
        # 2. Llamar al Agente Especialista en Preferencias (con rechazos temporales)
        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)
            
        # 3. Ejecutar Pipeline Multi-Agente (LangGraph: Generador → Revisor Médico)
        # Pasar el contexto completo (resúmenes + recientes) al pipeline
        result = run_plan_pipeline(data, history, taste_profile, 
                                   memory_context=memory.get("full_context_str", "") if session_id else "")
        
        # 4. Persistir los datos del formulario en user_profiles.health_profile
        if user_id and user_id != "guest":
            hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
            if hp_data:
                update_user_health_profile(user_id, hp_data)
                print(f"💾 [SYNC] health_profile guardado para user {user_id}")
        
        if session_id:
            goal = data.get('mainGoal', 'Desconocido')
            save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
            save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada generada analíticamente!")
            
            # 🧠 Background: Resumir y podar mensajes si el historial creció demasiado
            background_tasks.add_task(summarize_and_prune, session_id)
            
        # 👇 NUEVO: Registramos uso de API de Gemini
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_analyze")
            
        # 👇 NUEVO: Guardar el plan generado en la base de datos
        if user_id and user_id != "guest":
            try:
                from db import supabase
                from datetime import datetime
                if supabase:
                    calories = result.get("calories", 0)
                    macros = result.get("macros", {})
                    supabase.table("meal_plans").insert({
                        "user_id": user_id,
                        "plan_data": result,
                        "name": f"Plan Evolutivo - {datetime.now().strftime('%d/%m/%Y')}",
                        "calories": int(calories) if calories else 0,
                        "macros": macros,
                    }).execute()
                    print(f"💾 [DB] Plan guardado exitosamente en meal_plans para {user_id}")
            except Exception as db_e:
                print(f"⚠️ [DB ERROR] No se pudo guardar el plan en Supabase: {db_e}")

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/swap-meal")
def api_swap_meal(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        
        # Guardar en memoria el rechazo para que el Agente de Preferencias aprenda
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        rejected_meal = data.get("rejected_meal")
        meal_type = data.get("meal_type", "")
        
        if session_id and rejected_meal:
            get_or_create_session(session_id)
            save_message(session_id, "user", f"Rechacé explícitamente: {rejected_meal}")
        
        # Guardar rechazo TEMPORAL (expira en 7 días) en la tabla meal_rejections
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
            print(f"📝 Rechazo temporal guardado: '{rejected_meal}' (expira en 7 días)")
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_swap_meal")
            
        result = swap_meal(data)
        return result
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/like")
def api_like(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        insert_like(data)
        return {"success": True, "message": "Tu like/dislike ha sido guardado exitosamente."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/user/credits/{user_id}")
def api_get_user_credits(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Consulta los créditos consumidos en el mes usando api_usage."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        from db import get_monthly_api_usage
        if not user_id or user_id == "guest":
            return {"credits": 0}
        credits_used = get_monthly_api_usage(user_id)
        return {"credits": credits_used}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/user/credits GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user-facts/{user_id}")
def api_get_user_facts(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"facts": []}
        facts = get_all_user_facts(user_id)
        return {"facts": facts}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/user-facts GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/user-facts/{fact_id}")
def api_delete_user_fact(fact_id: str):
    try:
        result = delete_user_fact(fact_id)
        return {"success": True, "message": "Hecho eliminado de la IA."}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/user-facts DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/sessions/{user_id}")
def api_get_chat_sessions(user_id: str, session_ids: Optional[str] = None, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    from db import get_guest_chat_sessions, get_user_chat_sessions
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        sessions: list = get_user_chat_sessions(user_id) or []
        
        # Siempre leer los session_ids del frontend (localStorage) como capa de seguridad. 
        # Si la BD no tiene la columna user_id, los sessions de arriba regresan vacíos, pero aquí los recuperamos.
        if session_ids:
            guest_sessions = get_guest_chat_sessions(session_ids.split(","))
            if guest_sessions:
                # Merge lists deduplicating by 'id'
                existing_ids = {s["id"] for s in sessions}
                for gs in guest_sessions:
                    if gs["id"] not in existing_ids:
                        sessions.append(gs)
                        
        # Sort again by last_activity descending after merge
        sessions.sort(key=lambda x: x.get("last_activity") or x.get("created_at") or "1970-01-01T00:00:00", reverse=True)
            
        return {"sessions": sessions}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/chat/sessions GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}")
def api_get_chat_history(session_id: str):
    from db import get_session_messages
    try:
        messages = get_session_messages(session_id)
        # Ocultar mensajes de sistema como el system_title
        filtered_messages = [m for m in messages if not m.get("content", "").startswith("[SYSTEM_TITLE]")]
        return {"messages": filtered_messages}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/chat/history GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/message")
def api_save_chat_message(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    session_id = data.get("session_id")
    role = data.get("role")
    content = data.get("content")
    user_id = data.get("user_id", session_id)
    
    # Validación de seguridad IDOR
    if user_id and user_id != "guest" and user_id != session_id:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
            
    if session_id and role and content:
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, role, content)
        return {"success": True}
    return {"success": False, "error": "Faltan parámetros"}

from fastapi.responses import StreamingResponse
import asyncio

@app.post("/api/chat/stream")
async def api_chat_stream(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", session_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        # --- LÍMITE DE USO (PAYWALL) ---
        if user_id and user_id != "guest":
            from db import get_monthly_api_usage
            credits_used = get_monthly_api_usage(user_id)
            plan_tier = "gratis"
            
            profile = get_user_profile(user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                
            limit = 15 if plan_tier == "gratis" else (100 if plan_tier == "plus" else 999999)
            
            if credits_used >= limit:
                raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")
        # -------------------------------
                
        print(f"🔍 [DEBUG API CHAT STREAM] session_id={session_id}, user_id={user_id}")
        
        # Estas operaciones son síncronas pero muy rápidas
        await asyncio.to_thread(get_or_create_session, session_id, user_id=user_id if user_id != "guest" else None)
        await asyncio.to_thread(save_message, session_id, "user", prompt)
        
        # Sincronizar form_data al health_profile si está vacío en la DB
        if form_data and user_id and user_id != "guest" and user_id != session_id:
            profile = get_user_profile(user_id)
            if profile:
                existing_hp = profile.get("health_profile") or {}
                if not existing_hp:
                    print(f"🔄 [SYNC STREAM] health_profile vacío, sincronizando...")
                    await asyncio.to_thread(update_user_health_profile, user_id, form_data)
            else:
                try:
                    from db import supabase as sb_client
                    if sb_client:
                        def _upsert_profile():
                            sb_client.table("user_profiles").upsert({
                                "id": user_id,
                                "health_profile": form_data
                            }).execute()
                        await asyncio.to_thread(_upsert_profile)
                except Exception as e:
                    print(f"❌ [SYNC STREAM] Error creando perfil: {e}")
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = await asyncio.to_thread(get_latest_meal_plan, user_id)
            
        from agent import achat_with_agent_stream
        
        async def event_generator():
            try:
                # Obtenemos los chunks del generador de LangGraph
                async for chunk in achat_with_agent_stream(
                    session_id=session_id, 
                    prompt=prompt, 
                    current_plan=current_plan, 
                    user_id=user_id, 
                    form_data=form_data
                ):
                    yield chunk
                    
                    # Interceptar el evento 'done' para lanzar background tasks
                    if chunk.startswith("data: "):
                        import json
                        try:
                            data_obj = json.loads(chunk[len("data: "):].strip())
                            if data_obj.get("type") == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    # Guardamos la respuesta final en DB  
                                    await asyncio.to_thread(save_message, session_id, "model", response_text)
                                    
                                # Lógica Background (resumir, uso de API, embeddings)
                                def bg_tasks():
                                    if user_id and user_id != "guest" and user_id != session_id:
                                        log_api_usage(user_id, "gemini_chat")
                                        
                                    try:
                                        from db import get_session_messages
                                        raw_history = get_session_messages(session_id)
                                        recent_history_str = ""
                                        if raw_history:
                                            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
                                        
                                        is_plus = False
                                        if user_id and user_id != "guest":
                                            profile_sync = get_user_profile(user_id)
                                            if profile_sync:
                                                plan_tier_sync = profile_sync.get("plan_tier", "gratis")
                                                is_plus = plan_tier_sync in ["plus", "admin", "ultra"]
                                                
                                        if is_plus:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)
                                            
                                        generate_chat_title_background(session_id, prompt)
                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        print(f"Error en bg tasks: {inner_e}")
                                
                                await asyncio.to_thread(bg_tasks)
                        except Exception as e_json:
                            print(f"Error parseando chunk de fin: {e_json}")
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                import json
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def api_chat(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", session_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        # --- LÍMITE DE USO (PAYWALL) ---
        if user_id and user_id != "guest":
            from db import get_monthly_api_usage
            credits_used = get_monthly_api_usage(user_id)
            plan_tier = "gratis"
            
            profile = get_user_profile(user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                
            limit = 15 if plan_tier == "gratis" else (100 if plan_tier == "plus" else 999999)
            
            if credits_used >= limit:
                raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")
        # -------------------------------
                
        print(f"🔍 [DEBUG API CHAT] session_id={session_id}, user_id={user_id}")
        
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, "user", prompt)
        
        # Sincronizar form_data al health_profile si está vacío en la DB
        if form_data and user_id and user_id != "guest" and user_id != session_id:
            profile = get_user_profile(user_id)
            if profile:
                existing_hp = profile.get("health_profile") or {}
                if not existing_hp:
                    print(f"🔄 [SYNC] health_profile vacío, sincronizando desde formData del frontend...")
                    print(f"🔍 [SYNC] form_data a guardar: {list(form_data.keys())}")
                    result = update_user_health_profile(user_id, form_data)
                    print(f"🔍 [SYNC] Resultado update: {result}")
                    # Verificar que se guardó
                    verify = get_user_profile(user_id)
                    if verify:
                        print(f"✅ [SYNC] Verificación: health_profile ahora tiene {list((verify.get('health_profile') or {}).keys())}")
            else:
                # No existe el perfil, crear uno con upsert
                print(f"⚠️ [SYNC] No existe user_profile para {user_id}, intentando crear...")
                try:
                    from db import supabase as sb_client
                    if sb_client:
                        sb_client.table("user_profiles").upsert({
                            "id": user_id,
                            "health_profile": form_data
                        }).execute()
                        print(f"✅ [SYNC] Perfil creado con health_profile")
                except Exception as e:
                    print(f"❌ [SYNC] Error creando perfil: {e}")
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
        
        response_text, updated_fields, new_plan = chat_with_agent(session_id, prompt, current_plan=current_plan, user_id=user_id, form_data=form_data)
        
        save_message(session_id, "model", response_text)
        
        # 🧠 Background: Resumir y podar mensajes si el historial creció demasiado
        background_tasks.add_task(summarize_and_prune, session_id)
        
        if user_id and user_id != "guest" and user_id != session_id:
            log_api_usage(user_id, "gemini_chat")
        
        # === CONTEXTO PARA HECHOS (Debounce Semántico) ===
        # Obtenemos el historial de la sesión para darle contexto al LLM extractor
        from db import get_session_messages
        raw_history = get_session_messages(session_id)
        recent_history_str = ""
        if raw_history:
            # Tomar solo los últimos 6 mensajes para contexto rápido
            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
        
        # Verificar tier para usar la Memoria a Largo Plazo
        is_plus = False
        if user_id and user_id != "guest":
            profile = get_user_profile(user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                is_plus = plan_tier in ["plus", "admin", "ultra"]
        
        if is_plus:
            # 🧠 Background: Extraer hechos y vectorizarlos
            background_tasks.add_task(async_extract_and_save_facts, user_id, prompt, recent_history_str)
        else:
            print("INFO: Memoria a Largo Plazo deshabilitada para usuario Gratis.")
        
        # 🧠 Background: Generar un título si es el primer mensaje
        background_tasks.add_task(generate_chat_title_background, session_id, prompt)
        
        result = {"response": response_text, "updated_fields": updated_fields}
        if new_plan:
            result["new_plan"] = new_plan
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diary/upload")
async def api_diary_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("guest"),
    session_id: str = Form(None),
    verified_user_id: str = Depends(get_verified_user_id)
):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        # --- LÍMITE DE USO (PAYWALL) ---
        actual_user_id = user_id if user_id != "guest" else session_id
        if actual_user_id and actual_user_id != session_id:
            from db import get_monthly_api_usage
            credits_used = get_monthly_api_usage(actual_user_id)
            plan_tier = "gratis"
            
            profile = get_user_profile(actual_user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                
            limit = 15 if plan_tier == "gratis" else (100 if plan_tier == "plus" else 999999)
            
            if credits_used >= limit:
                raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")
        # -------------------------------

        from vision_agent import process_image_with_vision
        from db import supabase
        
        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        file_bytes = b""
        while chunk := await file.read(1024 * 1024):
            file_bytes += chunk
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="La imagen es demasiado grande. Máximo 20MB permitidos.")
        
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        actual_user_id = user_id if user_id != "guest" else session_id
        unique_filename = f"{actual_user_id}/{uuid.uuid4().hex}.{file_ext}"
        
        image_url = ""
        upload_success = False

        # 1. Intentar subir a Supabase Storage
        if supabase:
            try:
                res = await asyncio.to_thread(
                    supabase.storage.from_("visual_diary_images").upload,
                    path=unique_filename,
                    file=file_bytes,
                    file_options={"content-type": file.content_type}
                )
                image_url = supabase.storage.from_("visual_diary_images").get_public_url(unique_filename)
                upload_success = True
                print(f"☁️ Imagen guardada en Supabase: {image_url}")
            except Exception as sb_err:
                print(f"⚠️ Error subiendo a Supabase (¿Existe el bucket 'visual_diary_images'?): {sb_err}")
                upload_success = False

        # 2. Si no se pudo subir a Supabase, fallar (evitar guardar localmente en la nube)
        if not upload_success:
            print("❌ No se pudo subir la imagen a Supabase. Abortando.")
            raise HTTPException(status_code=500, detail="Error uploading image to cloud storage.")
            
            
        # 3. Procesar imagen con Visión SINCRÓNICAMENTE
        print("\n-------------------------------------------------------------")
        print("📸 [VISION AGENT] Procesando nueva imagen subida...")
        vision_result = await process_image_with_vision(file_bytes)
        
        description = vision_result.get("description", "No se pudo analizar la imagen.")
        is_food = vision_result.get("is_food", False)
        
        if is_food:
            print(f"✅ Descripción generada: '{description}'")
            
            if actual_user_id and actual_user_id != "guest" and actual_user_id != session_id:
                from db import log_api_usage
                log_api_usage(actual_user_id, "gemini_vision")
                
            # 4. Guardar en DB en segundo plano (embedding + insert)
            background_tasks.add_task(
                _save_visual_entry_background,
                actual_user_id, image_url, description
            )
        else:
            print("➡️ La imagen fue ignorada porque no se detectaron alimentos.")
        
        return {
            "success": True, 
            "is_food": is_food,
            "description": description,
            "image_url": image_url
        }
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/diary/upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _save_visual_entry_background(user_id: str, image_url: str, description: str):
    """Background task: genera embedding y guarda en la tabla visual_diary."""
    from vision_agent import get_multimodal_embedding
    from db import save_visual_entry
    
    embedding = get_multimodal_embedding(description)
    if embedding:
        print(f"📦 Guardando entrada visual en la DB (Vector 768d)...")
        save_visual_entry(user_id=user_id, image_url=image_url, description=description, embedding=embedding)
        print("✅ ¡Imagen registrada en el Diario Visual con éxito!")
    else:
        print("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")

@app.get("/api/shopping/custom/{user_id}")
def api_get_custom_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene los items custom de la lista de compras añadidos por la IA."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"items": []}
        items = get_custom_shopping_items(user_id)
        return {"items": items}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/shopping/custom GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/shopping/custom/{item_id}")
def api_delete_custom_shopping_item(item_id: str):
    """Elimina un item custom de la lista de compras."""
    try:
        from db import delete_custom_shopping_item
        delete_custom_shopping_item(item_id)
        return {"success": True, "message": "Item eliminado de la lista."}
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/shopping/custom DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diary/consumed/{user_id}")
def api_get_consumed_today(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene las métricas agregadas de las comidas registradas en el día por la IA."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"meals": [], "totals": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}}
        
        from db import get_consumed_meals_today
        meals = get_consumed_meals_today(user_id)
        
        total_cal = sum(m.get("calories", 0) for m in meals)
        total_pro = sum(m.get("protein", 0) for m in meals)
        total_car = sum(m.get("carbs", 0) for m in meals)
        total_fat = sum(m.get("healthy_fats", 0) for m in meals)
        
        return {
            "meals": meals,
            "totals": {
                "calories": total_cal,
                "protein": total_pro,
                "carbs": total_car,
                "healthy_fats": total_fat
            }
        }
    except Exception as e:
        print(f"❌ [ERROR] Error en /api/diary/consumed GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3001, reload=True)