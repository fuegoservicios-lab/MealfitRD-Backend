from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
import traceback
import json

from auth import get_verified_user_id, verify_api_quota
from rate_limiter import RateLimiter
from db import (
    get_user_chat_sessions, get_guest_chat_sessions, get_session_owner, delete_user_agent_sessions,
    delete_single_agent_session, update_session_title, get_session_messages, get_or_create_session,
    save_message, save_message_feedback, log_api_usage
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import generate_chat_title_background, chat_with_agent, chat_with_agent_stream
from services import merge_form_data_with_profile
from db_profiles import get_user_profile
from db_plans import get_latest_meal_plan
from fact_extractor import async_extract_and_save_facts
import threading

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

@router.get("/sessions/{user_id}")
def api_get_chat_sessions(user_id: str, session_ids: Optional[str] = None, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
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
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{user_id}")
def api_delete_chat_sessions(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
            delete_user_agent_sessions(user_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel
class RenameSessionReq(BaseModel):
    title: str


@router.put("/session/{session_id}")
def api_rename_chat_session(session_id: str, data: RenameSessionReq, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        session_owner = get_session_owner(session_id)
        if session_owner and session_owner != "guest":
            if not verified_user_id or verified_user_id != session_owner:
                raise HTTPException(status_code=403, detail="Prohibido.")
        update_session_title(session_id, data.title)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/session PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
def api_get_chat_history(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # 🛡️ Validación IDOR: Verificar que el session pertenece al usuario autenticado
        session_owner = get_session_owner(session_id)
        if session_owner and session_owner != "guest":
            if not verified_user_id or verified_user_id != session_owner:
                logger.warning(f"🚫 [HISTORY AUTH FAILED] REJECTED. owner={session_owner} != verified={verified_user_id}")
                raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

        messages = get_session_messages(session_id)
        # Ocultar mensajes de sistema como el system_title
        filtered_messages = [m for m in messages if not m.get("content", "").startswith("[SYSTEM_TITLE]")]
        return {"messages": filtered_messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/history GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
def api_delete_chat_session(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Elimina una sesión de chat. Requiere autenticación pero sin validación IDOR 
    (RLS desactivado — la auth se maneja aquí)."""
    from db import delete_chat_session
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="Token requerido para eliminar chats.")
        
        success, error_msg = delete_chat_session(session_id)
        if success:
            logger.info(f"🗑️ Chat {session_id} eliminado por usuario {verified_user_id}")
            return {"success": True, "message": "Chat eliminado correctamente."}
        else:
            logger.error(f"❌ Fallo al eliminar chat {session_id}: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Error: {error_msg}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en DELETE chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
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


@router.post("/feedback")
async def api_chat_feedback(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    session_id = data.get("session_id")
    content = data.get("content")
    feedback = data.get("feedback")
    
    if not session_id or not content:
        raise HTTPException(status_code=400, detail="Missing session_id or content")

    # Asegurarnos de que exista la sesión en la base de datos antes de guardar feedback
    from db import get_or_create_session
    await asyncio.to_thread(get_or_create_session, session_id, user_id=verified_user_id)

    success = await asyncio.to_thread(save_message_feedback, session_id, content, feedback)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="Error saving feedback")


@router.post("/stream")
def api_chat_stream(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", session_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        local_date = data.get("local_date", None)
        tz_offset = data.get("tz_offset", None)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        logger.info(f"🔍 [DEBUG API CHAT STREAM] session_id={session_id}, user_id={user_id}")
        
        # Operaciones síncronas directas (ya estamos en un threadpool worker)
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, "user", prompt)
        
        # Handle form_data: merge frontend data with DB health_profile (DRY — shared in services.py)
        form_data = merge_form_data_with_profile(
            user_id if user_id != "guest" and user_id != session_id else "",
            form_data
        )
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            
        
        # Iniciar generación del título de inmediato en paralelo
        threading.Thread(
            target=generate_chat_title_background,
            args=(user_id, session_id, prompt),
            daemon=True
        ).start()
        
        def event_generator():
            try:
                for chunk in chat_with_agent_stream(
                    session_id=session_id, 
                    prompt=prompt, 
                    current_plan=current_plan, 
                    user_id=user_id, 
                    form_data=form_data,
                    local_date=local_date,
                    tz_offset=tz_offset
                ):
                    yield chunk
                    
                    # Interceptar el evento 'done' para lanzar background tasks
                    if chunk.startswith("data: "):
                        try:
                            data_obj = json.loads(chunk[len("data: "):].strip())
                            if data_obj.get("type") == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    save_message(session_id, "model", response_text)
                                    
                                # Lógica Background (resumir, uso de API, embeddings)
                                def bg_tasks():
                                    if user_id and user_id != "guest" and user_id != session_id:
                                        log_api_usage(user_id, "gemini_chat")
                                        
                                    try:
                                        raw_history = get_session_messages(session_id)
                                        recent_history_str = ""
                                        if raw_history:
                                            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
                                        
                                        is_plus = False
                                        if user_id and user_id != "guest":
                                            profile_sync = get_user_profile(user_id)
                                            if profile_sync:
                                                plan_tier_sync = profile_sync.get("plan_tier", "gratis")
                                                is_plus = plan_tier_sync in ["basic", "plus", "admin", "ultra"]
                                                
                                        if is_plus:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)
                                            
                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        logger.error(f"Error en bg tasks: {inner_e}")
                                
                                threading.Thread(target=bg_tasks, daemon=True).start()
                        except Exception as e_json:
                            logger.error(f"Error parseando chunk de fin: {e_json}")
                            
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/")
def api_chat(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_api_quota)):
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
                
        logger.info(f"🔍 [DEBUG API CHAT] session_id={session_id}, user_id={user_id}")
        
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, "user", prompt)
        
        # Handle form_data: merge frontend data with DB health_profile (DRY — shared in services.py)
        form_data = merge_form_data_with_profile(
            user_id if user_id != "guest" and user_id != session_id else "",
            form_data
        )
        
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
                is_plus = plan_tier in ["basic", "plus", "admin", "ultra"]
        
        if is_plus:
            # 🧠 Background: Extraer hechos y vectorizarlos
            background_tasks.add_task(async_extract_and_save_facts, user_id, prompt, recent_history_str)
        else:
            logger.info("INFO: Memoria a Largo Plazo deshabilitada para usuario Gratis.")
        
        # 🧠 Background: Generar un título si es el primer mensaje
        background_tasks.add_task(generate_chat_title_background, user_id, session_id, prompt)
        
        result = {"response": response_text, "updated_fields": updated_fields}
        if new_plan:
            result["new_plan"] = new_plan
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

