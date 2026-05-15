from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from error_utils import safe_error_detail
from typing import Optional
import logging
import traceback
import json

from auth import get_verified_user_id, verify_api_quota
from path_validators import assert_valid_uuid
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
# [P1-BG-THREAD-TIMEOUT · 2026-05-15] SSOT para fire-and-forget con timeout
# duro + alert. Reemplaza los `threading.Thread(target=..., daemon=True).start()`
# que vivían inline en este router. Ver `backend/bg_executor.py`.
from bg_executor import submit_bg_task

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

@router.get("/sessions/{user_id}")
def api_get_chat_sessions(user_id: str, session_ids: Optional[str] = None, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(user_id, allow_guest=True)
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/sessions/{user_id}")
def api_delete_chat_sessions(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(user_id, allow_guest=True)
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
            delete_user_agent_sessions(user_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


from pydantic import BaseModel
class RenameSessionReq(BaseModel):
    title: str


@router.put("/session/{session_id}")
def api_rename_chat_session(session_id: str, data: RenameSessionReq, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(session_id)
        session_owner = get_session_owner(session_id)
        if session_owner and session_owner != "guest":
            if not verified_user_id or verified_user_id != session_owner:
                raise HTTPException(status_code=403, detail="Prohibido.")
        update_session_title(session_id, data.title)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/session PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/history/{session_id}")
def api_get_chat_history(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(session_id)
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/session/{session_id}")
def api_delete_chat_session(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Elimina una sesión de chat. Requiere autenticación pero sin validación IDOR
    (RLS desactivado — la auth se maneja aquí)."""
    from db import delete_chat_session
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(session_id)
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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

from fastapi.responses import StreamingResponse, Response
import asyncio
import os
import httpx

# [P1-CHAT-TTS-1 · 2026-05-11] Rate limiter para el TTS proxy. 60 calls/min
# por user_id autenticado (IP fallback para anon — pero el endpoint rechaza
# anons explícitamente abajo, así que el IP-bucket es defensa adicional contra
# burst pre-auth). Voice mode genera chunks ~1/seg, 60/min cubre conversación
# fluida sin pegarle al cap. Singleton módulo-level — mismo patrón que
# `_PLAN_GEN_LIMITER` en routers/plans.py.
_CHAT_TTS_LIMITER = RateLimiter(max_calls=60, period_seconds=60)

# [P1-CHAT-TTS-1 · 2026-05-11] Cap de longitud del texto enviado a
# ElevenLabs. La API factura por carácter; sin cap, un user (o un atacante
# autenticado) puede mandar 100kB y agotar el cupo del OWNER. Voice mode
# del frontend produce chunks cortos (<300 chars típico, response completa
# <1500). 1500 es safe upper bound; si en el futuro se requieren textos
# más largos, considerar streaming chunked TTS.
_CHAT_TTS_MAX_TEXT_CHARS = 1500


@router.post("/tts")
async def api_chat_tts(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_CHAT_TTS_LIMITER),
):
    """[P1-CHAT-TTS-1 · 2026-05-11] TTS proxy a ElevenLabs.

    ANTES (pre-fix):
      - Sin `Depends(...)`. Cualquiera con la URL podía POSTear texto
        arbitrario; el servidor reenviaba a ElevenLabs con la API key
        server-side. Vector: bot scraper + texto largo → cup del owner
        consumido en horas.
      - Sin cap de `len(text)`. Forwarding pasaba al request a ElevenLabs
        tal cual.
      - Sin `log_api_usage`. Cero accounting per-user de costo TTS.

    DESPUÉS:
      - `RateLimiter(max_calls=60, period_seconds=60)` requiere
        Authorization Bearer válido (`get_verified_user_id` injerida por
        el limiter). Rechazamos 401 si la cadena de auth no resolvió.
      - `len(text) <= _CHAT_TTS_MAX_TEXT_CHARS` (1500). 413 si excede.
      - `log_api_usage(verified_user_id, "elevenlabs_tts")` por call —
        permite SRE auditar costo per-user, detectar bursts anómalos, y
        es el path correcto para un futuro paywall TTS (hoy NO bypassea
        `verify_api_quota` porque TTS es UX feature, NO genera plan; pero
        el registro queda para reasoning).

    Tooltip-anchor: P1-CHAT-TTS-1-AUTH
    """
    if not verified_user_id:
        # `RateLimiter` resuelve el bucket por IP cuando no hay auth, pero
        # NO rechaza la request. Para TTS necesitamos auth obligatoria
        # (cost-bearing endpoint).
        raise HTTPException(status_code=401, detail="Authentication required for TTS.")

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Missing text")
    text = text.strip()
    if len(text) > _CHAT_TTS_MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"TTS text exceeds {_CHAT_TTS_MAX_TEXT_CHARS} chars (got {len(text)}).",
        )

    # Load and strip the API key to prevent whitespace or quotation errors
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API Key no configurada.")

    # Voz "Rachel" predeterminada (fuerte en inglés/multilingüe) o equivalente
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL") # Bella
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    # [P1-TTS-FINALLY-LOG · 2026-05-15] Billing idempotente vía finally — mismo
    # patrón que P2-AUDIT-NEW-2 cerró para `/chat/stream`.
    #
    # ANTES (P1-CHAT-TTS-1 original):
    #   `log_api_usage` vivía DENTRO del `async with httpx.AsyncClient` solo en
    #   path de éxito. Si la request a ElevenLabs lanzaba `TimeoutError`,
    #   `httpx.HTTPStatusError`, `httpx.ConnectError`, etc, el accounting NO
    #   se ejecutaba. ElevenLabs factura POR CARÁCTER SUBMITIDO (no por
    #   response devuelto): si la request llegó al servidor antes del timeout,
    #   ellos cobraron al owner y el usuario quedó sin charge en su cuota
    #   mensual. Vector: usuario malicioso provoca timeouts deliberados (URL
    #   bloqueada en cliente / network throttling) → tokens TTS gastados sin
    #   cobrar a cuota.
    #
    # FIX:
    #   - `_billed` flag dedupea (defensivo, finally corre una sola vez).
    #   - `_request_started` activa cuando entramos al `async with` httpx
    #     (api_key validado, url construida). Si fallamos ANTES de eso
    #     (auth, missing api_key), finally NO factura — esos errores no
    #     consumieron crédito en ElevenLabs.
    #   - Timeout-specific catch emite `pipeline_metrics` con
    #     `node='tts_timeout'` para que SRE pueda graficar la incidencia.
    _tts_billed = False
    _tts_request_started = False

    try:
        _tts_request_started = True
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return Response(content=resp.content, media_type="audio/mpeg")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error ElevenLabs {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=500, detail="Error en generación TTS")
    except (httpx.TimeoutException, asyncio.TimeoutError) as _e_timeout:
        logger.error(f"[P1-TTS-FINALLY-LOG] Timeout ElevenLabs ({_e_timeout!r})")
        try:
            from db_core import execute_sql_write
            import json as _json_tts
            execute_sql_write(
                """
                INSERT INTO pipeline_metrics
                    (user_id, session_id, node, duration_ms, retries,
                     tokens_estimated, confidence, metadata)
                VALUES (%s, NULL, %s, 0, 0, %s, 0, %s::jsonb)
                """,
                (
                    verified_user_id,
                    "tts_timeout",
                    len(text),
                    _json_tts.dumps({
                        "provider": "elevenlabs",
                        "timeout_threshold_s": 15.0,
                        "char_count": len(text),
                    }, ensure_ascii=False),
                ),
            )
        except Exception as _tick_err:
            logger.debug(f"[P1-TTS-FINALLY-LOG] tick timeout falló: {_tick_err}")
        raise HTTPException(status_code=504, detail="Timeout en generación TTS")
    except Exception as e:
        logger.error(f"Error general llamando a ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en generación TTS")
    finally:
        # [P1-TTS-FINALLY-LOG · 2026-05-15] Cobrar accounting siempre que
        # se haya iniciado la request a ElevenLabs, sin importar success/
        # error/timeout. Best-effort try/except: un fallo de DB no debe
        # tumbar la response del usuario.
        if _tts_request_started and not _tts_billed and verified_user_id:
            try:
                log_api_usage(verified_user_id, "elevenlabs_tts")
                _tts_billed = True
            except Exception as _audit_err:
                logger.warning(f"[P1-TTS-FINALLY-LOG] log_api_usage tts falló: {_audit_err}")

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
        is_call_mode = data.get("is_call_mode", False)
        
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
        
        plan_tier = "gratis"
        if user_id and user_id != "guest":
            profile_sync = get_user_profile(user_id)
            if profile_sync:
                plan_tier = profile_sync.get("plan_tier", "gratis")
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            
        
        # Iniciar generación del título de inmediato en paralelo
        # [P1-BG-THREAD-TIMEOUT · 2026-05-15] submit al pool compartido con
        # timeout + alert si excede (Gemini cuelga, etc.). Ver bg_executor.py.
        submit_bg_task(
            generate_chat_title_background,
            user_id, session_id, prompt,
            task_name="chat_title_generation",
        )
        
        # [P2-AUDIT-NEW-2 · 2026-05-12] Billing idempotente vía flag + finally.
        # ANTES: `log_api_usage(user_id, "gemini_chat")` vivía DENTRO de
        # `bg_tasks()` que solo se invocaba en path `type=="done"`. Si el
        # SSE se abortaba a mitad (Ctrl+C, cerrar tab, AbortController,
        # network drop) o lanzaba excepción mid-stream, el LLM YA había
        # consumido tokens reales (chunks de texto emitidos) pero la
        # quota mensual del usuario NO se decrementaba.
        #
        # Vector de explotación: usuario malicioso aborta cada SSE
        # deliberadamente tras recibir el 80% útil del output → tokens
        # gastados del owner sin cobrar al user. Mismo gap que P2-LIVE-7
        # cerró para 5 endpoints pero `/chat/stream` quedó fuera.
        #
        # Fix:
        #   - `_billed` flag dedupea (defensivo, finally corre una sola vez).
        #   - `_chunk_observed` se activa cuando llega el primer chunk
        #     `type=="chunk"` (texto del LLM principal). Eso evita facturar
        #     si solo se enviaron `progress`/`sentiment` (preamble fast
        #     antes del LLM principal; sentiment usa modelo separado de
        #     costo marginal — no justifica cobrar quota completa).
        #   - `finally` cobra una vez tras done OK, abort, o exception
        #     mid-stream — todos paths donde el LLM ya consumió tokens.
        _billed = False
        _chunk_observed = False

        def event_generator():
            nonlocal _billed, _chunk_observed
            try:
                for chunk in chat_with_agent_stream(
                    session_id=session_id,
                    prompt=prompt,
                    current_plan=current_plan,
                    user_id=user_id,
                    form_data=form_data,
                    local_date=local_date,
                    tz_offset=tz_offset,
                    is_call_mode=is_call_mode,
                    plan_tier=plan_tier
                ):
                    yield chunk

                    # Interceptar el evento 'done' para lanzar background tasks
                    if chunk.startswith("data: "):
                        try:
                            data_obj = json.loads(chunk[len("data: "):].strip())
                            _chunk_type = data_obj.get("type")

                            # [P2-AUDIT-NEW-2] Marcar consumo de tokens. Solo
                            # `type=="chunk"` (texto streaming del LLM principal)
                            # cuenta como tokens reales. `progress`/`sentiment`/
                            # `error` no justifican facturar la cuota.
                            if _chunk_type == "chunk":
                                _chunk_observed = True

                            if _chunk_type == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    save_message(session_id, "model", response_text)
                                    # `done` con response no-vacío también garantiza
                                    # consumo de tokens incluso si por alguna razón
                                    # los chunks intermedios no se observaron.
                                    _chunk_observed = True

                                # Lógica Background (resumir, embeddings).
                                # [P2-AUDIT-NEW-2] log_api_usage SE MOVIÓ al finally
                                # — no va aquí. bg_tasks ahora solo cubre summarization
                                # + facts extraction.
                                def bg_tasks():
                                    try:
                                        raw_history = get_session_messages(session_id)
                                        recent_history_str = ""
                                        if raw_history:
                                            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])

                                        is_plus = plan_tier in ["basic", "plus", "admin", "ultra"]

                                        # [LONG-TERM-MEMORY-TOGGLE · 2026-05-13]
                                        # Además del gate de tier, respetar el flag user-controlled.
                                        # Default TRUE si el perfil no expone el campo (legacy).
                                        ltm_enabled = True
                                        if is_plus and user_id and user_id != "guest":
                                            try:
                                                _p = get_user_profile(user_id)
                                                if _p and "long_term_memory_enabled" in _p:
                                                    ltm_enabled = bool(_p.get("long_term_memory_enabled", True))
                                            except Exception:
                                                ltm_enabled = True

                                        if is_plus and ltm_enabled:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)

                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        logger.error(f"Error en bg tasks: {inner_e}")

                                # [P1-BG-THREAD-TIMEOUT · 2026-05-15] submit
                                # al pool compartido con timeout + alert.
                                submit_bg_task(bg_tasks, task_name="chat_sse_bg_tasks")
                        except Exception as e_json:
                            logger.error(f"Error parseando chunk de fin: {e_json}")

            except GeneratorExit:
                # [P2-AUDIT-NEW-2] Cliente cerró el SSE (AbortController,
                # tab close, network drop). NO re-emite chunks (la conexión
                # ya está muerta) pero el finally SÍ cobra si chunk_observed.
                # `raise` propaga para que StreamingResponse cierre limpio.
                logger.info(
                    f"[P2-AUDIT-NEW-2] SSE abortado por cliente "
                    f"session={session_id} chunk_observed={_chunk_observed}"
                )
                raise
            except Exception as e:
                # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
                logger.exception(f"[CHAT STREAM] Error mid-stream: {e}")
                # `chunk_observed` puede ser True (excepción tras emitir chunks)
                # o False (excepción pre-LLM). El finally factura solo si True.
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # [P2-AUDIT-NEW-2] Billing idempotente. Cubre TODOS los exits:
                # done OK, GeneratorExit (abort), exception mid-stream.
                # Solo factura si:
                #   (a) Aún no se cobró (`_billed` flag, defensivo).
                #   (b) El LLM emitió al menos un chunk de texto.
                #   (c) Usuario autenticado (no guest, no session-only).
                if (
                    not _billed
                    and _chunk_observed
                    and user_id
                    and user_id != "guest"
                    and user_id != session_id
                ):
                    try:
                        log_api_usage(user_id, "gemini_chat")
                        _billed = True
                    except Exception as _bill_err:
                        logger.warning(
                            f"[P2-AUDIT-NEW-2] log_api_usage falló "
                            f"(best-effort): {_bill_err}"
                        )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"[CHAT STREAM] Error en api_chat_stream: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))



@router.post("")
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
        # [LONG-TERM-MEMORY-TOGGLE · 2026-05-13] Además del tier, respetar el flag
        # `long_term_memory_enabled` controlado por el usuario desde Settings.
        # Default TRUE si el campo no existe (perfil legacy pre-migración).
        ltm_enabled = True
        if user_id and user_id != "guest":
            profile = get_user_profile(user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                is_plus = plan_tier in ["basic", "plus", "admin", "ultra"]
                if "long_term_memory_enabled" in profile:
                    ltm_enabled = bool(profile.get("long_term_memory_enabled", True))

        if is_plus and ltm_enabled:
            # 🧠 Background: Extraer hechos y vectorizarlos
            background_tasks.add_task(async_extract_and_save_facts, user_id, prompt, recent_history_str)
        elif is_plus and not ltm_enabled:
            logger.info(f"[LONG-TERM-MEMORY-TOGGLE] Captura pausada por user toggle (user={user_id}).")
        else:
            logger.info("INFO: Memoria a Largo Plazo deshabilitada para usuario Gratis.")
        
        # 🧠 Background: Generar un título si es el primer mensaje
        background_tasks.add_task(generate_chat_title_background, user_id, session_id, prompt)
        
        result = {"response": response_text, "updated_fields": updated_fields}
        if new_plan:
            result["new_plan"] = new_plan
        return result
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"[CHAT] Error en api_chat: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

