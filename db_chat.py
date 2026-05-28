from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write
# [P2-CHAT-SAVE-MSG-RETRY · 2026-05-19] Tenacity para retry exponencial
# en `save_message`. Ya es dep declarada (requirements.txt:11 — tenacity==9.1.4)
# y usada en todo el repo para retries de DB y LLM transients.
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

def delete_user_agent_sessions(user_id: str) -> bool:
    """Elimina todas las sesiones de agente para un usuario."""
    if not supabase: return False
    try:
        supabase.table("agent_sessions").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error eliminando sesiones de agente de {user_id}: {e}")
        return False

def delete_single_agent_session(session_id: str) -> bool:
    """Elimina una sesión específica de agente."""
    if not supabase: return False
    try:
        supabase.table("agent_sessions").delete().eq("id", session_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error eliminando sesion de agente {session_id}: {e}")
        return False

def update_session_title(session_id: str, new_title: str) -> bool:
    """Actualiza el título de una sesión."""
    if not supabase: return False
    try:
        # Buscar si ya existe el mensaje de titulo
        res = supabase.table("agent_messages").select("id").eq("session_id", session_id).like("content", "[SYSTEM_TITLE] %").execute()
        if res.data and len(res.data) > 0:
            msg_id = res.data[0]["id"]
            supabase.table("agent_messages").update({"content": f"[SYSTEM_TITLE] {new_title}"}).eq("id", msg_id).execute()
        else:
            # Si no existe, insertar un mensaje especial al inicio (fecha muy antigua o la misma de la sesion)
            save_message(session_id, "model", f"[SYSTEM_TITLE] {new_title}")
        return True
    except Exception as e:
        logger.error(f"Error actualizando titulo de sesion {session_id}: {e}")
        return False

def get_or_create_session(session_id: str, user_id: str = None):
    if not connection_pool: 
        logger.error(f"🚨 [SESSION] connection_pool es None! No se puede crear sesión {session_id}")
        return None
    # [P3-SELECT-STAR-AGENT-SESSIONS · 2026-05-15] Columnas explícitas en lugar
    # de `SELECT *` / `RETURNING *`. Callers de `get_or_create_session` solo
    # consumen los 4 campos básicos del session (id, user_id, locked_at,
    # created_at). Convención post-P1-NEW-3 (explicit columns) cierra el
    # gap inconsistente entre módulos.
    _AGENT_SESSION_COLS = "id, user_id, locked_at, created_at"
    try:
        logger.info(f"📋 [SESSION] get_or_create_session(id={session_id}, user_id={user_id})")
        query = f"SELECT {_AGENT_SESSION_COLS} FROM agent_sessions WHERE id = %s"
        res = execute_sql_query(query, (session_id,), fetch_one=True)

        if res:
            existing_session = res
            logger.info(f"📋 [SESSION] Sesión {session_id} ya existe. user_id en DB: {existing_session.get('user_id')}")
            if not existing_session.get("user_id") and user_id:
                try:
                    update_q = (
                        f"UPDATE agent_sessions SET user_id = %s WHERE id = %s "
                        f"RETURNING {_AGENT_SESSION_COLS}"
                    )
                    update_res = execute_sql_write(update_q, (user_id, session_id), returning=True)
                    if update_res:
                        logger.info(f"✅ [SESSION] user_id actualizado a {user_id} para sesión {session_id}")
                        return update_res[0]
                except Exception as update_e:
                    logger.error(f"Error actualizando user_id en sesión: {update_e}")
            return existing_session

        logger.info(f"📋 [SESSION] Sesión {session_id} NO existe. Creando nueva...")
        if user_id:
            insert_q = (
                f"INSERT INTO agent_sessions (id, user_id, locked_at) VALUES (%s, %s, %s) "
                f"RETURNING {_AGENT_SESSION_COLS}"
            )
            new_res = execute_sql_write(insert_q, (session_id, user_id, None), returning=True)
        else:
            insert_q = (
                f"INSERT INTO agent_sessions (id, locked_at) VALUES (%s, %s) "
                f"RETURNING {_AGENT_SESSION_COLS}"
            )
            new_res = execute_sql_write(insert_q, (session_id, None), returning=True)

        logger.info(f"✅ [SESSION] Sesión creada: {new_res}")
        return new_res[0] if new_res else None
    except Exception as e:
        logger.info(f"Fallback creando sesión: {e}")
        try:
            insert_q = (
                f"INSERT INTO agent_sessions (id, locked_at) VALUES (%s, %s) "
                f"RETURNING {_AGENT_SESSION_COLS}"
            )
            new_res_fallback = execute_sql_write(insert_q, (session_id, None), returning=True)
            if new_res_fallback:
                return new_res_fallback[0]
            return None
        except Exception as inner_e:
            logger.error(f"Error fatal creando sesión: {inner_e}")
            return None

def get_session_owner(session_id: str) -> Optional[str]:
    """Retorna el user_id propietario de una sesión, o None si no existe/no tiene dueño.
    Usado para validación IDOR en endpoints de historial."""
    if not connection_pool: return None
    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = execute_sql_query(
                "SELECT user_id FROM agent_sessions WHERE id = %s",
                (session_id,), fetch_one=True
            )
            return str(res.get("user_id")) if res and res.get("user_id") else None
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.3)
                continue
            logger.error(f"Error consultando session owner: {e}")
            return None

def get_guest_chat_sessions(session_ids: list):
    """Obtiene la lista de sesiones para invitados, mediante sus IDs."""
    if not supabase or not session_ids: return []
    try:
        # Solo obtenemos hasta un límite prudente para evitar URLs muy largas
        res = supabase.table("agent_sessions").select("*").in_("id", session_ids[:20]).execute()
        sessions = res.data
        return _process_and_sort_sessions(sessions)
    except Exception as e:
        logger.error(f"Error en get_guest_chat_sessions: {e}")
        return []

def get_user_chat_sessions(user_id: str):
    """Obtiene la lista de sesiones, ordenadas por actividad reciente."""
    if not supabase: return[]
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Fallback de seguridad si no existe user_id en la base de datos
            res = supabase.table("agent_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(60).execute()
            sessions = res.data
            return _process_and_sort_sessions(sessions)
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            error_msg = str(e)
            if "42703" not in error_msg and "PGRST204" not in error_msg:
                logger.error(f"Error en getsessions: {e}")
            return []

def _process_and_sort_sessions(sessions: list):
    if not sessions: return []
    try:
        valid_sessions = []
        import re
        
        session_ids = [s["id"] for s in sessions]
        
        all_messages = []
        max_retries = 2
        for attempt in range(max_retries):
            try:
                msg_res = supabase.table("agent_messages").select("session_id, content, created_at, role").in_("session_id", session_ids).order("created_at", desc=False).execute()
                if msg_res and msg_res.data:
                    all_messages = msg_res.data
                break
            except Exception as db_e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(f"Error recuperando batch de mensajes para {len(session_ids)} sesiones: {db_e}")
        
        messages_by_session = {s_id: [] for s_id in session_ids}
        for m in all_messages:
            if "session_id" in m:
                messages_by_session[m["session_id"]].append(m)
                
        for s in sessions:
            s_msgs = messages_by_session[s["id"]]
            
            if s_msgs and len(s_msgs) > 0:
                title_msgs = [m for m in s_msgs if str(m.get("content", "")).startswith("[SYSTEM_TITLE] ")]
                user_msgs = [m for m in s_msgs if m.get("role") == "user"]
                
                if title_msgs:
                    s["title"] = str(title_msgs[0].get("content", "")).replace("[SYSTEM_TITLE] ", "")
                    s["is_fallback"] = False
                elif user_msgs:
                    content_str = str(user_msgs[0].get("content", ""))
                    # Limpiar el prefijo inyectado por el frontend con la hora, sistema y comandos
                    clean_str = re.sub(r'\[?\(Hora actual del usuario:[^)]*\)\]?', '', content_str, flags=re.IGNORECASE|re.DOTALL)
                    clean_str = re.sub(r'\[Sistema:[^\]]*\]', '', clean_str, flags=re.IGNORECASE)
                    clean_str = re.sub(r'Instrucción:.*?$', '', clean_str, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
                    clean_str = re.sub(r'\[IMAGE:[^\]]*\]', '', clean_str, flags=re.IGNORECASE)
                    
                    # SIEMPRE limpiar "Mensaje del usuario:" ya sea que haya imagen o no
                    clean_str = re.sub(r'Mensaje del usuario:\s*', '', clean_str, flags=re.IGNORECASE|re.DOTALL)
                    
                    if '[El usuario subió una imagen.' in clean_str:
                        clean_str = re.sub(r'\[El usuario subió una imagen\..+?\]', '', clean_str, flags=re.DOTALL)
                            
                    clean_str = clean_str.strip()
                    if not clean_str:
                        clean_str = "Interacción con imagen o sistema"
                        
                    if clean_str:
                        s["title"] = clean_str[:45] + ("..." if len(clean_str) > 45 else "")
                    else:
                        s["title"] = "Nuevo chat"
                    # Como no hay SYSTEM_TITLE, es fallback
                    s["is_fallback"] = True
                else:
                    s["title"] = "Nuevo Chat"
                    
                s["last_activity"] = s_msgs[-1].get("created_at", s.get("created_at", ""))
                valid_sessions.append(s)
            else:
                s["title"] = "Nuevo Chat"
                s["last_activity"] = s.get("created_at", "")
                valid_sessions.append(s)
                
        # Ordenar los chats para que el más reciente suba al primer lugar
        valid_sessions.sort(key=lambda x: x.get("last_activity") or x.get("created_at") or "1970-01-01T00:00:00", reverse=True)
        return valid_sessions
    except Exception as e:
        logger.error(f"Error en getsessions: {e}")
        return

def get_session_messages(session_id: str):
    """Obtiene todos los mensajes de una sesion, ordenados cronologicamente."""
    if not supabase: return []
    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = supabase.table("agent_messages").select("*").eq("session_id", session_id).order("created_at", desc=False).execute()
            return res.data
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.3)
                continue
            logger.error(f"Error get_session_messages: {e}")
            return []

def acquire_summarizing_lock(session_id: str) -> bool:
    """Intenta adquirir el bloqueo para resumir. Retorna True si lo logra, False si ya está bloqueado."""
    if not supabase: return True
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        
        # Verificar estado actual
        res = supabase.table("agent_sessions").select("locked_at").eq("id", session_id).execute()
        if res.data and len(res.data) > 0:
            locked_at_str = res.data[0].get("locked_at")
            if locked_at_str:
                try:
                    if locked_at_str.endswith("Z"):
                        locked_at_str = locked_at_str[:-1] + "+00:00"
                    locked_at = datetime.fromisoformat(locked_at_str)
                    # Si el lock tiene menos de 5 minutos, está bloqueado
                    if now - locked_at < timedelta(minutes=5):
                        return False
                except ValueError:
                    pass # Asumimos expirado si falla el parseo
                
        # Intentar establecer el timestamp
        update_res = supabase.table("agent_sessions").update({"locked_at": now.isoformat()}).eq("id", session_id).execute()
            
        return len(update_res.data) > 0
    except Exception as e:
        logger.error(f"Error acquiring summarizing lock: {e}")
        return False

def release_summarizing_lock(session_id: str):
    """Libera el bloqueo de resumen."""
    if not supabase: return
    try:
        supabase.table("agent_sessions").update({"locked_at": None}).eq("id", session_id).execute()
    except Exception as e:
        error_msg = str(e)
        if "Server disconnected" not in error_msg:
            logger.error(f"Error releasing summarizing lock: {e}")

def delete_chat_session(session_id: str, user_id: str) -> Tuple[bool, str]:
    """[P0-CHAT-DELETE-IDOR · 2026-05-26] Elimina sesión + chats + summaries.

    Valida ownership server-side antes de los DELETEs. Patrón simétrico al
    guard de `GET /api/chat/history/{session_id}` (routers/chat.py:266).

    Pre-fix (P2-AUDIT-HARDENING · 2026-05-25): la signature era
    `delete_chat_session(session_id)` sin user_id; el router solo validaba
    "está autenticado" pero NO "session.user_id == verified_user_id".
    Atacante autenticado podía DELETE conversation_summaries +
    agent_messages + agent_sessions ajenas pasando session_id enumerado.

    Retorna:
      - (True, "") si ownership validado y DELETEs ejecutados.
      - (False, "not_found") si la sesión no existe (404 en caller).
      - (False, "forbidden") si session.user_id != user_id (403 en caller).
      - (False, "<exc>") en error genérico (500 en caller).

    Defensa-en-profundidad: el DELETE de `agent_sessions` incluye
    `AND user_id = %s` adicional. Aunque el pre-check ya cubre TOCTTOU
    teórico, el predicate redundante cumple invariante I2 del repo.
    """
    try:
        owner = get_session_owner(session_id)
        if owner is None:
            return False, "not_found"
        if str(owner) != str(user_id):
            logger.warning(
                f"🚫 [P0-CHAT-DELETE-IDOR] REJECTED. session={session_id} "
                f"owner={owner} != caller={user_id}"
            )
            return False, "forbidden"

        # Ownership validado — orden respeta FK constraints.
        execute_sql_write("DELETE FROM conversation_summaries WHERE session_id = %s", (session_id,))
        execute_sql_write("DELETE FROM agent_messages WHERE session_id = %s", (session_id,))
        execute_sql_write(
            "DELETE FROM agent_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id),
        )
        logger.info(f"🗑️ [DB] Sesión {session_id} eliminada (owner={user_id})")
        return True, ""
    except Exception as e:
        logger.error(f"❌ [DB] Error eliminando la sesión {session_id}: {e}", exc_info=True)
        return False, str(e)

# [P2-CHAT-SAVE-MSG-RETRY · 2026-05-19] El INSERT a `agent_messages` ahora
# va envuelto en tenacity retry exponencial (3 intentos, base 0.5s,
# multiplier 2, max 4s). Vector cerrado: si Supabase tiene un blip
# transient mid-stream (network jitter, momentary 5xx, connection pool
# exhausto), pre-fix el `.execute()` levantaba la excepción → el caller
# en routers/chat.py la atrapaba como `Error en bg tasks` (warning) y
# el flujo seguía. El usuario veía la respuesta del agente RENDERIZADA
# en su pantalla pero la respuesta NO existía en `agent_messages` →
# mensaje fantasma. Al refrescar la sesión, el LLM perdía contexto del
# turno anterior y respondía como si nada hubiera pasado.
#
# Retry específico para excepciones de Supabase/Postgrest: NO atrapamos
# ValueError/KeyError (bug del caller, no transient). El `before_sleep_log`
# emite WARNING a logs para que SRE pueda graficar la frecuencia.
# Tooltip-anchor: P2-CHAT-SAVE-MSG-RETRY.
#
# [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] El helper ahora recibe `user_id`
# (puede ser None para guests) y lo persiste en la columna nueva
# `agent_messages.user_id`. Caller resuelve el user_id antes (explícito
# en routers/chat.py donde está disponible, o vía lookup defensivo en
# `save_message` para callsites legacy que no lo pasan).
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _save_message_insert_with_retry(
    session_id: str, role: str, content: str, user_id: Optional[str]
) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY · 2026-05-19] Helper aislado del INSERT
    para que tenacity sólo envuelva el roundtrip a Supabase. El side-effect
    `handle_nudge_response` queda FUERA — su falla no debe disparar
    re-INSERT (re-procesaría el nudge response múltiples veces).

    [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] `user_id` se persiste en la
    columna nueva (puede ser None para guests — legítimo). RLS policies
    `authenticated_*_own_messages` enforzan `auth.uid() = user_id` para
    callsites con anon/authenticated keys; service_role (que es lo que
    usa el backend) bypassea RLS.

    Tooltip-anchor: P2-CHAT-SAVE-MSG-RETRY + P1-CHAT-DB-USER-ID-RLS."""
    supabase.table("agent_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "user_id": user_id,
    }).execute()


def save_message(
    session_id: str,
    role: str,
    content: str,
    user_id: Optional[str] = None,
):
    """[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] `user_id` opcional — los
    callsites en routers/chat.py lo pasan explícitamente (ya está en
    scope post-auth); callsites legacy (db_plans.py, proactive_agent.py,
    services.py, etc.) no lo pasan y la función hace lookup vía
    `get_session_owner` como fallback. Tooltip-anchor: P1-CHAT-DB-USER-ID-RLS."""
    if not supabase: return None

    # [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Resolver user_id: prefer el
    # explícito (passed by caller post-auth); fallback al lookup por
    # session.owner para preserve backward compat. Guests legítimamente
    # quedan con None.
    if user_id is None:
        user_id = get_session_owner(session_id)

    if role == "user":
        if user_id:
            try:
                from proactive_agent import handle_nudge_response
                handle_nudge_response(user_id, content)
            except Exception as e:
                logger.error(f"Error procesando respuesta al nudge en save_message: {e}")

    # [P2-CHAT-SAVE-MSG-RETRY · 2026-05-19] INSERT envuelto en retry.
    # Si los 3 intentos fallan, la excepción se re-raises al caller — el
    # finally idempotente del SSE billing (P2-AUDIT-NEW-2) corre igual,
    # pero el caller decide cómo manejar el log. NO usamos `pass` silente:
    # 3 fallos consecutivos a Supabase es un incidente real, no transient.
    _save_message_insert_with_retry(session_id, role, content, user_id)

def save_message_feedback(session_id: str, content: str, feedback: str):
    """Guarda o remueve la retroalimentación (up/down/null) para un mensaje del modelo."""
    if not supabase: return False
    import logging
    logger = logging.getLogger(__name__)
    try:
        # feedback can be 'up', 'down', or None.
        logger.info(f"Intentando guardar feedback '{feedback}' para session '{session_id}'")
        
        # We use a robust matching strategy because long strings with newlines often fail exact match
        robust_prefix = content.strip()[:60]
        # Escape SQL LIKE special chars just in case
        robust_prefix = robust_prefix.replace("%", "").replace("_", "")
        search_pattern = robust_prefix + "%"
        
        res = supabase.table("agent_messages").update({
            "feedback": feedback
        }).eq("session_id", session_id).eq("role", "model").ilike("content", search_pattern).execute()
        
        rows_affected = len(res.data) if res.data else 0
        logger.info(f"Feedback guardado, rows affected: {rows_affected}")
        
        # Si aun así falla, intenta actualizar el ULTIMO mensaje del modelo en la sesion
        if rows_affected == 0:
            logger.warning("Fallo el match por sufijo. Actualizando el ultimo mensaje de modelo de esta sesion.")
            last_msg = supabase.table("agent_messages").select("id").eq("session_id", session_id).eq("role", "model").order("created_at", desc=True).limit(1).execute()
            if last_msg.data and len(last_msg.data) > 0:
                msg_id = last_msg.data[0]["id"]
                supabase.table("agent_messages").update({"feedback": feedback}).eq("id", msg_id).execute()
                rows_affected = 1
            else:
                logger.warning("No hay mensajes previos del modelo. Insertando el mensaje (posiblemente bienvenida autogenerada).")
                supabase.table("agent_messages").insert({
                    "session_id": session_id,
                    "role": "model",
                    "content": content,
                    "feedback": feedback
                }).execute()
                rows_affected = 1

        return rows_affected > 0
    except Exception as e:
        logger.error(f"Error saving message feedback: {e}")
        return False

def get_memory(session_id: str):
    if not supabase: return []
    res = supabase.table("agent_messages").select("*").eq("session_id", session_id).order("created_at", desc=False).execute()
    return res.data

def save_summary(session_id: str, summary: str, messages_start: str, messages_end: str, message_count: int):
    """Guarda un resumen de conversación en la tabla conversation_summaries."""
    if not supabase: return None
    res = supabase.table("conversation_summaries").insert({
        "session_id": session_id,
        "summary": summary,
        "messages_start": messages_start,
        "messages_end": messages_end,
        "message_count": message_count,
    }).execute()
    return res.data

def get_summaries(session_id: str):
    """Obtiene todos los resúmenes de una sesión, ordenados cronológicamente."""
    if not supabase: return []
    res = supabase.table("conversation_summaries").select("*").eq("session_id", session_id).order("messages_start", desc=False).execute()
    return res.data

def archive_summaries(summaries_list: list):
    """Guarda (archiva) una lista de resúmenes en cold storage antes de borrarlos."""
    if not supabase or not summaries_list: return None
    try:
        data_to_insert = []
        for s in summaries_list:
            data_to_insert.append({
                "session_id": s.get("session_id"),
                "summary": s.get("summary"),
                "messages_start": s.get("messages_start"),
                "messages_end": s.get("messages_end"),
                "message_count": s.get("message_count"),
                "original_created_at": s.get("created_at")
            })
        if data_to_insert:
            res = supabase.table("summary_archive").insert(data_to_insert).execute()
            return res.data
        return None
    except Exception as e:
        logger.error(f"Error archivando resúmenes en cold storage: {e}")
        return None

def search_deep_memory(user_id: str, query: str, limit: int = 5):
    """
    Busca en el archivo frío (summary_archive) los resúmenes históricos de un usuario.
    Hace un JOIN lógico: primero obtiene los session_ids del usuario,
    luego busca en summary_archive por esos session_ids con búsqueda textual.
    """
    if not supabase: return []
    try:
        # 1. Obtener todos los session_ids del usuario
        sessions_res = supabase.table("agent_sessions").select("id").eq("user_id", user_id).execute()
        if not sessions_res.data:
            return []
        
        session_ids = [s["id"] for s in sessions_res.data]
        
        # 2. Buscar en summary_archive por esos session_ids con texto parcial
        res = supabase.table("summary_archive") \
            .select("summary, messages_start, messages_end, message_count, original_created_at") \
            .in_("session_id", session_ids) \
            .ilike("summary", f"%{query}%") \
            .order("original_created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return res.data
    except Exception as e:
        logger.error(f"Error buscando en deep memory (summary_archive): {e}")
        return []

def delete_summaries(summary_ids: list):
    """Elimina múltiples resúmenes por sus IDs."""
    if not supabase: return None
    try:
        res = supabase.table("conversation_summaries").delete().in_("id", summary_ids).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando resúmenes: {e}")
        return None

def delete_old_messages(session_id: str, before_timestamp: str):
    """Elimina mensajes de agent_messages anteriores o iguales al timestamp dado."""
    if not supabase: return None
    res = supabase.table("agent_messages").delete().eq("session_id", session_id).lte("created_at", before_timestamp).execute()
    return res.data

# [P1-CHAT-SUPABASE-RETRY · 2026-05-20] Detector best-effort de errores
# transient de HTTP/2 al hablar con PostgREST de Supabase. `supabase-py`
# usa httpx con conexión HTTP/2 keep-alive a `*.supabase.co`. Kong/PostgREST
# cierra esas conexiones idle agresivamente (~30-60s); la PRIMER request
# post-idle suele lanzar `httpx.RemoteProtocolError: Server disconnected`,
# la SEGUNDA funciona porque httpx detecta el socket muerto y abre uno
# nuevo. La librería NO maneja este reconnect automáticamente.
#
# Detección por type-name + message para evitar importar httpx a este
# módulo (httpx es transitive dep de supabase-py, no first-class import
# aquí). Patrón análogo a `_is_rate_limit_error` en agent.py (P1-CHAT-LLM-429).
# Tooltip-anchor: P1-CHAT-SUPABASE-RETRY.
def _is_transient_supabase_http_error(exc: BaseException) -> bool:
    try:
        name = type(exc).__name__
        if name in (
            "RemoteProtocolError",
            "ConnectError",
            "ReadTimeout",
            "PoolTimeout",
            "ConnectTimeout",
        ):
            return True
        msg = str(exc).lower()
        if "server disconnected" in msg or "connection reset" in msg:
            return True
        return False
    except Exception:
        return False


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=1.5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def get_recent_messages(session_id: str, limit: int = 10):
    """Obtiene solo los N mensajes más recientes de una sesión (los no resumidos).

    [P1-CHAT-SUPABASE-RETRY · 2026-05-20] Retry 2 intentos con backoff 0.3-1.5s
    para cubrir el caso `httpx.RemoteProtocolError: Server disconnected`
    cuando Kong/PostgREST cierra la conexión HTTP/2 idle. La primera request
    post-idle falla pero httpx abre socket nuevo para la segunda. Tenacity
    re-raise tras agotar intentos para que callers (build_memory_context →
    chat_with_agent_stream) puedan propagar la excepción si persiste.

    Conservador: retry_if_exception_type=Exception es amplio pero stop=2
    limita el blast radius (max 1 reintento). Si el error es genuino (parse,
    schema mismatch), el segundo intento también falla y el caller lo sabe.
    """
    if not supabase: return []
    res = supabase.table("agent_messages").select("*").eq("session_id", session_id).order("created_at", desc=True).limit(limit).execute()
    # Revertir el orden para que estén cronológicos (el query los trae desc)
    return list(reversed(res.data)) if res.data else []

def insert_like(like_data: dict):
    if not supabase: return None
    res = supabase.table("meal_likes").insert(like_data).execute()
    return res.data

def get_user_likes(user_id: str):
    if not supabase: return []
    # Fetch all liked meals for this user
    res = supabase.table("meal_likes").select("meal_name, meal_type").eq("user_id", user_id).execute()
    return res.data

def insert_rejection(rejection_data: dict):
    """Guarda un rechazo de comida con timestamp automático."""
    if not supabase: return None
    res = supabase.table("meal_rejections").insert(rejection_data).execute()
    return res.data

def get_active_rejections(user_id: str = None, session_id: str = None):
    """
    Obtiene todos los rechazos permanentes del usuario.
    Una vez que el usuario rechaza un plato, nunca vuelve a sugerirse.
    """
    if not supabase: return []

    query = supabase.table("meal_rejections").select("meal_name, meal_type, rejected_at")

    if user_id:
        query = query.eq("user_id", user_id)
    elif session_id:
        query = query.eq("session_id", session_id)
    else:
        return []

    res = query.order("rejected_at", desc=True).execute()
    return res.data

