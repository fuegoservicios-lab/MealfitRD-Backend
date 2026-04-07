from functools import lru_cache as _lru_cache
import json
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write

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
    try:
        logger.info(f"📋 [SESSION] get_or_create_session(id={session_id}, user_id={user_id})")
        query = "SELECT * FROM agent_sessions WHERE id = %s"
        res = execute_sql_query(query, (session_id,), fetch_one=True)
        
        if res:
            existing_session = res
            logger.info(f"📋 [SESSION] Sesión {session_id} ya existe. user_id en DB: {existing_session.get('user_id')}")
            if not existing_session.get("user_id") and user_id:
                try:
                    update_q = "UPDATE agent_sessions SET user_id = %s WHERE id = %s RETURNING *"
                    update_res = execute_sql_write(update_q, (user_id, session_id), returning=True)
                    if update_res:
                        logger.info(f"✅ [SESSION] user_id actualizado a {user_id} para sesión {session_id}")
                        return update_res[0]
                except Exception as update_e:
                    logger.error(f"Error actualizando user_id en sesión: {update_e}")
            return existing_session
        
        logger.info(f"📋 [SESSION] Sesión {session_id} NO existe. Creando nueva...")
        if user_id:
            insert_q = "INSERT INTO agent_sessions (id, user_id, locked_at) VALUES (%s, %s, %s) RETURNING *"
            new_res = execute_sql_write(insert_q, (session_id, user_id, None), returning=True)
        else:
            insert_q = "INSERT INTO agent_sessions (id, locked_at) VALUES (%s, %s) RETURNING *"
            new_res = execute_sql_write(insert_q, (session_id, None), returning=True)
        
        logger.info(f"✅ [SESSION] Sesión creada: {new_res}")
        return new_res[0] if new_res else None
    except Exception as e:
        logger.info(f"Fallback creando sesión: {e}")
        try:
            insert_q = "INSERT INTO agent_sessions (id, locked_at) VALUES (%s, %s) RETURNING *"
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

def delete_chat_session(session_id: str) -> Tuple[bool, str]:
    """Elimina una sesión de chat y todos sus mensajes y resúmenes asociados.
    Usa SQL directo (psycopg) para bypassear completamente el REST API de Supabase."""
    try:
        # Borramos en orden para respetar Foreign Key constraints
        execute_sql_write("DELETE FROM conversation_summaries WHERE session_id = %s", (session_id,))
        execute_sql_write("DELETE FROM agent_messages WHERE session_id = %s", (session_id,))
        execute_sql_write("DELETE FROM agent_sessions WHERE id = %s", (session_id,))
        logger.info(f"🗑️ [DB] Sesión {session_id} eliminada exitosamente (SQL directo)")
        return True, ""
    except Exception as e:
        logger.error(f"❌ [DB] Error eliminando la sesión {session_id}: {e}", exc_info=True)
        return False, str(e)

def save_message(session_id: str, role: str, content: str):
    if not supabase: return None
    supabase.table("agent_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

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

def get_recent_messages(session_id: str, limit: int = 10):
    """Obtiene solo los N mensajes más recientes de una sesión (los no resumidos)."""
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
    Obtiene los rechazos activos (últimos 7 días).
    Después de 7 días, el rechazo expira y la IA puede volver a sugerir esa comida.
    """
    if not supabase: return []
    from datetime import datetime, timedelta, timezone
    
    # Calcular la fecha límite (hace 7 días)
    one_week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    query = supabase.table("meal_rejections").select("meal_name, meal_type, rejected_at")
    
    # Filtrar por user_id o session_id
    if user_id:
        query = query.eq("user_id", user_id)
    elif session_id:
        query = query.eq("session_id", session_id)
    else:
        return []
    
    # Solo rechazos de los últimos 7 días
    query = query.gte("rejected_at", one_week_ago)
    
    res = query.order("rejected_at", desc=True).execute()
    return res.data

