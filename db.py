import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from supabase import create_client, Client
from dotenv import load_dotenv

from constants import parse_ingredient_qty

logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# Configuración del ConnectionPool para psycopg (Usado por LangGraph PostgresSaver)
connection_pool = None
if SUPABASE_DB_URL:
    try:
        from psycopg_pool import ConnectionPool
        # Limpiar comillas basura por si acaso
        clean_url = SUPABASE_DB_URL.strip().strip("'").strip('"')
        def get_client_kwargs():
            return {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }

        connection_pool = ConnectionPool(
            conninfo=clean_url, 
            max_size=20, 
            kwargs=get_client_kwargs(),
            open=False
        )
        logger.info("🔌 [psycopg] ConnectionPool de Postgres configurado con keepalives.")
    except Exception as pool_err:
        logger.error(f"⚠️ [psycopg] Error configurando ConnectionPool: {pool_err}")

def close_connection_pool():
    if connection_pool:
        connection_pool.close()
        logger.info("Connection pool cerrado.")

# ============================================================
# SQL HELPERS (MIGRACIÓN REST -> PSYCOPG DIRECTO)
# ============================================================
def execute_sql_query(query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
    """Ejecuta una consulta SQL directa usando el pool de psycopg."""
    if not connection_pool:
        # Fallback de seguridad si el pool no estuviera disponible (aunque no debería)
        raise RuntimeError("db connection_pool is not available.")
    
    import psycopg
    from psycopg.rows import dict_row
    
    with connection_pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            return []

def execute_sql_write(query: str, params: tuple = None, returning: bool = False):
    """Ejecuta una transacción INSERT/UPDATE/DELETE."""
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available.")
    
    import psycopg
    from psycopg.rows import dict_row
    
    with connection_pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            if returning:
                return cursor.fetchall()
            return True

# ============================================================
# WRAPPERS PARA CONTROLADORES (Refactorización)
# ============================================================

def check_recent_meal_plan_exists(user_id: str, max_seconds: int = 30) -> bool:
    """Verifica si ya se ha guardado un plan para este usuario recientemente."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                if last_saved.endswith("Z"):
                    last_saved = last_saved[:-1] + "+00:00"
                last_saved = datetime.fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
                
            now_utc = datetime.now(timezone.utc)
            if (now_utc - last_saved).total_seconds() < max_seconds:
                return True
    except Exception as e:
        logger.warning(f"⚠️ [DEDUP] Error en check_recent_meal_plan_exists: {e}")
    return False

def check_meal_plan_generated_today(user_id: str) -> bool:
    """Valida si el último plan generado por el usuario se realizó el día actual."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                if last_saved.endswith("Z"): 
                    last_saved = last_saved[:-1] + "+00:00"
                last_saved = datetime.fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
            
            now_utc = datetime.now(timezone.utc)
            if last_saved.date() == now_utc.date():
                return True
    except Exception as e:
        logger.error(f"Error comprobando si plan fue hoy: {e}")
    return False

def save_new_meal_plan_robust(insert_data: dict) -> bool:
    """Guarda un nuevo plan nutricional con fallback por si faltan columnas optimizadas."""
    if not connection_pool: return False
    try:
        import copy
        from psycopg.types.json import Jsonb
        safe_data = copy.deepcopy(insert_data)
        
        def dict_to_insert(d):
            cols = list(d.keys())
            vals = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in d.values()]
            placeholders = ", ".join(["%s"] * len(cols))
            col_str = ", ".join(cols)
            return f"INSERT INTO meal_plans ({col_str}) VALUES ({placeholders})", vals
            
        query, vals = dict_to_insert(safe_data)
        execute_sql_write(query, tuple(vals))
        return True
    except Exception as try_db_e:
        err_msg = str(try_db_e)
        if "column" in err_msg and ("meal_names" in err_msg or "techniques" in err_msg):
            logger.warning("⚠️ [DB] Faltan columnas optimizadas (meal_names/techniques). Guardando sin ellas.")
            safe_data.pop("meal_names", None)
            safe_data.pop("ingredients", None)
            safe_data.pop("techniques", None)
            query, vals = dict_to_insert(safe_data)
            try:
                execute_sql_write(query, tuple(vals))
                return True
            except Exception as e2:
                raise e2
        else:
            raise try_db_e

def check_fact_ownership(fact_id: str, user_id: str) -> bool:
    """Verifica si un hecho corresponde a un determinado usuario."""
    if not connection_pool: return False
    try:
        query = "SELECT user_id FROM user_facts WHERE id = %s"
        res = execute_sql_query(query, (fact_id,), fetch_one=True)
        if res:
            return str(res.get("user_id")) == str(user_id)
    except Exception as e:
        logger.error(f"Error en check_fact_ownership: {e}")
    return False

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

def upsert_user_profile(user_id: str, health_profile: dict) -> bool:
    """Hace upsert del perfil de usuario y health_profile en user_profiles."""
    if not supabase: return False
    try:
        supabase.table("user_profiles").upsert({
            "id": user_id,
            "health_profile": health_profile
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error en upsert_user_profile: {e}")
        return False

# Pre-computar mapa de sinónimos sin acentos para track_meal_friction() (O(1) por llamada)
import unicodedata as _uc
from constants import GLOBAL_REVERSE_MAP as _GLOBAL_REVERSE_MAP, strip_accents as _strip_accents_canonical
_ACCENT_SAFE_REVERSE_MAP = {
    ''.join(c for c in _uc.normalize('NFD', k) if _uc.category(c) != 'Mn'): v
    for k, v in _GLOBAL_REVERSE_MAP.items()
}

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
    try:
        res = execute_sql_query(
            "SELECT user_id FROM agent_sessions WHERE id = %s",
            (session_id,), fetch_one=True
        )
        return str(res.get("user_id")) if res and res.get("user_id") else None
    except Exception as e:
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
    try:
        res = supabase.table("agent_messages").select("*").eq("session_id", session_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
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

def acquire_fact_lock(user_id: str) -> bool:
    """Intenta adquirir el bloqueo para extracción de hechos. Retorna True si lo logra, False si ya está bloqueado."""
    if not supabase: return True
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        
        # Verificar estado actual en user_profiles
        res = supabase.table("user_profiles").select("fact_locked_at").eq("id", user_id).execute()
        if res.data and len(res.data) > 0:
            locked_at_str = res.data[0].get("fact_locked_at")
            if locked_at_str:
                try:
                    if locked_at_str.endswith("Z"):
                        locked_at_str = locked_at_str[:-1] + "+00:00"
                    locked_at = datetime.fromisoformat(locked_at_str)
                    if now - locked_at < timedelta(minutes=5):
                        return False
                except ValueError:
                    pass
                
        # Intentar establecer el timestamp
        update_res = supabase.table("user_profiles").update({"fact_locked_at": now.isoformat()}).eq("id", user_id).execute()
            
        return len(update_res.data) > 0
    except Exception as e:
        # Si falla (ej. la columna no existe), imprimimos el error pero no bloqueamos
        error_msg = str(e)
        if "PGRST204" not in error_msg:
            logger.error(f"Error acquiring fact lock: {e}")
        return True

def release_fact_lock(user_id: str):
    """Libera el bloqueo de extracción de hechos."""
    if not supabase: return
    try:
        supabase.table("user_profiles").update({"fact_locked_at": None}).eq("id", user_id).execute()
    except Exception as e:
        error_msg = str(e)
        if "PGRST204" not in error_msg:
            logger.error(f"Error releasing fact lock: {e}")

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

# ============================================================
# GESTIÓN DE MEMORIA — Resúmenes de Conversación
# ============================================================

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

# ============================================================
# VECTORIZED KNOWLEDGE BASE (VÍA 1)
# ============================================================

def save_user_fact(user_id: str, fact: str, embedding: list, metadata: dict = None):
    """Guarda un hecho, su embedding y sus metadatos en la base de datos."""
    if not supabase: return None
    try:
        data_to_insert = {
            "user_id": user_id,
            "fact": fact,
            "embedding": embedding
        }
        if metadata:
            data_to_insert["metadata"] = metadata
            
        res = supabase.table("user_facts").insert(data_to_insert).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error guardando user_fact: {e}")
        return None

def delete_expired_temporal_facts(user_id: str = None, hours: int = 48):
    """Elimina los hechos con categoría 'sintoma_temporal' que son más antiguos que 'hours'."""
    if not supabase: return None
    from datetime import datetime, timedelta, timezone
    
    threshold_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    
    try:
        query = supabase.table("user_facts").delete().contains("metadata", {"category": "sintoma_temporal"}).lt("created_at", threshold_time)
        if user_id:
            query = query.eq("user_id", user_id)
        
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando temporal facts expirados: {e}")
        return None

def get_user_facts_by_metadata(user_id: str, key: str, value: str):
    """Busca hechos Exactos filtrando dentro del JSONB de metadata.
    Ejemplo: get_user_facts_by_metadata(user_id, 'category', 'alergia')
    """
    if not supabase: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        # Supabase Python client filter for JSONB: metadata->>key = value
        # we can use eq() if we query a specific path, but simpler is using contains
        filter_dict = {key: value}
        # Añadimos el filtro is_active
        res = supabase.table("user_facts").select("*").eq("user_id", user_id).eq("is_active", True).contains("metadata", filter_dict).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error buscando facts por metadata: {e}")
        return []

def delete_user_facts_by_metadata(user_id: str, filter_dict: dict):
    """Soft delete filtrando dentro del JSONB de metadata."""
    if not supabase: return None
    try:
        res = supabase.table("user_facts").update({"is_active": False}).eq("user_id", user_id).contains("metadata", filter_dict).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error haciendo soft delete a facts por metadata: {e}")
        return None

def search_user_facts(user_id: str, query_embedding: list, query_text: str = None, threshold: float = 0.5, limit: int = 5):
    """Busca hechos similares usando búsqueda híbrida (si hay texto) o vectorial pura en Supabase."""
    if not connection_pool: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        # Array Python a string pgvector '[1.2, 0.4, ...]'
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        if query_text:
            # Búsqueda híbrida (vector + full-text search)
            query = "SELECT * FROM hybrid_search_user_facts(query_text => %s, query_embedding => %s, match_count => %s, p_user_id => %s)"
            res = execute_sql_query(query, (query_text, emb_str, limit, user_id), fetch_all=True)
            return res
        else:
            # Búsqueda vectorial pura
            query = "SELECT * FROM match_user_facts(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s)"
            res = execute_sql_query(query, (emb_str, threshold, limit, user_id), fetch_all=True)
            return res
    except Exception as e:
        logger.error(f"Error buscando facts: {e}")
        return []

def search_user_facts_hybrid(user_id: str, query_embedding: list, filter_metadata: dict = None, threshold: float = 0.5, limit: int = 5):
    """Búsqueda Híbrida Vectorial: similitud de vectores pre-filtrada por metadatos (JSONB) usando pgvector RPC."""
    if not connection_pool: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        from psycopg.types.json import Jsonb
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        query = "SELECT * FROM match_user_facts_hybrid_metadata(query_embedding => %s, match_threshold => %s, match_count => %s, p_user_id => %s, p_metadata => %s)"
        meta_param = Jsonb(filter_metadata) if filter_metadata else None
        
        res = execute_sql_query(query, (emb_str, threshold, limit, user_id, meta_param), fetch_all=True)
        return res
    except Exception as e:
        logger.error(f"Error en búsqueda híbrida vectorial (metadatos): {e}")
        return []
        
def delete_user_fact(fact_id: str):
    """Hace un soft delete cambiando is_active a False"""
    if not supabase: return None
    try:
        # En lugar de .delete(), usamos .update()
        res = supabase.table("user_facts").update({"is_active": False}).eq("id", fact_id).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error haciendo soft delete a user_fact: {e}")
        return None

# ============================================================
# COLA PERSISTENTE DE HECHOS PENDIENTES (pending_facts_queue)
# Reemplaza la antigua cola volátil en RAM.
# ============================================================

def enqueue_pending_fact(user_id: str, message: str, recent_history: str = ""):
    """Encola un mensaje pendiente de extracción de hechos en Supabase."""
    if not supabase: return None
    try:
        res = supabase.table("pending_facts_queue").insert({
            "user_id": user_id,
            "message": message,
            "recent_history": recent_history or ""
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error encolando hecho pendiente: {e}")
        return None

def dequeue_pending_facts(user_id: str):
    """Obtiene todos los hechos pendientes de un usuario, ordenados cronológicamente."""
    if not supabase: return []
    try:
        res = supabase.table("pending_facts_queue").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error obteniendo hechos pendientes: {e}")
        return []

def delete_pending_facts(fact_ids: list):
    """Elimina los registros procesados de la cola de pendientes."""
    if not supabase or not fact_ids: return None
    try:
        res = supabase.table("pending_facts_queue").delete().in_("id", fact_ids).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error eliminando hechos pendientes procesados: {e}")
        return None

def get_all_user_facts(user_id: str):
    """Obtiene todos los hechos (facts) de un usuario para mostrarlos en la UI de Ajustes."""
    if not supabase: return []
    try:
        # Añadimos el filtro is_active
        res = supabase.table("user_facts").select("id, fact, metadata, created_at").eq("user_id", user_id).eq("is_active", True).order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error obteniendo ALL user facts: {e}")
        return []

# ============================================================
# DIARIO VISUAL Y MEMORIA MULTIMODAL (VÍA 2)
# ============================================================

def save_visual_entry(user_id: str, image_url: str, description: str, embedding: list):
    """
    Guarda una entrada del diario visual con deduplicación semántica.
    Antes de insertar, busca si ya existe un vector con >0.95 de similitud.
    Si existe, actualiza frequency += 1 y last_seen = NOW() en lugar de crear un duplicado.
    Esto previene que el RAG visual se sature con 15 fotos idénticas de mangú.
    """
    if not supabase: return None
    try:
        # === DEDUPLICACIÓN SEMÁNTICA: Buscar duplicados cercanos ===
        try:
            similar = supabase.rpc("match_visual_diary", {
                "query_embedding": embedding,
                "match_threshold": 0.95,
                "match_count": 1,
                "p_user_id": user_id
            }).execute()
            
            if similar.data and len(similar.data) > 0:
                existing = similar.data[0]
                existing_id = existing.get("id")
                old_freq = existing.get("frequency", 1) or 1
                print(f"🔄 [DEDUP VISUAL] Entrada similar detectada (sim>{0.95}). "
                      f"Actualizando frequency {old_freq} → {old_freq + 1} en lugar de insertar duplicado.")
                
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                
                supabase.table("visual_diary").update({
                    "frequency": old_freq + 1,
                    "last_seen": now,
                    "image_url": image_url,      # Actualizar con la foto más reciente
                    "description": description   # Actualizar con la descripción más reciente
                }).eq("id", existing_id).execute()
                
                logger.debug(f"✅ [DEDUP VISUAL] Registro {str(existing_id)[:8]}... actualizado.")
                return similar.data
        except Exception as dedup_err:
            # Si la deduplicación falla (ej. columnas aún no existen), insertar normalmente
            logger.error(f"⚠️ [DEDUP VISUAL] Error en deduplicación, insertando normalmente: {dedup_err}")
        
        # === INSERCIÓN NORMAL (no hay duplicado) ===
        res = supabase.table("visual_diary").insert({
            "user_id": user_id,
            "image_url": image_url,
            "description": description,
            "embedding": embedding,
            "frequency": 1
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error guardando visual_entry: {e}")
        return None

def search_visual_diary(user_id: str, query_embedding: list, threshold: float = 0.5, limit: int = 5):
    """Busca fotos/entradas visuales similares usando la función RPC match_visual_diary."""
    if not supabase: return []
    try:
        res = supabase.rpc("match_visual_diary", {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": limit,
            "p_user_id": user_id
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error buscando visual_diary: {e}")
        return []

# ============================================================
# CEREBRO CENTRAL: GESTIÓN DE PERFIL UNIFICADO
# ============================================================

def get_user_profile(user_id: str):
    """Obtiene el perfil completo del usuario, incluyendo el health_profile."""
    if not supabase: return None
    try:
        from datetime import datetime, timezone
        res = supabase.table("user_profiles").select("*").eq("id", user_id).execute()
        if not res.data:
            return None
            
        profile = res.data[0]
        
        # --- Graceful Degradation Middleware ---
        if profile.get("subscription_status") == "CANCELLED" and profile.get("subscription_end_date"):
            end_date_str = profile.get("subscription_end_date")
            try:
                if end_date_str.endswith("Z"):
                    end_date_str = end_date_str[:-1] + "+00:00"
                end_date = datetime.fromisoformat(end_date_str)
                now_utc = datetime.now(timezone.utc)
                
                # Si ya cruzamos la hora exacta de terminación, lo degradamos
                if now_utc > end_date:
                    logger.info(f"⬇️ Degradando perfil de {user_id} a 'gratis'. El tiempo de su cancelación (Graceful) terminó.")
                    supabase.table("user_profiles").update({
                        "plan_tier": "gratis",
                        "subscription_status": "INACTIVE"
                    }).eq("id", user_id).execute()
                    
                    profile["plan_tier"] = "gratis"
                    profile["subscription_status"] = "INACTIVE"
            except Exception as d_e:
                logger.error(f"Error parseando fechas en graceful degradation para {user_id}: {d_e}")
        # ---------------------------------------
        
        return profile
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {e}")
        return None

def update_user_health_profile(user_id: str, health_profile: dict):
    """Sobreescribe el JSONB de health_profile en la base de datos."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").update({
            "health_profile": health_profile
        }).eq("id", user_id).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error actualizando health_profile: {e}")
        return None

def get_shopping_plan_hash(user_id: str) -> str:
    """Obtiene el hash del plan usado para la última auto-generación de shopping list."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").select("shopping_plan_hash").eq("id", user_id).execute()
        if res.data and res.data[0].get("shopping_plan_hash"):
            return res.data[0]["shopping_plan_hash"]
        return None
    except Exception as e:
        # Columna no existe aún → no hay cache
        return None

def save_shopping_plan_hash(user_id: str, plan_hash: str):
    """Guarda el hash del plan para cache de auto-generación de shopping list."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").update({"shopping_plan_hash": plan_hash}).eq("id", user_id).execute()
        return res.data
    except Exception as e:
        # Columna no existe → silenciar (cache es opcional)
        logger.warning(f"⚠️ [DB] No se pudo guardar shopping_plan_hash: {e}")
        return None

def get_latest_meal_plan(user_id: str):
    """Obtiene el JSON del plan de comidas más reciente del usuario."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("plan_data")
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan actual: {e}")
        return None

def get_recent_meals_from_plans(user_id: str, days: int = 5):
    """Obtiene una lista de nombres de comidas de los planes recientes para evitar repeticiones."""
    if not supabase: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, meal_names").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
        meals = set() # 👈 Usar un Set evita enviar nombres duplicados al LLM y ahorra tokens
        if res.data:
            for row in res.data:
                meal_names_sql = row.get("meal_names")
                if meal_names_sql:
                    # 🚀 Fast Path O(1)
                    for n in meal_names_sql:
                        meals.add(n)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                         for day in plan_data.get("days", []):
                             for meal in day.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
                         if "meals" in plan_data:
                             for meal in plan_data.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
        return list(meals)
    except Exception as e:
        error_msg = str(e)
        if "meal_names" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna meal_names ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
                meals_fb = set()
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                             for day in plan_data.get("days", []):
                                 for meal in day.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                             if "meals" in plan_data:
                                 for meal in plan_data.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                return list(meals_fb)
            except Exception as e2:
                logger.error(f"Error obteniendo comidas recientes (fallback): {e2}")
                return []
                
        logger.error(f"Error obteniendo comidas recientes: {e}")
        return []

def get_recent_techniques(user_id: str, limit: int = 5) -> list:
    """Obtiene las técnicas de cocción usadas en planes recientes desde la columna `techniques` (text[]).
    Retorna una lista de tuplas (technique, created_at) para que el caller pueda aplicar decaimiento temporal.
    Ejemplo: [('Horneado Saludable', '2026-03-18T...'), ('Al Vapor', '2026-03-15T...')]
    """
    if not supabase or not user_id or user_id == "guest": return []
    try:
        res = supabase.table("meal_plans").select("techniques, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        # Retornar lista de tuplas CON duplicados y timestamps para decaimiento temporal.
        techniques = []
        if res.data:
            for row in res.data:
                techs = row.get("techniques")
                created_at = row.get("created_at", "")
                if techs and isinstance(techs, list):
                    for t in techs:
                        if t:
                            techniques.append((t, created_at))
        return techniques
    except Exception as e:
        error_msg = str(e)
        if "techniques" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # La columna aún no existe en la DB → retornar vacío silenciosamente
            return []
        logger.error(f"Error obteniendo técnicas recientes: {e}")
        return []

def get_ingredient_frequencies_from_plans(user_id: str, limit: int = 5) -> list:
    """Extrae los ingredientes crudos directamente del JSON o de la columna optimizada si existe.
    Retorna una lista plana de strings de ingredientes."""
    if not supabase or not user_id: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, ingredients").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        all_ingredients = []
        if res.data:
            for row in res.data:
                ingredients_sql = row.get("ingredients")
                if ingredients_sql:
                    # 🚀 Fast Path O(1)
                    all_ingredients.extend(ingredients_sql)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                        for day in plan_data.get("days", []):
                            for meal in day.get("meals", []):
                                ingredients = meal.get("ingredients", [])
                                if isinstance(ingredients, list):
                                    all_ingredients.extend(ingredients)
        return all_ingredients
    except Exception as e:
        error_msg = str(e)
        if "ingredients" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna ingredients ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
                all_ings_fb = []
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                            for day in plan_data.get("days", []):
                                for meal in day.get("meals", []):
                                    ings = meal.get("ingredients", [])
                                    if isinstance(ings, list):
                                        all_ings_fb.extend(ings)
                return all_ings_fb
            except Exception as e2:
                logger.error(f"Error extrayendo ingredientes de planes (fallback): {e2}")
                return []
                
        logger.error(f"Error extrayendo ingredientes de planes: {e}")
        return []


def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0):
    """Guarda una comida consumida en la tabla consumed_meals de Supabase."""
    if not supabase: return None
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        res = supabase.table("consumed_meals").insert({
            "user_id": user_id,
            "meal_name": meal_name,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "healthy_fats": healthy_fats,
            "created_at": now
        }).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error guardando comida consumida: {e}")
        return None

def get_consumed_meals_today(user_id: str, date_str: str = None, tz_offset_mins: int = None):
    """Obtiene las comidas consumidas del día especificado en base a la zona horaria del usuario."""
    if not supabase: return []
    max_retries = 2
    for attempt in range(max_retries):
        try:
            from datetime import datetime, timezone, timedelta
            
            if date_str and tz_offset_mins is not None:
                # El frontend envía tz_offset en minutos (JS: getTimezoneOffset() - diferencia a UTC)
                # Parsear el inicio del día local
                try:
                    local_start = datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S")
                    local_end = datetime.strptime(f"{date_str} 23:59:59", "%Y-%m-%d %H:%M:%S")
                    
                    # Convertir local a UTC sumando tz_offset_mins (para UTC-4 el offset es 240)
                    utc_start = local_start + timedelta(minutes=tz_offset_mins)
                    utc_end = local_end + timedelta(minutes=tz_offset_mins)
                    
                    start_str = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                    end_str = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception as e:
                    logger.error(f"⚠️ Error procesando la zona horaria, usando fallback a UTC: {e}")
                    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    start_str = f"{today_date}T00:00:00Z"
                    end_str = f"{today_date}T23:59:59Z"
            else:
                today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                start_str = f"{today_date}T00:00:00Z"
                end_str = f"{today_date}T23:59:59Z"
            
            # Consultar la DB filtrando las comidas dentro del rango de horas local
            res = supabase.table("consumed_meals")\
                .select("*")\
                .eq("user_id", user_id)\
                .gte("created_at", start_str)\
                .lt("created_at", end_str)\
                .execute()
                
            return res.data
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            logger.error(f"Error obteniendo comidas consumidas de hoy: {e}")
            return []

# ============================================================
# MODIFICACIÓN DE PLAN INDIVIDUAL
# ============================================================

def get_latest_meal_plan_with_id(user_id: str):
    """Obtiene el plan más reciente del usuario incluyendo su ID para poder actualizarlo."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").select("id, plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan con ID: {e}")
        return None

def update_meal_plan_data(plan_id: str, new_plan_data: dict):
    """Actualiza el plan_data JSONB de un plan existente por su ID."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").update({"plan_data": new_plan_data}).eq("id", plan_id).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error actualizando plan_data: {e}")
        return None

# ============================================================
# LISTA DE COMPRAS CUSTOM (Items añadidos por la IA)
# ============================================================

# --- Lock por usuario para operaciones de shopping list ---
# Previene race conditions cuando deduplicación, auto-generación
# o purga se ejecutan concurrentemente para el mismo usuario.
# RLock (reentrant) permite que un flujo externo (auto-generate)
# adquiera el lock y llame a deduplicate_shopping_items internamente
# sin deadlock, ya que ambos corren en el mismo thread.
import threading as _threading
from functools import lru_cache as _lru_cache
import time as _time
from datetime import datetime as _datetime, timezone as _timezone, timedelta as _timedelta

class DistributedShoppingLock:
    """Un Distributed Lock (RLock reentrante) apoyado en Supabase para evitar 
    race conditions en instancias distribuidas (Serverless/Varios workers)."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._acquired = False
        self._local_lock = _threading.RLock()
        self._recursions = 0

    def acquire(self, timeout=10):
        # 1. Bloqueo local para hilos en el mismo proceso (Reentrancy)
        local_acq = self._local_lock.acquire(timeout=timeout)
        if not local_acq: return False
        
        # Si ya teníamos el lock distribuido en este proceso, somos reentrantes
        if self._recursions > 0:
            self._recursions += 1
            return True
            
        # 2. Bloqueo Distribuido (Intentar en la Base de Datos)
        start = _time.time()
        while _time.time() - start < timeout:
            try:
                # Si inserta, nadie lo tenía. Si falla por Primary Key, alguien más lo tiene.
                if supabase:
                    supabase.table("shopping_locks").insert({"user_id": self.user_id}).execute()
                self._acquired = True
                self._recursions = 1
                return True
            except Exception as e:
                error_str = str(e)
                # Manejar conflicto de llave primaria (lock existente) o tabla inexistente
                if "23505" in error_str or "duplicate" in error_str.lower() or "already exists" in error_str.lower() or "PGRST116" in error_str:
                    # Limpiar Locks Rancios (Stale Locks > 2 minutos)
                    try:
                        if supabase:
                            check = supabase.table("shopping_locks").select("locked_at").eq("user_id", self.user_id).execute()
                            if check.data:
                                locked_at_str = check.data[0].get("locked_at")
                                if locked_at_str:
                                    locked_at = _datetime.fromisoformat(locked_at_str.replace("Z", "+00:00"))
                                    if _datetime.now(_timezone.utc) - locked_at > _timedelta(minutes=2):
                                        supabase.table("shopping_locks").delete().eq("user_id", self.user_id).execute()
                                        continue
                    except Exception:
                        pass
                elif "relation \"shopping_locks\" does not exist" in error_str.lower() or "404" in error_str:
                    # Fallback si la migración no se ha corrido: actuar solo como RLock local
                    logger.warning("⚠️ [LOCK] Tabla shopping_locks no existe. Cayendo a fallback RLock en memoria. Ejecuta migration_shopping_locks.sql")
                    self._acquired = False 
                    self._recursions = 1
                    return True
            _time.sleep(0.5)
            
        # Timeout distribuido
        self._local_lock.release()
        return False

    def release(self):
        if self._recursions > 0:
            self._recursions -= 1
            if self._recursions == 0 and self._acquired:
                try:
                    if supabase:
                        supabase.table("shopping_locks").delete().eq("user_id", self.user_id).execute()
                except Exception as e:
                    logger.error(f"Error liberando distributed lock: {e}")
                finally:
                    self._acquired = False
        self._local_lock.release()

    def __enter__(self):
        self.acquire(timeout=15)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

@_lru_cache(maxsize=1024)
def get_user_shopping_lock(user_id: str) -> DistributedShoppingLock:
    """Obtiene un Distributed RLock exclusivo por usuario para serializar
    operaciones de shopping list que requieren atomicidad en entornos serverless.
    LRU(1024) previene memory leaks devolviendo la misma instancia por usuario."""
    return DistributedShoppingLock(user_id)

def add_custom_shopping_items(user_id: str, items: list, source: str = "manual", overwrite: bool = False):
    """Inserta uno o más items custom a la lista de compras del usuario.
    source: 'auto' (IA auto-generados), 'chat' (añadidos vía chat), 'manual' (default/legacy)
    Dual-write: guarda JSON en item_name (legacy) + columnas estructuradas (category, display_name, qty, emoji)."""
    if not supabase or not items: return None
    import json
    
    if overwrite:
        try:
            clear_all_shopping_items(user_id)
        except Exception as cle:
            logger.warning(f"⚠️ Aviso: No se pudieron borrar los items viejos durante overwrite: {cle}")
    
    def _extract_fields(item):
        """Extrae campos estructurados de un item (dict, model, o string)."""
        if hasattr(item, 'model_dump'):
            d = item.model_dump()
        elif isinstance(item, dict):
            d = item
        elif isinstance(item, str) and item.strip():
            return item.strip(), {"category": "", "display_name": item.strip(), "qty": "", "emoji": ""}
        else:
            return None, None
        item_name_json = json.dumps(d, ensure_ascii=False)
        structured = {
            "category": d.get("category", ""),
            "display_name": d.get("name", ""),
            "qty": d.get("qty", ""),
            "emoji": d.get("emoji", ""),
            "is_checked": d.get("is_checked", False)
        }
        return item_name_json, structured
    
    try:
        rows = []
        for item in items:
            item_name, fields = _extract_fields(item)
            if item_name is None:
                continue
            row = {
                "user_id": user_id,
                "item_name": item_name,
                "source": source,
                "category": fields["category"],
                "display_name": fields["display_name"],
                "qty": fields["qty"],
                "emoji": fields["emoji"],
                "is_checked": fields["is_checked"]
            }
            if fields["is_checked"]:
                from datetime import datetime, timezone
                row["checked_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
                
        if rows:
            res = supabase.table("custom_shopping_items").insert(rows).execute()
            return res.data
        return None
    except Exception as e:
        error_msg = str(e)
        # Fallback 1: columnas estructuradas no existen → insertar sin ellas
        if "category" in error_msg or "display_name" in error_msg or "qty" in error_msg or "emoji" in error_msg:
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes. Ejecute migration_shopping_structured_columns.sql")
            try:
                rows_fb = []
                for item in items:
                    item_name, _ = _extract_fields(item)
                    if item_name:
                        rows_fb.append({"user_id": user_id, "item_name": item_name, "source": source})
                if rows_fb:
                    res_fb = supabase.table("custom_shopping_items").insert(rows_fb).execute()
                    return res_fb.data
                return None
            except Exception as e2:
                error_msg2 = str(e2)
                if "source" in error_msg2 or "PGRST205" in error_msg2:
                    # Fallback 2: ni source ni columnas estructuradas
                    return _add_shopping_items_minimal(user_id, items)
                logger.error(f"Error añadiendo items (fallback sin columnas estructuradas): {e2}")
                return None
        if "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Fallback 2: columna source tampoco existe
            logger.warning("⚠️ [DB] Columna source ausente. Ejecute migration_shopping_is_checked.sql")
            return _add_shopping_items_minimal(user_id, items)
        logger.error(f"Error añadiendo items a shopping list: {e}")
        return None

def _add_shopping_items_minimal(user_id: str, items: list):
    """Fallback mínimo: solo user_id + item_name (pre-migración)."""
    import json
    try:
        rows = []
        for item in items:
            if hasattr(item, 'model_dump'):
                rows.append({"user_id": user_id, "item_name": json.dumps(item.model_dump(), ensure_ascii=False)})
            elif isinstance(item, dict):
                rows.append({"user_id": user_id, "item_name": json.dumps(item, ensure_ascii=False)})
            elif isinstance(item, str) and item.strip():
                rows.append({"user_id": user_id, "item_name": item.strip()})
        if rows:
            res = supabase.table("custom_shopping_items").insert(rows).execute()
            return res.data
        return None
    except Exception as e:
        logger.error(f"Error añadiendo items a shopping list (minimal fallback): {e}")
        return None

def delete_auto_generated_shopping_items(user_id: str, exclude_ids: list = None):
    """Elimina los items auto-generados de la lista de compras del usuario.
    Usa columna source='auto' para borrado O(1). Fallback a JSON parsing si la columna no existe.
    exclude_ids: IDs de items recién insertados que NO deben borrarse (patrón insert-first / delete-old)."""
    if not supabase: return False
    try:
        # 🚀 Borrado directo por columna source (O(1) con índice, sin parsear JSON)
        # 🛡️ NUNCA borrar items ya comprados (is_checked=True) — son inventario físico del usuario
        query = supabase.table("custom_shopping_items").delete().eq("user_id", user_id).eq("source", "auto").eq("is_checked", False)
        if exclude_ids:
            # PostgREST NOT IN: 1 sola cláusula SQL vs N cláusulas neq encadenadas
            query = query.not_.in_("id", exclude_ids)
        query.execute()
        return True
    except Exception as e:
        error_msg = str(e)
        if "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Columna source no existe → fallback al método legacy (JSON parsing)
            logger.warning("⚠️ [DB] Columna source ausente. Usando fallback JSON. Ejecute migration_shopping_is_checked.sql")
            return _delete_auto_shopping_items_legacy(user_id, exclude_ids)
        logger.error(f"Error borrando items auto-generados: {e}")
        return False

def _delete_auto_shopping_items_legacy(user_id: str, exclude_ids: list = None):
    """Fallback legacy: borra items auto-generados parseando JSON (O(N) full table scan)."""
    import json
    try:
        res = supabase.table("custom_shopping_items").select("id, item_name").eq("user_id", user_id).execute()
        existing = res.data
        if not existing: return True
        
        exclude_set = set(exclude_ids) if exclude_ids else set()
        ids_to_delete = []
        for item in existing:
            if item['id'] in exclude_set:
                continue
            try:
                parsed = json.loads(item['item_name'])
                if isinstance(parsed, dict) and 'category' in parsed:
                    ids_to_delete.append(item['id'])
            except (json.JSONDecodeError, ValueError):
                pass
        
        if ids_to_delete:
            supabase.table("custom_shopping_items").delete().in_("id", ids_to_delete).execute()
        return True
    except Exception as e:
        logger.error(f"Error borrando items auto-generados (legacy fallback): {e}")
        return False

def get_custom_shopping_items(user_id: str, limit: int = 200, offset: int = 0, sort_by: str = "category", sort_order: str = "asc"):
    """Obtiene los items custom de la lista de compras del usuario con paginación y ordenamiento.
    sort_by: 'category' | 'created_at' | 'display_name' | 'is_checked' (default: 'category')
    sort_order: 'asc' | 'desc' (default: 'asc')
    Retorna columnas estructuradas (category, display_name, qty, emoji) si están disponibles."""
    if not supabase: return {"data": [], "total": 0}
    
    # Whitelist de campos para evitar inyección
    ALLOWED_SORT = {"category", "created_at", "display_name", "is_checked", "name"}
    if sort_by not in ALLOWED_SORT:
        sort_by = "category"
    is_desc = sort_order.lower() == "desc"
    
    try:
        # Intento 1: con columnas estructuradas → permite ordenar por categoría a nivel DB
        query = supabase.table("custom_shopping_items").select(
            "id, item_name, is_checked, checked_at, source, category, display_name, qty, emoji, created_at",
            count="exact"
        ).eq("user_id", user_id).order(sort_by, desc=is_desc).range(offset, offset + limit - 1)
        res = query.execute()
        return {"data": res.data, "total": res.count or 0}
    except Exception as e:
        error_msg = str(e)
        if "category" in error_msg or "display_name" in error_msg or "qty" in error_msg or "emoji" in error_msg:
            # Fallback 2: sin columnas estructuradas pero con is_checked/source
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes. Ejecute migration_shopping_structured_columns.sql")
            try:
                fb_sort = sort_by if sort_by in {"created_at", "is_checked"} else "created_at"
                query_fb = supabase.table("custom_shopping_items").select(
                    "id, item_name, is_checked, source, created_at", count="exact"
                ).eq("user_id", user_id).order(fb_sort, desc=is_desc).range(offset, offset + limit - 1)
                res_fb = query_fb.execute()
                return {"data": res_fb.data, "total": res_fb.count or 0}
            except Exception as e2:
                error_msg2 = str(e2)
                if "is_checked" in error_msg2 or "source" in error_msg2:
                    # Fallback 3: schema mínimo
                    return _get_shopping_items_minimal(user_id, limit, offset)
                logger.error(f"Error obteniendo items (fallback sin columnas estructuradas): {e2}")
                return {"data": [], "total": 0}
        if "is_checked" in error_msg or "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Fallback 3: schema mínimo (solo id, item_name, created_at)
            logger.warning("⚠️ [DB] Columnas is_checked/source ausentes. Ejecute migration_shopping_is_checked.sql")
            return _get_shopping_items_minimal(user_id, limit, offset)
        logger.error(f"Error obteniendo custom shopping items: {e}")
        return {"data": [], "total": 0}

def _get_shopping_items_minimal(user_id: str, limit: int = 200, offset: int = 0, sort_order: str = "desc"):
    """Fallback mínimo: solo id, item_name, created_at (pre-migración)."""
    is_desc = sort_order.lower() == "desc"
    try:
        res = supabase.table("custom_shopping_items").select(
            "id, item_name, created_at", count="exact"
        ).eq("user_id", user_id).order("created_at", desc=is_desc).range(offset, offset + limit - 1).execute()
        return {"data": res.data, "total": res.count or 0}
    except Exception as e:
        logger.error(f"Error obteniendo custom shopping items (minimal fallback): {e}")
        return {"data": [], "total": 0}

def clear_all_shopping_items(user_id: str):
    """Elimina TODOS los items de la lista de compras del usuario."""
    if not supabase: return False
    try:
        supabase.table("custom_shopping_items").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error limpiando lista de compras: {e}")
        return False

def uncheck_all_shopping_items(user_id: str):
    """Desmarca (is_checked=false) TODOS los items de la lista de compras del usuario."""
    if not supabase: return False
    try:
        supabase.table("custom_shopping_items").update({"is_checked": False, "checked_at": None}).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        error_msg = str(e)
        if "is_checked" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            logger.warning("⚠️ [DB] Columna is_checked ausente. Ejecute migration_shopping_is_checked.sql")
            return False
        logger.error(f"Error desmarcando items: {e}")
        return False

def deduplicate_shopping_items(user_id: str):
    """Encuentra y fusiona items duplicados en la lista de compras del usuario.
    Agrupa por display_name normalizado. Suma cantidades numéricas cuando es posible.
    Retorna el número de duplicados eliminados.
    
    Thread-safe: usa un RLock per-user para serializar el SELECT→process→DELETE/UPDATE
    y evitar race conditions con auto-generación o purga concurrentes."""
    if not supabase: return {"removed": 0, "merged": []}
    
    lock = get_user_shopping_lock(user_id)
    acquired = lock.acquire(timeout=10)
    if not acquired:
        logger.warning(f"⚠️ [DEDUP] Timeout adquiriendo lock para {user_id}. Otra operación en curso.")
        return {"removed": 0, "merged": [], "error": "Deduplicación en curso para este usuario, intenta más tarde."}
    
    import re, json as _json
    try:
        return _deduplicate_shopping_items_impl(user_id, re, _json)
    finally:
        lock.release()

def _deduplicate_shopping_items_impl(user_id: str, re, _json):
    
    try:
        # Obtener todos los items del usuario
        res = supabase.table("custom_shopping_items").select(
            "id, item_name, display_name, qty, category, emoji, source, is_checked, created_at"
        ).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not res.data or len(res.data) < 2:
            return {"removed": 0, "merged": []}
        
        items = res.data
    except Exception as e:
        # Fallback: sin columnas estructuradas
        try:
            res = supabase.table("custom_shopping_items").select("id, item_name, created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
            if not res.data or len(res.data) < 2:
                return {"removed": 0, "merged": []}
            items = res.data
        except Exception as e2:
            logger.error(f"Error obteniendo items para dedup: {e2}")
            return {"removed": 0, "merged": []}
    
    def _normalize(text: str) -> str:
        """Normaliza texto para comparación: minúsculas, sin acentos, sin espacios extra."""
        if not text: return ""
        import unicodedata
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))
    
    # Agrupar por nombre normalizado
    groups = {}  # normalized_name -> [items]
    for item in items:
        # Preferir display_name (estructurado) sobre item_name (JSON legacy)
        name = item.get("display_name") or ""
        if not name:
            # Intentar extraer name del JSON en item_name
            raw = item.get("item_name", "")
            if raw.startswith("{"):
                try:
                    parsed = _json.loads(raw)
                    name = parsed.get("name", raw)
                except Exception:
                    name = raw
            else:
                name = raw
        
        key = _normalize(name)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append({"item": item, "name": name})
    
    ids_to_delete = []
    ids_to_update = {}  # id -> {updates}
    merged_info = []
    
    # ===== FASE 1: Deduplicación exacta/regex =====
    unique_survivors = []
    
    for key, group in groups.items():
        # El primer item es el más reciente (ordenamos por created_at desc)
        keeper = group[0]
        duplicates = group[1:]
        
        # Intentar sumar cantidades
        keeper_qty_str = keeper["item"].get("qty", "")
        keeper_name = keeper["name"]
        keeper_num, keeper_unit, _ = parse_ingredient_qty(f"{keeper_qty_str} {keeper_name}", to_metric=False)
        total = keeper_num
        can_sum = keeper_num is not None
        
        # Si no podemos sumar matemáticamente, concatenamos strings
        concatenated_qtys = []
        if keeper_qty_str:
            concatenated_qtys.append(keeper_qty_str)
            
        for dup in duplicates:
            dup_qty_str = dup["item"].get("qty", "")
            dup_num, dup_unit, _ = parse_ingredient_qty(f"{dup_qty_str} {dup['name']}", to_metric=False)
            
            if can_sum and dup_num is not None and _normalize(dup_unit) == _normalize(keeper_unit):
                total += dup_num
            else:
                can_sum = False
                if dup_qty_str and dup_qty_str not in concatenated_qtys:
                    concatenated_qtys.append(dup_qty_str)
            
            ids_to_delete.append(dup["item"]["id"])
            
            # Si algún duplicado estaba checked, mantener ese estado
            if dup["item"].get("is_checked") and not keeper["item"].get("is_checked"):
                keeper["item"]["is_checked"] = True
        
        # Actualizar cantidad del keeper
        update_payload = {}
        if can_sum and total is not None:
            new_qty = format_qty(total, keeper_unit)
            if new_qty != keeper_qty_str:
                keeper["item"]["qty"] = new_qty
                update_payload["qty"] = new_qty
                if len(duplicates) > 0:
                    merged_info.append(f"{keeper_name}: {new_qty}")
        elif not can_sum and len(concatenated_qtys) > 1:
            # Concatenar como '1 lb + 2 unidades'
            new_qty = " + ".join(filter(bool, concatenated_qtys))
            if new_qty != keeper_qty_str:
                keeper["item"]["qty"] = new_qty
                update_payload["qty"] = new_qty
                merged_info.append(f"{keeper_name}: {new_qty} (texto combinado)")
        elif len(duplicates) > 0:
            merged_info.append(f"{keeper_name}: {len(duplicates)} duplicados removidos sin sumar")
            
        if keeper["item"].get("is_checked"):
            update_payload["is_checked"] = True
            
        if update_payload:
            ids_to_update[keeper["item"]["id"]] = update_payload
            
        unique_survivors.append(keeper)
        
    if not ids_to_delete and not ids_to_update:
        return {"removed": 0, "merged": []}
    
    # Ejecutar deletes y updates en batch (reducir N+1 queries)
    try:
        # 🚀 Batch DELETE: Supabase .in_() ya es batch, pero chunkeamos por si hay >500 IDs
        CHUNK_SIZE = 100
        for i in range(0, len(ids_to_delete), CHUNK_SIZE):
            chunk = ids_to_delete[i:i + CHUNK_SIZE]
            supabase.table("custom_shopping_items").delete().in_("id", chunk).execute()
        
        # 🚀 UPDATEs serializados (en vez de batch complex porque enviamos múltiples columnas)
        if ids_to_update:
            for item_id, payload in ids_to_update.items():
                if "is_checked" in payload:
                    from datetime import datetime, timezone
                    payload["checked_at"] = datetime.now(timezone.utc).isoformat()
                supabase.table("custom_shopping_items").update(payload).eq("id", item_id).execute()
        
        logger.debug(f"🧹 [DEDUP] Eliminados {len(ids_to_delete)} duplicados, {len(ids_to_update)} cantidades actualizadas")
        return {"removed": len(ids_to_delete), "merged": merged_info}
    except Exception as e:
        logger.error(f"Error en deduplicación: {e}")
        return {"removed": 0, "merged": [], "error": str(e)}

MAX_SHOPPING_ITEMS_PER_USER = 500
CHECKED_ITEM_EXPIRY_DAYS = 30

def purge_old_shopping_items(user_id: str):
    """Auto-purga items viejos de la lista de compras.
    1) Elimina items checked con checked_at > 30 días.
    2) Si aún hay más de 500 items, elimina los más antiguos.
    Retorna el número total de items purgados."""
    if not supabase: return 0
    
    from datetime import datetime, timezone, timedelta
    total_purged = 0
    
    try:
        # --- Fase 1: Purgar items checked hace más de 30 días ---
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CHECKED_ITEM_EXPIRY_DAYS)).isoformat()
        try:
            res = supabase.table("custom_shopping_items").delete()\
                .eq("user_id", user_id)\
                .eq("is_checked", True)\
                .lt("checked_at", cutoff)\
                .execute()
            phase1 = len(res.data) if res.data else 0
            total_purged += phase1
            if phase1 > 0:
                logger.info(f"🧹 [PURGE] Fase 1: eliminados {phase1} items checked hace >{CHECKED_ITEM_EXPIRY_DAYS} días")
        except Exception as e:
            # Columna is_checked/checked_at puede no existir aún
            logger.warning(f"⚠️ [PURGE] Fase 1 skipped (columnas ausentes): {e}")
        
        # --- Fase 2: Enforce tope global ---
        try:
            count_res = supabase.table("custom_shopping_items")\
                .select("id", count="exact")\
                .eq("user_id", user_id)\
                .execute()
            total_items = count_res.count if hasattr(count_res, 'count') and count_res.count else len(count_res.data or [])
            
            if total_items > MAX_SHOPPING_ITEMS_PER_USER:
                excess = total_items - MAX_SHOPPING_ITEMS_PER_USER
                # Obtener los IDs más antiguos a eliminar
                oldest_res = supabase.table("custom_shopping_items")\
                    .select("id")\
                    .eq("user_id", user_id)\
                    .order("created_at", desc=False)\
                    .limit(excess)\
                    .execute()
                if oldest_res.data:
                    old_ids = [r["id"] for r in oldest_res.data]
                    supabase.table("custom_shopping_items").delete().in_("id", old_ids).execute()
                    total_purged += len(old_ids)
                    logger.info(f"🧹 [PURGE] Fase 2: eliminados {len(old_ids)} items (tope {MAX_SHOPPING_ITEMS_PER_USER} excedido)")
        except Exception as e:
            logger.warning(f"⚠️ [PURGE] Fase 2 error: {e}")
        
        return total_purged
    except Exception as e:
        logger.error(f"Error en purge_old_shopping_items: {e}")
        return 0

def delete_custom_shopping_item(item_id: str, user_id: str = None):
    """Elimina un item custom de la lista de compras. Si se provee user_id, verifica ownership."""
    if not supabase: return None
    try:
        query = supabase.table("custom_shopping_items").delete().eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando custom shopping item: {e}")
        return None

def delete_custom_shopping_items_batch(item_ids: list, user_id: str = None):
    """Elimina múltiples items custom de la lista de compras de una vez. Si se provee user_id, verifica ownership."""
    if not supabase or not item_ids: return None
    try:
        query = supabase.table("custom_shopping_items").delete().in_("id", item_ids)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando custom shopping items en batch: {e}")
        return None

def update_custom_shopping_item(item_id: str, updates: dict, user_id: str = None):
    """Actualiza campos editables de un item (display_name, qty, category, emoji).
    Si se provee user_id, verifica ownership (IDOR protection).
    Si las columnas estructuradas no existen, cae al fallback legacy (JSON en item_name)."""
    if not supabase: return None
    
    # Solo permitir campos editables
    allowed_fields = {"display_name", "qty", "category", "emoji"}
    clean_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
    
    if not clean_updates:
        return []
    
    try:
        # 🚀 Método directo: UPDATE a columnas estructuradas
        query = supabase.table("custom_shopping_items").update(clean_updates).eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        
        # NOTA: Ya no sincronizamos item_name JSON (dual-write legacy eliminado).
        # Las columnas estructuradas son la fuente de verdad.
        # Si necesitas el fallback legacy, se activa automáticamente en el except.
        
        return res.data
    except Exception as e:
        error_msg = str(e)
        if any(col in error_msg for col in ["display_name", "qty", "category", "emoji", "PGRST205"]):
            # Columnas estructuradas no existen → fallback a JSON en item_name
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes. Usando fallback JSON para update.")
            return _update_shopping_item_legacy(item_id, clean_updates, user_id)
        logger.error(f"Error actualizando custom shopping item: {e}")
        return None

def _update_shopping_item_legacy(item_id: str, updates: dict, user_id: str = None):
    """Fallback: actualiza el JSON embebido en item_name."""
    import json
    try:
        query = supabase.table("custom_shopping_items").select("id, item_name").eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        if not res.data:
            return []
        
        row = res.data[0]
        raw = row.get("item_name", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        
        field_map = {"display_name": "name", "qty": "qty", "category": "category", "emoji": "emoji"}
        for k, v in updates.items():
            if k in field_map:
                parsed[field_map[k]] = v
        
        supabase.table("custom_shopping_items").update(
            {"item_name": json.dumps(parsed, ensure_ascii=False)}
        ).eq("id", item_id).execute()
        
        return [{"id": item_id, **updates}]
    except Exception as e:
        logger.error(f"Error actualizando item (legacy fallback): {e}")
        return None

def update_custom_shopping_item_status(item_id: str, is_checked: bool, user_id: str = None):
    """Actualiza el estado is_checked de un item.
    Intenta usar la columna nativa is_checked (1 query atómica, sin race conditions).
    Si la columna no existe aún, hace fallback al método legacy (JSON en item_name).
    Si se provee user_id, verifica ownership (IDOR protection)."""
    if not supabase: return None
    try:
        # 🚀 Método atómico: UPDATE directo a columna nativa (O(1), sin race conditions)
        from datetime import datetime, timezone
        update_data = {"is_checked": is_checked}
        update_data["checked_at"] = datetime.now(timezone.utc).isoformat() if is_checked else None
        update_query = supabase.table("custom_shopping_items").update(update_data).eq("id", item_id)
        if user_id:
            update_query = update_query.eq("user_id", user_id)
        update_res = update_query.execute()
        # Verificar que se actualizó al menos 1 fila
        if update_res.data:
            return update_res.data
        return None
    except Exception as e:
        error_msg = str(e)
        if "is_checked" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Columna nativa no existe → fallback al método legacy (JSON en item_name)
            logger.warning("⚠️ [DB] Columna is_checked ausente. Usando fallback JSON. Ejecute migration_shopping_is_checked.sql")
            return _update_shopping_item_status_legacy(item_id, is_checked, user_id)
        logger.error(f"Error actualizando estado de item: {e}")
        return None

def _update_shopping_item_status_legacy(item_id: str, is_checked: bool, user_id: str = None):
    """Fallback legacy: guarda is_checked dentro del JSON de item_name (read-modify-write)."""
    import json
    try:
        query = supabase.table("custom_shopping_items").select("item_name").eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        if not res.data: return None
        
        current_name = res.data[0]['item_name']
        try:
            parsed = json.loads(current_name)
            if isinstance(parsed, dict):
                parsed['is_checked'] = is_checked
                new_name = json.dumps(parsed, ensure_ascii=False)
            else:
                raise ValueError("Not a dict")
        except (json.JSONDecodeError, ValueError, KeyError):
            parsed = {
                "category": "Otros",
                "emoji": "📝",
                "name": current_name,
                "qty": "",
                "is_checked": is_checked
            }
            new_name = json.dumps(parsed, ensure_ascii=False)
        
        update_query = supabase.table("custom_shopping_items").update({"item_name": new_name}).eq("id", item_id)
        if user_id:
            update_query = update_query.eq("user_id", user_id)
        update_res = update_query.execute()
        return update_res.data
    except Exception as e:
        logger.error(f"Error actualizando estado de item (legacy fallback): {e}")
        return None

# ============================================================
# SISTEMA DE CRÉDITOS Y USO DE API
# ============================================================

def log_api_usage(user_id: str, endpoint: str = "gemini"):
    """Guarda un registro de uso de la API (consume 1 crédito)."""
    if not supabase or not user_id or user_id == "guest": return None
    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = supabase.table("api_usage").insert({
                "user_id": user_id,
                "endpoint": endpoint
            }).execute()
            return res.data
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            logger.error(f"Error registrando api_usage: {e}")
            return None

def get_monthly_api_usage(user_id: str) -> int:
    """Cuenta cuántas llamadas a la API ha hecho el usuario este mes."""
    if not supabase or not user_id or user_id == "guest": return 0
    from datetime import datetime
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Obtener inicio de mes actual
            now = datetime.now()
            start_date = datetime(now.year, now.month, 1).isoformat()
            
            # En Supabase count es más eficiente
            res = supabase.table("api_usage").select("*", count="exact").eq("user_id", user_id).gte("created_at", start_date).execute()
            return res.count if res.count is not None else 0
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            logger.error(f"Error obteniendo api_usage mensual: {e}")
            return 0
    return 0

# ============================================================
# MEMORIA PROACTIVA: TRACKING DE FRICCIÓN SILENCIOSA
# ============================================================

def track_meal_friction(user_id: str, session_id: str, rejected_meal: str):
    """
    Memoria Conductual: Trackea cuántas veces el usuario rechaza platos con la misma proteína base.
    Al tercer rechazo (strike 3), inserta el ingrediente en rechazos temporales y notifica proactivamente.
    """
    if not user_id or user_id == "guest" or not rejected_meal: return False
    
    from constants import DOMINICAN_PROTEINS
    from constants import GLOBAL_REVERSE_MAP
    
    _strip_accents = _strip_accents_canonical
    
    # Usar el mapa pre-computado a nivel de módulo (O(1)) en vez de reconstruirlo por llamada.
    # Crear versión sin acentos para matching robusto
    # (el LLM no siempre preserva tildes: "platano" vs "plátano")
    accent_safe_map = _ACCENT_SAFE_REVERSE_MAP
    
    base_ingredient = None
    lower_meal = _strip_accents(rejected_meal.lower())
    
    # Resolver sinónimos con n-gramas (trigrams → bigrams → unigrams)
    # para detectar multipalabra como "carne molida" → "res", "queso de freír" → "queso de freír"
    words = [w.strip(".,;:!?()\"'") for w in lower_meal.split()]
    for n in range(min(3, len(words)), 0, -1):  # 3-grams primero, luego 2, luego 1
        if base_ingredient:
            break
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in accent_safe_map:
                base_ingredient = accent_safe_map[ngram].capitalize()
                break
    
    # Fallback: búsqueda directa por nombre de proteína base
    if not base_ingredient:
        for p in DOMINICAN_PROTEINS:
            if p.lower() in lower_meal:
                base_ingredient = p
                break
            
    if not base_ingredient: return False
    
    # --- MÉTODO ATÓMICO (RPC) para evitar Race Conditions ---
    # Si dos requests de swap llegan simultáneos, el FOR UPDATE en PostgreSQL
    # serializa las escrituras garantizando que ambos incrementos se registren.
    try:
        rpc_result = supabase.rpc("increment_friction_rpc", {
            "p_user_id": user_id,
            "p_ingredient": base_ingredient
        }).execute()
        
        # El RPC retorna el conteo PRE-RESET (ej: 3 si alcanzó el umbral)
        new_count = rpc_result.data if isinstance(rpc_result.data, int) else 0
        
        if new_count >= 3:
            logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (vía RPC atómico).")
            
            rejection_record = {
                "meal_name": base_ingredient,
                "meal_type": "Ingrediente Fricción",
                "user_id": user_id,
                "session_id": session_id if session_id else None
            }
            insert_rejection(rejection_record)
            
            if session_id:
                msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
                save_message(session_id, "model", msg)
            return True
        
        return False
        
    except Exception as rpc_e:
        error_msg = str(rpc_e)
        if "Could not find the function" in error_msg or "PGRST202" in error_msg:
            # RPC aún no desplegado → fallback al método clásico (read-modify-write)
            pass
        else:
            logger.error(f"⚠️ [FRICCIÓN] Error en RPC atómico, usando fallback: {rpc_e}")
    
    # --- FALLBACK CLÁSICO (read-modify-write, vulnerable a race condition) ---
    # ⚠️ En producción, desplegar rpc_increment_friction.sql en Supabase para eliminar esto.
    profile = get_user_profile(user_id)
    if not profile: return False
    
    hp = profile.get("health_profile") or {}
    frictions = hp.get("frictions", {})
    
    current_count = frictions.get(base_ingredient, 0) + 1
    
    if current_count >= 3:
        logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (fallback).")
        
        rejection_record = {
            "meal_name": base_ingredient,
            "meal_type": "Ingrediente Fricción",
            "user_id": user_id,
            "session_id": session_id if session_id else None
        }
        insert_rejection(rejection_record)
        
        frictions[base_ingredient] = 0
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        
        if session_id:
            msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
            save_message(session_id, "model", msg)
        return True
    else:
        frictions[base_ingredient] = current_count
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        return False

# ============================================================
# MIGRACIÓN DE DATOS (De Guest a Usuario Registrado)
# ============================================================

def migrate_guest_data(session_ids: list, new_user_id: str):
    """
    Migra todos los datos asociados a uno o varios session_ids temporales
    (creados cuando el usuario era invitado) hacia el nuevo UUID del usuario registrado.
    """
    if not supabase or not session_ids or not new_user_id: 
        return False
    
    try:
        # 1. Actualizar agent_sessions para vincular historiales de chat
        supabase.table("agent_sessions").update({"user_id": new_user_id}).in_("id", session_ids).execute()
        
        # 2. Actualizar visual_diary (Vectores/Diario Visual)
        supabase.table("visual_diary").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 3. Actualizar user_facts (Vectores/Memoria a largo plazo)
        supabase.table("user_facts").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 4. Actualizar meal_plans (Planes guardados como guest, si los hubiera)
        supabase.table("meal_plans").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 5. Actualizar consumed_meals (Comidas registradas)
        supabase.table("consumed_meals").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 6. Actualizar pending_facts_queue
        supabase.table("pending_facts_queue").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 7. Actualizar meal_rejections (Ojo: estas usan session_id o user_id)
        # Primero intentamos por session_id (como se guardan usualmente los rechazos guest)
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("session_id", session_ids).execute()
        # Y también por user_id por si acaso
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 8. Actualizar meal_likes
        supabase.table("meal_likes").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 9. Actualizar custom_shopping_items (con deduplicación)
        try:
            # Obtener items existentes del usuario registrado
            existing_res = supabase.table("custom_shopping_items").select("item_name").eq("user_id", new_user_id).execute()
            existing_names = set()
            if existing_res.data:
                existing_names = {row.get("item_name", "") for row in existing_res.data}
            
            # Obtener items del guest
            guest_res = supabase.table("custom_shopping_items").select("id, item_name").in_("user_id", session_ids).execute()
            if guest_res.data:
                ids_to_migrate = []
                ids_to_delete = []
                for item in guest_res.data:
                    if item.get("item_name", "") in existing_names:
                        ids_to_delete.append(item["id"])  # Duplicado → eliminar
                    else:
                        ids_to_migrate.append(item["id"])
                        existing_names.add(item.get("item_name", ""))  # Evitar dup entre guests
                
                if ids_to_migrate:
                    supabase.table("custom_shopping_items").update({"user_id": new_user_id}).in_("id", ids_to_migrate).execute()
                if ids_to_delete:
                    supabase.table("custom_shopping_items").delete().in_("id", ids_to_delete).execute()
                    logger.warning(f"🧹 [MIGRATION] {len(ids_to_delete)} items duplicados eliminados, {len(ids_to_migrate)} migrados.")
        except Exception as shop_mig_e:
            logger.error(f"⚠️ [MIGRATION] Error migrando shopping items: {shop_mig_e}")
            # Fallback: migrar todo sin deduplicar
            supabase.table("custom_shopping_items").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 10. Recalcular frecuencias de ingredientes a partir de los planes migrados
        # Sin esto, el usuario registrado parte con freq=0 y pierde el historial de variedad.
        try:
            from constants import normalize_ingredient_for_tracking
            migrated_ings = get_ingredient_frequencies_from_plans(new_user_id, limit=10)
            if migrated_ings:
                normalized = [normalize_ingredient_for_tracking(i) for i in migrated_ings if i]
                normalized = [n for n in normalized if n]  # Filtrar vacíos
                if normalized:
                    increment_ingredient_frequencies(new_user_id, normalized)
                    logger.info(f"✅ [MIGRACIÓN] Frecuencias recalculadas para {new_user_id} ({len(normalized)} ingredientes)")
        except Exception as freq_e:
            # No bloquear la migración si falla el recálculo de frecuencias
            logger.error(f"⚠️ [MIGRACIÓN] Error recalculando frecuencias (no crítico): {freq_e}")
        
        logger.info(f"✅ Migración exitosa de {session_ids} a UUID {new_user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error migrando datos de invitado: {e}")
        return False


def log_unknown_ingredients(user_id: str, unknown_ings: list, raw_map: dict = None):
    """Loguea ingredientes que el LLM genera pero que el sistema de sinónimos no reconoce.
    Se guardan en la tabla `unknown_ingredients` para revisión periódica y expansión del catálogo.
    Usa RPC atómico con fallback a upsert clásico.
    """
    if not supabase or not user_id or user_id == "guest" or not unknown_ings:
        return
    
    try:
        for ing in unknown_ings[:20]:  # Cap a 20 por plan para no saturar
            raw_text = raw_map.get(ing, "") if raw_map else ""
            try:
                # Intentar RPC atómico
                supabase.rpc("log_unknown_ingredient_rpc", {
                    "p_user_id": user_id,
                    "p_ingredient": ing,
                    "p_raw_text": raw_text or None
                }).execute()
            except Exception as rpc_e:
                err = str(rpc_e)
                if "Could not find the function" in err or "PGRST202" in err or "unknown_ingredients" in err:
                    # RPC o tabla no desplegados aún → silenciar
                    return
                # Fallback: upsert directo
                try:
                    from datetime import datetime, timezone
                    supabase.table("unknown_ingredients").upsert({
                        "user_id": user_id,
                        "ingredient": ing,
                        "raw_text": raw_text or None,
                        "occurrences": 1,
                        "last_seen": datetime.now(timezone.utc).isoformat()
                    }, on_conflict="user_id,ingredient").execute()
                except Exception as fb_e:
                    if "unknown_ingredients" in str(fb_e) or "PGRST205" in str(fb_e):
                        return  # Tabla no existe aún → silenciar
                    logger.error(f"⚠️ [UNKNOWN ING] Error en fallback: {fb_e}")
                    return
        
        logger.info(f"📝 [UNKNOWN ING] {len(unknown_ings)} ingredientes no reconocidos logueados para revisión")
    except Exception as e:
        logger.error(f"⚠️ [UNKNOWN ING] Error logueando ingredientes desconocidos: {e}")

def increment_ingredient_frequencies(user_id: str, ingredients: list[str]):
    """Incrementa la frecuencia histórica de los ingredientes consumidos por un usuario.
    Intenta usar un RPC atómico O(1) robusto ante Race Conditions,
    con fallback al viejo método Select+Upsert si la función SQL no se ha creado.
    """
    if not supabase or not user_id or user_id == "guest": return
    
    try:
        from collections import Counter
        from datetime import datetime, timezone
        
        strip_accents = _strip_accents_canonical
            
        normalized_ings = [strip_accents(i.lower()).strip() for i in ingredients if i]
        if not normalized_ings: return
        
        incoming_counts = Counter(normalized_ings)
        ingredients_list = list(incoming_counts.keys())
        counts_list = list(incoming_counts.values())
        
        # 1. Intentar método atómico (RPC) para evitar Race Conditions
        try:
            supabase.rpc("increment_ingredient_frequencies_rpc", {
                "p_user_id": user_id,
                "p_ingredients": ingredients_list,
                "p_counts": counts_list
            }).execute()
            logger.info(f"✅ [DB] Frecuencia atómica (RPC) incrementada para {user_id} ({len(ingredients_list)} items)")
            return
        except Exception as rpc_e:
            error_msg = str(rpc_e)
            if "Could not find the function" in error_msg or "PGRST202" in error_msg:
                # El usuario aún no corre el código SQL en Supabase, pasamos al fallback silenciosamente
                pass
            else:
                logger.warning(f"⚠️ [DB] Aviso de RPC, recurriendo a fallback... Detalles: {rpc_e}")

        # 2. Fallback clásico: Leer estado actual y luego hacer upsert
        # ⚠️ RACE CONDITION: Si dos requests concurrentes leen el mismo count antes de que
        # cualquiera escriba, un incremento se pierde (lost update).
        # En producción, desplegar el RPC `increment_ingredient_frequencies_rpc` en Supabase
        # para garantizar atomicidad. Este fallback solo existe para desarrollo local.
        res = supabase.table("ingredient_frequencies").select("ingredient, count").eq("user_id", user_id).execute()
        current_map = {row["ingredient"]: row["count"] for row in res.data} if res.data else {}
        
        upsert_rows = []
        now_str = datetime.now(timezone.utc).isoformat()
        
        for ing, inc_val in incoming_counts.items():
            new_val = current_map.get(ing, 0) + inc_val
            upsert_rows.append({
                "user_id": user_id,
                "ingredient": ing,
                "count": new_val,
                "last_used": now_str
            })
            
        if upsert_rows:
            supabase.table("ingredient_frequencies").upsert(upsert_rows).execute()
            logger.info(f"✅ [DB] Frecuencia (Fallback Clásico) incrementada para {user_id} ({len(upsert_rows)} items)")
            
    except Exception as e:
        logger.error(f"⚠️ [DB] Error incrementando frecuecia de ingredientes: {e}")

def get_user_ingredient_frequencies(user_id: str, days_limit: int = 60) -> dict:
    """Retorna un diccionario {ingrediente_normalizado: conteo_decaimiento} de la DB.
    Implementa Decaimiento Temporal Continuo Matemático: count * (0.9 ^ dias_transcurridos).
    Se amplía days_limit a 60 días por defecto para dar margen al decaimiento suave.
    """
    if not supabase or not user_id or user_id == "guest": return {}
    try:
        from datetime import datetime, timedelta, timezone
        
        now = datetime.now(timezone.utc)
        cutoff_date = (now - timedelta(days=days_limit)).isoformat()
        
        res = supabase.table("ingredient_frequencies").select("ingredient, count, last_used").eq("user_id", user_id).gte("last_used", cutoff_date).execute()
        
        if not res.data:
            return {}
            
        freq_dict = {}
        decay_factor = 0.9  # Retiene 90% del peso por cada día que pasa sin uso
        
        for row in res.data:
            ingredient = row["ingredient"]
            count = row["count"]
            last_used_str = row.get("last_used")
            
            if not last_used_str:
                freq_dict[ingredient] = count
                continue
                
            try:
                # Parse robusto para last_used asumiendo formato de Supabase
                if last_used_str.endswith("Z"):
                    last_used_dt = datetime.fromisoformat(last_used_str[:-1]).replace(tzinfo=timezone.utc)
                else:
                    last_used_dt = datetime.fromisoformat(last_used_str)
                    if last_used_dt.tzinfo is None:
                        last_used_dt = last_used_dt.replace(tzinfo=timezone.utc)
                        
                days_elapsed = max(0, (now - last_used_dt).days)
                
                # Fórmula decaimiento matemático: count * (decay_factor ^ days_elapsed)
                decayed_count = count * (decay_factor ** days_elapsed)
                
                freq_dict[ingredient] = round(decayed_count, 2)
            except Exception as parse_e:
                logger.error(f"⚠️ [DB] Error parseando fecha {last_used_str}: {parse_e}")
                freq_dict[ingredient] = count
                
        return freq_dict
    except Exception as e:
        logger.error(f"⚠️ [DB] Error obteniendo diccionario de frecuencias: {e}")
        return {}


def format_qty(qty: float, unit: str) -> str:
    """Aplica pretty-formatting convirtiendo unidades base largas (g, ml) a legibles (kg, lt)."""
    if not unit:
        return str(int(qty)) if qty == int(qty) else f"{qty:.2f}".rstrip('0').rstrip('.')
        
    if unit == "g":
        if qty >= 1000:
            val = qty / 1000.0
            val_str = str(int(val)) if val == int(val) else f"{val:.2f}".rstrip('0').rstrip('.')
            return f"{val_str} kg"
        return f"{int(round(qty))} g"
        
    if unit == "ml":
        if qty >= 1000:
            val = qty / 1000.0
            val_str = str(int(val)) if val == int(val) else f"{val:.2f}".rstrip('0').rstrip('.')
            return f"{val_str} lt"
        return f"{int(round(qty))} ml"
        
    qty_str = str(int(qty)) if qty == int(qty) else f"{qty:.2f}".rstrip('0').rstrip('.')
    return f"{qty_str} {unit}".strip()


def deduct_inventory_items(user_id: str, consumed_ingredients: list) -> int:
    """Resta ingredientes consumidos de la lista de compras/despensa (is_checked=True)."""
    if not supabase or not consumed_ingredients or user_id == "guest": 
        return 0
        
    logger.info(f"📦 [WMS] Buscando ítems en despensa para deducir {len(consumed_ingredients)} ingredientes consumidos...")
    try:
        import unicodedata
        import re
        import json
        
        def _normalize(text: str) -> str:
            if not text: return ""
            nfkd = unicodedata.normalize('NFKD', str(text).lower().strip())
            return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))
            
        # 1. Obtener los ítems del inventario actual (solo los que están "comprados" -> en casa)
        res = supabase.table("custom_shopping_items").select("*").eq("user_id", user_id).eq("is_checked", True).execute()
        inventory = res.data if res.data else []
        
        if not inventory:
            logger.info("📦 [WMS] El usuario no tiene ingredientes marcados como comprados en la despensa.")
            return 0
            
        # Mapa para búsqueda rápida: nombre normalizado -> list de items en BD
        normalized_inventory = {}
        for row in inventory:
            name_val = row.get("display_name") or row.get("name")
            if not name_val:
                try:
                    parsed = json.loads(row.get("item_name", "{}"))
                    name_val = parsed.get("name") if isinstance(parsed, dict) else str(row.get("item_name", ""))
                except Exception:
                    name_val = str(row.get("item_name", ""))
                    
            norm_name = _normalize(name_val)
            if norm_name not in normalized_inventory:
                normalized_inventory[norm_name] = []
            normalized_inventory[norm_name].append(row)
            
        items_to_delete = []
        items_to_update = []
        deducted_count = 0
        
        # 2. Procesar cada ingrediente consumido
        for consumed_raw in consumed_ingredients:
            if not isinstance(consumed_raw, str) or not consumed_raw.strip():
                continue
                
            cons_num, cons_unit, cons_name = parse_ingredient_qty(consumed_raw, to_metric=True)
            if cons_num is None: 
                continue
                
            norm_cons_name = _normalize(cons_name)
            
            # Buscar coincidencia exacta por nombre en la despensa
            if norm_cons_name in normalized_inventory:
                candidates = normalized_inventory[norm_cons_name]
                for candidate in candidates:
                    # Traer cantidad del inventario
                    inv_raw = candidate.get("qty", "")
                    if not inv_raw:
                        continue
                        
                    # Extraer unidad y valor de lo que hay en BD (ej. "3 taza")
                    inv_num, inv_unit, _ = parse_ingredient_qty(f"{inv_raw} {norm_cons_name}", to_metric=True)
                    if inv_num is None:
                        continue
                        
                    # Si la unidad combinada coincide lógicamente
                    if (cons_unit == inv_unit) or (not cons_unit and not inv_unit):
                        new_qty = float(inv_num) - float(cons_num)
                        
                        if new_qty <= 0.05:
                            items_to_delete.append(candidate["id"])
                            logger.info(f"🗑️ [WMS] Agotado de despensa: {cons_name} (se comieron {cons_num} {cons_unit})")
                        else:
                            new_qty_label = format_qty(new_qty, inv_unit)
                            items_to_update.append({
                                "id": candidate["id"],
                                "qty": new_qty_label
                            })
                            logger.info(f"📉 [WMS] Dedcido en despensa: Quedan {new_qty_label} de {cons_name} (se comieron {cons_num} {cons_unit})")
                            
                        # Actualizamos el elemento en el diccionario en memoria para no seguir descontándole el mismo plato si hay duplicados
                        candidates.remove(candidate)
                        deducted_count += 1
                        break # Ya dedujimos de este match exitoso
                        
        # 3. Ejecutar transacciones SQL
        if items_to_delete:
            supabase.table("custom_shopping_items").delete().in_("id", items_to_delete).execute()
            
        for update_item in items_to_update:
            supabase.table("custom_shopping_items").update({"qty": update_item["qty"]}).eq("id", update_item["id"]).execute()
            
        if deducted_count > 0:
            logger.info(f"✅ [WMS] Se dedujeron con éxito {deducted_count} ingredientes consumidos del almacenamiento del usuario.")
        return deducted_count
        
    except Exception as e:
        logger.error(f"⚠️ [WMS] Error en deduct_inventory_items: {e}")
        return 0
