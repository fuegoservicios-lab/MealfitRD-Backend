import os
from supabase import create_client, Client
from dotenv import load_dotenv

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
        connection_pool = ConnectionPool(conninfo=clean_url, max_size=20, open=False)
        print("🔌 [psycopg] ConnectionPool de Postgres configurado.")
    except Exception as pool_err:
        print(f"⚠️ [psycopg] Error configurando ConnectionPool: {pool_err}")

def get_or_create_session(session_id: str, user_id: str = None):
    if not supabase: return None
    try:
        res = supabase.table("agent_sessions").select("*").eq("id", session_id).execute()
        
        if res.data and len(res.data) > 0:
            existing_session = res.data[0]
            if not existing_session.get("user_id") and user_id:
                try:
                    update_res = supabase.table("agent_sessions").update({"user_id": user_id}).eq("id", session_id).execute()
                    if update_res.data:
                        return update_res.data[0]
                except Exception as update_e:
                    print(f"Error actualizando user_id en sesión: {update_e}")
            return existing_session
        
        insert_data = {"id": session_id, "locked_at": None}
        if user_id:
            insert_data["user_id"] = user_id
            
        new_res = supabase.table("agent_sessions").insert(insert_data).execute()
        return new_res.data[0]
    except Exception as e:
        print(f"Fallback creando sesión: {e}")
        try:
            # Si falló la inserción normal (probablemente porque 'user_id' no existe en DB),
            # intentamos crear la sesión *sin* 'user_id' para no romper el chat.
            insert_data_fallback = {"id": session_id, "locked_at": None}
            new_res_fallback = supabase.table("agent_sessions").insert(insert_data_fallback).execute()
            if new_res_fallback.data:
                return new_res_fallback.data[0]
            return None
        except Exception as inner_e:
            print(f"Error fatal creando sesión: {inner_e}")
            # Si aún así falla, es posible que la sesión ya existiera pero el .select() fallara
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
        print(f"Error en get_guest_chat_sessions: {e}")
        return []

def get_user_chat_sessions(user_id: str):
    """Obtiene la lista de sesiones, ordenadas por actividad reciente."""
    if not supabase: return[]
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Fallback de seguridad si no existe user_id en la base de datos
            res = supabase.table("agent_sessions").select("*").eq("user_id", user_id).execute()
            sessions = res.data
            return _process_and_sort_sessions(sessions)
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            error_msg = str(e)
            if "42703" not in error_msg and "PGRST204" not in error_msg:
                print(f"Error en getsessions: {e}")
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
                    print(f"Error recuperando batch de mensajes para {len(session_ids)} sesiones: {db_e}")
        
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
                elif user_msgs:
                    content_str = str(user_msgs[0].get("content", ""))
                    clean_str = re.sub(r'\[IMAGE:\s*.*?\]', '', content_str)
                    if '[El usuario subió una imagen.' in clean_str:
                        match = re.search(r'Mensaje del usuario:\s*(.+)$', clean_str, re.DOTALL)
                        if match:
                            clean_str = match.group(1)
                        else:
                            clean_str = re.sub(r'\[El usuario subió una imagen\..+?\]', '', clean_str, flags=re.DOTALL)
                    clean_str = clean_str.strip()
                    
                    if clean_str:
                        s["title"] = clean_str[:40] + ("..." if len(clean_str) > 40 else "")
                    else:
                        s["title"] = "Chat con Imagen"
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
        print(f"Error en getsessions: {e}")
        return

def get_session_messages(session_id: str):
    """Obtiene todos los mensajes de una sesion, ordenados cronologicamente."""
    if not supabase: return []
    try:
        res = supabase.table("agent_messages").select("*").eq("session_id", session_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
        print(f"Error get_session_messages: {e}")
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
        print(f"Error acquiring summarizing lock: {e}")
        return False

def release_summarizing_lock(session_id: str):
    """Libera el bloqueo de resumen."""
    if not supabase: return
    try:
        supabase.table("agent_sessions").update({"locked_at": None}).eq("id", session_id).execute()
    except Exception as e:
        error_msg = str(e)
        if "Server disconnected" not in error_msg:
            print(f"Error releasing summarizing lock: {e}")

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
            print(f"Error acquiring fact lock: {e}")
        return True

def release_fact_lock(user_id: str):
    """Libera el bloqueo de extracción de hechos."""
    if not supabase: return
    try:
        supabase.table("user_profiles").update({"fact_locked_at": None}).eq("id", user_id).execute()
    except Exception as e:
        error_msg = str(e)
        if "PGRST204" not in error_msg:
            print(f"Error releasing fact lock: {e}")

def save_message(session_id: str, role: str, content: str):
    if not supabase: return None
    supabase.table("agent_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

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
        print(f"Error archivando resúmenes en cold storage: {e}")
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
        print(f"Error buscando en deep memory (summary_archive): {e}")
        return []

def delete_summaries(summary_ids: list):
    """Elimina múltiples resúmenes por sus IDs."""
    if not supabase: return None
    try:
        res = supabase.table("conversation_summaries").delete().in_("id", summary_ids).execute()
        return res.data
    except Exception as e:
        print(f"Error borrando resúmenes: {e}")
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
        print(f"Error guardando user_fact: {e}")
        return None

def delete_expired_temporal_facts(user_id: str = None, hours: int = 48):
    """Elimina los hechos con categoría 'sintoma_temporal' que son más antiguos que 'hours'."""
    if not supabase: return None
    from datetime import datetime, timedelta, timezone
    
    threshold_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    
    try:
        query = supabase.table("user_facts").delete().contains("metadata", {"categoria": "sintoma_temporal"}).lt("created_at", threshold_time)
        if user_id:
            query = query.eq("user_id", user_id)
        
        res = query.execute()
        return res.data
    except Exception as e:
        print(f"Error borrando temporal facts expirados: {e}")
        return None

def get_user_facts_by_metadata(user_id: str, key: str, value: str):
    """Busca hechos Exactos filtrando dentro del JSONB de metadata.
    Ejemplo: get_user_facts_by_metadata(user_id, 'categoria', 'alergia')
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
        print(f"Error buscando facts por metadata: {e}")
        return []

def delete_user_facts_by_metadata(user_id: str, filter_dict: dict):
    """Soft delete filtrando dentro del JSONB de metadata."""
    if not supabase: return None
    try:
        res = supabase.table("user_facts").update({"is_active": False}).eq("user_id", user_id).contains("metadata", filter_dict).execute()
        return res.data
    except Exception as e:
        print(f"Error haciendo soft delete a facts por metadata: {e}")
        return None

def search_user_facts(user_id: str, query_embedding: list, query_text: str = None, threshold: float = 0.5, limit: int = 5):
    """Busca hechos similares usando búsqueda híbrida (si hay texto) o vectorial pura en Supabase."""
    if not supabase: return []
    
    # Auto-Limpieza de síntomas temporales antes de buscar
    delete_expired_temporal_facts(user_id)
    
    try:
        if query_text:
            # Búsqueda híbrida (vector + full-text search)
            res = supabase.rpc("hybrid_search_user_facts", {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_count": limit,
                "p_user_id": user_id
            }).execute()
        else:
            # Búsqueda vectorial pura
            res = supabase.rpc("match_user_facts", {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit,
                "p_user_id": user_id
            }).execute()
        return res.data
    except Exception as e:
        print(f"Error buscando facts: {e}")
        return []

def delete_user_fact(fact_id: str):
    """Hace un soft delete cambiando is_active a False"""
    if not supabase: return None
    try:
        # En lugar de .delete(), usamos .update()
        res = supabase.table("user_facts").update({"is_active": False}).eq("id", fact_id).execute()
        return res.data
    except Exception as e:
        print(f"Error haciendo soft delete a user_fact: {e}")
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
        print(f"Error encolando hecho pendiente: {e}")
        return None

def dequeue_pending_facts(user_id: str):
    """Obtiene todos los hechos pendientes de un usuario, ordenados cronológicamente."""
    if not supabase: return []
    try:
        res = supabase.table("pending_facts_queue").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
        print(f"Error obteniendo hechos pendientes: {e}")
        return []

def delete_pending_facts(fact_ids: list):
    """Elimina los registros procesados de la cola de pendientes."""
    if not supabase or not fact_ids: return None
    try:
        res = supabase.table("pending_facts_queue").delete().in_("id", fact_ids).execute()
        return res.data
    except Exception as e:
        print(f"Error eliminando hechos pendientes procesados: {e}")
        return None

def get_all_user_facts(user_id: str):
    """Obtiene todos los hechos (facts) de un usuario para mostrarlos en la UI de Ajustes."""
    if not supabase: return []
    try:
        # Añadimos el filtro is_active
        res = supabase.table("user_facts").select("id, fact, metadata, created_at").eq("user_id", user_id).eq("is_active", True).order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        print(f"Error obteniendo ALL user facts: {e}")
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
                
                print(f"✅ [DEDUP VISUAL] Registro {str(existing_id)[:8]}... actualizado.")
                return similar.data
        except Exception as dedup_err:
            # Si la deduplicación falla (ej. columnas aún no existen), insertar normalmente
            print(f"⚠️ [DEDUP VISUAL] Error en deduplicación, insertando normalmente: {dedup_err}")
        
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
        print(f"Error guardando visual_entry: {e}")
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
        print(f"Error buscando visual_diary: {e}")
        return []

# ============================================================
# CEREBRO CENTRAL: GESTIÓN DE PERFIL UNIFICADO
# ============================================================

def get_user_profile(user_id: str):
    """Obtiene el perfil completo del usuario, incluyendo el health_profile."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").select("*").eq("id", user_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"Error obteniendo perfil: {e}")
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
        print(f"Error actualizando health_profile: {e}")
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
        print(f"Error obteniendo plan actual: {e}")
        return None

def get_recent_meals_from_plans(user_id: str, days: int = 5):
    """Obtiene una lista de nombres de comidas de los planes recientes para evitar repeticiones."""
    if not supabase: return []
    try:
        res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
        meals = set() # 👈 Usar un Set evita enviar nombres duplicados al LLM y ahorra tokens
        if res.data:
            for row in res.data:
                plan_data = row.get("plan_data", {})
                if isinstance(plan_data, dict):
                     # ✅ BÚSQUEDA CORREGIDA: Iterar sobre "days" primero y luego sobre "meals"
                     for day in plan_data.get("days", []):
                         for meal in day.get("meals", []):
                             meal_name = meal.get("name")
                             if meal_name:
                                 meals.add(meal_name)
                     # Fallback por si en la base de datos hay planes muy antiguos
                     if "meals" in plan_data:
                         for meal in plan_data.get("meals", []):
                             meal_name = meal.get("name")
                             if meal_name:
                                 meals.add(meal_name)
        return list(meals)
    except Exception as e:
        print(f"Error obteniendo comidas recientes: {e}")
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
        print(f"Error guardando comida consumida: {e}")
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
                    print(f"⚠️ Error procesando la zona horaria, usando fallback a UTC: {e}")
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
            print(f"Error obteniendo comidas consumidas de hoy: {e}")
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
        print(f"Error obteniendo plan con ID: {e}")
        return None

def update_meal_plan_data(plan_id: str, new_plan_data: dict):
    """Actualiza el plan_data JSONB de un plan existente por su ID."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").update({"plan_data": new_plan_data}).eq("id", plan_id).execute()
        return res.data
    except Exception as e:
        print(f"Error actualizando plan_data: {e}")
        return None

# ============================================================
# LISTA DE COMPRAS CUSTOM (Items añadidos por la IA)
# ============================================================

def add_custom_shopping_items(user_id: str, items: list):
    """Inserta uno o más items custom a la lista de compras del usuario."""
    if not supabase or not items: return None
    try:
        rows = [{"user_id": user_id, "item_name": item.strip()} for item in items if item.strip()]
        if rows:
            res = supabase.table("custom_shopping_items").insert(rows).execute()
            return res.data
        return None
    except Exception as e:
        print(f"Error añadiendo items a shopping list: {e}")
        return None

def get_custom_shopping_items(user_id: str):
    """Obtiene todos los items custom de la lista de compras del usuario."""
    if not supabase: return []
    try:
        res = supabase.table("custom_shopping_items").select("id, item_name, created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        print(f"Error obteniendo custom shopping items: {e}")
        return []

def delete_custom_shopping_item(item_id: str):
    """Elimina un item custom de la lista de compras."""
    if not supabase: return None
    try:
        res = supabase.table("custom_shopping_items").delete().eq("id", item_id).execute()
        return res.data
    except Exception as e:
        print(f"Error borrando custom shopping item: {e}")
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
            print(f"Error registrando api_usage: {e}")
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
            return res.count if res.count is not None else 0
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            print(f"Error obteniendo api_usage mensual: {e}")
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
    
    from agent import DOMINICAN_PROTEINS
    base_ingredient = None
    lower_meal = rejected_meal.lower()
    for p in DOMINICAN_PROTEINS:
        if p.lower() in lower_meal:
            base_ingredient = p
            break
            
    if not base_ingredient: return False
            
    profile = get_user_profile(user_id)
    if not profile: return False
    
    hp = profile.get("health_profile") or {}
    frictions = hp.get("frictions", {})
    
    current_count = frictions.get(base_ingredient, 0) + 1
    
    if current_count >= 3:
        # 3er strike -> Bloqueo automático del ingrediente base
        print(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente.")
        
        rejection_record = {
            "meal_name": base_ingredient,
            "meal_type": "Ingrediente Fricción",
            "user_id": user_id,
            "session_id": session_id if session_id else None
        }
        insert_rejection(rejection_record)
        
        # Reset counter
        frictions[base_ingredient] = 0
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        
        # Mensaje proactivo en el chat del agente
        if session_id:
            msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
            save_message(session_id, "model", msg)
        return True
    else:
        frictions[base_ingredient] = current_count
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        return False

