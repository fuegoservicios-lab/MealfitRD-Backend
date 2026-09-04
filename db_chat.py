import json
import uuid
from typing import Optional, List, Dict, Any, Tuple, cast
import os
import logging
import hashlib
import hmac
import time
logger = logging.getLogger(__name__)
from db_core import connection_pool, execute_sql_query, execute_sql_write, execute_sql_transaction
# [P2-CHAT-SAVE-MSG-RETRY · 2026-05-19] Tenacity para retry exponencial
# en `save_message`. Ya es dep declarada (requirements.txt:11 — tenacity==9.1.4)
# y usada en todo el repo para retries de DB y LLM transients.
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_not_exception_type,
    before_sleep_log,
)

# [P1-NEON-DB-MIGRATION · 2026-06-12] Módulo migrado de PostgREST (cliente legado)
# a SQL directo vía psycopg (datos viven en Neon). Los SELECT castean
# uuid/timestamptz a ::text para preservar la
# paridad de tipos con PostgREST (que devolvía JSON con strings ISO): los
# consumers hacen slicing/`.endswith`/`fromisoformat`/comparaciones string
# sobre `created_at` y usan ids como keys de dicts/sets.
_AGENT_MESSAGE_COLS_SQL = (
    "id::text AS id, session_id::text AS session_id, role, content, "
    "created_at::text AS created_at, feedback, user_id::text AS user_id, "
    "attachments, client_message_id::text AS client_message_id"
)

_last_chat_attachment_cleanup = 0.0


def _cleanup_orphan_chat_attachments() -> None:
    """Best-effort, acotado a 100 blobs y como máximo una vez por hora/proceso."""
    global _last_chat_attachment_cleanup
    now = time.monotonic()
    if now - _last_chat_attachment_cleanup < 3600:
        return
    _last_chat_attachment_cleanup = now
    try:
        execute_sql_write(
            """
            WITH stale AS (
                SELECT id FROM public.chat_attachments
                WHERE message_id IS NULL
                  AND created_at < now() - interval '24 hours'
                ORDER BY created_at
                LIMIT 100
            )
            DELETE FROM public.chat_attachments attachment
            USING stale WHERE attachment.id = stale.id
            """,
            (),
        )
    except Exception as exc:
        logger.warning("No se pudieron limpiar adjuntos huérfanos del chat: %s", exc)


def _chat_attachment_signing_secret() -> Optional[bytes]:
    value = os.environ.get("MEALFIT_CHAT_ATTACHMENT_SIGNING_SECRET") or os.environ.get("MEALFIT_SESSION_SECRET")
    if not value or len(value) < 32:
        return None
    return value.encode("utf-8")


def build_chat_attachment_url(attachment_id: str, ttl_seconds: int = 7 * 24 * 3600) -> str:
    """URL HMAC temporal. Sin secreto, el endpoint aún acepta la sesión propietaria."""
    secret = _chat_attachment_signing_secret()
    base = f"/api/chat/attachments/{attachment_id}"
    if not secret:
        return base
    expires = int(time.time()) + max(60, min(int(ttl_seconds), 30 * 24 * 3600))
    payload = f"{attachment_id}:{expires}".encode("utf-8")
    signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{base}?expires={expires}&sig={signature}"


def verify_chat_attachment_signature(attachment_id: str, expires: int, signature: str) -> bool:
    secret = _chat_attachment_signing_secret()
    if not secret or not signature or int(expires or 0) < int(time.time()):
        return False
    payload = f"{attachment_id}:{int(expires)}".encode("utf-8")
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, str(signature))


def create_chat_attachment(
    session_id: str,
    user_id: str,
    content: bytes,
    content_type: str,
    original_name: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[str]:
    """Persiste media privada del chat. Nunca devuelve el contenido en listados."""
    if not connection_pool or not session_id or not user_id or not content:
        return None
    _cleanup_orphan_chat_attachments()
    rows = execute_sql_write(
        """
        INSERT INTO public.chat_attachments
            (session_id, user_id, content, content_type, byte_size, original_name, width, height)
        SELECT %s, %s, %s, %s, %s, %s, %s, %s
        WHERE EXISTS (
            SELECT 1 FROM public.agent_sessions
            WHERE id = %s AND user_id = %s
        )
        RETURNING id::text AS id
        """,
        (
            session_id, user_id, content, content_type, len(content),
            (original_name or "")[:255] or None, width, height,
            session_id, user_id,
        ),
        returning=True,
    )
    return str(rows[0]["id"]) if rows else None


def get_chat_attachment(attachment_id: str) -> Optional[Dict[str, Any]]:
    """Obtiene una imagen por id para el endpoint privado; lookup sin SELECT *."""
    if not connection_pool:
        return None
    row = execute_sql_query(
        """
        SELECT id::text AS id, session_id::text AS session_id,
               user_id::text AS user_id, content, content_type, byte_size,
               original_name, width, height, created_at::text AS created_at
        FROM public.chat_attachments WHERE id = %s
        """,
        (attachment_id,),
        fetch_one=True,
    )
    return cast(Optional[Dict[str, Any]], row)


def _owned_attachment_metadata(
    attachment_ids: List[str], session_id: str, user_id: str,
    client_message_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not attachment_ids:
        return []
    rows = execute_sql_query(
        """
        SELECT id::text AS attachment_id, original_name AS name, content_type,
               byte_size, width, height
        FROM public.chat_attachments
        WHERE id = ANY(%s::uuid[]) AND session_id = %s AND user_id = %s
          AND (
              message_id IS NULL OR message_id = (
                  SELECT id FROM public.agent_messages
                  WHERE session_id = %s AND client_message_id = %s
              )
          )
        """,
        (attachment_ids, session_id, user_id, session_id, client_message_id),
        fetch_all=True,
    ) or []
    by_id = {str(row["attachment_id"]): row for row in rows}
    return [by_id[attachment_id] for attachment_id in attachment_ids if attachment_id in by_id]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_not_exception_type(ValueError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def save_message_with_attachments(
    session_id: str,
    content: str,
    user_id: str,
    attachment_ids: List[str],
    client_message_id: Optional[str] = None,
    vision_items: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """INSERT idempotente del turno + claim de hasta 4 adjuntos propios."""
    if not connection_pool:
        return ""
    ordered_ids = list(dict.fromkeys(str(value) for value in attachment_ids if value))[:4]
    metadata = _owned_attachment_metadata(ordered_ids, session_id, user_id, client_message_id)
    if len(metadata) != len(ordered_ids):
        raise ValueError("Uno o más adjuntos no pertenecen a esta sesión")
    contexts = vision_items if isinstance(vision_items, list) else []
    allowed_kinds = {"plato", "items", "otro", "unavailable"}
    for index, item in enumerate(metadata):
        context = contexts[index] if index < len(contexts) and isinstance(contexts[index], dict) else {}
        kind = str(context.get("kind") or "")
        if kind in allowed_kinds:
            item["kind"] = kind
        description = str(context.get("description") or "").strip()[:1500]
        if description:
            item["description"] = description
        reason = str(context.get("reason") or "")
        if reason in {"busy", "down"}:
            item["reason"] = reason

    message_id = str(uuid.uuid4())
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    insert_sql = """
        INSERT INTO public.agent_messages
            (id, session_id, role, content, user_id, attachments, client_message_id)
        VALUES (%s, %s, 'user', %s, %s, %s::jsonb, %s)
        ON CONFLICT (session_id, client_message_id)
            WHERE client_message_id IS NOT NULL
        DO NOTHING
    """
    claim_sql = """
        UPDATE public.chat_attachments
        SET message_id = COALESCE(
                (SELECT id FROM public.agent_messages
                 WHERE session_id = %s AND client_message_id = %s),
                %s::uuid
            ),
            claimed_at = COALESCE(claimed_at, now())
        WHERE id = ANY(%s::uuid[]) AND session_id = %s AND user_id = %s
          AND message_id IS NULL
    """
    execute_sql_transaction([
        (insert_sql, (message_id, session_id, content, user_id, metadata_json, client_message_id)),
        (claim_sql, (session_id, client_message_id, message_id, ordered_ids, session_id, user_id)),
    ])

    try:
        from proactive_agent import handle_nudge_response
        handle_nudge_response(user_id, content)
    except Exception as exc:
        logger.error(f"Error procesando respuesta al nudge en save_message_with_attachments: {exc}")

    if client_message_id:
        existing = execute_sql_query(
            "SELECT id::text AS id FROM public.agent_messages WHERE session_id = %s AND client_message_id = %s",
            (session_id, client_message_id),
            fetch_one=True,
        )
        if existing:
            return str(existing["id"])
    return message_id

def delete_user_agent_sessions(user_id: str) -> bool:
    """Elimina todas las sesiones de agente para un usuario."""
    if not connection_pool: return False
    try:
        # FKs de agent_messages/conversation_summaries son ON DELETE CASCADE.
        execute_sql_write("DELETE FROM public.agent_sessions WHERE user_id = %s", (user_id,))
        return True
    except Exception as e:
        logger.error(f"Error eliminando sesiones de agente de {user_id}: {e}")
        return False

def delete_single_agent_session(session_id: str) -> bool:
    """Elimina una sesión específica de agente."""
    if not connection_pool: return False
    try:
        execute_sql_write("DELETE FROM public.agent_sessions WHERE id = %s", (session_id,))
        return True
    except Exception as e:
        logger.error(f"Error eliminando sesion de agente {session_id}: {e}")
        return False

def update_session_title(session_id: str, new_title: str) -> bool:
    """Actualiza el título de una sesión."""
    if not connection_pool: return False
    try:
        # Buscar si ya existe el mensaje de titulo
        res = execute_sql_query(
            "SELECT id::text AS id FROM public.agent_messages "
            "WHERE session_id = %s AND content LIKE %s",
            (session_id, "[SYSTEM_TITLE] %"),
            fetch_all=True,
        )
        if res and len(res) > 0:
            msg_id = res[0]["id"]
            execute_sql_write(
                "UPDATE public.agent_messages SET content = %s WHERE id = %s",
                (f"[SYSTEM_TITLE] {new_title}", msg_id),
            )
        else:
            # Si no existe, insertar un mensaje especial al inicio (fecha muy antigua o la misma de la sesion)
            save_message(session_id, "model", f"[SYSTEM_TITLE] {new_title}")
        return True
    except Exception as e:
        logger.error(f"Error actualizando titulo de sesion {session_id}: {e}")
        return False

def get_or_create_session(session_id: str, user_id: Optional[str] = None):
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
    """Obtiene la lista de sesiones para invitados, mediante sus IDs.

    [P1-CHAT-GUEST-IDOR · 2026-05-30] El filtro `.is_("user_id", "null")` es
    OBLIGATORIO por seguridad. Este path recupera sesiones por id crudo (los
    `session_ids` vienen del localStorage del cliente, sin verificar ownership),
    así que SIN el filtro un usuario autenticado podía llamar
    `GET /api/chat/sessions/<su_uid>?session_ids=<sesión_ajena>` y recibir la
    sesión + el snippet del primer mensaje de OTRO usuario (IDOR + leak de PII).
    Solo las sesiones genuinas de invitado (sin dueño, `user_id IS NULL`) deben
    ser recuperables por id crudo; las sesiones propias de un usuario logueado
    ya las trae `get_user_chat_sessions(user_id)` filtrando por owner. Una
    sesión creada como guest y luego "reclamada" al loguearse queda con
    `user_id` no-nulo → sale por el path de owner, no por este. Tooltip-anchor:
    P1-CHAT-GUEST-IDOR (no remover el `user_id IS NULL`).
    """
    if not connection_pool or not session_ids: return []
    try:
        # [P1-CHAT-GUEST-IDOR] el predicado `user_id IS NULL` es OBLIGATORIO:
        # solo sesiones sin dueño son recuperables por id crudo (no IDOR).
        sessions = execute_sql_query(
            "SELECT id::text AS id, created_at::text AS created_at, "
            "locked_at::text AS locked_at, user_id::text AS user_id "
            "FROM public.agent_sessions "
            "WHERE id = ANY(%s::uuid[]) AND user_id IS NULL",
            ([str(s) for s in session_ids[:20]],),
            fetch_all=True,
        )
        return _process_and_sort_sessions(sessions or [])
    except Exception as e:
        logger.error(f"Error en get_guest_chat_sessions: {e}")
        return []

# [P2-CHAT-SESSIONS-PAGING · 2026-09-03] Tope y página de la lista de Recientes. Antes
# `LIMIT 60` fijo: el chat 61 dejaba de existir para el usuario sin aviso ni forma de
# llegar a él. Ahora el caller pagina por `offset` y `count_user_chat_sessions` le dice
# si queda más. Tooltip-anchor: P2-CHAT-SESSIONS-PAGING.
CHAT_SESSIONS_PAGE_SIZE = 60


def count_user_chat_sessions(user_id: str) -> int:
    """Total de sesiones del usuario (para `has_more`). 0 si la DB falla: fail-quiet."""
    if not connection_pool or not user_id:
        return 0
    try:
        row = execute_sql_query(
            "SELECT count(*) AS total FROM public.agent_sessions WHERE user_id = %s",
            (user_id,),
            fetch_one=True,
        )
        return int((row or {}).get("total") or 0)
    except Exception as e:
        logger.error(f"Error contando sesiones: {e}")
        return 0


def get_user_chat_sessions(user_id: str, limit: int = CHAT_SESSIONS_PAGE_SIZE, offset: int = 0):
    """Obtiene la lista de sesiones, ordenadas por actividad reciente."""
    if not connection_pool: return[]

    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Fallback de seguridad si no existe user_id en la base de datos
            sessions = execute_sql_query(
                "SELECT id::text AS id, created_at::text AS created_at, "
                "locked_at::text AS locked_at, user_id::text AS user_id "
                "FROM public.agent_sessions WHERE user_id = %s "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (user_id, int(limit), int(offset)),
                fetch_all=True,
            )
            return _process_and_sort_sessions(sessions or [])
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
                # [P1-NEON-DB-MIGRATION · 2026-06-12] session_id::text para que
                # las keys de `messages_by_session` matcheen los ids (también
                # text) de las sesiones; created_at::text para el sort string.
                msg_res = execute_sql_query(
                    "SELECT session_id::text AS session_id, content, "
                    "created_at::text AS created_at, role "
                    "FROM public.agent_messages "
                    "WHERE session_id = ANY(%s::uuid[]) "
                    "ORDER BY created_at ASC",
                    (session_ids,),
                    fetch_all=True,
                )
                if msg_res:
                    all_messages = msg_res
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
                
                # [P1-I18N-CHAT-TITULOS-SERVIDOR · 2026-08-22] El backend deja de decidir el
                # TEXTO de los rótulos genéricos y devuelve un DISCRIMINADOR (`title_key`).
                #
                # Antes componía «Nuevo Chat» / «Interacción con imagen o sistema» en español
                # y el cliente los pintaba crudos: con la app en inglés, la barra lateral del
                # Agente listaba «Nuevo Chat», «Nuevo Chat»… bajo un encabezado de grupo que
                # SÍ decía «Today». Medido en producción: 171 de 186 sesiones vivas.
                #
                # `title` se conserva SÓLO cuando es contenido real —el `[SYSTEM_TITLE]`, que
                # ya nace en el idioma del usuario vía `build_title_language_directive`, o el
                # texto que el usuario escribió—. Un rótulo genérico no es contenido: es
                # interfaz, y la interfaz la traduce el cliente, que es quien sabe el idioma
                # activo en ese instante (el perfil puede haber cambiado desde el último
                # mensaje). `title: None` + `title_key` deja esa decisión donde corresponde.
                if title_msgs:
                    s["title"] = str(title_msgs[0].get("content", "")).replace("[SYSTEM_TITLE] ", "")
                    s["title_key"] = None
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
                        
                    if clean_str:
                        # Contenido REAL del usuario: viaja tal cual — lo que el usuario
                        # escribió no se traduce.
                        s["title"] = clean_str[:45] + ("..." if len(clean_str) > 45 else "")
                        s["title_key"] = None
                    else:
                        # Lo que queda tras limpiar los prefijos de sistema no es texto del
                        # usuario: es «aquí no hubo nada legible», o sea interfaz.
                        s["title"] = None
                        s["title_key"] = "image_or_system"
                    # Como no hay SYSTEM_TITLE, es fallback
                    s["is_fallback"] = True
                else:
                    s["title"] = None
                    s["title_key"] = "empty"
                    
                s["last_activity"] = s_msgs[-1].get("created_at", s.get("created_at", ""))
                valid_sessions.append(s)
            else:
                s["title"] = None
                s["title_key"] = "empty"
                s["last_activity"] = s.get("created_at", "")
                valid_sessions.append(s)
                
        # Ordenar los chats para que el más reciente suba al primer lugar
        valid_sessions.sort(key=lambda x: x.get("last_activity") or x.get("created_at") or "1970-01-01T00:00:00", reverse=True)
        return valid_sessions
    except Exception as e:
        logger.error(f"Error en getsessions: {e}")
        return

def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Obtiene todos los mensajes de una sesion, ordenados cronologicamente."""
    if not connection_pool: return []
    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = execute_sql_query(
                f"SELECT {_AGENT_MESSAGE_COLS_SQL} FROM public.agent_messages "
                "WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,),
                fetch_all=True,
            )
            return cast(List[Dict[str, Any]], res or [])
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.3)
                continue
            logger.error(f"Error get_session_messages: {e}")
            return []
    return []  # defensivo: range(max_retries) siempre itera (max_retries=2), inalcanzable

def acquire_summarizing_lock(session_id: str) -> bool:
    """Intenta adquirir el bloqueo para resumir. Retorna True si lo logra, False si ya está bloqueado."""
    if not connection_pool: return True
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)

        # Verificar estado actual. [P1-NEON-DB-MIGRATION · 2026-06-12]
        # psycopg devuelve `locked_at` como datetime tz-aware nativo —
        # ya no hace falta el parseo string ISO de PostgREST.
        res = execute_sql_query(
            "SELECT locked_at FROM public.agent_sessions WHERE id = %s",
            (session_id,),
            fetch_one=True,
        )
        if res:
            locked_at = res.get("locked_at")
            # Si el lock tiene menos de 5 minutos, está bloqueado
            if locked_at is not None and (now - locked_at) < timedelta(minutes=5):
                return False

        # Intentar establecer el timestamp
        update_res = execute_sql_write(
            "UPDATE public.agent_sessions SET locked_at = %s WHERE id = %s RETURNING id",
            (now, session_id),
            returning=True,
        )

        return bool(update_res)
    except Exception as e:
        logger.error(f"Error acquiring summarizing lock: {e}")
        return False

def release_summarizing_lock(session_id: str):
    """Libera el bloqueo de resumen."""
    if not connection_pool: return
    try:
        execute_sql_write(
            "UPDATE public.agent_sessions SET locked_at = NULL WHERE id = %s",
            (session_id,),
        )
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
# multiplier 2, max 4s). Vector cerrado: si la DB tiene un blip transient
# mid-stream (network jitter, momentary 5xx, connection pool exhausto),
# pre-fix el `.execute()` levantaba la excepción → el caller
# en routers/chat.py la atrapaba como `Error en bg tasks` (warning) y
# el flujo seguía. El usuario veía la respuesta del agente RENDERIZADA
# en su pantalla pero la respuesta NO existía en `agent_messages` →
# mensaje fantasma. Al refrescar la sesión, el LLM perdía contexto del
# turno anterior y respondía como si nada hubiera pasado.
#
# Retry específico para excepciones de DB/red: NO atrapamos
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
    para que tenacity sólo envuelva el roundtrip a la DB. El side-effect
    `handle_nudge_response` queda FUERA — su falla no debe disparar
    re-INSERT (re-procesaría el nudge response múltiples veces).

    [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] `user_id` se persiste en la
    columna nueva (puede ser None para guests — legítimo). RLS policies
    `authenticated_*_own_messages` enforzan `auth.uid() = user_id` para
    callsites con anon/authenticated keys; service_role (que es lo que
    usa el backend) bypassea RLS.

    Tooltip-anchor: P2-CHAT-SAVE-MSG-RETRY + P1-CHAT-DB-USER-ID-RLS."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] INSERT directo con named params
    # (psycopg acepta dict + placeholders %(name)s).
    execute_sql_write(
        "INSERT INTO public.agent_messages (session_id, role, content, user_id) "
        "VALUES (%(session_id)s, %(role)s, %(content)s, %(user_id)s)",
        {  # pyright: ignore[reportArgumentType]
            "session_id": session_id,
            "role": role,
            "content": content,
            "user_id": user_id,
        },
    )


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
    if not connection_pool: return None

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
    # 3 fallos consecutivos a la DB es un incidente real, no transient.
    _save_message_insert_with_retry(session_id, role, content, user_id)


def get_model_response_id_for_regeneration(
    session_id: str,
    *,
    message_id: Optional[str] = None,
    content: Optional[str] = None,
) -> Optional[str]:
    """Resuelve exactamente la respuesta del modelo que será regenerada.

    El ``session_id`` ya pasó el control de ownership del router. Se exige
    además role=model y pertenencia a esa sesión; el fallback por contenido
    permite regenerar una respuesta recién emitida cuyo id todavía no conoce
    el cliente. No modifica la fila: la respuesta previa se conserva si falla
    la generación nueva.
    """
    if not connection_pool:
        return None
    if message_id:
        row = execute_sql_query(
            "SELECT id::text AS id FROM public.agent_messages "
            "WHERE id = %s AND session_id = %s AND role = 'model'",
            (message_id, session_id),
            fetch_one=True,
        )
        if row:
            return str(row["id"])
    if not content:
        return None
    row = execute_sql_query(
        "SELECT id::text AS id FROM public.agent_messages "
        "WHERE session_id = %s AND role = 'model' AND content = %s "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (session_id, content),
        fetch_one=True,
    )
    return str(row["id"]) if row else None


def replace_model_response_for_regeneration(
    session_id: str,
    message_id: str,
    content: str,
) -> bool:
    """Sustituye la respuesta solo después de generar la nueva con éxito."""
    rows = execute_sql_write(
        "UPDATE public.agent_messages SET content = %s, feedback = NULL "
        "WHERE id = %s AND session_id = %s AND role = 'model' RETURNING id",
        (content, message_id, session_id),
        returning=True,
    )
    return bool(rows)

def save_message_feedback(session_id: str, content: str, feedback: Optional[str]):
    """Guarda o remueve la retroalimentación (up/down/null) para un mensaje del modelo."""
    if not connection_pool: return False
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

        res = execute_sql_write(
            "UPDATE public.agent_messages SET feedback = %s "
            "WHERE session_id = %s AND role = 'model' AND content ILIKE %s "
            "RETURNING id",
            (feedback, session_id, search_pattern),
            returning=True,
        )

        rows_affected = len(res) if res else 0
        logger.info(f"Feedback guardado, rows affected: {rows_affected}")

        # Si aun así falla, intenta actualizar el ULTIMO mensaje del modelo en la sesion
        if rows_affected == 0:
            logger.warning("Fallo el match por sufijo. Actualizando el ultimo mensaje de modelo de esta sesion.")
            last_msg = execute_sql_query(
                "SELECT id::text AS id FROM public.agent_messages "
                "WHERE session_id = %s AND role = 'model' "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
                fetch_one=True,
            )
            if last_msg:
                execute_sql_write(
                    "UPDATE public.agent_messages SET feedback = %s WHERE id = %s",
                    (feedback, last_msg["id"]),
                )
                rows_affected = 1
            else:
                logger.warning("No hay mensajes previos del modelo. Insertando el mensaje (posiblemente bienvenida autogenerada).")
                execute_sql_write(
                    "INSERT INTO public.agent_messages (session_id, role, content, feedback) "
                    "VALUES (%s, 'model', %s, %s)",
                    (session_id, content, feedback),
                )
                rows_affected = 1

        return rows_affected > 0
    except Exception as e:
        logger.error(f"Error saving message feedback: {e}")
        return False

def get_memory(session_id: str):
    if not connection_pool: return []
    res = execute_sql_query(
        f"SELECT {_AGENT_MESSAGE_COLS_SQL} FROM public.agent_messages "
        "WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,),
        fetch_all=True,
    )
    return res or []

def save_summary(session_id: str, summary: str, messages_start: str, messages_end: str, message_count: int):
    """Guarda un resumen de conversación en la tabla conversation_summaries."""
    if not connection_pool: return None
    res = execute_sql_write(
        "INSERT INTO public.conversation_summaries "
        "(session_id, summary, messages_start, messages_end, message_count) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (session_id, summary, messages_start, messages_end, message_count),
        returning=True,
    )
    return res

def get_summaries(session_id: str):
    """Obtiene todos los resúmenes de una sesión, ordenados cronológicamente."""
    if not connection_pool: return []
    res = execute_sql_query(
        "SELECT id::text AS id, session_id::text AS session_id, summary, "
        "messages_start::text AS messages_start, messages_end::text AS messages_end, "
        "message_count, created_at::text AS created_at, user_id::text AS user_id "
        "FROM public.conversation_summaries "
        "WHERE session_id = %s ORDER BY messages_start ASC",
        (session_id,),
        fetch_all=True,
    )
    return res or []

def archive_summaries(summaries_list: list):
    """Guarda (archiva) una lista de resúmenes en cold storage antes de borrarlos.

    Retorna lista truthy si el INSERT confirmó, None si falló — el caller
    (P3-SUMMARY-ARCHIVE-GUARD en memory_manager) gatea el delete en esto."""
    if not connection_pool or not summaries_list: return None
    try:
        data_to_insert = []
        for s in summaries_list:
            data_to_insert.append((
                # [P1-NEON-DB-MIGRATION · 2026-06-12] summary_archive.session_id
                # es TEXT (no uuid): normalizar a str para el INSERT.
                str(s.get("session_id")) if s.get("session_id") is not None else None,
                s.get("summary"),
                s.get("messages_start"),
                s.get("messages_end"),
                s.get("message_count"),
                s.get("created_at"),
            ))
        if data_to_insert:
            values_sql = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * len(data_to_insert))
            params = tuple(v for row in data_to_insert for v in row)
            res = execute_sql_write(
                "INSERT INTO public.summary_archive "
                "(session_id, summary, messages_start, messages_end, message_count, original_created_at) "
                f"VALUES {values_sql} RETURNING id",
                params,
                returning=True,
            )
            return res
        return None
    except Exception as e:
        logger.error(f"Error archivando resúmenes en cold storage: {e}")
        return None

def search_deep_memory(user_id: str, query: str, limit: int = 5):
    """
    Busca en la memoria profunda del usuario. Capa legacy: ILIKE sobre el archivo
    frío (summary_archive). [P1-DREAMING-1 · 2026-06-13] Capa nueva (gateada por
    MEALFIT_DREAMING_RETRIEVAL_ENABLED): antepone el "modelo del usuario"
    consolidado por el Dreaming via match semántico (RPC match_user_memory).
    Fail-secure: sin COHERE_API_KEY o cualquier error → solo el ILIKE legacy.
    """
    if not connection_pool: return []
    deep: list = []

    # --- Capa Dreaming (vectorial): el user_model de alto nivel ---
    try:
        import dreaming
        if dreaming._dreaming_retrieval_enabled():
            from fact_extractor import get_embedding
            emb = get_embedding(query, purpose="query")  # asimetría: lado de búsqueda
            if emb:
                emb_str = f"[{','.join(map(str, emb))}]"
                rows = execute_sql_query(
                    "SELECT user_model AS summary, 'consolidado' AS messages_start, "
                    "'memoria' AS messages_end, 0 AS message_count, "
                    "now()::text AS original_created_at "
                    "FROM match_user_memory(query_embedding => %s::extensions.vector, "
                    "match_threshold => %s, match_count => %s, p_user_id => %s)",
                    (emb_str, 0.20, 1, user_id),
                    fetch_all=True,
                ) or []
                deep.extend(rows)
    except Exception as e:
        logger.debug(f"[P1-DREAMING-1] branch vectorial de deep memory falló (fallback ILIKE): {e}")

    # --- Capa legacy (ILIKE sobre el archivo frío) ---
    try:
        # session_ids del usuario (::text — summary_archive.session_id es TEXT)
        sessions_res = execute_sql_query(
            "SELECT id::text AS id FROM public.agent_sessions WHERE user_id = %s",
            (user_id,),
            fetch_all=True,
        )
        if sessions_res:
            session_ids = [s["id"] for s in sessions_res]
            res = execute_sql_query(
                "SELECT summary, messages_start::text AS messages_start, "
                "messages_end::text AS messages_end, message_count, "
                "original_created_at::text AS original_created_at "
                "FROM public.summary_archive "
                "WHERE session_id = ANY(%s::text[]) AND summary ILIKE %s "
                "ORDER BY original_created_at DESC LIMIT %s",
                (session_ids, f"%{query}%", limit),
                fetch_all=True,
            )
            deep.extend(res or [])
    except Exception as e:
        logger.error(f"Error buscando en deep memory (summary_archive): {e}")

    return deep[:limit]

def delete_summaries(summary_ids: list):
    """Elimina múltiples resúmenes por sus IDs."""
    if not connection_pool: return None
    try:
        res = execute_sql_write(
            "DELETE FROM public.conversation_summaries WHERE id = ANY(%s::uuid[]) RETURNING id",
            ([str(x) for x in summary_ids],),
            returning=True,
        )
        return res
    except Exception as e:
        logger.error(f"Error borrando resúmenes: {e}")
        return None

def delete_old_messages(session_id: str, before_timestamp: str):
    """Elimina mensajes de agent_messages anteriores o iguales al timestamp dado."""
    if not connection_pool: return None
    res = execute_sql_write(
        "DELETE FROM public.agent_messages WHERE session_id = %s AND created_at <= %s RETURNING id",
        (session_id, before_timestamp),
        returning=True,
    )
    return res


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
    cuando el gateway/PostgREST cerraba la conexión HTTP/2 idle (proveedor
    legado). La primera request post-idle fallaba pero httpx abría socket nuevo
    para la segunda. Tenacity re-raise tras agotar intentos para que callers
    (build_memory_context → chat_with_agent_stream) puedan propagar la
    excepción si persiste.

    Conservador: retry_if_exception_type=Exception es amplio pero stop=2
    limita el blast radius (max 1 reintento). Si el error es genuino (parse,
    schema mismatch), el segundo intento también falla y el caller lo sabe.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Post-migración a SQL directo el
    retry sigue siendo útil: cubre blips transient del pool psycopg contra
    Neon (connection reset, pooler EOF) con la misma semántica.
    """
    if not connection_pool: return []
    res = execute_sql_query(
        f"SELECT {_AGENT_MESSAGE_COLS_SQL} FROM public.agent_messages "
        "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
        (session_id, limit),
        fetch_all=True,
    )
    # Revertir el orden para que estén cronológicos (el query los trae desc)
    return list(reversed(res)) if res else []

# [P1-NEON-DB-MIGRATION · 2026-06-12] insert_like/insert_rejection reciben
# dicts construidos por callers (incl. el body crudo del endpoint /like) —
# whitelist de columnas para construir el INSERT sin riesgo de injection
# por keys arbitrarias (paridad con PostgREST, que rechazaba columnas
# desconocidas; aquí se filtran silenciosamente).
def insert_like(like_data: dict):
    if not connection_pool: return None
    allowed = ("user_id", "meal_name", "meal_type")
    cols = [c for c in allowed if c in like_data]
    if not cols:
        return None
    placeholders = ", ".join(["%s"] * len(cols))
    res = execute_sql_write(
        f"INSERT INTO public.meal_likes ({', '.join(cols)}) VALUES ({placeholders}) RETURNING id",
        tuple(like_data[c] for c in cols),
        returning=True,
    )
    return res

def get_user_likes(user_id: str):
    if not connection_pool: return []
    # Fetch all liked meals for this user
    res = execute_sql_query(
        "SELECT meal_name, meal_type FROM public.meal_likes WHERE user_id = %s",
        (user_id,),
        fetch_all=True,
    )
    return res or []

def insert_rejection(rejection_data: dict):
    """Guarda un rechazo de comida con timestamp automático."""
    if not connection_pool: return None
    allowed = ("user_id", "session_id", "meal_name", "meal_type")
    cols = [c for c in allowed if c in rejection_data]
    if not cols:
        return None
    placeholders = ", ".join(["%s"] * len(cols))
    res = execute_sql_write(
        f"INSERT INTO public.meal_rejections ({', '.join(cols)}) VALUES ({placeholders}) RETURNING id",
        tuple(rejection_data[c] for c in cols),
        returning=True,
    )
    return res

def get_active_rejections(user_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Obtiene todos los rechazos permanentes del usuario.
    Una vez que el usuario rechaza un plato, nunca vuelve a sugerirse.
    """
    if not connection_pool: return []

    # rejected_at::text — los consumers tratan los rows como JSON-serializables
    # (paridad PostgREST; datetime nativo rompería json.dumps en prompts).
    if user_id:
        where_clause, params = "user_id = %s", (user_id,)
    elif session_id:
        where_clause, params = "session_id = %s", (session_id,)
    else:
        return []

    res = execute_sql_query(
        "SELECT meal_name, meal_type, rejected_at::text AS rejected_at "
        f"FROM public.meal_rejections WHERE {where_clause} ORDER BY rejected_at DESC",
        params,
        fetch_all=True,
    )
    return res or []

