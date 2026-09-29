from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from error_utils import safe_error_detail
from typing import Optional
import hashlib
import logging
import traceback
import json

from auth import get_verified_user_id, verify_api_quota, verify_coach_quota, coach_quota_snapshot
from path_validators import assert_valid_uuid
from rate_limiter import RateLimiter
from db import (
    get_user_chat_sessions, get_guest_chat_sessions, get_session_owner, delete_user_agent_sessions,
    delete_single_agent_session, update_session_title, get_session_messages, get_or_create_session,
    save_message, save_message_with_attachments, save_message_feedback, log_api_usage,
    get_model_response_id_for_regeneration, replace_model_response_for_regeneration,
    get_chat_attachment, build_chat_attachment_url, verify_chat_attachment_signature,
    count_user_chat_sessions, CHAT_SESSIONS_PAGE_SIZE,
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import generate_chat_title_background, chat_with_agent, chat_with_agent_stream, LLMCircuitBreakerOpen, LLMRateLimitedError, strip_ui_action_tags_for_persist, is_turn_active
from services import merge_form_data_with_profile
from db_profiles import get_user_profile
from db_plans import get_latest_meal_plan
from fact_extractor import async_extract_and_save_facts
# [P1-PLAN-LOTE-717 · 2026-09-28] La lectura del interruptor de memoria, fail-CLOSED (SSOT en su módulo).
from memoria_largo_plazo import memoria_activa
# [P1-BG-THREAD-TIMEOUT · 2026-05-15] SSOT para fire-and-forget con timeout
# duro + alert. Reemplaza los `threading.Thread(target=..., daemon=True).start()`
# que vivían inline en este router. Ver `backend/bg_executor.py`.
from bg_executor import submit_bg_task
# [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Helper SSOT del registry de knobs
# `MEALFIT_*`. Cualquier env var leída aquí se auto-registra en
# `_KNOBS_REGISTRY` y es visible en `/health/version`.
from knobs import _env_int, _env_float

logger = logging.getLogger(__name__)


# [P1-CHAT-LOG-CTX · 2026-05-19] Logger correlacionable por (session_id,
# user_id) para incidentes reportados por usuarios. Pre-fix: cada log line
# del chat-flow (router + stream + tools) usaba el logger crudo del módulo
# sin contexto. Un user reporta "mi chat no funcionó hace 10min" → grep en
# Sentry/CloudWatch revela `session_id` solo en algunos logs (los que el
# autor del log decidió incluir manualmente). Reconstruir la cronología
# requiere correlación visual + suerte. Con `LoggerAdapter`, cada record
# carga `extra={'session_id': '...', 'user_id_hash': '...'}` y Sentry/
# OpenSearch puede filtrar por esos atributos sin que cada log explicite.
#
# `user_id` se hashea (SHA-256[:12]) en endpoints públicos del chat —
# mismo patrón canónico que `routers/system.py::_hash_uuid_for_public()`
# (P2-HEALTH-UID-STRIP · 2026-05-12). El log queda grep-friendly sin
# leakeo de UUIDs raw a sinks de retención larga (Sentry retención ~30d,
# data residency cross-region).
#
# Tooltip-anchor: P1-CHAT-LOG-CTX.


def _hash_user_id_for_log(user_id: Optional[str]) -> str:
    """[P1-CHAT-LOG-CTX · 2026-05-19] Hash determinístico del user_id
    para inyectar en log records. Devuelve `"guest"` para guests/None
    (NO hasheamos esos casos — son legítimamente públicos)."""
    if not user_id or user_id == "guest":
        return "guest"
    try:
        return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def _chat_logger(session_id: Optional[str], user_id: Optional[str]) -> logging.LoggerAdapter:
    """[P1-CHAT-LOG-CTX · 2026-05-19] LoggerAdapter con contexto
    correlacionable. Uso: `clog = _chat_logger(session_id, user_id); clog.info(...)`.
    Cualquier sink estructurado (Sentry, OpenSearch, Datadog) verá los
    atributos en `record.__dict__` y los puede filtrar/agregar."""
    return logging.LoggerAdapter(
        logger,
        extra={
            "session_id": session_id or "unknown",
            "user_id_hash": _hash_user_id_for_log(user_id),
        },
    )


# [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap de longitud del texto que el
# usuario envía al chat. Aplica a:
#   - `/api/chat/stream` (campo `prompt`): texto que el LLM consumirá.
#   - `/api/chat/message` (campo `content`): mensaje persistido en
#     `agent_messages` (rol `user` o `model` desde el cliente).
#
# Pre-fix: ninguno de los dos endpoints validaba longitud. Vectores:
#   (a) DoS económico — un autenticado envía 100KB → Gemini consume tokens
#       del owner desproporcionados al payload útil; bajo abuso sostenido,
#       cuota mensual del provider se agota.
#   (b) Context window saturation — Gemini Flash 3.5 acepta ~1M tokens
#       pero el LLM gasta tiempo+latencia procesando el blob; el endpoint
#       puede colgar hasta el timeout total-graph (60s, P0-CHAT-LLM-TIMEOUT).
#   (c) Storage abuse — `/message` permite insertar texto arbitrario en
#       `agent_messages.content` (text, sin cap DB). Un user puede crecer
#       la tabla sin facturar quota LLM.
#
# Defaults:
#   - 8192 chars (~8KB) cubre el 99.9% de mensajes legítimos en chat
#     conversacional español. Voice mode TTS está aún más acotado por su
#     propio cap de 1500 chars (P1-CHAT-TTS-1). Mensajes que necesiten
#     >8KB legítimamente son extremadamente raros (un copy-paste de receta
#     full o de paper técnico cabe en 8KB).
#   - Clamp [256, 65536]: el límite inferior evita env vars patológicas
#     que rompan el chat completo (256 cubre saludo + pregunta corta);
#     el superior bloquea caps absurdos (>64KB ya es archivo, no chat).
#
# Knob: `MEALFIT_CHAT_PROMPT_MAX_CHARS` (auto-registrado). HTTPException
# 413 (PAYLOAD_TOO_LARGE) es el código semánticamente correcto.
# Tooltip-anchor: P0-CHAT-PROMPT-MAXLEN.
_CHAT_PROMPT_MAX_CHARS_DEFAULT = 8192
_CHAT_PROMPT_MAX_CHARS_CLAMP_MIN = 256
_CHAT_PROMPT_MAX_CHARS_CLAMP_MAX = 65536


def _chat_prompt_max_chars() -> int:
    """[P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap actual aplicado por
    `_enforce_chat_prompt_cap`. Lee `MEALFIT_CHAT_PROMPT_MAX_CHARS` con
    clamp defensivo. Tooltip-anchor: P0-CHAT-PROMPT-MAXLEN."""
    raw = _env_int("MEALFIT_CHAT_PROMPT_MAX_CHARS", _CHAT_PROMPT_MAX_CHARS_DEFAULT)
    if raw < _CHAT_PROMPT_MAX_CHARS_CLAMP_MIN:
        return _CHAT_PROMPT_MAX_CHARS_CLAMP_MIN
    if raw > _CHAT_PROMPT_MAX_CHARS_CLAMP_MAX:
        return _CHAT_PROMPT_MAX_CHARS_CLAMP_MAX
    return raw


def _enforce_chat_prompt_cap(value, field_name: str = "prompt") -> None:
    """[P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Levanta HTTP 413 si el texto
    excede el cap configurado. `None`/`""` pasan sin error (validación
    de "missing" pertenece al caller). Tooltip-anchor: P0-CHAT-PROMPT-MAXLEN."""
    if not value:
        return
    if not isinstance(value, str):
        return
    n = len(value)
    cap = _chat_prompt_max_chars()
    if n > cap:
        logger.warning(
            f"[P0-CHAT-PROMPT-MAXLEN] rechazado field={field_name} len={n} cap={cap}"
        )
        raise HTTPException(
            status_code=413,
            detail=f"Mensaje demasiado largo ({n} caracteres). Máximo permitido: {cap} caracteres.",
        )


def _resolve_user_id_for_db(
    user_id_input: Optional[str], session_id: Optional[str]
) -> Optional[str]:
    """[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Normaliza el `user_id` que
    persistimos en `agent_messages.user_id`:

      - `None`, `""`, `"guest"` → `None` (guest legítimo).
      - `user_id == session_id` → `None` (frontend default cuando no hay
        auth — el endpoint usa `session_id` como placeholder de user_id).
      - UUID real distinto → retorna tal cual.

    NO valida que sea UUID válido (eso lo hace el FK a `auth.users` en
    DB — si es UUID malformado, el INSERT falla con `invalid uuid` y el
    retry tenacity captura el error). Tooltip-anchor: P1-CHAT-DB-USER-ID-RLS."""
    if not user_id_input:
        return None
    if user_id_input == "guest":
        return None
    if session_id and user_id_input == session_id:
        return None
    return user_id_input


def _resolve_chat_identity(
    body_user_id: Optional[str], session_id: Optional[str], verified_user_id: Optional[str]
) -> str:
    """[P0-CHAT-IDENTITY-FROM-TOKEN · 2026-09-14] La identidad del turno sale SOLO del token.

    Antes, los endpoints del chat tomaban `user_id` del BODY y lo validaban contra el
    token únicamente cuando `user_id != session_id`. Una petición SIN token con
    `session_id = user_id = <UUID de la víctima>` saltaba ese guard y también el de
    dueño de sesión (la sesión aún no existía), y `verify_coach_quota` deja pasar sin
    token. El agente recibía ese UUID como si estuviera autenticado: cargaba plan,
    alergias, Nevera y diario de la víctima en el prompt, y el override P0-AGENT-1
    fijaba el `user_id` de las TOOLS a ese mismo valor — escrituras sobre la víctima.

    Contrato:
      - Con token: el turno es del `verified_user_id`. Un `user_id` del body distinto
        (que no sea "guest" ni el session_id del propio cliente) es 401, como antes.
      - Sin token: SIEMPRE "guest", diga lo que diga el body. El invitado sigue
        teniendo su conversación (vive en `session_id`); lo que pierde es la
        posibilidad de nombrar a otra persona.

    tooltip-anchor: _resolve_chat_identity (test_p0_chat_identity_from_token.py)
    """
    if verified_user_id:
        if body_user_id and body_user_id not in ("guest", session_id, verified_user_id):
            raise HTTPException(status_code=401, detail="No autorizado.")
        return verified_user_id
    # [P1-PLAN-LOTE-161 · 2026-09-22] El invitado no puede tomar como `session_id` el id de una CUENTA. Su turno ya
    # era «guest», pero las herramientas del coach usan el `session_id` del invitado como su identidad (es donde
    # vive su diario y su Nevera de prueba): con el UUID de otra persona leían su perfil clínico y escribían en su
    # diario. Una sesión de invitado nace de un UUID aleatorio del cliente; si coincide con una cuenta, no es de
    # un invitado. `None` (no se pudo comprobar) falla abierto, como el resto de chequeos que dependen de la base.
    from db import uuid_es_de_una_cuenta
    if session_id and uuid_es_de_una_cuenta(session_id) is True:
        raise HTTPException(status_code=403, detail="Prohibido. Inicia sesión para continuar esta conversación.")
    return "guest"


def _resolve_chat_local_time(local_date, tz_offset, verified_user_id):
    """[P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD · 2026-08-23] Resuelve el "hoy" del usuario
    SERVER-SIDE cuando el cliente no lo manda.

    El contexto temporal del coach (`build_temporal_context`, `DIARIO DE HOY`, los días
    pasados, las comidas que te quedan hoy) se arma con `local_date`/`tz_offset`. Los dos
    endpoints de chat los leen del BODY y, si el cliente no colabora, aguas abajo caen a un
    huso por defecto: para un usuario en Madrid a las 00:30, el coach cree que es el día
    anterior a las 18:30.

    **Por qué server-side y no "que lo mande el cliente"**: estos endpoints tienen delante un
    `verified_user_id` y un perfil con el huso que el propio navegador del usuario ya persistió
    (`health_profile.tzOffset`). Depender de que cada cliente lo reenvíe en cada turno es
    depender de la colaboración de un caller que puede no existir — y el path no-stream no
    tiene NINGÚN caller en el frontend hoy, aunque está registrado, autenticado y tarifado.

    Contrato:
      - Si el cliente mandó AMBOS, no se toca nada (ni un round-trip a la DB).
      - Si mandó sólo `tz_offset`, la fecha se deriva de ESE offset — el del cliente gana al
        del perfil, que puede ser viejo (el usuario viaja).
      - Invitado sin identidad verificada, o lectura de perfil que falla: se devuelve lo que
        llegó, así que la conducta es EXACTAMENTE la previa. Este helper no puede empeorar
        ningún caso; sólo puede rellenar huecos.
      - `0` es un offset legítimo (UTC, y Canarias en invierno): la resolución es por
        `is not None`, JAMÁS por truthiness.

    tooltip-anchor: _resolve_chat_local_time (test_p3_chat_nostream_contexto_temporal_rd.py)
    """
    if local_date is not None and tz_offset is not None:
        return local_date, tz_offset

    resolved_tz = tz_offset
    if resolved_tz is None and verified_user_id:
        try:
            from db import user_tz_offset_min
            resolved_tz = user_tz_offset_min(verified_user_id)
        except Exception as e:
            logger.warning(
                f"[P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD] no se pudo resolver el huso de "
                f"{_hash_user_id_for_log(verified_user_id)}: {e}. Se deja el contexto temporal como llegó."
            )
            return local_date, tz_offset

    if resolved_tz is None:
        # Ni cliente ni perfil: no hay nada que aportar. Conducta previa, intacta.
        return local_date, tz_offset

    resolved_date = local_date
    if resolved_date is None:
        from datetime import datetime, timedelta, timezone
        try:
            resolved_date = (
                datetime.now(timezone.utc) - timedelta(minutes=int(resolved_tz))
            ).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            return local_date, tz_offset

    return resolved_date, resolved_tz

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)


def _hydrate_chat_attachment_urls(messages: list) -> list:
    """Renueva URLs firmadas al leer historial sin exponer el bytea."""
    hydrated = []
    for message in messages or []:
        copy = dict(message)
        raw = copy.get("attachments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        attachments = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            attachment_id = item.get("attachment_id") or item.get("id")
            if not attachment_id:
                continue
            attachments.append({
                **item,
                "attachment_id": str(attachment_id),
                "url": build_chat_attachment_url(str(attachment_id)),
            })
        copy["attachments"] = attachments
        hydrated.append(copy)
    return hydrated


@router.get("/attachments/{attachment_id}")
def api_get_chat_attachment(
    attachment_id: str,
    expires: int = 0,
    sig: str = "",
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    assert_valid_uuid(attachment_id)
    attachment = get_chat_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    signed = verify_chat_attachment_signature(attachment_id, expires, sig)
    owned = bool(verified_user_id and verified_user_id == attachment.get("user_id"))
    if not signed and not owned:
        raise HTTPException(status_code=403, detail="Prohibido.")
    return Response(
        content=attachment.get("content") or b"",
        media_type=attachment.get("content_type") or "application/octet-stream",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
        },
    )

@router.get("/sessions/{user_id}")
def api_get_chat_sessions(
    user_id: str,
    session_ids: Optional[str] = None,
    offset: int = 0,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    # [P2-CHAT-SESSIONS-PAGING · 2026-09-03] `offset` pagina Recientes de 60 en 60 y
    # `has_more` le dice al cliente si mostrar «Ver más». El invitado no pagina: lista
    # sus ids (`session_ids`) y ya. Tooltip-anchor: P2-CHAT-SESSIONS-PAGING.
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(user_id, allow_guest=True)
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        _offset = max(0, min(int(offset or 0), 10_000))
        sessions: list = get_user_chat_sessions(user_id, limit=CHAT_SESSIONS_PAGE_SIZE, offset=_offset) or []
        has_more = False
        if user_id and user_id != "guest":
            has_more = count_user_chat_sessions(user_id) > _offset + CHAT_SESSIONS_PAGE_SIZE
        
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
            
        return {"sessions": sessions, "has_more": has_more}
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
        filtered_messages = _hydrate_chat_attachment_urls(
            [m for m in messages if not m.get("content", "").startswith("[SYSTEM_TITLE]")]
        )
        # [P1-CHAT-ORPHAN-TURN-TRUTH · 2026-09-03] `turn_active`: el cliente que recarga con
        # un mensaje sin respuesta solo debe seguir «recuperando» si el turno sigue vivo.
        return {"messages": filtered_messages, "turn_active": is_turn_active(session_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/history GET: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/session/{session_id}")
def api_delete_chat_session(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """[P0-CHAT-DELETE-IDOR · 2026-05-26] Elimina sesión propia del usuario.

    Pre-fix: el endpoint solo validaba `if not verified_user_id` (¿está
    logueado?), pero NO `session.user_id == verified_user_id`. Cualquier
    autenticado podía DELETE chats ajenos pasando session_id enumerado.

    Post-fix: el helper `delete_chat_session(session_id, user_id)` ejecuta
    pre-check de ownership server-side (patrón simétrico al GET /history).
    Mapeo de error_msg a HTTP status:
      - "not_found" → 404
      - "forbidden" → 403
      - otros → 500
    """
    from db import delete_chat_session
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(session_id)
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="Token requerido para eliminar chats.")

        success, error_msg = delete_chat_session(session_id, verified_user_id)
        if success:
            logger.info(f"🗑️ Chat {session_id} eliminado por usuario {verified_user_id}")
            return {"success": True, "message": "Chat eliminado correctamente."}
        if error_msg == "not_found":
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")
        if error_msg == "forbidden":
            raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")
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
    # [P0-CHAT-IDENTITY-FROM-TOKEN · 2026-09-14] Identidad solo del token (ver helper).
    user_id = _resolve_chat_identity(data.get("user_id"), session_id, verified_user_id)

    # [P2-CHAT-WRITE-IDOR · 2026-05-28] El guard de arriba se SALTA cuando
    # user_id == session_id (atacante envía session_id=<sesión de la víctima>,
    # user_id=<mismo UUID>): la condición `user_id != session_id` es False y no
    # se valida ownership; luego save_message(user_id=None) resuelve el dueño
    # real vía get_session_owner e inyecta el mensaje bajo la víctima. Espejo de
    # P0-CHAT-DELETE-IDOR: si la sesión YA tiene dueño, exigir que coincida con
    # el token verificado. Sesión sin dueño (guest) o inexistente → permitido.
    from db_chat import get_session_owner
    _sess_owner = get_session_owner(session_id) if session_id else None
    if _sess_owner and _sess_owner != verified_user_id:
        raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

    # [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap longitud antes de INSERT en
    # `agent_messages`. Storage abuse: text column sin cap nativo → un
    # autenticado podría inflar la tabla sin pasar por LLM. Ver helper.
    _enforce_chat_prompt_cap(content, field_name="content")

    if session_id and role and content:
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        # [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Pasar user_id explícito al
        # save_message (ya en scope post-IDOR check + cap). Helper normaliza
        # guest/session_id/UUID. Evita el lookup defensivo via
        # get_session_owner que save_message hace cuando user_id es None.
        save_message(
            session_id, role, content,
            user_id=_resolve_user_id_for_db(user_id, session_id),
        )
        return {"success": True}
    return {"success": False, "error": "Faltan parámetros"}

from fastapi.responses import Response
import asyncio
import os



# [P1-CHAT-STREAM-RL · 2026-05-19] Rate limiter de los endpoints
# `/api/chat/stream` (SSE LLM principal) y `/api/chat` (non-stream, mismo
# path al LLM). Pre-fix: solo el paywall `verify_api_quota` filtraba
# tráfico (gratis=15, basic=50, plus=200, ultra/admin=999999 por MES).
# A escala mensual eso no protege contra bursts: un user `plus` (200/mes)
# puede gastar todo su cupo en 30 segundos enviando 200 prompts seguidos,
# (a) saturando el upstream Gemini, (b) elevando p95 para usuarios
# legítimos, (c) abriendo el `LLMCircuitBreaker` y gatillando 503s.
#
# Per-minute limit es el complemento correcto del paywall mensual: el
# paywall acota costo total, el rate limiter acota burst. Patrón canónico
# del repo (P1-6, P1-CHAT-TTS-1, P1-CHAT-TTS-1-AUTH).
#
# Default 30 calls/60s:
#   - Conversación humana típica: ~1-2 prompts/min. 30/min cubre con margen
#     amplio para un user que itera fast (correcciones rápidas, "más",
#     "no, otra cosa").
#   - Voice mode (call_mode): el audio lo dice el propio dispositivo
#     (P1-PLAN-LOTE-682); /stream solo recibe lo que el usuario dijo,
#     una vez por turno, igual que un mensaje escrito.
#   - Bots/scrapers: 30/min limita hard el daño en burst — incluso si un
#     atacante autenticado intenta spam, 30 prompts × 30 segundos = 15
#     respuestas LLM, no 200.
#
# Knob `MEALFIT_CHAT_STREAM_LIMITER_PER_MIN` (auto-registrado vía `_env_int`,
# clamp [1, 600]). Floor 1: nunca dejes el limiter en 0 (chat queda roto);
# techo 600 (= 10/seg) es el peor caso razonable antes de que el limiter
# se vuelva no-protector. Tooltip-anchor: P1-CHAT-STREAM-RL.
_CHAT_STREAM_LIMITER_PER_MIN_DEFAULT = 30
_CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MIN = 1
_CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MAX = 600


def _chat_stream_limiter_per_min() -> int:
    """[P1-CHAT-STREAM-RL · 2026-05-19] Lee el knob con clamp defensivo.
    Llamado UNA vez al module-load (en la construcción del singleton).
    Tooltip-anchor: P1-CHAT-STREAM-RL."""
    raw = _env_int(
        "MEALFIT_CHAT_STREAM_LIMITER_PER_MIN",
        _CHAT_STREAM_LIMITER_PER_MIN_DEFAULT,
    )
    if raw < _CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MIN:
        return _CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MIN
    if raw > _CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MAX:
        return _CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MAX
    return raw


# [P1-COACH-QUOTA-METER · 2026-09-02] Lectura de la cuota del coach para el medidor del chat.
# Exenta del paywall (read-only, cero LLM): al llegar al tope el usuario necesita VER cuánto
# le queda y cuándo se renueva, no otro 402. Anti-spam por RateLimiter, no por cuota.
_COACH_QUOTA_LIMITER = RateLimiter(max_calls=30, period_seconds=60)
# [P1-PLAN-LOTE-760] «Detener» del chat: barato, pero con tope como todo endpoint sin cuota.
_CHAT_STOP_LIMITER = RateLimiter(max_calls=30, period_seconds=60)


@router.post("/stop")
def api_chat_stop(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id),
                  _rl: None = Depends(_CHAT_STOP_LIMITER)):
    """[P1-PLAN-LOTE-760] «Detener»: corta el turno en curso de ese chat. Desde que el turno sigue aunque el cliente se
    vaya (`turno_desacoplado`), cortar la conexión ya no para la generación; esto sí. Mismo control de dueño que
    `/stream`: un chat con dueño solo lo detiene su dueño. Cero LLM."""
    session_id = str((data or {}).get("session_id") or "").strip()
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="Falta el chat.")
    from db_chat import get_session_owner
    _dueno = get_session_owner(session_id)
    if _dueno and _dueno != verified_user_id:
        raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")
    from turno_desacoplado import detener
    return {"stopped": detener(session_id)}


@router.get("/quota")
def api_chat_quota(verified_user_id: str = Depends(get_verified_user_id), _rl: None = Depends(_COACH_QUOTA_LIMITER)):
    """[P1-COACH-QUOTA-METER] `{used, limit, remaining, tier, period, resets_at}` del mes en curso."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return coach_quota_snapshot(verified_user_id)


_CHAT_STREAM_LIMITER = RateLimiter(
    max_calls=_chat_stream_limiter_per_min(),
    period_seconds=60,
)

# [P1-PLAN-LOTE-685 · 2026-09-28] La voz del coach en el modo voz: Gemini 3.8 Flash-Lite TTS (`coach_voz.py`). El dueño:
# «la voz no es nada realista» (era el lector de accesibilidad del teléfono). Exenta del paywall como todo el modo voz:
# el gasto va a `llm_usage_events` (node="coach_voice_tts"), NUNCA a `api_usage` —cada frase quemaría un crédito del
# plan—; el tope es de DINERO (presupuesto diario, knob) y el anti-spam, el RateLimiter (una respuesta son 2-4 frases).
# Soft-fail: 204 sin cuerpo y el teléfono pone su propia voz —apagada por knob, sin presupuesto o Google caído—, nunca
# un error en rojo. `asyncio.to_thread`: la síntesis tarda ~2 s y el backend corre con UN worker de uvicorn.
_VOZ_LIMITER = RateLimiter(max_calls=40, period_seconds=60)
_VOZ_EN_VUELO = None


def _semaforo_de_voz():
    global _VOZ_EN_VUELO
    if _VOZ_EN_VUELO is None:
        import asyncio
        _VOZ_EN_VUELO = asyncio.Semaphore(4)   # no más de 4 síntesis a la vez: el resto espera su turno
    return _VOZ_EN_VUELO


@router.post("/voz")
async def api_chat_voz(background_tasks: BackgroundTasks, data: dict = Body(...),
                       verified_user_id: Optional[str] = Depends(get_verified_user_id),
                       _rl: None = Depends(_VOZ_LIMITER)):
    """[P1-PLAN-LOTE-685] Una frase del coach → WAV con su voz. `{texto, locale}`; 204 = «usa la voz del teléfono»."""
    import asyncio
    from coach_voz import hay_presupuesto, registrar_uso, sintetizar, voz_en_la_nube_activa
    texto = str((data or {}).get("texto") or "").strip()
    locale = str((data or {}).get("locale") or "es-DO").strip()[:10]
    if not texto:
        raise HTTPException(status_code=400, detail="Falta el texto.")
    if not voz_en_la_nube_activa():
        return Response(status_code=204, headers={"X-Voz-Motivo": "apagada"})
    if not await asyncio.to_thread(hay_presupuesto):
        logger.warning("⚠️ [P1-PLAN-LOTE-685] voz del coach sin presupuesto hoy: el teléfono pone la suya.")
        return Response(status_code=204, headers={"X-Voz-Motivo": "presupuesto"})
    try:
        async with _semaforo_de_voz():
            voz = await asyncio.to_thread(sintetizar, texto, locale)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-685] la voz del coach falló, el teléfono pone la suya: "
                       f"{type(e).__name__}: {str(e)[:160]}")
        return Response(status_code=204, headers={"X-Voz-Motivo": "error"})
    if voz is None:
        return Response(status_code=204, headers={"X-Voz-Motivo": "vacio"})
    background_tasks.add_task(registrar_uso, voz, verified_user_id)
    return Response(content=voz.wav, media_type="audio/wav",
                    headers={"Cache-Control": "no-store", "X-Voz-Ms": str(voz.ms)})


# [P1-PLAN-LOTE-682 · 2026-09-28] Aquí vivía `POST /tts`: el proxy a ElevenLabs del viejo Modo Llamada.
# Sin llamadores desde mayo (P1-DEADCODE-TTS) y reemplazado por la voz del propio dispositivo
# (`speechSynthesis`, frontend `utils/vozDelCoach.js`): ya no se manda texto a ElevenLabs, y un endpoint
# muerto que aún podía gastar su saldo (y créditos de plan: `elevenlabs_tts` caía en la lista negativa) sale.

@router.post("/feedback")
async def api_chat_feedback(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    session_id = data.get("session_id")
    content = data.get("content")
    feedback = data.get("feedback")
    
    if not session_id or not content:
        raise HTTPException(status_code=400, detail="Missing session_id or content")

    # [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] Validación IDOR de ownership (espejo
    # de `/history/{session_id}` y `DELETE /session/{session_id}`). Pre-fix el
    # endpoint requería JWT pero NO verificaba que el `session_id` del body
    # perteneciera al caller → un usuario autenticado que enumerara/adivinara el
    # session_id de otro podía escribir feedback en su sesión ajena (y forzar
    # get_or_create_session sobre ella). Si la sesión NO existe aún (owner None),
    # el check pasa y get_or_create la crea a nombre del caller.
    # Tooltip-anchor: P2-CHAT-FEEDBACK-OWNERSHIP.
    assert_valid_uuid(session_id)
    session_owner = await asyncio.to_thread(get_session_owner, session_id)
    if session_owner and session_owner != "guest":
        if not verified_user_id or verified_user_id != session_owner:
            logger.warning(
                f"🚫 [FEEDBACK AUTH FAILED] REJECTED. owner={session_owner} != "
                f"verified={verified_user_id}"
            )
            raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

    # Asegurarnos de que exista la sesión en la base de datos antes de guardar feedback
    from db import get_or_create_session
    await asyncio.to_thread(get_or_create_session, session_id, user_id=verified_user_id)

    success = await asyncio.to_thread(save_message_feedback, session_id, content, feedback)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="Error saving feedback")


@router.post("/stream", dependencies=[Depends(_CHAT_STREAM_LIMITER)])
def api_chat_stream(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_coach_quota)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        # [P0-CHAT-IDENTITY-FROM-TOKEN · 2026-09-14] Identidad solo del token (ver helper).
        user_id = _resolve_chat_identity(data.get("user_id"), session_id, verified_user_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        local_date = data.get("local_date", None)
        tz_offset = data.get("tz_offset", None)
        # [P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD · 2026-08-23] Relleno server-side de los
        # huecos. Este endpoint SÍ tiene caller y siempre los manda, así que aquí es una
        # red de seguridad; en `POST /api/chat` (abajo) es la única defensa que hay.
        local_date, tz_offset = _resolve_chat_local_time(local_date, tz_offset, verified_user_id)
        is_call_mode = data.get("is_call_mode", False)
        # [P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL · 2026-08-23] El contexto de la foto viene
        # ESTRUCTURADO; el servidor compone el bloque y lo pone en el system prompt.
        vision = data.get("vision") if isinstance(data.get("vision"), dict) else None
        raw_attachments = data.get("attachments") if isinstance(data.get("attachments"), list) else []
        attachment_ids = []
        for item in raw_attachments[:4]:
            attachment_id = item.get("attachment_id") if isinstance(item, dict) else None
            if attachment_id:
                assert_valid_uuid(str(attachment_id))
                attachment_ids.append(str(attachment_id))
        client_message_id = data.get("client_message_id")
        if client_message_id:
            assert_valid_uuid(str(client_message_id))
        regenerate_message_id = data.get("regenerate_message_id")
        if regenerate_message_id:
            assert_valid_uuid(str(regenerate_message_id))
        regenerate_response_content = data.get("regenerate_response_content")
        
        # [P2-CHAT-WRITE-IDOR · 2026-05-28] Cierra el bypass user_id==session_id
        # (ver /message): si la sesión ya tiene dueño, exigir match con el token.
        from db_chat import get_session_owner
        _sess_owner = get_session_owner(session_id) if session_id else None
        if _sess_owner and _sess_owner != verified_user_id:
            raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

        # [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap longitud ANTES de
        # `save_message` y ANTES de invocar el LLM. Vector cerrado: DoS
        # económico vía prompts gigantes (quema tokens del owner +
        # cuelga endpoint hasta timeout total-graph 60s). Ver helper.
        _enforce_chat_prompt_cap(prompt, field_name="prompt")

        # [P1-CHAT-LOG-CTX · 2026-05-19] LoggerAdapter con session_id +
        # user_id_hash. Reemplaza logs crudos del módulo en este endpoint
        # para que un incidente reportado por user sea grepable end-to-end.
        clog = _chat_logger(session_id, user_id)
        clog.info(f"🔍 [DEBUG API CHAT STREAM] session_id={session_id}, user_id={user_id}")

        # Operaciones síncronas directas (ya estamos en un threadpool worker)
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        # [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Pasar user_id explícito.
        _db_user_id = _resolve_user_id_for_db(user_id, session_id)
        is_regeneration = bool(regenerate_message_id or regenerate_response_content)
        _regenerate_target_id = None
        if is_regeneration:
            _regenerate_target_id = get_model_response_id_for_regeneration(
                session_id,
                message_id=str(regenerate_message_id) if regenerate_message_id else None,
                content=str(regenerate_response_content) if regenerate_response_content else None,
            )
            if not _regenerate_target_id:
                raise HTTPException(status_code=409, detail="La respuesta que intentas regenerar ya cambió. Recarga el chat.")
        if attachment_ids and not _db_user_id:
            raise HTTPException(status_code=401, detail="Se requiere sesión para adjuntar imágenes.")
        if is_regeneration:
            # El turno de usuario original ya existe. Persistirlo otra vez
            # produce pares user/model duplicados al recargar el historial.
            pass
        elif _db_user_id and client_message_id:
            vision_items = (
                vision.get("items", [])
                if isinstance(vision, dict) and vision.get("kind") == "multi"
                else ([vision] if isinstance(vision, dict) else [])
            )
            save_message_with_attachments(
                session_id, prompt, _db_user_id, attachment_ids,
                client_message_id=str(client_message_id),
                vision_items=vision_items,
            )
        else:
            save_message(session_id, "user", prompt, user_id=_db_user_id)
        
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
        # ANTES: `log_api_usage(user_id, "llm_chat")` vivía DENTRO de
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

        # [P1-PHOTO-ONLY-TURN · 2026-09-06] Una foto sin texto dejaba el turno del usuario VACÍO.
        #
        # Caso vivo (06-sep 16:48): el dueño abre un chat nuevo y sube la foto de un almuerzo
        # dominicano, sin escribir nada. El escáner acertó de lleno — «arroz blanco, espaguetis
        # guisados con salsa de tomate y aceitunas, carne de res guisada y plátano maduro frito…
        # (1065 kcal, 51 g de proteína)», guardado en `attachments[0].description`— y el bloque
        # `build_vision_context` llegó al system prompt como debe. Aun así el coach contestó con
        # el menú del día y terminó preguntando «¿Almorzaste algo distinto al ceviche?»: no miró
        # la foto.
        #
        # La causa es el turno vacío. Un mensaje de usuario sin una sola palabra no es una
        # pregunta; el modelo llena el vacío con lo que mejor encaja al abrir una sesión, que es
        # saludar y recitar el plan. La instrucción del bloque de foto («actúa proactivamente y
        # resume lo detectado») está ahí y perdió contra ese vacío.
        #
        # El marcador es UN EMOJI a propósito. `P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL` sacó del
        # turno del usuario los cuatro bloques en español justamente porque eran la señal más
        # fuerte hacia el español; meter ahora «El usuario subió una foto» sería deshacerlo. Un
        # emoji no tiene idioma.
        #
        # Y NO se persiste: lo que se guardó en `agent_messages` sigue siendo el texto del usuario
        # (vacío), así que la burbuja del chat no cambia. Es el mismo reparto que ya usa el resto
        # del sistema — lo que el usuario VE y lo que el modelo LEE no tienen por qué ser lo mismo.
        _prompt_para_el_modelo = prompt
        if not str(prompt or "").strip() and isinstance(vision, dict) and vision.get("kind"):
            _prompt_para_el_modelo = "\U0001f4f7"

        def event_generator():
            nonlocal _billed, _chunk_observed
            # [P2-CHAT-SINGLE-ERROR-EVENT · 2026-09-14] Si el agente ya emitió su evento
            # `error`, el router no emite un segundo.
            _error_seen = False
            try:
                for chunk in chat_with_agent_stream(
                    session_id=session_id,
                    prompt=_prompt_para_el_modelo,
                    current_plan=current_plan,
                    user_id=user_id,
                    form_data=form_data,
                    local_date=local_date,
                    tz_offset=tz_offset,
                    is_call_mode=is_call_mode,
                    plan_tier=plan_tier,
                    vision=vision,
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
                            elif _chunk_type == "error":
                                _error_seen = True

                            if _chunk_type == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    # [P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20]
                                    # Strip tags `[UI_ACTION: <NAME>]` ANTES de
                                    # persistir. El frontend ya strip + dispatch
                                    # en runtime (AgentPage.jsx), pero el refetch
                                    # de `/api/chat/history/<session_id>` traía
                                    # el tag RAW de DB y lo re-renderizaba —
                                    # síntoma reportado: "desapareció y volvió a
                                    # aparecer". Strip server-side cierra el ciclo.
                                    response_text = strip_ui_action_tags_for_persist(response_text)
                                    # [P1-CHAT-DB-USER-ID-RLS · 2026-05-19]
                                    # `_db_user_id` resuelto arriba en
                                    # closure scope. Persiste el ownership
                                    # de la respuesta del modelo al user
                                    # que envió el prompt.
                                    # [P2-CHAT-DONE-PERSIST-LOUD · 2026-09-14] Un fallo al
                                    # guardar caía en el `except` de «Error parseando chunk
                                    # de fin» y se saltaba también `bg_tasks` (hechos y
                                    # resumen). Ahora se registra como lo que es y el
                                    # turno sigue.
                                    try:
                                        if _regenerate_target_id:
                                            replaced = replace_model_response_for_regeneration(
                                                session_id,
                                                _regenerate_target_id,
                                                response_text,
                                            )
                                            if not replaced:
                                                raise RuntimeError("No se pudo sustituir la respuesta regenerada")
                                        else:
                                            save_message(
                                                session_id, "model", response_text,
                                                user_id=_db_user_id,
                                            )
                                    except Exception as _persist_err:
                                        clog.exception(
                                            f"[P2-CHAT-DONE-PERSIST-LOUD] la respuesta del modelo "
                                            f"NO se guardó en el historial: {type(_persist_err).__name__}"
                                        )
                                    # [P1-PLAN-LOTE-692] «Bioboros te respondió»: la push sale siempre y el
                                    # teléfono la calla si el usuario la está viendo (`solo_si_no_mira`).
                                    from aviso_respuesta_chat import avisar_respuesta
                                    avisar_respuesta(verified_user_id, response_text)
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

                                        # [P1-TIER-PARITY · 2026-07-12] Memoria a largo
                                        # plazo para TODOS los tiers (los planes solo
                                        # difieren en créditos). Guests fuera: sin
                                        # cuenta no hay identidad estable que recordar.
                                        is_plus = bool(user_id and user_id != "guest")

                                        # [LONG-TERM-MEMORY-TOGGLE · 2026-05-13]
                                        # Además del gate de tier, respetar el flag user-controlled
                                        # `long_term_memory_enabled`.
                                        # [P1-PLAN-LOTE-717 · 2026-09-28] Fail-CLOSED: un perfil ilegible
                                        # cuenta como PAUSADA. Antes contaba como «activada» y aprendía
                                        # justo cuando no se podía confirmar que el usuario lo quería.
                                        ltm_enabled = bool(is_plus) and memoria_activa(user_id, donde="chat/stream")

                                        if is_plus and ltm_enabled:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)
                                        elif is_plus:
                                            logger.info(f"[LONG-TERM-MEMORY-TOGGLE] Captura pausada (user={user_id}).")

                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        logger.error(f"Error en bg tasks: {inner_e}")

                                # [P1-BG-THREAD-TIMEOUT · 2026-05-15] submit
                                # al pool compartido con timeout + alert.
                                submit_bg_task(bg_tasks, task_name="chat_sse_bg_tasks")
                        except Exception as e_json:
                            logger.error(f"Error parseando chunk de fin: {e_json}")

            except (GeneratorExit, asyncio.CancelledError) as _cancel_exc:
                # [P2-AUDIT-NEW-2 · 2026-05-12] Cliente cerró el SSE
                # (AbortController, tab close, network drop).
                # [P1-CHAT-CANCEL-ASYNC · 2026-05-19] Extendido a
                # `asyncio.CancelledError` — los generators sync embebidos
                # en `StreamingResponse` se ejecutan en threadpool, pero
                # Starlette puede cancelar el wrapper async wrapper cuando
                # el cliente desconecta. `CancelledError` hereda de
                # `BaseException` (NO de `Exception`) en Python 3.8+ así
                # que el `except Exception` debajo NO la atrapaba — el
                # finally idempotente de billing corría, pero el log se
                # perdía como "Error mid-stream" con stack confuso. Ahora
                # ambas señales de aborto se loguean como `info`, no
                # `exception`. NO re-emite chunks (conexión ya muerta);
                # finally SÍ cobra si chunk_observed.
                _cancel_kind = type(_cancel_exc).__name__
                clog.info(
                    f"[P2-AUDIT-NEW-2] SSE abortado por cliente "
                    f"kind={_cancel_kind} chunk_observed={_chunk_observed}"
                )
                raise
            except Exception as e:
                # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
                clog.exception(f"[CHAT STREAM] Error mid-stream: {e}")
                # `chunk_observed` puede ser True (excepción tras emitir chunks)
                # o False (excepción pre-LLM). El finally factura solo si True.
                # [P2-CHAT-SINGLE-ERROR-EVENT · 2026-09-14] Sin `str(e)` (detalle interno
                # visible en Network) y sin segundo evento si el agente ya emitió el suyo.
                if not _error_seen:
                    yield f"data: {json.dumps({'type': 'error', 'code': 'internal', 'message': 'El asistente tuvo un problema. Intenta de nuevo.'})}\n\n"
            finally:
                # [P2-AUDIT-NEW-2] Billing idempotente. Cubre TODOS los exits:
                # done OK, GeneratorExit (abort), exception mid-stream.
                # Solo factura si:
                #   (a) Aún no se cobró (`_billed` flag, defensivo).
                #   (b) El LLM emitió al menos un chunk de texto.
                #   (c) Usuario autenticado (no guest, no session-only).
                # [P1-CHAT-BILL-VERIFIED-UID · 2026-05-30] Facturar contra la
                # identidad VERIFICADA por el token (`verified_user_id`), NO el
                # `user_id` del body. Pre-fix el gate `user_id != session_id`
                # permitía a un autenticado evadir el incremento del paywall
                # mensual enviando user_id==session_id==su-propio-UUID (la rama
                # "guest gratis" asume user_id==session_id solo para invitados,
                # pero un request crafteado puede igualarlos). El LLM corría y
                # `log_api_usage` nunca incrementaba → `verify_api_quota` (que
                # cuenta por verified_user_id) jamás alcanzaba el cap → Gemini
                # ilimitado gratis para un tier `gratis`. Facturar por
                # verified_user_id cierra el bypass: invitados (sin token →
                # verified_user_id None) siguen gratis + acotados por
                # `_CHAT_STREAM_LIMITER`; autenticados se facturan en la
                # identidad que el proveedor de Auth verificó (no spoofeable vía body).
                # Tooltip-anchor: P1-CHAT-BILL-VERIFIED-UID.
                if not _billed and _chunk_observed and verified_user_id:
                    try:
                        log_api_usage(verified_user_id, "llm_chat")
                        _billed = True
                    except Exception as _bill_err:
                        logger.warning(
                            f"[P2-AUDIT-NEW-2] log_api_usage falló "
                            f"(best-effort): {_bill_err}"
                        )

        # [P1-PLAN-LOTE-760] El turno corre en SU hilo: si el cliente se va (salió de la app), la respuesta se termina,
        # se guarda y avisa por push igual. «Detener» ya no es cortar la conexión: es `POST /api/chat/stop`.
        from turno_desacoplado import activo as _turno_desacoplado, desacoplar
        _flujo = desacoplar(event_generator(), session_id) if _turno_desacoplado() else event_generator()
        return StreamingResponse(_flujo, media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"[CHAT STREAM] Error en api_chat_stream: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))



@router.post("", dependencies=[Depends(_CHAT_STREAM_LIMITER)])
def api_chat(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_coach_quota)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        # [P0-CHAT-IDENTITY-FROM-TOKEN · 2026-09-14] Identidad solo del token (ver helper).
        user_id = _resolve_chat_identity(data.get("user_id"), session_id, verified_user_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        local_date = data.get("local_date", None)
        tz_offset = data.get("tz_offset", None)
        # [P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD + P3-CHAT-NONSTREAM-RD-DATE · 2026-08-23]
        # TODO el contexto temporal de este endpoint (hora, día de la semana, `DIARIO DE HOY`,
        # días pasados, comidas que quedan hoy) se armaba en hora dominicana cuando el cliente
        # no mandaba estos dos campos. Se resuelve desde el `verified_user_id`, que ya está
        # delante: es la opción que no depende de que el caller colabore.
        local_date, tz_offset = _resolve_chat_local_time(local_date, tz_offset, verified_user_id)

        # [P2-CHAT-WRITE-IDOR · 2026-05-30] Tercer hermano del guard de escritura
        # IDOR. El check inline de arriba se SALTA cuando user_id == session_id
        # (atacante manda session_id=<sesión de la víctima>, user_id=<mismo UUID>):
        # `user_id != session_id` es False → no se valida ownership; luego
        # `_resolve_user_id_for_db` → None → `save_message(user_id=None)` resuelve
        # el dueño real vía `get_session_owner` e INYECTA mensajes en
        # `agent_messages` + corrompe `nudge_outcomes`/`abandoned_meal_reasons` de
        # la víctima, sin su token. `/message` (P2-CHAT-WRITE-IDOR) y `/stream`
        # ya tenían este guard; este endpoint `POST /api/chat` lo había omitido.
        # Si la sesión ya tiene dueño, exigir match con el token verificado.
        from db_chat import get_session_owner
        _sess_owner = get_session_owner(session_id) if session_id else None
        if _sess_owner and _sess_owner != verified_user_id:
            raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

        # [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap longitud antes de invocar
        # el LLM (`chat_with_agent`). Mismo vector que `/stream` pero sin
        # streaming — un blob gigante cuelga el endpoint hasta el timeout
        # total-graph y quema tokens del owner. Ver helper.
        _enforce_chat_prompt_cap(prompt, field_name="prompt")

        # [P1-CHAT-LOG-CTX · 2026-05-19] Logger correlacionable.
        clog = _chat_logger(session_id, user_id)
        clog.info(f"🔍 [DEBUG API CHAT] session_id={session_id}, user_id={user_id}")

        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        # [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Pasar user_id explícito —
        # ya resuelto post-IDOR check. Mismo patrón que /stream.
        _db_user_id = _resolve_user_id_for_db(user_id, session_id)
        save_message(session_id, "user", prompt, user_id=_db_user_id)

        # Handle form_data: merge frontend data with DB health_profile (DRY — shared in services.py)
        form_data = merge_form_data_with_profile(
            user_id if user_id != "guest" and user_id != session_id else "",
            form_data
        )

        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)

        response_text, updated_fields, new_plan = chat_with_agent(
            session_id,
            prompt,
            current_plan=current_plan,
            user_id=user_id,
            form_data=form_data,
            local_date=local_date,
            tz_offset=tz_offset,
        )

        # [P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20] Mismo strip que el
        # endpoint /stream — cierra el ciclo del tag visible al refetch.
        response_text = strip_ui_action_tags_for_persist(response_text)
        save_message(session_id, "model", response_text, user_id=_db_user_id)
        
        # 🧠 Background: Resumir y podar mensajes si el historial creció demasiado
        background_tasks.add_task(summarize_and_prune, session_id)
        
        # [P1-CHAT-BILL-VERIFIED-UID · 2026-05-30] Facturar por la identidad
        # verificada por el token (ver el finally de /stream). Cierra el bypass
        # del paywall vía user_id==session_id en este endpoint non-stream.
        if verified_user_id:
            log_api_usage(verified_user_id, "llm_chat")

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
        # [P1-TIER-PARITY · 2026-07-12] La memoria a largo plazo es para TODOS
        # los tiers (decisión del owner: los planes solo difieren en créditos).
        # Pre-fix `is_plus` excluía a gratis — un usuario gratis con horas de
        # chat quedaba sin user_facts (y sin Dreaming, que come de ahí). Guests
        # (sin cuenta) siguen fuera: no hay user_id estable que recordar.
        is_plus = bool(user_id and user_id != "guest")
        # [P1-PLAN-LOTE-717 · 2026-09-28] Fail-CLOSED: `get_user_profile` devuelve None tanto si no hay fila como si
        # la base falla, y None contaba como «activada». Ahora un perfil ilegible cuenta como PAUSADA.
        ltm_enabled = is_plus and memoria_activa(user_id, donde="chat")

        if is_plus and ltm_enabled:
            # 🧠 Background: Extraer hechos y vectorizarlos
            background_tasks.add_task(async_extract_and_save_facts, user_id, prompt, recent_history_str)
        elif is_plus and not ltm_enabled:
            logger.info(f"[LONG-TERM-MEMORY-TOGGLE] Captura pausada por user toggle (user={user_id}).")
        else:
            logger.info("INFO: Memoria a Largo Plazo omitida (guest sin cuenta).")
        
        # 🧠 Background: Generar un título si es el primer mensaje
        background_tasks.add_task(generate_chat_title_background, user_id, session_id, prompt)
        
        result = {"response": response_text, "updated_fields": updated_fields}
        if new_plan:
            result["new_plan"] = new_plan
        return result
    except HTTPException:
        raise
    except LLMRateLimitedError as e:
        # [P1-CHAT-LLM-429 · 2026-05-20] Gemini ResourceExhausted detectado en
        # call_model. Distinto de CB abierto: el provider está vivo pero
        # throttleando este API key (saturación temporal). HTTP 429 con
        # Retry-After permite al cliente reintentar con backoff sin contaminar
        # el conteo del CB. Frontend ya muestra banner contextual; el
        # navegador respeta Retry-After si está set.
        logger.warning(f"[CHAT][P1-CHAT-LLM-429] Gemini rate-limit: {e}")
        raise HTTPException(
            status_code=429,
            detail="El asistente está procesando muchas peticiones. Intenta de nuevo en unos segundos.",
            headers={"Retry-After": "5"},
        )
    except LLMCircuitBreakerOpen as e:
        # [P1-CHAT-CB · 2026-05-19] Breaker per-modelo abierto: el provider
        # acumuló N fallos consecutivos (default 3) dentro de la ventana
        # MEALFIT_CB_RESET_TIMEOUT_S. Fail-fast con 503 SERVICE UNAVAILABLE
        # — semánticamente "intenta de nuevo más tarde". El frontend muestra
        # el banner sin reintentar automáticamente (evita amplificar la
        # condición). El breaker auto-resetea tras la ventana; el siguiente
        # request que pasa por can_proceed() probará el provider.
        logger.warning(f"[CHAT][P1-CHAT-CB] Circuit breaker abierto: {e}")
        raise HTTPException(
            status_code=503,
            detail="El asistente está temporalmente saturado. Intenta de nuevo en unos segundos.",
        )
    except TimeoutError as e:
        # [P0-CHAT-LLM-TIMEOUT · 2026-05-19] Total-graph timeout (default 60s) o
        # LLM per-invoke timeout (default 15s) excedido. 504 GATEWAY TIMEOUT
        # comunica al frontend que el LLM upstream no respondió a tiempo;
        # AgentPage muestra el banner de error sin re-intentar automáticamente
        # (evita amplificar el incidente). Sentry capture vía logger.exception.
        logger.exception(f"[CHAT][P0-CHAT-LLM-TIMEOUT] Gemini timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="El asistente tardó demasiado en responder. Intenta de nuevo en un momento.",
        )
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"[CHAT] Error en api_chat: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

