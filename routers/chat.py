from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from error_utils import safe_error_detail
from typing import Optional
import hashlib
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
from agent import generate_chat_title_background, chat_with_agent, chat_with_agent_stream, LLMCircuitBreakerOpen, LLMRateLimitedError, strip_ui_action_tags_for_persist
from services import merge_form_data_with_profile
from db_profiles import get_user_profile
from db_plans import get_latest_meal_plan
from fact_extractor import async_extract_and_save_facts
# [P1-BG-THREAD-TIMEOUT · 2026-05-15] SSOT para fire-and-forget con timeout
# duro + alert. Reemplaza los `threading.Thread(target=..., daemon=True).start()`
# que vivían inline en este router. Ver `backend/bg_executor.py`.
from bg_executor import submit_bg_task
# [P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Helper SSOT del registry de knobs
# `MEALFIT_*`. Cualquier env var leída aquí se auto-registra en
# `_KNOBS_REGISTRY` y es visible en `/health/version`.
from knobs import _env_int, _env_float

# [P1-CHAT-TTS-TIMEOUT-HARDCODED · 2026-05-24] Knob del timeout `httpx` para
# ElevenLabs TTS. Pre-fix el `httpx.AsyncClient(timeout=15.0)` era literal
# hardcoded — si ElevenLabs degradaba latencia (incident regional), no había
# rollback sin redeploy. El test blanket `test_p1_new_httpx_timeout` cubría
# `routers/billing.py`; `chat.py` quedó fuera del scope original y exhibía
# el patrón prohibido. Default 15.0s = preserva comportamiento previo. Clamp
# [1.0, 60.0]: el piso defiende contra `=0.001` accidental (rompe en
# latencia normal), el techo contra `=120` que dejaría workers FastAPI
# bloqueados más allá de la SLA del frontend (60s total-graph timeout).
# Auto-registrado en `_KNOBS_REGISTRY` → visible en `/health/version`.
# Tooltip-anchor: P1-CHAT-TTS-TIMEOUT-HARDCODED.
_TTS_HTTPX_TIMEOUT_S = _env_float(
    "MEALFIT_TTS_HTTPX_TIMEOUT_S",
    15.0,
    validator=lambda v: 1.0 <= v <= 60.0,
)

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
    user_id = data.get("user_id", session_id)

    # Validación de seguridad IDOR
    if user_id and user_id != "guest" and user_id != session_id:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")

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
#   - Voice mode (call_mode) genera ~1 chunk/seg pero cada chunk va al
#     endpoint /tts (cap separado 60/min), NO al /stream. /stream solo
#     recibe el prompt completo del user, una vez por turn.
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


_CHAT_STREAM_LIMITER = RateLimiter(
    max_calls=_chat_stream_limiter_per_min(),
    period_seconds=60,
)

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
        # [P1-CHAT-TTS-TIMEOUT-HARDCODED · 2026-05-24] timeout via knob, no literal.
        async with httpx.AsyncClient(timeout=_TTS_HTTPX_TIMEOUT_S) as client:
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
            # [P2-PROD-AUDIT-3 · 2026-05-30] INSERT síncrono offloaded del event
            # loop (handler async). Ver nota en el finally.
            await asyncio.to_thread(
                execute_sql_write,
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
                        "timeout_threshold_s": _TTS_HTTPX_TIMEOUT_S,
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
                # [P2-PROD-AUDIT-3 · 2026-05-30] `log_api_usage` es un INSERT
                # síncrono (roundtrip DB ~10-200ms); este `finally` corre en CADA
                # request TTS (incl. el success path), y este handler es `async def`
                # → llamarlo directo bloqueaba el event loop del worker uvicorn.
                # Hermano del contrato P1-ASYNC-SYNC-DB-BLOCKING (plans.py usa
                # asyncio.to_thread; billing.py usa _supabase_async).
                await asyncio.to_thread(log_api_usage, verified_user_id, "elevenlabs_tts")
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
                                    save_message(
                                        session_id, "model", response_text,
                                        user_id=_db_user_id,
                                    )
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
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
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
                # identidad que Supabase verificó (no spoofeable vía body).
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

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"[CHAT STREAM] Error en api_chat_stream: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))



@router.post("", dependencies=[Depends(_CHAT_STREAM_LIMITER)])
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

        response_text, updated_fields, new_plan = chat_with_agent(session_id, prompt, current_plan=current_plan, user_id=user_id, form_data=form_data)

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

