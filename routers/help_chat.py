"""[P2-HELP-CHATBOT · 2026-07-04] Chatbot de ayuda del menú "Obtener ayuda".

POST /api/help/chat — Q&A de producto (qué es MealfitRD, planes/precios,
cómo usar cada sección). SIN acceso a datos del usuario: cero tools, cero
DB, cero user_id en el prompt — el conocimiento vive en
`prompts.help_bot.HELP_BOT_SYSTEM_PROMPT` y la escalación es el correo de
soporte canónico. Para preguntas sobre "MI plan" el bot redirige al Agente.

Decisiones (test ancla `tests/test_p2_help_chatbot.py`):
  - **Quota-exempt**: NO `verify_api_quota` NI `log_api_usage`. Lección
    P1-NEVERA-QUOTA-EXEMPT: `get_monthly_api_usage` cuenta TODA fila de
    `api_usage` sin filtrar endpoint — loguear aquí quemaría crédito de
    planes por pedir ayuda, y el paywall 402 bloquearía soporte justo al
    usuario que llegó al cap (el que MÁS necesita entender los planes).
    Anti-spam: `RateLimiter` per-bucket (user_id o IP para invitados).
  - **Fail-cheap**: modelo flash del router por tier (el bot no razona
    clínica); override sin redeploy vía `MEALFIT_HELP_CHAT_MODEL`
    (convención P3-PREVIEW-MODEL-KNOB, helper `_help_chat_model_name`).
  - **Kill switch**: `MEALFIT_HELP_CHAT_ENABLED` (leído por request →
    flip sin tocar código; en el VPS requiere restart igual que todo env,
    pero deja el rollback como one-liner de .env).
  - Import de `llm_provider` LAZY dentro del handler: mantiene el import
    del router liviano para arranque/tests.
"""
from typing import Optional
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from rate_limiter import RateLimiter
from knobs import _env_bool, _env_int, _env_float, _env_str
from prompts.help_bot import (
    HELP_BOT_SYSTEM_PROMPT,
    HelpChatValidationError,
    sanitize_help_messages,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/help", tags=["help"])

# Anti-spam per-bucket (user autenticado o IP para invitados). 8/min cubre
# con holgura una conversación humana de soporte (~1-2 preguntas/min) y
# acota el burst de un abusador a ~$0.001/min en flash.
_HELP_CHAT_LIMITER = RateLimiter(
    max_calls=_env_int(
        "MEALFIT_HELP_CHAT_LIMITER_PER_MIN", 8, validator=lambda v: 1 <= v <= 120
    ),
    period_seconds=60,
)

# Bounds del payload (misma lección P0-CHAT-PROMPT-MAXLEN: sin caps, un
# cliente puede quemar tokens del owner con blobs gigantes).
_HELP_CHAT_MAX_TURNS = _env_int(
    "MEALFIT_HELP_CHAT_MAX_TURNS", 12, validator=lambda v: 1 <= v <= 60
)
_HELP_CHAT_MAX_CHARS = _env_int(
    "MEALFIT_HELP_CHAT_MAX_CHARS", 1500, validator=lambda v: 100 <= v <= 8192
)
_HELP_CHAT_MAX_OUTPUT_TOKENS = _env_int(
    "MEALFIT_HELP_CHAT_MAX_OUTPUT_TOKENS", 700, validator=lambda v: 100 <= v <= 4000
)
_HELP_CHAT_LLM_TIMEOUT_S = _env_float(
    "MEALFIT_HELP_CHAT_LLM_TIMEOUT_S", 30.0, validator=lambda v: 5.0 <= v <= 120.0
)

_HELP_CHAT_UNAVAILABLE_MSG = (
    "El asistente no está disponible en este momento. "
    "Escríbenos a fuego.servicios@gmail.com y te ayudamos por correo."
)


def _help_chat_model_name() -> str:
    """[P3-PREVIEW-MODEL-KNOB] Modelo del help-chat: override per-feature
    `MEALFIT_HELP_CHAT_MODEL` gana SIEMPRE; default = flash del tier gratis
    (fail-cheap — Q&A de producto no necesita el modelo pro).
    Tooltip-anchor: P2-HELP-CHATBOT."""
    override = _env_str("MEALFIT_HELP_CHAT_MODEL", "")
    if override:
        return override
    from llm_provider import model_free_tier

    return model_free_tier()


@router.post("/chat")
async def api_help_chat(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_HELP_CHAT_LIMITER),
):
    """Responde una duda de producto. Body: `{"messages": [{role, content}...]}`
    (historial client-held; el último mensaje debe ser del usuario).
    Retorna `{"reply": str}`. Invitados permitidos (bucket por IP)."""
    if not _env_bool("MEALFIT_HELP_CHAT_ENABLED", True):
        raise HTTPException(status_code=503, detail=_HELP_CHAT_UNAVAILABLE_MSG)

    try:
        messages = sanitize_help_messages(
            (data or {}).get("messages"),
            max_turns=_HELP_CHAT_MAX_TURNS,
            max_chars=_HELP_CHAT_MAX_CHARS,
        )
    except HelpChatValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Lazy: ver docstring del módulo.
    from llm_provider import ChatDeepSeek
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = [SystemMessage(content=HELP_BOT_SYSTEM_PROMPT)]
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    try:
        llm = ChatDeepSeek(
            model=_help_chat_model_name(),
            temperature=0.3,
            max_output_tokens=_HELP_CHAT_MAX_OUTPUT_TOKENS,
            timeout=_HELP_CHAT_LLM_TIMEOUT_S,
            max_retries=1,
        )
        reply = await llm.ainvoke(lc_messages)
        content = getattr(reply, "content", "")
        if isinstance(content, list):  # providers multi-part → aplanar defensivo
            content = "".join(str(part) for part in content)
        text = (content or "").strip()
    except Exception as exc:
        logger.warning(
            f"⚠️ [P2-HELP-CHATBOT] LLM del help-chat falló "
            f"(auth={'sí' if verified_user_id else 'no'}): {exc}"
        )
        raise HTTPException(status_code=502, detail=_HELP_CHAT_UNAVAILABLE_MSG)

    if not text:
        logger.warning("⚠️ [P2-HELP-CHATBOT] Respuesta vacía del LLM del help-chat.")
        raise HTTPException(status_code=502, detail=_HELP_CHAT_UNAVAILABLE_MSG)

    return {"reply": text}
