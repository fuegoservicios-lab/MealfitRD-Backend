# backend/sentiment_classifier.py
"""
Clasificador de Sentimiento Adaptativo para MealfitRD.
Analiza el tono emocional del mensaje del usuario y retorna una personalidad
dinámica que se inyecta al system prompt del agente principal.
"""

import os
import json
import logging
import time
from functools import lru_cache
from knobs import _env_float, _env_str

logger = logging.getLogger(__name__)

# ============================================================
# PERSONALITY PROFILES & SENTIMENT PROMPT (importados del paquete prompts/)
# ============================================================
from prompts.sentiment import PERSONALITY_PROFILES, SENTIMENT_PROMPT


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Cohorte omitida del sweep original: el
# clasificador de sentimiento construye `ChatGoogleGenerativeAI` SIN `timeout=`.
# Corre INLINE en el hot-path del chat (agent.py classify_sentiment) ANTES del
# wrapper `_graph_timeout`, así que un socket colgado bloquearía el thread del
# worker uvicorn. `_env_float` con clamp.
# [P3-PREVIEW-MODEL-KNOB · 2026-05-30] Modelo a knob (era hardcoded) — un preview
# de Gemini puede deprecarse sin aviso; permite swap sin redeploy.
def _sentiment_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_SENTIMENT_LLM_TIMEOUT_S",
        8.0,
        validator=lambda v: 0.0 < v <= 60.0,
    )


def _sentiment_model_name() -> str:
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default DeepSeek V4 Flash (aux barato).
    from llm_provider import DEEPSEEK_FLASH
    return _env_str("MEALFIT_SENTIMENT_MODEL", DEEPSEEK_FLASH)


@lru_cache(maxsize=1)
def _get_classifier_model():
    """Inicializa el modelo clasificador una sola vez (singleton).

    Nota: el `@lru_cache(maxsize=1)` fija el `timeout=`/`model=` al primer build;
    cambiar el knob requiere reinicio del worker (aceptable — son knobs de
    operación, no intra-request)."""
    from llm_provider import ChatDeepSeek  # [P0-DEEPSEEK-MIGRATION]
    return ChatDeepSeek(
        model=_sentiment_model_name(),
        temperature=0.0,
        max_output_tokens=10,
        # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] deadline duro del request.
        timeout=_sentiment_llm_timeout_s()
    )


def classify_sentiment(user_message: str) -> dict:
    """
    Clasifica el sentimiento del mensaje del usuario.
    Retorna un dict con: sentiment, name, emoji, instruction.
    Fallback a 'neutral' si falla.
    """
    if not user_message or len(user_message.strip()) < 3:
        return {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"}
    
    try:
        start = time.time()
        model = _get_classifier_model()
        
        response = model.invoke(SENTIMENT_PROMPT.format(message=user_message[:300]))
        
        # Manejar respuestas tipo lista (Gemini a veces devuelve content como list)
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        raw = str(content).strip().lower().replace('"', '').replace("'", "")
        
        # Extraer la categoría del response
        valid_sentiments = list(PERSONALITY_PROFILES.keys())
        detected = "neutral"
        for s in valid_sentiments:
            if s in raw:
                detected = s
                break
        
        elapsed = time.time() - start
        profile = PERSONALITY_PROFILES[detected]
        
        logger.info(f"🎭 [SENTIMENT] '{user_message[:50]}...' → {profile['emoji']} {profile['name']} ({elapsed:.2f}s)")
        
        return {
            "sentiment": detected,
            "name": profile["name"],
            "emoji": profile["emoji"],
            "instruction": profile["instruction"]
        }
        
    except Exception as e:
        logger.warning(f"⚠️ [SENTIMENT] Error en clasificador, fallback a neutral: {e}")
        return {
            "sentiment": "neutral",
            **PERSONALITY_PROFILES["neutral"]
        }
