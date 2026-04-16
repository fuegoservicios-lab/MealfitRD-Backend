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

logger = logging.getLogger(__name__)

# ============================================================
# PERSONALITY PROFILES & SENTIMENT PROMPT (importados del paquete prompts/)
# ============================================================
from prompts.sentiment import PERSONALITY_PROFILES, SENTIMENT_PROMPT


@lru_cache(maxsize=1)
def _get_classifier_model():
    """Inicializa el modelo clasificador una sola vez (singleton)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        max_output_tokens=10,
        google_api_key=os.environ.get("GEMINI_API_KEY")
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
