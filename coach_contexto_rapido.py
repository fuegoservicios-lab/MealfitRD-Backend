"""[P1-PLAN-LOTE-1020 · A-11] Fase de contexto sin las dos llamadas al modelo que no cambian la respuesta.

Medido en producción (7 días, 11 turnos con desglose, 15-nov): `context_ready_ms` p50 1593 / p90 2399, y de eso
`sentiment_rag_router_ms` p50 1352 / p90 1987: el clasificador de sentimiento y el router de la memoria (dos llamadas al
modelo en paralelo) son el 85 % de la fase. La búsqueda en la memoria en sí (`rag_ms`) es p50 190.

  · Router de la memoria: solo reescribe la frase para el embedding («Desayuno» → «Preferencias y opciones de alimentos
    para el desayuno»). El modo voz ya busca con la frase tal cual desde el lote 684 (`_rag_sin_router`) sin pérdida
    medida. Con el canario se hace igual en el chat escrito.
  · Sentimiento: 81 de 95 clasificaciones en 10 días de logs fueron «neutral» (85 %). Con el canario se salta solo cuando
    el mensaje no trae NINGUNA señal afectiva (`chat_prompt_opt.es_turno_objetivo_sin_sentimiento`: cansancio,
    frustración, culpa, alegría, «me siento», emojis, «!!»…) y está en español (esa lista de señales es en español; en
    otro idioma se sigue preguntando al modelo). Si hay señal, se clasifica como siempre: la empatía no se toca.

Por canario `contexto_rapido` (`MEALFIT_CHAT_CANARY_CONTEXTO_RAPIDO`, off/canary/on; por defecto canary = solo las cuentas
de `MEALFIT_CHAT_CANARY_USER_IDS`). Fail-open: cualquier error deja las decisiones como venían.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _mensaje_en_espanol(prompt: str) -> bool:
    try:
        from prompts.chat_agent import detect_message_locale
        return not detect_message_locale(prompt or "")
    except Exception:
        return False


def aplicar(tracker, user_id: Optional[str], prompt: str, do_sentiment: bool, do_rag_router: bool, do_rag: bool,
            rag_decision, rag_sin_router: Callable[[str], dict]):
    """Devuelve `(do_sentiment, do_rag_router, rag_decision)` ya ajustados para este turno."""
    original = (do_sentiment, do_rag_router, rag_decision)
    try:
        if not (do_sentiment or do_rag_router):
            return do_sentiment, do_rag_router, rag_decision
        from coach_canario import registrar_brazo
        if not registrar_brazo(tracker, user_id, "contexto_rapido"):
            return do_sentiment, do_rag_router, rag_decision
        if do_rag_router:
            do_rag_router = False
            rag_decision = rag_sin_router(prompt) if do_rag else None
        if do_sentiment:
            from chat_prompt_opt import es_turno_objetivo_sin_sentimiento
            if es_turno_objetivo_sin_sentimiento(prompt) and _mensaje_en_espanol(prompt):
                do_sentiment = False
        return do_sentiment, do_rag_router, rag_decision
    except Exception as exc:  # noqa: BLE001 — la fase de contexto nunca se rompe por esto
        try:
            from fail_open import fail_open
            fail_open("coach_contexto_rapido", exc)
        except Exception:
            logger.debug(f"[LOTE-1020] contexto rápido no aplicado: {exc!r}")
        return original
