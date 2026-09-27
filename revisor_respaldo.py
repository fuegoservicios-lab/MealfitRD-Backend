# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-615 · 2026-09-27] El revisor clínico no se queda sin modelo cuando OpenAI falla.

Producción, 27-sep 16:53-16:59 UTC (plan 3957a669, bloque 9): con perfil de riesgo el revisor es de OpenAI (Luna/Terra,
P1-REVIEWER-TIER-MODELS) y su breaker estaba ABIERTO —el saldo de OpenAI se había agotado—: «Error TRANSITORIO del
reviewer (LLMCircuitOpenError)» tres veces. Cada error transitorio devuelve el plan a `should_retry`, que lo REGENERA
entero: dos generaciones más pagadas que tampoco podían aprobarse, y el bloque se entregó degradado
(`plan_quality_degraded`, `review_failed_delivered_rate_high` al 33 %). El respaldo existía sólo para cuando falta la
clave (`_reviewer_model_name`: «el gate clínico NUNCA se queda sin modelo utilizable»); con la clave presente y la
cuenta sin saldo, sin red.

Aquí, cuando el revisor de OpenAI falla por infraestructura —breaker abierto, cuota o límite, autenticación, 5xx,
tiempo, respuesta que no parsea—, el mismo veredicto se pide al modelo de respaldo (`_REVIEWER_RISK_TIER_DEFAULT`, el del
fail-safe sin clave) antes de rendirse; el plan lo anota en `_reviewer_fallback`. Nunca ante el tope de gasto del plan
(eso no es un fallo del proveedor) ni si el revisor ya es el de respaldo. Knob `MEALFIT_REVIEWER_CROSS_PROVIDER_FALLBACK`
(True). tooltip-anchor: P1-PLAN-LOTE-615
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: fallos del cliente OpenAI que dejan al revisor sin veredicto sin ser un rechazo clínico
_OPENAI_API_ERRORS = ("RateLimitError", "AuthenticationError", "PermissionDeniedError", "NotFoundError",
                      "APIStatusError", "APIError", "APIConnectionError", "APITimeoutError", "InternalServerError",
                      "UnprocessableEntityError", "ConflictError")


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REVIEWER_CROSS_PROVIDER_FALLBACK", True)
    except Exception:                                                          # noqa: BLE001
        return True


def respaldo(modelo, exc, plan=None):
    """El modelo con el que repetir la revisión, o None si este fallo no se resuelve cambiando de proveedor."""
    try:
        if not activo() or exc is None:
            return None
        import graph_orchestrator as go
        if not go.is_openai_model(modelo) or go._is_plan_spend_cap_error(exc):
            return None
        destino = go._REVIEWER_RISK_TIER_DEFAULT
        if not destino or destino == modelo or go.is_openai_model(destino):
            return None
        if not (go._is_reviewer_transient_error(exc) or type(exc).__name__ in _OPENAI_API_ERRORS):
            return None
        if isinstance(plan, dict):
            plan["_reviewer_fallback"] = {"from": str(modelo), "to": str(destino), "error": type(exc).__name__}
        logger.warning(f"🛟 [P1-PLAN-LOTE-615] Revisor clínico {modelo} sin servicio ({type(exc).__name__}: "
                       f"{str(exc)[:120]}) → mismo veredicto con {destino}, sin regenerar el plan.")
        return destino
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-615] no-op: {type(e).__name__}: {e}")
        return None


__all__ = ["respaldo"]
