# -*- coding: utf-8 -*-
"""[P1-PORTION-HONORED · 2026-09-07] Clasificador de errores TRANSITORIOS del proveedor LLM.

Extraído de `graph_orchestrator.py` para bajar del techo de líneas congelado —la regla del repo
es «extrae un módulo, no subas el tope» (`test_p3_shopping_projection_pkg.py`)— y se eligió ESTE
porque era el candidato más barato de todo el fichero: **72 líneas con CERO dependencias del
módulo**, medido con AST sobre las 300 funciones. Una extracción sin dependencias no puede
provocar el import circular que descartó a las funciones del cap de huevo.

No es una extracción arbitraria: «¿este fallo del proveedor merece reintento?» es un asunto
propio, y ya lo consumían dos sitios del orquestador y dos tests.

`graph_orchestrator` lo re-exporta, así que todo lo que lo importaba de allí sigue funcionando
(mismo patrón que `dish_naming.py`).
"""
from __future__ import annotations

def _is_transient_upstream_error(exc: BaseException) -> bool:
    """[P1-LLM-TRANSIENT-5XX · 2026-05-21] Detecta errores 5xx transitorios
    de Google que NO deben contar como failure en el LLMCircuitBreaker.

    Bug observado 2026-05-21 02:58:37:
      Google retornó `502 Bad Gateway` en la compresión + planner. El CB
      contó esos 3 retries como fallas → abrió `gemini-3.5-flash` por 30s →
      Días 1/2/3 cayeron con `Circuit Breaker OPEN` aunque el modelo
      principal estaba sano (era infra de Google teniendo un hipo).

    Distinto a `_is_rate_limit_error` (429): los 5xx son **del lado de
    Google** (problemas internos suyos), no del usuario/proyecto. La
    estrategia correcta: backoff + retry SIN contaminar el CB. Por eso
    los excluimos del conteo de failures.

    Cubre 502/503/504/INTERNAL/UNAVAILABLE — los códigos transitorios que
    Google documenta como retryable. Match por string + por attributes
    porque LangChain wrappea estos errores de formas inconsistentes entre
    versiones.

    Tooltip-anchor: P1-LLM-TRANSIENT-5XX.
    """
    try:
        _type_name = type(exc).__name__
        # [P1-TRANSIENT-PRO-ERRORS · 2026-07-27] Taxonomía del cliente OpenAI-compatible, que es
        # el que usa GLM desde P0-LLM-PROVIDER-MIGRATION (2026-06-12).
        #
        # Esta función se escribió el 2026-05-21 para las firmas de GOOGLE y NUNCA se actualizó
        # al proveedor nuevo. Resultado: `APIConnectionError` —un fallo de RED puro, el error
        # más transitorio que existe— contaba como mala salud del modelo y abría el circuit
        # breaker. Es exactamente el bug que esta función existe para evitar, descrito en su
        # propio docstring, con otro proveedor.
        #
        # Medido en los logs del VPS (6 h): 25 correcciones del self-critique intentadas, 12
        # perdidas — 6 por `pro_error:APIConnectionError` y **6 por `pro_cb_open`**, o sea el
        # breaker que abrieron las primeras. De ahí salieron 2 regeneraciones COMPLETAS de plan
        # (el revisor rechaza por "misma proteína repetida" justo lo que la corrección perdida
        # iba a arreglar). Un hipo de red desactivaba PRO para todos, incluido el revisor médico
        # que va a PRO en todos los tiers.
        #
        # Coste del fix: cero llamadas LLM extra. Solo deja de castigar al modelo por la red.
        if _type_name in (
            "APIConnectionError", "APITimeoutError", "InternalServerError",
            "ConnectionError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",
        ):
            return True
        # Excepciones canónicas que documentan transient upstream
        if _type_name in (
            "ServiceUnavailable", "InternalServerError", "GatewayTimeout",
            "DeadlineExceeded", "Aborted", "ServerError", "BadGateway",
        ):
            return True
        _msg = str(exc).lower() if exc else ""
        # HTTP code 502/503/504 en el mensaje (LangChain/genai wrappean así)
        if any(code in f" {_msg} " for code in (" 502 ", " 503 ", " 504 ")):
            return True
        if any(s in _msg for s in ("(502)", "(503)", "(504)", '"code":502', '"code":503', '"code":504')):
            return True
        # gRPC / google.api_core status strings
        if "bad gateway" in _msg or "gateway timeout" in _msg or "service unavailable" in _msg:
            return True
        if "internal" in _msg and ("server error" in _msg or "google" in _msg):
            return True
        if "unavailable" in _msg and ("backend" in _msg or "google" in _msg or "service" in _msg):
            return True
        # google.genai ClientError expone `.code` numérico
        _code = getattr(exc, "code", None)
        if _code in (502, 503, 504):
            return True
        return False
    except Exception:
        return False
