# -*- coding: utf-8 -*-
"""[REVISION-LOTES-1001-1011 · 2026-11-14] La caché de contexto del coach (lote 1010 · G-12, 30 s) se invalida también
cuando el usuario escribe DESDE LA APP.

El lote 1010 guarda 30 s en memoria lo que el coach lee al empezar cada turno (diario de hoy, Nevera, agua,
suplementos) y solo la vaciaba cuando escribía una tool del propio coach. Lo que el usuario cambia desde la app (anota
una comida con el escáner, edita la Nevera, suma un vaso en la tarjeta) no la tocaba: si abría el chat enseguida y
preguntaba «¿cuánto me falta?», el coach contestaba con el diario de antes — y el bloque de la Nevera dice «cantidades
REALES de la base de datos en este instante».

Middleware ASGI puro (no envuelve el cuerpo ni toca el streaming del chat): en toda petición que escribe (POST, PUT,
PATCH, DELETE) fuera de `/api/chat/`, al arrancar la respuesta (el handler ya escribió) vacía la caché entera. Es un
dict de ≤ 256 entradas en un solo worker de uvicorn: vaciarla cuesta nada y no hace falta saber de quién es la
petición. Las escrituras del chat ya invalidan por usuario en `agent.execute_tools`.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_METODOS_QUE_ESCRIBEN = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_PREFIJO_CHAT = "/api/chat/"


def _vaciar_cache_de_contexto() -> None:
    agente = sys.modules.get("agent")   # sin importar agent.py aquí: si no está cargado, no hay caché que vaciar
    if agente is None:
        return
    try:
        agente.invalidate_chat_user_ctx_cache(None)
    except Exception as exc:  # noqa: BLE001 — nunca romper una respuesta por la caché
        from fail_open import fail_open   # [P1-PLAN-LOTE-1014 · A-04] una caché que no se vacía sirve datos viejos
        fail_open("ctx_cache_invalidate", exc, critico=True)


def escribe_fuera_del_chat(scope) -> bool:
    return (
        scope.get("type") == "http"
        and str(scope.get("method") or "").upper() in _METODOS_QUE_ESCRIBEN
        and not str(scope.get("path") or "").startswith(_PREFIJO_CHAT)
    )


class ContextoDelCoachFrescoMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not escribe_fuera_del_chat(scope):
            await self.app(scope, receive, send)
            return
        vaciada = False

        async def _send(message):
            nonlocal vaciada
            if not vaciada and message.get("type") == "http.response.start":
                vaciada = True
                _vaciar_cache_de_contexto()
            await send(message)

        await self.app(scope, receive, _send)
