"""[P1-AUDIT-3 · 2026-05-12] Path-param validators que rechazan input
no-UUID con HTTP 400 antes de que llegue a SQL.

Contexto:
    Antes, endpoints como `/api/chat/history/{session_id}` invocaban
    `get_session_owner(session_id)` que ejecuta `SELECT … WHERE id = %s`
    contra una columna `uuid`. Un probe externo con `session_id="test"` o
    `"0000"` producía:

        postgres ERROR: invalid input syntax for type uuid: "0000"

    y el handler devolvía 500 al cliente (sin información útil, polución de
    logs). El audit production-readiness 2026-05-12 observó 2 entries con
    estos valores en 24h de logs prod.

Decisión:
    Validar formato UUID al inicio del handler — antes de tocar DB. Retornar
    `HTTPException 400` con mensaje genérico (no leak del schema). El
    sentinel `"guest"` se acepta opcionalmente porque varios endpoints lo
    usan como bypass para sesiones anónimas (`/api/chat/sessions/{user_id}`
    permite `user_id="guest"`).

Patrón:
    Inline al inicio del handler:

        @router.get("/sessions/{user_id}")
        def api_get_chat_sessions(user_id: str, …):
            assert_valid_uuid(user_id, allow_guest=True)
            …

    Para `{session_id}`/`{plan_id}`/`{chunk_id}` (sin sentinel):

        assert_valid_uuid(session_id)

Tooltip-anchor: P1-AUDIT-3-UUID-VALIDATOR.
"""
from __future__ import annotations

import uuid as _uuid_mod

from fastapi import HTTPException

# Sentinel usado por endpoints de chat / sessions para distinguir flujo
# de usuario autenticado vs flujo anon. NO es un UUID — los handlers que
# lo aceptan deben pasar `allow_guest=True`.
_GUEST_SENTINEL = "guest"


def assert_valid_uuid(value: str, allow_guest: bool = False) -> str:
    """Verifica que `value` sea un UUID válido (o `"guest"` si
    `allow_guest=True`). Retorna el valor sin modificar para soportar
    `target = assert_valid_uuid(path_param)` en una sola línea.

    Raises:
        HTTPException 400: cuando `value` no es UUID parseable y no
            coincide con el sentinel `"guest"` (o `allow_guest=False`).

    Mensaje del 400 deliberadamente genérico ("Invalid identifier format.")
    para no leakear cuál columna es uuid ni qué validación se aplicó.
    """
    if value is None or value == "":
        raise HTTPException(status_code=400, detail="Invalid identifier format.")
    if allow_guest and value == _GUEST_SENTINEL:
        return value
    try:
        _uuid_mod.UUID(str(value))
        return value
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid identifier format.")
