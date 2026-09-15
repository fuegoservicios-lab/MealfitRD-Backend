"""[P0-CHAT-IDENTITY-FROM-TOKEN · 2026-09-14] La identidad del turno del coach sale SOLO del token.

## El agujero

Los tres endpoints que escriben en el chat (`/stream`, `POST /api/chat`, `/message`) tomaban
`user_id` del BODY y lo validaban contra el token solo cuando `user_id != session_id`. Una
petición SIN token con `session_id = user_id = <UUID de la víctima>`:

  1. saltaba el guard IDOR (la condición `user_id != session_id` era falsa);
  2. saltaba el de dueño de sesión (la sesión aún no existía: `get_session_owner` → None);
  3. pasaba `verify_coach_quota`, que devuelve None sin error cuando no hay token.

El agente recibía ese UUID como si estuviera autenticado — plan, alergias, Nevera y diario de
la víctima en el prompt — y el override P0-AGENT-1 fijaba el `user_id` de las TOOLS a ese
mismo valor: escrituras sobre la cuenta de la víctima.

## El contrato

Con token, el turno es del `verified_user_id` (un `user_id` ajeno en el body es 401). Sin
token, SIEMPRE "guest".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_ROUTER = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")

from routers.chat import _resolve_chat_identity  # noqa: E402

VICTIMA = "11111111-2222-3333-4444-555555555555"
ATACANTE = "99999999-8888-7777-6666-555555555555"


def test_sin_token_el_uuid_de_la_victima_como_session_y_user_es_invitado():
    """El vector exacto: sin token, `session_id == user_id == <víctima>`."""
    assert _resolve_chat_identity(VICTIMA, VICTIMA, None) == "guest"


@pytest.mark.parametrize("body_uid", [None, "", "guest", VICTIMA, "cualquier-cosa"])
def test_sin_token_siempre_invitado(body_uid):
    assert _resolve_chat_identity(body_uid, "sesion-local", None) == "guest"


def test_con_token_el_turno_es_del_token():
    assert _resolve_chat_identity(ATACANTE, "s1", ATACANTE) == ATACANTE
    # el frontend puede mandar "guest" o su propio session_id: gana el token
    assert _resolve_chat_identity("guest", "s1", ATACANTE) == ATACANTE
    assert _resolve_chat_identity("s1", "s1", ATACANTE) == ATACANTE
    assert _resolve_chat_identity(None, "s1", ATACANTE) == ATACANTE


def test_con_token_un_user_id_ajeno_en_el_body_es_401():
    with pytest.raises(HTTPException) as exc:
        _resolve_chat_identity(VICTIMA, "s1", ATACANTE)
    assert exc.value.status_code == 401


@pytest.mark.parametrize("endpoint", [
    "def api_save_chat_message(",
    "def api_chat_stream(",
    "def api_chat(",
])
def test_los_tres_endpoints_resuelven_con_el_helper(endpoint):
    i = _ROUTER.index(endpoint)
    j = _ROUTER.find("\n@router.", i)
    body = _ROUTER[i:j if j > 0 else None]
    assert "_resolve_chat_identity(data.get(\"user_id\"), session_id, verified_user_id)" in body, (
        f"{endpoint} no deriva la identidad del token"
    )
    # el patrón viejo (user_id del body con default al session_id) no puede volver
    assert not re.search(r"user_id\s*=\s*data\.get\(\"user_id\",\s*session_id\)", body), (
        f"{endpoint} vuelve a tomar user_id del body"
    )
