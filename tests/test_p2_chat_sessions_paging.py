"""[P2-CHAT-SESSIONS-PAGING · 2026-09-03] Recientes del coach se cortaba en 60 sesiones (LIMIT fijo)
y los chats más viejos dejaban de existir sin aviso. Ahora `GET /api/chat/sessions/{user_id}`
acepta `offset`, la consulta pagina con LIMIT/OFFSET parametrizados y la respuesta lleva
`has_more` (calculado con un count real, no con "vinieron 60"). El invitado no pagina.

Parser-based: el contrato vive en el texto de prod; un renombre lo rompe antes de tocar prod.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_DB = (_BACKEND / "db_chat.py").read_text(encoding="utf-8")
_ROUTER = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


def test_la_consulta_pagina_con_limit_y_offset_parametrizados():
    assert "CHAT_SESSIONS_PAGE_SIZE = 60" in _DB
    assert "def get_user_chat_sessions(user_id: str, limit: int = CHAT_SESSIONS_PAGE_SIZE, offset: int = 0):" in _DB
    assert re.search(r'"ORDER BY created_at DESC LIMIT %s OFFSET %s"', _DB), "el LIMIT volvió a ser fijo"
    assert "(user_id, int(limit), int(offset))," in _DB
    assert 'LIMIT 60"' not in _DB, "quedó un LIMIT 60 literal en db_chat.py"


def test_has_more_sale_de_un_count_real():
    assert "def count_user_chat_sessions(user_id: str) -> int:" in _DB
    assert "SELECT count(*) AS total FROM public.agent_sessions WHERE user_id = %s" in _DB


def test_el_endpoint_acepta_offset_acotado_y_devuelve_has_more():
    assert "offset: int = 0," in _ROUTER
    assert "_offset = max(0, min(int(offset or 0), 10_000))" in _ROUTER
    assert "get_user_chat_sessions(user_id, limit=CHAT_SESSIONS_PAGE_SIZE, offset=_offset)" in _ROUTER
    assert "has_more = count_user_chat_sessions(user_id) > _offset + CHAT_SESSIONS_PAGE_SIZE" in _ROUTER
    assert 'return {"sessions": sessions, "has_more": has_more}' in _ROUTER
    # el invitado no pagina: has_more solo se calcula para cuentas
    i = _ROUTER.index("has_more = False")
    j = _ROUTER.index("has_more = count_user_chat_sessions")
    assert 'if user_id and user_id != "guest":' in _ROUTER[i:j]


def test_el_router_importa_los_helpers_por_la_fachada_db():
    m = re.search(r"from db import \(([^)]*)\)", _ROUTER, re.S) or re.search(r"^from db import ([^\n]+)$", _ROUTER, re.M)
    assert m, "sin import de la fachada db en routers/chat.py"
    assert "count_user_chat_sessions" in m.group(1) and "CHAT_SESSIONS_PAGE_SIZE" in m.group(1)


def test_marker_bumpeado():
    assert '_LAST_KNOWN_PFIX = "P2-CHAT-SESSIONS-PAGING · 2026-09-03"' in _APP
