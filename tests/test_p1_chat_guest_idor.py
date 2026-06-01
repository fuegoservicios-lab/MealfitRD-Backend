"""[P1-CHAT-GUEST-IDOR · 2026-05-30] Cierre del IDOR en
GET /api/chat/sessions/{user_id}?session_ids=...

Pre-fix (descubierto en audit prod-readiness 2026-05-30, workflow multi-agente):
    El handler `api_get_chat_sessions` (routers/chat.py:192) protegía el path
    param `user_id` (rechaza si `verified_user_id != user_id`), pero luego
    leía el query param `session_ids` (CSV desde localStorage del cliente) y lo
    pasaba a `get_guest_chat_sessions(session_ids.split(","))`.

    `get_guest_chat_sessions` (db_chat.py) hacía:
        supabase.table("agent_sessions").select("*").in_("id", session_ids[:20]).execute()
    SIN filtro de ownership. `SUPABASE_KEY = service_role` bypassea RLS, así
    que cualquier id pasado se devolvía sin importar el dueño. Vector:

        usuario A autenticado llama
        GET /api/chat/sessions/<A_uid>?session_ids=<sesión_de_B>
        → recibe la sesión de B + el snippet del primer mensaje de B
        (IDOR + leak de PII; `_process_and_sort_sessions` deriva `title`
         del primer user-message de la sesión).

Post-fix:
    `get_guest_chat_sessions` añade `.is_("user_id", "null")` a la query, de
    modo que SOLO las sesiones genuinas de invitado (sin dueño) son
    recuperables por id crudo. Las sesiones propias de un usuario logueado las
    trae `get_user_chat_sessions(user_id)` filtrando por owner; una sesión
    creada como guest y luego reclamada queda con `user_id` no-nulo → sale por
    el path de owner, no por este.

Contratos anclados:
    1. Parser: el cuerpo de `get_guest_chat_sessions` contiene
       `.is_("user_id", "null")` y aparece DESPUÉS del `.in_("id"`.
    2. Parser: anchor `P1-CHAT-GUEST-IDOR` presente en db_chat.py.
    3. Funcional: la cadena de query realmente invoca `.is_("user_id", "null")`
       antes de `.execute()` (un mock-spy de supabase lo verifica).
    4. Funcional: si la query se construye sin el filtro, el test falla.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CHAT_PY = _BACKEND_ROOT / "db_chat.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:\n(?:[ \t]+.*\n|[ \t]*\n)+",
        re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"No se encontró def `{name}` en el source."
    return m.group(0)


# ---------------------------------------------------------------------------
# Parser-based
# ---------------------------------------------------------------------------

def test_guest_sessions_query_filters_user_id_null():
    """La query de `get_guest_chat_sessions` DEBE incluir `.is_("user_id", "null")`."""
    src = _read(_DB_CHAT_PY)
    body = _extract_function(src, "get_guest_chat_sessions")
    # Excluir el docstring (que cita el filtro narrativamente) para verificar
    # que el `.is_` está en el CÓDIGO real, no solo mencionado en la prosa.
    code_only = re.sub(r'""".*?"""', "", body, count=1, flags=re.DOTALL)
    assert '.in_("id"' in code_only, (
        "Falta el filtro `.in_(\"id\", ...)` en get_guest_chat_sessions."
    )
    assert '.is_("user_id", "null")' in code_only, (
        "Falta `.is_(\"user_id\", \"null\")` en el CÓDIGO de `get_guest_chat_sessions`. "
        "Sin ese filtro, un usuario autenticado puede recuperar sesiones de OTRO "
        "usuario pasando `?session_ids=<sesión ajena>` (IDOR P1-CHAT-GUEST-IDOR). "
        "Lee la memoria antes de remover el filtro."
    )


def test_anchor_present_in_db_chat():
    src = _read(_DB_CHAT_PY)
    assert "P1-CHAT-GUEST-IDOR" in src, (
        "Falta anchor `P1-CHAT-GUEST-IDOR` en db_chat.py."
    )


# ---------------------------------------------------------------------------
# Funcional: mock-spy de la cadena supabase
# ---------------------------------------------------------------------------

class _QuerySpy:
    """Registra los filtros aplicados a la cadena de query supabase-py."""

    def __init__(self, recorder: dict):
        self._rec = recorder
        self._rec.setdefault("calls", [])

    def select(self, *a, **k):
        self._rec["calls"].append(("select", a))
        return self

    def in_(self, col, vals):
        self._rec["calls"].append(("in_", col))
        return self

    def is_(self, col, val):
        self._rec["calls"].append(("is_", col, val))
        return self

    def execute(self):
        self._rec["executed"] = True
        # Devuelve un resultado vacío para no entrar a _process_and_sort_sessions.
        res = MagicMock()
        res.data = []
        return res


def test_guest_sessions_applies_user_id_null_filter_at_runtime():
    """En runtime la cadena DEBE invocar `.is_("user_id", "null")` antes de execute()."""
    import db_chat

    recorder: dict = {}
    fake_supabase = MagicMock()
    fake_supabase.table.return_value = _QuerySpy(recorder)

    with patch.object(db_chat, "supabase", fake_supabase):
        out = db_chat.get_guest_chat_sessions(["sess-a", "sess-b"])

    assert out == []  # data vacía → sin sesiones
    assert recorder.get("executed") is True, "La query nunca se ejecutó."
    is_calls = [c for c in recorder["calls"] if c[0] == "is_"]
    assert ("is_", "user_id", "null") in is_calls, (
        "La query de get_guest_chat_sessions NO aplicó `.is_(\"user_id\", \"null\")` "
        "en runtime. Eso reabre el IDOR P1-CHAT-GUEST-IDOR: se devolverían sesiones "
        f"de cualquier dueño. Filtros aplicados: {recorder['calls']}"
    )
