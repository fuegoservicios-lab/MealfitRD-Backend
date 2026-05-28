"""[P0-CHAT-DELETE-IDOR · 2026-05-26] Cierre del IDOR en DELETE /api/chat/session/{session_id}.

Pre-fix (descubierto en audit prod-readiness 2026-05-26):
    El endpoint `@router.delete("/session/{session_id}")` en `routers/chat.py:283`
    solo validaba `if not verified_user_id` (¿está logueado?), pero NO
    `session.user_id == verified_user_id`. El docstring mismo lo admitía
    literalmente:

        "Elimina una sesión de chat. Requiere autenticación pero sin
         validación IDOR (RLS desactivado — la auth se maneja aquí)."

    Vector: cualquier usuario autenticado podía DELETE el chat de otro
    usuario pasando un `session_id` enumerado/leakeado. La cascada borraba
    `conversation_summaries` + `agent_messages` + `agent_sessions`.
    `service_role` (SUPABASE_KEY) bypassea RLS → la única defensa era
    el guard del endpoint, y el guard no existía.

Post-fix:
    - `delete_chat_session(session_id, user_id)` en `db_chat.py:310` ahora
       hace pre-check via `get_session_owner` SSOT (mismo guard que el
       GET /history en `routers/chat.py:266`).
    - Router pasa `verified_user_id` al helper y mapea error_msg a HTTP:
       "not_found" → 404, "forbidden" → 403, otros → 500.
    - Defensa-en-profundidad: el DELETE de `agent_sessions` incluye
       `AND user_id = %s` adicional (cumple invariante I2 del repo).

Este test ancla 5 contratos parser-based + 2 tests funcionales:
    1. La signature de `delete_chat_session` toma `user_id`.
    2. El router pasa `verified_user_id` al helper.
    3. El helper invoca `get_session_owner` ANTES de los DELETEs.
    4. El helper retorna "forbidden" cuando owner != caller.
    5. El DELETE de `agent_sessions` incluye `AND user_id = %s`.
    6. Anchor `P0-CHAT-DELETE-IDOR` presente en ambos archivos.
    7. Tests funcionales: el helper retorna ("forbidden", ...) / ("not_found", ...)
       según el dueño retornado por un `get_session_owner` mockeado.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CHAT_PY = _BACKEND_ROOT / "db_chat.py"
_ROUTER_PY = _BACKEND_ROOT / "routers" / "chat.py"


# ---------------------------------------------------------------------------
# Helpers de extracción
# ---------------------------------------------------------------------------

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    """Devuelve el cuerpo de la función `name` desde `def <name>(` hasta el
    siguiente `def ` toplevel o EOF.
    """
    pattern = re.compile(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:\n(?:[ \t]+.*\n|[ \t]*\n)+",
        re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"No se encontró def `{name}` en el source."
    return m.group(0)


def _extract_endpoint_body(src: str, method: str, path: str) -> str:
    """Extrae el cuerpo del handler decorado por `@router.<method>("<path>")`."""
    pattern = re.compile(
        rf'@router\.{re.escape(method)}\(\s*[\"\']'
        + re.escape(path)
        + r'[\"\']\s*\)\s*\n'
        r'(?:def|async def)\s+\w+\s*\([^)]*\)[^:]*:\n'
        r'(?:[ \t]+.*\n|[ \t]*\n)+',
        re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"No se encontró @router.{method}({path!r}) en el source."
    return m.group(0)


# ---------------------------------------------------------------------------
# Tests parser-based
# ---------------------------------------------------------------------------

def test_db_chat_signature_includes_user_id():
    """`delete_chat_session` debe aceptar `user_id` además de `session_id`."""
    src = _read(_DB_CHAT_PY)
    m = re.search(
        r"def\s+delete_chat_session\s*\(\s*session_id\s*:\s*str\s*,"
        r"\s*user_id\s*:\s*str\s*\)\s*->\s*Tuple\[bool,\s*str\]",
        src,
    )
    assert m is not None, (
        "La signature de `delete_chat_session` debe ser "
        "`def delete_chat_session(session_id: str, user_id: str) -> Tuple[bool, str]`. "
        "Si revertiste a la signature vieja sin `user_id`, abriste de nuevo el IDOR "
        "del P0-CHAT-DELETE-IDOR. Lee la memoria antes de seguir."
    )


def test_router_passes_verified_user_id_to_helper():
    """El router debe llamar `delete_chat_session(session_id, verified_user_id)`."""
    src = _read(_ROUTER_PY)
    body = _extract_endpoint_body(src, "delete", "/session/{session_id}")
    m = re.search(
        r"delete_chat_session\s*\(\s*session_id\s*,\s*verified_user_id\s*\)",
        body,
    )
    assert m is not None, (
        "El handler `api_delete_chat_session` debe pasar `verified_user_id` al "
        "helper. Sin ese argumento el helper no puede validar ownership."
    )


def test_helper_calls_get_session_owner_before_deletes():
    """Pre-check de ownership debe ocurrir ANTES del primer execute_sql_write."""
    src = _read(_DB_CHAT_PY)
    body = _extract_function(src, "delete_chat_session")
    pos_owner = body.find("get_session_owner")
    pos_first_delete = body.find("execute_sql_write")
    assert pos_owner != -1, (
        "El helper debe invocar `get_session_owner` para validar ownership. "
        "Sin ese pre-check el IDOR P0-CHAT-DELETE-IDOR se re-abre."
    )
    assert pos_first_delete != -1, "Faltan execute_sql_write en el helper."
    assert pos_owner < pos_first_delete, (
        "`get_session_owner` debe ejecutarse ANTES del primer DELETE. "
        "Si invoca DELETE primero, el atacante ya borró antes de validar."
    )


def test_helper_returns_forbidden_when_owner_mismatch():
    """El helper debe retornar la cadena literal `forbidden` cuando owner != caller."""
    src = _read(_DB_CHAT_PY)
    body = _extract_function(src, "delete_chat_session")
    assert '"forbidden"' in body or "'forbidden'" in body, (
        "El helper debe retornar `(False, 'forbidden')` cuando el dueño de la "
        "sesión no coincide con el caller. El router mapea esta cadena a HTTP 403."
    )
    assert '"not_found"' in body or "'not_found'" in body, (
        "El helper debe retornar `(False, 'not_found')` cuando `get_session_owner` "
        "devuelve None. El router mapea esta cadena a HTTP 404."
    )


def test_agent_sessions_delete_filters_by_user_id():
    """Defensa-en-profundidad: DELETE de agent_sessions debe incluir AND user_id = %s."""
    src = _read(_DB_CHAT_PY)
    body = _extract_function(src, "delete_chat_session")
    m = re.search(
        r"DELETE\s+FROM\s+agent_sessions\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        body,
        re.IGNORECASE,
    )
    assert m is not None, (
        "El DELETE de `agent_sessions` debe incluir `AND user_id = %s` adicional al "
        "pre-check (cumple invariante I2 del repo). Si removiste el guard, "
        "el TOCTTOU teórico vuelve a abrirse."
    )


def test_router_maps_forbidden_to_403():
    """El handler debe traducir error_msg == 'forbidden' a HTTPException 403."""
    src = _read(_ROUTER_PY)
    body = _extract_endpoint_body(src, "delete", "/session/{session_id}")
    m = re.search(
        r'error_msg\s*==\s*[\"\']forbidden[\"\'].*?HTTPException\s*\(\s*status_code\s*=\s*403',
        body,
        re.DOTALL,
    )
    assert m is not None, (
        "El handler debe mapear `error_msg == 'forbidden'` a `HTTPException(403)`. "
        "Sin ese mapeo el cliente recibiría 500 para un intento IDOR — confundiría "
        "el modo del fallo y enmascararía intentos de abuso en logs."
    )


def test_router_maps_not_found_to_404():
    """El handler debe traducir error_msg == 'not_found' a HTTPException 404."""
    src = _read(_ROUTER_PY)
    body = _extract_endpoint_body(src, "delete", "/session/{session_id}")
    m = re.search(
        r'error_msg\s*==\s*[\"\']not_found[\"\'].*?HTTPException\s*\(\s*status_code\s*=\s*404',
        body,
        re.DOTALL,
    )
    assert m is not None, (
        "El handler debe mapear `error_msg == 'not_found'` a `HTTPException(404)`."
    )


def test_anchor_present_in_db_chat():
    src = _read(_DB_CHAT_PY)
    assert "P0-CHAT-DELETE-IDOR" in src, (
        "Falta anchor `P0-CHAT-DELETE-IDOR` en `db_chat.py`. Sin anchor, un "
        "reader del helper no sabrá de qué guard depende este código."
    )


def test_anchor_present_in_router():
    src = _read(_ROUTER_PY)
    assert "P0-CHAT-DELETE-IDOR" in src, (
        "Falta anchor `P0-CHAT-DELETE-IDOR` en `routers/chat.py`."
    )


# ---------------------------------------------------------------------------
# Tests funcionales (mockean get_session_owner para validar control flow)
# ---------------------------------------------------------------------------

@pytest.fixture
def import_helper():
    """Importa `delete_chat_session` desde db_chat."""
    from db_chat import delete_chat_session
    return delete_chat_session


def test_helper_rejects_cross_user_session(import_helper):
    """`get_session_owner` retorna ID distinto → helper retorna ('forbidden')."""
    with patch("db_chat.get_session_owner", return_value="victim-uuid-aaaa"):
        # execute_sql_write NO debe llamarse — el guard lo bloquea antes.
        with patch("db_chat.execute_sql_write") as mock_write:
            ok, err = import_helper("session-uuid-xxxx", "attacker-uuid-bbbb")
            assert ok is False
            assert err == "forbidden"
            mock_write.assert_not_called()


def test_helper_returns_not_found_when_session_missing(import_helper):
    """`get_session_owner` retorna None → helper retorna ('not_found')."""
    with patch("db_chat.get_session_owner", return_value=None):
        with patch("db_chat.execute_sql_write") as mock_write:
            ok, err = import_helper("session-uuid-xxxx", "user-uuid-aaaa")
            assert ok is False
            assert err == "not_found"
            mock_write.assert_not_called()


def test_helper_proceeds_when_owner_matches(import_helper):
    """`get_session_owner` retorna ID == caller → 3 DELETEs ejecutan."""
    user_id = "user-uuid-aaaa"
    with patch("db_chat.get_session_owner", return_value=user_id):
        with patch("db_chat.execute_sql_write") as mock_write:
            ok, err = import_helper("session-uuid-xxxx", user_id)
            assert ok is True
            assert err == ""
            # 3 DELETEs: conversation_summaries, agent_messages, agent_sessions.
            assert mock_write.call_count == 3
            # Último DELETE debe ser agent_sessions con (session_id, user_id).
            last_call = mock_write.call_args_list[-1]
            sql_str = last_call.args[0]
            params = last_call.args[1]
            assert "agent_sessions" in sql_str
            assert "user_id" in sql_str
            assert params == ("session-uuid-xxxx", user_id)
