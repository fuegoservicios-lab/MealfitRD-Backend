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
    `get_guest_chat_sessions` añade el predicado de ownership a la query, de
    modo que SOLO las sesiones genuinas de invitado (sin dueño) son
    recuperables por id crudo. Las sesiones propias de un usuario logueado las
    trae `get_user_chat_sessions(user_id)` filtrando por owner; una sesión
    creada como guest y luego reclamada queda con `user_id` no-nulo → sale por
    el path de owner, no por este.

    [P1-NEON-DB-MIGRATION · 2026-06-12] El transporte migró de PostgREST
    (`.in_("id", ...).is_("user_id", "null")`) a SQL directo via
    `execute_sql_query`. El predicado equivalente es:
        WHERE id = ANY(%s::uuid[]) AND user_id IS NULL
    La propiedad de seguridad verificada es LA MISMA: el filtro
    `user_id IS NULL` debe vivir en el SQL ejecutable de
    `get_guest_chat_sessions` (no en comentarios ni docstrings).

Contratos anclados:
    1. Parser (AST): los string-literals EJECUTABLES de
       `get_guest_chat_sessions` (docstring excluido, comentarios invisibles
       al AST) contienen el WHERE con `id = ANY(...)` Y `user_id IS NULL`
       conjugados con AND en la misma cláusula.
    2. Parser: anchor `P1-CHAT-GUEST-IDOR` presente en db_chat.py + cap
       `session_ids[:20]` en código ejecutable.
    3. Funcional: en runtime el SQL pasado a `execute_sql_query` contiene
       `user_id IS NULL` (mock-spy de db_chat.execute_sql_query lo verifica).
    4. Funcional: si la query se construye sin el filtro, el test falla.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CHAT_PY = _BACKEND_ROOT / "db_chat.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _function_node(src: str, name: str) -> ast.FunctionDef:
    """Localiza el FunctionDef via AST — los comentarios NO existen en el
    árbol, así que todo lo que extraigamos de acá es código ejecutable."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"No se encontró def `{name}` en el source.")


def _executable_string_literals(fn: ast.FunctionDef) -> str:
    """Concatena los string-literals del cuerpo de la función EXCLUYENDO el
    docstring (primera Expr-Constant-str). Así la prosa narrativa no puede
    satisfacer las assertions — solo strings que el código realmente usa
    (el SQL pasado a execute_sql_query)."""
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    parts: list[str] = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts.append(node.value)
    # Normaliza whitespace para que los regex no dependan del formateo.
    return " ".join(" ".join(parts).split())


# ---------------------------------------------------------------------------
# Parser-based (AST — código ejecutable, no comentarios)
# ---------------------------------------------------------------------------

def test_guest_sessions_query_filters_user_id_null():
    """El SQL de `get_guest_chat_sessions` DEBE incluir `user_id IS NULL`
    AND-conjugado con el filtro por ids en la MISMA cláusula WHERE."""
    src = _read(_DB_CHAT_PY)
    fn = _function_node(src, "get_guest_chat_sessions")
    sql_blob = _executable_string_literals(fn)
    assert re.search(r"id\s*=\s*ANY\(", sql_blob), (
        "Falta el filtro por ids (`WHERE id = ANY(%s::uuid[])`) en el SQL "
        "ejecutable de get_guest_chat_sessions."
    )
    assert re.search(
        r"id\s*=\s*ANY\(%s::uuid\[\]\)\s+AND\s+user_id\s+IS\s+NULL", sql_blob
    ), (
        "Falta `AND user_id IS NULL` junto al filtro por ids en el SQL "
        "ejecutable de `get_guest_chat_sessions`. Sin ese predicado, un "
        "usuario autenticado puede recuperar sesiones de OTRO usuario "
        "pasando `?session_ids=<sesión ajena>` (IDOR P1-CHAT-GUEST-IDOR). "
        "Lee la memoria antes de remover el filtro. "
        f"SQL ejecutable encontrado: {sql_blob!r}"
    )


def test_guest_sessions_caps_session_ids():
    """El cap `session_ids[:20]` (límite prudente del legacy PostgREST)
    sobrevive la migración a SQL directo — en código ejecutable."""
    src = _read(_DB_CHAT_PY)
    fn = _function_node(src, "get_guest_chat_sessions")
    code = ast.get_source_segment(src, fn) or ""
    # Solo líneas ejecutables: descarta comentarios full-line.
    code_only = "\n".join(
        ln for ln in code.splitlines() if not ln.strip().startswith("#")
    )
    assert re.search(r"session_ids\[:20\]", code_only), (
        "Falta el cap `session_ids[:20]` en get_guest_chat_sessions — sin "
        "límite, un cliente puede pasar miles de ids en un solo request."
    )


def test_anchor_present_in_db_chat():
    src = _read(_DB_CHAT_PY)
    assert "P1-CHAT-GUEST-IDOR" in src, (
        "Falta anchor `P1-CHAT-GUEST-IDOR` en db_chat.py."
    )


# ---------------------------------------------------------------------------
# Funcional: mock-spy de execute_sql_query (transporte SQL directo)
# ---------------------------------------------------------------------------

def test_guest_sessions_applies_user_id_null_filter_at_runtime(monkeypatch):
    """En runtime el SQL enviado a `execute_sql_query` DEBE contener
    `user_id IS NULL` — el filtro tiene que viajar en la query real, no
    solo existir en el source."""
    import db_chat

    captured: dict = {}

    def _spy_execute_sql_query(sql, params=None, *args, **kwargs):
        captured["sql"] = sql
        captured["params"] = params
        captured["fetch_all"] = kwargs.get("fetch_all")
        # Devuelve un resultado vacío para no entrar a _process_and_sort_sessions.
        return []

    # connection_pool truthy para pasar el guard `if not connection_pool`.
    monkeypatch.setattr(db_chat, "connection_pool", object())
    monkeypatch.setattr(db_chat, "execute_sql_query", _spy_execute_sql_query)

    out = db_chat.get_guest_chat_sessions(["sess-a", "sess-b"])

    assert out == []  # data vacía → sin sesiones
    assert "sql" in captured, "La query nunca se ejecutó."
    sql_norm = " ".join(str(captured["sql"]).split())
    assert "agent_sessions" in sql_norm, (
        f"La query no apunta a agent_sessions. SQL: {sql_norm!r}"
    )
    assert re.search(r"user_id\s+IS\s+NULL", sql_norm), (
        "La query de get_guest_chat_sessions NO incluye `user_id IS NULL` "
        "en runtime. Eso reabre el IDOR P1-CHAT-GUEST-IDOR: se devolverían "
        f"sesiones de cualquier dueño. SQL ejecutado: {sql_norm!r}"
    )
    # Los ids viajan como lista de strings (cast ::uuid[] lo hace el SQL).
    assert captured["params"] == (["sess-a", "sess-b"],), (
        f"Params inesperados: {captured['params']!r} — los session_ids deben "
        "viajar como única param list (str) para el `= ANY(%s::uuid[])`."
    )
    assert captured["fetch_all"] is True, (
        "get_guest_chat_sessions debe pedir fetch_all=True (lista de sesiones)."
    )
