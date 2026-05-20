"""[P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Tests anti-regresión del split
de pools entre el tráfico genérico (Transaction Pooler, 6543) y el
LangGraph `PostgresSaver` (Session mode, 5432).

Bug que cierra:
    LangGraph chat stream completaba la response al user pero al hacer
    `put_writes` final del checkpoint → `SSL error: bad length` /
    `SSL SYSCALL error: EOF detected`. Root cause: Transaction Pooler
    de Supabase (Supavisor 6543) mata conexiones idle agresivamente
    (~10-30s). Durante el chat-flow, el PostgresSaver mantiene la
    conexión abierta ~5-15s mientras espera el LLM call → Supavisor
    cierra → `put_writes` falla.

Fix:
    - `db_core.py` crea `chat_checkpoint_pool` separado usando el URL
      original (port 5432 session mode), sin el rewrite a 6543.
    - `agent.py` callsites de `PostgresSaver(connection_pool)` cambian
      a `PostgresSaver(_checkpoint_pool)` con fallback defensivo.

Tests parser-based — anchor literal en código para que un refactor
accidental restaure el bug.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE_PY = _BACKEND_ROOT / "db_core.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_APP_PY = _BACKEND_ROOT / "app.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# db_core.py: chat_checkpoint_pool existe + usa session URL
# ============================================================

def test_chat_checkpoint_pool_declared_module_level():
    """[P1-CHECKPOINT-POOL-SPLIT] `chat_checkpoint_pool` debe declararse
    a nivel de módulo (con `= None` default para garantizar importabilidad
    cuando el setup del pool falla)."""
    src = _read(_DB_CORE_PY)
    assert re.search(r"^chat_checkpoint_pool\s*=\s*None", src, re.MULTILINE), (
        "`chat_checkpoint_pool = None` ausente del top-level de db_core.py. "
        "Importadores no podrán hacer `from db import chat_checkpoint_pool` "
        "cuando el setup del pool falla — ImportError. Ver "
        "P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )


def test_chat_checkpoint_pool_uses_original_session_url():
    """[P1-CHECKPOINT-POOL-SPLIT] El `chat_checkpoint_pool` debe construirse
    con `original_session_url` (preservada ANTES del rewrite a 6543) — no
    con `clean_url` (que ya fue rewrited a Transaction Pooler).
    """
    src = _read(_DB_CORE_PY)
    # Anchor directo: `chat_checkpoint_pool = ConnectionPool(` seguido por
    # `conninfo=<var>` dentro del mismo bloque de paréntesis.
    pool_block = re.search(
        r"chat_checkpoint_pool\s*=\s*ConnectionPool\(\s*conninfo\s*=\s*(\w+)",
        src,
        re.DOTALL,
    )
    assert pool_block, (
        "`chat_checkpoint_pool = ConnectionPool(conninfo=...)` no encontrado "
        "— refactor inesperado del bloque del split."
    )
    conninfo_arg = pool_block.group(1)
    assert conninfo_arg == "original_session_url", (
        f"`chat_checkpoint_pool` usa `{conninfo_arg}` en lugar de "
        f"`original_session_url`. Si pasa `clean_url`, hereda el rewrite "
        f"a port 6543 y el bug del SSL bad length vuelve. Ver "
        f"P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )


def test_original_session_url_preserved_before_rewrite():
    """[P1-CHECKPOINT-POOL-SPLIT] `original_session_url` debe asignarse
    ANTES del `.replace(":5432", ":6543")` que el setup hace al
    `clean_url`. Si se asigna después, contendría el URL rewrited y
    el chat_checkpoint_pool quedaría con el bug."""
    src = _read(_DB_CORE_PY)
    # Buscamos el line number de cada asignación clave.
    lines = src.splitlines()

    assign_lineno = next(
        (i for i, ln in enumerate(lines) if re.search(r"original_session_url\s*=\s*clean_url\b", ln)),
        None,
    )
    rewrite_lineno = next(
        (i for i, ln in enumerate(lines) if 'replace(":5432", ":6543")' in ln),
        None,
    )
    assert assign_lineno is not None, "Asignación `original_session_url = clean_url` ausente."
    assert rewrite_lineno is not None, "Rewrite `:5432 → :6543` ausente — refactor del setup?"
    assert assign_lineno < rewrite_lineno, (
        f"`original_session_url` se asigna en línea {assign_lineno + 1} pero el "
        f"rewrite ocurre en {rewrite_lineno + 1}. Order inverted → "
        f"`original_session_url` contiene el URL post-rewrite, perdiendo la "
        f"intención del split. Ver P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )


# ============================================================
# agent.py: PostgresSaver usa el pool nuevo, no el general
# ============================================================

def test_agent_imports_chat_checkpoint_pool():
    """[P1-CHECKPOINT-POOL-SPLIT] El import desde `db` debe incluir
    `chat_checkpoint_pool` para que los callsites puedan referenciarlo."""
    src = _read(_AGENT_PY)
    assert "chat_checkpoint_pool" in src, (
        "agent.py NO importa `chat_checkpoint_pool` — los callsites de "
        "PostgresSaver caen al fallback `connection_pool` y el bug "
        "persiste. Ver P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )
    # Sanity: en la línea de `from db import ...`.
    assert re.search(
        r"from db import [^\n]*chat_checkpoint_pool",
        src,
    ), "chat_checkpoint_pool no aparece en el import explícito desde db."


def test_postgressaver_callsites_use_checkpoint_pool():
    """[P1-CHECKPOINT-POOL-SPLIT] TODOS los callsites de `PostgresSaver(...)`
    en agent.py deben usar `_checkpoint_pool` (con fallback a
    `connection_pool` si chat_checkpoint_pool es None) — NUNCA pasar
    `connection_pool` directo.

    Anchor: regex contra `PostgresSaver\\(connection_pool\\)` literal.
    """
    src = _read(_AGENT_PY)
    direct_callsites = re.findall(r"PostgresSaver\(\s*connection_pool\s*\)", src)
    assert not direct_callsites, (
        f"{len(direct_callsites)} callsites de `PostgresSaver(connection_pool)` "
        f"encontrados — debe usarse `PostgresSaver(_checkpoint_pool)` con "
        f"fallback `chat_checkpoint_pool or connection_pool`. Ver "
        f"P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )
    # Sanity positiva: al menos 2 callsites de _checkpoint_pool con fallback.
    fallback_pattern = re.findall(
        r"chat_checkpoint_pool\s+or\s+connection_pool",
        src,
    )
    assert len(fallback_pattern) >= 2, (
        f"Esperaba >=2 callsites con fallback `chat_checkpoint_pool or "
        f"connection_pool` (chat_with_agent + chat_with_agent_stream); "
        f"encontrados: {len(fallback_pattern)}. Refactor incompleto?"
    )


# ============================================================
# app.py: el pool se abre en startup (sin esto: PoolClosed runtime error)
# ============================================================

def test_app_startup_opens_chat_checkpoint_pool():
    """[P1-CHECKPOINT-POOL-SPLIT] `app.py` debe (a) importar
    `chat_checkpoint_pool` desde `db` y (b) llamar `.open()` en el lifespan
    de startup junto a `connection_pool.open()`.

    Pre-fix: el pool quedaba con `open=False` (default del setup en db_core)
    y el primer `chat_graph_app.get_state(...)` fallaba con
    `psycopg_pool.PoolClosed: the pool 'pool-3' is not open yet`. Bug
    observado en runtime tras aplicar el split en la primera iteración."""
    src = _read(_APP_PY)
    assert "chat_checkpoint_pool" in src, (
        "app.py NO referencia `chat_checkpoint_pool` — ni import ni .open() "
        "ni close. El pool quedará cerrado y el chat fallará con PoolClosed. "
        "Ver P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )
    # Sanity import.
    assert re.search(
        r"from db import\s*\([^)]*chat_checkpoint_pool",
        src,
        re.DOTALL,
    ), "Import explícito de chat_checkpoint_pool ausente."
    # Anchor del open() en startup — debe estar dentro de un `if chat_checkpoint_pool:`
    # guard porque el pool es None cuando el setup en db_core falla.
    open_pattern = re.search(
        r"if\s+chat_checkpoint_pool\s*:\s*\n\s+chat_checkpoint_pool\.open\(\)",
        src,
    )
    assert open_pattern, (
        "`if chat_checkpoint_pool: chat_checkpoint_pool.open()` ausente del "
        "startup. Sin esto, PoolClosed runtime error en el primer chat. Ver "
        "P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
    )
