"""[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Cierre de DB-P1 + DB-P2 del audit
Supabase MCP del Agente (2026-05-19) post P0+P1+P2+P3 bundles de código.

VECTORES CERRADOS

  - **DB-P1**: agent_messages / conversation_summaries solo tenían
    `session_id` (nullable). Ownership requería siempre join contra
    `agent_sessions` — frágil ante cualquier callsite futuro que omita
    el join (cron de cleanup, query ad-hoc, refactor que asume `user_id`).

  - **DB-P2**: RLS estaba enabled+forced pero la ÚNICA policy era
    `DELETE service_role`. Sin SELECT/INSERT/UPDATE policies, un futuro
    callsite frontend que intentara leer directo con anon/authenticated
    key recibiría 0 rows silenciosamente.

MIGRACIÓN SSOT

  Aplicada vía Supabase MCP a producción (project_id=mpoodlmnzaeuuazsazbj)
  el 2026-05-19. Archivo del repo:
  `supabase/migrations/db_p1_chat_user_id_rls_2026_05_19.sql`.

  Cambios:
    1. ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
       en ambas tablas (nullable — guests legítimos).
    2. Backfill UPDATE FROM agent_sessions.user_id (122 mensajes + 2
       summaries afectados, todos quedan NULL porque los 4 sessions son
       guests/tests — esperado).
    3. 3 índices: idx_agent_messages_user_id (partial NOT NULL),
       idx_conversation_summaries_user_id (partial NOT NULL),
       idx_agent_messages_session_id_created_at (compuesto).
    4. 6 RLS policies (SELECT/INSERT/UPDATE × 2 tablas) basadas en
       auth.uid() = user_id. DELETE legacy preservado (solo service_role).
    5. COMMENTs en columnas + índices.
    6. Sanity DO $$ post-apply.

COORDINACIÓN BACKEND

  - `save_message(session_id, role, content, user_id=None)` — `user_id`
    opcional para preservar backward compat. Si None, hace lookup defensivo
    via `get_session_owner(session_id)`.
  - `_save_message_insert_with_retry(session_id, role, content, user_id)`
    persiste el user_id en la columna nueva.
  - `_resolve_user_id_for_db(user_id_input, session_id)` helper en
    `routers/chat.py` normaliza "guest"/None/session_id_placeholder → None.
  - Los 3 callsites principales en routers/chat.py (/message, /stream
    user+model, /api/chat root user+model) pasan user_id explícito post-IDOR.

Este test valida:
  - **Migración SSOT** (archivo en repo) contiene todos los cambios
    esperados con la idempotencia obligatoria (P3-MIGRATION-IDEMPOTENCE-DOC).
  - **Backend coordination** está completa: signature de `save_message`
    + helper + helper `_resolve_user_id_for_db` + 3 callsites updateados.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_FP = (
    _REPO_ROOT / "supabase" / "migrations" / "db_p1_chat_user_id_rls_2026_05_19.sql"
)
_DB_CHAT_FP = _REPO_ROOT / "backend" / "db_chat.py"
_CHAT_ROUTER_FP = _REPO_ROOT / "backend" / "routers" / "chat.py"


@pytest.fixture(scope="module")
def migration_sql() -> str:
    return _MIGRATION_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def db_chat_src() -> str:
    return _DB_CHAT_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_router_src() -> str:
    return _CHAT_ROUTER_FP.read_text(encoding="utf-8")


# ===========================================================================
# Migración SSOT
# ===========================================================================

def test_migration_file_exists() -> None:
    """[P1-CHAT-DB-USER-ID-RLS] el archivo de migración debe existir en
    `supabase/migrations/`. SSOT obligatorio (CLAUDE.md: 'DDL en runtime:
    prohibido')."""
    assert _MIGRATION_FP.exists(), (
        f"[P1-CHAT-DB-USER-ID-RLS] falta {_MIGRATION_FP}. La migración "
        f"debe estar en `supabase/migrations/` para ser SSOT."
    )


def test_migration_adds_user_id_to_both_tables(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] ambas tablas reciben la columna user_id
    con FK a auth.users + ON DELETE CASCADE (GDPR account-deletion)."""
    for table in ("agent_messages", "conversation_summaries"):
        pattern = re.compile(
            rf"ALTER TABLE public\.{table}\s+"
            rf"ADD COLUMN IF NOT EXISTS user_id UUID\s+"
            rf"REFERENCES auth\.users\(id\) ON DELETE CASCADE",
            re.MULTILINE,
        )
        assert pattern.search(migration_sql), (
            f"[P1-CHAT-DB-USER-ID-RLS] falta ADD COLUMN user_id en "
            f"{table}. Debe ser `UUID REFERENCES auth.users(id) ON DELETE "
            f"CASCADE` para GDPR cleanup automático."
        )


def test_migration_idempotent_patterns(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] idempotencia obligatoria
    (P3-MIGRATION-IDEMPOTENCE-DOC): IF NOT EXISTS en ADD COLUMN +
    CREATE INDEX, DROP POLICY IF EXISTS antes de CREATE POLICY,
    sanity DO $$ RAISE EXCEPTION al final."""
    # ADD COLUMN IF NOT EXISTS × 2 tablas
    assert migration_sql.count("ADD COLUMN IF NOT EXISTS user_id") == 2, (
        "[P1-CHAT-DB-USER-ID-RLS] se requieren 2 `ADD COLUMN IF NOT EXISTS user_id` "
        "(uno por tabla). Sin IF NOT EXISTS, re-aplicación falla."
    )
    # CREATE INDEX IF NOT EXISTS × 3 índices (matchear callsites reales,
    # NO menciones en comments del header). Usamos lookahead para nombre.
    idx_callsites = len(
        re.findall(
            r"CREATE INDEX IF NOT EXISTS\s+\w+\s+ON\s+public\.",
            migration_sql,
        )
    )
    assert idx_callsites == 3, (
        f"[P1-CHAT-DB-USER-ID-RLS] esperaba 3 callsites reales de "
        f"`CREATE INDEX IF NOT EXISTS <name> ON public.<table>` "
        f"(idx_agent_messages_user_id + idx_conversation_summaries_user_id "
        f"+ idx_agent_messages_session_id_created_at). Encontrados: {idx_callsites}."
    )
    # DROP POLICY IF EXISTS × 6 — matchear callsites reales (con
    # `ON public.<table>` para excluir menciones en comments).
    drop_callsites = len(
        re.findall(
            r"DROP POLICY IF EXISTS\s+\w+\s+ON\s+public\.",
            migration_sql,
        )
    )
    assert drop_callsites == 6, (
        f"[P1-CHAT-DB-USER-ID-RLS] esperaba 6 callsites reales de "
        f"`DROP POLICY IF EXISTS <name> ON public.<table>` "
        f"(SELECT/INSERT/UPDATE × 2 tablas). Encontrados: {drop_callsites}."
    )
    # Sanity DO $$ con RAISE EXCEPTION
    assert "DO $$" in migration_sql, (
        "[P1-CHAT-DB-USER-ID-RLS] falta el sanity check `DO $$ ... END $$`. "
        "Convención P3-MIGRATION-IDEMPOTENCE-DOC."
    )
    assert "RAISE EXCEPTION" in migration_sql, (
        "[P1-CHAT-DB-USER-ID-RLS] el sanity check debe usar RAISE EXCEPTION "
        "para fail-loud post-apply si algún ADD/CREATE falló silente."
    )


def test_migration_backfill_present(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] backfill UPDATE FROM agent_sessions debe
    estar en la migración. Sin el backfill, todos los mensajes existentes
    quedan con user_id=NULL incluso si su session.user_id no era NULL."""
    for table in ("agent_messages", "conversation_summaries"):
        pattern = re.compile(
            rf"UPDATE public\.{table}.*?\n"
            rf".*?SET user_id = ases\.user_id.*?\n"
            rf".*?FROM public\.agent_sessions ases.*?\n"
            rf".*?WHERE.*?session_id = ases\.id.*?\n"
            rf".*?user_id IS NULL",
            re.DOTALL,
        )
        assert pattern.search(migration_sql), (
            f"[P1-CHAT-DB-USER-ID-RLS] falta el backfill UPDATE FROM "
            f"para {table}. Pattern: UPDATE...SET user_id = ases.user_id "
            f"FROM agent_sessions WHERE session_id = ases.id AND user_id IS NULL."
        )


def test_migration_creates_6_rls_policies(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] 6 policies nuevas: SELECT/INSERT/UPDATE
    × 2 tablas. TO authenticated. USING/WITH CHECK basadas en
    `auth.uid() = user_id`. NO INSERT policy para anon (no escribimos
    chat sin auth desde el cliente)."""
    expected_policies = [
        ("agent_messages", "authenticated_select_own_messages", "SELECT"),
        ("agent_messages", "authenticated_insert_own_messages", "INSERT"),
        ("agent_messages", "authenticated_update_own_messages", "UPDATE"),
        ("conversation_summaries", "authenticated_select_own_summaries", "SELECT"),
        ("conversation_summaries", "authenticated_insert_own_summaries", "INSERT"),
        ("conversation_summaries", "authenticated_update_own_summaries", "UPDATE"),
    ]
    for table, policy_name, cmd in expected_policies:
        assert f"CREATE POLICY {policy_name}" in migration_sql, (
            f"[P1-CHAT-DB-USER-ID-RLS] falta `CREATE POLICY {policy_name}` "
            f"sobre {table}."
        )
        assert f"ON public.{table}" in migration_sql, (
            f"[P1-CHAT-DB-USER-ID-RLS] policy {policy_name} debe ser "
            f"`ON public.{table}`."
        )
    # auth.uid() debe aparecer al menos 6 veces (USING + WITH CHECK).
    auth_uid_count = migration_sql.count("auth.uid()")
    assert auth_uid_count >= 6, (
        f"[P1-CHAT-DB-USER-ID-RLS] esperaba ≥6 referencias a `auth.uid()` "
        f"en las policies (USING/WITH CHECK). Encontradas: {auth_uid_count}."
    )


def test_migration_composite_index_for_ordered_lookup(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] índice compuesto `(session_id,
    created_at DESC)` para la query principal "mensajes ordenados por
    sesión". Pre-migración había solo `(session_id)` simple + sort en
    memoria — OK con 122 rows, degrada con escala."""
    assert "idx_agent_messages_session_id_created_at" in migration_sql
    assert re.search(
        r"ON public\.agent_messages\s*\(session_id,\s*created_at DESC\)",
        migration_sql,
    ), (
        "[P1-CHAT-DB-USER-ID-RLS] el índice compuesto debe ser "
        "`(session_id, created_at DESC)`. Si está sin DESC o invertido, "
        "no sirve para la query ORDER BY created_at ASC con backward scan."
    )


def test_migration_partial_indexes_exclude_guests(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] índices en user_id son `WHERE user_id IS
    NOT NULL` (partial) para evitar indexar las filas de guest (~30-50%
    reducción de tamaño según el ratio guest:authenticated)."""
    for idx_name in ("idx_agent_messages_user_id", "idx_conversation_summaries_user_id"):
        pattern = re.compile(
            rf"CREATE INDEX IF NOT EXISTS {idx_name}.*?WHERE user_id IS NOT NULL",
            re.DOTALL,
        )
        assert pattern.search(migration_sql), (
            f"[P1-CHAT-DB-USER-ID-RLS] {idx_name} debe ser partial con "
            f"`WHERE user_id IS NOT NULL`."
        )


def test_migration_has_column_comments(migration_sql: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] COMMENT ON COLUMN para documentar la
    semántica (nullable para guests, FK CASCADE GDPR, backfill source)."""
    for table in ("agent_messages", "conversation_summaries"):
        assert f"COMMENT ON COLUMN public.{table}.user_id" in migration_sql, (
            f"[P1-CHAT-DB-USER-ID-RLS] falta `COMMENT ON COLUMN "
            f"public.{table}.user_id`. Convención del repo para SRE + "
            f"advisor context."
        )


# ===========================================================================
# Coordinación backend — db_chat.py
# ===========================================================================

def test_save_message_signature_accepts_user_id(db_chat_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] `save_message` ahora acepta `user_id:
    Optional[str] = None`. Mantiene backward compat — callsites legacy
    no necesitan cambiar."""
    pattern = re.compile(
        r"def save_message\s*\(\s*\n?\s*session_id:\s*str\s*,"
        r"\s*\n?\s*role:\s*str\s*,"
        r"\s*\n?\s*content:\s*str\s*,"
        r"\s*\n?\s*user_id:\s*Optional\[str\]\s*=\s*None\s*,?",
        re.MULTILINE,
    )
    assert pattern.search(db_chat_src), (
        "[P1-CHAT-DB-USER-ID-RLS] signature canónica: "
        "`save_message(session_id: str, role: str, content: str, "
        "user_id: Optional[str] = None)`. Si cambias el orden o el "
        "default, callsites legacy se rompen."
    )


def test_save_message_fallback_lookup_when_user_id_none(db_chat_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] cuando `user_id is None`, `save_message`
    hace lookup via `get_session_owner(session_id)` para preservar
    backward compat con ~10 callsites legacy que no lo pasan."""
    fn_idx = db_chat_src.find("def save_message(")
    assert fn_idx >= 0
    next_def = re.search(r"\ndef\s", db_chat_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(db_chat_src)
    body = db_chat_src[fn_idx:end]
    # Patrón canónico: `if user_id is None: user_id = get_session_owner(...)`.
    assert "user_id is None" in body, (
        "[P1-CHAT-DB-USER-ID-RLS] `save_message` debe tener un branch "
        "`if user_id is None: ... get_session_owner(session_id)` para "
        "fallback. Sin él, callsites legacy persisten user_id=None aunque "
        "la session tenga owner conocido."
    )
    assert "get_session_owner(session_id)" in body


def test_insert_helper_persists_user_id(db_chat_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] el helper retry-able incluye `user_id`
    en el dict del INSERT a `agent_messages`."""
    helper_idx = db_chat_src.find("def _save_message_insert_with_retry(")
    assert helper_idx >= 0
    next_def = re.search(r"\ndef\s", db_chat_src[helper_idx + 10:])
    end = (helper_idx + 10 + next_def.start()) if next_def else len(db_chat_src)
    body = db_chat_src[helper_idx:end]
    # El INSERT debe incluir `"user_id": user_id`.
    assert '"user_id": user_id' in body, (
        "[P1-CHAT-DB-USER-ID-RLS] `_save_message_insert_with_retry` debe "
        "incluir `\"user_id\": user_id` en el dict del INSERT. Sin esto, "
        "la columna nueva queda NULL incluso cuando el caller pasa user_id."
    )


# ===========================================================================
# Coordinación backend — routers/chat.py
# ===========================================================================

def test_resolve_user_id_helper_defined(chat_router_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] helper `_resolve_user_id_for_db` existe
    para normalizar "guest"/None/session_id_placeholder → None."""
    assert "def _resolve_user_id_for_db(" in chat_router_src, (
        "[P1-CHAT-DB-USER-ID-RLS] falta helper `_resolve_user_id_for_db(...)` "
        "que normalice user_id_input → user_id_for_db para los 3 callsites."
    )


def test_resolve_user_id_helper_returns_none_for_guests(chat_router_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] el helper debe retornar None para 3 cases:
    None/""/`"guest"` y cuando user_id_input == session_id."""
    fn_idx = chat_router_src.find("def _resolve_user_id_for_db(")
    assert fn_idx >= 0
    # Body hasta la siguiente `def ` top-level.
    next_def = re.search(r"\ndef\s", chat_router_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(chat_router_src)
    body = chat_router_src[fn_idx:end]
    # Las 3 ramas de guest detection deben estar presentes.
    assert 'user_id_input == "guest"' in body
    assert "user_id_input == session_id" in body
    # Al menos 3 `return None`.
    return_none_count = body.count("return None")
    assert return_none_count >= 3, (
        f"[P1-CHAT-DB-USER-ID-RLS] el helper debe `return None` en al menos "
        f"3 ramas (None/empty, 'guest', user_id==session_id). Encontradas: "
        f"{return_none_count}."
    )


def test_callsites_pass_user_id_explicit(chat_router_src: str) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] los 3 callsites principales de
    save_message en routers/chat.py pasan `user_id=` explícito vía
    `_resolve_user_id_for_db(...)` (o usando la var `_db_user_id`
    pre-resuelta en closure scope)."""
    # Cuento callsites con user_id= explícito.
    # Pattern: save_message(...) seguido eventualmente por user_id=
    callsites_with_user_id = len(
        re.findall(
            r"save_message\(\s*\n?\s*session_id[^)]*?user_id\s*=",
            chat_router_src,
            re.DOTALL,
        )
    )
    # /message + /stream user + /stream model + /api/chat user + /api/chat model = 5
    # Aceptamos ≥4 (uno podría usar _resolve directamente sin _db_user_id var).
    assert callsites_with_user_id >= 4, (
        f"[P1-CHAT-DB-USER-ID-RLS] esperaba ≥4 callsites con `user_id=` "
        f"explícito en routers/chat.py (/message, /stream user+model, "
        f"/api/chat user+model). Encontrados: {callsites_with_user_id}."
    )


# ===========================================================================
# Tooltip-anchor preservado
# ===========================================================================

def test_tooltip_anchor_present_across_files(
    migration_sql: str, db_chat_src: str, chat_router_src: str
) -> None:
    """[P1-CHAT-DB-USER-ID-RLS] marker textual presente en migración +
    backend coordinator. Cross-link con
    test_p2_hist_audit_14_marker_test_link.py."""
    assert migration_sql.count("P1-CHAT-DB-USER-ID-RLS") >= 5, (
        "marker en migración SQL (header + comments en columns/indexes + "
        "sanity RAISE EXCEPTION)."
    )
    assert db_chat_src.count("P1-CHAT-DB-USER-ID-RLS") >= 2, (
        "marker en db_chat.py (helper docstrings + save_message)."
    )
    assert chat_router_src.count("P1-CHAT-DB-USER-ID-RLS") >= 3, (
        "marker en routers/chat.py (_resolve_user_id_for_db + 3 callsites)."
    )
