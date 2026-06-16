"""[P1-CHAT-CHECKPOINT-FIX · 2026-05-20 · actualizado P1-NEON-DB-MIGRATION
2026-06-12] El `chat_checkpoint_pool` DEBE conectar en session mode contra el
endpoint DIRECTO, separado del pool principal (pooler/transaction mode).

Bug histórico observado en runtime el 2026-05-20 (era Supabase):
    Tras P1-CHECKPOINT-POOL-SPLIT (mismo día), un chat productivo
    reportó el banner rojo "El asistente tuvo un problema". El user
    SÍ vio la respuesta completa del LLM, pero el `put_writes` final
    del `PostgresSaver` falló con:

        psycopg.OperationalError: sending query and params failed:
        SSL error: bad length
        SSL SYSCALL error: EOF detected

    Causa raíz: el `chat_checkpoint_pool` se creaba contra el
    transaction-pooler (puerto :6543 de Supavisor), que mata conexiones
    idle agresivamente mid-stream. El fix Supabase-era forzaba un
    rewrite `:6543`→`:5432` para que el checkpointer usara session mode.

Estado actual (post-migración a Neon, 2026-06-12):
    Supabase fue eliminado por completo. Neon provee DOS URLs separados
    (no hace falta el rewrite de puerto):
      - `NEON_DATABASE_URL_POOLED` → `clean_url` → pools principales
        (PgBouncer transaction mode).
      - `NEON_DATABASE_URL` (endpoint directo, session mode) →
        `original_session_url` → `chat_checkpoint_pool`.
    El INVARIANTE que protegía el fix se preserva: el checkpointer usa
    un URL session-mode DISTINTO del URL pooled de los pools principales.
    El rewrite `:6543`→`:5432` y el guard `".supabase." in clean_url`
    ya NO existen — fueron reemplazados por la separación de URLs de Neon.

Este test enforza (post-Neon):
    1. `original_session_url` se deriva de `NEON_DATABASE_URL` (directo).
    2. `clean_url` se deriva de `NEON_DATABASE_URL_POOLED` (pooler) — son
       fuentes DISTINTAS (el checkpointer no comparte el URL pooled).
    3. `chat_checkpoint_pool` se construye con `conninfo=original_session_url`
       (session mode separado).
    4. El WARN legacy ("SUPABASE_DB_URL ya contiene :6543") sigue removido.

Cross-link convention (P2-HIST-AUDIT-14): slug `p1_chat_checkpoint_fix`
matchea este archivo `test_p1_chat_checkpoint_fix.py`.

Tooltip-anchor: P1-CHAT-CHECKPOINT-FIX.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE_PY = _BACKEND_ROOT / "db_core.py"


@pytest.fixture(scope="module")
def db_core_src() -> str:
    return _DB_CORE_PY.read_text(encoding="utf-8")


def test_session_url_derives_from_neon_direct(db_core_src: str):
    """[P1-NEON-DB-MIGRATION] `original_session_url` DEBE derivar de
    `NEON_DATABASE_URL` (endpoint directo, session mode). Es la fuente que
    garantiza que el checkpointer NO use el transaction-pooler."""
    pattern = re.compile(
        r"original_session_url\s*=\s*NEON_DATABASE_URL\b"
    )
    assert pattern.search(db_core_src), (
        "P1-CHAT-CHECKPOINT-FIX regresión: `original_session_url` ya no se "
        "deriva de `NEON_DATABASE_URL` (endpoint directo). Sin el endpoint "
        "directo session-mode, chat_checkpoint_pool caería al pooler "
        "(transaction mode) y el SSL bad length / EOF reaparece al "
        "`put_writes` final."
    )


def test_session_url_distinct_from_pooled_url(db_core_src: str):
    """[P1-NEON-DB-MIGRATION] Las dos URLs deben venir de fuentes DISTINTAS:
    `clean_url` (pools principales) del POOLED, `original_session_url`
    (checkpointer) del DIRECTO. El antiguo guard `".supabase." in clean_url`
    + rewrite `:6543`→`:5432` fue reemplazado por la separación de URLs de
    Neon — el checkpointer no comparte el URL pooled."""
    clean_from_pooled = re.search(
        r"clean_url\s*=\s*NEON_DATABASE_URL_POOLED\b", db_core_src
    )
    session_from_direct = re.search(
        r"original_session_url\s*=\s*NEON_DATABASE_URL\b", db_core_src
    )
    assert clean_from_pooled, (
        "P1-CHAT-CHECKPOINT-FIX regresión: `clean_url` ya no se deriva de "
        "`NEON_DATABASE_URL_POOLED` (pooler). Refactor inesperado."
    )
    assert session_from_direct, (
        "P1-CHAT-CHECKPOINT-FIX regresión: `original_session_url` ya no se "
        "deriva de `NEON_DATABASE_URL` (directo). Refactor inesperado."
    )
    # Y el rewrite legacy de puerto NO debe reaparecer (Neon no lo necesita).
    legacy_rewrite = re.search(
        r"""clean_url\.replace\(\s*["']:6543["']\s*,\s*["']:5432["']\s*\)""",
        db_core_src,
    )
    assert not legacy_rewrite, (
        "P1-CHAT-CHECKPOINT-FIX regresión: reapareció el rewrite legacy "
        "`clean_url.replace(':6543', ':5432')`. Post-Neon NO debe existir — "
        "Neon usa URLs separados (pooled vs direct), no rewrite de puerto."
    )


def test_checkpoint_pool_uses_session_url(db_core_src: str):
    """El `chat_checkpoint_pool` DEBE construirse con
    `conninfo=original_session_url` (session mode, separado del pool
    principal que usa `clean_url`/pooler). Es el SSOT del split."""
    pattern = re.compile(
        r"chat_checkpoint_pool\s*=\s*ConnectionPool\(\s*conninfo\s*=\s*original_session_url",
        re.DOTALL,
    )
    assert pattern.search(db_core_src), (
        "P1-CHAT-CHECKPOINT-FIX regresión: `chat_checkpoint_pool` ya no usa "
        "`conninfo=original_session_url`. Si usa `clean_url` (pooler), el "
        "checkpointer vuelve a transaction mode y el SSL bad length / EOF "
        "reaparece al `put_writes` final."
    )


def test_obsolete_warn_removed(db_core_src: str):
    """El WARN legacy del split (`logger.warning("⚠️ [P1-CHECKPOINT-POOL-SPLIT]
    SUPABASE_DB_URL ya contiene...")`) fue removido — post-fix es
    inalcanzable (la nueva lógica garantiza `original_session_url !=
    clean_url` cuando entra esa rama). Mantenerlo confunde al operator
    (WARN nunca dispara, queda como code rot).

    Test específico: busca `logger.warning(...)` con el legacy marker
    `[P1-CHECKPOINT-POOL-SPLIT]` Y el string `SUPABASE_DB_URL`. Menciones
    en comments están OK (documentan el cierre)."""
    # Pattern intencionalmente específico: logger.warning + marker legacy.
    legacy_warn_pattern = re.compile(
        r"""logger\.warning\(\s*['"f][^)]*P1-CHECKPOINT-POOL-SPLIT[^)]*SUPABASE_DB_URL""",
        re.DOTALL,
    )
    assert not legacy_warn_pattern.search(db_core_src), (
        "P1-CHAT-CHECKPOINT-FIX regresión: el `logger.warning(...)` con "
        "el marker [P1-CHECKPOINT-POOL-SPLIT] sobre 'SUPABASE_DB_URL ya "
        "contiene :6543' sigue activo. Post-fix es inalcanzable — "
        "removerlo evita confundir al SRE con un WARN que NUNCA dispara."
    )


def test_tooltip_anchor_present(db_core_src: str):
    """Marker `P1-CHAT-CHECKPOINT-FIX` aparece ≥1× en db_core.py.

    Post-migración a Neon (P1-NEON-DB-MIGRATION) el bloque de force-rewrite
    Supabase-era se eliminó; el marker queda en el comment del bloque del
    checkpoint pool (1×). Es el tripwire contra rename del slug."""
    count = db_core_src.count("P1-CHAT-CHECKPOINT-FIX")
    assert count >= 1, (
        f"P1-CHAT-CHECKPOINT-FIX regresión: tooltip-anchor aparece "
        f"{count}× en db_core.py, esperado ≥1. Si un rename del slug "
        f"ocurrió, restaurar el marker en el bloque del checkpoint pool."
    )
