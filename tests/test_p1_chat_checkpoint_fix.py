"""[P1-CHAT-CHECKPOINT-FIX · 2026-05-20] Force-rewrite del puerto del
`chat_checkpoint_pool` URL.

Bug observado en runtime el 2026-05-20:
    Tras P1-CHECKPOINT-POOL-SPLIT (mismo día), un chat productivo
    reportó el banner rojo "El asistente tuvo un problema". El user
    SÍ vio la respuesta completa del LLM, pero el `put_writes` final
    del `PostgresSaver` falló con:

        psycopg.OperationalError: sending query and params failed:
        SSL error: bad length
        SSL SYSCALL error: EOF detected
        File "langgraph/checkpoint/postgres/__init__.py", line 358,
        in put_writes -> cur.executemany(...)

    Causa raíz: el operator tenía `SUPABASE_DB_URL` con `:6543`
    hardcoded (caso común si copió "Transaction pooler" del dashboard
    de Supabase). Pre-fix, la lógica de `db_core.py` capturaba
    `original_session_url = clean_url` ANTES del rewrite a 6543, pero
    si el URL ya venía con 6543, `original_session_url` también
    quedaba con 6543 — el `chat_checkpoint_pool` se creaba contra
    Supavisor transaction mode, perpetuando el bug que el split
    pretendía cerrar.

Fix:
    Force-rewrite explícito a `:5432` para `original_session_url`
    cuando detectamos `:6543` en el URL Supabase. Garantiza que
    `chat_checkpoint_pool` SIEMPRE conecte en session mode,
    independientemente del valor del env var.

Este test enforza:
    1. La rama `clean_url.replace(":6543", ":5432")` existe asignada
       a `original_session_url`.
    2. La rama está gateada por `if ".supabase." in clean_url and
       ":6543" in clean_url`.
    3. La rama aparece ANTES del rewrite del pool principal a 6543
       (orden crítico: si el principal rewrite va primero, sobreescribe
       la variable que mi fix lee).
    4. El `else: original_session_url = clean_url` sigue presente
       para casos non-Supabase (local dev, otras DBs).
    5. El WARN legacy ("SUPABASE_DB_URL ya contiene :6543") fue
       removido — post-fix es inalcanzable, mantenerlo confunde.

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


def test_force_rewrite_branch_exists(db_core_src: str):
    """La asignación `original_session_url = clean_url.replace(":6543", ":5432")`
    DEBE existir. Es el SSOT del fix."""
    pattern = re.compile(
        r"""original_session_url\s*=\s*clean_url\.replace\(\s*["']:6543["']\s*,\s*["']:5432["']\s*\)"""
    )
    assert pattern.search(db_core_src), (
        "P1-CHAT-CHECKPOINT-FIX regresión: el force-rewrite "
        "`original_session_url = clean_url.replace(':6543', ':5432')` "
        "no se encuentra en db_core.py. Sin él, si SUPABASE_DB_URL viene "
        "con :6543, chat_checkpoint_pool cae a Supavisor transaction mode "
        "y el SSL bad length / EOF reaparece al `put_writes` final."
    )


def test_force_rewrite_gated_by_supabase_check(db_core_src: str):
    """El force-rewrite debe estar dentro de un `if ".supabase." in clean_url
    and ":6543" in clean_url:` — sin el guard, romperíamos URLs locales
    (postgres://localhost:6543 no es Supabase, no aplicar el rewrite)."""
    # Buscamos el bloque del if + replace adentro.
    block = re.search(
        r'if\s+["\']\.supabase\.["\']\s+in\s+clean_url\s+and\s+["\']:6543["\']\s+in\s+clean_url\s*:'
        r'.*?clean_url\.replace\(\s*["\']:6543["\']\s*,\s*["\']:5432["\']\s*\)',
        db_core_src,
        re.DOTALL,
    )
    assert block, (
        "P1-CHAT-CHECKPOINT-FIX regresión: el force-rewrite NO está gateado "
        "por `if '.supabase.' in clean_url and ':6543' in clean_url:`. Sin el "
        "guard, romperíamos URLs locales o non-Supabase. Reintroducir el "
        "guard exacto."
    )


def test_force_rewrite_before_main_rewrite(db_core_src: str):
    """El force-rewrite a `:5432` para `original_session_url` debe aparecer
    ANTES del rewrite `clean_url.replace(":5432", ":6543")` del pool
    principal. Si fuera al revés, `clean_url` ya estaría rewrited y el
    chequeo `:6543 in clean_url` siempre matchearía → el flujo se rompe."""
    lines = db_core_src.splitlines()
    force_lineno = next(
        (i for i, ln in enumerate(lines)
         if 'clean_url.replace(":6543", ":5432")' in ln
            or "clean_url.replace(':6543', ':5432')" in ln),
        None,
    )
    main_rewrite_lineno = next(
        (i for i, ln in enumerate(lines)
         if 'clean_url.replace(":5432", ":6543")' in ln
            or "clean_url.replace(':5432', ':6543')" in ln),
        None,
    )
    assert force_lineno is not None, (
        "P1-CHAT-CHECKPOINT-FIX regresión: force-rewrite a :5432 ausente."
    )
    assert main_rewrite_lineno is not None, (
        "Rewrite del pool principal a :6543 ausente — refactor inesperado."
    )
    assert force_lineno < main_rewrite_lineno, (
        f"P1-CHAT-CHECKPOINT-FIX regresión: force-rewrite (línea "
        f"{force_lineno + 1}) debe ir ANTES del main rewrite (línea "
        f"{main_rewrite_lineno + 1}). Order inverted → el guard "
        f"`:6543 in clean_url` siempre matchea tras el main rewrite y el "
        f"flujo se rompe."
    )


def test_else_branch_preserves_clean_url(db_core_src: str):
    """El `else: original_session_url = clean_url` debe seguir presente
    — cubre URLs non-Supabase (local dev con postgres://localhost:5432,
    otras DBs hosteadas)."""
    # Buscamos `original_session_url = clean_url` plain (NO la del replace).
    # Negative lookahead para excluir `.replace(...)` después.
    plain_assign = re.search(
        r"original_session_url\s*=\s*clean_url\s*(?:#[^\n]*)?\n",
        db_core_src,
    )
    assert plain_assign, (
        "P1-CHAT-CHECKPOINT-FIX regresión: el branch fallback "
        "`original_session_url = clean_url` plain ausente. Sin él, URLs "
        "non-Supabase (local dev, otras DBs) dejarían `original_session_url` "
        "indefinida → NameError al construir chat_checkpoint_pool."
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
    """Marker `P1-CHAT-CHECKPOINT-FIX` aparece ≥2× en db_core.py
    (al menos: docstring del bloque + comment del WARN removido)."""
    count = db_core_src.count("P1-CHAT-CHECKPOINT-FIX")
    assert count >= 2, (
        f"P1-CHAT-CHECKPOINT-FIX regresión: tooltip-anchor aparece "
        f"{count}× en db_core.py, esperado ≥2. Si un rename del slug "
        f"ocurrió, restaurar el marker en el bloque del force-rewrite."
    )
