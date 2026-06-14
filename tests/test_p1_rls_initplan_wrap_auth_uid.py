"""[P1-RLS-INITPLAN · 2026-05-20] Regression guard parser-based.

Garantiza que la migración SSOT `p1_rls_initplan_wrap_auth_uid_2026_05_20.sql`
escriba las 10 policies user-facing (water_intake_log + agent_messages +
conversation_summaries) envolviendo `auth.uid()` con `(select auth.uid())`.

Si un futuro DROP + CREATE de cualquiera de estas policies olvida el wrap,
el advisor Supabase `auth_rls_initplan` volverá a reportar WARN y degradará
performance. Este test bloquea esa regresión en CI antes de tocar la DB.

NO valida la DB live (eso es responsabilidad de la propia migración SSOT
con su DO $$ RAISE sanity check). Parsea source-de-prod.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATION_FILES = (
    _REPO_ROOT / "migrations" / "p1_rls_initplan_wrap_auth_uid_2026_05_20.sql",
    _REPO_ROOT / "backend" / "migrations" / "p1_rls_initplan_wrap_auth_uid_2026_05_20.sql",
)

# (table, policy_name) — las 10 user-facing policies envueltas.
# service_role_delete_* NO incluidas (qual = true, no usan auth.uid()).
_POLICIES = (
    ("water_intake_log", "water_intake_log_select_own"),
    ("water_intake_log", "water_intake_log_insert_own"),
    ("water_intake_log", "water_intake_log_update_own"),
    ("water_intake_log", "water_intake_log_delete_own"),
    ("agent_messages", "authenticated_select_own_messages"),
    ("agent_messages", "authenticated_insert_own_messages"),
    ("agent_messages", "authenticated_update_own_messages"),
    ("conversation_summaries", "authenticated_select_own_summaries"),
    ("conversation_summaries", "authenticated_insert_own_summaries"),
    ("conversation_summaries", "authenticated_update_own_summaries"),
)


@pytest.mark.parametrize("migration_path", _MIGRATION_FILES)
def test_migration_files_exist_in_both_dirs(migration_path: Path) -> None:
    """SSOT contract: la migración vive en workspace-root Y backend dir."""
    assert migration_path.exists(), (
        f"Migración SSOT esperada en {migration_path}. "
        "Convención P3-MIGRATIONS-SSOT exige presencia en ambos dirs."
    )


def test_workspace_and_backend_migrations_identical() -> None:
    """SSOT contract: ambos archivos byte-idénticos (sin drift)."""
    a = _MIGRATION_FILES[0].read_text(encoding="utf-8")
    b = _MIGRATION_FILES[1].read_text(encoding="utf-8")
    assert a == b, (
        "Drift entre migrations/ y backend/migrations/ — "
        "actualizar ambos en el mismo commit (P3-MIGRATIONS-SSOT)."
    )


@pytest.mark.parametrize("table,policy", _POLICIES)
def test_policy_uses_select_auth_uid_wrap(table: str, policy: str) -> None:
    """Cada policy debe envolver auth.uid() con (select auth.uid())."""
    text = _MIGRATION_FILES[0].read_text(encoding="utf-8")

    # Encontrar el bloque CREATE POLICY <policy> ON <schema>.<table> ... ;
    pattern = re.compile(
        r"CREATE\s+POLICY\s+" + re.escape(policy)
        + r"\s+ON\s+public\." + re.escape(table)
        + r"(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, (
        f"CREATE POLICY {policy} ON public.{table} no encontrado en migración."
    )

    body = match.group(1)
    # El bloque debe contener el wrap `(select auth.uid())` —
    # case-insensitive, tolerante a whitespace.
    wrap_pattern = re.compile(r"\(\s*select\s+auth\.uid\(\)\s*\)", re.IGNORECASE)
    assert wrap_pattern.search(body), (
        f"Policy {policy} en {table} NO usa el wrap `(select auth.uid())`. "
        f"Bloque encontrado:\n{body}\n\n"
        "Sin el wrap, postgres re-evalúa auth.uid() por cada row (advisor "
        "auth_rls_initplan WARN). Ver "
        "https://supabase.com/docs/guides/database/postgres/row-level-security#call-functions-with-select"
    )


def test_no_unwrapped_auth_uid_in_policy_blocks() -> None:
    """Defensa: ningún CREATE POLICY del archivo usa `auth.uid()` desnudo
    (sin wrap select). Atrapa typos como `USING (auth.uid() = user_id)`."""
    text = _MIGRATION_FILES[0].read_text(encoding="utf-8")

    # Buscar TODOS los bloques CREATE POLICY ... ; del archivo
    policy_blocks = re.findall(
        r"CREATE\s+POLICY\s+\S+\s+ON\s+public\.\S+.*?;",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    assert len(policy_blocks) == 10, (
        f"Esperadas 10 CREATE POLICY, encontradas {len(policy_blocks)}. "
        "Si añadiste/removiste policies, actualizar este test."
    )

    # Para cada bloque, contar `auth.uid()` matches que NO estén dentro de
    # un `select ... auth.uid()`. Si quedó alguno desnudo → fallo.
    bare_pattern = re.compile(
        r"(?<!select\s)(?<!select)auth\.uid\(\)",
        re.IGNORECASE,
    )
    for block in policy_blocks:
        # Sanitize: remove "(select auth.uid())" substrings y luego buscar bare.
        cleaned = re.sub(
            r"\(\s*select\s+auth\.uid\(\)\s*\)",
            "<WRAPPED>",
            block,
            flags=re.IGNORECASE,
        )
        bare_matches = bare_pattern.findall(cleaned)
        assert not bare_matches, (
            f"Bloque CREATE POLICY contiene `auth.uid()` desnudo (sin wrap):\n"
            f"{block}\n\n"
            "Cada uso DEBE estar dentro de `(select auth.uid())`."
        )


def test_migration_contains_do_raise_sanity() -> None:
    """Defensa runtime: la migración tiene un `DO $$ ... RAISE EXCEPTION` que
    valida post-CREATE que todas las policies tienen el wrap. Si alguien
    elimina ese bloque, atrapamos en CI antes que se aplique."""
    text = _MIGRATION_FILES[0].read_text(encoding="utf-8")
    assert "DO $$" in text or "DO\n$$" in text or "do $$" in text.lower(), (
        "Migración debe incluir DO $$ ... RAISE EXCEPTION sanity check."
    )
    assert "RAISE EXCEPTION" in text.upper(), (
        "Migración debe incluir RAISE EXCEPTION en el sanity check."
    )
    assert "P1-RLS-INITPLAN" in text, (
        "Sanity check debe mencionar [P1-RLS-INITPLAN] para grepability."
    )
