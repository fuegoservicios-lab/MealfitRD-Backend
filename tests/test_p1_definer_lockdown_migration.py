"""[P1-DEFINER-LOCKDOWN · 2026-05-12] La migración SSOT
`p1_definer_functions_lockdown_2026_05_12.sql` DEBE lockear las 3
functions SECURITY DEFINER nuevas (`handle_new_user`,
`get_monthly_plan_count`, `log_unknown_ingredient_rpc`) al pattern
P3-NEW-2 (`SET search_path = ''`) Y emitir REVOKE EXECUTE explícito
desde PUBLIC/anon/authenticated, manteniendo solo `service_role`.

Bug original (audit production-readiness 2026-05-12):
    Snapshot de pg_proc detectó 3 functions SECURITY DEFINER que NO
    estaban listadas en la tabla "Functions ya bajo el pattern" de
    CLAUDE.md. Estado pre-migración:
      - search_path = 'public' (no '') → drift del pattern P3-NEW-2.
      - Grants implícitos: solo service_role (default seguro), pero
        SIN REVOKE explícito. Un futuro `GRANT EXECUTE ... TO
        authenticated` por error en otra migración pisaría el
        lockdown silenciosamente.

    Riesgo IDOR: TEÓRICO no explotable hoy (anon/authenticated no
    pueden invocar las 3 via PostgREST `/rest/v1/rpc/*`), pero
    `get_monthly_plan_count(user_uuid)` y `log_unknown_ingredient_rpc(
    p_user_id, ...)` aceptan UUID arbitrario sin validar contra
    `auth.uid()` — si quedaran expuestas, leak/poison cross-user.

Fix:
    Migración SSOT que (a) flippea search_path a '', (b) emite REVOKE
    explícito triple (PUBLIC, anon, authenticated), (c) GRANT a
    service_role, (d) COMMENT ON FUNCTION cross-link al test.

Estrategia del test (parser estático sobre la migración):
    1. Verificar que la migración existe en `migrations/`.
    2. Para cada una de las 3 functions:
       a. `CREATE OR REPLACE FUNCTION public.<name>(...)`
       b. `SET search_path = ''` (cadena vacía exactamente)
       c. `SECURITY DEFINER` preservado
       d. REVOKE EXECUTE FROM PUBLIC | anon | authenticated (las 3
          líneas presentes — un solo REVOKE no basta).
       e. GRANT EXECUTE TO service_role (preservar el caller legítimo).
       f. COMMENT ON FUNCTION con anchor `P1-DEFINER-LOCKDOWN`.
    3. Verificar anchor textual `P1-DEFINER-LOCKDOWN` en la migración.

Sin behavior test directo contra Postgres porque:
  - Requeriría conexión Supabase en CI (no garantizada).
  - El audit post-aplicación (`execute_sql` sobre pg_proc + grants)
    ya confirmó:
      search_path_settings = ['search_path=""']
      grants = anon=false, authenticated=false, public=false,
               service_role=true
      has_comment = true
    para las 3 functions. El contrato producción↔código está en la
    SQL del archivo de migración (SSOT).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATION_PATH = (
    _REPO_ROOT
    / "migrations"
    / "p1_definer_functions_lockdown_2026_05_12.sql"
)

_FUNCTIONS = [
    # (proname, signature en SQL — exacto como aparece en REVOKE/GRANT/COMMENT)
    ("handle_new_user", "()"),
    ("get_monthly_plan_count", "(uuid)"),
    ("log_unknown_ingredient_rpc", "(uuid, text, text)"),
]


@pytest.fixture(scope="module")
def migration_src() -> str:
    if not _MIGRATION_PATH.exists():
        pytest.fail(
            f"P1-DEFINER-LOCKDOWN regresión: la migración "
            f"{_MIGRATION_PATH.name} no existe en `migrations/`. "
            f"Si fue renombrada o eliminada, restaurarla o actualizar "
            f"este test. La migración es SSOT del lockdown; sin ella el "
            f"contrato REVOKE EXECUTE FROM authenticated es solo "
            f"implícito (default) y vulnerable a GRANT futuro accidental."
        )
    return _MIGRATION_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor textual + presencia de las 3 functions
# ---------------------------------------------------------------------------
def test_anchor_present(migration_src: str):
    """Anchor textual `P1-DEFINER-LOCKDOWN` en comentarios + COMMENTs."""
    assert "P1-DEFINER-LOCKDOWN" in migration_src, (
        "P1-DEFINER-LOCKDOWN regresión: anchor textual desapareció. "
        "Restaurar para `grep -r P1-DEFINER-LOCKDOWN migrations/`."
    )


@pytest.mark.parametrize("proname,_sig", _FUNCTIONS)
def test_create_or_replace_function_present(
    migration_src: str, proname: str, _sig: str
):
    """Cada function debe usar `CREATE OR REPLACE FUNCTION`
    (idempotente). DROP/CREATE rompería el trigger en handle_new_user
    o callsites de las otras."""
    pattern = re.compile(
        rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.{re.escape(proname)}\s*\(",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        f"P1-DEFINER-LOCKDOWN regresión: la migración no usa "
        f"`CREATE OR REPLACE FUNCTION public.{proname}`. "
        f"Si pasó a DROP+CREATE, reconectar dependientes (trigger "
        f"para handle_new_user; callsite db_plans.py:1196 para "
        f"log_unknown_ingredient_rpc)."
    )


# ---------------------------------------------------------------------------
# 2. search_path locked al pattern P3-NEW-2 (cadena vacía)
# ---------------------------------------------------------------------------
def test_search_path_locked_to_empty_string(migration_src: str):
    """Al menos 3 ocurrencias de `SET search_path = ''` (una por
    function). Cadena vacía es el lock estricto del pattern P3-NEW-2;
    `'public'` permite shadowing por temp tables."""
    pattern = re.compile(r"SET\s+search_path\s*=\s*''", re.IGNORECASE)
    matches = pattern.findall(migration_src)
    assert len(matches) >= 3, (
        f"P1-DEFINER-LOCKDOWN regresión: encontré {len(matches)} "
        f"ocurrencia(s) de `SET search_path = ''`. Esperaba ≥3 (una "
        f"por function: handle_new_user, get_monthly_plan_count, "
        f"log_unknown_ingredient_rpc). Si pasó a `'public'` o se "
        f"removió, el pattern P3-NEW-2 (CLAUDE.md) queda violado."
    )


@pytest.mark.parametrize("proname,_sig", _FUNCTIONS)
def test_security_definer_preserved(
    migration_src: str, proname: str, _sig: str
):
    """`SECURITY DEFINER` debe permanecer — sin él la función
    perdería el privilege escalation y los callsites con SERVICE_ROLE
    seguirían funcionando, pero la semántica del REVOKE explícito
    perdería sentido."""
    # Match el bloque de la función: CREATE OR REPLACE FUNCTION ... AS $$
    block_pattern = re.compile(
        rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.{re.escape(proname)}\s*\([^)]*\).*?AS\s*\$\$",
        re.IGNORECASE | re.DOTALL,
    )
    block_match = block_pattern.search(migration_src)
    assert block_match, (
        f"No pude localizar el bloque CREATE para `{proname}` — el "
        f"test test_create_or_replace_function_present debió fallar antes."
    )
    block = block_match.group(0)
    assert re.search(r"SECURITY\s+DEFINER", block, re.IGNORECASE), (
        f"P1-DEFINER-LOCKDOWN regresión: `SECURITY DEFINER` ausente "
        f"en el bloque CREATE de `{proname}`. Si la función pasó a "
        f"SECURITY INVOKER, el patrón cambia y los REVOKE+GRANT "
        f"deben revisarse — INVOKER usa privilegios del caller."
    )


# ---------------------------------------------------------------------------
# 3. REVOKE explícito triple por function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("proname,sig", _FUNCTIONS)
@pytest.mark.parametrize("grantee", ["PUBLIC", "anon", "authenticated"])
def test_revoke_execute_present(
    migration_src: str, proname: str, sig: str, grantee: str
):
    """Cada function debe tener REVOKE EXECUTE explícito desde los 3
    grantees no-trusted (PUBLIC, anon, authenticated). Las 3 líneas
    son load-bearing como contrato SSOT contra GRANT futuro accidental."""
    # Signature normalization: `()` → ``, `(uuid)` → `\(uuid\)`,
    # `(uuid, text, text)` → match flexible whitespace.
    sig_pattern = re.escape(sig).replace(r",\ ", r",\s*")
    pattern = re.compile(
        rf"REVOKE\s+EXECUTE\s+ON\s+FUNCTION\s+public\.{re.escape(proname)}\s*{sig_pattern}\s+FROM\s+{re.escape(grantee)}\s*;",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        f"P1-DEFINER-LOCKDOWN regresión: falta "
        f"`REVOKE EXECUTE ON FUNCTION public.{proname}{sig} FROM "
        f"{grantee};` en la migración. Sin esta línea explícita, un "
        f"futuro `GRANT EXECUTE ... TO {grantee}` ejecutado por error "
        f"(en otra migración o desde el dashboard) abriría la función "
        f"a un caller no-trusted sin que el code review lo note."
    )


@pytest.mark.parametrize("proname,sig", _FUNCTIONS)
def test_grant_to_service_role_present(
    migration_src: str, proname: str, sig: str
):
    """Cada function debe mantener `GRANT EXECUTE ... TO service_role`
    — es el único caller legítimo en backend (Python supabase client
    con SERVICE_ROLE key)."""
    sig_pattern = re.escape(sig).replace(r",\ ", r",\s*")
    pattern = re.compile(
        rf"GRANT\s+EXECUTE\s+ON\s+FUNCTION\s+public\.{re.escape(proname)}\s*{sig_pattern}\s+TO\s+service_role\s*;",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        f"P1-DEFINER-LOCKDOWN regresión: falta "
        f"`GRANT EXECUTE ON FUNCTION public.{proname}{sig} TO "
        f"service_role;`. Sin esta línea, los 3 REVOKE previos dejarían "
        f"la función inejecutable incluso para el backend legítimo, "
        f"rompiendo (a) el trigger de handle_new_user (registros nuevos "
        f"no podrían crear user_profiles), (b) db_plans.py:1196 "
        f"(log_unknown_ingredient_rpc fallback path se silenciaría)."
    )


# ---------------------------------------------------------------------------
# 4. COMMENT ON FUNCTION con anchor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("proname,sig", _FUNCTIONS)
def test_comment_on_function_present(
    migration_src: str, proname: str, sig: str
):
    """Cada function debe tener `COMMENT ON FUNCTION ... IS '...'`
    con anchor `P1-DEFINER-LOCKDOWN`. El COMMENT es lo que un futuro
    operador ve al hacer `\\df+ public.<name>` y entiende "esto está
    locked deliberadamente, no abrir sin leer primero".
    """
    sig_pattern = re.escape(sig).replace(r",\ ", r",\s*")
    pattern = re.compile(
        rf"COMMENT\s+ON\s+FUNCTION\s+public\.{re.escape(proname)}\s*{sig_pattern}\s+IS\s+'[^']*P1-DEFINER-LOCKDOWN[^']*'",
        re.IGNORECASE | re.DOTALL,
    )
    assert pattern.search(migration_src), (
        f"P1-DEFINER-LOCKDOWN regresión: COMMENT ON FUNCTION "
        f"public.{proname}{sig} ausente o sin anchor "
        f"`P1-DEFINER-LOCKDOWN`. El COMMENT es la documentación "
        f"in-database visible vía `obj_description(<oid>, 'pg_proc')`."
    )


# ---------------------------------------------------------------------------
# 5. Idempotencia textual
# ---------------------------------------------------------------------------
def test_migration_uses_begin_commit(migration_src: str):
    """La migración debe envolver los DDL en BEGIN/COMMIT — atomicidad
    transaccional contra fallos parciales."""
    assert re.search(r"\bBEGIN\s*;", migration_src, re.IGNORECASE), (
        "P1-DEFINER-LOCKDOWN regresión: falta `BEGIN;` — sin "
        "transacción explícita un fallo a media migración deja la DB "
        "en estado inconsistente (algunas functions lockeadas, otras no)."
    )
    assert re.search(r"\bCOMMIT\s*;", migration_src, re.IGNORECASE), (
        "P1-DEFINER-LOCKDOWN regresión: falta `COMMIT;`."
    )


def test_notify_pgrst_reload_present(migration_src: str):
    """`NOTIFY pgrst, 'reload schema'` post-COMMIT — PostgREST cachea
    signatures de functions; sin reload, los nuevos search_path/grants
    pueden no reflejarse hasta el próximo restart del PostgREST worker."""
    pattern = re.compile(
        r"NOTIFY\s+pgrst\s*,\s*'reload\s+schema'", re.IGNORECASE
    )
    assert pattern.search(migration_src), (
        "P1-DEFINER-LOCKDOWN regresión: falta "
        "`NOTIFY pgrst, 'reload schema';` post-COMMIT. PostgREST "
        "puede servir signatures stale de las 3 functions hasta su "
        "próximo restart natural."
    )
