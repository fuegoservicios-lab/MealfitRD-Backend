"""[P1-NEON-DB-MIGRATION · 2026-06-12] Anclas de la migración de datos
Supabase → Neon (arquitectura híbrida: Postgres en Neon, Auth+Storage en
Supabase).

Contratos cubiertos (todos parser-based — no requieren DB viva):

1. `db_core.py` define el knob `MEALFIT_DB_BACKEND` con default "supabase",
   y la rama neon es FAIL-LOUD (RuntimeError) si faltan NEON_DATABASE_URL /
   NEON_DATABASE_URL_POOLED — un fallback silencioso a Supabase escribiría
   en la DB equivocada post-cutover (split-brain).
2. `db_core.DB_SESSION_MODE_URL` existe y `app.py::_build_session_mode_db_url`
   lo consume (el leader lock del scheduler necesita session mode; la
   heurística legacy ':6543'→':5432' era NO-OP con hostnames Neon).
3. `db_profiles.ensure_user_profile_exists` reemplaza el trigger
   `handle_new_user` (vivía sobre auth.users — schema inexistente en Neon):
   INSERT ... ON CONFLICT (id) DO NOTHING, y `auth.py` lo invoca tras
   validar el JWT.
4. `cron_tasks.py` registra el job semanal `delete_old_meal_rejections_weekly`
   (reemplaza el pg_cron `cleanup_old_meal_rejections` interno de Supabase).
5. BLANKET: ningún archivo productivo del backend usa PostgREST
   (`supabase.table( / supabase.rpc( / supabase.from_(`). Los únicos usos
   legítimos del cliente supabase son Auth (auth.py) y Storage
   (db_profiles._purge_visual_diary_storage, vision/diary uploads).
6. `scripts/migrate_db_to_neon.py` (script repetible de cutover) conserva
   sus kill-patterns y post-condiciones.
7. Las búsquedas vectoriales de db_facts castean `id::text` (paridad
   PostgREST — el cache LLM hace json.dumps de esas rows; uuid.UUID nativo
   lanzaría TypeError).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_BACKEND_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Knob de backend + fail-loud
# ---------------------------------------------------------------------------

def test_db_core_defines_backend_knob_with_supabase_default():
    src = _read("db_core.py")
    assert re.search(
        r'MEALFIT_DB_BACKEND\s*=\s*\(os\.environ\.get\("MEALFIT_DB_BACKEND"\)\s*or\s*"supabase"\)',
        src,
    ), (
        "db_core.py debe definir MEALFIT_DB_BACKEND con default 'supabase' "
        "(cutover a Neon es opt-in explícito via env)."
    )


def test_db_core_neon_branch_is_fail_loud():
    src = _read("db_core.py")
    m = re.search(
        r'if MEALFIT_DB_BACKEND == "neon":\s*\n(.*?)(?=\n\s+else:)',
        src, re.DOTALL,
    )
    assert m, "Rama `if MEALFIT_DB_BACKEND == \"neon\":` no encontrada en db_core.py"
    branch = m.group(1)
    assert "raise RuntimeError" in branch, (
        "La rama neon DEBE lanzar RuntimeError si faltan los URLs — NO "
        "degradar silenciosamente a SUPABASE_DB_URL (split-brain: escrituras "
        "irían a la DB vieja tras el cutover)."
    )
    assert "NEON_DATABASE_URL_POOLED" in branch and "NEON_DATABASE_URL" in branch, (
        "La rama neon debe consumir NEON_DATABASE_URL (direct→checkpoint pool) "
        "y NEON_DATABASE_URL_POOLED (pooler→pools principales)."
    )


def test_db_core_exports_session_mode_url():
    src = _read("db_core.py")
    assert re.search(r"^DB_SESSION_MODE_URL = None", src, re.MULTILINE), (
        "db_core.py debe declarar DB_SESSION_MODE_URL a nivel de módulo "
        "(importabilidad garantizada aunque la config de pools falle)."
    )
    assert "DB_SESSION_MODE_URL = original_session_url" in src, (
        "DB_SESSION_MODE_URL debe asignarse desde original_session_url "
        "(resuelto según backend activo)."
    )


# ---------------------------------------------------------------------------
# 2. Leader lock usa la URL resuelta por db_core
# ---------------------------------------------------------------------------

def test_app_leader_lock_uses_db_core_session_url():
    src = _read("app.py")
    m = re.search(
        r"def _build_session_mode_db_url\(\).*?(?=\ndef )", src, re.DOTALL
    )
    assert m, "_build_session_mode_db_url no encontrada en app.py"
    body = m.group(0)
    assert "from db_core import DB_SESSION_MODE_URL" in body, (
        "_build_session_mode_db_url debe consumir db_core.DB_SESSION_MODE_URL "
        "— la heurística local ':6543'→':5432' es NO-OP con hostnames Neon y "
        "el advisory lock caería en el pooler transaction-mode (lock liberado "
        "por sentencia = inútil)."
    )


# ---------------------------------------------------------------------------
# 3. ensure_user_profile_exists (reemplazo de handle_new_user)
# ---------------------------------------------------------------------------

def test_ensure_user_profile_exists_on_conflict_do_nothing():
    src = _read("db_profiles.py")
    m = re.search(
        r"def ensure_user_profile_exists\(.*?(?=\ndef )", src, re.DOTALL
    )
    assert m, "ensure_user_profile_exists no encontrada en db_profiles.py"
    body = m.group(0)
    assert "ON CONFLICT (id) DO NOTHING" in body, (
        "ensure_user_profile_exists debe usar ON CONFLICT (id) DO NOTHING — "
        "jamás pisar un profile existente (a diferencia del upsert de "
        "health_profile)."
    )
    assert "INSERT INTO user_profiles" in body
    # El payload espejo del trigger original handle_new_user.
    for col in ("id", "email", "full_name", "created_at"):
        assert col in body, f"Columna `{col}` del trigger original ausente."


def test_auth_invokes_ensure_profile_after_jwt_validation():
    src = _read("auth.py")
    assert "ensure_user_profile_exists" in src, (
        "auth.py debe invocar ensure_user_profile_exists tras validar el JWT "
        "— sin esto, usuarios nuevos de Supabase Auth jamás obtienen fila en "
        "public.user_profiles (el trigger handle_new_user no existe en Neon)."
    )
    assert re.search(
        r"asyncio\.to_thread\(\s*ensure_user_profile_exists", src
    ), (
        "La invocación debe ir via asyncio.to_thread (INSERT sync no debe "
        "bloquear el event loop — convención P2-AUTH-ASYNC-SLEEP)."
    )


# ---------------------------------------------------------------------------
# 4. Cron semanal reemplazo de pg_cron
# ---------------------------------------------------------------------------

def test_meal_rejections_weekly_cron_registered():
    src = _read("cron_tasks.py")
    m = re.search(
        r"def _delete_old_meal_rejections_weekly\(.*?(?=\ndef )", src, re.DOTALL
    )
    assert m, "_delete_old_meal_rejections_weekly no encontrada en cron_tasks.py"
    assert "public.delete_old_meal_rejections()" in m.group(0), (
        "El job debe invocar la función SQL public.delete_old_meal_rejections() "
        "(existe en Neon — restaurada por el dump)."
    )
    assert 'id="delete_old_meal_rejections_weekly"' in src, (
        "El job debe registrarse en register_plan_chunk_scheduler (SSOT de "
        "crons) con id estable."
    )


# ---------------------------------------------------------------------------
# 5. BLANKET: cero PostgREST en código productivo
# ---------------------------------------------------------------------------

_POSTGREST_RE = re.compile(r"supabase\.(table|rpc|from_)\(")

# Archivos donde el cliente supabase sigue siendo legítimo (Auth/Storage) —
# pero NUNCA PostgREST. La whitelist es de ARCHIVOS para el grep del cliente,
# el patrón PostgREST se prohíbe incluso en ellos.
_PROD_DIRS = ("", "routers")


def _prod_python_files():
    for rel_dir in _PROD_DIRS:
        base = _BACKEND_ROOT / rel_dir if rel_dir else _BACKEND_ROOT
        for p in base.glob("*.py"):
            if p.name.startswith("test_"):
                continue
            yield p


def test_blanket_no_postgrest_in_prod_backend():
    """Si este test falla: alguien reintrodujo un call site PostgREST.
    Post-Neon, PostgREST apunta al Postgres de SUPABASE (datos stale) —
    leerlo es servir data vieja; escribirlo es split-brain. Usar los
    helpers execute_sql_* de db_core."""
    offenders: list[str] = []
    for p in _prod_python_files():
        src = p.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _POSTGREST_RE.search(line):
                offenders.append(f"{p.relative_to(_BACKEND_ROOT)}:{i}: {stripped[:100]}")
    assert not offenders, (
        "Call sites PostgREST detectados en código productivo (prohibido "
        "post P1-NEON-DB-MIGRATION — los datos viven en Neon):\n"
        + "\n".join(offenders)
    )


def test_supabase_client_only_for_auth_and_storage():
    """`supabase.auth.` solo en auth.py / app.py (migrate de cuentas) /
    routers/system.py (admin purge: auth.admin.delete_user — Auth queda en
    Supabase); `supabase.storage.` permitido (Storage queda en Supabase)."""
    allowed_auth_files = {"auth.py", "app.py", "db_core.py", "system.py"}
    offenders: list[str] = []
    for p in _prod_python_files():
        if p.name in allowed_auth_files:
            continue
        src = p.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"supabase\.auth\.", line):
                offenders.append(f"{p.relative_to(_BACKEND_ROOT)}:{i}")
    assert not offenders, (
        "supabase.auth.* fuera de auth.py/app.py — la verificación de JWT "
        "está centralizada (P0-AUDIT-1):\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 6. Script de migración repetible
# ---------------------------------------------------------------------------

def test_migration_script_contracts():
    src = _read("scripts/migrate_db_to_neon.py")
    for needle, why in (
        ("_AUTH_DEPENDENT_FUNCTIONS", "lista SSOT de funciones auth-dep a dropear"),
        ("handle_new_user", "trigger de signup (reemplazado app-side)"),
        ("increment_inventory_quantity", "RPC frontend (reemplazada por endpoint)"),
        ("update_health_profile_merge", "RPC frontend (reemplazada por endpoint)"),
        ("REFERENCES auth.users", "post-condición FKs"),
        ("ROW LEVEL SECURITY", "kill pattern RLS"),
        ("--reset-neon", "re-sync de cutover"),
        ("ensure_neon_extensions", "schema extensions espejo (vector qualified)"),
        ("FROM stdin;", "manejo de bloques COPY (apóstrofes crudos)"),
    ):
        assert needle in src, f"migrate_db_to_neon.py perdió `{needle}` ({why})."


# ---------------------------------------------------------------------------
# 7. Paridad de tipos en búsquedas vectoriales (cache LLM hace json.dumps)
# ---------------------------------------------------------------------------

def test_db_facts_vector_searches_cast_id_to_text():
    src = _read("db_facts.py")
    for fn_name in ("search_user_facts", "search_user_facts_hybrid"):
        m = re.search(
            rf"def {fn_name}\(.*?(?=\ndef )", src, re.DOTALL
        )
        assert m, f"{fn_name} no encontrada"
        body = m.group(0)
        assert "id::text" in body, (
            f"{fn_name} debe castear id::text — las funciones de matching "
            "RETURNS TABLE(id uuid,...) devolverían uuid.UUID y el cache LLM "
            "(json.dumps en graph_orchestrator) lanzaría TypeError."
        )
        assert "SELECT *" not in body, (
            f"{fn_name} no debe usar SELECT * sobre las funciones de matching "
            "(columnas enumeradas con casts para paridad PostgREST)."
        )


# ---------------------------------------------------------------------------
# 8. Marker bumpeado
# ---------------------------------------------------------------------------

def test_last_known_pfix_bumped_to_neon_migration():
    """[P1-NEON-AUTH-MIGRATION · 2026-06-13] Acepta cualquier marker `P1-NEON-`
    (DB o AUTH): ambos son fases de la misma migración a Neon. El cierre de la
    fase Auth (posterior) movió el marker de DB→AUTH; lo que importa es que el
    marker refleje la migración Neon, no la sub-fase exacta."""
    src = _read("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    assert m.group(1).startswith("P1-NEON-"), (
        f"Marker actual: {m.group(1)!r} — el cierre de la migración Neon "
        "(DB o AUTH) debe bumpearlo (contrato /health/version + deploy-lag)."
    )
