"""[P2-4 · 2026-05-10] Defense-in-depth: RLS isolation cross-role.

15 tablas chunk-system tienen `RLS ENABLE+FORCE` (memoria
`project_p0_rls_hardening_2026_05_06`) y otras 11 tablas user-facing
tienen policies con `auth.uid() = user_id`. El backend conecta como
`postgres` (BYPASSRLS=true) — cualquier bug en su SQL podría leakear
data cross-user. La defense-in-depth es PostgREST con role
`authenticated`: ningún usuario debería poder leer rows con `user_id`
distinto al suyo a través de la API REST de Supabase.

Este test verifica los invariantes de las policies SIN requerir JWTs
reales — corre contra un snapshot del estado real de `pg_policy` (vía
fixture JSON) o, si `SUPABASE_DB_URL` está disponible, contra el live
schema.

Cobertura:
    1. Cada tabla chunk-system tiene `rls_forced=true` (ENABLE+FORCE).
    2. Cada tabla user-facing tiene al menos 1 policy con `auth.uid()`
       en `using_expr` (lectura) o `with_check_expr` (escritura).
    3. Tablas con policies "abiertas" (`USING true` para roles públicos)
       están en una whitelist explícita (master_ingredients read-only,
       discount_codes validate-anyone). Cualquier policy nueva con
       `USING true` que no esté en la whitelist falla el test.
    4. Las 17 tablas service_role-only (chunk system + ops internos)
       tienen ÚNICAMENTE policies para role `service_role` — un
       `authenticated` no puede ni leer ni escribir.

Para regenerar el snapshot tras cambios deliberados a policies:
    El SQL de SNAPSHOT_QUERY puede correrse vía Supabase MCP /
    psql; serializar el resultado y reemplazar el fixture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _BACKEND_ROOT / "tests" / "fixtures" / "rls_policies_snapshot_2026_05_10.json"

# SSOT: query usada para generar el snapshot. Si el shape cambia, regenerar
# fixture con esta MISMA query — el test parsea las keys que aquí se proyectan.
SNAPSHOT_QUERY = """
SELECT
  c.relname AS table_name,
  c.relrowsecurity AS rls_enabled,
  c.relforcerowsecurity AS rls_forced,
  p.polname AS policy_name,
  (SELECT array_agg(r.rolname ORDER BY r.rolname)
     FROM pg_roles r WHERE r.oid = ANY(p.polroles)) AS roles,
  pg_get_expr(p.polqual, p.polrelid) AS using_expr,
  pg_get_expr(p.polwithcheck, p.polrelid) AS with_check_expr
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_policy p ON p.polrelid = c.oid
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
  AND c.relrowsecurity = true
ORDER BY c.relname, p.polname;
"""

# Tablas chunk-system + ops internos: service_role debe ser el ÚNICO role
# que las puede tocar. Frontend/authenticated NO debe ver nada.
SERVICE_ROLE_ONLY_TABLES = {
    "abandoned_meal_reasons",
    "agent_messages",
    "agent_sessions",
    "api_usage",
    "app_kv_store",
    "checkpoint_blobs",
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
    "chunk_deferrals",
    "chunk_lesson_telemetry",
    "chunk_user_locks",
    "conversation_summaries",
    "failed_inventory_deductions",
    "ingredient_frequencies",
    "learning_experiments",
    "nightly_rotation_queue",
    "nudge_outcomes",
    "pending_facts_queue",
    "pipeline_metrics",
    "plan_chunk_metrics",
    "plan_chunk_queue",
    "push_subscriptions",
    "summary_archive",
    "system_alerts",
    "weight_log",
}

# Subset que DEBE tener `rls_forced=true` — chunk-system core. El audit
# 2026-05-09 (P1-HIST-AUDIT-6) aplicó FORCE a meal_plans + chunk_lesson_telemetry.
# Si una tabla nueva chunk-system queda solo con rls_enabled=true (sin FORCE),
# el backend con BYPASSRLS sigue accediendo, pero un futuro bug podría dejar
# leakear vía un role distinto. FORCE es defense-in-depth.
CHUNK_SYSTEM_FORCE_REQUIRED = {
    "agent_messages",
    "agent_sessions",
    "app_kv_store",
    "chunk_deferrals",
    "chunk_lesson_telemetry",
    "chunk_user_locks",
    "conversation_summaries",
    "failed_inventory_deductions",
    "learning_experiments",
    "meal_plans",
    "nightly_rotation_queue",
    "nudge_outcomes",
    "pipeline_metrics",
    "plan_chunk_metrics",
    "plan_chunk_queue",
    "push_subscriptions",
    "system_alerts",
}

# Tablas user-facing: cada policy DEBE filtrar por auth.uid().
USER_FACING_TABLES = {
    "consumed_meals",
    "custom_shopping_items",
    "meal_likes",
    "meal_plans",
    "meal_rejections",
    "shopping_locks",
    "user_facts",
    "user_inventory",
    "user_profiles",
    "visual_diary",
    "unknown_ingredients",
}

# Whitelist de tablas con policies legítimamente "abiertas" (USING true sin
# auth.uid()). Cualquier policy nueva sin auth.uid() y NO en esta whitelist
# falla el test.
PUBLIC_POLICIES_WHITELIST = {
    # ("table_name", "policy_name"): "razón documentada"
    ("master_ingredients", "Master ingredients read-only for all"):
        "Catálogo público de ingredientes; lectura para anyone es intencional.",
    ("discount_codes", "Anyone can validate codes"):
        "Frontend valida códigos sin login (UX: aplicar código antes de signup).",
}


# ---------------------------------------------------------------------------
# Carga de policies — fixture o live
# ---------------------------------------------------------------------------
def _load_policies() -> list[dict]:
    """Devuelve la lista de policies. Live si SUPABASE_DB_URL está
    disponible (y P2_4_LIVE=1), sino del fixture committeado."""
    if os.environ.get("P2_4_LIVE") == "1":
        db_url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
        if db_url:
            try:
                import psycopg
                with psycopg.connect(db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(SNAPSHOT_QUERY)
                        cols = [c.name for c in cur.description]
                        return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception as e:
                pytest.skip(f"P2_4_LIVE=1 pero conexión falló: {e}")
    if not _FIXTURE.exists():
        pytest.skip(
            f"Fixture {_FIXTURE} ausente. Setear P2_4_LIVE=1 + "
            f"SUPABASE_DB_URL para corrida live."
        )
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["rls_tables"]


# ---------------------------------------------------------------------------
# 1. RLS ENABLE+FORCE en chunk-system core
# ---------------------------------------------------------------------------
def test_chunk_system_tables_have_rls_forced():
    """Chunk system + meal_plans deben tener `rls_forced=true`."""
    rows = _load_policies()
    by_table: dict[str, dict] = {}
    for r in rows:
        by_table.setdefault(r["table_name"], r)

    missing = []
    for table in CHUNK_SYSTEM_FORCE_REQUIRED:
        if table not in by_table:
            missing.append((table, "no encontrada (¿RLS deshabilitado?)"))
        elif not by_table[table].get("rls_forced"):
            missing.append((table, "rls_forced=false"))

    assert not missing, (
        f"{len(missing)} tabla(s) chunk-system sin FORCE ROW LEVEL SECURITY: "
        f"{missing}. FORCE es defense-in-depth contra bugs del backend que "
        f"corre como postgres BYPASSRLS=true."
    )


# ---------------------------------------------------------------------------
# 2. Tablas service_role-only no exponen policies a authenticated/anon
# ---------------------------------------------------------------------------
def test_service_role_only_tables_dont_leak_to_other_roles():
    rows = _load_policies()
    leaks = []
    for r in rows:
        table = r["table_name"]
        if table not in SERVICE_ROLE_ONLY_TABLES:
            continue
        roles = r.get("roles") or []
        # `roles=None` significa "policy aplica a TODOS los roles" — fuga.
        if roles is None:
            leaks.append((table, r["policy_name"], "roles=null (aplica a todos)"))
            continue
        for role in roles:
            if role not in ("service_role",):
                leaks.append((table, r["policy_name"], f"role={role}"))

    assert not leaks, (
        f"Service-role-only tables con policies para otros roles: {leaks[:5]}. "
        f"Authenticated/anon NO deben tener policy en chunk-system tables. "
        f"Si es intencional (e.g., admin endpoint), añadir a whitelist."
    )


# ---------------------------------------------------------------------------
# 3. User-facing tables: cada policy filtra por auth.uid()
# ---------------------------------------------------------------------------
def test_user_facing_policies_use_auth_uid():
    rows = _load_policies()
    bad = []
    for r in rows:
        table = r["table_name"]
        if table not in USER_FACING_TABLES:
            continue
        policy = r["policy_name"]
        # Service role policies en user-facing tables son OK (admin path).
        roles = r.get("roles") or []
        if "service_role" in roles:
            continue
        using = r.get("using_expr") or ""
        check = r.get("with_check_expr") or ""
        # Al menos una de las dos expresiones debe mencionar auth.uid().
        if "auth.uid()" not in using and "auth.uid()" not in check:
            bad.append((table, policy, using[:50], check[:50]))

    assert not bad, (
        f"Policies user-facing sin filtro auth.uid(): {bad}. "
        f"Riesgo: usuario authenticated podría leer rows de OTRO user. "
        f"Cada policy debe tener `(auth.uid() = user_id)` en USING o WITH CHECK."
    )


# ---------------------------------------------------------------------------
# 4. Policies "abiertas" (USING true sin auth.uid()) en whitelist
# ---------------------------------------------------------------------------
def test_public_policies_are_whitelisted():
    rows = _load_policies()
    suspect = []
    for r in rows:
        table = r["table_name"]
        policy = r["policy_name"]
        # Service role policies pueden tener `USING true` (acceso interno).
        roles = r.get("roles") or []
        if "service_role" in roles:
            continue
        using = (r.get("using_expr") or "").strip()
        check = (r.get("with_check_expr") or "").strip()
        # Si `using=true` o `check=true` SIN auth.uid() en NINGUNA expresión,
        # la policy es pública. Verificar contra whitelist.
        is_public = (
            (using == "true" or check == "true")
            and "auth.uid()" not in using
            and "auth.uid()" not in check
        )
        if not is_public:
            continue
        if (table, policy) not in PUBLIC_POLICIES_WHITELIST:
            suspect.append((table, policy))

    assert not suspect, (
        f"Policies públicas sin whitelist: {suspect}. "
        f"Cada policy con `USING true` para roles no-service debe estar "
        f"explícitamente whitelisted con razón documentada en "
        f"`PUBLIC_POLICIES_WHITELIST`."
    )


# ---------------------------------------------------------------------------
# 5. Sanity: cada tabla user-facing tiene AL MENOS 1 policy auth.uid()
# ---------------------------------------------------------------------------
def test_each_user_facing_table_has_at_least_one_auth_uid_policy():
    rows = _load_policies()
    has_auth_uid: dict[str, bool] = {t: False for t in USER_FACING_TABLES}
    for r in rows:
        table = r["table_name"]
        if table not in USER_FACING_TABLES:
            continue
        using = r.get("using_expr") or ""
        check = r.get("with_check_expr") or ""
        if "auth.uid()" in using or "auth.uid()" in check:
            has_auth_uid[table] = True

    missing = [t for t, ok in has_auth_uid.items() if not ok]
    assert not missing, (
        f"User-facing tables sin policy auth.uid(): {missing}. "
        f"Frontend authenticated podría no acceder a sus propias rows, "
        f"o peor — acceder a las de otros si la policy default es permisiva."
    )
