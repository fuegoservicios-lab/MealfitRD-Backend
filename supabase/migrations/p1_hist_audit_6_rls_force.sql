-- [P1-HIST-AUDIT-6 · 2026-05-09] FORCE ROW LEVEL SECURITY en
-- `meal_plans` y `chunk_lesson_telemetry`.
--
-- Motivación (audit historial 2026-05-08):
--   Ambas tablas tenían `relrowsecurity=true` pero
--   `relforcerowsecurity=false`. Esto significa que el TABLE OWNER
--   (sin BYPASSRLS) podía ver/modificar todos los rows ignorando las
--   policies. Las otras tablas chunk_* (`plan_chunk_queue`,
--   `chunk_user_locks`, `plan_chunk_metrics`, `chunk_deferrals`) ya
--   tenían FORCE; estas dos eran inconsistentes.
--
-- Defense-in-depth: el backend conecta como `postgres` (BYPASSRLS=true)
-- así que sigue funcionando sin cambios. FORCE cierra el caso edge
-- donde un role custom o una migración futura execute SQL como un
-- role no-superuser que sea OWNER de la tabla. Hoy no aplica; mañana
-- podría. Cerrar el gap ahora previene una clase de bug invisible.
--
-- Roles que SIGUEN bypaseando RLS post-FORCE (verificado contra DB
-- real, project mpoodlmnzaeuuazsazbj):
--   - postgres (BYPASSRLS=true)              ← backend SUPABASE_DB_URL
--   - service_role (BYPASSRLS=true)          ← backend Supabase JS
--   - supabase_admin (BYPASSRLS=true)        ← migraciones admin UI
--
-- Roles que QUEDAN sometidos a RLS post-FORCE (sin cambio funcional):
--   - authenticated (BYPASSRLS=false)        ← clientes JS authenticated
--   - anon (BYPASSRLS=false)                 ← clientes JS pre-login
--
-- Verificación de paths del frontend:
--   - History.jsx::_loadPlanDataLazy: usa Supabase client authenticated;
--     filtra `eq('id', ...).eq('user_id', user.id)` — la policy
--     "Users can view own meal plans" cubre con `auth.uid() = user_id`.
--   - chunk_lesson_telemetry: nunca leída directo desde frontend (solo
--     via backend endpoint /api/plans/lessons-counts).
--
-- Idempotencia: `ALTER TABLE ... FORCE ROW LEVEL SECURITY` es no-op
-- si ya está FORCE. Re-aplicar la migración no rompe nada.

DO $$
BEGIN
    -- ============================================================
    -- 1. meal_plans
    -- ============================================================
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname='public' AND c.relname='meal_plans'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND c.relname='meal_plans'
              AND c.relforcerowsecurity = true
        ) THEN
            ALTER TABLE public.meal_plans FORCE ROW LEVEL SECURITY;
            RAISE NOTICE '[P1-HIST-AUDIT-6] meal_plans FORCE RLS aplicado.';
        ELSE
            RAISE NOTICE '[P1-HIST-AUDIT-6] meal_plans ya tenía FORCE RLS. Skip.';
        END IF;
    END IF;

    -- ============================================================
    -- 2. chunk_lesson_telemetry
    -- ============================================================
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname='public' AND c.relname='chunk_lesson_telemetry'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND c.relname='chunk_lesson_telemetry'
              AND c.relforcerowsecurity = true
        ) THEN
            ALTER TABLE public.chunk_lesson_telemetry FORCE ROW LEVEL SECURITY;
            RAISE NOTICE '[P1-HIST-AUDIT-6] chunk_lesson_telemetry FORCE RLS aplicado.';
        ELSE
            RAISE NOTICE '[P1-HIST-AUDIT-6] chunk_lesson_telemetry ya tenía FORCE RLS. Skip.';
        END IF;
    END IF;
END $$;

-- Anotación documental sobre las tablas (visible en pg_description).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname='public' AND c.relname='meal_plans'
    ) THEN
        EXECUTE 'COMMENT ON TABLE public.meal_plans IS ''[P1-HIST-AUDIT-6] FORCE ROW LEVEL SECURITY: aplicado para defense-in-depth. Backend conecta como postgres (BYPASSRLS=true) y sigue funcionando sin cambios; frontend authenticated cumple las 4 policies con auth.uid() = user_id.''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname='public' AND c.relname='chunk_lesson_telemetry'
    ) THEN
        EXECUTE 'COMMENT ON TABLE public.chunk_lesson_telemetry IS ''[P1-HIST-AUDIT-6] FORCE ROW LEVEL SECURITY: aplicado para defense-in-depth. Solo accedida por backend (BYPASSRLS=true). Policy service_role_all permite todo a service_role; ningún role authenticated tiene policy aquí.''';
    END IF;
END $$;
