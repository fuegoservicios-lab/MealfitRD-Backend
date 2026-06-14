-- [P1-RLS-INITPLAN · 2026-05-20] Wrap `auth.uid()` con `(select auth.uid())` en
-- 10 RLS policies de 3 tablas (water_intake_log, agent_messages, conversation_summaries).
--
-- Razón: el advisor Supabase `auth_rls_initplan` reporta WARN porque postgres
-- re-evalúa `auth.uid()` para cada row de la tabla durante una query con WHERE
-- match-all. Envolviendo en `(select auth.uid())` postgres trata la expresión
-- como un parámetro stable (InitPlan), evaluado una vez por query — performance
-- O(1) vs O(n).
--
-- Docs: https://supabase.com/docs/guides/database/postgres/row-level-security#call-functions-with-select
-- Advisor: https://supabase.com/docs/guides/database/database-linter?lint=0003_auth_rls_initplan
--
-- Patrón de migración: DROP POLICY IF EXISTS + CREATE POLICY (idempotente).
-- Las policies de `service_role` con `qual = true` (DELETE policies) NO se
-- tocan — no invocan `auth.uid()`, no son afectadas por el advisor.
--
-- Sanity check al final via DO $$ RAISE EXCEPTION si alguna policy
-- queda sin el wrap esperado.

-- ============================================================================
-- water_intake_log (4 policies authenticated)
-- ============================================================================

DROP POLICY IF EXISTS water_intake_log_select_own ON public.water_intake_log;
CREATE POLICY water_intake_log_select_own ON public.water_intake_log
  FOR SELECT TO authenticated
  USING ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS water_intake_log_insert_own ON public.water_intake_log;
CREATE POLICY water_intake_log_insert_own ON public.water_intake_log
  FOR INSERT TO authenticated
  WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS water_intake_log_update_own ON public.water_intake_log;
CREATE POLICY water_intake_log_update_own ON public.water_intake_log
  FOR UPDATE TO authenticated
  USING ((select auth.uid()) = user_id)
  WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS water_intake_log_delete_own ON public.water_intake_log;
CREATE POLICY water_intake_log_delete_own ON public.water_intake_log
  FOR DELETE TO authenticated
  USING ((select auth.uid()) = user_id);

-- ============================================================================
-- agent_messages (3 policies authenticated)
-- ============================================================================

DROP POLICY IF EXISTS authenticated_select_own_messages ON public.agent_messages;
CREATE POLICY authenticated_select_own_messages ON public.agent_messages
  FOR SELECT TO authenticated
  USING ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS authenticated_insert_own_messages ON public.agent_messages;
CREATE POLICY authenticated_insert_own_messages ON public.agent_messages
  FOR INSERT TO authenticated
  WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS authenticated_update_own_messages ON public.agent_messages;
CREATE POLICY authenticated_update_own_messages ON public.agent_messages
  FOR UPDATE TO authenticated
  USING ((select auth.uid()) = user_id)
  WITH CHECK ((select auth.uid()) = user_id);

-- ============================================================================
-- conversation_summaries (3 policies authenticated)
-- ============================================================================

DROP POLICY IF EXISTS authenticated_select_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_select_own_summaries ON public.conversation_summaries
  FOR SELECT TO authenticated
  USING ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS authenticated_insert_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_insert_own_summaries ON public.conversation_summaries
  FOR INSERT TO authenticated
  WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS authenticated_update_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_update_own_summaries ON public.conversation_summaries
  FOR UPDATE TO authenticated
  USING ((select auth.uid()) = user_id)
  WITH CHECK ((select auth.uid()) = user_id);

-- ============================================================================
-- Sanity: las 10 policies DEBEN existir post-migración y su qual/with_check
-- DEBE contener el wrap `( SELECT auth.uid() )` (postgres rewrite normaliza
-- el texto, así que matchear sustring case-insensitive sin paréntesis exactos).
-- ============================================================================

DO $$
DECLARE
  missing_wrap text[] := ARRAY[]::text[];
  rec record;
BEGIN
  FOR rec IN
    SELECT tablename, policyname, qual, with_check
    FROM pg_policies
    WHERE schemaname = 'public'
      AND (
        (tablename = 'water_intake_log' AND policyname IN ('water_intake_log_select_own', 'water_intake_log_insert_own', 'water_intake_log_update_own', 'water_intake_log_delete_own'))
        OR (tablename = 'agent_messages' AND policyname IN ('authenticated_select_own_messages', 'authenticated_insert_own_messages', 'authenticated_update_own_messages'))
        OR (tablename = 'conversation_summaries' AND policyname IN ('authenticated_select_own_summaries', 'authenticated_insert_own_summaries', 'authenticated_update_own_summaries'))
      )
  LOOP
    -- combine qual + with_check, must contain "SELECT auth.uid()" substring (case-insensitive)
    IF coalesce(rec.qual, '') !~* 'select\s+auth\.uid\(\)' AND coalesce(rec.with_check, '') !~* 'select\s+auth\.uid\(\)' THEN
      missing_wrap := array_append(missing_wrap, rec.tablename || '.' || rec.policyname);
    END IF;
  END LOOP;

  IF array_length(missing_wrap, 1) IS NOT NULL THEN
    RAISE EXCEPTION '[P1-RLS-INITPLAN] policies sin wrap (select auth.uid()): %', missing_wrap;
  END IF;
END $$;
