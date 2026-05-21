-- [P2-UNUSED-IDX · 2026-05-20] Documenta 3 índices que el advisor Supabase
-- marca como `unused_index` pero son INFO intencionales:
--
--   1. idx_agent_messages_user_id          — FK-cover (auth.users CASCADE)
--   2. idx_conversation_summaries_user_id  — FK-cover (auth.users CASCADE)
--   3. idx_llm_usage_events_model_created  — analytics (cost-by-node endpoint)
--
-- Los 2 primeros YA tenían COMMENT vía db_p1_chat_user_id_rls_2026_05_19.sql.
-- Este archivo:
--   (a) refuerza el comment con marker [P2-UNUSED-IDX] para que el advisor
--       anchor sea trivialmente grep-able desde docs/CLAUDE.md;
--   (b) añade el comment faltante para idx_llm_usage_events_model_created;
--   (c) sanity check (DO $$ RAISE) que falla si alguno de los 3 índices
--       desaparece — prevén regresión silenciosa por DROP INDEX manual.
--
-- Patrón documentado en CLAUDE.md → "Advisors aceptados > Performance".
-- Hermana de p2_perf_1_consolidate_unused_index_comments y p3_final_1_meal_plans_audit_advisor_anchors.

COMMENT ON INDEX public.idx_agent_messages_user_id IS
  '[P2-UNUSED-IDX · 2026-05-20] [DB-P1-CHAT-USER-ID-RLS] Partial index on user_id (excludes guests). Serves cross-session queries + covers FK CASCADE to auth.users. Advisor unused_index NO observa uso interno por FK — mantener pese al WARN.';

COMMENT ON INDEX public.idx_conversation_summaries_user_id IS
  '[P2-UNUSED-IDX · 2026-05-20] [DB-P1-CHAT-USER-ID-RLS] Partial index on user_id. Serves search_deep_memory filter + covers FK CASCADE to auth.users. Advisor unused_index NO observa uso interno por FK — mantener pese al WARN.';

COMMENT ON INDEX public.idx_llm_usage_events_model_created IS
  '[P2-UNUSED-IDX · 2026-05-20] [P1-COST-INSTRUMENTATION] Index on (model, created_at DESC) sirve queries analytics de cost-by-model en window N días. Callsite: routers/system.py /api/admin/cost-by-node (GROUP BY node, model). Advisor unused_index marca 0 scans porque el endpoint es admin-only (uso esporádico, NO continuo) — mantener para soportar diagnóstico de incidentes de costo.';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_agent_messages_user_id') THEN
    RAISE EXCEPTION '[P2-UNUSED-IDX] index idx_agent_messages_user_id missing — restore via db_p1_chat_user_id_rls_2026_05_19.sql';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_conversation_summaries_user_id') THEN
    RAISE EXCEPTION '[P2-UNUSED-IDX] index idx_conversation_summaries_user_id missing — restore via db_p1_chat_user_id_rls_2026_05_19.sql';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_llm_usage_events_model_created') THEN
    RAISE EXCEPTION '[P2-UNUSED-IDX] index idx_llm_usage_events_model_created missing — restore via p1_cost_instrumentation_2026_05_15.sql';
  END IF;
END $$;
