-- [P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Añade columna `user_id` directa a
-- `agent_messages` y `conversation_summaries` + 6 policies RLS basadas en
-- `auth.uid()`. Cierra los dos hallazgos del audit Supabase MCP del
-- Agente (2026-05-19) post P0+P1+P2+P3 bundles de código.
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTORES CERRADOS
-- ─────────────────────────────────────────────────────────────────────────
--
-- 1) **DB-P1**: Pre-fix ambas tablas tenían SOLO `session_id` (nullable). El
--    ownership requería siempre un join contra `agent_sessions`. Modos de
--    fallo:
--    - Si un futuro callsite olvida el join (e.g. cron de cleanup, query
--      ad-hoc desde dashboard, refactor que asume `user_id` está en la
--      tabla), un leak cross-user es trivial.
--    - El advisor `unused_index` reporta los índices del join como ruido
--      cuando no se usan en el patrón esperado.
--    - `agent_messages.session_id` es nullable → potencial de mensajes
--      huérfanos sin trazabilidad de owner.
--
-- 2) **DB-P2**: RLS estaba enabled + forced en ambas tablas pero la ÚNICA
--    policy era `DELETE service_role`. Sin SELECT/INSERT/UPDATE policies:
--    - Funciona hoy porque el backend usa `SUPABASE_KEY = SERVICE_ROLE` que
--      bypassea RLS (P0-AUDIT-1 documenta esta dependencia).
--    - Si en el FUTURO algún callsite frontend intenta leer directo via
--      `supabase.from('agent_messages').select(...)` con anon/authenticated
--      key, recibirá 0 rows silenciosamente — bug invisible que se manifiesta
--      como "el chat no muestra mi historial".
--    - La defensa-en-profundidad simétrica al patrón `meal_plans` requiere
--      policies explícitas anclando el ownership a `auth.uid()`.
--
-- ─────────────────────────────────────────────────────────────────────────
-- DISEÑO
-- ─────────────────────────────────────────────────────────────────────────
--
-- - Columna `user_id` es **NULLABLE** (no NOT NULL). Vector legítimo:
--   sesiones de guest (chat sin auth) crean rows con `session_id`
--   válido pero `user_id IS NULL`. Forzar NOT NULL rompería esos flows.
--   Las policies RLS `auth.uid() = user_id` NO matchean NULL → guests
--   solo accesibles vía service_role (correcto).
--
-- - FK a `auth.users(id) ON DELETE CASCADE`. Si Supabase elimina un user
--   (deletion request, GDPR), sus mensajes/summaries se borran
--   automáticamente. Patrón canónico del repo (`user_inventory`,
--   `consumed_meals`, `user_facts`).
--
-- - Backfill via UPDATE FROM contra `agent_sessions`. Los 4 sessions
--   actuales tienen `user_id IS NULL` (guests/tests), así que los 122
--   mensajes + 2 summaries existentes quedan con `user_id = NULL` post-
--   migración. RLS sigue bloqueándolos correctamente para anon/authenticated;
--   service_role los lee.
--
-- - Índices con `WHERE user_id IS NOT NULL` (partial) — no indexar las
--   filas de guest reduce tamaño del índice ~30-50% cuando el ratio
--   guest:authenticated es alto (típico en early product).
--
-- - Índice compuesto `(session_id, created_at DESC)` cierra el gap
--   identificado en el audit: la query principal del chat es
--   "lista mensajes ordenados por una sesión". Pre-fix usaba el índice
--   simple `session_id` + sort en memoria — OK ahora con 122 rows pero
--   degrada con escala.
--
-- - 6 RLS policies (SELECT/INSERT/UPDATE × 2 tablas) con `TO authenticated`.
--   service_role bypassea RLS automáticamente, no necesita policy.
--   DELETE solo via service_role (mantiene política existente).
--   `WITH CHECK` en INSERT/UPDATE previene write cross-user.
--
-- - Idempotencia: `ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`,
--   `DROP POLICY IF EXISTS` antes de `CREATE POLICY`. Re-aplicación
--   sin efectos colaterales.
--
-- - Sanity DO $$ verifica: columnas creadas, policies creadas, sin
--   rows de auth-users con user_id NO matching auth.users.id (FK lo
--   garantiza pero defensive check).
--
-- Tooltip-anchor: P1-CHAT-DB-USER-ID-RLS.
--
-- ─────────────────────────────────────────────────────────────────────────

BEGIN;

-- 1) Añadir columna `user_id` a agent_messages
ALTER TABLE public.agent_messages
  ADD COLUMN IF NOT EXISTS user_id UUID
  REFERENCES auth.users(id) ON DELETE CASCADE;

-- 2) Añadir columna `user_id` a conversation_summaries
ALTER TABLE public.conversation_summaries
  ADD COLUMN IF NOT EXISTS user_id UUID
  REFERENCES auth.users(id) ON DELETE CASCADE;

-- 3) Backfill desde agent_sessions (UPDATE FROM). Filas con session.user_id
--    NULL (guests / tests) quedan con `user_id = NULL` — correcto.
UPDATE public.agent_messages am
SET user_id = ases.user_id
FROM public.agent_sessions ases
WHERE am.session_id = ases.id
  AND am.user_id IS NULL
  AND ases.user_id IS NOT NULL;

UPDATE public.conversation_summaries cs
SET user_id = ases.user_id
FROM public.agent_sessions ases
WHERE cs.session_id = ases.id
  AND cs.user_id IS NULL
  AND ases.user_id IS NOT NULL;

-- 4) Índices partial en (user_id) — guest rows excluidas
CREATE INDEX IF NOT EXISTS idx_agent_messages_user_id
  ON public.agent_messages (user_id)
  WHERE user_id IS NOT NULL;

COMMENT ON INDEX public.idx_agent_messages_user_id IS
  '[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Partial index por user_id (excluye guests). Sirve queries "mensajes de un user across sessions" + cubre FK a auth.users ON DELETE CASCADE.';

CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user_id
  ON public.conversation_summaries (user_id)
  WHERE user_id IS NOT NULL;

COMMENT ON INDEX public.idx_conversation_summaries_user_id IS
  '[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Partial index por user_id. Sirve `search_deep_memory` filter + cubre FK CASCADE.';

-- 5) Índice compuesto `(session_id, created_at DESC)` para listado de
--    mensajes ordenados. Query típica: SELECT * FROM agent_messages WHERE
--    session_id = ? ORDER BY created_at ASC.
CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id_created_at
  ON public.agent_messages (session_id, created_at DESC);

COMMENT ON INDEX public.idx_agent_messages_session_id_created_at IS
  '[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Índice compuesto para listado ordenado. Reemplaza idx_agent_messages_session_id (simple) en queries con ORDER BY created_at.';

-- 6) Policies RLS para `agent_messages`
DROP POLICY IF EXISTS authenticated_select_own_messages ON public.agent_messages;
CREATE POLICY authenticated_select_own_messages
  ON public.agent_messages
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS authenticated_insert_own_messages ON public.agent_messages;
CREATE POLICY authenticated_insert_own_messages
  ON public.agent_messages
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS authenticated_update_own_messages ON public.agent_messages;
CREATE POLICY authenticated_update_own_messages
  ON public.agent_messages
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- 7) Policies RLS para `conversation_summaries`
DROP POLICY IF EXISTS authenticated_select_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_select_own_summaries
  ON public.conversation_summaries
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS authenticated_insert_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_insert_own_summaries
  ON public.conversation_summaries
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS authenticated_update_own_summaries ON public.conversation_summaries;
CREATE POLICY authenticated_update_own_summaries
  ON public.conversation_summaries
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- 8) COMMENTs en columnas para SRE / advisors
COMMENT ON COLUMN public.agent_messages.user_id IS
  '[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Owner del mensaje. NULL para guest sessions (legítimo). FK a auth.users ON DELETE CASCADE — eliminación de cuenta limpia los mensajes automáticamente (GDPR). Backfill desde agent_sessions.user_id en migración. RLS policies authenticated_*_own_messages anclan el ownership a auth.uid().';

COMMENT ON COLUMN public.conversation_summaries.user_id IS
  '[P1-CHAT-DB-USER-ID-RLS · 2026-05-19] Owner del summary archivado. NULL para guest sessions (legítimo). FK a auth.users ON DELETE CASCADE. Backfill desde agent_sessions.user_id en migración.';

-- 9) Sanity check post-migración
DO $$
DECLARE
  am_col_count INT;
  cs_col_count INT;
  am_policy_count INT;
  cs_policy_count INT;
  am_idx_count INT;
  cs_idx_count INT;
BEGIN
  -- Columnas creadas
  SELECT COUNT(*) INTO am_col_count
  FROM information_schema.columns
  WHERE table_schema='public' AND table_name='agent_messages' AND column_name='user_id';

  IF am_col_count != 1 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] agent_messages.user_id no se creó (count=%)', am_col_count;
  END IF;

  SELECT COUNT(*) INTO cs_col_count
  FROM information_schema.columns
  WHERE table_schema='public' AND table_name='conversation_summaries' AND column_name='user_id';

  IF cs_col_count != 1 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] conversation_summaries.user_id no se creó (count=%)', cs_col_count;
  END IF;

  -- Policies creadas (≥3 nuevas por tabla: SELECT/INSERT/UPDATE; DELETE legacy ya existía)
  SELECT COUNT(*) INTO am_policy_count
  FROM pg_policies
  WHERE schemaname='public' AND tablename='agent_messages'
    AND policyname IN ('authenticated_select_own_messages',
                       'authenticated_insert_own_messages',
                       'authenticated_update_own_messages');

  IF am_policy_count != 3 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] agent_messages: esperaba 3 policies nuevas, encontradas %', am_policy_count;
  END IF;

  SELECT COUNT(*) INTO cs_policy_count
  FROM pg_policies
  WHERE schemaname='public' AND tablename='conversation_summaries'
    AND policyname IN ('authenticated_select_own_summaries',
                       'authenticated_insert_own_summaries',
                       'authenticated_update_own_summaries');

  IF cs_policy_count != 3 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] conversation_summaries: esperaba 3 policies nuevas, encontradas %', cs_policy_count;
  END IF;

  -- Índices creados
  SELECT COUNT(*) INTO am_idx_count
  FROM pg_indexes
  WHERE schemaname='public' AND tablename='agent_messages'
    AND indexname IN ('idx_agent_messages_user_id',
                      'idx_agent_messages_session_id_created_at');

  IF am_idx_count != 2 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] agent_messages: esperaba 2 índices nuevos, encontrados %', am_idx_count;
  END IF;

  SELECT COUNT(*) INTO cs_idx_count
  FROM pg_indexes
  WHERE schemaname='public' AND tablename='conversation_summaries'
    AND indexname='idx_conversation_summaries_user_id';

  IF cs_idx_count != 1 THEN
    RAISE EXCEPTION '[P1-CHAT-DB-USER-ID-RLS] conversation_summaries: esperaba 1 índice nuevo, encontrado %', cs_idx_count;
  END IF;

  RAISE NOTICE '[P1-CHAT-DB-USER-ID-RLS] sanity OK: 2 columnas + 6 policies + 3 índices creados';
END $$;

COMMIT;
