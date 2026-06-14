-- [P1-DREAMING-1 · 2026-06-13] Fundación del sistema híbrido RAG + Dreaming.
--
-- Combina el RAG actual (user_facts + Cohere embed-v4 @1536 + match_user_facts)
-- con una capa de "Dreaming" (consolidación offline tipo sueño): salience
-- scoring + decay sobre user_facts, una capa de reflexión por usuario
-- (user_memory_profile), y la infraestructura de cola/audit del cron nocturno.
--
-- ADAPTADO A NEON (post P1-NEON-DB/AUTH-MIGRATION): sin `auth.users` (no existe),
-- sin RLS/`auth.uid()` (el backend hace scoping app-side `AND user_id=%s`); las
-- FKs apuntan a `public.user_profiles(id)`; las RPC espejan el estilo de
-- `match_user_facts` (LANGUAGE sql + SET search_path TO 'public','extensions',
-- sin SECURITY DEFINER ni REVOKE — no hay roles authenticated/anon en Neon).
--
-- F0-NEUTRAL: 100% aditivo. NO modifica `match_user_facts` (que ordena por
-- distancia cruda) — el re-rank por salience vive en una fase de lectura gateada
-- por knob (MEALFIT_DREAMING_RETRIEVAL_ENABLED, default OFF). Con los flags
-- MEALFIT_DREAMING_* en OFF el comportamiento es idéntico al de hoy.
--
-- Idempotente (IF NOT EXISTS / DROP CONSTRAINT IF EXISTS / CREATE OR REPLACE +
-- DO $$ RAISE EXCEPTION sanity), re-ejecutable sin efecto. SSOT dual-dir:
-- copia byte-a-byte en migrations/ Y backend/migrations/.

-- ===========================================================================
-- 1) Salience scoring sobre user_facts (la tabla que YA indexa pgvector).
--    Máximo reuso, mínimo footprint. F0: salience uniforme 0.5 (neutral).
-- ===========================================================================
ALTER TABLE public.user_facts ADD COLUMN IF NOT EXISTS salience_score REAL NOT NULL DEFAULT 0.5;
ALTER TABLE public.user_facts ADD COLUMN IF NOT EXISTS last_consolidated_at TIMESTAMPTZ NULL;
ALTER TABLE public.user_facts ADD COLUMN IF NOT EXISTS consolidation_source TEXT NULL; -- 'dream_merge_canonical' | 'online' | NULL

ALTER TABLE public.user_facts DROP CONSTRAINT IF EXISTS user_facts_salience_range;
ALTER TABLE public.user_facts ADD CONSTRAINT user_facts_salience_range
  CHECK (salience_score >= 0 AND salience_score <= 1);

-- Sirve el re-rank por salience del Dreaming + el pickup de facts de alta salience.
CREATE INDEX IF NOT EXISTS idx_user_facts_salience
  ON public.user_facts (user_id, salience_score DESC) WHERE is_active = TRUE;

-- ===========================================================================
-- 2) user_memory_profile: capa de reflexión. UNA fila viva por usuario (no un
--    árbol O(N^2)). Síntesis citable de alto nivel inyectada al prompt.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS public.user_memory_profile (
  user_id                UUID PRIMARY KEY REFERENCES public.user_profiles(id) ON DELETE CASCADE,
  user_model             TEXT NOT NULL,                       -- 6-8 frases inyectadas al system prompt
  embedding              extensions.vector(1536) NULL,        -- Cohere v4 search_document; NULL => sin vector branch
  evidence_fact_ids      UUID[] NOT NULL DEFAULT '{}',        -- FK-verificados en runtime vs user_facts del MISMO user_id
  source_model           TEXT NULL,                           -- p.ej. deepseek-v4-flash
  facts_synthesized_from INT  NOT NULL DEFAULT 0,
  is_active              BOOLEAN NOT NULL DEFAULT TRUE,
  created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- HNSW para match_user_memory (búsqueda semántica del corpus consolidado).
CREATE INDEX IF NOT EXISTS idx_user_memory_profile_embedding
  ON public.user_memory_profile USING hnsw (embedding extensions.vector_cosine_ops) WHERE is_active;

-- ===========================================================================
-- 3) dream_work_queue: cola idempotente leader-safe (qué user_ids necesitan
--    ciclo). Multi-worker via FOR UPDATE SKIP LOCKED + dedup por unique partial.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS public.dream_work_queue (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
  trigger_reason TEXT NULL,                                    -- master_summary | nightly_sweep | manual_admin
  enqueued_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  processed_at   TIMESTAMPTZ NULL,
  attempts       SMALLINT NOT NULL DEFAULT 0,
  last_error     TEXT NULL
);
-- dedup: a lo sumo 1 trabajo pendiente por usuario (ON CONFLICT DO NOTHING al encolar).
CREATE UNIQUE INDEX IF NOT EXISTS idx_dream_queue_user_pending
  ON public.dream_work_queue (user_id) WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_dream_queue_pickup
  ON public.dream_work_queue (enqueued_at) WHERE processed_at IS NULL;

-- ===========================================================================
-- 4) dream_consolidation_log: audit append-only de cada corrida (forensics +
--    rollback de merges via facts_soft_deleted).
-- ===========================================================================
CREATE TABLE IF NOT EXISTS public.dream_consolidation_log (
  id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                 UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
  run_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
  facts_in                INT NULL,
  merges_applied          INT NULL,
  facts_soft_deleted      JSONB NULL,                          -- [{fact_id, fact_text}] para revertir
  contradictions_detected INT NULL,
  profile_updated         BOOLEAN NOT NULL DEFAULT FALSE,
  model_id                TEXT NULL,
  tokens_estimated        INT NULL,
  cost_usd                NUMERIC(10,6) NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_log_user_id ON public.dream_consolidation_log (user_id);

-- ===========================================================================
-- 5) match_user_memory: RPC de búsqueda semántica sobre la capa de reflexión.
--    Espejo EXACTO del estilo de match_user_facts en Neon (LANGUAGE sql +
--    SET search_path TO 'public','extensions', filtra p_user_id internamente,
--    SECURITY INVOKER default). El caller backend pasa el p_user_id autenticado
--    (NUNCA del LLM — simétrico a P0-AGENT-1). 1 fila/usuario máx → barato.
-- ===========================================================================
CREATE OR REPLACE FUNCTION public.match_user_memory(
  query_embedding extensions.vector,
  match_threshold double precision,
  match_count     integer,
  p_user_id       uuid
)
RETURNS TABLE(user_id uuid, user_model text, similarity double precision)
LANGUAGE sql
SET search_path TO 'public', 'extensions'
AS $function$
  SELECT
    ump.user_id,
    ump.user_model,
    1 - (ump.embedding <=> query_embedding) AS similarity
  FROM user_memory_profile ump
  WHERE ump.user_id = p_user_id
    AND ump.is_active = TRUE
    AND ump.embedding IS NOT NULL
    AND (1 - (ump.embedding <=> query_embedding)) > match_threshold
  ORDER BY ump.embedding <=> query_embedding
  LIMIT match_count;
$function$;

-- ===========================================================================
-- Sanity (idempotencia segura: falla ruidoso si el schema quedó a medias).
-- ===========================================================================
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema='public' AND table_name='user_facts' AND column_name='salience_score') THEN
    RAISE EXCEPTION 'P1-DREAMING-1: user_facts.salience_score ausente';
  END IF;
  IF to_regclass('public.user_memory_profile') IS NULL THEN
    RAISE EXCEPTION 'P1-DREAMING-1: user_memory_profile ausente';
  END IF;
  IF to_regclass('public.dream_work_queue') IS NULL THEN
    RAISE EXCEPTION 'P1-DREAMING-1: dream_work_queue ausente';
  END IF;
  IF to_regclass('public.dream_consolidation_log') IS NULL THEN
    RAISE EXCEPTION 'P1-DREAMING-1: dream_consolidation_log ausente';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname='match_user_memory') THEN
    RAISE EXCEPTION 'P1-DREAMING-1: match_user_memory ausente';
  END IF;
END $$;
