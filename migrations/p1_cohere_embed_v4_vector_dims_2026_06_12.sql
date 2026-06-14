-- ============================================================================
-- [P1-COHERE-EMBED-V4 · 2026-06-12] Dimensiones pgvector 768 → 1536
-- ----------------------------------------------------------------------------
-- Contexto: migración de embeddings Gemini (gemini-embedding-2, 768d
-- truncado) → Cohere Embed v4 (`embed-v4.0`, output_dimension=1536, default
-- del knob MEALFIT_EMBEDDINGS_DIMENSION). Los vectores legacy de Gemini se
-- ANULAN: espacios vectoriales de modelos distintos no son comparables — un
-- cosine cross-espacio produce similitudes sin significado (peor que NULL).
--
-- Estado prod al escribir (verificado via MCP 2026-06-12):
--   user_facts.embedding            768d · 0 filas totales      · idx hnsw
--   meal_plans.profile_embedding    768d · 0 de 4 con vector    · idx hnsw
--   visual_diary.embedding          768d · 3 filas con vector   · sin índice
--   nudge_outcomes.context_embedding 1536d YA (el código Gemini truncaba a
--     768 → los INSERT fallaban en silencio; con Cohere@1536 ese path
--     queda funcional por primera vez)
--   RPCs (match_user_facts, hybrid_search_user_facts, match_visual_diary,
--     match_similar_plan, match_successful_nudges, match_user_facts_hybrid_
--     metadata): parámetro `vector` SIN typmod → aceptan cualquier dimensión,
--     no requieren cambios.
--
-- Backfill: NO se re-embebe nada aquí (0 facts, 3 entradas de diario cuyo
-- texto `description` se conserva — el retrieval para esas 3 entradas
-- degrada hasta que se regeneren orgánicamente). Los vectores nuevos los
-- escribe el backend con purpose=document via embeddings_provider.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): cada bloque verifica el
-- atttypmod actual y solo actúa si ≠1536; índices con IF EXISTS/IF NOT
-- EXISTS; sanity final con RAISE EXCEPTION.
-- ============================================================================

-- ---- user_facts.embedding --------------------------------------------------
DO $$
DECLARE
  v_dim int;
BEGIN
  SELECT a.atttypmod INTO v_dim
  FROM pg_attribute a
  WHERE a.attrelid = 'public.user_facts'::regclass
    AND a.attname = 'embedding' AND NOT a.attisdropped;

  IF v_dim IS DISTINCT FROM 1536 THEN
    DROP INDEX IF EXISTS public.idx_user_facts_embedding;
    UPDATE public.user_facts SET embedding = NULL WHERE embedding IS NOT NULL;
    ALTER TABLE public.user_facts ALTER COLUMN embedding TYPE vector(1536);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_facts_embedding
  ON public.user_facts USING hnsw (embedding vector_cosine_ops);

COMMENT ON COLUMN public.user_facts.embedding IS
  'Cohere embed-v4.0 @1536, input_type=search_document (P1-COHERE-EMBED-V4 2026-06-12). Vectores Gemini legacy (768) anulados en la migración.';

-- ---- meal_plans.profile_embedding (semantic plan cache) --------------------
DO $$
DECLARE
  v_dim int;
BEGIN
  SELECT a.atttypmod INTO v_dim
  FROM pg_attribute a
  WHERE a.attrelid = 'public.meal_plans'::regclass
    AND a.attname = 'profile_embedding' AND NOT a.attisdropped;

  IF v_dim IS DISTINCT FROM 1536 THEN
    DROP INDEX IF EXISTS public.meal_plans_profile_emb_idx;
    UPDATE public.meal_plans SET profile_embedding = NULL WHERE profile_embedding IS NOT NULL;
    ALTER TABLE public.meal_plans ALTER COLUMN profile_embedding TYPE vector(1536);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS meal_plans_profile_emb_idx
  ON public.meal_plans USING hnsw (profile_embedding vector_cosine_ops);

COMMENT ON COLUMN public.meal_plans.profile_embedding IS
  'Cohere embed-v4.0 @1536, espacio simétrico query↔query del semantic plan cache (P1-COHERE-EMBED-V4 2026-06-12).';

-- ---- visual_diary.embedding ------------------------------------------------
-- (Sin índice vectorial histórico — no se crea uno aquí: volumen diminuto y
-- la política de advisors del repo penaliza índices sin uso observado.)
DO $$
DECLARE
  v_dim int;
BEGIN
  SELECT a.atttypmod INTO v_dim
  FROM pg_attribute a
  WHERE a.attrelid = 'public.visual_diary'::regclass
    AND a.attname = 'embedding' AND NOT a.attisdropped;

  IF v_dim IS DISTINCT FROM 1536 THEN
    UPDATE public.visual_diary SET embedding = NULL WHERE embedding IS NOT NULL;
    ALTER TABLE public.visual_diary ALTER COLUMN embedding TYPE vector(1536);
  END IF;
END $$;

COMMENT ON COLUMN public.visual_diary.embedding IS
  'Cohere embed-v4.0 @1536, input_type=search_document (P1-COHERE-EMBED-V4 2026-06-12). Vectores Gemini legacy (768) anulados; description se conserva.';

-- ---- Sanity final ----------------------------------------------------------
DO $$
DECLARE
  bad text;
BEGIN
  SELECT string_agg(c.relname || '.' || a.attname || '=' || a.atttypmod, ', ')
  INTO bad
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public' AND c.relkind = 'r' AND NOT a.attisdropped
    AND a.atttypid = (SELECT oid FROM pg_type WHERE typname = 'vector')
    AND a.atttypmod IS DISTINCT FROM 1536;

  IF bad IS NOT NULL THEN
    RAISE EXCEPTION 'P1-COHERE-EMBED-V4 sanity: columnas vector fuera de 1536: %', bad;
  END IF;
END $$;
