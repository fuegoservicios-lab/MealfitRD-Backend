-- ============================================================================
-- [P1-1 · 2026-05-08] Consolidación al SSOT del DDL del semantic cache.
-- ----------------------------------------------------------------------------
-- Origen del drift: el script standalone `backend/scripts/add_semantic_cache.py`
-- declaraba 3 objetos del semantic cache en runtime (column + index + RPC).
-- Producción tiene esos objetos vivos (vía las migraciones históricas
-- `semantic_cache_migration` 2026-04-17 y `fix_match_similar_plan_search_path_extensions`
-- 2026-05-06), pero el repo NO conservaba el archivo `.sql` correspondiente
-- en `supabase/migrations/`. Resultado: un greenfield clon del repo no podría
-- reproducir el schema del semantic cache desde el SSOT del filesystem y
-- dependía del script Python (mismo anti-patrón que cerró P1-A para los
-- scripts price_per_*/paypal_*).
--
-- Verificado vía MCP execute_sql (2026-05-08):
--   - meal_plans.profile_embedding existe como `vector(768)`.
--   - meal_plans_profile_emb_idx existe (HNSW, vector_cosine_ops).
--   - public.match_similar_plan existe con search_path hardeneado a
--     ('public', 'pg_catalog', 'extensions').
--   - Extensión `vector` v0.8.0 instalada.
--
-- Esta migración es idempotente (CREATE EXTENSION IF NOT EXISTS,
-- ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, CREATE OR REPLACE
-- FUNCTION). Sobre el schema actual de producción es NOOP funcional; su
-- valor está en formalizar el schema en el SSOT del repo para reproducción
-- greenfield y para que el script standalone pueda archivarse sin perder
-- el rastro del DDL canónico.
--
-- Consumidores en backend (verificación de uso real):
--   - backend/db_plans.py:487   (column profile_embedding en INSERT)
--   - backend/db_plans.py:1112  (RPC match_similar_plan)
--   - backend/graph_orchestrator.py:2367,9102 (state field profile_embedding)
-- ============================================================================

-- ============================================================
-- 1. Extensión pgvector (defensivo para greenfield)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. meal_plans.profile_embedding: column vectorial 768-dim
-- ============================================================
-- Dimensión 768 alineada con el modelo de embeddings text-only del backend
-- (gemini-embedding-001 / gemini-embedding-2-preview, ambos a 768 dims).
-- Cambiar la dimensión requiere DROP+ADD del column y reindex del HNSW.
ALTER TABLE meal_plans
    ADD COLUMN IF NOT EXISTS profile_embedding vector(768);

COMMENT ON COLUMN meal_plans.profile_embedding IS
    'Embedding 768-dim del perfil del usuario al momento de generar el plan. '
    'Origen histórico: backend/scripts/add_semantic_cache.py (deprecated). '
    'Consolidado al SSOT por P1-1 2026-05-08. Consumido por '
    'search_similar_plan() en db_plans.py vía la RPC match_similar_plan.';

-- ============================================================
-- 3. Índice HNSW para similitud coseno
-- ============================================================
-- HNSW elegido sobre IVFFLAT por mejor recall a costa de mayor espacio
-- (acceptable con la cardinalidad esperada de meal_plans). vector_cosine_ops
-- alinea con el operador `<=>` usado en la RPC.
CREATE INDEX IF NOT EXISTS meal_plans_profile_emb_idx
    ON meal_plans USING hnsw (profile_embedding vector_cosine_ops);

COMMENT ON INDEX meal_plans_profile_emb_idx IS
    'HNSW vector_cosine_ops sobre profile_embedding. Habilita búsqueda '
    'aproximada por similitud coseno usada en match_similar_plan. '
    'Consolidado al SSOT por P1-1 2026-05-08.';

-- ============================================================
-- 4. RPC match_similar_plan
-- ============================================================
-- Refleja el estado de producción tras `fix_match_similar_plan_search_path_extensions`
-- (2026-05-06): el SET search_path es hardening contra search_path attack
-- (advisor Supabase function_search_path_mutable). Mantener este SET; quitarlo
-- regresaría la función al estado vulnerable.
--
-- El parámetro `query_embedding vector` (sin `(768)`) coincide con prod:
-- la función acepta cualquier dimensión y delega la validación al operador
-- `<=>` que falla con "different vector dimensions" si el caller pasa el
-- tamaño equivocado. Mantener la firma sin dimensionar evita un
-- "function does not exist" por mismatch trivial entre caller y schema.
CREATE OR REPLACE FUNCTION public.match_similar_plan (
    query_embedding vector,
    match_threshold double precision,
    match_count integer
)
RETURNS TABLE (
    id uuid,
    user_id uuid,
    plan_data jsonb,
    similarity double precision
)
LANGUAGE sql
SET search_path TO 'public', 'pg_catalog', 'extensions'
AS $$
    SELECT
        id,
        user_id,
        plan_data,
        1 - (meal_plans.profile_embedding <=> query_embedding) AS similarity
    FROM meal_plans
    WHERE 1 - (meal_plans.profile_embedding <=> query_embedding) > match_threshold
    ORDER BY meal_plans.profile_embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION public.match_similar_plan(vector, double precision, integer) IS
    'Busca planes con profile_embedding similar al query_embedding por '
    'similitud coseno. SET search_path es hardening contra search_path '
    'attacks (no remover). Consolidado al SSOT por P1-1 2026-05-08.';
