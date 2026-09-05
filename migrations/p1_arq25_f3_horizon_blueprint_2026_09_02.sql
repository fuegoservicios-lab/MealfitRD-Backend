-- [P1-ARQ25-F3-HORIZON · 2026-09-02] Fase 3 del roadmap 2.5: el blueprint del horizonte
-- (§6.5) se persiste CON el run. Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC).
--
--   plan_generation_runs.blueprint          jsonb   reparto 7/15/30 (días, anclas, chunks, ventanas)
--   plan_generation_runs.blueprint_hash     text    sha256 canónico (misma política ⇒ mismo hash)
--   plan_generation_runs.allocator_version  text    versión del allocator que lo produjo
--
-- Las rebanadas por chunk NO tienen columna propia: viajan dentro de
-- plan_chunk_queue.pipeline_snapshot["_blueprint_slice"] y su hash entra en input_hash.

ALTER TABLE public.plan_generation_runs
    ADD COLUMN IF NOT EXISTS blueprint         JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS blueprint_hash    TEXT,
    ADD COLUMN IF NOT EXISTS allocator_version TEXT;

COMMENT ON COLUMN public.plan_generation_runs.blueprint IS
    '[P1-ARQ25-F3-HORIZON] Full-Horizon Blueprint (§6.5): días/anclas/chunks/ventanas del run.';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'plan_generation_runs' AND column_name = 'blueprint'
    ) THEN
        RAISE EXCEPTION '[P1-ARQ25-F3-HORIZON] plan_generation_runs.blueprint no existe tras la migración';
    END IF;
END $$;
