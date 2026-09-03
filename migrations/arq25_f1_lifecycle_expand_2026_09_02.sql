-- [P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Fase 1 del roadmap 2.5 «Núcleo único»
-- (docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md §5.1, §12).
--
-- EXPAND, sin contract: nada de lo que existe cambia de semántica. Cuatro piezas:
--
--   1. `plan_generation_runs`  — idempotencia de la SOLICITUD (I9) + snapshot de
--      la política (Fase 2 la llena). `run_status` NO se almacena: se deriva
--      (generation_lifecycle.derive_run_status) para no crear una sexta fuente
--      de verdad del lifecycle.
--   2. `plan_jobs`             — el outbox (I13). Misma disciplina de claim/CAS
--      que `plan_chunk_queue`. La Fase 5 conecta los consumidores; aquí nace vacía.
--   3. `plan_chunk_queue`      — columnas nuevas, TODAS nullable: `run_id`,
--      `claimed_by`, `input_hash`, `output_hash`. `attempts` sigue siendo el
--      token de fencing (I10): no se añade `lease_token`.
--   4. `meal_plans.revision`   — I12 por TRIGGER, no por convención de código:
--      55 sitios escriben `plan_data` hoy y un writer futuro no tendría que
--      acordarse. `BEFORE UPDATE OF plan_data` + `IS DISTINCT FROM` ⇒ solo sube
--      cuando el contenido cambia de verdad.
--
-- Idempotente (IF NOT EXISTS / OR REPLACE / DROP TRIGGER IF EXISTS) y con
-- sanity DO $$ al final. SSOT dual: copia byte-idéntica en `migrations/` del
-- workspace-root y en `backend/migrations/`. Ledger: `scripts/apply_migration.py`.

-- ============================================================
-- 1. plan_generation_runs
-- ============================================================
CREATE TABLE IF NOT EXISTS public.plan_generation_runs (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    plan_id                UUID REFERENCES public.meal_plans(id) ON DELETE SET NULL,
    idempotency_key        TEXT NOT NULL,
    request_fingerprint    TEXT NOT NULL,
    requested_days         INT  NOT NULL,
    market_country         TEXT,
    locale                 TEXT,
    workflow_version       TEXT NOT NULL DEFAULT 'arq25-f1',
    policy_schema_version  INT  NOT NULL DEFAULT 0,
    policy_hash            TEXT,
    requested_policy       JSONB NOT NULL DEFAULT '{}'::jsonb,
    effective_policy       JSONB NOT NULL DEFAULT '{}'::jsonb,
    relaxations            JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_snapshot         JSONB NOT NULL DEFAULT '{}'::jsonb,
    correlation_id         TEXT,
    engine_versions        JSONB NOT NULL DEFAULT '{}'::jsonb,
    cancel_requested_at    TIMESTAMPTZ,
    error_code             TEXT,
    error_redacted         TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at           TIMESTAMPTZ
);

-- I9: la MISMA clave lógica del MISMO usuario nunca crea dos runs. La clave no
-- caduca al terminar el run: un replay tardío devuelve el run anterior.
CREATE UNIQUE INDEX IF NOT EXISTS plan_generation_runs_user_idem_uq
    ON public.plan_generation_runs (user_id, idempotency_key);
CREATE INDEX IF NOT EXISTS plan_generation_runs_plan_id_idx
    ON public.plan_generation_runs (plan_id);
CREATE INDEX IF NOT EXISTS plan_generation_runs_user_created_idx
    ON public.plan_generation_runs (user_id, created_at DESC);

-- ============================================================
-- 2. plan_jobs (outbox genérico; consumidores en Fase 5)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.plan_jobs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type         TEXT NOT NULL,
    plan_id          UUID REFERENCES public.meal_plans(id) ON DELETE CASCADE,
    user_id          UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    plan_revision    INT,
    dedup_key        TEXT NOT NULL,
    payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    status           TEXT NOT NULL DEFAULT 'pending',
    attempts         INT  NOT NULL DEFAULT 0,
    execute_after    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_by       TEXT,
    heartbeat_at     TIMESTAMPTZ,
    error_code       TEXT,
    error_redacted   TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at     TIMESTAMPTZ,
    dead_lettered_at TIMESTAMPTZ
);

ALTER TABLE public.plan_jobs DROP CONSTRAINT IF EXISTS plan_jobs_status_check;
ALTER TABLE public.plan_jobs ADD CONSTRAINT plan_jobs_status_check
    CHECK (status IN ('pending', 'processing', 'done', 'failed', 'dead', 'stale'));

CREATE UNIQUE INDEX IF NOT EXISTS plan_jobs_dedup_key_uq
    ON public.plan_jobs (dedup_key);
CREATE INDEX IF NOT EXISTS plan_jobs_claim_idx
    ON public.plan_jobs (status, execute_after)
    WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS plan_jobs_plan_idx
    ON public.plan_jobs (plan_id, plan_revision);

-- ============================================================
-- 3. plan_chunk_queue: columnas nuevas (nullable, sin backfill)
-- ============================================================
ALTER TABLE public.plan_chunk_queue
    ADD COLUMN IF NOT EXISTS run_id      UUID REFERENCES public.plan_generation_runs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS claimed_by  TEXT,
    ADD COLUMN IF NOT EXISTS input_hash  TEXT,
    ADD COLUMN IF NOT EXISTS output_hash TEXT;

CREATE INDEX IF NOT EXISTS plan_chunk_queue_run_id_idx
    ON public.plan_chunk_queue (run_id)
    WHERE run_id IS NOT NULL;

-- ============================================================
-- 4. meal_plans.revision (I12) + run_id, con trigger
-- ============================================================
ALTER TABLE public.meal_plans
    ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS run_id   UUID;

-- SET search_path = '' (patrón P3-NEW-2): todo calificado con public.
CREATE OR REPLACE FUNCTION public.meal_plans_bump_revision()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
    IF NEW.plan_data IS DISTINCT FROM OLD.plan_data THEN
        NEW.revision := COALESCE(OLD.revision, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS meal_plans_bump_revision_trg ON public.meal_plans;
CREATE TRIGGER meal_plans_bump_revision_trg
    BEFORE UPDATE OF plan_data ON public.meal_plans
    FOR EACH ROW
    EXECUTE FUNCTION public.meal_plans_bump_revision();

-- ============================================================
-- Sanity
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'meal_plans' AND column_name = 'revision'
    ) THEN
        RAISE EXCEPTION '[ARQ25-F1] meal_plans.revision no existe tras la migración';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'meal_plans_bump_revision_trg' AND NOT tgisinternal
    ) THEN
        RAISE EXCEPTION '[ARQ25-F1] trigger meal_plans_bump_revision_trg no existe';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'plan_chunk_queue' AND column_name = 'run_id'
    ) THEN
        RAISE EXCEPTION '[ARQ25-F1] plan_chunk_queue.run_id no existe';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'plan_generation_runs') THEN
        RAISE EXCEPTION '[ARQ25-F1] plan_generation_runs no existe';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'plan_jobs') THEN
        RAISE EXCEPTION '[ARQ25-F1] plan_jobs no existe';
    END IF;
END $$;
