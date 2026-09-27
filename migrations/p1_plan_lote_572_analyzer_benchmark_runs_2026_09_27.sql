-- [P1-PLAN-LOTE-572 · 2026-09-27] Corridas del banco de pruebas del analizador de fotos
-- (docs/superpowers/specs/2026-09-27-analizador-banco-de-pruebas-design.md §2). Una fila por corrida; la escribe
-- SOLO backend/scripts/banco_analizador_correr.py --guardar y la lee SOLO el panel de administración. Idempotente.
CREATE TABLE IF NOT EXISTS public.analyzer_benchmark_runs (
    id BIGSERIAL PRIMARY KEY,
    ran_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT NOT NULL,
    prompt_sha TEXT NOT NULL,
    codigo_sha TEXT NOT NULL DEFAULT '',
    manifest_sha TEXT NOT NULL,
    n INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    metrics JSONB NOT NULL,
    per_dish JSONB NOT NULL,
    tokens JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS analyzer_benchmark_runs_ran_at_idx ON public.analyzer_benchmark_runs (ran_at DESC);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'analyzer_benchmark_runs') THEN
        RAISE EXCEPTION 'p1_plan_lote_572: analyzer_benchmark_runs no existe tras la migracion';
    END IF;
END $$;
