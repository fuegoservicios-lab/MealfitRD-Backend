-- [P2-NEW-E · 2026-05-07] Consolidación del DDL runtime de `app.py:lifespan`
-- al SSOT de migraciones. Antes, 7 tablas (push_subscriptions, system_alerts,
-- pipeline_metrics, plan_chunk_queue, plan_chunk_metrics, chunk_deferrals,
-- chunk_lesson_telemetry) + ALTERs sobre user_profiles/user_inventory se
-- creaban con `CREATE TABLE IF NOT EXISTS` cada startup desde Python. Mismo
-- patrón estructural que P1-NEW-A 2026-05-08 (runtime DDL recreaba dup
-- indexes): un cambio del schema vía SQL editor no era visible en código →
-- el siguiente edit del bloque Python podía revertir o pisar el cambio sin
-- detección. Migración idempotente: NOOP sobre el schema actual de producción.
-- ============================================================================

-- ============================================================
-- 1. push_subscriptions (Web Push notifications)
-- ============================================================
CREATE TABLE IF NOT EXISTS push_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    subscription_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- 2. system_alerts (alertas SRE persistentes)
-- ============================================================
CREATE TABLE IF NOT EXISTS system_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_key TEXT NOT NULL UNIQUE,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'warning',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    affected_user_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE NULL
);

-- ============================================================
-- 3. pipeline_metrics (P1-Q10: schema canónico, user_id NULL allowed)
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NULL,
    session_id TEXT,
    node TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    tokens_estimated INTEGER NOT NULL DEFAULT 0,
    confidence NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
ALTER TABLE pipeline_metrics
    ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE pipeline_metrics
    ADD COLUMN IF NOT EXISTS is_guest BOOLEAN
    GENERATED ALWAYS AS (user_id IS NULL) STORED;
CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_node_created
    ON pipeline_metrics (node, created_at DESC);

-- ============================================================
-- 4. user_profiles ALTERs (quality_alert_at + pantry_tolerance + CHECK)
-- ============================================================
ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS quality_alert_at TIMESTAMP WITH TIME ZONE;

-- [P1-D] Tolerancia per-usuario en [1.00, 1.50]; NULL = default global.
ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS pantry_tolerance NUMERIC(4,2);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'pantry_tolerance_range_check'
    ) THEN
        ALTER TABLE user_profiles
            ADD CONSTRAINT pantry_tolerance_range_check
            CHECK (pantry_tolerance IS NULL OR (pantry_tolerance >= 1.00 AND pantry_tolerance <= 1.50));
    END IF;
END
$$;

-- ============================================================
-- 5. user_inventory ALTERs (reservation tracking) + backfill
-- ============================================================
ALTER TABLE user_inventory
    ADD COLUMN IF NOT EXISTS reserved_quantity NUMERIC(12,4) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reservation_details JSONB DEFAULT '{}'::jsonb;

UPDATE user_inventory
SET reserved_quantity = COALESCE(reserved_quantity, 0),
    reservation_details = COALESCE(reservation_details, '{}'::jsonb)
WHERE reserved_quantity IS NULL OR reservation_details IS NULL;

-- ============================================================
-- 6. plan_chunk_queue (Background Chunking JIT) + ALTERs + dedupe + indexes
-- ============================================================
CREATE TABLE IF NOT EXISTS plan_chunk_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    meal_plan_id UUID NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE,
    week_number INT NOT NULL,
    chunk_kind VARCHAR(32) NOT NULL DEFAULT 'initial_plan',
    days_offset INT NOT NULL,
    days_count INT NOT NULL DEFAULT 3,
    pipeline_snapshot JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    attempts INT DEFAULT 0,
    execute_after TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS execute_after TIMESTAMP WITH TIME ZONE DEFAULT NOW();

UPDATE plan_chunk_queue
SET execute_after = NOW()
WHERE execute_after IS NULL AND status = 'pending';

-- [GAP A] SLA + lag al pickup
ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS quality_tier VARCHAR(20),
    ADD COLUMN IF NOT EXISTS lag_seconds_at_pickup INT,
    ADD COLUMN IF NOT EXISTS expected_preemption_seconds INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS effective_lag_seconds_at_pickup INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMP WITH TIME ZONE;

-- [GAP F] Métricas de aprendizaje inter-chunk
ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS learning_metrics JSONB;

-- [P0-2] Atomicidad de persistencia de lección
ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS learning_persisted_at TIMESTAMP WITH TIME ZONE;

-- Dead-letter
ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS dead_lettered_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS dead_letter_reason TEXT;

-- [P1-2] chunk_kind explícito (rolling_refill vs initial_plan)
ALTER TABLE plan_chunk_queue
    ADD COLUMN IF NOT EXISTS chunk_kind VARCHAR(32) DEFAULT 'initial_plan';

UPDATE plan_chunk_queue
SET chunk_kind = CASE
    WHEN COALESCE((pipeline_snapshot->>'_is_rolling_refill')::boolean, false) THEN 'rolling_refill'
    ELSE 'initial_plan'
END
WHERE chunk_kind IS NULL OR chunk_kind = '';

-- [GAP E] Dedupe filas vivas duplicadas (meal_plan_id, week_number) antes del UNIQUE
UPDATE plan_chunk_queue
SET status = 'cancelled', updated_at = NOW()
WHERE id IN (
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY meal_plan_id, week_number
                   ORDER BY updated_at DESC, created_at DESC
               ) AS rn
        FROM plan_chunk_queue
        WHERE status IN ('pending', 'processing', 'stale')
    ) t
    WHERE t.rn > 1
);

-- [GAP E] UNIQUE parcial sobre filas vivas
CREATE UNIQUE INDEX IF NOT EXISTS ux_plan_chunk_queue_live_week
    ON plan_chunk_queue (meal_plan_id, week_number)
    WHERE status IN ('pending', 'processing', 'stale', 'failed');

CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_pending
    ON plan_chunk_queue (execute_after, created_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_processing
    ON plan_chunk_queue (updated_at)
    WHERE status = 'processing';

-- [GAP A] Detección rápida de chunks atrasados
CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_stuck
    ON plan_chunk_queue (execute_after)
    WHERE status IN ('pending', 'stale');

-- ============================================================
-- 7. plan_chunk_metrics (GAP G: histórico de pipeline de chunks)
-- ============================================================
CREATE TABLE IF NOT EXISTS plan_chunk_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID,
    meal_plan_id UUID,
    user_id UUID,
    week_number INT,
    days_count INT,
    duration_ms INT,
    quality_tier VARCHAR(20),
    was_degraded BOOLEAN DEFAULT FALSE,
    retries INT DEFAULT 0,
    lag_seconds INT,
    learning_repeat_pct NUMERIC(5,2),
    rejection_violations INT DEFAULT 0,
    allergy_violations INT DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plan_chunk_metrics_recent
    ON plan_chunk_metrics (created_at DESC);

-- [P2-5 2026-05-07] FK plan_chunk_metrics_meal_plan_id_fkey ON DELETE SET NULL
-- requiere este índice internamente; advisor unused_index es false-positive.
CREATE INDEX IF NOT EXISTS idx_plan_chunk_metrics_plan
    ON plan_chunk_metrics (meal_plan_id);

-- ============================================================
-- 8. chunk_deferrals (P1-3: telemetría de gates que difieren chunks)
-- ============================================================
CREATE TABLE IF NOT EXISTS chunk_deferrals (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    meal_plan_id UUID,
    week_number INT NOT NULL,
    reason TEXT NOT NULL,
    days_until_prev_end INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_deferrals_user_recent
    ON chunk_deferrals (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chunk_deferrals_plan_week
    ON chunk_deferrals (meal_plan_id, week_number, created_at DESC);

-- ============================================================
-- 9. chunk_lesson_telemetry (P0-A/P1.3: telemetría de resolución de lecciones)
--
-- NOTA: los índices canónicos `_event_created_at`, `_user_created_at`,
-- `_plan_week` viven en `p1_3_chunk_lesson_telemetry_table.sql`. NO recrear
-- `_event_recent`/`_user_recent` aquí — los dropea
-- `p1a_drop_dup_indexes_chunk_telemetry.sql` y este archivo es la causa
-- raíz que P1-NEW-A cerró al eliminar el bucle de recreación runtime.
-- ============================================================
CREATE TABLE IF NOT EXISTS chunk_lesson_telemetry (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    meal_plan_id UUID NOT NULL,
    week_number INT NOT NULL,
    event TEXT NOT NULL,
    synthesized_count INT NOT NULL DEFAULT 0,
    queue_count INT NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
