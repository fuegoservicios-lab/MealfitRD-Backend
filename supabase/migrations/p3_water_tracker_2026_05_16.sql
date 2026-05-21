-- [P3-WATER-TRACKER · 2026-05-16] Tracker diario de hidratacion (8 vasos default)
-- reemplazo del card "Mi Nevera" en Dashboard que duplicaba la pagina Pantry.
--
-- Diseno:
--   - PK compuesta (user_id, log_date) → un row por dia por usuario, upsert simple.
--   - log_date es la fecha LOCAL del cliente (DATE, no TIMESTAMPTZ) — el reset
--     "a medianoche local" emerge naturalmente: el cliente envia su fecha local
--     YYYY-MM-DD y el GET para una nueva fecha devuelve 0.
--   - CHECK glasses BETWEEN 0 AND 50 — cap defensivo contra click-spam o bug.
--   - FK a auth.users(id) ON DELETE CASCADE — al borrar el usuario sus logs se
--     limpian solos (mismo patron que weight_log/consumed_meals).
--   - Idempotente: IF NOT EXISTS + DROP POLICY IF EXISTS (convencion
--     P3-MIGRATION-IDEMPOTENCE-DOC).

CREATE TABLE IF NOT EXISTS public.water_intake_log (
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    log_date DATE NOT NULL,
    glasses INT NOT NULL DEFAULT 0 CHECK (glasses >= 0 AND glasses <= 50),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, log_date)
);

CREATE INDEX IF NOT EXISTS idx_water_intake_log_user_date
    ON public.water_intake_log (user_id, log_date DESC);

COMMENT ON TABLE public.water_intake_log IS
    'P3-WATER-TRACKER . 2026-05-16: hidratacion diaria por usuario. '
    'log_date es la fecha local del cliente (no UTC) — reset emerge del rollover '
    'de fecha local. PK (user_id, log_date) permite upsert simple.';

COMMENT ON COLUMN public.water_intake_log.log_date IS
    'Fecha LOCAL del usuario (no UTC). El cliente envia YYYY-MM-DD construido '
    'desde su timezone. Esto evita la complejidad de almacenar timezone por '
    'usuario para el caso simple de "reset a medianoche local".';

ALTER TABLE public.water_intake_log ENABLE ROW LEVEL SECURITY;

-- Policies idempotentes. Patron simetrico a consumed_meals / weight_log.
DROP POLICY IF EXISTS water_intake_log_select_own ON public.water_intake_log;
CREATE POLICY water_intake_log_select_own ON public.water_intake_log
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS water_intake_log_insert_own ON public.water_intake_log;
CREATE POLICY water_intake_log_insert_own ON public.water_intake_log
    FOR INSERT TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS water_intake_log_update_own ON public.water_intake_log;
CREATE POLICY water_intake_log_update_own ON public.water_intake_log
    FOR UPDATE TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS water_intake_log_delete_own ON public.water_intake_log;
CREATE POLICY water_intake_log_delete_own ON public.water_intake_log
    FOR DELETE TO authenticated
    USING (auth.uid() = user_id);

-- Sanity check (P3-MIGRATION-IDEMPOTENCE-DOC pattern).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'water_intake_log'
    ) THEN
        RAISE EXCEPTION 'water_intake_log table missing after migration';
    END IF;
END $$;
