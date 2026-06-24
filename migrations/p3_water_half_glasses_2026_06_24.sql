-- [P3-WATER-HALF-GLASS · 2026-06-24] Medios vasos (0.5) en el tracker de
-- hidratacion. El rediseño del card de Hidratacion ofrece "Sorbo" (½ vaso) +
-- un stepper de ½, asi que `glasses` pasa de INT a NUMERIC(5,1).
--
-- - El CHECK de rango [0,50] de la migracion original (p3_water_tracker) se
--   mantiene (sobrevive al cambio de tipo de columna).
-- - Se agrega un CHECK de paso 0.5 (defensa-en-profundidad; la app y el endpoint
--   POST /water-intake ya validan). `glasses*2` entero ⇔ multiplo de 0.5.
-- - Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): guard por data_type +
--   DROP CONSTRAINT IF EXISTS antes de ADD.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'water_intake_log'
          AND column_name = 'glasses' AND data_type = 'integer'
    ) THEN
        ALTER TABLE public.water_intake_log
            ALTER COLUMN glasses TYPE numeric(5,1) USING glasses::numeric(5,1);
    END IF;
END $$;

-- Paso de 0.5 (medios vasos).
ALTER TABLE public.water_intake_log
    DROP CONSTRAINT IF EXISTS water_intake_log_glasses_half_step;
ALTER TABLE public.water_intake_log
    ADD CONSTRAINT water_intake_log_glasses_half_step
    CHECK ((glasses * 2) = floor(glasses * 2));

COMMENT ON COLUMN public.water_intake_log.glasses IS
    'P3-WATER-HALF-GLASS . 2026-06-24: conteo de vasos del dia. NUMERIC(5,1) '
    'para soportar medios vasos (Sorbo = 0.5). Rango [0,50] + paso 0.5 via CHECK.';

-- Sanity check (P3-MIGRATION-IDEMPOTENCE-DOC pattern).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'water_intake_log'
          AND column_name = 'glasses' AND data_type = 'numeric'
    ) THEN
        RAISE EXCEPTION 'water_intake_log.glasses no es numeric tras la migracion';
    END IF;
END $$;
