-- [P3-WATER-TRACKER · 2026-05-16] Toggle del tracker de hidratacion.
-- Default TRUE (decision producto: predeterminado activado). Usuario lo
-- apaga desde Settings/Preferencias para no ver el card en Dashboard.
--
-- Semantica:
--   TRUE  (default) = WaterTracker se renderiza en Dashboard, endpoints
--                     /api/plans/water-intake funcionan normalmente.
--   FALSE          = WaterTracker NO se renderiza. Los rows previos en
--                     water_intake_log quedan intactos (reversible). El
--                     backend NO bloquea los endpoints — la fuente de
--                     verdad es el frontend que oculta el componente.
--
-- IF NOT EXISTS para idempotencia (P3-MIGRATION-IDEMPOTENCE-DOC).
ALTER TABLE public.user_profiles
ADD COLUMN IF NOT EXISTS water_tracker_enabled BOOLEAN NOT NULL DEFAULT TRUE;

COMMENT ON COLUMN public.user_profiles.water_tracker_enabled IS
'P3-WATER-TRACKER . 2026-05-16: toggle del tracker de hidratacion del Dashboard. TRUE (default) = card visible. FALSE = card oculto en Dashboard pero el historial water_intake_log queda intacto (reversible).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'user_profiles'
          AND column_name = 'water_tracker_enabled'
    ) THEN
        RAISE EXCEPTION 'water_tracker_enabled column missing after migration';
    END IF;
END $$;
