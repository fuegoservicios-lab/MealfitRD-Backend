-- [P1-PLAN-MODE · 2026-08-11] La postura del usuario sobre la generación de planes.
-- (copia IDÉNTICA en migrations/ del workspace-root — P3-MIGRATIONS-SSOT, mismo commit)
--
-- UNA columna, no dos: los diseños de partida proponían `plan_generation_enabled`
-- BOOLEAN y `plan_mode` TEXT — el mismo interruptor descrito dos veces, que es el
-- patrón de drift que este repo lleva un año pagando. `plan_mode` es el SSOT único:
-- lo leen el pickup del worker (el ÚNICO sitio por el que pasa todo el gasto LLM de
-- chunks), el bg-refill, el shift, el dashboard, el wizard y la nav.
--
-- Vive en `user_profiles` y no en `plan_data` porque sobrevive al borrado del plan,
-- a generar uno nuevo y a cambiar de dispositivo — y porque `get_user_profile` hace
-- SELECT *, así que llega al frontend sin tocar ningún fetch. Precedente exacto:
-- `logging_preference`, la única preferencia de usuario que el worker ya honra.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC).
BEGIN;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS plan_mode TEXT NOT NULL DEFAULT 'plan';

ALTER TABLE public.user_profiles
    DROP CONSTRAINT IF EXISTS user_profiles_plan_mode_valid;

ALTER TABLE public.user_profiles
    ADD CONSTRAINT user_profiles_plan_mode_valid CHECK (plan_mode IN ('plan', 'tracking'));

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS plan_mode_changed_at TIMESTAMPTZ;

-- El pickup corre CADA MINUTO y evalúa un anti-join contra esta tabla. Índice
-- parcial sobre las filas 'tracking' (la minoría esperada): el NOT EXISTS del gate
-- resuelve por index-only scan y el modo 'plan' —el 99% de las filas— no paga nada.
CREATE INDEX IF NOT EXISTS idx_user_profiles_plan_mode_tracking
    ON public.user_profiles (id) WHERE plan_mode = 'tracking';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'user_profiles'
                     AND column_name = 'plan_mode')
    THEN RAISE EXCEPTION 'P1-PLAN-MODE sanity: falta la columna plan_mode'; END IF;
END $$;

COMMIT;
