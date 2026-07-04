-- [P2-AI-TRAINING-CONSENT · 2026-07-04] Consentimiento OPT-IN para el uso
-- futuro de datos del usuario en entrenamiento de modelos propios de MealfitRD.
--
-- Por qué AHORA (antes de que exista el pipeline de ML): el consentimiento no
-- es retroactivo — datos recolectados sin permiso explícito no pueden usarse
-- después. Capturar el flag desde hoy hace que el corpus futuro nazca limpio.
--
-- Contrato:
--   - DEFAULT FALSE (opt-in explícito; fail-secure: ausencia = NO consiente).
--   - Toggle en Configuración → Privacidad ("Entrenamiento de modelos de IA").
--   - TODO pipeline futuro de training data DEBE filtrar por este flag vía
--     db_profiles.get_ai_training_consented_user_ids() (SSOT del gate).
--   - Al activarse el entrenamiento real: actualizar /ai-policy y /privacy
--     ANTES de encender cualquier ETL.

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS ai_training_consent boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN public.user_profiles.ai_training_consent IS
    '[P2-AI-TRAINING-CONSENT 2026-07-04] Opt-in EXPLÍCITO para uso futuro de '
    'datos (planes/conversaciones, anonimizados) en entrenamiento de modelos '
    'propios. DEFAULT false. Gate SSOT: get_ai_training_consented_user_ids(). '
    'No activar ETL de training sin actualizar /ai-policy y /privacy.';

-- Sanity check idempotente (patrón p2_next_4).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'user_profiles'
          AND column_name = 'ai_training_consent'
    ) THEN
        RAISE EXCEPTION 'p2_ai_training_consent: la columna no existe tras el ALTER';
    END IF;
END $$;
