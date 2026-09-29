-- [P1-PLAN-LOTE-843 · 2026-09-29] Permiso explícito para enviar datos personales a la IA de terceros
-- (App Review 5.1.2(i), texto de nov-2025; RGPD art. 9(2)(a) para los datos de salud y art. 49(1)(a) para la
-- transferencia a DeepSeek en China). Auditoría 2026-09-29, fila 4 y §A.1.
--
-- `user_consents` es el REGISTRO: de solo inserción (cada decisión es una fila nueva, nunca se edita), para que el
-- permiso sea demostrable (art. 7.1). El titular es una cuenta (`user_id`) o un invitado (`guest_hash` = sha256 del
-- session_id del invitado: el id crudo no se guarda). Se va con la cuenta (ON DELETE CASCADE).
--
-- Las columnas nuevas de `user_profiles` son el ESTADO derivado: el gate (recogida de bloques, coach, crons) decide con
-- una lectura por clave primaria en vez de recorrer el registro. Las dos cosas las escribe en la misma transacción
-- `backend/consentimientos.py`, y nadie más. `analytics_consent` NULL = no preguntado.
-- Idempotente.
CREATE TABLE IF NOT EXISTS public.user_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    guest_hash TEXT,
    consent_key TEXT NOT NULL,
    version TEXT NOT NULL,
    granted BOOLEAN NOT NULL,
    text_sha256 TEXT,
    locale TEXT,
    platform TEXT,
    app_build TEXT,
    origen TEXT NOT NULL DEFAULT 'cuenta',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- De dónde sale cada fila: 'cuenta' (la decidió la cuenta), 'invitado' (el invitado, con su guest_hash) o 'adopcion'
-- (la del invitado copiada a su cuenta al adoptar el plan, con su fecha original). El ADD COLUMN y el UPDATE solo
-- hacen algo en una base que aplicó la primera versión de esta migración (con la tabla nueva no hay filas).
ALTER TABLE public.user_consents ADD COLUMN IF NOT EXISTS origen TEXT NOT NULL DEFAULT 'cuenta';
UPDATE public.user_consents SET origen = 'invitado' WHERE guest_hash IS NOT NULL AND origen <> 'invitado';

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_origen_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_origen_chk
    CHECK (origen IN ('cuenta', 'invitado', 'adopcion'));

-- Una fila de invitado es la que lleva guest_hash, y ninguna otra.
ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_origen_titular_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_origen_titular_chk
    CHECK ((origen = 'invitado') = (guest_hash IS NOT NULL));

-- Exactamente un titular: la cuenta O el invitado, nunca los dos ni ninguno.
ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_titular_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_titular_chk
    CHECK (num_nonnulls(user_id, guest_hash) = 1);

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_guest_hash_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_guest_hash_chk
    CHECK (guest_hash IS NULL OR guest_hash ~ '^[0-9a-f]{64}$');

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_key_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_key_chk
    CHECK (consent_key IN ('ai_processing', 'ai_transfer_cn', 'analytics'));

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_version_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_version_chk
    CHECK (char_length(version) BETWEEN 1 AND 32);

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_text_sha256_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_text_sha256_chk
    CHECK (text_sha256 IS NULL OR text_sha256 ~ '^[0-9a-f]{64}$');

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_platform_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_platform_chk
    CHECK (platform IS NULL OR platform IN ('ios', 'android', 'web'));

ALTER TABLE public.user_consents DROP CONSTRAINT IF EXISTS user_consents_textos_chk;
ALTER TABLE public.user_consents ADD CONSTRAINT user_consents_textos_chk
    CHECK ((locale IS NULL OR char_length(locale) <= 16) AND (app_build IS NULL OR char_length(app_build) <= 64));

CREATE INDEX IF NOT EXISTS user_consents_user_key_idx
    ON public.user_consents (user_id, consent_key, created_at DESC);
CREATE INDEX IF NOT EXISTS user_consents_guest_idx
    ON public.user_consents (guest_hash, created_at DESC);

REVOKE ALL ON public.user_consents FROM PUBLIC;

COMMENT ON TABLE public.user_consents IS
    'P1-PLAN-LOTE-843: registro de solo inserción del permiso para la IA de terceros (ai_processing, ai_transfer_cn) y la analítica. Lo escribe solo backend/consentimientos.py.';

-- El estado derivado (lo lee el gate con una lectura por PK).
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ai_consent_version TEXT;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ai_consent_at TIMESTAMPTZ;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ai_consent_revoked_at TIMESTAMPTZ;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ai_cn_transfer_at TIMESTAMPTZ;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS analytics_consent BOOLEAN;
-- La hora de la pausa del generador que puso la retirada: el MISMO now() que estampó en plan_mode_changed_at. Volver a
-- conceder reanuda solo si el plan sigue en 'tracking' desde esa hora exacta; un encendido o apagado a mano después la
-- deja atrás. NULL = la retirada no pausó nada (ya estaba en seguimiento) o ya se concedió otra vez.
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ai_consent_paused_at TIMESTAMPTZ;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'user_consents') THEN
        RAISE EXCEPTION 'p1_plan_lote_843: user_consents no existe tras la migracion';
    END IF;
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'user_profiles'
          AND column_name IN ('ai_consent_version', 'ai_consent_at', 'ai_consent_revoked_at',
                              'ai_cn_transfer_at', 'analytics_consent', 'ai_consent_paused_at')) <> 6 THEN
        RAISE EXCEPTION 'p1_plan_lote_843: faltan columnas del permiso en user_profiles';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'user_consents' AND column_name = 'origen'
                     AND is_nullable = 'NO') THEN
        RAISE EXCEPTION 'p1_plan_lote_843: falta user_consents.origen';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_consents_titular_chk') THEN
        RAISE EXCEPTION 'p1_plan_lote_843: falta el CHECK de un solo titular';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_consents_origen_titular_chk') THEN
        RAISE EXCEPTION 'p1_plan_lote_843: falta el CHECK del origen';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes
                   WHERE schemaname = 'public' AND indexname = 'user_consents_user_key_idx') THEN
        RAISE EXCEPTION 'p1_plan_lote_843: falta el indice por cuenta';
    END IF;
END $$;
