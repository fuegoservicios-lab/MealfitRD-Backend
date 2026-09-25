-- [P1-PLAN-LOTE-280 · 2026-09-25] Push NATIVA (Firebase Cloud Messaging) para la app de Android. SSOT: backend/fcm_push.py.
--
-- La app nativa no tiene Service Worker: `push_subscriptions` (Web Push) no le sirve. Cada teléfono registra aquí su
-- token de FCM al abrir la app; `utils_push.send_push_notification` (el cuello de botella de TODO aviso del servidor)
-- envía a las dos tablas. El token es único por instalación: si cambia de cuenta en el mismo teléfono, la fila pasa
-- a la cuenta nueva (UPSERT por token), así nunca le llegan avisos de la cuenta anterior. Idempotente.

CREATE TABLE IF NOT EXISTS public.device_push_tokens (
    token       TEXT PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    platform    TEXT NOT NULL DEFAULT 'android',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS device_push_tokens_user_idx ON public.device_push_tokens (user_id);

ALTER TABLE public.device_push_tokens DROP CONSTRAINT IF EXISTS device_push_tokens_platform_check;
ALTER TABLE public.device_push_tokens ADD CONSTRAINT device_push_tokens_platform_check
    CHECK (platform IN ('android', 'ios'));

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'device_push_tokens'
    ) THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-280] falta public.device_push_tokens tras la migración';
    END IF;
END $$;
