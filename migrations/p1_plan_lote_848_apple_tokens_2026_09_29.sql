-- [P1-PLAN-LOTE-848 · 2026-09-29] Sign in with Apple: el refresh token, CIFRADO, para poder revocarlo al borrar la
-- cuenta (App Review 5.1.1(v): la guía de borrado de cuentas de Apple pide revocar los tokens con `/auth/revoke`).
-- Auditoría App Store 2026-09-29, fila 3.2, §A.6.
--
-- Una fila por cuenta: el último `authorizationCode` canjeado gana (revocar un refresh token retira la autorización de
-- la app entera en Apple). `refresh_token_enc` es un token Fernet (clave `MEALFIT_TOKEN_ENC_KEY`, fuera de la base):
-- quien lea la tabla sin la clave no puede usarlo. Se va con la cuenta (ON DELETE CASCADE): el borrado revoca ANTES de
-- purgar (`backend/app.py`, `/api/account/delete`), y la purga arrastra la fila.
-- Lo escribe y lo lee solo `backend/apple_tokens.py`. Idempotente.
CREATE TABLE IF NOT EXISTS public.apple_signin_tokens (
    user_id UUID PRIMARY KEY REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    refresh_token_enc TEXT NOT NULL,
    client_id TEXT NOT NULL DEFAULT 'com.bioboros.app',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Un token Fernet es base64 urlsafe y empieza por «gAAAAA» (versión 0x80): así un refresh token EN CLARO no entra.
ALTER TABLE public.apple_signin_tokens DROP CONSTRAINT IF EXISTS apple_signin_tokens_cifrado_chk;
ALTER TABLE public.apple_signin_tokens ADD CONSTRAINT apple_signin_tokens_cifrado_chk
    CHECK (refresh_token_enc ~ '^gAAAAA[A-Za-z0-9_=-]+$' AND char_length(refresh_token_enc) <= 8192);

ALTER TABLE public.apple_signin_tokens DROP CONSTRAINT IF EXISTS apple_signin_tokens_client_id_chk;
ALTER TABLE public.apple_signin_tokens ADD CONSTRAINT apple_signin_tokens_client_id_chk
    CHECK (char_length(client_id) BETWEEN 1 AND 255);

REVOKE ALL ON public.apple_signin_tokens FROM PUBLIC;

COMMENT ON TABLE public.apple_signin_tokens IS
    'P1-PLAN-LOTE-848: refresh token de Sign in with Apple cifrado con Fernet (MEALFIT_TOKEN_ENC_KEY), para revocarlo en Apple al borrar la cuenta. Lo escribe y lo lee solo backend/apple_tokens.py.';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'apple_signin_tokens') THEN
        RAISE EXCEPTION 'p1_plan_lote_848: apple_signin_tokens no existe tras la migracion';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_class r ON r.oid = c.confrelid
        WHERE t.relname = 'apple_signin_tokens' AND r.relname = 'user_profiles'
          AND c.contype = 'f' AND c.confdeltype = 'c'
    ) THEN
        RAISE EXCEPTION 'p1_plan_lote_848: falta el FK en cascada a user_profiles';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'apple_signin_tokens_cifrado_chk') THEN
        RAISE EXCEPTION 'p1_plan_lote_848: falta el CHECK del token cifrado';
    END IF;
END $$;
