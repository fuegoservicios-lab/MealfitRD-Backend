-- [P1-PLAN-LOTE-771 · 2026-09-28] Regalos de la cuenta (spec 2026-09-28-admin-cuentas-regalos-design §3.1): créditos
-- extra y planes de cortesía que el dueño da desde /admin → Cuentas. Lo pagado (user_profiles.plan_tier, que escribe
-- PayPal) NO se toca: esto se superpone al leer (backend/regalos_cuenta.py). Un regalo se revoca, nunca se borra; se
-- va con la cuenta (ON DELETE CASCADE) y el rastro de quién lo dio vive en admin_access_log, que no tiene FK.
-- Idempotente.
CREATE TABLE IF NOT EXISTS public.account_grants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    amount INTEGER,
    plan TEXT,
    starts_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ends_at TIMESTAMPTZ,
    reason TEXT NOT NULL,
    granted_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ,
    revoked_by UUID,
    revoke_reason TEXT
);

ALTER TABLE public.account_grants DROP CONSTRAINT IF EXISTS account_grants_kind_chk;
ALTER TABLE public.account_grants ADD CONSTRAINT account_grants_kind_chk
    CHECK (kind IN ('creditos_generacion', 'creditos_coach', 'plan'));

-- Un plan de cortesía jamás es 'admin'; los créditos siempre caducan.
ALTER TABLE public.account_grants DROP CONSTRAINT IF EXISTS account_grants_forma_chk;
ALTER TABLE public.account_grants ADD CONSTRAINT account_grants_forma_chk CHECK (
    (kind = 'plan' AND plan IN ('basic', 'plus', 'ultra') AND amount IS NULL)
    OR (kind <> 'plan' AND amount BETWEEN 1 AND 1000 AND plan IS NULL AND ends_at IS NOT NULL)
);

ALTER TABLE public.account_grants DROP CONSTRAINT IF EXISTS account_grants_ventana_chk;
ALTER TABLE public.account_grants ADD CONSTRAINT account_grants_ventana_chk
    CHECK (ends_at IS NULL OR ends_at > starts_at);

ALTER TABLE public.account_grants DROP CONSTRAINT IF EXISTS account_grants_motivo_chk;
ALTER TABLE public.account_grants ADD CONSTRAINT account_grants_motivo_chk
    CHECK (char_length(reason) BETWEEN 3 AND 300);

CREATE INDEX IF NOT EXISTS account_grants_user_vivos_idx
    ON public.account_grants (user_id) WHERE revoked_at IS NULL;

-- Una sola cortesía viva por persona: la nueva revoca la anterior en la misma transacción.
CREATE UNIQUE INDEX IF NOT EXISTS account_grants_una_cortesia_idx
    ON public.account_grants (user_id) WHERE kind = 'plan' AND revoked_at IS NULL;

REVOKE ALL ON public.account_grants FROM PUBLIC;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'account_grants') THEN
        RAISE EXCEPTION 'p1_plan_lote_771: account_grants no existe tras la migracion';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes
                   WHERE schemaname = 'public' AND indexname = 'account_grants_una_cortesia_idx') THEN
        RAISE EXCEPTION 'p1_plan_lote_771: falta el indice unico de la cortesia';
    END IF;
END $$;
