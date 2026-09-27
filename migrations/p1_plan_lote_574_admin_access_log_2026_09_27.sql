-- [P1-PLAN-LOTE-574 · 2026-09-27] Rastro del panel de administración (spec 2026-09-27-panel-admin-design §1): quién
-- abrió qué y cuándo. Sin FK a user_profiles a propósito: el rastro sobrevive a la cuenta. Idempotente.
CREATE TABLE IF NOT EXISTS public.admin_access_log (
    id BIGSERIAL PRIMARY KEY,
    admin_user_id UUID NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS admin_access_log_at_idx ON public.admin_access_log (at DESC);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'admin_access_log') THEN
        RAISE EXCEPTION 'p1_plan_lote_574: admin_access_log no existe tras la migracion';
    END IF;
END $$;
