-- Additive migration: historic usage and gifts retain their web scope.
ALTER TABLE public.api_usage ADD COLUMN IF NOT EXISTS usage_scope text NOT NULL DEFAULT 'web';
ALTER TABLE public.account_grants ADD COLUMN IF NOT EXISTS usage_scope text NOT NULL DEFAULT 'web';
ALTER TABLE public.meal_plans ADD COLUMN IF NOT EXISTS usage_scope text NOT NULL DEFAULT 'web';
CREATE INDEX IF NOT EXISTS api_usage_scope_meter ON public.api_usage(user_id, usage_scope, created_at);
CREATE INDEX IF NOT EXISTS account_grants_scope_user ON public.account_grants(user_id, usage_scope);
