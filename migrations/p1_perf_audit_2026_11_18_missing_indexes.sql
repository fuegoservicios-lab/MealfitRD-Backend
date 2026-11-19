-- [P1-PERF-AUDIT-2026-11-18] Índices faltantes en tablas calientes + UNIQUE parcial
-- en user_profiles.paypal_subscription_id (cierre P0-BILLING-OWNERSHIP).
-- Idempotente (CREATE INDEX IF NOT EXISTS). Sincronizado en migrations/ y backend/migrations/.

CREATE INDEX IF NOT EXISTS idx_consumed_meals_user_consumed_at
    ON public.consumed_meals (user_id, consumed_at DESC);

CREATE INDEX IF NOT EXISTS idx_consumed_meals_user_created_at
    ON public.consumed_meals (user_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS ux_user_profiles_paypal_subscription_id
    ON public.user_profiles (paypal_subscription_id)
    WHERE paypal_subscription_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_meal_plan_week
    ON public.plan_chunk_queue (meal_plan_id, week_number);

CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_user_id
    ON public.plan_chunk_queue (user_id);

CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_paused_action
    ON public.plan_chunk_queue (meal_plan_id)
    WHERE status = 'pending_user_action';

CREATE INDEX IF NOT EXISTS idx_llm_usage_events_node_created_at
    ON public.llm_usage_events (node, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_events_plan_id
    ON public.llm_usage_events (plan_id)
    WHERE plan_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_plan_jobs_plan_type_created
    ON public.plan_jobs (plan_id, job_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_plan_jobs_user_id
    ON public.plan_jobs (user_id);

CREATE INDEX IF NOT EXISTS idx_user_inventory_reserved
    ON public.user_inventory (user_id)
    WHERE reserved_quantity > 0;

CREATE INDEX IF NOT EXISTS idx_agent_messages_user_role_created
    ON public.agent_messages (user_id, role, created_at DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_system_alerts_triggered_at
    ON public.system_alerts (triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user_id
    ON public.push_subscriptions (user_id);
