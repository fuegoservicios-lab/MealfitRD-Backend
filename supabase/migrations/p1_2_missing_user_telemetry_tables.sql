-- [P1.2] Tablas y columna faltantes consumidas por cron_tasks/proactive_agent.
--
-- Estado previo (verificado vía information_schema antes de aplicar):
--   - public.weight_log               → NO existía (cron_tasks.py:7461,8077,8363)
--   - public.abandoned_meal_reasons   → NO existía (cron_tasks.py:7784,8395; routers/plans.py:821,1167,1522; proactive_agent.py:177)
--   - public.nudge_outcomes.response_sentiment → columna faltante (cron_tasks.py:7817,8111,8151,8377; proactive_agent.py:158-167)
--
-- Sin estas estructuras, los SELECTs caen en except y degradan SELF-EVALUATION,
-- METABOLISMO EVOLUTIVO y NUDGE SYNC sin levantar errores fatales (best-effort).
-- proactive_agent.py:164 incluso tenía un ALTER TABLE inline como parche en
-- runtime cuando detectaba la columna missing — esa rama queda obsoleta tras
-- esta migration.
--
-- Convenciones aplicadas (consistentes con el resto del proyecto):
--   - timestamps con `timestamp with time zone DEFAULT now()`
--   - RLS habilitado sin policies → solo service_role accede (backend cron),
--     mismo patrón que chunk_lesson_telemetry, api_usage, summary_archive.
--   - abandoned_meal_reasons usa SERIAL para alinearse con nudge_outcomes
--     (ambas son del subsistema proactive_agent y comparten convención).

-- =========================================================================
-- 1) weight_log
-- =========================================================================
CREATE TABLE IF NOT EXISTS public.weight_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL,
    weight      NUMERIC(6, 2) NOT NULL,
    unit        VARCHAR(8) NOT NULL DEFAULT 'lb',
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Soporta SELECT ... WHERE user_id = %s ORDER BY created_at [DESC] (snapshot reciente y serie histórica).
CREATE INDEX IF NOT EXISTS idx_weight_log_user_created_at
    ON public.weight_log (user_id, created_at DESC);

ALTER TABLE public.weight_log ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.weight_log IS
    '[P1.2] Historial de peso por usuario. Consumido por SELF-EVALUATION y METABOLISMO EVOLUTIVO en cron_tasks.';

-- =========================================================================
-- 2) abandoned_meal_reasons
-- =========================================================================
CREATE TABLE IF NOT EXISTS public.abandoned_meal_reasons (
    id          SERIAL PRIMARY KEY,
    user_id     UUID,
    meal_type   VARCHAR(50),
    reason      VARCHAR(50),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Soporta queries por usuario (cron_tasks.py:7784,8395) y agregaciones globales por motivo (routers/system.py:51).
CREATE INDEX IF NOT EXISTS idx_abandoned_meal_reasons_user_created_at
    ON public.abandoned_meal_reasons (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_abandoned_meal_reasons_reason
    ON public.abandoned_meal_reasons (reason);

ALTER TABLE public.abandoned_meal_reasons ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.abandoned_meal_reasons IS
    '[P1.2] Razones de abandono de comidas (insertadas desde routers/plans y proactive_agent). Consumida por causal-loop / FEEDBACK LOOP del cron.';

-- =========================================================================
-- 3) nudge_outcomes.response_sentiment
-- =========================================================================
ALTER TABLE public.nudge_outcomes
    ADD COLUMN IF NOT EXISTS response_sentiment VARCHAR(50);

-- Soporta filtros por sentiment + agregaciones (cron_tasks.py:8151, routers/system.py:59).
CREATE INDEX IF NOT EXISTS idx_nudge_outcomes_response_sentiment
    ON public.nudge_outcomes (response_sentiment)
    WHERE response_sentiment IS NOT NULL;

COMMENT ON COLUMN public.nudge_outcomes.response_sentiment IS
    '[P1.2] Sentimiento clasificado del usuario al responder un nudge (motivation/positive/neutral/negative/etc.).';
