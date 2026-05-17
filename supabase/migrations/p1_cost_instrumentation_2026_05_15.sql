-- [P1-COST-INSTRUMENTATION · 2026-05-15] Tabla `llm_usage_events` para
-- instrumentar costo real por llamada LLM (tokens + cost_usd_micros) que
-- `api_usage` no captura (solo cuenta invocaciones para el paywall mensual).
--
-- Motivación: pre-fix no había forma de responder "¿cuánto cuesta generar
-- un plan?" desde SQL. Audit 2026-05-15 estimó ~$0.06-$0.15/plan pero
-- estaba a ciegas — sin observabilidad no se pueden tomar decisiones
-- de optimización (context caching, retry budgets, model swaps).
--
-- Separación con `api_usage`:
--   - `api_usage`: count-based, gates paywall (gratis=15/basic=50/plus=200).
--     INTACTA, este P-fix no la toca.
--   - `llm_usage_events`: granular per-LLM-call, financial accounting.
--     Emitida desde `_safe_ainvoke` (graph_orchestrator.py) cuando el
--     response trae `usage_metadata`. Best-effort; fallo no rompe la
--     llamada LLM.
--
-- Queries esperadas (SRE post-deploy):
--   - SELECT model, SUM(cost_usd_micros)/1e6 AS usd, COUNT(*) AS calls
--     FROM llm_usage_events WHERE created_at > NOW() - INTERVAL '7 days'
--     GROUP BY model ORDER BY usd DESC;
--   - SELECT date_trunc('day', created_at), SUM(cost_usd_micros)/1e6
--     FROM llm_usage_events GROUP BY 1 ORDER BY 1 DESC LIMIT 30;
--
-- Idempotente: CREATE TABLE IF NOT EXISTS + ADD COLUMN IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS public.llm_usage_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NULL,
  plan_id uuid NULL,
  model text NOT NULL,
  node text NULL,
  input_tokens int NULL,
  output_tokens int NULL,
  cached_tokens int NULL,
  cost_usd_micros bigint NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- FK opcional a auth.users — ON DELETE SET NULL preserva la fila para
-- accounting agregado tras account deletion (GDPR-compatible: PII removida,
-- agregado de costo persiste).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'llm_usage_events_user_id_fkey'
  ) THEN
    ALTER TABLE public.llm_usage_events
      ADD CONSTRAINT llm_usage_events_user_id_fkey
      FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL;
  END IF;
END $$;

-- Índices: queries esperadas son time-window + agrupado por model/node/user.
CREATE INDEX IF NOT EXISTS idx_llm_usage_events_created_at
  ON public.llm_usage_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_events_model_created
  ON public.llm_usage_events (model, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_events_user_id
  ON public.llm_usage_events (user_id, created_at DESC)
  WHERE user_id IS NOT NULL;

-- RLS: tabla operacional, solo service_role. Frontend NO debe leer ni
-- escribir directamente (mismo patrón que meal_plans_audit / pipeline_metrics).
ALTER TABLE public.llm_usage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.llm_usage_events FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all ON public.llm_usage_events;
CREATE POLICY service_role_all ON public.llm_usage_events
  FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMENT ON TABLE public.llm_usage_events IS
'[P1-COST-INSTRUMENTATION · 2026-05-15] Financial accounting per-LLM-call. '
'Emitida por _safe_ainvoke (graph_orchestrator.py) tras éxito; best-effort. '
'Separada de api_usage (que es count-based para paywall). RLS: service_role only.';

COMMENT ON COLUMN public.llm_usage_events.cost_usd_micros IS
'USD * 1e6 (micros). 1.25e6 = $1.25. Calculado en backend via '
'compute_gemini_cost_micros() con pricing dict + knob MEALFIT_GEMINI_PRICING_JSON.';

COMMENT ON COLUMN public.llm_usage_events.cached_tokens IS
'Tokens servidos desde Gemini Context Caching (cuesta ~25% del input price). '
'Útil para medir ROI de cache habilitación post P1-COST-INSTRUMENTATION.';
