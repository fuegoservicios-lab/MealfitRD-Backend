-- [P2-NEW-5 · 2026-05-11] Tabla `meal_plans_audit` referenciada en
-- el SOP P3-AUDIT-6 de CLAUDE.md como "backup defensivo" pre-mutación
-- ante `plan_data_corrupted`. Audit 2026-05-11 detectó que la tabla
-- NO existía en producción — el SOP era inútil.
--
-- Diseño:
--   - Append-only: solo INSERT, no UPDATE ni DELETE de filas
--     (write-once log defensivo).
--   - Captura el `plan_data` ANTES de la mutación correctiva.
--   - Indexada por `meal_plan_id` para que el SRE pueda recuperar
--     el último estado pre-fix de un plan específico.
--   - Retención inicial sin GC — incidentes son raros. Si la tabla
--     crece a >10K rows, añadir cron de purga (>1y old).
--   - RLS habilitado + forced para que solo service_role escriba/lea
--     (operadores SRE acceden via dashboard server-side).

CREATE TABLE IF NOT EXISTS public.meal_plans_audit (
    id BIGSERIAL PRIMARY KEY,
    meal_plan_id UUID NOT NULL,
    user_id UUID,  -- nullable porque el plan puede haber sido borrado
    plan_data_before JSONB NOT NULL,
    action TEXT NOT NULL CHECK (action IN (
        'corruption_repair',
        'manual_rollback',
        'pre_delete_backup',
        'schema_migration'
    )),
    actor TEXT NOT NULL,  -- 'sre_manual', 'cron_<name>', 'endpoint_<name>'
    note TEXT,            -- libre, contexto del incidente
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meal_plans_audit_meal_plan_id
    ON public.meal_plans_audit (meal_plan_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_meal_plans_audit_user_id
    ON public.meal_plans_audit (user_id)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meal_plans_audit_action_created
    ON public.meal_plans_audit (action, created_at DESC);

COMMENT ON TABLE public.meal_plans_audit IS
    'P2-NEW-5 . 2026-05-11: backup defensivo append-only de plan_data '
    'pre-mutacion correctiva. Referenciado por SOP P3-AUDIT-6 en CLAUDE.md. '
    'Solo INSERT (write-once log). El SRE consulta para recuperar el '
    'estado de un plan antes de un fix manual / hotfix de corrupcion.';

COMMENT ON COLUMN public.meal_plans_audit.action IS
    'Tipo de mutacion que se hizo POST-backup. Enum cerrado para '
    'analytics estables (CHECK constraint). Si necesitas otro tipo, '
    'extender el CHECK con migracion explicita.';

COMMENT ON COLUMN public.meal_plans_audit.actor IS
    'Quien hizo la mutacion. Convencion: "sre_manual" (operador), '
    '"cron_<name>" (cron task), "endpoint_<name>" (handler API). '
    'Para forensics post-incident.';

-- RLS: la tabla solo es accesible via service_role. Las apps cliente
-- no necesitan leer/escribir aqui — es operacional puro.
ALTER TABLE public.meal_plans_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.meal_plans_audit FORCE ROW LEVEL SECURITY;

-- Sin policies: con RLS habilitado + ningun policy + FORCE, solo
-- bypass roles (service_role, supabase_admin) pueden leer/escribir.
-- Esto es deliberado.
