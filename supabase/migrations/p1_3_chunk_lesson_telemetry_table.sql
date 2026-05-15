-- [P1.3] Tabla de telemetría para resolución de lecciones por chunk.
--
-- Sin esta tabla, `cron_tasks._record_chunk_lesson_telemetry` (cron_tasks.py:9478)
-- falla en cada INSERT con UndefinedTable, contaminando logs y bloqueando la
-- alerta de "synth ratio alto" (cron_tasks.py:9758) que detecta cuando el
-- aprendizaje continuo está degradado silenciosamente — el síntoma es que
-- plan_chunk_queue.learning_metrics está NULL para chunks completados, así que
-- el chunk N+1 recibe contadores en cero y los platos no responden a las
-- repeticiones del N.
--
-- Eventos canónicos (controlados por el código, no por CHECK constraint para no
-- bloquear roll-outs futuros que añadan eventos):
--   - 'lesson_synthesized_low_confidence': fallback a synthesize_last_chunk_learning
--   - 'recent_lessons_partial_synthesis': ventana rolling regenerada desde plan_data.days
--
-- No se añaden FKs a meal_plans/users: la telemetría es append-only y debe
-- sobrevivir a la eliminación de planes/usuarios para análisis histórico.

CREATE TABLE IF NOT EXISTS public.chunk_lesson_telemetry (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL,
    meal_plan_id      UUID NOT NULL,
    week_number       INTEGER NOT NULL,
    event             TEXT NOT NULL,
    synthesized_count INTEGER NOT NULL DEFAULT 0,
    queue_count       INTEGER NOT NULL DEFAULT 0,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Soporta la query de alerta cron_tasks.py:9758 (filtro por event + ventana temporal).
CREATE INDEX IF NOT EXISTS idx_chunk_lesson_telemetry_event_created_at
    ON public.chunk_lesson_telemetry (event, created_at DESC);

-- Debug por plan/chunk (admin / postmortem).
CREATE INDEX IF NOT EXISTS idx_chunk_lesson_telemetry_plan_week
    ON public.chunk_lesson_telemetry (meal_plan_id, week_number);

-- Vista por usuario (admin / soporte).
CREATE INDEX IF NOT EXISTS idx_chunk_lesson_telemetry_user_created_at
    ON public.chunk_lesson_telemetry (user_id, created_at DESC);

-- Telemetría server-side: solo service_role lee/escribe. Sin políticas para
-- anon/authenticated, con RLS habilitado, los clientes no pueden tocarla.
ALTER TABLE public.chunk_lesson_telemetry ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.chunk_lesson_telemetry IS
    '[P0-A/P1.3] Telemetría de resolución de lecciones del aprendizaje continuo entre chunks. Append-only, server-side.';
