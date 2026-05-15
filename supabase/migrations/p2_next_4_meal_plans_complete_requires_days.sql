-- [P2-NEXT-4 · 2026-05-11] CHECK constraint en meal_plans: si
-- `plan_data->>'generation_status' = 'complete'` entonces
-- `jsonb_array_length(plan_data->'days') > 0`. Cierra el modo de
-- corrupción detectado en el audit 2026-05-11 (plan 005c5a99: status
-- complete + days=0 vivió ~14h en prod hasta ser detectado).
--
-- Root cause: chunk worker T1 marcó status='complete' pero el merge
-- `plan_data.days = merged_days` se perdió (race / rollback parcial
-- intermedio entre estado del chunk y plan_data). El CHECK fuerza
-- atomicidad lógica: si el plan dice estar completo, DEBE tener al
-- menos un día renderizable.
--
-- Pre-requisito: plan 005c5a99 ya fue migrado a status='abandoned'
-- via P2-NEXT-4 backup audit + UPDATE (2026-05-11 18:46-18:47 UTC).
-- Ese es el único violador conocido. Verificación abajo bloquea si
-- aparece otro pre-deploy.
--
-- Idempotente: usa DROP CONSTRAINT IF EXISTS + ADD CONSTRAINT, así
-- una re-aplicación no falla.

-- Sanity check: ningún plan viola la nueva regla AHORA.
DO $$
DECLARE
  violators_count int;
BEGIN
  SELECT COUNT(*) INTO violators_count
  FROM meal_plans
  WHERE plan_data->>'generation_status' = 'complete'
    AND jsonb_array_length(COALESCE(plan_data->'days', '[]'::jsonb)) = 0;
  IF violators_count > 0 THEN
    RAISE EXCEPTION
      'P2-NEXT-4: % planes violan la nueva regla (status=complete + days=0). '
      'Migrarlos a status=abandoned ANTES de aplicar el CHECK. Plan 005c5a99 '
      'ya fue migrado el 2026-05-11; investigar si hay nuevos violators.',
      violators_count;
  END IF;
END $$;

-- DROP previo idempotente + ADD constraint.
ALTER TABLE public.meal_plans
  DROP CONSTRAINT IF EXISTS meal_plans_complete_requires_days;

ALTER TABLE public.meal_plans
  ADD CONSTRAINT meal_plans_complete_requires_days
  CHECK (
    plan_data IS NULL
    OR plan_data->>'generation_status' IS NULL
    OR plan_data->>'generation_status' != 'complete'
    OR jsonb_array_length(COALESCE(plan_data->'days', '[]'::jsonb)) > 0
  )
  NOT VALID;

-- VALIDATE en pasos separados — permite que filas legacy con violations
-- sean detectadas por el sanity check arriba sin abortar la migración.
-- Si el sanity pasó, VALIDATE es no-op.
ALTER TABLE public.meal_plans
  VALIDATE CONSTRAINT meal_plans_complete_requires_days;

COMMENT ON CONSTRAINT meal_plans_complete_requires_days ON public.meal_plans IS
'[P2-NEXT-4 · 2026-05-11] Si generation_status=complete, plan_data.days debe '
'tener al menos 1 día. Cierra el modo de corrupción del plan 005c5a99 donde '
'chunk T1 marcaba complete sin que el merge plan_data.days persistiera. '
'Si esta constraint falla en runtime, el bug está en el chunk worker — '
'investigar antes de relaxar el constraint.';
