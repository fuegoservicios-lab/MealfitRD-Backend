-- [P3-MULTIPLIER-DB-CHECK · 2026-05-14] CHECK constraint en meal_plans:
-- si `plan_data->>'calc_household_multiplier'` está presente, debe ser
-- numérico en el rango [1, 100]. Defense-in-depth a nivel BD análogo
-- al I8 (`meal_plans_complete_requires_days`, P2-NEXT-4).
--
-- Motivación (audit 2026-05-14):
--   El clamp `[1, MEALFIT_MAX_HOUSEHOLD_SIZE=20]` vive en
--   `compute_household_multiplier` ([constants.py:2136](backend/constants.py))
--   y se aplica también inline en `/recalculate-shopping-list`
--   (P3-PDF-POLISH-4-B-RECALC). Pero el enforcement es PURAMENTE en
--   código — una regresión en uno de los callsites (alguien añadiendo
--   un 4to write path sin pasar por el helper SSOT, o un knob mal
--   calibrado que sube el cap a 999) persistiría valores absurdos en
--   `plan_data.calc_household_multiplier` sin que la DB lo bloquee.
--
--   Esta migración es la "última red de seguridad": defense-in-depth
--   a nivel persistencia. Análoga al I8 que enforza `complete →
--   days > 0` independientemente del código del chunk worker.
--
-- Rango [1, 100]:
--   - Lower bound 1: helper ya clampa a >=1.0 (planner requiere >=1
--     persona). 0 o negativo viola el contrato del aggregator.
--   - Upper bound 100: 5× sobre el cap default del knob (20). Permite
--     que el knob `MEALFIT_MAX_HOUSEHOLD_SIZE` se ajuste hasta 100 sin
--     que la BD lo bloquee. Sobre 100 viola el contrato del aggregator
--     (memory/perf concerns en `get_shopping_list_delta`).
--
-- NULL-friendly:
--   Planes legacy pre-P1-3 (sin la key `calc_household_multiplier`)
--   pasan el check porque `plan_data->>'calc_household_multiplier'`
--   es NULL para JSONB que no tiene la key. Los planes nuevos siempre
--   la persisten via `update_plan_data_atomic` callback de
--   /recalculate-shopping-list (P1-RECALC-LOSTUPDATE).
--
-- Idempotente: DROP CONSTRAINT IF EXISTS + ADD CONSTRAINT (mismo patrón
-- que P2-NEXT-4).
--
-- Aplicación staged: ADD ... NOT VALID (barato, no scan) + VALIDATE
-- separado (scan completo), igual que P2-NEXT-4. El sanity check
-- pre-deploy bloquea si hay violators existentes.

-- Sanity check: ningún plan viola la nueva regla AHORA.
DO $$
DECLARE
  violators_count int;
BEGIN
  SELECT COUNT(*) INTO violators_count
  FROM meal_plans
  WHERE plan_data ? 'calc_household_multiplier'
    AND (
      -- Cast falla con exception si el value no es castable a numeric.
      -- Capturamos via try/regex también, defense-in-depth.
      (plan_data->>'calc_household_multiplier') !~ '^-?\d+(\.\d+)?$'
      OR (plan_data->>'calc_household_multiplier')::numeric < 1
      OR (plan_data->>'calc_household_multiplier')::numeric > 100
    );
  IF violators_count > 0 THEN
    RAISE EXCEPTION
      'P3-MULTIPLIER-DB-CHECK: % planes violan la nueva regla '
      '(calc_household_multiplier fuera de [1, 100] o no-numérico). '
      'Investigar root cause: regresión del clamp en compute_household_multiplier '
      '(constants.py:2113), knob MEALFIT_MAX_HOUSEHOLD_SIZE mal calibrado, '
      'o write path nuevo que bypassea el helper SSOT.',
      violators_count;
  END IF;
END $$;

-- DROP previo idempotente + ADD constraint.
ALTER TABLE public.meal_plans
  DROP CONSTRAINT IF EXISTS meal_plans_calc_household_multiplier_range;

ALTER TABLE public.meal_plans
  ADD CONSTRAINT meal_plans_calc_household_multiplier_range
  CHECK (
    plan_data IS NULL
    OR NOT (plan_data ? 'calc_household_multiplier')
    OR (
      (plan_data->>'calc_household_multiplier') ~ '^-?\d+(\.\d+)?$'
      AND (plan_data->>'calc_household_multiplier')::numeric BETWEEN 1 AND 100
    )
  )
  NOT VALID;

-- VALIDATE en paso separado — el sanity check arriba ya garantizó cero
-- violators, así que el VALIDATE es no-op pero confirma la garantía
-- transaccionalmente.
ALTER TABLE public.meal_plans
  VALIDATE CONSTRAINT meal_plans_calc_household_multiplier_range;

COMMENT ON CONSTRAINT meal_plans_calc_household_multiplier_range ON public.meal_plans IS
'[P3-MULTIPLIER-DB-CHECK · 2026-05-14] Si plan_data tiene la key '
'calc_household_multiplier, su valor debe ser numérico en [1, 100]. '
'Defense-in-depth análoga al I8 (P2-NEXT-4): cierra el modo de '
'corrupción donde una regresión del clamp `compute_household_multiplier` '
'(constants.py) o un knob MEALFIT_MAX_HOUSEHOLD_SIZE mal calibrado '
'persistiría multipliers fuera del rango sin que la BD lo bloquee. '
'Si esta constraint falla en runtime, el bug está aguas arriba en el '
'helper SSOT — investigar ANTES de relaxar el constraint.';
