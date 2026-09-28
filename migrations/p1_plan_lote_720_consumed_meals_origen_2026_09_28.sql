-- [P1-PLAN-LOTE-720 · 2026-09-28] De dónde vino cada comida del diario (spec 2026-09-28-ficha-plato-registrado).
--
-- `source`: el vocabulario del ledger de la Nevera (`inventory_consumption_events.source`) más los dos caminos que
-- el ledger no ve porque no descuentan: photo | manual | estimate | plan_meal | chat | repeat. SIN CHECK a propósito:
-- es una etiqueta de la ficha, y un CHECK convertiría un valor nuevo en un registro de comida fallido. El código
-- normaliza lo desconocido a NULL (`db_facts._origen_de_comida`).
-- `plan_ref`: las coordenadas de «Me lo comí» ({plan_id, day_index, meal_index}); NULL en el resto.
--
-- Relleno de las filas viejas desde el ledger: solo las que descontaron algo lo tienen (38 de 52 el 28-sep); el
-- resto se queda en NULL y la ficha no dice origen. Idempotente: solo toca `source IS NULL`.
ALTER TABLE public.consumed_meals ADD COLUMN IF NOT EXISTS source TEXT;
ALTER TABLE public.consumed_meals ADD COLUMN IF NOT EXISTS plan_ref JSONB;

UPDATE public.consumed_meals cm
   SET source = e.source
  FROM (
        SELECT DISTINCT ON (consumed_meal_id) consumed_meal_id, source
          FROM public.inventory_consumption_events
         WHERE consumed_meal_id IS NOT NULL
           AND source IN ('photo', 'manual', 'plan_meal', 'chat')
         ORDER BY consumed_meal_id, created_at
       ) e
 WHERE cm.id = e.consumed_meal_id
   AND cm.source IS NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'consumed_meals' AND column_name = 'source') THEN
        RAISE EXCEPTION 'p1_plan_lote_720: consumed_meals.source no existe tras la migracion';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'consumed_meals' AND column_name = 'plan_ref') THEN
        RAISE EXCEPTION 'p1_plan_lote_720: consumed_meals.plan_ref no existe tras la migracion';
    END IF;
END $$;
