-- [P3-DENSITY-BACKFILL · 2026-07-01] (audit v2 micros GAP-7, batch P3-AUDIT-V2-RESIDUALS)
-- Un ingrediente medido en "taza"/"unidad" cuya fila master NO tiene density_g_per_cup NI
-- density_g_per_unit descarta TODOS sus micros silenciosamente (nutrition_db devuelve None →
-- WARN dedup, P1-VERIFIED-ONLY-OBSERVABILITY lo hizo observable pero no lo corrigió) → baja la
-- cobertura por-nutriente → `estimado_bajo` → el micro-closer salta ese micro.
-- Audit en vivo 2026-07-01: quedaban EXACTAMENTE 15 filas sin ninguna densidad. Backfill con
-- porciones típicas CURADAS (conservadoras, es-DO/USDA household measures):
--   Despensa: casabe 30 g/torta-porción; galleta de soda 7 g/unidad; sal 292 g/taza (USDA).
--   Quesos: lonja 28-30 g/unidad; rallado/cubos 113-132 g/taza (USDA shredded/cubed).
--   Proteínas: camarón 8 g/u; lonja jamón pavo 28 g; rebanada salami 25 g; longaniza 60 g/porción;
--   bistec/chuleta/porción de res/cerdo/bacalao 80-100 g; res molida 220 g/taza (USDA raw).
-- COALESCE = idempotente (no pisa valores existentes). DO-block: NINGUNA fila puede quedar con
-- AMBAS densidades NULL — contrato de datos para alimentos nuevos (siempre al menos una densidad,
-- o "1 unidad de X" del LLM descarta micros en silencio). Sync: migrations/ + backend/migrations/.

BEGIN;

UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 30) WHERE name = 'Casabe';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 30) WHERE name = 'Casabe albahaca';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 7) WHERE name = 'Galletas de soda';
UPDATE public.master_ingredients SET density_g_per_cup = COALESCE(density_g_per_cup, 292) WHERE name = 'Sal';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 30),
                                     density_g_per_cup = COALESCE(density_g_per_cup, 132) WHERE name = 'Queso blanco';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 30),
                                     density_g_per_cup = COALESCE(density_g_per_cup, 132) WHERE name = 'Queso de hoja';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 28),
                                     density_g_per_cup = COALESCE(density_g_per_cup, 113) WHERE name = 'Queso mozzarella';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 80) WHERE name = 'Bacalao';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 8) WHERE name = 'Camarones';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 100) WHERE name = 'Carne de res';
UPDATE public.master_ingredients SET density_g_per_cup = COALESCE(density_g_per_cup, 220),
                                     density_g_per_unit = COALESCE(density_g_per_unit, 100) WHERE name = 'Carne de res molida';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 100) WHERE name = 'Cerdo';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 28) WHERE name = 'Jamón de pavo';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 60) WHERE name = 'Longaniza dominicana';
UPDATE public.master_ingredients SET density_g_per_unit = COALESCE(density_g_per_unit, 25) WHERE name = 'Salami';

-- Sanity: cero filas con AMBAS densidades NULL (contrato de datos para altas futuras).
DO $$
DECLARE n_null integer;
BEGIN
  SELECT count(*) INTO n_null FROM public.master_ingredients
  WHERE density_g_per_cup IS NULL AND density_g_per_unit IS NULL;
  IF n_null > 0 THEN
    RAISE EXCEPTION 'P3-DENSITY-BACKFILL: % filas sin NINGUNA densidad — añadir density_g_per_unit o _per_cup (una mención "1 unidad de X" del LLM descartaría sus micros en silencio)', n_null;
  END IF;
END $$;

COMMIT;
