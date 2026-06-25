-- [P1-MICRONUTRIENT-CATALOG-BACKFILL · 2026-06-24]
-- Rellena los micronutrientes del PANEL (magnesio, fósforo, potasio, calcio, hierro, vit D,
-- B12, azúcares) que estaban en NULL en `master_ingredients`. Un NULL se computa como 0.0 en
-- `micronutrients.compute_plan_micronutrient_totals` → SUBESTIMA el total del panel y dispara
-- falsos "bajo" (caso real: magnesio reportado 350.9/420 mg porque Yogurt/Casabe/Ajo/Brócoli
-- —ingredientes de alta frecuencia— tenían magnesium_mg_per_100g = NULL → contaban 0 mg).
--
-- Valores per-100g anclados a USDA FoodData Central (mismo primary_source que data_provenance).
-- Idempotente: COALESCE solo rellena NULL (no pisa valores ya poblados). Excepción: 'Ajo' era un
-- STUB con k/ca/fe/vd/b12 = 0 (erróneo, no NULL) → corrección explícita a garlic raw (FDC 169230).
--
-- DATA-only (DML), NO DDL. Solo afecta planes NUEVOS / recálculo (los planes ya persistidos
-- guardan su micronutrient_report calculado al momento de generación).

BEGIN;

-- Ajo: stub con ceros erróneos → garlic raw USDA (FDC 169230).
UPDATE public.master_ingredients SET
  magnesium_mg_per_100g = 25, phosphorus_mg_per_100g = 153, potassium_mg_per_100g = 401,
  calcium_mg_per_100g = 181, iron_mg_per_100g = 1.7, sugars_g_per_100g = 1.0,
  vitamin_d_mcg_per_100g = 0, vitamin_b12_mcg_per_100g = 0
WHERE name = 'Ajo';

-- Alimentos enteros de alta frecuencia (magnesio/fósforo).
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,11), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,95)  WHERE name='Yogurt';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,11), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,137) WHERE name='Yogurt griego sin azúcar';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,21), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,66)  WHERE name='Brócoli';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,28), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,140) WHERE name='Atún en agua';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,12), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,15)  WHERE name='Melón';
UPDATE public.master_ingredients SET phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,642), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0), vitamin_b12_mcg_per_100g=COALESCE(vitamin_b12_mcg_per_100g,0) WHERE name='Linaza';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,12), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,27)  WHERE name='Casabe';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,12), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,27)  WHERE name='Casabe albahaca';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,48), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,106) WHERE name='Mostaza';
UPDATE public.master_ingredients SET phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,220) WHERE name='Jamón de pavo';

-- Embutidos / quesos / condimentos (valores USDA-aligned; uso en cantidades pequeñas).
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,15), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,180), potassium_mg_per_100g=COALESCE(potassium_mg_per_100g,250), calcium_mg_per_100g=COALESCE(calcium_mg_per_100g,12), sugars_g_per_100g=COALESCE(sugars_g_per_100g,0), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0) WHERE name='Longaniza dominicana';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,15), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,200), potassium_mg_per_100g=COALESCE(potassium_mg_per_100g,350), calcium_mg_per_100g=COALESCE(calcium_mg_per_100g,13), sugars_g_per_100g=COALESCE(sugars_g_per_100g,0), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0) WHERE name='Salami';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,20), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,400), potassium_mg_per_100g=COALESCE(potassium_mg_per_100g,90), calcium_mg_per_100g=COALESCE(calcium_mg_per_100g,700), sugars_g_per_100g=COALESCE(sugars_g_per_100g,2), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0) WHERE name='Queso de hoja';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,1), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,0), potassium_mg_per_100g=COALESCE(potassium_mg_per_100g,8), calcium_mg_per_100g=COALESCE(calcium_mg_per_100g,24), iron_mg_per_100g=COALESCE(iron_mg_per_100g,0.33), sugars_g_per_100g=COALESCE(sugars_g_per_100g,0), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0), vitamin_b12_mcg_per_100g=COALESCE(vitamin_b12_mcg_per_100g,0) WHERE name='Sal';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,0), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,0), potassium_mg_per_100g=COALESCE(potassium_mg_per_100g,2), calcium_mg_per_100g=COALESCE(calcium_mg_per_100g,6), iron_mg_per_100g=COALESCE(iron_mg_per_100g,0.03), sugars_g_per_100g=COALESCE(sugars_g_per_100g,0.04), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0), vitamin_b12_mcg_per_100g=COALESCE(vitamin_b12_mcg_per_100g,0) WHERE name='Vinagre blanco';
UPDATE public.master_ingredients SET magnesium_mg_per_100g=COALESCE(magnesium_mg_per_100g,7), phosphorus_mg_per_100g=COALESCE(phosphorus_mg_per_100g,19), vitamin_d_mcg_per_100g=COALESCE(vitamin_d_mcg_per_100g,0), vitamin_b12_mcg_per_100g=COALESCE(vitamin_b12_mcg_per_100g,0) WHERE name='Vinagre balsámico';

-- Sanity: tras el backfill, ninguna fila debe quedar con un micro del panel en NULL.
DO $$
DECLARE n_null integer;
BEGIN
  SELECT count(*) INTO n_null FROM public.master_ingredients
  WHERE magnesium_mg_per_100g IS NULL OR phosphorus_mg_per_100g IS NULL
     OR potassium_mg_per_100g IS NULL OR calcium_mg_per_100g IS NULL
     OR iron_mg_per_100g IS NULL OR vitamin_d_mcg_per_100g IS NULL
     OR vitamin_b12_mcg_per_100g IS NULL OR sugars_g_per_100g IS NULL
     OR sodium_mg_per_100g IS NULL OR fiber_g_per_100g IS NULL;
  IF n_null > 0 THEN
    RAISE EXCEPTION 'P1-MICRONUTRIENT-CATALOG-BACKFILL: % filas con micro de panel en NULL tras el backfill', n_null;
  END IF;
END $$;

COMMIT;
