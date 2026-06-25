-- [P1-FOOD-DB-EXTENDED-MICROS · 2026-06-25] Panel exhaustivo: 8 micronutrientes nuevos en
-- master_ingredients (zinc, folato DFE, vit A RAE, vit C, vit E, vit K, selenio, omega-3 ALA).
-- Convención de columnas idéntica a las existentes: <nutriente>_<unidad>_per_100g.
-- Todas NULLABLE → degradación honesta (lo no poblado sale 'estimado', nunca rompe el validador,
-- ver micronutrients.py P1-CEILING/FLOOR-COVERAGE-AWARE). Idempotente (ADD COLUMN IF NOT EXISTS).
-- Pobladas por scripts/backfill_extended_micros.py (USDA FDC por fdc_id, nutrient IDs inmutables).

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS zinc_mg_per_100g           numeric,
    ADD COLUMN IF NOT EXISTS folate_mcg_dfe_per_100g    numeric,
    ADD COLUMN IF NOT EXISTS vitamin_a_mcg_rae_per_100g numeric,
    ADD COLUMN IF NOT EXISTS vitamin_c_mg_per_100g      numeric,
    ADD COLUMN IF NOT EXISTS vitamin_e_mg_per_100g      numeric,
    ADD COLUMN IF NOT EXISTS vitamin_k_mcg_per_100g     numeric,
    ADD COLUMN IF NOT EXISTS selenium_mcg_per_100g      numeric,
    ADD COLUMN IF NOT EXISTS omega3_ala_g_per_100g      numeric;

COMMENT ON COLUMN public.master_ingredients.zinc_mg_per_100g           IS '[P1-FOOD-DB-EXTENDED-MICROS] Zinc (mg/100g). USDA "Zinc, Zn" id 1095. Inmunidad, síntesis proteica.';
COMMENT ON COLUMN public.master_ingredients.folate_mcg_dfe_per_100g    IS '[P1-FOOD-DB-EXTENDED-MICROS] Folato (mcg DFE/100g). USDA "Folate, DFE" id 1190 / "Folate, total" id 1177. Clave en embarazo.';
COMMENT ON COLUMN public.master_ingredients.vitamin_a_mcg_rae_per_100g IS '[P1-FOOD-DB-EXTENDED-MICROS] Vitamina A (mcg RAE/100g). USDA "Vitamin A, RAE" id 1106. Visión, inmunidad.';
COMMENT ON COLUMN public.master_ingredients.vitamin_c_mg_per_100g      IS '[P1-FOOD-DB-EXTENDED-MICROS] Vitamina C (mg/100g). USDA "Vitamin C, total ascorbic acid" id 1162. Absorción de hierro.';
COMMENT ON COLUMN public.master_ingredients.vitamin_e_mg_per_100g      IS '[P1-FOOD-DB-EXTENDED-MICROS] Vitamina E alfa-tocoferol (mg/100g). USDA id 1109. Antioxidante.';
COMMENT ON COLUMN public.master_ingredients.vitamin_k_mcg_per_100g     IS '[P1-FOOD-DB-EXTENDED-MICROS] Vitamina K filoquinona (mcg/100g). USDA id 1185. Coagulación; interacción warfarina.';
COMMENT ON COLUMN public.master_ingredients.selenium_mcg_per_100g      IS '[P1-FOOD-DB-EXTENDED-MICROS] Selenio (mcg/100g). USDA "Selenium, Se" id 1103. Tiroides, antioxidante.';
COMMENT ON COLUMN public.master_ingredients.omega3_ala_g_per_100g      IS '[P1-FOOD-DB-EXTENDED-MICROS] Omega-3 ALA (g/100g). USDA "PUFA 18:3 n-3 (ALA)" id 1404. Cardiovascular.';

DO $$
BEGIN
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_schema='public' AND table_name='master_ingredients'
          AND column_name IN ('zinc_mg_per_100g','folate_mcg_dfe_per_100g','vitamin_a_mcg_rae_per_100g',
                              'vitamin_c_mg_per_100g','vitamin_e_mg_per_100g','vitamin_k_mcg_per_100g',
                              'selenium_mcg_per_100g','omega3_ala_g_per_100g')) <> 8 THEN
        RAISE EXCEPTION 'P1-FOOD-DB-EXTENDED-MICROS: no se crearon las 8 columnas';
    END IF;
END $$;
