-- [P3-MICRONUTRIENTS · 2026-06-13] Columnas de micronutrientes por-100g en
-- master_ingredients. Extiende el cimiento del cerebro dividido (MDDA) del panel de
-- macros (P1-FOOD-DB-NUTRITION) al panel CLÍNICO completo: el validador determinista
-- (FS4) computa sodio/fibra/azúcar/vit D/calcio/hierro/B12/potasio del plan desde estos
-- valores y los compara vs los pisos/techos DRI/WHO, inyectando un reporte advisory
-- (con sugerencia de suplemento para los gaps estructurales como la vit D).
-- Todas NULLABLE → degradación grácil (sin micro → no se contabiliza ese ingrediente).
-- Idempotente (re-ejecutable). SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en
-- supabase/migrations/ Y backend/supabase/migrations/.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS vitamin_d_mcg_per_100g    numeric,
    ADD COLUMN IF NOT EXISTS calcium_mg_per_100g       numeric,
    ADD COLUMN IF NOT EXISTS iron_mg_per_100g          numeric,
    ADD COLUMN IF NOT EXISTS vitamin_b12_mcg_per_100g  numeric,
    ADD COLUMN IF NOT EXISTS sugars_g_per_100g         numeric,
    ADD COLUMN IF NOT EXISTS potassium_mg_per_100g     numeric;

COMMENT ON COLUMN public.master_ingredients.vitamin_d_mcg_per_100g IS
    '[P3-MICRONUTRIENTS] Vitamina D (D2+D3) en mcg por 100g comestibles (USDA "Vitamin D (D2 + D3)"). NULL si la fila USDA no la reporta — frecuente en alimentos enteros sin fortificar.';
COMMENT ON COLUMN public.master_ingredients.sugars_g_per_100g IS
    '[P3-MICRONUTRIENTS] Azúcares totales en g por 100g (USDA "Sugars, total including NLEA"). Proxy para el cómputo de azúcares libres del plan (miel/glaseados añadidos se cuentan aparte por el validador).';
COMMENT ON COLUMN public.master_ingredients.potassium_mg_per_100g IS
    '[P3-MICRONUTRIENTS] Potasio (K) en mg por 100g (USDA "Potassium, K"). Para el balance Na/K del reporte advisory.';

-- Sanity: idempotencia + presencia de columnas clave.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'vitamin_d_mcg_per_100g'
    ) THEN
        RAISE EXCEPTION 'P3-MICRONUTRIENTS: columna vitamin_d_mcg_per_100g no fue creada';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'potassium_mg_per_100g'
    ) THEN
        RAISE EXCEPTION 'P3-MICRONUTRIENTS: columna potassium_mg_per_100g no fue creada';
    END IF;
END $$;
