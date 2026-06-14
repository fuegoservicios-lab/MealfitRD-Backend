-- [P1-FOOD-DB-NUTRITION · 2026-06-13] Columnas de macros nutricionales por-100g en
-- master_ingredients. Cimiento del "cerebro dividido" (MDDA): el solver determinista
-- computa los macros de cada comida desde estos valores en vez de que el LLM los
-- alucine. Todas NULLABLE → degradación grácil (sin macros → fallback estimado del
-- assembler). Idempotente (re-ejecutable sin error). SSOT dual-dir (P3-MIGRATIONS-SSOT):
-- este archivo vive en migrations/ Y backend/migrations/.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS kcal_per_100g          numeric,
    ADD COLUMN IF NOT EXISTS protein_g_per_100g     numeric,
    ADD COLUMN IF NOT EXISTS carbs_g_per_100g       numeric,
    ADD COLUMN IF NOT EXISTS fats_g_per_100g        numeric,
    ADD COLUMN IF NOT EXISTS fiber_g_per_100g       numeric,
    ADD COLUMN IF NOT EXISTS sodium_mg_per_100g     numeric,
    ADD COLUMN IF NOT EXISTS nutrition_source       text,
    ADD COLUMN IF NOT EXISTS nutrition_source_date  date,
    ADD COLUMN IF NOT EXISTS fdc_id                 bigint,
    ADD COLUMN IF NOT EXISTS is_dominican_cultivar  boolean DEFAULT false;

-- Enum de la fuente (idempotente: DROP antes de ADD).
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_nutrition_source_check;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_nutrition_source_check
    CHECK (nutrition_source IS NULL OR nutrition_source IN ('usda', 'off', 'faoinfoods', 'manual'));

COMMENT ON COLUMN public.master_ingredients.kcal_per_100g IS
    '[P1-FOOD-DB-NUTRITION] kcal por 100g comestibles. Por consistencia con el solver se usa Atwater 4/4/9 (= 4*protein + 4*carbs + 9*fats) cuando la fuente no trae Energy directo.';
COMMENT ON COLUMN public.master_ingredients.fdc_id IS
    '[P1-FOOD-DB-NUTRITION] FoodData Central id (USDA) de la fila origen — trazabilidad. NULL para fuente manual/INFOODS.';
COMMENT ON COLUMN public.master_ingredients.is_dominican_cultivar IS
    '[P1-FOOD-DB-NUTRITION] true para viandas/cultivares DD cuyos macros son estimación (no estudio publicado): yuca roja, casabe, yautía, etc. — pendiente de validación por nutricionista (nutrition_source=manual).';

-- Sanity: idempotencia + presencia de columnas clave.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'protein_g_per_100g'
    ) THEN
        RAISE EXCEPTION 'P1-FOOD-DB-NUTRITION: columna protein_g_per_100g no fue creada';
    END IF;
END $$;
