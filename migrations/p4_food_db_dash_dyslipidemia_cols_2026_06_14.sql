-- [P4-UNIFIED-RESOLVER · 2026-06-14] Columnas por-100g que faltaban para volver ENFORCED (con datos
-- reales, no solo prompt) dos condiciones del set Pareto cardiometabólico:
--   • magnesium_mg / potassium ya existía → DASH (hipertensión): el balance K/Mg/Ca del patrón DASH.
--   • saturated_fat_g → dislipidemia: el techo de grasa saturada (<7% de las kcal, AHA/ACC).
--   • phosphorus_mg → ERC (enfermedad renal): el fósforo es crítico y hoy no se modela.
--   • cholesterol_mg → dislipidemia: colesterol dietético (advisory).
-- Pobladas desde USDA FoodData Central (nutrient ids 1090/1091/1258/1253) por
-- scripts/populate_nutrition_db.py para los 97 ingredientes con fdc_id. Todas NULLABLE → degradación
-- grácil (sin dato → no se contabiliza ese ingrediente en el validador). Idempotente (re-ejecutable).
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS magnesium_mg_per_100g     numeric,
    ADD COLUMN IF NOT EXISTS phosphorus_mg_per_100g    numeric,
    ADD COLUMN IF NOT EXISTS saturated_fat_g_per_100g  numeric,
    ADD COLUMN IF NOT EXISTS cholesterol_mg_per_100g   numeric;

COMMENT ON COLUMN public.master_ingredients.magnesium_mg_per_100g IS
    '[P4-UNIFIED-RESOLVER] Magnesio (Mg) en mg por 100g (USDA "Magnesium, Mg" id 1090). Para el patrón DASH (hipertensión): Mg/K/Ca altos bajan la presión.';
COMMENT ON COLUMN public.master_ingredients.phosphorus_mg_per_100g IS
    '[P4-UNIFIED-RESOLVER] Fósforo (P) en mg por 100g (USDA "Phosphorus, P" id 1091). Crítico en enfermedad renal crónica (se modera junto con el potasio).';
COMMENT ON COLUMN public.master_ingredients.saturated_fat_g_per_100g IS
    '[P4-UNIFIED-RESOLVER] Grasa saturada en g por 100g (USDA "Fatty acids, total saturated" id 1258). Para el techo de dislipidemia (<7% de las kcal, AHA/ACC) — pasa de prompt-confiado a dato real.';
COMMENT ON COLUMN public.master_ingredients.cholesterol_mg_per_100g IS
    '[P4-UNIFIED-RESOLVER] Colesterol dietético en mg por 100g (USDA "Cholesterol" id 1253). Advisory para dislipidemia.';

-- Sanity: idempotencia + presencia de las columnas nuevas.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'saturated_fat_g_per_100g'
    ) THEN
        RAISE EXCEPTION 'P4-UNIFIED-RESOLVER: columna saturated_fat_g_per_100g no fue creada';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'magnesium_mg_per_100g'
    ) THEN
        RAISE EXCEPTION 'P4-UNIFIED-RESOLVER: columna magnesium_mg_per_100g no fue creada';
    END IF;
END $$;
