-- Nullable composition: absent evidence is never a manufactured zero.
BEGIN;
ALTER TABLE master_ingredients ADD COLUMN IF NOT EXISTS vitamin_b6_mg_per_100g DOUBLE PRECISION;
ALTER TABLE master_ingredients ADD COLUMN IF NOT EXISTS iodine_mcg_per_100g DOUBLE PRECISION;
ALTER TABLE master_ingredients ADD COLUMN IF NOT EXISTS micronutrient_provenance JSONB NOT NULL DEFAULT '{}'::jsonb;
-- Retire the historical zero-fill invariant for nutrients that may be unreported.
ALTER TABLE master_ingredients ALTER COLUMN magnesium_mg_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN zinc_mg_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN vitamin_b12_mcg_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN folate_mcg_dfe_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN vitamin_e_mg_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN vitamin_k_mcg_per_100g DROP NOT NULL;
ALTER TABLE master_ingredients ALTER COLUMN selenium_mcg_per_100g DROP NOT NULL;
COMMENT ON COLUMN master_ingredients.folate_mcg_dfe_per_100g IS 'Micrograms dietary folate equivalents per 100g; total folate alone is not interchangeable for fortified foods.';
COMMENT ON COLUMN master_ingredients.micronutrient_provenance IS 'Per nutrient source, food identifier, retrieval date and composition version. Missing metadata identifies legacy composition.';
COMMIT;
