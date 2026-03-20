-- ============================================================
-- Migración: Añadir columna shopping_plan_hash a user_profiles
-- Para cache de auto-generación de lista de compras
-- ============================================================

ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS shopping_plan_hash TEXT DEFAULT NULL;
