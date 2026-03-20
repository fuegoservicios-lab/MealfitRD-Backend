-- ============================================================
-- Migración: Añadir columna checked_at a custom_shopping_items
-- Registra CUÁNDO se marcó un item como comprado
-- ============================================================

ALTER TABLE custom_shopping_items ADD COLUMN IF NOT EXISTS checked_at TIMESTAMPTZ DEFAULT NULL;
