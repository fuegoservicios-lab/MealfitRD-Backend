-- [USER-INVENTORY-CATEGORY · 2026-05-18]
-- Añade columna `category` (nullable) a public.user_inventory para soportar
-- ítems personalizados (sin master_ingredient_id) creados manualmente desde
-- la nevera del frontend. La lectura del frontend agrupa por categoría con
-- prioridad: user_inventory.category → master_ingredients.category → "OTROS".
--
-- Semántica:
--   NULL  = item canónico con master_ingredient_id; el grupo se resuelve por
--           join contra master_ingredients.category (comportamiento legacy).
--   TEXT  = item personalizado donde el usuario eligió la categoría manualmente
--           durante el flujo "Añadir Alimento" → "Crear personalizado".
--
-- Nullable + sin default: rows previas quedan intactas; INSERT existentes
-- (chunk worker, log_consumed_meal, deduct, restock) tampoco requieren cambio
-- porque omitir la columna deja NULL.
--
-- IF NOT EXISTS para idempotencia (re-aplicar no debe romper).
ALTER TABLE public.user_inventory
ADD COLUMN IF NOT EXISTS category TEXT NULL;

COMMENT ON COLUMN public.user_inventory.category IS
'Categoría visual (Despensa/Vegetales/Proteínas/Frutas/Lácteos/Víveres/Otros) elegida por el usuario al añadir un ítem personalizado sin master_ingredient_id desde la nevera. NULL para items canónicos: el grupo se resuelve por master_ingredients.category vía join. Lectura del frontend prioriza este campo sobre el del master.';
