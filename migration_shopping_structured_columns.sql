-- ============================================================
-- Migración: Añadir columnas estructuradas a custom_shopping_items
-- Permite queries, sorting y filtering por categoría a nivel DB
-- ============================================================

-- Paso 1: Añadir columnas nuevas (si no existen)
ALTER TABLE custom_shopping_items ADD COLUMN IF NOT EXISTS category TEXT DEFAULT '';
ALTER TABLE custom_shopping_items ADD COLUMN IF NOT EXISTS display_name TEXT DEFAULT '';
ALTER TABLE custom_shopping_items ADD COLUMN IF NOT EXISTS qty TEXT DEFAULT '';
ALTER TABLE custom_shopping_items ADD COLUMN IF NOT EXISTS emoji TEXT DEFAULT '';

-- Paso 2: Backfill — extraer datos del JSON en item_name a las nuevas columnas
-- Solo para filas donde item_name es JSON válido y display_name está vacío
UPDATE custom_shopping_items
SET
    category     = COALESCE((item_name::json ->> 'category'), ''),
    display_name = COALESCE((item_name::json ->> 'name'), ''),
    qty          = COALESCE((item_name::json ->> 'qty'), ''),
    emoji        = COALESCE((item_name::json ->> 'emoji'), '')
WHERE
    display_name = ''
    AND item_name LIKE '{%'
    AND item_name LIKE '%}';

-- Paso 3: Para items que son strings planos (no JSON), usar item_name como display_name
UPDATE custom_shopping_items
SET display_name = item_name
WHERE
    display_name = ''
    AND item_name NOT LIKE '{%';

-- Paso 4: Índice para ordenar por categoría eficientemente
CREATE INDEX IF NOT EXISTS idx_shopping_category_user
ON custom_shopping_items (user_id, category);
