-- ============================================================
-- MIGRACIÓN: Columna nativa is_checked en custom_shopping_items
-- ============================================================
-- Reemplaza el hack de guardar is_checked dentro del JSON de item_name.
-- Ahora el estado de checkbox es una columna booleana nativa, lo que permite:
--   1. UPDATE atómico en 1 sola query (sin read-modify-write)
--   2. Cero race conditions
--   3. Separación limpia entre datos semánticos (item_name) y estado UI (is_checked)
--
-- INSTRUCCIONES: Ejecutar en Supabase SQL Editor.
-- ============================================================

-- 1. Añadir columna booleana con default FALSE
ALTER TABLE custom_shopping_items
ADD COLUMN IF NOT EXISTS is_checked BOOLEAN DEFAULT false;

-- 2. Migrar valores existentes: extraer is_checked del JSON embebido en item_name
-- Solo aplica a items cuyo item_name contiene un JSON válido con la key "is_checked"
UPDATE custom_shopping_items
SET is_checked = (item_name::jsonb ->> 'is_checked')::boolean
WHERE item_name LIKE '%"is_checked"%'
  AND item_name::jsonb ? 'is_checked';

-- 3. (Opcional) Limpiar is_checked del JSON para dejar item_name limpio
-- Descomentar si quieres remover is_checked del JSON embebido tras migrar:
-- UPDATE custom_shopping_items
-- SET item_name = (item_name::jsonb - 'is_checked')::text
-- WHERE item_name LIKE '%"is_checked"%'
--   AND item_name::jsonb ? 'is_checked';

-- ============================================================
-- COLUMNA source: Diferenciar origen de items sin parsear JSON
-- ============================================================
-- Elimina el full table scan O(N) en delete_auto_generated_shopping_items.
-- Valores: 'auto' (generados por IA), 'chat' (añadidos vía chat), 'manual' (legacy/default)

-- 4. Añadir columna source con default 'manual'
ALTER TABLE custom_shopping_items
ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual';

-- 5. Migrar items existentes: los que tienen JSON con 'category' son auto-generados
UPDATE custom_shopping_items
SET source = 'auto'
WHERE source = 'manual'
  AND item_name LIKE '{%'
  AND item_name::jsonb ? 'category';

-- 6. Índice para borrado rápido por source + user_id
CREATE INDEX IF NOT EXISTS idx_shopping_source_user
ON custom_shopping_items(user_id, source);
