-- Ejecuta esto en el SQL Editor de tu Dashboard de Supabase.
-- Crea una tabla para almacenar ingredientes que el LLM genera pero que 
-- el sistema de sinónimos no reconoce. Úsala para revisión periódica
-- y expansión del catálogo en constants.py.

CREATE TABLE IF NOT EXISTS unknown_ingredients (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  ingredient TEXT NOT NULL,
  raw_text TEXT,           -- Texto original del LLM antes de normalizar
  occurrences INT DEFAULT 1,
  first_seen TIMESTAMPTZ DEFAULT NOW(),
  last_seen TIMESTAMPTZ DEFAULT NOW(),
  reviewed BOOLEAN DEFAULT FALSE,
  UNIQUE(user_id, ingredient)
);

-- Índices para queries comunes
CREATE INDEX IF NOT EXISTS idx_unknown_ingredients_reviewed 
  ON unknown_ingredients(reviewed) WHERE reviewed = FALSE;
CREATE INDEX IF NOT EXISTS idx_unknown_ingredients_occurrences 
  ON unknown_ingredients(occurrences DESC);

-- RPC para upsert atómico (incrementar occurrences si ya existe)
CREATE OR REPLACE FUNCTION log_unknown_ingredient_rpc(
  p_user_id UUID,
  p_ingredient TEXT,
  p_raw_text TEXT DEFAULT NULL
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
  INSERT INTO unknown_ingredients (user_id, ingredient, raw_text)
  VALUES (p_user_id, p_ingredient, p_raw_text)
  ON CONFLICT (user_id, ingredient)
  DO UPDATE SET
    occurrences = unknown_ingredients.occurrences + 1,
    last_seen = NOW(),
    raw_text = COALESCE(EXCLUDED.raw_text, unknown_ingredients.raw_text);
END;
$$;
