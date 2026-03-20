-- Ejecuta esto en el SQL Editor de tu Dashboard de Supabase
-- Esto crea una función (Stored Procedure) para incrementar los conteos de forma atómica.
-- Evita que dos peticiones simultáneas sobreescriban el valor incorrectamente.

CREATE OR REPLACE FUNCTION increment_ingredient_frequencies_rpc(
  p_user_id UUID,
  p_ingredients TEXT[],
  p_counts INT[]
) 
RETURNS VOID 
LANGUAGE plpgsql
AS $$
DECLARE
  i INT;
BEGIN
  -- Verificar que los arreglos tengan el mismo tamaño por seguridad
  IF array_length(p_ingredients, 1) != array_length(p_counts, 1) THEN
    RAISE EXCEPTION 'Mismatched array lengths';
  END IF;

  FOR i IN 1..array_length(p_ingredients, 1) LOOP
    INSERT INTO ingredient_frequencies (user_id, ingredient, count, last_used)
    VALUES (p_user_id, p_ingredients[i], p_counts[i], NOW())
    ON CONFLICT (user_id, ingredient) 
    DO UPDATE SET 
      count = ingredient_frequencies.count + EXCLUDED.count,
      last_used = EXCLUDED.last_used;
  END LOOP;
END;
$$;
