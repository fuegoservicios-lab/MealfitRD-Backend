-- Ejecuta esto en el SQL Editor de tu Dashboard de Supabase.
-- Crea una función atómica para incrementar el contador de fricción de un ingrediente.
-- Evita la Race Condition del patrón read-modify-write en Python.
-- Si el contador llega a 3, lo resetea a 0 y retorna 3 para que Python auto-bloquee.

CREATE OR REPLACE FUNCTION increment_friction_rpc(
  p_user_id UUID,
  p_ingredient TEXT
)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
  current_val INT;
  new_val INT;
  hp JSONB;
BEGIN
  -- Leer health_profile con bloqueo de fila (FOR UPDATE) para evitar escrituras concurrentes
  SELECT health_profile INTO hp
  FROM user_profiles
  WHERE id = p_user_id
  FOR UPDATE;

  IF hp IS NULL THEN
    hp := '{}'::jsonb;
  END IF;

  -- Leer el valor actual del contador de fricción (default 0)
  current_val := COALESCE((hp -> 'frictions' ->> p_ingredient)::int, 0);
  new_val := current_val + 1;

  -- Si llega a 3 strikes, resetear a 0
  IF new_val >= 3 THEN
    new_val := 0;
  END IF;

  -- Asegurar que el objeto 'frictions' exista
  IF hp -> 'frictions' IS NULL THEN
    hp := jsonb_set(hp, ARRAY['frictions'], '{}'::jsonb);
  END IF;

  -- Escribir el nuevo valor atómicamente
  hp := jsonb_set(hp, ARRAY['frictions', p_ingredient], to_jsonb(new_val));

  UPDATE user_profiles
  SET health_profile = hp
  WHERE id = p_user_id;

  -- Retornar current_val + 1 (el valor ANTES del reset) para que Python sepa si alcanzó 3
  RETURN current_val + 1;
END;
$$;
