-- [P1-CULINARY-CONTRACT · 2026-07-31] Hueco de metadata cazado por el golden set
-- (test_capa1_cero_fp): el backfill por categoría dio a Lácteos ['ninguno','crudo']
-- y "Hierve la leche" es cocina básica — V1 daba falso positivo sobre planes buenos.
-- Alcance leche%: cierra la misma mina en las demás leches del catálogo, no solo la
-- fila que el examen pisó. Idempotente: array_append solo si falta.
UPDATE public.master_ingredients
SET prep_methods = array_append(prep_methods, 'hervir')
WHERE name ILIKE 'leche%'
  AND prep_methods IS NOT NULL
  AND NOT ('hervir' = ANY(prep_methods));

DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE name ILIKE 'leche%' AND prep_methods IS NOT NULL
      AND NOT ('hervir' = ANY(prep_methods));
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-CULINARY-CONTRACT] % fila(s) leche%% sin hervir tras el backfill', _n;
    END IF;
END $$;
