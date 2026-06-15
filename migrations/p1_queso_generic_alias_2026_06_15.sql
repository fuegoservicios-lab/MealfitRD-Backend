-- [P1-QUESO-ALIAS · 2026-06-15] Alias genérico "queso" → "Queso blanco".
--
-- Contexto: el LLM emite con frecuencia "queso" genérico (o "lonjas/pedazos de queso", "X de queso")
-- sin especificar el tipo. El catálogo solo tenía aliases específicos ("queso blanco fresco",
-- "queso de freír", etc.) → el resolver de macros NO resolvía el genérico y caía a "0 silencioso"
-- (aporta 0 macros al solver) + se logueaba como unknown_ingredient (9 ocurrencias en prod 2026-06-14).
--
-- En RD el queso genérico de receta es queso blanco (fresco/de freír) por defecto → mapeamos "queso"
-- a "Queso blanco". SEGURIDAD: el resolver (nutrition_db._match_row) ordena los aliases por longitud
-- DESC y prueba match exacto primero, así que los quesos ESPECÍFICOS ("queso crema", "queso mozzarella",
-- "queso de hoja", "queso parmesano rallado") matchean su propia fila ANTES de caer al "queso" pelado.
-- Verificado con sonda offline (inyección de filas reales): genéricos → Queso blanco; específicos intactos.
--
-- Idempotente: solo agrega si no está ya presente.

UPDATE public.master_ingredients
SET aliases = array_append(aliases, 'queso')
WHERE name = 'Queso blanco'
  AND NOT ('queso' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- Sanity: tras la migración, "Queso blanco" DEBE tener el alias "queso".
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM public.master_ingredients
        WHERE name = 'Queso blanco' AND 'queso' = ANY(aliases)
    ) THEN
        RAISE EXCEPTION '[P1-QUESO-ALIAS] sanity falló: "Queso blanco" no tiene el alias "queso" tras la migración';
    END IF;
END $$;
