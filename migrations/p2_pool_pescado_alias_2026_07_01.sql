-- [P2-POOL-PRICE-CONTRACT · 2026-07-01] (audit v2 creatividad GAP-3/GAP-4, batch P2-AUDIT-V2-BATCH)
-- El item 'Pescado' del pool DOMINICAN_PROTEINS era el ÚNICO de los 142 items de los 4 pools del
-- planner que NO resolvía a una fila priced de master_ingredients (verificación en vivo
-- scripts/check_pool_prices.py, 2026-07-01): 'pescado fresco' ya era alias de 'Filete de pescado
-- blanco' (p1_resolver_coverage_ingredients_2026_06_16) pero 'pescado' pelado no mapeaba → si el
-- LLM escribía "150g de pescado" el ingrediente se DROPEABA de la lista por VERIFIED-ONLY en
-- silencio. Alias word-boundary 'pescado' → 'Filete de pescado blanco' (el genérico de pescado del
-- catálogo, mapping culinario correcto para compras/precio; los específicos mero/tilapia/salmón
-- tienen fila propia y ganan por match de nombre antes que el alias).
-- Idempotente (array_append condicional). Sync: migrations/ + backend/migrations/.

BEGIN;

UPDATE public.master_ingredients
SET aliases = array_append(aliases, 'pescado')
WHERE name = 'Filete de pescado blanco'
  AND NOT ('pescado' = ANY(COALESCE(aliases, ARRAY[]::text[])));

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients
                   WHERE name = 'Filete de pescado blanco' AND 'pescado' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P2-POOL-PRICE-CONTRACT] sanity: alias "pescado" no añadido';
    END IF;
END $$;

COMMIT;
