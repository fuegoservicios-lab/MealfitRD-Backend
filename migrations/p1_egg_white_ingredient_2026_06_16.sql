-- [P1-RESOLVER-COVERAGE · 2026-06-16 · follow-up] Separa "Clara de huevo" (clara) de "Huevo" (entero).
--
-- BUG de precisión pre-existente surfaceado al medir: "claras de huevo" era alias de Huevo (entero) →
-- 45 líneas de clara en comidas reales (31 distintas) resolvían a macros de huevo ENTERO (~143 kcal/100g,
-- 9.5 g grasa) cuando la clara es ~52 kcal/100g, 0.17 g grasa. Sobre-contaba ~2.7x kcal + grasa fantasma
-- en el panel de macros/coherencia/compra para un ingrediente STAPLE del desayuno dominicano (revoltillo/
-- tortilla de claras). El LLM computa el meal con macros de clara → el resolver discrepaba 2.7x.
--
-- Fix: fila propia "Clara de huevo" (macros USDA vía populate_nutrition_db.py) + se MUEVEN los aliases de
-- clara desde Huevo a la fila nueva. Routing limpio: "2 huevos"/"huevos enteros" → Huevo; "claras de
-- huevo"/"claras" → Clara de huevo. Tier1-exacto + longest-first garantizan que el entero no se afecta.
--
-- Idempotente: INSERT WHERE NOT EXISTS; array_remove (no-op si ya quitado). Sync: migrations/ + backend/migrations/.

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'clara-de-huevo', 'Clara de huevo', 'Proteínas', 'unidad',
       ARRAY['claras de huevo','clara de huevo','claras','clara de huevos']::text[],
       33, 243, 21, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Clara de huevo');

-- Quita los aliases de clara de Huevo (entero) para que el routing sea exclusivo.
UPDATE public.master_ingredients
SET aliases = array_remove(array_remove(aliases, 'claras de huevo'), 'clara de huevo')
WHERE name = 'Huevo';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Clara de huevo') THEN
        RAISE EXCEPTION '[P1-EGG-WHITE] sanity: falta la fila Clara de huevo';
    END IF;
    IF EXISTS (SELECT 1 FROM public.master_ingredients
               WHERE name = 'Huevo' AND 'claras de huevo' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P1-EGG-WHITE] sanity: Huevo aún tiene el alias claras de huevo';
    END IF;
END $$;
