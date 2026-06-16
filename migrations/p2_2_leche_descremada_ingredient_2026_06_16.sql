-- [P2-2 · 2026-06-16] (gap-audit P2-2) Separa "Leche descremada" de "Leche" (entera 3.25%).
--
-- La fila Leche (fdc 171265 = whole 3.25%, satfat 1.865) conflaciona entera+descremada en aliases →
-- 'leche descremada' resolvía a la grasa/satfat de la ENTERA, y el swap de dislipidemia
-- 'leche entera'→descremada era un no-op clínico (ambas resolvían a la misma fila). Esta fila nueva
-- (macros USDA reales vía populate, 'milk nonfat fat free skim') habilita el swap real. 'leche deslactosada'
-- y 'leche líquida' (ambiguas) quedan en Leche a propósito.
--
-- Idempotente. Sync: migrations/ + backend/migrations/.

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'leche-descremada', 'Leche descremada', 'Lácteos', 'cartón',
       ARRAY['leche descremada','leche desnatada','leche 0% grasa','leche sin grasa',
             'leche descremada en polvo reconstituida']::text[],
       NULL, 244, 7, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Leche descremada');

UPDATE public.master_ingredients
SET aliases = array_remove(aliases, 'leche descremada')
WHERE name = 'Leche';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Leche descremada') THEN
        RAISE EXCEPTION '[P2-2] sanity: falta la fila Leche descremada';
    END IF;
    IF EXISTS (SELECT 1 FROM public.master_ingredients
               WHERE name = 'Leche' AND 'leche descremada' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P2-2] sanity: Leche aún tiene el alias leche descremada';
    END IF;
END $$;
