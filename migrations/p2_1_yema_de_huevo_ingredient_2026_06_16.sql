-- [P2-1 · 2026-06-16] (gap-audit P2-1) Separa "Yema de huevo" (yema sola) de "Huevo" (entero).
--
-- BUG de precisión: 'yema de huevo' era alias de Huevo (entero: 9.51 g grasa/100g, 372 mg colesterol).
-- La yema real es ~26 g grasa/100g y ~1085 mg colesterol/100g (~3x el entero) → subestimaba grasa y
-- colesterol del plato y del panel dislipidemia/clínico. Análogo del fix de la clara
-- (p1_egg_white_ingredient_2026_06_16). Macros USDA reales vía populate_nutrition_db.py ('egg yolk raw fresh').
--
-- Idempotente: INSERT WHERE NOT EXISTS; array_remove no-op si ya quitado. Sync: migrations/ + backend/migrations/.

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'yema-de-huevo', 'Yema de huevo', 'Proteínas', 'unidad',
       ARRAY['yema de huevo','yemas de huevo','yemas','yema de huevos']::text[],
       17, 243, 21, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Yema de huevo');

-- Quita el alias de yema de Huevo (entero) para que el routing sea exclusivo.
UPDATE public.master_ingredients
SET aliases = array_remove(aliases, 'yema de huevo')
WHERE name = 'Huevo';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Yema de huevo') THEN
        RAISE EXCEPTION '[P2-1] sanity: falta la fila Yema de huevo';
    END IF;
    IF EXISTS (SELECT 1 FROM public.master_ingredients
               WHERE name = 'Huevo' AND 'yema de huevo' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P2-1] sanity: Huevo aún tiene el alias yema de huevo';
    END IF;
END $$;
