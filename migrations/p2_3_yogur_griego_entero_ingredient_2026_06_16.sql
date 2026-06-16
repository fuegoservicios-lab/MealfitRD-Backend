-- [P2-3 · 2026-06-16] (gap-audit P2-3) Separa "Yogur griego entero" del "Yogurt griego sin azúcar" (NONFAT).
--
-- La fila nonfat (fdc 330137, 0.37 g grasa/100g) tenía los aliases BARE 'yogur griego'/'yogurt griego'/
-- 'yogurt griego natural' → un plan con yogur ENTERO resolvía a la grasa del nonfat (subestima ~5 g/100g).
-- DECISIÓN: el bare 'yogur griego' se asigna al ENTERO (default cultural DR salvo que el texto diga
-- 'sin azúcar/0%/sin grasa') → no subestima grasa. Los aliases explícitos sin-azúcar/0% se quedan en el
-- nonfat. Macros USDA reales vía populate ('yogurt greek plain whole milk').
--
-- Idempotente. Sync: migrations/ + backend/migrations/.

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'yogurt-griego-entero', 'Yogurt griego entero', 'Lácteos', 'pote',
       ARRAY['yogur griego entero','yogurt griego entero','yogur entero','yogurt entero',
             'yogur natural entero','yogur griego natural entero',
             'yogur griego','yogurt griego','yogurt griego natural']::text[],
       NULL, 245, 14, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Yogurt griego entero');

-- Quita los aliases BARE/ambiguos del nonfat (quedan los explícitos sin-azúcar/0%/sin-grasa).
UPDATE public.master_ingredients
SET aliases = array_remove(array_remove(array_remove(aliases, 'yogur griego'), 'yogurt griego'), 'yogurt griego natural')
WHERE name = 'Yogurt griego sin azúcar';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Yogurt griego entero') THEN
        RAISE EXCEPTION '[P2-3] sanity: falta la fila Yogur griego entero';
    END IF;
    IF EXISTS (SELECT 1 FROM public.master_ingredients
               WHERE name = 'Yogurt griego sin azúcar' AND 'yogur griego' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P2-3] sanity: el nonfat aún tiene el alias bare yogur griego';
    END IF;
END $$;
