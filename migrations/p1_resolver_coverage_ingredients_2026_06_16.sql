-- [P1-RESOLVER-COVERAGE · 2026-06-16] Cierra el gap de cobertura del resolver de macros medido
-- empíricamente contra 871 líneas de ingrediente DISTINTAS de planes generados reales (cobertura
-- previa 89.8% distintas / 80.6% por ocurrencia). El gap NO eran platos compuestos (0 aparecen como
-- línea) sino: (a) 2 alimentos atómicos comunes ausentes del catálogo de 105 — manzana (~9 ocurrencias)
-- y pepino (~2) — más granola y maní (1 c/u); (b) 2 aliases faltantes sobre filas existentes.
--
-- Esta migración SOLO crea las filas + aliases. Los MACROS de las 4 filas nuevas los puebla
-- populate_nutrition_db.py desde USDA FoodData Central (mismo patrón que las 105 existentes:
-- schema/seed aquí, nutrición vía USDA). Hasta poblarlas, kcal_per_100g IS NULL → el resolver
-- degrada grácil (lookup retorna None, no inventa). Las líneas reales de manzana/pepino traen hint
-- "(NNg)" → los gramos salen del hint del LLM; la densidad es respaldo.
--
-- SEGURIDAD de aliases: nutrition_db._match_row ordena aliases por longitud DESC y prueba match
-- EXACTO antes que word-boundary → "mantequilla de maní" (alias exacto de Mantequilla de maní) gana
-- ANTES de caer al "maní" pelado de la fila nueva; los quesos/tortillas específicos quedan intactos.
-- Verificado vía snapshot diff offline de las 871 líneas (ninguna ya-resuelta cambia sus macros).
--
-- Idempotente: INSERT con WHERE NOT EXISTS; alias array_append condicional. Sync: migrations/ + backend/migrations/.

-- ── 4 alimentos atómicos nuevos (macros NULL → populate_nutrition_db.py los llena desde USDA) ──
INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'manzana', 'Manzana', 'Frutas', 'unidad',
       ARRAY['manzanas','apple']::text[], 180, NULL, 21, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Manzana');

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'pepino', 'Pepino', 'Vegetales', 'unidad',
       ARRAY['pepinos','cohombro']::text[], 200, NULL, 10, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Pepino');

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'granola', 'Granola', 'Despensa', 'g',
       ARRAY['granola sin azucar']::text[], NULL, 120, 180, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Granola');

INSERT INTO public.master_ingredients (slug, name, category, default_unit, aliases,
        density_g_per_unit, density_g_per_cup, shelf_life_days, price_per_lb, price_per_unit)
SELECT 'mani', 'Maní', 'Despensa', 'g',
       ARRAY['mani','mani tostado','mani tostado sin sal','cacahuate','cacahuates','peanuts']::text[],
       NULL, 146, 365, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Maní');

-- ── 2 aliases sobre filas existentes (alimento idéntico/cercano, gramos vía hint del LLM) ──
-- "pescado fresco (mero o chillo)" no resolvía: "pescado fresco" no era alias y "pescado" pelado no mapea.
UPDATE public.master_ingredients
SET aliases = array_append(aliases, 'pescado fresco')
WHERE name = 'Filete de pescado blanco'
  AND NOT ('pescado fresco' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- "tortilla de harina de trigo (wrap, 60g)" → Tortilla integral. USDA: tortilla de harina blanca (~310
-- kcal/100g, por la grasa añadida) ≈ integral (~310) → macros equivalentes; la magnitud sale del hint en g.
UPDATE public.master_ingredients
SET aliases = aliases || ARRAY['tortilla de harina de trigo','tortilla de trigo','tortillas de harina de trigo']::text[]
WHERE name = 'Tortilla integral'
  AND NOT ('tortilla de harina de trigo' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- ── Sanity ──
DO $$
DECLARE missing text;
BEGIN
    FOREACH missing IN ARRAY ARRAY['Manzana','Pepino','Granola','Maní'] LOOP
        IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = missing) THEN
            RAISE EXCEPTION '[P1-RESOLVER-COVERAGE] sanity: falta la fila %', missing;
        END IF;
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients
                   WHERE name = 'Filete de pescado blanco' AND 'pescado fresco' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P1-RESOLVER-COVERAGE] sanity: alias "pescado fresco" no añadido';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients
                   WHERE name = 'Tortilla integral' AND 'tortilla de harina de trigo' = ANY(aliases)) THEN
        RAISE EXCEPTION '[P1-RESOLVER-COVERAGE] sanity: alias "tortilla de harina de trigo" no añadido';
    END IF;
END $$;
