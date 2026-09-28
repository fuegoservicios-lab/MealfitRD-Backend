-- [P1-CATALOGO-RD-2026-09-28 · completar] Completa las 5 filas del paquete del 28-sep (con el OK del dueño, 28-sep tarde).
--
-- El gate del backend cazó en el catálogo VIVO que las 5 filas de `p1_catalogo_rd_polvos_soya_tostadas_tomatillo_2026_09_28`
-- nacieron incompletas, y lo cazaba en el gate de TODAS las sesiones:
--   1) test_p2_objective_v4_batch::test_catalog_micros_fully_populated — 4 celdas de micros en NULL (reabren
--      `estimado_bajo`, que el micro-closer salta a propósito).
--   2) test_p2_unpriced_keep_invariant (A y C) — filas RD sin precio: el agregador quita de la lista de la compra lo que
--      no tiene precio, EN SILENCIO; y desde el 10-sep (P1-DO-DESPENSA-DE-SU-MERCADO) RD no tiene filas sin precio.
--   3) test_p1_country_system_f2::test_c3_durable_guard_do_corpus_retarget_baseline (fase «country catalog live») — el
--      alias «tostadas de tortilla de maiz» no resolvía a su fila en ningún modo (con el sistema de países encendido lo
--      intercepta la preparación de «tortilla de maíz») y hacía el baseline DO dependiente del knob. Fuera ese alias.
--
-- Micros (USDA FoodData Central): «Soybeans, mature seeds, roasted, salted» (172440) y FNDDS «Soy nuts» (2707433) → 4,2 g
-- de azúcares; «Taco shells, baked» (172800) y FNDDS «Taco shell, corn» (2707826) → 1,5 g de azúcares, 0 vitamina D, 0 B12.
-- Precio: leche en polvo con el precio de referencia La Sirena 2026-07 de `supermarket_products` (Nestlé Ideal bolsa 800 g
-- RD$450; Milex Slim funda 1600 g RD$1.585). Soya tostada, tostadas de maíz y tomatillo NO están en el supermercado:
-- precio ESTIMADO, a ajustar por el dueño en /supermercado (tostadas ~370 g ≈ RD$250 → RD$306,48/lb y RD$8,31 la
-- tostada de 12,3 g; soya tostada 227 g ≈ RD$250 → RD$499,55/lb; tomatillo lata ~794 g ≈ RD$275 → RD$157,10/lb y
-- RD$11,78 la pieza de 34 g). Idempotente (cada UPDATE sólo actúa sobre lo que falta). Sync: migrations/ + backend/migrations/.

UPDATE public.master_ingredients SET sugars_g_per_100g = 4.2
 WHERE slug = 'soya-tostada' AND sugars_g_per_100g IS NULL;

UPDATE public.master_ingredients
   SET sugars_g_per_100g = COALESCE(sugars_g_per_100g, 1.5),
       vitamin_d_mcg_per_100g = COALESCE(vitamin_d_mcg_per_100g, 0),
       vitamin_b12_mcg_per_100g = COALESCE(vitamin_b12_mcg_per_100g, 0)
 WHERE slug = 'tostadas-de-maiz'
   AND (sugars_g_per_100g IS NULL OR vitamin_d_mcg_per_100g IS NULL OR vitamin_b12_mcg_per_100g IS NULL);

UPDATE public.master_ingredients SET price_per_lb = 255.15, price_per_unit = 450
 WHERE slug = 'leche-entera-en-polvo' AND COALESCE(price_per_lb, 0) = 0 AND COALESCE(price_per_unit, 0) = 0;
UPDATE public.master_ingredients SET price_per_lb = 449.34, price_per_unit = 1585
 WHERE slug = 'leche-descremada-en-polvo' AND COALESCE(price_per_lb, 0) = 0 AND COALESCE(price_per_unit, 0) = 0;
UPDATE public.master_ingredients SET price_per_lb = 499.55, price_per_unit = 250
 WHERE slug = 'soya-tostada' AND COALESCE(price_per_lb, 0) = 0 AND COALESCE(price_per_unit, 0) = 0;
UPDATE public.master_ingredients SET price_per_lb = 306.48, price_per_unit = 8.31
 WHERE slug = 'tostadas-de-maiz' AND COALESCE(price_per_lb, 0) = 0 AND COALESCE(price_per_unit, 0) = 0;
UPDATE public.master_ingredients SET price_per_lb = 157.10, price_per_unit = 11.78
 WHERE slug = 'tomatillo' AND COALESCE(price_per_lb, 0) = 0 AND COALESCE(price_per_unit, 0) = 0;

UPDATE public.master_ingredients SET aliases = array_remove(aliases, 'tostadas de tortilla de maiz')
 WHERE slug = 'tostadas-de-maiz' AND 'tostadas de tortilla de maiz' = ANY(aliases);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.master_ingredients
               WHERE slug IN ('leche-descremada-en-polvo', 'leche-entera-en-polvo', 'soya-tostada', 'tostadas-de-maiz', 'tomatillo')
                 AND (COALESCE(price_per_lb, 0) <= 0 AND COALESCE(price_per_unit, 0) <= 0
                      OR sugars_g_per_100g IS NULL OR vitamin_d_mcg_per_100g IS NULL OR vitamin_b12_mcg_per_100g IS NULL
                      OR 'tostadas de tortilla de maiz' = ANY(aliases))) THEN
        RAISE EXCEPTION '[P1-CATALOGO-RD-2026-09-28 · completar] sanity: alguna fila sigue incompleta';
    END IF;
END $$;
