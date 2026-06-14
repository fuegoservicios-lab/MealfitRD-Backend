-- [P4-UNIFIED-RESOLVER · 2026-06-14] Backfill CONSERVADOR de densidades para los ingredientes que no
-- tenían ninguna (density_g_per_cup/unit/container todas NULL) → mata el "0 silencioso" #2: sin densidad,
-- to_grams() retorna None para medidas de taza/unidad sin hint de gramos → el ingrediente aporta 0.
--
-- SOLO valores de alta confianza (pesos USDA por household measure):
--   • g_per_cup: harina (125g), espinaca cruda (30g), auyama cubo (116g), vainitas (100g), molondrón (100g).
--   • g_per_unit: pechuga de pollo (170g), filete de pescado (150g), jengibre pieza (15g).
--   • container: galletas de soda (paquete/sleeve ~200g).
-- INTENCIONALMENTE NO se tocan los embutidos/carnes vendidos por lb (bacalao, camarones, carne de res/
-- molida, cerdo, jamón de pavo, longaniza, salami): el `default_unit='lb'` ya convierte lb→g sin densidad,
-- y el peso por "lonja/unidad" es demasiado variable para curarlo sin riesgo (una densidad ERRADA es peor
-- que None — mete gramos/macros falsos). Idempotente (solo escribe donde está NULL). SSOT dual-dir.

UPDATE public.master_ingredients SET density_g_per_cup = 125 WHERE slug = 'harina-de-trigo' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 30  WHERE slug = 'espinacas'       AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 116 WHERE slug = 'auyama'          AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 100 WHERE slug = 'vainitas'        AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 100 WHERE slug = 'molondrones'     AND density_g_per_cup IS NULL;

UPDATE public.master_ingredients SET density_g_per_unit = 170 WHERE slug = 'pechuga-de-pollo' AND density_g_per_unit IS NULL;
UPDATE public.master_ingredients SET density_g_per_unit = 150 WHERE slug = 'filete-pescado'   AND density_g_per_unit IS NULL;
UPDATE public.master_ingredients SET density_g_per_unit = 15  WHERE slug = 'jengibre'         AND density_g_per_unit IS NULL;

UPDATE public.master_ingredients SET container_weight_g = 200 WHERE slug = 'galletas-de-soda' AND container_weight_g IS NULL;
