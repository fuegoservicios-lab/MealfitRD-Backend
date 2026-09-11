-- [P1-DO-DESPENSA-DE-SU-MERCADO · 2026-09-10 · envases] Los cuatro productos que el dueño trajo de
-- su supermercado se venden en ENVASES, y la primera migración (p1_do_despensa_de_su_mercado)
-- sólo les puso `price_per_lb`. Medido con el agregador real contra el catálogo de producción:
--   «1 sobre de sazón con culantro y achiote» → «¼ lb · alcanza ~1 de 7 días», RD$280,66: el
--      sobre caía al peso por defecto de la categoría Despensa (450 g) y lo recortaba el tope de
--      especias;
--   «1 taza de azúcar morena»  → «¼ lb», RD$8,88: sin densidad de taza;
--   «1 taza de pan rallado»    → «5 Uds. (~150g total)» y SIN coste: sin densidad de taza, la
--      palabra «pan» le prestaba el peso de una unidad de pan (30 g);
--   «1 pote de hummus»         → «1 lb», RD$478,40: el pote de sus capturas es de 10 oz, RD$299.
-- Lo que se compra es lo que muestran sus capturas: caja de 8 sobres (1,41 oz) a RD$99, funda
-- Wala de 2 lb a RD$71 o de 5 lb a RD$157, paquete Buenhorno de 1 lb a RD$73 y pote Dietz &
-- Watson de 10 oz a RD$299.
--
-- `price_per_lb` NO se toca: el coste por ración del plato lo lee por gramo y ya es correcto.
-- Densidades de taza: porción «1 cup» de USDA SR Legacy, la misma referencia de sus valores
-- nutricionales (fdc 168833, azúcar morena compacta: 220 g; fdc 174928, pan rallado seco: 108 g).
-- El sobre del sazón: 1,41 oz = 40 g entre 8 sobres = 5 g.
--
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.

UPDATE public.master_ingredients SET
    market_container   = 'caja',
    container_weight_g = 40,
    available_sizes_g  = '[40]'::jsonb,
    market_packages    = '[{"unit": "caja", "grams": 40, "label": "8 sobres · 1,41 oz", "price": 99}]'::jsonb,
    density_g_per_unit = 5
WHERE name = 'Sazón con culantro y achiote';

UPDATE public.master_ingredients SET
    market_container   = 'funda',
    container_weight_g = 907,
    available_sizes_g  = '[907, 2268]'::jsonb,
    market_packages    = '[{"unit": "funda", "grams": 907, "label": "2 lb", "price": 71}, {"unit": "funda", "grams": 2268, "label": "5 lb", "price": 157}]'::jsonb,
    density_g_per_cup  = 220
WHERE name = 'Azúcar morena';

UPDATE public.master_ingredients SET
    market_container   = 'paquete',
    container_weight_g = 454,
    available_sizes_g  = '[454]'::jsonb,
    market_packages    = '[{"unit": "paquete", "grams": 454, "label": "1 lb", "price": 73}]'::jsonb,
    density_g_per_cup  = 108
WHERE name = 'Pan rallado';

UPDATE public.master_ingredients SET
    market_container   = 'pote',
    container_weight_g = 283,
    available_sizes_g  = '[283]'::jsonb,
    market_packages    = '[{"unit": "pote", "grams": 283, "label": "10 oz", "price": 299}]'::jsonb
WHERE name = 'Hummus';

-- Sanity: las cuatro filas quedan con envase, el sobre pesa 5 g y hay densidad de taza.
DO $$
DECLARE
    n int;
BEGIN
    SELECT count(*) INTO n FROM public.master_ingredients
    WHERE name IN ('Sazón con culantro y achiote', 'Azúcar morena', 'Pan rallado', 'Hummus')
      AND container_weight_g > 0 AND market_container IS NOT NULL AND market_packages IS NOT NULL;
    IF n <> 4 THEN
        RAISE EXCEPTION 'P1-DO-DESPENSA-DE-SU-MERCADO envases: esperaba 4 filas con envase, hay %', n;
    END IF;
    IF (SELECT density_g_per_unit FROM public.master_ingredients
        WHERE name = 'Sazón con culantro y achiote') IS DISTINCT FROM 5 THEN
        RAISE EXCEPTION 'P1-DO-DESPENSA-DE-SU-MERCADO envases: el sobre del sazón no pesa 5 g';
    END IF;
    SELECT count(*) INTO n FROM public.master_ingredients
    WHERE name IN ('Azúcar morena', 'Pan rallado') AND density_g_per_cup > 0;
    IF n <> 2 THEN
        RAISE EXCEPTION 'P1-DO-DESPENSA-DE-SU-MERCADO envases: faltan densidades de taza (% de 2)', n;
    END IF;
END $$;
