-- [P1-DO-DESPENSA-DE-SU-MERCADO · 2026-09-10]
-- Seis ingredientes de platos dominicanos sin precio, resueltos con lo que el dueno trajo de su mercado.
--
-- De donde sale: tras pasar el «queso de papa» a gouda quedaban 12 platos sin precio por 6
-- productos. Busque sinonimos en `supermarket_products` y no encontre ninguno; el dueno contesto con
-- capturas de su supermercado y con lo que sabe de su cocina:
--
--   Azucar morena   en RD se llama «AZUCAR CREMA»   Wala 2 lb RD$71 (5 lb RD$157)   -> RD$35,50/lb
--   Pan rallado     en RD se dice «PAN MOLIDO»      Buenhorno 1 lb RD$73            -> RD$73,00/lb
--   Sazon con culantro y achiote   SAZON GOYA 1,41 oz (8 sobres) RD$99             -> RD$1.122,64/lb
--   Hummus          Dietz & Watson 10 oz RD$299                                     -> RD$478,40/lb
--
-- De cada presentacion se toma el precio por libra MAS ALTO (el paquete pequeno del azucar): un precio
-- que subestima asciende en el ranking de presupuesto lo que no se puede pagar (P1-CANDIDATO-CON-PRECIO).
-- `price_source = 'owner_verified'`, el mismo valor que ya usan las filas con precio dado por el dueno.
--
-- El nombre canonico NO se toca (identidad del motor: Nevera, alergias, coherencia): el nombre
-- dominicano va en `gloss_es`, que es display-only y la lista de compras ya pinta.
--
-- QUESO DE HOJA: el dueno explico que se vende en «funditas transparentes» — no hay etiqueta que
-- leer. Sus valores (299 kcal, 20 g prot, 23 g grasa, 600 mg Na, 700 mg Ca) caen dentro de la familia
-- de pasta hilada: mozzarella entera (USDA 170845: 299/22,2/22,1/486/505) y queso en hebras (USDA
-- 171244: 295/23,75/19,78/666/697). No se cambian sin evidencia; se DOCUMENTA su procedencia en
-- `nutrition_source_ref`. Poner un `fdc_id` con valores distintos seria el puntero que miente que se
-- corrigio tres veces hoy.
--
-- Los otros dos los resuelven los JSON del backend: el SOFRITO se hace, no se compra (se desglosa en
-- aji cubanela, cilantro, cebolla y ajo), y los tres platos de SEMOLA DE MAIZ salen de la biblioteca
-- dominicana (el dueno no la conoce; la Maizena no la sustituye).
--
-- Idempotente: re-aplicarla no cambia el resultado.

UPDATE public.master_ingredients SET
    price_per_lb = 35.50, price_source = 'owner_verified', price_captured_at = DATE '2026-09-10',
    price_confidence = 'high', gloss_es = 'azúcar crema'
WHERE name = 'Azúcar morena';

UPDATE public.master_ingredients SET
    price_per_lb = 73.00, price_source = 'owner_verified', price_captured_at = DATE '2026-09-10',
    price_confidence = 'high', gloss_es = 'pan molido'
WHERE name = 'Pan rallado';

UPDATE public.master_ingredients SET
    price_per_lb = 1122.64, price_source = 'owner_verified', price_captured_at = DATE '2026-09-10',
    price_confidence = 'high'
WHERE name = 'Sazón con culantro y achiote';

UPDATE public.master_ingredients SET
    price_per_lb = 478.40, price_source = 'owner_verified', price_captured_at = DATE '2026-09-10',
    price_confidence = 'high'
WHERE name = 'Hummus';

UPDATE public.master_ingredients SET
    nutrition_source_ref = 'sin etiqueta: se vende en funditas transparentes (dueno, 2026-09-10); valores '
                        || 'dentro de la familia pasta hilada: USDA 170845 (mozzarella entera) y 171244 '
                        || '(queso en hebras)'
WHERE name = 'Queso de hoja';

-- Sanity: las cinco filas existen y quedaron como se dice; ningun nombre se movio.
DO $$
DECLARE
    _n INT;
BEGIN
    SELECT count(*) INTO _n FROM public.master_ingredients
    WHERE (name = 'Azúcar morena' AND price_per_lb = 35.50 AND gloss_es = 'azúcar crema')
       OR (name = 'Pan rallado' AND price_per_lb = 73.00 AND gloss_es = 'pan molido')
       OR (name = 'Sazón con culantro y achiote' AND price_per_lb = 1122.64)
       OR (name = 'Hummus' AND price_per_lb = 478.40)
       OR (name = 'Queso de hoja' AND nutrition_source_ref LIKE 'sin etiqueta%' AND nutrition_source = 'manual');
    IF _n <> 5 THEN
        RAISE EXCEPTION '[P1-DO-DESPENSA-DE-SU-MERCADO] esperaba 5 filas corregidas, hay %', _n;
    END IF;
END $$;
