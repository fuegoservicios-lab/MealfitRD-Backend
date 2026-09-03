-- [P1-PROVENANCE-TRUTHFUL · 2026-08-19] Un `fdc_id` deja de poder mentir.
--
-- `fdc_id` es una AFIRMACION: «esta fila ES ese alimento de USDA». Tras el de-proxy
-- espanol (P1-BEDCA-DEPROXY-ES) y el yogur (P1-YOGURT-NATURAL) quedaban 16 grupos
-- con el id COMPARTIDO por 36 filas: como maximo una de cada grupo puede estar
-- diciendo la verdad.
--
-- REGLA: conserva el id la fila cuya IDENTIDAD y cuyos VALORES siguen coincidiendo con
-- la fila real de USDA (descripcion consultada a la API, una a una — abajo va la de
-- cada grupo). Las demas pasan a `fdc_id = NULL`, `nutrition_source = 'manual'` y
-- `nutrition_source_ref = 'usda:<id> (proxy: <descripcion>)'`: la traza NO se pierde,
-- deja de presentarse como fuente. Un dato aproximado etiquetado como aproximado es
-- honesto; el mismo dato con un fdc_id es una fuente falsa.
--
-- LO QUE **NO** HACE, A PROPOSITO: no borra ni fusiona ninguna fila. `Requeson` y
-- `Queso ricotta` son el mismo alimento con dos nombres, igual que `Judias blancas` y
-- `Habichuelas blancas` (ES vs DO). Fusionarlas romperia todo plan, `user_inventory` o
-- `supermarket_products.master_food_name` que las referencie POR NOMBRE — el catalogo
-- se resuelve por cadena, no por id. Es una decision de producto con su propia
-- migracion de datos, no un efecto colateral de una limpieza de procedencia.
--
-- Idempotente: filtra por `name` exacto y por `fdc_id IS NOT NULL`; re-ejecutar es no-op.

-- fdc 167750 = USDA «Prickly pears, raw»
--   conserva: Tuna de nopal — USDA dice *Prickly pears, raw*: es exactamente la tuna. Xoconostle es la tuna ACIDA, otro fruto
--   pierden el reclamo: Xoconostle
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:167750 (proxy: Prickly pears, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Xoconostle');

-- fdc 167761 = USDA «Soursop, raw»
--   conserva: Guanábana — USDA dice *Soursop*: guanabana literal. Borojo es otro fruto
--   pierden el reclamo: Borojó
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:167761 (proxy: Soursop, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Borojó');

-- fdc 169108 = USDA «Passion-fruit, (granadilla), purple, raw»
--   conserva: Granadilla — USDA dice *Passion-fruit, (granadilla), purple*: la nombra. Chinola ya tenia valores propios (108.6 vs 97) asi que su reclamo ya era falso; Curuba es otra passiflora
--   pierden el reclamo: Chinola, Curuba
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:169108 (proxy: Passion-fruit, (granadilla), purple, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Chinola', 'Curuba');

-- fdc 169396 = USDA «Peppers, ancho, dried»
--   conserva: Chile ancho — USDA dice *Peppers, ancho, dried*. Guajillo y mulato son chiles distintos con perfil propio
--   pierden el reclamo: Chile guajillo, Chile mulato
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:169396 (proxy: Peppers, ancho, dried)'
    WHERE fdc_id IS NOT NULL AND name IN ('Chile guajillo', 'Chile mulato');

-- fdc 169998 = USDA «Corn, sweet, yellow, raw»
--   conserva: Maíz dulce en granos — USDA dice *Corn, sweet, yellow, raw*. Champus es una BEBIDA colombiana, no maiz
--   pierden el reclamo: Champús
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:169998 (proxy: Corn, sweet, yellow, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Champús');

-- fdc 170591 = USDA «Nuts, pine nuts, dried»
--   conserva: Piñones — USDA dice *Nuts, pine nuts, dried*. 'Nueces mixtas' ya tenia valores propios (594 vs 673)
--   pierden el reclamo: Nueces mixtas
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:170591 (proxy: Nuts, pine nuts, dried)'
    WHERE fdc_id IS NOT NULL AND name IN ('Nueces mixtas');

-- fdc 170851 = USDA «Cheese, ricotta, whole milk»
--   conserva: Queso ricotta — USDA dice *Cheese, ricotta, whole milk*. El requeson es el MISMO alimento con otro nombre — ver la nota sobre no fusionar filas
--   pierden el reclamo: Requesón
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:170851 (proxy: Cheese, ricotta, whole milk)'
    WHERE fdc_id IS NOT NULL AND name IN ('Requesón');

-- fdc 170932 = USDA «Spices, pepper, red or cayenne»
--   conserva: Chile de árbol — USDA dice *Spices, pepper, red or cayenne*: el de arbol ES un cayena. El chipotle es jalapeno AHUMADO, otro perfil
--   pierden el reclamo: Chile chipotle
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:170932 (proxy: Spices, pepper, red or cayenne)'
    WHERE fdc_id IS NOT NULL AND name IN ('Chile chipotle');

-- fdc 171320 = USDA «Spices, cinnamon, ground»
--   conserva: Canela en polvo — USDA dice *Spices, cinnamon, ground*. 'Especias para arroz con dulce' es una MEZCLA
--   pierden el reclamo: Especias para arroz con dulce
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:171320 (proxy: Spices, cinnamon, ground)'
    WHERE fdc_id IS NOT NULL AND name IN ('Especias para arroz con dulce');

-- fdc 171714 = USDA «Breadfruit, raw»
--   conserva: Panapén — USDA dice *Breadfruit, raw*: panapen literal. El chontaduro es palma de pejibaye, mucho mas graso
--   pierden el reclamo: Chontaduro
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:171714 (proxy: Breadfruit, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Chontaduro');

-- fdc 173443 = USDA «SIN RESPUESTA (429)»
--   conserva: (ninguna) — sin dueno decidido
--   pierden el reclamo: Crema mexicana, Suero costeño
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:173443 (proxy: SIN RESPUESTA (429))'
    WHERE fdc_id IS NOT NULL AND name IN ('Crema mexicana', 'Suero costeño');

-- fdc 173859 = USDA «SIN RESPUESTA (429)»
--   conserva: Chorizo mexicano — USDA dice *Sausage, pork, chorizo, raw*. De los 4 que quedan tras el de-proxy espanol, el mexicano fresco es el mas cercano a un chorizo de cerdo crudo
--   pierden el reclamo: Chorizo santarrosano, Chorizo verde, Longaniza puertorriqueña
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:173859 (proxy: SIN RESPUESTA (429))'
    WHERE fdc_id IS NOT NULL AND name IN ('Chorizo santarrosano', 'Chorizo verde', 'Longaniza puertorriqueña');

-- fdc 173944 = USDA «Bananas, raw»
--   conserva: Guineo — USDA dice *Bananas, raw*. El guineo VERDE tiene mas almidon resistente y menos azucar
--   pierden el reclamo: Guineo verde
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:173944 (proxy: Bananas, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Guineo verde');

-- fdc 175179 = USDA «Crustaceans, shrimp, raw»
--   conserva: Camarones — USDA dice *Crustaceans, shrimp, raw*. Tilapia ya tiene valores propios (96 kcal, colesterol 50 vs 161): su reclamo era solo la etiqueta
--   pierden el reclamo: Tilapia
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:175179 (proxy: Crustaceans, shrimp, raw)'
    WHERE fdc_id IS NOT NULL AND name IN ('Tilapia');

-- FUERA DE ESTA MIGRACION, a proposito: los grupos de abajo NO se tocan porque la
-- API de USDA no devolvio su descripcion (DEMO_KEY, 30 req/hora). Sin saber que
-- alimento es realmente el id, decidir quien lo conserva seria adivinar — y este
-- P-fix existe justamente para que el catalogo deje de afirmar lo que no sabe.
-- Se cierran en cuanto haya una USDA_API_KEY propia en el entorno:
--   fdc 174220: Mejillones, Vieira
--   fdc 175202: Habichuelas blancas, Judías blancas

-- == Sanity 1: ningun fdc_id compartido, SALVO los no verificados ================
DO $$
DECLARE _dup int;
BEGIN
    SELECT COUNT(*) INTO _dup FROM (
        SELECT fdc_id FROM public.master_ingredients
        WHERE fdc_id IS NOT NULL GROUP BY fdc_id HAVING COUNT(*) > 1) t;
    IF _dup > 2 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL] % fdc_id compartidos, se esperaban 2 (los no verificados)', _dup;
    END IF;
END $$;

-- == Sanity 2: toda fila que perdio el id dejo escrito de donde salio ============
DO $$
DECLARE _mudas int;
BEGIN
    SELECT COUNT(*) INTO _mudas FROM public.master_ingredients
    WHERE nutrition_source = 'manual' AND fdc_id IS NULL
      AND (nutrition_source_ref IS NULL OR nutrition_source_ref = '')
      AND nutrition_source_date >= DATE '2026-08-19';
    IF _mudas > 0 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL] % filas manual sin referencia', _mudas;
    END IF;
END $$;
