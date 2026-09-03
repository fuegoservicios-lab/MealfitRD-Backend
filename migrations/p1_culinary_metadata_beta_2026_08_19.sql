-- [P1-CULINARY-METADATA-BETA · 2026-08-19] Backfill ronda 3 de metadata culinaria:
-- las 141 filas de paises beta que entraron el 2026-08-17 (altas P1-COUNTRY-SYSTEM-F2
-- T5-T8) nacieron con prep_methods / ready_to_eat en NULL al 100%, devolviendo la
-- cobertura del catalogo de 100% (cerrada por P2-CULINARY-METADATA-ROUND2) a 206/347
-- = 59%. NULL es fail-open POR CHECK (docs/culinary_coherence.md): el scan no falla,
-- se SALTA V1/V2 para ese alimento. Un plan dominicano sigue midiendo ~100% porque usa
-- filas DO; un plan beta colapsaba a 24% (medido sobre corpus sintetico, ver abajo).
--
-- Metodologia (misma vara que la ronda 2): cada asignacion se valido por SIMULACION
-- contra el catalogo REAL de Neon ANTES de escribir este archivo, en dos corpus:
--   (a) golden set dominicano (tests/fixtures/culinary_golden): 0 violaciones NUEVAS
--       sobre los 5 'buenos', y los defectos capa1:* de los 5 'mutados' se siguen
--       cazando 100%. Limitacion DECLARADA: el golden set no contiene ni un alimento
--       beta, asi que prueba no-regresion, no cobertura de lo nuevo.
--   (b) corpus sintetico de 22 recetas beta realistas + 8 absurdos plantados
--       (backend/tests/fixtures/culinary_beta/): 0 violaciones sobre los limpios,
--       8/8 absurdos correctamente clasificados, cobertura 24% -> 100%.
-- Esa simulacion es la fuente de verdad de las asignaciones, no intuicion.
--
-- Cuatro asignaciones NACIERON MAL y las corrigio la simulacion, no el criterio:
--   Cuajada / Bolitas de papa / Papas ralladas sin 'saltear' (se doran en sarten);
--   y el bare ARRAY['ninguno'] para TODO condimento producia un V1 falso positivo
--   real: "Guisa con el azucar morena hasta espesar". De ahi la particion entre
--   condimento-de-mesa (nunca toca calor) y condimento-de-olla (lleva sus verbos).
--
-- Idempotente: TODO UPDATE lleva `AND prep_methods IS NULL` (solo filas virgenes);
-- re-ejecutar es no-op y NO pisa ninguna de las 206 filas dominicanas.
--
-- ORDEN INTERNO (load-bearing): los overrides por alimento van ANTES de los defaults
-- por categoria. Al reves, el default deja prep_methods no-NULL y el `IS NULL` del
-- override no casa NUNCA: las 43 asignaciones de Proteinas/Lacteos/Frutas quedan
-- MUERTAS y los curados (jamones, chorizos, pepperoni) se guardan como carne CRUDA.
-- Lo cazo el dry-run transaccional contra Neon, no la simulacion en Python.
--
-- ORDEN DE ARCHIVOS: este va ANTES de p1_culinary_metadata_beta_not_null_check.sql.
-- El CHECK aplicado antes de este backfill revienta contra las 141 filas vivas.

-- == Paso 1: asignaciones por alimento (LO ESPECIFICO VA PRIMERO) =================


-- condimentos y liquidos que SI entran a la olla (ver nota _SAZON_OLLA)
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'saltear', 'guisar', 'hervir', 'hornear', 'freir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Aceite de achiote', 'Achiote', 'Adobo', 'Alcaparrado', 'Azafrán', 'Azúcar morena',
        'Chile en polvo', 'Jarabe de arce', 'Kétchup', 'Pique', 'Ron de cocina',
        'Salsa inglesa', 'Sazonador para tacos', 'Sazón con culantro y achiote'
    );

-- emulsiones frias y dulces de mesa: nunca tocan calor
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Aceitunas rellenas', 'Aderezo ranch', 'Alioli', 'Arequipe',
        'Ensalada de macarrones', 'Galletas Graham', 'Huevos rellenos', 'Lomo embuchado',
        'Mazapán', 'Membrillo dulce', 'Pretzels', 'Turrón'
    );

-- chiles secos: se tuestan en comal y se hidratan para salsas
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'tostar', 'hervir', 'licuar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Chile ancho', 'Chile chipotle', 'Chile de árbol', 'Chile guajillo', 'Chile mulato',
        'Chile pasilla'
    );

-- pastas y granos secos
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'guisar'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Coditos', 'Fideos', 'Frijol cargamanto', 'Judías blancas', 'Judías pintas',
        'Sémola de maíz'
    );

-- frutos secos (patron ronda 2)
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'tostar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Almendra marcona', 'Malvaviscos', 'Nueces pecanas', 'Nuez de Castilla', 'Piñones'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Champús', 'Chocolate de mesa', 'Especias para arroz con dulce', 'Flor de Jamaica'
    );

-- curados: el default de Proteinas los daria CRUDOS, que es falso
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'plancha'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Jamón de cocinar', 'Jamón de sándwich', 'Jamón ibérico', 'Jamón serrano'
    );

-- panes y bolleria (patron 'pan %' de la ronda 1)
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['tostar', 'ninguno'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Bagels', 'Pan de maíz', 'Panecillos ingleses'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'saltear', 'hornear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Anchoas', 'Sobrasada'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'plancha', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Cecina', 'Chuleta ahumada'
    );

-- enlatados listos que ademas se recalientan
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Chili con carne', 'Salsa de salchicha'
    );

-- quesos que se funden/gratinan: sin verbo de coccion, 'horno' daria V1 falso
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'crudo', 'hornear', 'plancha', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Queso en hebras', 'Queso provolone'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'freir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Bacalaítos'
    );

-- congelados que SI requieren coccion
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['freir', 'hornear', 'saltear'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Bolitas de papa'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'freir', 'plancha'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Boquerones'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'saltear', 'hornear', 'freir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Chicharrón'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'plancha', 'saltear', 'guisar', 'freir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Chorizo español'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'crudo', 'plancha', 'freir', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Cuajada'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir', 'hornear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Frijoles horneados'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'saltear', 'hervir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Frijoles refritos'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hornear', 'freir', 'hervir'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Harina de yuca'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['crudo', 'licuar', 'ninguno', 'hornear', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Higo'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'licuar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Hummus'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hornear'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Masa para pie'
    );

-- fruta astringente: NO se come cruda, es base de dulce cocido
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'hornear', 'guisar', 'ninguno'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Membrillo'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['plancha', 'freir'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Mezcla para panqueques'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'plancha', 'freir', 'saltear', 'hornear', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Morcilla'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'tostar', 'freir', 'hornear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Pan rallado'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['tostar', 'ninguno', 'hornear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Panecillos de mantequilla'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir', 'licuar', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Panela'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['freir', 'plancha', 'hornear', 'saltear'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Papas ralladas'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hornear', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Pepperoni'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'crudo', 'hornear', 'plancha', 'saltear', 'freir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Queso de papa'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'crudo', 'licuar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Requesón'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir', 'plancha', 'saltear', 'freir', 'hornear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Salchichas'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hornear', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Salsa barbacoa'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'saltear', 'guisar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Sofrito'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['tostar', 'ninguno', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Tortilla de maíz'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['tostar', 'ninguno', 'plancha'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Wafles'
    );

UPDATE public.master_ingredients SET
    prep_methods = ARRAY['crudo', 'licuar', 'ninguno', 'guisar', 'hervir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Xoconostle'
    );

-- == Paso 2: defaults por categoria, para lo que el paso 1 no nombro ==============
-- ready_to_eat se deja NULL en Vegetales y Viveres: las 39 y 10 filas DO de esas
-- categorias estan asi por diseno de la ronda 1 y copiarlo es lo correcto.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['crudo', 'licuar', 'ninguno'], ready_to_eat = true
    WHERE category = 'Frutas' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'saltear', 'plancha', 'hornear', 'guisar', 'crudo']
    WHERE category = 'Vegetales' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'plancha', 'freir', 'hornear', 'guisar', 'saltear'], ready_to_eat = false
    WHERE category = 'Proteínas' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'crudo'], ready_to_eat = true
    WHERE category = 'Lácteos' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'freir', 'hornear', 'guisar']
    WHERE category = 'Víveres' AND prep_methods IS NULL;
-- Despensa: heterogenea, sin default (misma decision que la ronda 1). Sus 65 filas
-- ya quedaron nombradas una a una en el paso 1.


-- == Sanity 1: vocabulario canonico (mismo check que rondas 1 y 2) ================
DO $$
DECLARE _bad int;
BEGIN
    SELECT COUNT(*) INTO _bad FROM public.master_ingredients
    WHERE prep_methods IS NOT NULL
      AND NOT (prep_methods <@ ARRAY['hervir','plancha','freir','hornear','guisar',
                                     'saltear','licuar','tostar','crudo','ninguno']);
    IF _bad > 0 THEN
        RAISE EXCEPTION '[P1-CULINARY-METADATA-BETA] % filas con prep_methods fuera del vocabulario', _bad;
    END IF;
END $$;

-- == Sanity 2: cero NULL restantes. A DIFERENCIA de la ronda 2 (que dejaba margen
-- de 10 filas y solo avisaba por RAISE NOTICE), aqui es EXCEPTION dura: el CHECK
-- constraint del archivo hermano exige cero, y descubrir el hueco al aplicar el
-- CHECK es peor que descubrirlo aqui. =============================================
DO $$
DECLARE _rest int;
BEGIN
    SELECT COUNT(*) INTO _rest FROM public.master_ingredients WHERE prep_methods IS NULL;
    IF _rest > 0 THEN
        RAISE EXCEPTION '[P1-CULINARY-METADATA-BETA] quedan % filas con prep_methods NULL; el CHECK hermano fallaria', _rest;
    END IF;
END $$;
