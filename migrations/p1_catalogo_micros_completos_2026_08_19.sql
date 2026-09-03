-- [P1-CATALOGO-MICROS · 2026-08-19] Colesterol y fosforo: cerrar los huecos SIN
-- inventar, distinguiendo los dos casos que parecen el mismo.
--
-- LA DISTINCION QUE ORDENA TODO: un NULL en `cholesterol_mg` y un NULL en
-- `phosphorus_mg` NO son el mismo problema.
--
--   COLESTEROL es un esterol ANIMAL. En un alimento vegetal el valor correcto es 0, y
--   ponerlo no es inventar: es escribir lo que la bioquimica ya determina. De las 50
--   filas sin colesterol, 48 son Despensa/Vegetales/Frutas/Viveres.
--
--   FOSFORO esta en todo alimento ENTERO, asi que un NULL no se puede rellenar con 0
--   «por prudencia»: seria falso y engañaria al steering en sentido contrario. Los 11
--   huecos exigen dato real, uno a uno.
--
--   MATIZ QUE ME CORRIGIO LA PROPIA USDA: «todo alimento» era demasiado fuerte. Un
--   producto REFINADO de un solo componente si puede tener 0 legitimamente, y USDA lo
--   reporta asi — 0,0 mg en los cinco aceites del catalogo, y la sal y el vinagre igual.
--   La regla correcta distingue alimento entero de aislado (grasa pura, cloruro sodico,
--   acido acetico). Se descubrio porque un sanity mio, escrito sobre la premisa fuerte,
--   bloqueo esta misma migracion contra 10 filas preexistentes que estaban BIEN.
--
-- Es la misma clase de error que ya costo caro hoy: tratar igual dos cosas que se
-- parecen (COPIADO vs DIFERENCIADO en la procedencia). Aqui la regla se aplica antes.
--
-- Los valores que se escriben salen de la fila USDA de cada alimento, consultada una a
-- una. Las filas SIN fdc_id (Achiote, Borojo, Champus, Chontaduro, Hoja santa) se dejan
-- como estan: no hay de donde sacarlo y rellenarlas seria exactamente lo que este P-fix
-- evita.
--
-- Idempotente: valores absolutos por nombre; el bloque vegetal filtra por IS NULL.

-- == Colesterol = 0 en todo lo vegetal (bioquimica, no estimacion) =================
UPDATE public.master_ingredients SET cholesterol_mg_per_100g = 0
    WHERE cholesterol_mg_per_100g IS NULL
      AND category IN ('Despensa', 'Vegetales', 'Frutas', 'Víveres');

-- == Fosforo: dato real de USDA, alimento por alimento ============================
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 71.0,  cholesterol_mg_per_100g = 0
    WHERE name = 'Arracacha';          -- USDA «Parsnips, raw»
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 66.0,  cholesterol_mg_per_100g = 0
    WHERE name = 'Guascas';            -- USDA «Dandelion greens, raw»
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 120.0, cholesterol_mg_per_100g = 0
    WHERE name = 'Huitlacoche';        -- USDA «Mushrooms, crimini, raw»
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 40.0,  cholesterol_mg_per_100g = 0
    WHERE name = 'Uchuva';             -- USDA «Groundcherries (cape-gooseberries), raw»
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 1.0
    WHERE name = 'Panela';             -- USDA «Sugar, turbinado»
UPDATE public.master_ingredients SET phosphorus_mg_per_100g = 85.0
    WHERE name = 'Chicharrón';         -- USDA «Snacks, pork skins, plain»

-- == Sanity 1: ningun alimento VEGETAL se queda sin colesterol ====================
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE cholesterol_mg_per_100g IS NULL
      AND category IN ('Despensa', 'Vegetales', 'Frutas', 'Víveres');
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-MICROS] % filas vegetales sin colesterol', _n;
    END IF;
END $$;

-- == Sanity 2: fosforo 0 solo en aislados, nunca en un alimento entero ===========
-- La version fuerte de este guard («ningun alimento tiene 0 fosforo») era FALSA y la
-- refuto USDA: los aceites, la sal y el vinagre reportan 0,0 mg de verdad. Lo que si
-- sigue siendo cierto — y es lo que hay que impedir — es que un alimento ENTERO
-- (proteina, lacteo, fruta, vegetal, vivere) acabe con fosforo 0 porque alguien
-- «cerro el hueco» con un cero.
DO $$
DECLARE _malos text;
BEGIN
    SELECT string_agg(name, ', ') INTO _malos FROM public.master_ingredients
    WHERE phosphorus_mg_per_100g = 0
      AND category IN ('Proteínas', 'Lácteos', 'Frutas', 'Vegetales', 'Víveres');
    IF _malos IS NOT NULL THEN
        RAISE EXCEPTION '[P1-CATALOGO-MICROS] alimentos ENTEROS con fosforo 0 (dato falso, no conservador): %', _malos;
    END IF;
END $$;
