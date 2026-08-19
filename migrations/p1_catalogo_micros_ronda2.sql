-- [P1-CATALOGO-MICROS · ronda 2 · 2026-08-19] Los dos ultimos colesteroles, y por que
-- los cinco fosforos que quedan NO se rellenan.
--
-- COLESTEROL — cerrado. `Atun en agua` y `Yogurt griego sin azucar` son los dos unicos
-- alimentos de origen animal que quedaban sin el dato. Sus fdc_id son de tipo
-- `Foundation`, que el endpoint de DETALLE de USDA no sirve (devuelve 404) pero el
-- BUSCADOR si — de ahi salen estos valores.
--
-- FOSFORO — 5 huecos que se quedan abiertos A PROPOSITO: Achiote, Champus y Hoja santa
-- no tienen fdc_id (son filas manuales) y Borojo y Chontaduro son de la TCAC.
--
-- La TCAC **si** trae tabla de minerales con las dos filas. No se usa, y esa es la
-- decision que importa: en esa tabla el numero de columnas VARIA por fila igual que en
-- la proximal (Borojo trae 7 bloques, Chontaduro 6), asi que el mapeo por posicion no
-- es fiable — y aqui NO hay un validador equivalente a Atwater que diga si la columna
-- leida es la correcta. Los valores que saldrian del mapeo ingenuo (fosforo 1,5 mg en
-- el borojo, 359 mg en el chontaduro) son ambos implausibles, lo que confirma que el
-- mapeo esta mal.
--
-- En la tabla proximal se extrajo con confianza porque Atwater cruzaba cada fila. Sin
-- ese cruce, extraer seria adivinar con pinta de dato — exactamente lo que toda esta
-- tanda vino a eliminar del catalogo. Cinco huecos declarados valen mas que cinco
-- numeros inventados.

UPDATE public.master_ingredients SET cholesterol_mg_per_100g = 36.0
    WHERE name = 'Atún en agua';       -- USDA «Fish, tuna, light, canned in water» (Foundation)
UPDATE public.master_ingredients SET cholesterol_mg_per_100g = 5.0
    WHERE name = 'Yogurt griego sin azúcar';  -- USDA «Yogurt, Greek, plain, nonfat» (Foundation)

-- == Sanity: ningun alimento ANIMAL se queda sin colesterol =======================
DO $$
DECLARE _falta text;
BEGIN
    SELECT string_agg(name, ', ') INTO _falta FROM public.master_ingredients
    WHERE cholesterol_mg_per_100g IS NULL AND category IN ('Proteínas', 'Lácteos');
    IF _falta IS NOT NULL THEN
        RAISE EXCEPTION '[P1-CATALOGO-MICROS r2] animales sin colesterol: %', _falta;
    END IF;
END $$;
