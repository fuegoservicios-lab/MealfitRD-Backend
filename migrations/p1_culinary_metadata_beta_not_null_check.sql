-- [P1-CULINARY-METADATA-BETA · 2026-08-19] EL ANCLA: prep_methods deja de poder
-- nacer NULL en master_ingredients.
--
-- POR QUE UN CHECK Y NO UN TEST. El hueco que este P-fix cierra (141 filas beta
-- insertadas el 2026-08-17 sin metadata culinaria, cobertura del catalogo de vuelta
-- a 59%) ocurrio CON los tests existentes en verde, porque son parser-based sobre
-- las migraciones: prueban que el archivo existe, que es byte-identico en los dos
-- dirs SSOT y que es idempotente. Ninguno mira el DATO. Un test `e2e` contra Neon
-- tampoco serviria: el gate deselecciona ese marker, y un guard que no puede fallar
-- es peor que no tenerlo. La invariante tiene que vivir donde vive el dato.
--
-- Mismo patron que I8 (`meal_plans_complete_requires_days`, P2-NEXT-4): constraint
-- con nombre para que el error sea buscable, no un `SET NOT NULL` anonimo.
--
-- CONSECUENCIA BUSCADA: a partir de aqui, un script de altas de catalogo
-- (`scripts/add_foods_*.py`) que no rellene prep_methods FALLA EN EL INSERT, ruidoso
-- y en el momento, en vez de degradar en silencio meses despues por el fail-open del
-- scan. Si estas leyendo esto porque tu INSERT reviento: el arreglo es poblar
-- prep_methods con el vocabulario canonico, NO relajar la constraint.
--
-- ready_to_eat NO lleva constraint a proposito: tiene 49 NULLs LEGITIMOS
-- preexistentes (Vegetales y Viveres, que la ronda 1 dejo asi por diseno). Un CHECK
-- ahi seria falso y romperia filas dominicanas sanas.
--
-- ORDEN: este archivo va DESPUES de p1_culinary_metadata_beta_2026_08_19.sql.
-- Aplicado antes del backfill revienta contra las 141 filas vivas — que es
-- exactamente lo que debe hacer, pero no es la forma de enterarse.
--
-- Idempotente: DROP IF EXISTS antes de ADD (P3-MIGRATION-IDEMPOTENCE-DOC).

-- == Guarda previa: no intentes crear la constraint sobre datos que la violan =====
DO $$
DECLARE _null int;
BEGIN
    SELECT COUNT(*) INTO _null FROM public.master_ingredients WHERE prep_methods IS NULL;
    IF _null > 0 THEN
        RAISE EXCEPTION '[P1-CULINARY-METADATA-BETA] % filas con prep_methods NULL: corre PRIMERO p1_culinary_metadata_beta_2026_08_19.sql', _null;
    END IF;
END $$;

ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_prep_methods_not_null;

ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_prep_methods_not_null
    CHECK (prep_methods IS NOT NULL);

COMMENT ON CONSTRAINT master_ingredients_prep_methods_not_null ON public.master_ingredients IS
    '[P1-CULINARY-METADATA-BETA] Toda alta de catalogo debe traer metadata culinaria. NULL era fail-open silencioso en culinary_coherence (V1/V2 saltados para ese alimento).';

-- == Sanity: la constraint existe y muerde ========================================
DO $$
DECLARE _existe int;
BEGIN
    SELECT COUNT(*) INTO _existe FROM pg_constraint
    WHERE conname = 'master_ingredients_prep_methods_not_null'
      AND conrelid = 'public.master_ingredients'::regclass;
    IF _existe <> 1 THEN
        RAISE EXCEPTION '[P1-CULINARY-METADATA-BETA] la constraint no quedo creada';
    END IF;
END $$;
