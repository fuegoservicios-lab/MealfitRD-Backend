-- [P1-CATALOGO-SINONIMOS · fix de referencias · 2026-08-19] Una fila no puede ser
-- sinonimo de si misma.
--
-- El UPDATE de la sincronizacion aplicaba `nutrition_source_ref = COALESCE(ref,
-- 'sinonimo:<canonico>')` a las DOS filas del par. Las canonicas tenian el ref en NULL,
-- asi que se quedaron apuntandose a si mismas: `Judias blancas` -> «sinonimo:Judias
-- blancas». Absurdo y, peor, borra la senal util: el canonico se identifica porque
-- conserva su `fdc_id`, no por una referencia circular.
--
-- Y `Habichuelas blancas` seguia diciendo «id previo; valores propios», que era cierto
-- ANTES de la sincronizacion y dejo de serlo justo al aplicarla: sus valores ya no son
-- propios, son los del canonico verificado. Una etiqueta correcta puede caducar por un
-- cambio en OTRA columna.
--
-- Queda asi:
--   canonico (conserva su fdc_id)  -> nutrition_source_ref = NULL
--   sinonimo regional              -> 'sinonimo:<canonico>'
--
-- ⚠️ ORDEN, NO SOLO IDEMPOTENCIA. Este archivo va DESPUES de
-- p1_provenance_truthful_round_3_cierre.sql, que asigna a `Habichuelas blancas` el ref
-- 'usda:175202 (id previo; valores propios)'. Esa etiqueta era cierta hasta que la
-- sincronizacion de sinonimos le dio los valores del canonico; ahora la correcta es
-- 'sinonimo:Judias blancas'.
--
-- Se descubrio de la peor forma: re-ejecutando la ronda 3 como «prueba de idempotencia»
-- DESPUES de este fix, que la deshizo. **«Idempotente» no significa «seguro de
-- re-ejecutar en cualquier orden»**: la ronda 3 es idempotente consigo misma, pero
-- re-correrla despues de una migracion posterior revierte a la posterior. Las
-- migraciones se aplican UNA VEZ y EN ORDEN; re-ejecutar una vieja contra una base ya
-- avanzada no es una comprobacion inocua.

UPDATE public.master_ingredients SET nutrition_source_ref = NULL
    WHERE name IN ('Queso ricotta', 'Judías blancas') AND fdc_id IS NOT NULL;

UPDATE public.master_ingredients SET nutrition_source_ref = 'sinonimo:Queso ricotta'
    WHERE name = 'Requesón';
UPDATE public.master_ingredients SET nutrition_source_ref = 'sinonimo:Judías blancas'
    WHERE name = 'Habichuelas blancas';

-- == Sanity 1: ninguna fila se declara sinonimo de si misma =======================
DO $$
DECLARE _circular int;
BEGIN
    SELECT COUNT(*) INTO _circular FROM public.master_ingredients
    WHERE nutrition_source_ref = 'sinonimo:' || name;
    IF _circular > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-SINONIMOS fix] % fila(s) sinonimo de si mismas', _circular;
    END IF;
END $$;

-- == Sanity 2: todo 'sinonimo:X' apunta a una fila que EXISTE y tiene fdc_id ======
-- Sin esto, un renombre del canonico dejaria referencias colgando en silencio.
DO $$
DECLARE _huerfanas int;
BEGIN
    SELECT COUNT(*) INTO _huerfanas FROM public.master_ingredients s
    WHERE s.nutrition_source_ref LIKE 'sinonimo:%'
      AND NOT EXISTS (
        SELECT 1 FROM public.master_ingredients c
        WHERE c.name = substring(s.nutrition_source_ref from 10)
          AND c.fdc_id IS NOT NULL);
    IF _huerfanas > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-SINONIMOS fix] % referencia(s) sinonimo apuntan a una fila inexistente o sin procedencia', _huerfanas;
    END IF;
END $$;

-- == Sanity 3: los pares siguen con los mismos macros =============================
DO $$
DECLARE _difs int;
BEGIN
    SELECT COUNT(*) INTO _difs FROM public.master_ingredients s
    JOIN public.master_ingredients c
      ON c.name = substring(s.nutrition_source_ref from 10)
    WHERE s.nutrition_source_ref LIKE 'sinonimo:%'
      AND (s.kcal_per_100g IS DISTINCT FROM c.kcal_per_100g
        OR s.protein_g_per_100g IS DISTINCT FROM c.protein_g_per_100g
        OR s.carbs_g_per_100g IS DISTINCT FROM c.carbs_g_per_100g
        OR s.fats_g_per_100g IS DISTINCT FROM c.fats_g_per_100g);
    IF _difs > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-SINONIMOS fix] % sinonimo(s) divergen del canonico', _difs;
    END IF;
END $$;
