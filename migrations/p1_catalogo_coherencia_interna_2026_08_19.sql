-- [P1-CATALOGO-COHERENCIA-INTERNA · 2026-08-19] Cinco filas eran fisicamente
-- imposibles. Las cinco las creo el propio de-proxy de hoy.
--
-- EL MECANISMO, que es lo que hay que recordar. Cuando una fila cambia de fuente
-- (USDA -> BEDCA, USDA -> TCAC) la fuente nueva casi nunca cubre TODAS las columnas.
-- La deuda declarada de P1-BEDCA-DEPROXY-ES y P1-LATINFOODS-TCAC fue «las columnas que
-- la fuente nueva no cubre conservan el valor del proxy». Eso suena inocuo y no lo es:
-- deja SUB-COMPONENTES de un alimento junto a los TOTALES de otro, y el resultado puede
-- violar la aritmetica basica.
--
--   Curuba          fibra 10,4 > carbos 7,8   y azucares 11,2 > carbos 7,8
--                   (fibra y azucares del maracuya; carbos de la TCAC)
--   Cecina          azucares 2,7 con carbos 0,0
--   Lomo embuchado  azucares 0,9 con carbos 0,0
--                   (azucares de USDA; carbos de BEDCA)
--   Suero costeno   grasa SATURADA 6,6 > grasa TOTAL 1,5
--                   (saturada de la crema agria; total de la TCAC)
--
-- La ultima no es cosmetica: un backstop cardiovascular que lea grasa saturada leeria
-- 6,6 g sobre un alimento que tiene 1,5 g de grasa en total.
--
-- LA REGLA APLICADA: un sub-componente se escala por el MISMO factor en que encogio su
-- padre. No es inventar: es conservar la proporcion que la fuente anterior medía,
-- aplicada al total que la fuente nueva mide.
--   Curuba:         carbos 23,4 -> 7,8   factor 0,333
--   Suero costeno:  grasa 10,6 -> 1,5    factor 0,1415
--   Cecina / Lomo:  carbos -> 0          factor 0 (azucares son subconjunto de carbos)
--
-- Y hay una senal de que la regla es sana: en Suero costeno el escalado da 0,93 g de
-- saturada, que es el ~62% de la grasa total — justo la proporcion que la quimica de un
-- lacteo predice. El metodo aterriza solo donde deberia.
--
-- Idempotente: valores absolutos filtrados por `name`.

-- == Los cinco arreglos ============================================================
UPDATE public.master_ingredients SET
    fiber_g_per_100g = 3.47, sugars_g_per_100g = 3.73
    WHERE name = 'Curuba';

UPDATE public.master_ingredients SET sugars_g_per_100g = 0.0
    WHERE name IN ('Cecina', 'Lomo embuchado');

UPDATE public.master_ingredients SET saturated_fat_g_per_100g = 0.93
    WHERE name = 'Suero costeño';

-- Y de paso: `Suero costeno` habia perdido su fuente. Re-ejecutar la ronda 3 del
-- P-fix de procedencia DESPUES de P1-LATINFOODS-TCAC (como «prueba de idempotencia»)
-- la devolvio a 'manual' + el proxy de la crema agria, pisando la TCAC. Misma leccion
-- que ya quedo escrita en p1_catalogo_sinonimos_fix_refs.sql: «idempotente» no es
-- «seguro de re-ejecutar en cualquier orden».
UPDATE public.master_ingredients SET
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:648 (Suero costeño)'
    WHERE name = 'Suero costeño';

-- == El ancla: que la aritmetica basica no pueda volver a romperse =================
-- Patron I8: la invariante vive donde vive el dato. Un de-proxy futuro que deje un
-- sub-componente huerfano falla en el UPDATE, no meses despues en un guard clinico.
-- Las cuatro toleran NULL: no obligan a rellenar, solo prohiben lo imposible.
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_fibra_no_supera_carbos;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_fibra_no_supera_carbos
    CHECK (fiber_g_per_100g IS NULL OR carbs_g_per_100g IS NULL
           OR fiber_g_per_100g <= carbs_g_per_100g + 0.5);

ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_azucares_no_superan_carbos;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_azucares_no_superan_carbos
    CHECK (sugars_g_per_100g IS NULL OR carbs_g_per_100g IS NULL
           OR sugars_g_per_100g <= carbs_g_per_100g + 0.5);

ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_saturada_no_supera_grasa;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_saturada_no_supera_grasa
    CHECK (saturated_fat_g_per_100g IS NULL OR fats_g_per_100g IS NULL
           OR saturated_fat_g_per_100g <= fats_g_per_100g + 0.5);

ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_macros_no_superan_100g;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_macros_no_superan_100g
    CHECK (protein_g_per_100g IS NULL OR carbs_g_per_100g IS NULL OR fats_g_per_100g IS NULL
           OR protein_g_per_100g + carbs_g_per_100g + fats_g_per_100g <= 100.5);

-- == Sanity: las cuatro constraints existen y la tabla las cumple =================
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM pg_constraint
    WHERE conrelid = 'public.master_ingredients'::regclass
      AND conname IN ('master_ingredients_fibra_no_supera_carbos',
                      'master_ingredients_azucares_no_superan_carbos',
                      'master_ingredients_saturada_no_supera_grasa',
                      'master_ingredients_macros_no_superan_100g');
    IF _n <> 4 THEN
        RAISE EXCEPTION '[P1-CATALOGO-COHERENCIA-INTERNA] solo % de 4 constraints creadas', _n;
    END IF;
END $$;
