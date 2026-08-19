-- [P1-LATINFOODS-TCAC · 2026-08-19] Cinco alimentos andinos dejan de vivir sobre un
-- proxy de USDA y pasan a su tabla nacional.
--
-- FUENTE: Tabla de Composición de Alimentos Colombianos (TCAC 2015, ICBF), la tabla
-- oficial de Colombia dentro de la red FAO/INFOODS-LATINFOODS. Se extrajo del PDF
-- publicado — no hay API ni Excel — leyendo la «Tabla del análisis proximal».
--
-- LA MAGNITUD DEL ERROR QUE CORRIGE, que es el motivo de hacerlo:
--
--   Chontaduro     103 -> 332 kcal   3,2x. Vivía sobre *Breadfruit* (panapén). El
--                                    chontaduro es palma de pejibaye: 25,7 g de GRASA
--                                    por 100 g. El panapén tiene 0,23. No se parecen
--                                    en nada.
--   Curuba          97 -> 35 kcal    2,8x al revés. Vivía sobre *Passion-fruit purple*.
--   Borojó          66 -> 134 kcal   2x. Vivía sobre *Soursop* (guanábana).
--   Chinola        108,6 -> 59 kcal  vivía sobre la misma fila de granadilla; la TCAC
--                                    la tiene como «Maracuyá, maduro, pulpa».
--   Suero costeño  136 -> 83 kcal    y proteína 3,5 -> 11,0. Vivía sobre *Sour cream*:
--                                    es suero fermentado, no crema. El error no era de
--                                    magnitud sino de CATEGORÍA de alimento.
--
-- CÓMO SE VALIDÓ LA EXTRACCIÓN. Un PDF no tiene contrato: la tabla proximal cambia de
-- número de columnas según si la fila trae desviación estándar o no, así que partir por
-- posición se rompe. Se parte por las LETRAS de calificación (a/b/c), y **cada fila se
-- acepta solo si cruza Atwater** (4P + 4C + 9G vs las kcal declaradas) dentro del 5%.
-- Las cinco cruzan con <1,5% de divergencia. Ese cruce no es decorativo: es lo que
-- distingue una extracción correcta de un mapeo de columnas equivocado, y de hecho cazó
-- dos errores de parseo míos antes de que llegaran aquí (la columna «N» confundida con
-- las kcal, y un `finditer` que consumía el par kcal/kJ antes de poder evaluarlo).
--
-- LO QUE **NO** SE TOCA, Y POR QUÉ: la tabla proximal de la TCAC trae macros y cenizas,
-- pero NO fibra ni minerales ni vitaminas — viven en tablas aparte del mismo PDF, con
-- otra estructura. Esas columnas conservan el valor heredado del proxy de USDA. Es deuda
-- CONOCIDA y declarada, igual que en P1-BEDCA-DEPROXY-ES: sobrescribirlas exigiría
-- inventarlas, y vitamin_a_mcg_rae / vitamin_k_mcg son además NOT NULL.
--
-- `Champús` NO entra: no está en la TCAC (es una bebida preparada, no un alimento base).
-- Sigue etiquetado como proxy, que es lo honesto.
--
-- Idempotente: filtra por `name` exacto; re-ejecutar reescribe lo mismo.

-- == Paso 0: el enum de procedencia admite 'latinfoods' ============================
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_nutrition_source_check;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_nutrition_source_check
    CHECK (nutrition_source IS NULL OR nutrition_source = ANY
           (ARRAY['usda'::text, 'off'::text, 'faoinfoods'::text, 'manual'::text,
                  'bedca'::text, 'latinfoods'::text]));

-- == Los cinco, uno a uno, con su código en la TCAC ================================

-- TCAC 289 «Chontaduro, maduro, pulpa» · Atwater 0,2%
UPDATE public.master_ingredients SET
    kcal_per_100g = 332.0, protein_g_per_100g = 6.3,
    fats_g_per_100g = 25.7, carbs_g_per_100g = 19.0,
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:289 (Chontaduro, maduro, pulpa)',
    nutrition_source_date = DATE '2026-08-19', fdc_id = NULL
WHERE name = 'Chontaduro';

-- TCAC 298 «Curuba, maduro, pulpa» · Atwater 1,4%
UPDATE public.master_ingredients SET
    kcal_per_100g = 35.0, protein_g_per_100g = 0.6,
    fats_g_per_100g = 0.1, carbs_g_per_100g = 7.8,
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:298 (Curuba, maduro, pulpa)',
    nutrition_source_date = DATE '2026-08-19', fdc_id = NULL
WHERE name = 'Curuba';

-- TCAC 272 «Borojó, maduro, pulpa» · Atwater 0,4%
UPDATE public.master_ingredients SET
    kcal_per_100g = 134.0, protein_g_per_100g = 3.0,
    fats_g_per_100g = 0.6, carbs_g_per_100g = 29.0,
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:272 (Borojó, maduro, pulpa)',
    nutrition_source_date = DATE '2026-08-19', fdc_id = NULL
WHERE name = 'Borojó';

-- TCAC 333 «Maracuyá, maduro, pulpa» · Atwater 0,8%
-- En RD el maracuyá se llama chinola: es el MISMO fruto (Passiflora edulis).
UPDATE public.master_ingredients SET
    kcal_per_100g = 59.0, protein_g_per_100g = 1.5,
    fats_g_per_100g = 0.5, carbs_g_per_100g = 12.0,
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:333 (Maracuyá, maduro, pulpa)',
    nutrition_source_date = DATE '2026-08-19', fdc_id = NULL
WHERE name = 'Chinola';

-- TCAC 648 «Suero costeño» · Atwater 0,1%
UPDATE public.master_ingredients SET
    kcal_per_100g = 83.0, protein_g_per_100g = 11.0,
    fats_g_per_100g = 1.5, carbs_g_per_100g = 6.4,
    nutrition_source = 'latinfoods',
    nutrition_source_ref = 'tcac:648 (Suero costeño)',
    nutrition_source_date = DATE '2026-08-19', fdc_id = NULL
WHERE name = 'Suero costeño';

-- == Sanity 1: las 5 filas quedaron marcadas y con su código TCAC ==================
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE nutrition_source = 'latinfoods' AND nutrition_source_ref LIKE 'tcac:%'
      AND fdc_id IS NULL;
    IF _n <> 5 THEN
        RAISE EXCEPTION '[P1-LATINFOODS-TCAC] % filas latinfoods completas, esperadas 5', _n;
    END IF;
END $$;

-- == Sanity 2: Atwater sobre lo escrito ===========================================
-- El mismo cruce que valido la extraccion, ahora sobre la DB: si alguien re-escribe
-- una de estas filas con numeros incoherentes, salta aqui.
DO $$
DECLARE _mal int;
BEGIN
    SELECT COUNT(*) INTO _mal FROM public.master_ingredients
    WHERE nutrition_source = 'latinfoods' AND kcal_per_100g > 0
      AND ABS((4*protein_g_per_100g + 4*carbs_g_per_100g + 9*fats_g_per_100g)
              - kcal_per_100g) / kcal_per_100g > 0.05;
    IF _mal > 0 THEN
        RAISE EXCEPTION '[P1-LATINFOODS-TCAC] % filas latinfoods divergen >5%% en Atwater', _mal;
    END IF;
END $$;

-- == Sanity 3: el chontaduro dejo de parecerse al panapen =========================
-- Prueba directa del efecto: eran la MISMA fila (103 kcal, 0,23 g de grasa).
DO $$
DECLARE _kcal numeric; _grasa numeric;
BEGIN
    SELECT kcal_per_100g, fats_g_per_100g INTO _kcal, _grasa
    FROM public.master_ingredients WHERE name = 'Chontaduro';
    IF _kcal < 300 OR _grasa < 20 THEN
        RAISE EXCEPTION '[P1-LATINFOODS-TCAC] Chontaduro sigue con perfil de panapen (% kcal, % g grasa)', _kcal, _grasa;
    END IF;
END $$;
