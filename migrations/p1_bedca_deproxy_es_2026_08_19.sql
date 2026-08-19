-- [P1-BEDCA-DEPROXY-ES · 2026-08-19] Los embutidos y curados espanoles dejan de vivir
-- sobre una fila de USDA que no es la suya.
--
-- DIAGNOSTICO (auditoria de procedencia, 2026-08-19): de 347 filas de master_ingredients,
-- 47 comparten fdc_id con otra fila; en 13 de esos 20 grupos los macros estan literalmente
-- COPIADOS. El peor: `fdc 173859` (USDA *Sausage, pork, chorizo, raw*, 296 kcal) hacia de
-- sustituto de SIETE embutidos a la vez, Sobrasada incluida. Detalle completo y clasificacion
-- de los 20 grupos: backend/docs/catalog_provenance_audit.md
--
-- La magnitud no era cosmetica:
--     Sobrasada        296 -> 595 kcal   (el catalogo contaba la MITAD)
--     Lomo embuchado   110 -> 321 kcal   (~3x; ademas prot 20.3 -> 34.0)
--     Chistorra        296 -> 512 kcal
--     Jamon serrano    195 -> 318 kcal   (grasa 8.32 -> 22.6)
--     Cecina           153 -> 242 kcal
--
-- FUENTE: BEDCA (Base de Datos Espanola de Composicion de Alimentos, AESAN/MICINN),
-- consultada via su servicio publico `procquery.php`. Cada fila trae abajo su f_id, la
-- descripcion exacta de BEDCA y POR QUE se eligio esa entrada y no otra.
--
-- CONVERSION: BEDCA publica la energia en **kJ**, no en kcal. kcal = kJ / 4.184. Los 11
-- alimentos cruzan Atwater (4P + 4C + 9G vs kcal) con <2% de divergencia — muy por debajo
-- del 12% que marcan los scripts de alta, y confirmacion independiente de que la conversion
-- y el mapeo de componentes son correctos.
--
-- LO QUE **NO** SE TOCA, Y POR QUE: BEDCA no reporta azucares, vitamina A (RAE), vitamina D
-- ni vitamina K. Esas cuatro columnas conservan el valor heredado del proxy de USDA. Es
-- deuda CONOCIDA y declarada, no un descuido: sobrescribirlas exigiria inventarlas, y
-- vitamin_a_mcg_rae / vitamin_k_mcg son ademas NOT NULL (P1-EXTENDED-MICROS-GUARD).
--
-- `Lomo embuchado` entra aqui aunque NO estaba en ningun grupo de fdc_id compartido: tenia
-- su propio fdc, apuntando a lomo de cerdo CRUDO en vez de al curado. Es la prueba de que la
-- clase de error es mas amplia que los ids compartidos — un fdc_id unico tambien puede
-- apuntar al alimento equivocado, y la auditoria por duplicados NO lo ve.
--
-- Idempotente: filtra por `name` exacto; re-ejecutar reescribe los mismos valores.

-- == Paso 0: el enum de procedencia admite 'bedca' ================================
-- El CHECK vigente solo permite usda|off|faoinfoods|manual. Sin esto, todo UPDATE de
-- abajo falla.
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_nutrition_source_check;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_nutrition_source_check
    CHECK (nutrition_source IS NULL OR nutrition_source = ANY
           (ARRAY['usda'::text, 'off'::text, 'faoinfoods'::text, 'manual'::text, 'bedca'::text]));

-- == Paso 1: columna de referencia de procedencia ==================================
-- `fdc_id` solo sabe hablar de USDA. Sin un campo donde anotar el f_id de BEDCA, estas
-- filas nacerian IMPOSIBLES de re-verificar contra su fuente — repitiendo a sabiendas el
-- problema que destapo el fdc 330137 (compartido por dos yogures y ademas HTTP 404 hoy:
-- procedencia podrida sin que nada lo notara).
ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS nutrition_source_ref text DEFAULT NULL;
COMMENT ON COLUMN public.master_ingredients.nutrition_source_ref IS
    '[P1-BEDCA-DEPROXY-ES] Id del alimento en la fuente NO-USDA (p.ej. bedca:2264). fdc_id solo cubre USDA.';

-- == Paso 2: los valores, uno a uno ================================================

-- Boquerones  <-  BEDCA 2316 "Boquerón"
--   sin freir, misma razon
--   Atwater: 0.1% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 127.2,
    protein_g_per_100g          = 17.6,
    carbs_g_per_100g            = 0.0,
    fats_g_per_100g             = 6.3,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 1.65,
    sodium_mg_per_100g          = 116.0,
    cholesterol_mg_per_100g     = 69.0,
    calcium_mg_per_100g         = 30.0,
    iron_mg_per_100g            = 1.0,
    potassium_mg_per_100g       = 331.0,
    magnesium_mg_per_100g       = 29.0,
    phosphorus_mg_per_100g      = 182.0,
    zinc_mg_per_100g            = 0.5,
    selenium_mcg_per_100g       = 36.5,
    folate_mcg_dfe_per_100g     = 8.7,
    vitamin_b12_mcg_per_100g    = 1.9,
    vitamin_e_mg_per_100g       = 0.02,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2316',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Boquerones';

-- Butifarra  <-  BEDCA 2254 "Butifarra"
--   entrada propia
--   Atwater: 0.9% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 239.8,
    protein_g_per_100g          = 10.0,
    carbs_g_per_100g            = 5.5,
    fats_g_per_100g             = 20.0,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 6.8,
    sodium_mg_per_100g          = 703.0,
    cholesterol_mg_per_100g     = 50.0,
    calcium_mg_per_100g         = 51.0,
    iron_mg_per_100g            = 1.9,
    potassium_mg_per_100g       = 140.0,
    magnesium_mg_per_100g       = 15.0,
    phosphorus_mg_per_100g      = 51.0,
    selenium_mcg_per_100g       = 11.5,
    folate_mcg_dfe_per_100g     = 2.0,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.01,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2254',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Butifarra';

-- Cecina  <-  BEDCA 2256 "Cecina"
--   entrada propia
--   Atwater: 0.4% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 242.5,
    protein_g_per_100g          = 39.0,
    carbs_g_per_100g            = 0.0,
    fats_g_per_100g             = 9.5,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 4.4,
    sodium_mg_per_100g          = 2100.0,
    cholesterol_mg_per_100g     = 120.0,
    calcium_mg_per_100g         = 48.0,
    iron_mg_per_100g            = 9.8,
    potassium_mg_per_100g       = 62.2,
    magnesium_mg_per_100g       = 39.0,
    phosphorus_mg_per_100g      = 321.0,
    folate_mcg_dfe_per_100g     = 0.0,
    vitamin_b12_mcg_per_100g    = 8.89,
    vitamin_c_mg_per_100g       = 0.0,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2256',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Cecina';

-- Chistorra  <-  BEDCA 2262 "Chistorra"
--   entrada propia
--   Atwater: 1.3% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 512.4,
    protein_g_per_100g          = 15.3,
    carbs_g_per_100g            = 0.9,
    fats_g_per_100g             = 50.5,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 26.8,
    sodium_mg_per_100g          = 889.0,
    cholesterol_mg_per_100g     = 76.0,
    calcium_mg_per_100g         = 13.0,
    iron_mg_per_100g            = 1.7,
    potassium_mg_per_100g       = 232.0,
    magnesium_mg_per_100g       = 23.0,
    phosphorus_mg_per_100g      = 90.0,
    zinc_mg_per_100g            = 1.4,
    selenium_mcg_per_100g       = 21.1,
    folate_mcg_dfe_per_100g     = 2.0,
    vitamin_b12_mcg_per_100g    = 0.85,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.57,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2262',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Chistorra';

-- Chorizo español  <-  BEDCA 2264 "Chorizo"
--   entrada generica del embutido espanol
--   Atwater: 0.6% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 321.7,
    protein_g_per_100g          = 27.0,
    carbs_g_per_100g            = 1.9,
    fats_g_per_100g             = 23.1,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 9.6,
    sodium_mg_per_100g          = 1060.0,
    cholesterol_mg_per_100g     = 72.6,
    calcium_mg_per_100g         = 18.4,
    iron_mg_per_100g            = 2.1,
    potassium_mg_per_100g       = 180.0,
    magnesium_mg_per_100g       = 10.3,
    phosphorus_mg_per_100g      = 270.0,
    zinc_mg_per_100g            = 1.2,
    selenium_mcg_per_100g       = 21.1,
    folate_mcg_dfe_per_100g     = 0.9,
    vitamin_b12_mcg_per_100g    = 0.9,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.29,
    omega3_ala_g_per_100g       = 0.151,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2264',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Chorizo español';

-- Jamón ibérico  <-  BEDCA 1777 "Jamón ibérico de cebo"
--   de cebo, no de bellota: el de cebo es el que se compra a diario
--   Atwater: 0.3% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 301.4,
    protein_g_per_100g          = 32.3,
    carbs_g_per_100g            = 0.1,
    fats_g_per_100g             = 19.2,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 7.81,
    sodium_mg_per_100g          = 1942.0,
    cholesterol_mg_per_100g     = 69.0,
    calcium_mg_per_100g         = 10.2,
    iron_mg_per_100g            = 4.3,
    potassium_mg_per_100g       = 701.0,
    magnesium_mg_per_100g       = 37.2,
    phosphorus_mg_per_100g      = 141.0,
    zinc_mg_per_100g            = 4.0,
    selenium_mcg_per_100g       = 12.1,
    folate_mcg_dfe_per_100g     = 13.5,
    vitamin_b12_mcg_per_100g    = 15.7,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.08,
    omega3_ala_g_per_100g       = 0.084,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:1777',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Jamón ibérico';

-- Jamón serrano  <-  BEDCA 2273 "Jamón serrano"
--   entrada propia
--   Atwater: 0.5% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 317.7,
    protein_g_per_100g          = 28.8,
    carbs_g_per_100g            = 0.2,
    fats_g_per_100g             = 22.6,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 7.94,
    sodium_mg_per_100g          = 2130.0,
    cholesterol_mg_per_100g     = 84.0,
    calcium_mg_per_100g         = 9.0,
    iron_mg_per_100g            = 1.7,
    potassium_mg_per_100g       = 250.0,
    magnesium_mg_per_100g       = 22.0,
    phosphorus_mg_per_100g      = 167.0,
    zinc_mg_per_100g            = 2.1,
    selenium_mcg_per_100g       = 12.1,
    folate_mcg_dfe_per_100g     = 2.0,
    vitamin_b12_mcg_per_100g    = 0.6,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.2,
    omega3_ala_g_per_100g       = 0.068,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2273',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Jamón serrano';

-- Lomo embuchado  <-  BEDCA 2277 "Lomo embuchado"
--   entrada propia
--   Atwater: 0.3% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 321.2,
    protein_g_per_100g          = 34.0,
    carbs_g_per_100g            = 0.0,
    fats_g_per_100g             = 20.7,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 6.68,
    sodium_mg_per_100g          = 1470.0,
    cholesterol_mg_per_100g     = 69.0,
    calcium_mg_per_100g         = 20.0,
    iron_mg_per_100g            = 3.7,
    potassium_mg_per_100g       = 230.0,
    magnesium_mg_per_100g       = 20.0,
    phosphorus_mg_per_100g      = 180.0,
    zinc_mg_per_100g            = 2.6,
    selenium_mcg_per_100g       = 20.0,
    folate_mcg_dfe_per_100g     = 5.0,
    vitamin_b12_mcg_per_100g    = 2.0,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.44,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2277',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Lomo embuchado';

-- Morcilla  <-  BEDCA 2280 "Morcilla"
--   sin freir, misma razon
--   Atwater: 1.2% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 323.0,
    protein_g_per_100g          = 11.0,
    carbs_g_per_100g            = 3.2,
    fats_g_per_100g             = 30.0,
    saturated_fat_g_per_100g    = 11.7,
    sodium_mg_per_100g          = 700.0,
    cholesterol_mg_per_100g     = 110.0,
    calcium_mg_per_100g         = 40.0,
    iron_mg_per_100g            = 18.0,
    potassium_mg_per_100g       = 150.0,
    magnesium_mg_per_100g       = 11.0,
    phosphorus_mg_per_100g      = 60.0,
    zinc_mg_per_100g            = 0.3,
    selenium_mcg_per_100g       = 11.8,
    folate_mcg_dfe_per_100g     = 5.0,
    vitamin_b12_mcg_per_100g    = 0.5,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.5,
    omega3_ala_g_per_100g       = 0.356,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2280',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Morcilla';

-- Panceta ibérica  <-  BEDCA 2260 "Cerdo, panceta, cruda"
--   CRUDA: la convencion del catalogo es carnes en crudo
--   Atwater: 1.4% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 464.9,
    protein_g_per_100g          = 12.5,
    carbs_g_per_100g            = 0.5,
    fats_g_per_100g             = 46.6,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 19.39,
    sodium_mg_per_100g          = 1470.0,
    cholesterol_mg_per_100g     = 72.0,
    calcium_mg_per_100g         = 6.0,
    iron_mg_per_100g            = 0.9,
    potassium_mg_per_100g       = 230.0,
    magnesium_mg_per_100g       = 13.0,
    phosphorus_mg_per_100g      = 120.0,
    zinc_mg_per_100g            = 1.5,
    selenium_mcg_per_100g       = 1.0,
    folate_mcg_dfe_per_100g     = 1.5,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.08,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2260',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Panceta ibérica';

-- Sobrasada  <-  BEDCA 2311 "Sobrasada"
--   entrada propia
--   Atwater: 1.5% de divergencia
UPDATE public.master_ingredients SET
    kcal_per_100g               = 595.4,
    protein_g_per_100g          = 12.9,
    carbs_g_per_100g            = 0.0,
    fats_g_per_100g             = 61.4,
    fiber_g_per_100g            = 0.0,
    saturated_fat_g_per_100g    = 20.7,
    sodium_mg_per_100g          = 1264.0,
    cholesterol_mg_per_100g     = 72.0,
    calcium_mg_per_100g         = 15.0,
    iron_mg_per_100g            = 1.9,
    potassium_mg_per_100g       = 259.0,
    magnesium_mg_per_100g       = 25.0,
    phosphorus_mg_per_100g      = 310.0,
    zinc_mg_per_100g            = 1.5,
    selenium_mcg_per_100g       = 21.1,
    folate_mcg_dfe_per_100g     = 2.0,
    vitamin_b12_mcg_per_100g    = 0.9,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.57,
    nutrition_source            = 'bedca',
    nutrition_source_ref        = 'bedca:2311',
    nutrition_source_date       = DATE '2026-08-19',
    fdc_id                      = NULL
WHERE name = 'Sobrasada';

-- == Sanity 1: las 11 filas quedaron marcadas y con su referencia ==================
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE nutrition_source = 'bedca' AND nutrition_source_ref LIKE 'bedca:%' AND fdc_id IS NULL;
    IF _n <> 11 THEN
        RAISE EXCEPTION '[P1-BEDCA-DEPROXY-ES] % filas bedca completas, esperadas 11', _n;
    END IF;
END $$;

-- == Sanity 2: el cluster de los 7 embutidos DEJO de ser un bloque identico ========
-- Es la prueba directa de que la de-proxyficacion surtio efecto: antes las 7 filas
-- tenian las MISMAS kcal.
DO $$
DECLARE _distintas int;
BEGIN
    SELECT COUNT(DISTINCT kcal_per_100g) INTO _distintas
    FROM public.master_ingredients WHERE name IN (
        'Chistorra', 'Chorizo español', 'Chorizo mexicano', 'Chorizo santarrosano',
        'Chorizo verde', 'Longaniza puertorriqueña', 'Sobrasada');
    IF _distintas < 3 THEN
        RAISE EXCEPTION '[P1-BEDCA-DEPROXY-ES] los embutidos siguen con % valor(es) de kcal', _distintas;
    END IF;
END $$;

-- == Sanity 3: Atwater sobre lo escrito ============================================
DO $$
DECLARE _mal int;
BEGIN
    SELECT COUNT(*) INTO _mal FROM public.master_ingredients
    WHERE nutrition_source = 'bedca' AND kcal_per_100g > 0
      AND ABS((4*protein_g_per_100g + 4*carbs_g_per_100g + 9*fats_g_per_100g)
              - kcal_per_100g) / kcal_per_100g > 0.12;
    IF _mal > 0 THEN
        RAISE EXCEPTION '[P1-BEDCA-DEPROXY-ES] % filas bedca divergen >12%% en Atwater', _mal;
    END IF;
END $$;
