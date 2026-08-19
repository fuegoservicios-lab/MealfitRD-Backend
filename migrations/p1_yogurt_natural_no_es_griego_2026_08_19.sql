-- [P1-YOGURT-NATURAL · 2026-08-19] La fila `Yogurt` cargaba el perfil del GRIEGO.
--
-- Diagnostico verificado en Neon prod (SELECT, 2026-08-19): `Yogurt` (slug 'yogurt',
-- alias literal 'yogurt regular') y `Yogurt griego sin azucar` (slug 'yogurt-griego')
-- comparten fdc_id 330137 y tienen valores BYTE-IDENTICOS:
--     59.1 kcal | 10.3 g prot | 0.37 g grasa | 3.64 g carbs
-- Ese es el perfil del yogur griego 0%. Un yogur natural entero ronda 3.5 g de
-- proteina, no 10.3: la fila sobreestimaba proteina ~3x, en tráfico dominicano real
-- y en la columna que MAS pesa en el portion_solver.
--
-- [CORRECCION · 2026-08-19, mismo dia] La version original de este comentario decia
-- que la procedencia estaba MUERTA porque `GET /fdc/v1/food/330137` devuelve HTTP 404.
-- El 404 es real, la conclusion era FALSA: 330137 es un registro de tipo `Foundation`,
-- y el endpoint de DETALLE no sirve ese tipo — el BUSCADOR si lo conoce, y devuelve
-- «Yogurt, Greek, plain, nonfat» con los macros exactos de la fila. El id esta vivo;
-- lo que estaba mal era mi sonda. Un barrido posterior de los 288 fdc_id del catalogo
-- confirmo CERO ids muertos: los 8 sospechosos eran los 7 `Foundation` + uno que fallo
-- transitoriamente.
--
-- Lo que SI era cierto, y es lo que esta migracion arregla, no cambia: `Yogurt` (alias
-- literal «yogurt regular») y `Yogurt griego sin azucar` compartian fdc_id y tenian
-- valores BYTE-IDENTICOS, y los de la fila generica eran los del griego.
-- (Auditoria completa de los 20 grupos con fdc_id compartido: ver
-- docs/superpowers/plans/2026-08-19-catalogo-metadata-beta.md §3.)
--
-- Decision del dueno (2026-08-19): `Yogurt` = yogur NATURAL. Fuente nueva
-- `Yogurt, plain, whole milk` (SR Legacy, **fdc 171284**), panel de 23 columnas
-- traido con el mismo mapeo NUM del script de altas (scripts/fetch_usda_foods_*.py),
-- no transcrito a mano.
--   Atwater: 4(3.47) + 4(4.66) + 9(3.25) = 61.77 vs 61.0 declaradas = 1.3% de
--   divergencia, muy por debajo del 12% que los scripts de alta marcan.
--
-- La fila del GRIEGO no se toca aqui: sus valores son un perfil griego 0% plausible y
-- correcto, y conserva legitimamente el fdc 330137 — que, corregida la sonda, resulta
-- ser justo «Yogurt, Greek, plain, nonfat».
--
-- RESIDUAL DECLARADO: omega3_ala_g_per_100g se deja en 0.007 (heredado del proxy
-- griego) porque SR Legacy no reporta el nutriente 851 para fdc 171284. Un yogur
-- entero tiene mas ALA que uno desnatado, asi que 0.007 SUBESTIMA — direccion
-- conservadora, y la columna es NOT NULL (P1-EXTENDED-MICROS-GUARD). Inventar el
-- numero seria peor que dejarlo bajo: no se inventa nada sin fuente.
--
-- Idempotente: filtra por slug exacto; re-ejecutar reescribe los mismos valores.

UPDATE public.master_ingredients SET
    kcal_per_100g               = 61.0,
    protein_g_per_100g          = 3.47,
    carbs_g_per_100g            = 4.66,
    fats_g_per_100g             = 3.25,
    fiber_g_per_100g            = 0.0,
    sugars_g_per_100g           = 4.66,     -- era 0: imposible en un yogur (lactosa)
    saturated_fat_g_per_100g    = 2.096,
    sodium_mg_per_100g          = 46.0,
    cholesterol_mg_per_100g     = 13.0,     -- era NULL
    calcium_mg_per_100g         = 121.0,
    iron_mg_per_100g            = 0.05,
    potassium_mg_per_100g       = 155.0,
    magnesium_mg_per_100g       = 12.0,
    phosphorus_mg_per_100g      = 95.0,
    zinc_mg_per_100g            = 0.59,
    vitamin_d_mcg_per_100g      = 0.1,
    vitamin_b12_mcg_per_100g    = 0.37,
    folate_mcg_dfe_per_100g     = 7.0,
    vitamin_a_mcg_rae_per_100g  = 27.0,
    vitamin_c_mg_per_100g       = 0.5,
    vitamin_e_mg_per_100g       = 0.06,
    vitamin_k_mcg_per_100g      = 0.2,
    selenium_mcg_per_100g       = 2.2,
    fdc_id                      = 171284,
    nutrition_source            = 'usda',
    nutrition_source_date       = DATE '2026-08-19'
WHERE slug = 'yogurt';

-- == Sanity 1: la fila quedo escrita y ya NO comparte fdc_id con el griego =======
DO $$
DECLARE _prot numeric; _fdc int; _comparten int;
BEGIN
    SELECT protein_g_per_100g, fdc_id INTO _prot, _fdc
    FROM public.master_ingredients WHERE slug = 'yogurt';
    IF _prot IS NULL THEN
        RAISE EXCEPTION '[P1-YOGURT-NATURAL] no existe la fila slug=yogurt';
    END IF;
    IF _prot > 5 THEN
        RAISE EXCEPTION '[P1-YOGURT-NATURAL] proteina % g sigue en rango griego (>5)', _prot;
    END IF;
    IF _fdc <> 171284 THEN
        RAISE EXCEPTION '[P1-YOGURT-NATURAL] fdc_id quedo en % en vez de 171284', _fdc;
    END IF;
    SELECT COUNT(*) INTO _comparten FROM public.master_ingredients
    WHERE fdc_id = 171284 AND slug <> 'yogurt';
    IF _comparten > 0 THEN
        RAISE EXCEPTION '[P1-YOGURT-NATURAL] el fdc 171284 quedo compartido con % fila(s) mas', _comparten;
    END IF;
END $$;

-- == Sanity 2: las dos filas dejaron de ser gemelas ==============================
DO $$
DECLARE _iguales int;
BEGIN
    SELECT COUNT(*) INTO _iguales FROM public.master_ingredients a
    JOIN public.master_ingredients b ON b.slug = 'yogurt-griego'
    WHERE a.slug = 'yogurt'
      AND a.kcal_per_100g = b.kcal_per_100g
      AND a.protein_g_per_100g = b.protein_g_per_100g;
    IF _iguales > 0 THEN
        RAISE EXCEPTION '[P1-YOGURT-NATURAL] Yogurt y Yogurt griego siguen con macros identicos';
    END IF;
END $$;
