-- [P1-CATALOGO-DICE-QUE-PRODUCTO · 2026-09-10]
-- El catalogo tenia numeros de productos CONCRETOS y no decia cuales.
--
-- De donde sale: tercera ronda de juicio humano sobre los 19 platos de proteina. Nueve notas
-- pedian lo mismo: «especificar el porcentaje de grasa del cottage y verificar macros y sodio
-- segun su etiqueta», «especificar salsa de soya reducida en sodio o recalcular», «verificar
-- macros segun el tofu utilizado». Verificado contra USDA FoodData Central (API, 2026-09-10) y
-- contra la tabla `supermarket_products` del dueno:
--
--   fila                      fdc_id   lo que USDA dice que es                 lo que vende su super
--   Queso cottage             173417   cottage lowfat 1 % milkfat              2 % y 4 % — NO 1 %
--   Yogurt griego sin azucar  330137   Greek, plain, NONFAT                    (ninguno mapeado)
--   Salsa de soya             172474   soy sauce, REDUCED SODIUM               Kikkoman Baja en Sal,
--                                                                              La Choy Lite (y normales)
--   Tofu firme                172476   tofu raw, REGULAR  (76 kcal)            firme/extra/super firme
--
-- Cuatro decisiones:
--
-- 1. QUESO COTTAGE pasa a 2 % (fdc 172182). El dueno pidio verificar «segun SU etiqueta», y su
--    supermercado no vende cottage 1 %: los numeros eran de un producto que no puede comprar. Se
--    reescribe la fila ENTERA, no cuatro columnas — una fila mitad 1 % y mitad 2 % seria peor que
--    cualquiera de las dos (precedente: P1-YOGURT-NATURAL, 2026-08-19). Por 100 g: proteina
--    12,4 -> 10,45 g, grasa 1,02 -> 2,27 g, sodio 406 -> 308 mg, calcio 61 -> 111 mg.
--    kcal toma el valor de ENERGIA de USDA (81), como hizo el precedente, no el Atwater de la fila.
--
-- 2. TOFU FIRME: los VALORES ya eran de tofu firme (144 kcal, 17,3 g, 8,7 g = fdc 172475
--    «Tofu, raw, firm, prepared with calcium sulfate», verificado) y el PUNTERO apuntaba al tofu
--    normal (172476, 76 kcal). Se corrige el puntero; ningun numero cambia. Es la clase que
--    P1-BEDCA-DEPROXY-ES dejo escrita: auditar ids DUPLICADOS no ve el UNICO mal apuntado.
--
-- 3. SALSA DE SOYA: los numeros son de soya REDUCIDA en sodio (2.890 mg/100 g). Una normal ronda el
--    doble. Se glosa «reducida en sodio» para que la lista de compras pida la que cuadra con los
--    numeros — y su super la vende. Se glosa en vez de repuntar a la normal porque es la opcion que
--    el propio dueno escribio primero y porque el techo de sodio del dia (P1-SODIO-DEL-DIA-
--    DETERMINISTA) agradece la reducida.
--
-- 4. YOGURT GRIEGO SIN AZUCAR: los numeros son de griego NATURAL 0 %. Se glosa asi. El nombre decia
--    «sin azucar» y callaba la grasa, que es lo que el dueno pregunto.
--
-- RESTRICCION DURA (la de P1-GLOSS-MAPUEY-DO): `master_ingredients.name` es identidad canonica y
-- NO se toca. `gloss_es` es display-only — nunca entra en aliases ni en matching.
--
-- Queda FUERA, dicho: «Queso de hoja» tiene valores metidos a mano (nutrition_source 'manual', sin
-- fdc_id) y USDA no tiene ese queso. Verificarlo pide una etiqueta de San Juan, La Zarina o Aguila;
-- no se inventa una referencia.
--
-- Idempotente: re-aplicarla no cambia el resultado.

UPDATE public.master_ingredients SET
    kcal_per_100g               = 81.0,
    protein_g_per_100g          = 10.45,
    carbs_g_per_100g            = 4.76,
    fats_g_per_100g             = 2.27,
    fiber_g_per_100g            = 0.0,
    sugars_g_per_100g           = 4.0,
    saturated_fat_g_per_100g    = 1.235,
    sodium_mg_per_100g          = 308.0,
    cholesterol_mg_per_100g     = 12.0,
    calcium_mg_per_100g         = 111.0,
    iron_mg_per_100g            = 0.13,
    potassium_mg_per_100g       = 125.0,
    magnesium_mg_per_100g       = 9.0,
    phosphorus_mg_per_100g      = 150.0,
    zinc_mg_per_100g            = 0.51,
    vitamin_d_mcg_per_100g      = 0.0,
    vitamin_b12_mcg_per_100g    = 0.47,
    folate_mcg_dfe_per_100g     = 8.0,
    vitamin_a_mcg_rae_per_100g  = 68.0,
    vitamin_c_mg_per_100g       = 0.0,
    vitamin_e_mg_per_100g       = 0.08,
    vitamin_k_mcg_per_100g      = 0.0,
    selenium_mcg_per_100g       = 11.9,
    omega3_ala_g_per_100g       = 0.007,
    fdc_id                      = 172182,
    nutrition_source            = 'usda',
    nutrition_source_date       = DATE '2026-09-10',
    gloss_es                    = '2 % de grasa'
WHERE name = 'Queso cottage';

UPDATE public.master_ingredients
SET fdc_id = 172475
WHERE name = 'Tofu firme' AND fdc_id IS DISTINCT FROM 172475;

UPDATE public.master_ingredients
SET gloss_es = 'reducida en sodio'
WHERE name = 'Salsa de soya' AND gloss_es IS DISTINCT FROM 'reducida en sodio';

UPDATE public.master_ingredients
SET gloss_es = 'natural, 0 % de grasa'
WHERE name = 'Yogurt griego sin azúcar' AND gloss_es IS DISTINCT FROM 'natural, 0 % de grasa';

-- Sanity: las cuatro filas existen y quedaron como se dice arriba; el nombre canonico no se movio.
DO $$
DECLARE
    _n INT;
BEGIN
    SELECT count(*) INTO _n FROM public.master_ingredients
    WHERE (name = 'Queso cottage' AND fdc_id = 172182 AND protein_g_per_100g = 10.45 AND gloss_es = '2 % de grasa')
       OR (name = 'Tofu firme' AND fdc_id = 172475 AND kcal_per_100g = 144)
       OR (name = 'Salsa de soya' AND gloss_es = 'reducida en sodio')
       OR (name = 'Yogurt griego sin azúcar' AND gloss_es = 'natural, 0 % de grasa');
    IF _n <> 4 THEN
        RAISE EXCEPTION '[P1-CATALOGO-DICE-QUE-PRODUCTO] esperaba 4 filas corregidas, hay %', _n;
    END IF;
END $$;
