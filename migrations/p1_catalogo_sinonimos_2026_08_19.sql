-- [P1-CATALOGO-SINONIMOS · 2026-08-19] El mismo alimento deja de tener dos perfiles
-- segun como lo llames.
--
-- `Queso ricotta`/`Requeson` y `Judias blancas`/`Habichuelas blancas` son el MISMO
-- alimento con el nombre de dos paises. Y tenian numeros distintos: 151 vs 150 kcal y
-- 342,4 vs 333. Un usuario espanol y uno dominicano comiendo lo mismo recibian macros
-- distintas — y el guard de coherencia, el solver de porciones y el contador del diario
-- resuelven POR CADENA, asi que ninguno de los tres podia notarlo.
--
-- POR QUE NO SE BORRA LA DUPLICADA. Medido en prod (2026-08-19): `Requeson` y
-- `Judias blancas` tienen CERO referencias en user_inventory, supermarket_products,
-- consumed_meals, ingredient_frequencies y plan_data. Borrarlas hoy no romperia nada.
-- Pero tienen cero referencias porque son los nombres ESPANOLES y Espana aun no tiene
-- usuarios reales — no porque sobren. Se anadieron a proposito en el catalogo de ES
-- (P1-COUNTRY-SYSTEM-F2) y borrarlas romperia los planes espanoles el dia del flip.
-- «Cero referencias» en un pais que aun no ha arrancado no es «no se usa».
--
-- REGLA DE FUSION: manda la fila con procedencia VERIFICADA (la que conserva su
-- fdc_id); pero donde esa no tiene dato (NULL, o un 0 de relleno) y la otra si, se
-- conserva el dato de la otra. No es un detalle: la densidad por taza la tiene el
-- canonico en un par y el SINONIMO en el otro, asi que un «gana el canonico» a secas
-- habria borrado la densidad de Habichuelas blancas.
--
-- Idempotente: escribe valores absolutos filtrando por `name` exacto.

-- == «Queso ricotta» ≡ «Requesón» ======================================
UPDATE public.master_ingredients SET
    density_g_per_cup = 246,
    density_g_per_unit = NULL,
    kcal_per_100g = 151,
    protein_g_per_100g = 7.54,
    carbs_g_per_100g = 7.27,
    fats_g_per_100g = 10.2,
    fiber_g_per_100g = 0,
    sodium_mg_per_100g = 110,
    vitamin_d_mcg_per_100g = 0.2,
    calcium_mg_per_100g = 206,
    iron_mg_per_100g = 0.13,
    vitamin_b12_mcg_per_100g = 0.85,
    sugars_g_per_100g = 0.27,
    potassium_mg_per_100g = 219,
    magnesium_mg_per_100g = 20,
    phosphorus_mg_per_100g = 154,
    saturated_fat_g_per_100g = 6.42,
    cholesterol_mg_per_100g = 49,
    zinc_mg_per_100g = 0.53,
    folate_mcg_dfe_per_100g = 4,
    vitamin_a_mcg_rae_per_100g = 120,
    vitamin_c_mg_per_100g = 0,
    vitamin_e_mg_per_100g = 0.11,
    vitamin_k_mcg_per_100g = 1.1,
    selenium_mcg_per_100g = 5.9,
    omega3_ala_g_per_100g = 0.039,
    nutrition_source_ref = COALESCE(nutrition_source_ref, 'sinonimo:Queso ricotta')
    WHERE name IN ('Queso ricotta', 'Requesón');

-- == «Judías blancas» ≡ «Habichuelas blancas» ======================================
--   density_g_per_cup: se toma 180 del sinonimo (el canonico no lo tiene)
--   omega3_ala_g_per_100g: se toma 0.166 del sinonimo (el canonico trae 0 (relleno))
UPDATE public.master_ingredients SET
    density_g_per_cup = 180,
    density_g_per_unit = NULL,
    kcal_per_100g = 333,
    protein_g_per_100g = 23.4,
    carbs_g_per_100g = 60.3,
    fats_g_per_100g = 0.85,
    fiber_g_per_100g = 15.2,
    sodium_mg_per_100g = 16,
    vitamin_d_mcg_per_100g = 0,
    calcium_mg_per_100g = 240,
    iron_mg_per_100g = 10.4,
    vitamin_b12_mcg_per_100g = 0,
    sugars_g_per_100g = 2.11,
    potassium_mg_per_100g = 1800,
    magnesium_mg_per_100g = 190,
    phosphorus_mg_per_100g = 301,
    saturated_fat_g_per_100g = 0.219,
    cholesterol_mg_per_100g = 0,
    zinc_mg_per_100g = 3.67,
    folate_mcg_dfe_per_100g = 388,
    vitamin_a_mcg_rae_per_100g = 0,
    vitamin_c_mg_per_100g = 0,
    vitamin_e_mg_per_100g = 0.21,
    vitamin_k_mcg_per_100g = 5.6,
    selenium_mcg_per_100g = 12.8,
    omega3_ala_g_per_100g = 0.166,
    nutrition_source_ref = COALESCE(nutrition_source_ref, 'sinonimo:Judías blancas')
    WHERE name IN ('Judías blancas', 'Habichuelas blancas');

-- == Sanity: los pares quedan con nutrientes IDENTICOS ============================
DO $$
DECLARE _difs int;
BEGIN
    SELECT COUNT(*) INTO _difs FROM (
        SELECT 1 FROM public.master_ingredients a, public.master_ingredients b
        WHERE a.name = 'Queso ricotta' AND b.name = 'Requesón'
          AND (a.kcal_per_100g IS DISTINCT FROM b.kcal_per_100g
            OR a.protein_g_per_100g IS DISTINCT FROM b.protein_g_per_100g
            OR a.carbs_g_per_100g IS DISTINCT FROM b.carbs_g_per_100g
            OR a.fats_g_per_100g IS DISTINCT FROM b.fats_g_per_100g)
        UNION ALL
        SELECT 1 FROM public.master_ingredients a, public.master_ingredients b
        WHERE a.name = 'Judías blancas' AND b.name = 'Habichuelas blancas'
          AND (a.kcal_per_100g IS DISTINCT FROM b.kcal_per_100g
            OR a.protein_g_per_100g IS DISTINCT FROM b.protein_g_per_100g
            OR a.carbs_g_per_100g IS DISTINCT FROM b.carbs_g_per_100g
            OR a.fats_g_per_100g IS DISTINCT FROM b.fats_g_per_100g)
    ) t;
    IF _difs > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-SINONIMOS] % par(es) de sinonimos siguen con macros distintos', _difs;
    END IF;
END $$;
