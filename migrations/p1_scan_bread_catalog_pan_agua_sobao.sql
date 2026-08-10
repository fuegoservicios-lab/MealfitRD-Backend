-- [P1-BREAD-CATALOG · 2026-08-10] Pan de agua y pan sobao entran al catálogo.
--
-- POR QUÉ: son los dos panes de mesa más comunes de RD y no existían entre los 204
-- alimentos. El dueño fotografió pan de agua con el escáner de la Nevera y, aunque el
-- modelo lo hubiera nombrado bien, no había fila que recibiera esa lectura: salía
-- «sin match en el catálogo» y no se podía agregar. Era un techo de DATOS, no de
-- código — arreglar el matcher y la resolución de la cámara no podía cerrarlo.
--
-- POR QUÉ NO BASTABA UN ALIAS SOBRE «Pan blanco familiar»: ese se vende por PAQUETE
-- (754 g, RD$140) y estos se venden por UNIDAD (~60-70 g, ~RD$15). Aliasarlos habría
-- hecho que «tengo 2 panes de agua» se contabilizara como 2 paquetes de pan de molde
-- — cantidad y precio de la lista de compras equivocados por un factor de diez. Y su
-- caducidad real es de días, no de dos semanas: son de panadería, sin conservantes.
--
-- PROCEDENCIA DE LAS CIFRAS, declarada en las propias filas para que sea auditable:
--   · `nutrition_source = 'manual'` — el panel está introducido a mano, informado por
--     los análogos de USDA FoodData Central (pan francés/vienés para el de agua; pan
--     blanco enriquecido con grasa y azúcar añadidos para el sobao). NO se marca
--     `'usda'` a propósito: no existe ficha FDC de «pan de agua», y ponerlo dejaría
--     la columna `fdc_id` vacía contradiciendo su propia etiqueta. Mismo criterio que
--     la fila del Casabe. Si aparecen datos de laboratorio dominicanos, se sustituye.
--   · `price_confidence = 'low'` — el precio unitario es una estimación de colmado,
--     no una captura de tienda como las de `nacional_tienda`. Los crons de precio
--     que ponderan por confianza lo tratarán en consecuencia.
--
-- Idempotente: `ON CONFLICT (slug) DO NOTHING` + sanity check al final.

INSERT INTO public.master_ingredients (
    slug, name, category, aliases,
    density_g_per_unit, shelf_life_days,
    price_per_lb, price_per_unit, market_container, container_weight_g, default_unit,
    kcal_per_100g, protein_g_per_100g, carbs_g_per_100g, fats_g_per_100g,
    fiber_g_per_100g, sodium_mg_per_100g, sugars_g_per_100g, saturated_fat_g_per_100g,
    cholesterol_mg_per_100g, vitamin_d_mcg_per_100g, calcium_mg_per_100g,
    iron_mg_per_100g, vitamin_b12_mcg_per_100g, potassium_mg_per_100g,
    magnesium_mg_per_100g, phosphorus_mg_per_100g,
    zinc_mg_per_100g, folate_mcg_dfe_per_100g, vitamin_a_mcg_rae_per_100g,
    vitamin_c_mg_per_100g, vitamin_e_mg_per_100g, vitamin_k_mcg_per_100g,
    selenium_mcg_per_100g, omega3_ala_g_per_100g,
    nutrition_source, nutrition_source_date, is_dominican_cultivar,
    price_per_unit_base, price_base_period, price_source, price_captured_at,
    price_confidence, prep_methods, ready_to_eat
) VALUES
-- Pan de agua: harina, agua, sal y levadura. Sin grasa ni azúcar añadidos — por eso
-- su perfil se parece al del pan francés y NO al del pan de molde.
(
    'pan-de-agua', 'Pan de agua', 'Despensa',
    ARRAY['pan de agua', 'pan agua', 'pan de agua dominicano', 'pan criollo', 'pan de panaderia'],
    60, 3,
    0, 15, 'unidad', 60, 'unidad',
    270, 9.0, 52.0, 2.0,
    2.3, 550, 2.5, 0.5,
    0, 0, 40,
    3.3, 0, 120,
    27, 100,
    0.8, 130, 0,
    0, 0.2, 1.5,
    25, 0.06,
    'manual', DATE '2026-08-10', TRUE,
    15, '2026-08', 'estimado_colmado', DATE '2026-08-10',
    'low', ARRAY['tostar', 'ninguno'], TRUE
),
-- Pan sobao: enriquecido con grasa y azúcar — más denso en calorías que el de agua.
-- La diferencia NO es cosmética: son ~30 kcal y 4 g de grasa por cada 100 g.
(
    'pan-sobao', 'Pan sobao', 'Despensa',
    ARRAY['pan sobao', 'pan sobado', 'sobao', 'pan dulce de mesa'],
    70, 3,
    0, 18, 'unidad', 70, 'unidad',
    300, 8.0, 53.0, 6.0,
    2.0, 480, 6.0, 2.5,
    0, 0, 45,
    3.2, 0, 115,
    25, 95,
    0.8, 125, 0,
    0, 0.4, 2.0,
    24, 0.08,
    'manual', DATE '2026-08-10', TRUE,
    18, '2026-08', 'estimado_colmado', DATE '2026-08-10',
    'low', ARRAY['tostar', 'ninguno'], TRUE
)
ON CONFLICT (slug) DO NOTHING;

-- Sanity: las dos filas existen y se venden por UNIDAD (si alguien las "corrige" a
-- paquete, vuelve el error de magnitud que motivó darles fila propia).
DO $$
DECLARE
    n INTEGER;
BEGIN
    SELECT COUNT(*) INTO n
    FROM public.master_ingredients
    WHERE slug IN ('pan-de-agua', 'pan-sobao')
      AND default_unit = 'unidad'
      AND container_weight_g BETWEEN 40 AND 120;
    IF n <> 2 THEN
        RAISE EXCEPTION 'P1-BREAD-CATALOG: se esperaban 2 panes de mesa por unidad, hay %', n;
    END IF;
END $$;
