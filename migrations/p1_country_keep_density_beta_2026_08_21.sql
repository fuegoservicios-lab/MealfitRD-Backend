-- [P1-COUNTRY-KEEP-RESPECT-QTY · 2026-08-21] Densidad (g por taza) de las filas VOLUMÉTRICAS del
-- lote de catálogo-país de Fase 2.
--
-- POR QUÉ AHORA, Y NO ANTES NI DESPUÉS. Mientras la rama `keep` del agregador servía TODA la
-- comida de catálogo-país a 150 g fijos, la ausencia de densidad estaba TAPADA: daba igual que la
-- receta dijera «1 taza de Nata», porque el resultado era 150 g pasara lo que pasara. Al invertir
-- la precedencia (la receta manda; el default queda como último recurso), esas líneas empiezan a
-- convertirse de verdad — y sin densidad caen al fallback genérico `DEFAULT_G_PER_TAZA = 150`,
-- que para una nata real (~238 g/taza) se queda a la mitad y para dos cucharadas de sirope se
-- pasa. Un arreglo bueno que destapa uno malo no está terminado: van en el mismo cambio.
--
-- ALCANCE DELIBERADAMENTE ESTRECHO. De las 136 filas del catálogo sin densidad sólo se tocan las
-- que las recetas piden POR VOLUMEN (cremas, siropes, aceites, líquidos y untables). Los sólidos
-- que se venden y se piden por peso o por envase (Chocolate de mesa en tableta, Masa para pie,
-- Especias para arroz con dulce en sobre) NO llevan densidad: inventarles una sería peor que no
-- tenerla — la lección de P1-CATALOGO-READY es que la densidad casi nunca hace falta y que el
-- criterio es el USO REAL, no la cuota de cobertura.
--
-- FUENTE. Tablas estándar volumen→peso (USDA FoodData Central, "cup" portions) para la referencia
-- genérica de cada producto. Son valores de PRESENTACIÓN, no de composición nutricional: no tocan
-- kcal ni macros, sólo cómo se traduce «1 taza» a gramos en la lista de compras.
--
-- Idempotente: el `WHERE density_g_per_cup IS NULL` hace que una segunda ejecución no pise un
-- valor curado a mano después. SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y
-- backend/migrations/.

UPDATE public.master_ingredients AS m
SET density_g_per_cup = v.g_per_cup
FROM (VALUES
    -- Cremas y lácteos líquidos
    ('Nata',                   238.0),  -- heavy/whipping cream
    ('Crema agria',            230.0),  -- sour cream
    ('Crema mexicana',         235.0),  -- crema de mesa, algo más fluida que la agria
    ('Crema mitad y mitad',    242.0),  -- half-and-half
    ('Suero de mantequilla',   245.0),  -- buttermilk
    ('Suero costeño',          240.0),  -- cultivada costeña, cuerpo entre buttermilk y crema agria
    ('Natilla',                250.0),  -- custard
    ('Arequipe',               300.0),  -- dulce de leche (denso, azucarado)
    -- Siropes, aceites y salsas embotelladas
    ('Jarabe de arce',         322.0),  -- maple syrup
    ('Aceite de achiote',      218.0),  -- aceite vegetal infusionado
    ('Salsa barbacoa',         285.0),  -- barbecue sauce
    ('Salsa de salchicha',     240.0),  -- sausage gravy
    -- Untables
    ('Hummus',                 246.0)
) AS v(name, g_per_cup)
WHERE m.name = v.name
  AND m.density_g_per_cup IS NULL;

-- Sanity ACOTADO A LAS FILAS DE ESTA MIGRACIÓN, y ésa es la corrección importante: la primera
-- versión barría la tabla entera con un rango [50, 500] y el dry-run transaccional la abortó
-- contra 12 filas PREEXISTENTES Y CORRECTAS — Laurel 14, Cilantro 16, Kale 21, Espinacas 30,
-- Lechuga 36: las hojas verdes y las hierbas secas pesan eso de verdad por taza. Un guard que
-- grita contra datos buenos se apaga en una semana; el rango sólo tiene sentido dentro de la
-- CLASE que esta migración toca (cremas, siropes, aceites y untables: 218-322 g/taza).
DO $$
DECLARE
    _lote text[] := ARRAY['Nata', 'Crema agria', 'Crema mexicana', 'Crema mitad y mitad',
                          'Suero de mantequilla', 'Suero costeño', 'Natilla', 'Arequipe',
                          'Jarabe de arce', 'Aceite de achiote', 'Salsa barbacoa',
                          'Salsa de salchicha', 'Hummus'];
    _sin_densidad int;
    _absurdas int;
BEGIN
    SELECT COUNT(*) INTO _sin_densidad
    FROM public.master_ingredients
    WHERE name = ANY(_lote) AND density_g_per_cup IS NULL;
    IF _sin_densidad > 0 THEN
        RAISE EXCEPTION 'P1-COUNTRY-KEEP-RESPECT-QTY: % filas volumetricas siguen sin densidad', _sin_densidad;
    END IF;

    SELECT COUNT(*) INTO _absurdas
    FROM public.master_ingredients
    WHERE name = ANY(_lote)
      AND (density_g_per_cup < 150 OR density_g_per_cup > 400);
    IF _absurdas > 0 THEN
        RAISE EXCEPTION 'P1-COUNTRY-KEEP-RESPECT-QTY: % cremas/siropes con densidad fuera de [150,400]', _absurdas;
    END IF;
END $$;
