-- [P1-1 · 2026-05-08] Backfill density_g_per_cup y density_g_per_unit en master_ingredients.
--
-- Contexto: convert_amount() en backend/db_inventory.py asumía 150 g/taza por
-- default cuando faltaba la densidad. Para grasas (~218), miel (~340), peanut
-- butter (~258), eso producía conversiones cruzadas masa↔volumen con error
-- ~30-50% que se propagaba a la deducción de inventario y a la coherencia
-- receta↔lista de compras.
--
-- Tras la refactorización (fallback en cadena master → VOLUMETRIC_DENSITIES →
-- knob MEALFIT_CROSS_UNIT_CONVERSION_STRICT), retornar None es seguro pero
-- los callers saltan la fila — por lo que aún así conviene rellenar el SSOT
-- para los items de mayor frecuencia.
--
-- Valores: g por taza estándar US (1 cup = 236.588 mL). Fuente: USDA + tablas
-- estándar de cocina. Se actualiza solo si el campo está NULL para no pisar
-- valores curados existentes.

UPDATE public.master_ingredients SET density_g_per_cup = 227 WHERE name = 'Mantequilla' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 258 WHERE name = 'Mantequilla de maní' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 248 WHERE name = 'Mostaza' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 248 WHERE name = 'Salsa de soya baja en sodio' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 246 WHERE name = 'Salsa de tomate' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 226 WHERE name = 'Queso crema' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 225 WHERE name = 'Queso cottage' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 100 WHERE name = 'Queso parmesano' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 246 WHERE name = 'Queso ricotta' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 252 WHERE name = 'Leche evaporada' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 135 WHERE name = 'Aceitunas' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 165 WHERE name = 'Atún en agua' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 165 WHERE name = 'Maíz dulce en granos' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 200 WHERE name = 'Gandules' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 208 WHERE name = 'Extracto de vainilla' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 115 WHERE name = 'Proteína en polvo' AND density_g_per_cup IS NULL;

-- Especias (envasadas en "sobre"). Densidades de polvos secos.
UPDATE public.master_ingredients SET density_g_per_cup = 150 WHERE name = 'Ajo en polvo' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 24  WHERE name = 'Albahaca seca' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 124 WHERE name = 'Canela en polvo' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 200 WHERE name = 'Estevia' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 24  WHERE name = 'Orégano dominicano' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 115 WHERE name = 'Pimentón' AND density_g_per_cup IS NULL;
UPDATE public.master_ingredients SET density_g_per_cup = 96  WHERE name = 'Pimienta negra' AND density_g_per_cup IS NULL;

-- Huevo: count→mass usa density_g_per_unit (1 huevo entero ≈ 50 g).
UPDATE public.master_ingredients SET density_g_per_unit = 50 WHERE name = 'Huevo' AND density_g_per_unit IS NULL;
