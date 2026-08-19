-- [P1-PROVENANCE-TRUTHFUL · fix round 2 · 2026-08-19] La corrección de la procedencia
-- tenía ella misma dos procedencias mal escritas.
--
-- DEFECTO 1 — el sentinel de fallo se coló como si fuera una fuente. El generador de la
-- ronda 1 descartaba las descripciones VACÍAS (`if not desc`), pero cuando la API
-- devolvía 429 el caché guardaba la cadena `'SIN RESPUESTA (429)'`, que es perfectamente
-- no-vacía. Resultado: CINCO filas quedaron en producción diciendo
-- `nutrition_source_ref = 'usda:173859 (proxy: SIN RESPUESTA (429))'`. Escribir un
-- mensaje de error donde va la fuente es exactamente el pecado que este P-fix corrige,
-- cometido por el propio P-fix. Un guard que filtra por «vacío» no filtra sentinels: hay
-- que filtrar por «es una descripción válida».
--
-- DEFECTO 2 — «proxy» no es cierto para todas. La auditoría había separado dos clases y
-- la ronda 1 las trató igual:
--   COPIADO       los macros son el valor LITERAL de la fila ajena  -> «proxy» es exacto
--   DIFERENCIADO  los valores se ajustaron a mano y son PROPIOS     -> «proxy» MIENTE
-- `Tilapia` (96 kcal, colesterol 50) no es un proxy de camarón (85 kcal, colesterol 161):
-- sus números son suyos, lo único que estaba mal era el id. Llamarla proxy degrada un
-- dato bueno igual que un fdc_id falso ascendía uno malo — el mismo error, al revés.
--
-- Tres etiquetas, una por situación real:
--   'usda:<id> (proxy: <desc>)'                     valores copiados de esa fila
--   'usda:<id> (id previo; valores propios)'        el id era falso, los datos no
--   'usda:<id> (id previo; desc sin verificar)'     no se pudo consultar la descripción
--
-- Idempotente: filtra por `name` exacto; re-ejecutar reescribe lo mismo.

-- == Clase COPIADO: los valores SÍ vienen de esa fila de USDA =====================
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:173859 (proxy: Sausage, pork, chorizo, link or ground, raw)'
    WHERE name IN ('Chorizo santarrosano', 'Chorizo verde', 'Longaniza puertorriqueña');

UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:169396 (proxy: Peppers, ancho, dried)'
    WHERE name IN ('Chile guajillo', 'Chile mulato');

UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:169108 (proxy: Passion-fruit, (granadilla), purple, raw)'
    WHERE name = 'Curuba';

UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:167750 (proxy: Prickly pears, raw)'      WHERE name = 'Xoconostle';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:167761 (proxy: Soursop, raw)'            WHERE name = 'Borojó';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:170851 (proxy: Cheese, ricotta, whole milk)' WHERE name = 'Requesón';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:170932 (proxy: Spices, pepper, red or cayenne)' WHERE name = 'Chile chipotle';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:171714 (proxy: Breadfruit, raw)'         WHERE name = 'Chontaduro';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:173944 (proxy: Bananas, raw)'            WHERE name = 'Guineo verde';

-- == Clase DIFERENCIADO: el id era falso, pero los valores son PROPIOS ============
-- Estas filas ya tenían números distintos a los de la fila de USDA que reclamaban, así
-- que llamarlas «proxy» las degrada sin motivo. Ejemplos medidos: Tilapia 96 kcal y
-- colesterol 50 (el camarón: 85 y 161); Nueces mixtas 594 (los piñones: 673); Chinola
-- 108.6 (la granadilla: 97).
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:175179 (id previo; valores propios)' WHERE name = 'Tilapia';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:170591 (id previo; valores propios)' WHERE name = 'Nueces mixtas';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:169108 (id previo; valores propios)' WHERE name = 'Chinola';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:169998 (id previo; valores propios)' WHERE name = 'Champús';
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:171320 (id previo; valores propios)' WHERE name = 'Especias para arroz con dulce';

-- == Sin verificar: la API nunca devolvió la descripción de 173443 ================
-- Se dice lo que se sabe y no más. Ambas filas comparten valores idénticos, así que una
-- de las dos es un proxy de la otra — pero sin la descripción real no se sabe cuál, y
-- afirmarlo sería volver a inventar.
UPDATE public.master_ingredients SET
    nutrition_source_ref = 'usda:173443 (id previo; desc sin verificar)'
    WHERE name IN ('Crema mexicana', 'Suero costeño');

-- == Sanity 1: ni un sentinel de error sobrevive en una columna de procedencia =====
DO $$
DECLARE _basura int;
BEGIN
    SELECT COUNT(*) INTO _basura FROM public.master_ingredients
    WHERE nutrition_source_ref IS NOT NULL
      AND (nutrition_source_ref ILIKE '%SIN RESPUESTA%'
        OR nutrition_source_ref ILIKE '%HTTP 4%'
        OR nutrition_source_ref ILIKE '%HTTP 5%'
        OR nutrition_source_ref ILIKE '%None%'
        OR nutrition_source_ref ILIKE '%?%');
    IF _basura > 0 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL fix2] % filas con un sentinel de error como procedencia', _basura;
    END IF;
END $$;

-- == Sanity 2: toda referencia usa una de las TRES etiquetas canónicas =============
DO $$
DECLARE _raras int;
BEGIN
    SELECT COUNT(*) INTO _raras FROM public.master_ingredients
    WHERE nutrition_source_ref LIKE 'usda:%'
      AND nutrition_source_ref !~ '^usda:[0-9]+ \((proxy: .+|id previo; (valores propios|desc sin verificar))\)$';
    IF _raras > 0 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL fix2] % referencias fuera de las 3 etiquetas canonicas', _raras;
    END IF;
END $$;

-- == Sanity 3: no se PIERDE ninguna referencia ====================================
-- Cota INFERIOR, no igualdad: la ronda 3 cierra los grupos que esta dejaba pendientes y
-- sube el total a 20. Un `<> 19` convertiria re-ejecutar esta migracion DESPUES de la
-- ronda 3 en un fallo espurio — el riesgo real es perder referencias, no ganarlas.
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE nutrition_source_ref LIKE 'usda:%';
    IF _n < 19 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL fix2] solo % filas con referencia usda, se esperaban >= 19', _n;
    END IF;
END $$;
