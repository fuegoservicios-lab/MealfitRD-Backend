-- [P1-PLAN-LOTE-856 · 2026-09-29] Alias que faltan en el catálogo de los países beta.
--
-- LO QUE SE VIO (validación beta G24, ES D1 Almuerzo): la lista dice «100 g de filete de merluza» y el paso «mide
-- 170 g de merluza». La comida sale con `_recipe_contract_final=None`: el índice del contrato de receta
-- (`culinary_coherence.build_culinary_index`, que carga nombre + `aliases`) no empareja «merluza» con NINGUNA fila, así
-- que ni el reparador (170 → 100) ni el medidor V4 lo tocan. Y la lista de compras la perdía entera: «Filete de
-- merluza» no resolvía a ninguna fila y el filtro verified-only la soltaba. En DO esos medidores están en 0.
--
-- LA FILA «Filete de merluza» NO EXISTE (354 filas leídas el 29-sep, SELECT de prod). La merluza es un pescado blanco
-- magro sin fila propia, igual que el chillo, que ya es alias de «Filete de pescado blanco» (95,7 kcal, 20,1 g de
-- proteína, 1,7 g de grasa por 100 g). La regla de P2-WHITE-FISH-ALIAS-SPLIT es «un alias resuelve a UNA fila»: mero y
-- tilapia salieron de la genérica porque tienen fila; la merluza no la tiene, así que va a la genérica, como el chillo.
--
-- CÓMO SE ELIGIERON (sin IA; catálogo = SELECT de prod; corpus = 834 planes guardados, 10 480 comidas, 432 beta):
--   1. Auditoría de las 140 filas beta (las altas del 17-ago de ES/MX/CO/PR/US): el núcleo con que una receta las
--      nombraría (sin «Filete de», «en lata», «fresco»…, y su cabeza) contra el índice del contrato, la Nevera
--      (`constants.pantry_names_match`), el tracking (`normalize_ingredient_for_tracking`) y la lista de compras.
--      74 núcleos: 3 ya resuelven; 64 AMBIGUOS (token o alias de otra fila, varias filas beta, forma genérica, o el
--      SSOT de tracking los colapsa a otro término); 7 sin ambigüedad en el dato, de los que 5 lo son de SENTIDO y no
--      entran: «chocolate» (en ES/US es chocolate negro, no el de mesa mexicano; el índice es de todos los países),
--      «jarabe» (agave, maíz, simple), «especias», «sémola» (de trigo) y «bolitas» (una forma: «divide la masa en 3
--      bolitas»). Entran los otros 2: «panceta» y «ron» (0 comidas hoy: sin uso medido, sin ambigüedad).
--   2. Pasada por el corpus: líneas y menciones numéricas de los pasos beta que el índice no resuelve. Salen, además
--      de «agua» (no es una fila) y de núcleos ambiguos: merluza (ES, 3 comidas), almendras laminadas (el nombre
--      español de las fileteadas; ES/CO y 1 DO, 3 comidas) y chile piquín en polvo (MX, 6 comidas).
--   3. Replay antes/después sobre el corpus entero: 0 claves del índice perdidas o con otro dueño; de 7 580 líneas
--      únicas re-parseadas por la lista de compras cambian 5 (merluza ×2 → Filete de pescado blanco, chile piquín en
--      polvo ×3 → Chile en polvo), ninguna DO; el contrato reescribe 1 comida (la de la evidencia: «mide 100 g de
--      merluza») y el scan capa 1 gana 1 V4 (el mismo 170 frente a 100, ahora medido).
--
-- DESCARTADOS POR AMBIGUOS (para el dueño; NO entran): chile (11 filas Chile *), frijol/frijoles, queso, chorizo,
-- jamón, crema, salsa, tortilla, pan, harina, masa, mezcla, aceite, almendra/almendras, nueces/nuez, azúcar, huevos,
-- papas, hoja, carne, suero, panecillos, galletas, lomo, longaniza, salchicha, sazonador, chuleta, aderezo, judías,
-- arándanos, membrillo, ensalada, chili, flor, aceitunas. Y del corpus: «plátano» a secas (en España es el guineo: es
-- una decisión de sentido, no un alias), «arroz rojo», «salsa verde», «cuscús» y «pulpa» (sin fila). Los alias
-- sueltos «maíz»/«soya» siguen pendientes del dueño: esta migración no los toca.
--
-- EFECTOS DECLARADOS: «merluza» y «almendras laminadas» pasan a ser sinónimos en la Nevera (tokens que no se
-- contienen); «panceta», «ron» y «chile piquín en polvo» NO (son la fila con un calificativo quitado o puesto, y la
-- Nevera no los empareja por diseño: P2-PANTRY-REGIONAL-SYNONYMS). Tres filas son beta (Chile en polvo, Panceta
-- ibérica, Ron de cocina); dos son DO y las usan los planes beta (Filete de pescado blanco, Almendras fileteadas).
--
-- Idempotente: cada alias se añade sólo si la fila no lo tiene; la sanity previa exige las filas destino y que
-- ninguna OTRA fila reclame ya el alias (sin acentos ni mayúsculas); la posterior, que cada alias esté UNA vez en su
-- fila. Sólo toca `aliases`. El contrato de receta ve los alias sin reinicio (knob
-- `MEALFIT_RECIPE_CONTRACT_INDEX_BY_CONTENT`); los demás índices en memoria, al siguiente despliegue.
-- Test: tests/test_p1_plan_lote_856.py

-- == Sanity previa: las filas existen y nadie más reclama el alias ======================================
DO $$
DECLARE _falta text; _choca text;
BEGIN
    SELECT string_agg(f.fila, ', ') INTO _falta
      FROM (VALUES ('Filete de pescado blanco'), ('Almendras fileteadas'), ('Chile en polvo'), ('Panceta ibérica'), ('Ron de cocina')) AS f(fila)
     WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients m WHERE m.name = f.fila);
    IF _falta IS NOT NULL THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-856] faltan filas destino: %', _falta;
    END IF;

    SELECT string_agg(v.alias || ' (ya en ' || m.name || ')', ', ') INTO _choca
      FROM (VALUES
        ('Filete de pescado blanco', 'merluza'),
        ('Filete de pescado blanco', 'filete de merluza'),
        ('Almendras fileteadas', 'almendras laminadas'),
        ('Chile en polvo', 'chile piquín en polvo'),
        ('Chile en polvo', 'chile piquin en polvo'),
        ('Panceta ibérica', 'panceta'),
        ('Ron de cocina', 'ron')
      ) AS v(fila, alias)
      JOIN public.master_ingredients m ON m.name <> v.fila
     WHERE translate(lower(m.name), 'áéíóúüñ', 'aeiouun') = translate(lower(v.alias), 'áéíóúüñ', 'aeiouun')
        OR EXISTS (SELECT 1 FROM unnest(COALESCE(m.aliases, ARRAY[]::text[])) a
                    WHERE translate(lower(a), 'áéíóúüñ', 'aeiouun') = translate(lower(v.alias), 'áéíóúüñ', 'aeiouun'));
    IF _choca IS NOT NULL THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-856] alias que ya reclama otra fila (sería ambiguo): %', _choca;
    END IF;
END $$;

-- == Alias ======================================================================================
-- Merluza: pescado blanco magro sin fila propia; la genérica ya lleva «chillo» por la misma razón.
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'merluza')
 WHERE name = 'Filete de pescado blanco'
   AND NOT ('merluza' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- La lista de compras busca el nombre parseado EXACTO («Filete de merluza»): como «filete de mero»/«filete de tilapia».
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'filete de merluza')
 WHERE name = 'Filete de pescado blanco'
   AND NOT ('filete de merluza' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- «Laminadas» es como se dice en España; «almendras» a secas NO (token de 5 filas: la regla 3 del índice lo descarta).
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'almendras laminadas')
 WHERE name = 'Almendras fileteadas'
   AND NOT ('almendras laminadas' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- El chile en polvo de los planes mexicanos; «chile piquín» a secas NO (el chile entero seco es otra compra).
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'chile piquín en polvo')
 WHERE name = 'Chile en polvo'
   AND NOT ('chile piquín en polvo' = ANY(COALESCE(aliases, ARRAY[]::text[])));

UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'chile piquin en polvo')
 WHERE name = 'Chile en polvo'
   AND NOT ('chile piquin en polvo' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- Núcleo de la fila beta: la única panceta del catálogo (la tocineta DO es otra fila con sus alias).
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'panceta')
 WHERE name = 'Panceta ibérica'
   AND NOT ('panceta' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- Núcleo de la fila beta: el único ron del catálogo.
UPDATE public.master_ingredients
   SET aliases = array_append(COALESCE(aliases, ARRAY[]::text[]), 'ron')
 WHERE name = 'Ron de cocina'
   AND NOT ('ron' = ANY(COALESCE(aliases, ARRAY[]::text[])));

-- == Sanity posterior: cada alias, UNA vez, en su fila ==========================================
DO $$
DECLARE _mal text;
BEGIN
    SELECT string_agg(v.fila || ' / ' || v.alias, ', ') INTO _mal
      FROM (VALUES
        ('Filete de pescado blanco', 'merluza'),
        ('Filete de pescado blanco', 'filete de merluza'),
        ('Almendras fileteadas', 'almendras laminadas'),
        ('Chile en polvo', 'chile piquín en polvo'),
        ('Chile en polvo', 'chile piquin en polvo'),
        ('Panceta ibérica', 'panceta'),
        ('Ron de cocina', 'ron')
      ) AS v(fila, alias)
     WHERE (SELECT count(*) FROM public.master_ingredients m, unnest(COALESCE(m.aliases, ARRAY[]::text[])) a
             WHERE m.name = v.fila AND a = v.alias) <> 1;
    IF _mal IS NOT NULL THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-856] alias ausente o repetido en su fila: %', _mal;
    END IF;
END $$;

-- == Sanity: ningún núcleo ambiguo entró como alias de una fila beta ==============================
DO $$
DECLARE _amb text;
BEGIN
    SELECT string_agg(DISTINCT m.name || ' / ' || a, ', ') INTO _amb
      FROM public.master_ingredients m, unnest(COALESCE(m.aliases, ARRAY[]::text[])) a
     WHERE m.name IN ('Chile en polvo', 'Panceta ibérica', 'Ron de cocina', 'Filete de pescado blanco')
       AND translate(lower(a), 'áéíóúüñ', 'aeiouun') IN ('chile', 'chiles', 'queso', 'chorizo', 'jamon', 'frijoles', 'nueces',
                                                        'chocolate', 'jarabe', 'especias', 'semola', 'maiz', 'soya', 'pollo');
    IF _amb IS NOT NULL THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-856] núcleo ambiguo como alias: %', _amb;
    END IF;
END $$;
