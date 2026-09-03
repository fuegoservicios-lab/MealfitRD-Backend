-- [P1-CATALOGO-DENSIDAD-ALIAS · 2026-08-19] Los dos huecos que quedaban en la capa de
-- RESOLUCION y de PORCIONES, cerrados con lo que la medicion dijo que hacia falta.
--
-- == DENSIDAD: el gap eran 3 alimentos, no 139 ==
-- La auditoria contaba 139 filas sin `density_g_per_cup` NI `density_g_per_unit`, y eso
-- parecia una montana. Medido contra planes REALES (60 planes, 4.241 ingredientes):
--   - el 12,2% de los ingredientes se mide por VOLUMEN (taza/cda/cdta/ml) — no es ruido,
--   - pero solo 73 alimentos del catalogo aparecen medidos asi,
--   - y 70 de esos 73 YA tenian densidad.
-- El gap real eran TRES. Las otras 136 filas nunca se miden en tazas: son proteinas por
-- libra y productos empaquetados. Rellenarlas habria sido trabajo por completismo.
--
-- Los tres valores salen de `foodPortions` de USDA, no de una estimacion:
--   Acelgas        1 cup  = 36 g   (y 1 leaf = 48 g -> density_g_per_unit)
--   Salsa inglesa  1 cup  = 275 g  (coherente con su 1 tbsp = 17 g: 16x17 = 272)
--   Aderezo ranch  1 tbsp = 15 g   -> 16 x 15 = 240 g/taza (USDA no publica la taza)
--
-- == ALIAS: 6 filas no tenian ninguno ==
-- Sin alias, el resolver solo las encuentra por su nombre exacto. Cada alias propuesto
-- se paso por un comprobador de COLISIONES contra los ~1.000 nombres y alias del
-- catalogo, porque este repo ya se quemo con subcadenas (`sal` dentro de `salsa`,
-- `pollo` dentro de `repollo`).
--
-- Tres salieron marcados y se ACEPTAN igualmente, porque el indice resuelve por alias
-- MAS LARGO y el token que colisiona es el largo: «ajo molido» vs «ajo», «mandioca» vs
-- «harina de mandioca», «cassava» vs «cassava flour». En los tres el mas largo gana y
-- cada cadena cae donde debe.
--
-- LO QUE EL COMPROBADOR NO PODIA VER. El chequeo de colisiones compara contra los
-- nombres y alias del catalogo VIVO, y eso no incluye las decisiones ya tomadas y
-- registradas en TESTS. «pure de tomate» paso el chequeo (ningun choque de cadenas) y
-- lo tumbo el gate: `test_fix_round_2026_07_29_bad_aliases.py` lo prohibe explicitamente
-- porque el pure tiene macros ~3x mas concentrados que la salsa. Un alias nuevo hay que
-- pasarlo por los dos filtros: colisiones de cadena Y prohibiciones ya escritas.
--
-- Dos se RECHAZAN. El primero, «china» para Naranja. Es el nombre dominicano de la naranja y seria
-- util, pero son cinco letras y colisiona con «col china» (Bok choy). Un token tan corto
-- y ambiguo es justo la clase que ya costo dos incidentes aqui. No compensa.
--
-- Idempotente: valores absolutos y alias reconstruidos por nombre exacto.

-- == Densidad ======================================================================
UPDATE public.master_ingredients SET
    density_g_per_cup = 36.0, density_g_per_unit = 48.0
    WHERE name = 'Acelgas';
UPDATE public.master_ingredients SET density_g_per_cup = 275.0
    WHERE name = 'Salsa inglesa';
UPDATE public.master_ingredients SET density_g_per_cup = 240.0
    WHERE name = 'Aderezo ranch';

-- == Alias =========================================================================
UPDATE public.master_ingredients SET aliases = ARRAY['palta', 'avocado']
    WHERE name = 'Aguacate';
UPDATE public.master_ingredients SET aliases = ARRAY['ajo molido', 'garlic powder']
    WHERE name = 'Ajo en polvo';
UPDATE public.master_ingredients SET aliases = ARRAY['cauliflower']
    WHERE name = 'Coliflor';
UPDATE public.master_ingredients SET aliases = ARRAY['naranja dulce', 'orange']
    WHERE name = 'Naranja';
-- «pure de tomate» NO entra: lo prohibe test_fix_round_2026_07_29_bad_aliases.py, y con
-- razon medida — el pure/pasta de tomate tiene macros ~3x mas concentrados que la salsa
-- (finding 4 de aquella auditoria). Anadirlo reintroducia el error que esa ronda cerro.
UPDATE public.master_ingredients SET aliases = ARRAY['tomato sauce']
    WHERE name = 'Salsa de tomate';
UPDATE public.master_ingredients SET aliases = ARRAY['mandioca', 'cassava']
    WHERE name = 'Yuca';

-- == Sanity 1: ninguna fila del catalogo se queda sin alias =======================
DO $$
DECLARE _sin int;
BEGIN
    SELECT COUNT(*) INTO _sin FROM public.master_ingredients
    WHERE aliases IS NULL OR cardinality(aliases) = 0;
    IF _sin > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-DENSIDAD-ALIAS] % filas siguen sin alias', _sin;
    END IF;
END $$;

-- == Sanity 2: los 3 alimentos medidos por volumen tienen densidad ================
DO $$
DECLARE _falta text;
BEGIN
    SELECT string_agg(name, ', ') INTO _falta FROM public.master_ingredients
    WHERE name IN ('Acelgas', 'Salsa inglesa', 'Aderezo ranch')
      AND density_g_per_cup IS NULL;
    IF _falta IS NOT NULL THEN
        RAISE EXCEPTION '[P1-CATALOGO-DENSIDAD-ALIAS] sin densidad: %', _falta;
    END IF;
END $$;

-- == Sanity 3: «pure de tomate» NO puede volver a entrar =========================
-- Lo cazo el gate, no el comprobador de colisiones. Aqui queda anclado en la DB para
-- que no dependa de que alguien vuelva a correr aquel test.
DO $$
DECLARE _pure int;
BEGIN
    SELECT COUNT(*) INTO _pure FROM public.master_ingredients
    WHERE name = 'Salsa de tomate'
      AND (aliases && ARRAY['pure de tomate', 'puré de tomate', 'pasta de tomate']);
    IF _pure > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-DENSIDAD-ALIAS] pure/pasta de tomate NO puede resolver a Salsa de tomate: macros ~3x mas concentrados';
    END IF;
END $$;

-- == Sanity 4: «china» NO entro como alias =======================================
-- Guard explicito de la decision: si alguien lo anade luego «porque es el nombre
-- dominicano», que se encuentre esto primero.
DO $$
DECLARE _china int;
BEGIN
    SELECT COUNT(*) INTO _china FROM public.master_ingredients
    WHERE 'china' = ANY(aliases);
    IF _china > 0 THEN
        RAISE EXCEPTION '[P1-CATALOGO-DENSIDAD-ALIAS] «china» entro como alias: colisiona con «col china» (Bok choy)';
    END IF;
END $$;
