-- [P1-PROVENANCE-TRUTHFUL · ronda 3 · 2026-08-19] Cierre: cero `fdc_id` compartidos.
--
-- La ronda 1 dejó fuera 3 grupos porque `DEMO_KEY` (30 req/hora) se agotó y decidir sin
-- la descripción real habría sido adivinar. Con una `USDA_API_KEY` propia (1.000/hora)
-- se consultaron los tres — y **DOS de las tres conjeturas razonadas estaban INVERTIDAS**.
-- Ese es el valor de haberlos dejado abiertos en vez de cerrarlos «con criterio»:
--
--   fdc 174220 = «Mollusks, scallop, mixed species, raw»
--       Se había razonado por divergencia de valores que el dueño era Mejillones.
--       Es VIEIRA: vieira = scallop, y sus macros cuadran EXACTO (69 / 12.1 / 0.49 /
--       3.18). Mejillones diverge un 25% porque tiene los suyos, los del mejillón (86).
--
--   fdc 175202 = «Beans, white, mature seeds, raw»
--       Se había razonado que el dueño era Habichuelas blancas. Es JUDIAS BLANCAS:
--       clava 333 = 333.0. Habichuelas blancas está en 342.4, ajustada a mano.
--
--   fdc 173443 = «Sour cream, light»
--       Ninguna de las dos conservaba el id tras la ronda 1. Es CREMA MEXICANA (crema
--       ácida) y cuadra exacto en las cuatro columnas. `Suero costeño` es suero
--       fermentado de la costa colombiana — otro producto — y copiaba sus valores.
--
-- El criterio que decide sigue siendo el mismo: **coincidencia EXACTA de valores** con
-- la fila real de USDA, no parecido. Habichuelas blancas queda a 2.8% del valor de USDA
-- y aun así pierde el reclamo frente a Judías blancas, que coincide al decimal.
--
-- Idempotente: filtra por `name` exacto; re-ejecutar reescribe lo mismo.

-- == 173443: Crema mexicana RECUPERA el reclamo (es la fila de USDA) ===============
-- La ronda 1 se lo quitó a las dos por no poder verificar. Ahora consta que es suyo.
UPDATE public.master_ingredients SET
    fdc_id = 173443, nutrition_source = 'usda', nutrition_source_ref = NULL
    WHERE name = 'Crema mexicana';
UPDATE public.master_ingredients SET
    nutrition_source = 'manual',
    nutrition_source_ref = 'usda:173443 (proxy: Sour cream, light)'
    WHERE name = 'Suero costeño';

-- == 174220: el dueño es Vieira, no Mejillones ====================================
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:174220 (id previo; valores propios)'
    WHERE name = 'Mejillones';

-- == 175202: el dueño es Judías blancas, no Habichuelas blancas ===================
UPDATE public.master_ingredients SET
    fdc_id = NULL, nutrition_source = 'manual',
    nutrition_source_ref = 'usda:175202 (id previo; valores propios)'
    WHERE name = 'Habichuelas blancas';

-- == Sanity 1: CERO fdc_id compartidos. Ya no hay excusa pendiente ================
DO $$
DECLARE _dup int;
BEGIN
    SELECT COUNT(*) INTO _dup FROM (
        SELECT fdc_id FROM public.master_ingredients
        WHERE fdc_id IS NOT NULL GROUP BY fdc_id HAVING COUNT(*) > 1) t;
    IF _dup > 0 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL r3] siguen % fdc_id compartidos', _dup;
    END IF;
END $$;

-- == Sanity 2: los tres dueños conservan su id ====================================
DO $$
DECLARE _falta text;
BEGIN
    SELECT string_agg(x.n, ', ') INTO _falta FROM (VALUES
        ('Crema mexicana', 173443), ('Vieira', 174220), ('Judías blancas', 175202)
    ) AS x(n, id)
    WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients m
                      WHERE m.name = x.n AND m.fdc_id = x.id);
    IF _falta IS NOT NULL THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL r3] estos duenos perdieron su id: %', _falta;
    END IF;
END $$;

-- == Sanity 3: ninguna referencia usa un sentinel ni sale de formato ==============
-- AGNOSTICO DE FUENTE **Y DE FORMATO DEL ID**, tras estorbar TRES veces: primero la
-- lista cerrada `usda|bedca` rechazaba `tcac:` (P1-LATINFOODS-TCAC), y luego exigir
-- `[0-9]+` rechazaba `sinonimo:Queso ricotta` (P1-CATALOGO-SINONIMOS), cuyo «id» es un
-- nombre. La leccion: este sanity estaba SOBRE-ESPECIFICADO. Lo que de verdad hay que
-- impedir — que un mensaje de error acabe en la columna de procedencia — ya lo cubre el
-- sanity de sentinels de la ronda 2, que mira el CONTENIDO. Aqui basta con exigir que
-- haya una fuente y algo detras.
DO $$
DECLARE _raras int;
BEGIN
    SELECT COUNT(*) INTO _raras FROM public.master_ingredients
    WHERE nutrition_source_ref IS NOT NULL
      AND nutrition_source_ref !~ '^[a-z]+:\S.*$';
    IF _raras > 0 THEN
        RAISE EXCEPTION '[P1-PROVENANCE-TRUTHFUL r3] % referencias fuera de formato', _raras;
    END IF;
END $$;
