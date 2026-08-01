-- [P2-CULINARY-METADATA-ROUND2 · 2026-08-01] Backfill ronda 2 de metadata culinaria
-- en master_ingredients. Cierra el gap de cobertura que bloqueaba F2
-- (P1-CULINARY-CONTRACT-BLOCK, docs/culinary_coherence.md): la ronda 1
-- (p1_culinary_metadata_master_ingredients_2026_07_31.sql) dejó la categoría
-- 'Despensa' entera SIN default a propósito ("heterogénea: aceites, granos,
-- enlatados") — las 56 filas de esa categoría quedaron con prep_methods NULL,
-- fail-open, midiendo cobertura ~61-69% en planes reales cuando la
-- precondición de F2 exige ≥80%.
--
-- Diagnóstico verificado en Neon PROD (SELECT, 2026-08-01): 56/204 filas con
-- prep_methods IS NULL, TODAS category='Despensa'. Cero filas nuevas
-- post-migración fuera de esa categoría — el hueco es exactamente el que la
-- ronda 1 documentó y dejó pendiente.
--
-- Metodología: cada grupo se validó por SIMULACIÓN (no solo lectura) contra
-- el catálogo real de Neon y los 5 fixtures "buenos" del golden set
-- (tests/fixtures/culinary_golden), corriendo culinary_coherence.
-- culinary_contract_scan() con la metadata propuesta ANTES de escribir este
-- archivo — 0 falsos positivos nuevos sobre los 5 buenos, y los defectos
-- capa1:* documentados en golden_manifest.json se siguen atrapando 100%. Esa
-- simulación es la fuente de verdad de las asignaciones de abajo, no
-- intuición sin verificar.
--
-- Idempotente: cada UPDATE trae `AND prep_methods IS NULL` (vírgenes
-- solamente); re-ejecutar es no-op. Vocabulario canónico enforzado por el
-- mismo sanity DO $$ de la ronda 1: hervir|plancha|freir|hornear|guisar|
-- saltear|licuar|tostar|crudo|ninguno.

-- ── Grupo 1: aceites / vinagres / condimentos líquidos-o-en-polvo listos ───
-- Se usan tal cual (aliñar, espolvorear) — nunca son objeto directo de un
-- verbo de cocción en una receta real. ready_to_eat=true, prep_methods=
-- ['ninguno'].
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Aceite de coco', 'Aceite de oliva', 'Aceite de sésamo', 'Aceite vegetal',
        'Vinagre balsámico', 'Vinagre blanco', 'Vinagre de manzana',
        'Salsa de soya', 'Miel', 'Mostaza', 'Vainilla', 'Aceitunas',
        'Polvo de hornear'
    );

-- ── Grupo 2: especias secas / sal ───────────────────────────────────────────
-- Se espolvorean o se agregan enteras al sofrito — jamás se "cocinan" ellas
-- mismas (el verbo de cocción del paso aplica a la proteína/vegetal, no a la
-- especia). ready_to_eat=true, prep_methods=['ninguno']. Nota: la mayoría ya
-- vive en CONDIMENT_EXEMPT (culinary_coherence.py) para V3 — esto además
-- cierra V1/V2 con metadata real en vez de fail-open silencioso.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Ajo en polvo', 'Albahaca seca', 'Canela en polvo', 'Cebolla en polvo',
        'Comino', 'Curry en polvo', 'Cúrcuma', 'Laurel', 'Orégano dominicano',
        'Pimentón', 'Pimienta negra', 'Tomillo', 'Sal'
    );

-- ── Grupo 2b: salsa de tomate ───────────────────────────────────────────────
-- A diferencia del grupo 2, esta SÍ puede recibir un verbo de cocción real
-- (se incorpora al sofrito y se guisa/hierve unos minutos — golden set real:
-- "Agrega...la salsa de tomate, cocina X minutos" seguido de "guisa"/"hierve"
-- en el mismo plato). ready_to_eat=true porque también se usa tal cual, sin
-- cocción adicional, sobre pastas/arroces ya listos.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'guisar', 'hervir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name = 'Salsa de tomate';

-- ── Grupo 3: granos secos / almidones que se hierven ────────────────────────
-- prep_methods=['hervir','guisar'] (verificado contra el golden set real: el
-- Arroz blanco de los 5 planes buenos recibe literalmente 'guisa'/'hierve' —
-- "Incorpora el arroz y el agua, guisa a fuego bajo..." / "Hierve con agua
-- hasta que el arroz absorba el líquido" — asignar solo ['hervir'] habría
-- introducido un falso positivo real, no hipotético). ready_to_eat=false:
-- crudos, nunca se comen sin cocción.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'guisar'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Arroz blanco', 'Arroz integral', 'Pasta integral', 'Quinoa', 'Bulgur',
        'Cebada'
    );

-- ── Grupo 4: legumbres secas ─────────────────────────────────────────────────
-- Mismo tratamiento que el _LEGUME_PROTEIN_HINT de la ronda 1 (guisante/
-- arveja/lenteja/garbanzo/habichuela): se hierven y se guisan, jamás
-- plancha/freír. ready_to_eat=false.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'guisar'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name IN (
        'Frijoles pintos', 'Gandules', 'Habas'
    );

-- ── Grupo 5: enlatados / listos-para-comer heterogéneos ─────────────────────
-- Maíz dulce en granos: se compra en lata, ya cocido — se come frío en
-- ensalada o se recalienta/sofríe. Granola / Galletas de soda: cereal ya
-- procesado, se come tal cual (con yogur/leche o solas), sin cocción.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name = 'Maíz dulce en granos';
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN ('Granola', 'Galletas de soda');

-- ── Grupo 6: frutos secos / semillas ────────────────────────────────────────
-- Se comen crudos o tostados (snack, topping); jamás se hierven/guisan.
-- ready_to_eat=true.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'tostar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN (
        'Almendras fileteadas', 'Merey', 'Maní', 'Nueces mixtas', 'Pistachos',
        'Semillas de calabaza', 'Semillas de girasol', 'Ajonjolí',
        'Semillas de chía', 'Linaza'
    );
-- Mantequilla de almendras: mismo tratamiento que el override de
-- 'mantequilla de maní' de la ronda 1 (untable, o base de batido).
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'licuar'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name = 'Mantequilla de almendras';

-- ── Grupo 7: panes / tortillas ───────────────────────────────────────────────
-- Mismo tratamiento que el override de 'casabe'/'pan %' de la ronda 1: se
-- comen tal cual o tostadas. 'saltear' incluido a propósito (no es un tercer
-- método genérico): medido contra 2/5 planes reales de producción
-- (8d3f246a-…, 2eec94a8-…, ambos "Revoltillo ... con Tortilla Integral") que
-- literalmente dicen "Dora la tortilla integral en otra sartén/en una sartén
-- seca durante 1 minuto por lado" — 'dora' resuelve a 'saltear' en
-- VERB_TO_METHOD (culinary_coherence.py, fusionado con sofreír: la misma
-- técnica de sellar/tostar en sartén con o sin grasa) y sin este método la
-- metadata nueva convertía una instrucción real y segura ("calienta la
-- tortilla en sartén seca") en un V1 falso positivo. Sin este hallazgo el
-- gap habría quedado invisible hasta la primera corrida en block (F2).
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['tostar', 'ninguno', 'saltear'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name IN ('Tortilla de trigo', 'Tortilla integral');
-- Corrección idempotente para el caso en que esta migración ya corrió antes
-- de que este hallazgo se documentara (prep_methods ya no-NULL): mismo
-- patrón array_append condicional que p1_culinary_metadata_leche_hervir.
UPDATE public.master_ingredients
SET prep_methods = array_append(prep_methods, 'saltear')
WHERE name IN ('Tortilla de trigo', 'Tortilla integral')
  AND prep_methods IS NOT NULL
  AND NOT ('saltear' = ANY(prep_methods));

-- ── Grupo 8: casos delicados (uno a uno, razonamiento individual) ──────────
--
-- Cacao en polvo: se espolvorea crudo (topping) O se hierve disuelto en
-- leche/agua para chocolate caliente. ready_to_eat=true (no requiere
-- cocción para ser seguro, a diferencia de un grano crudo).
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno', 'hervir'], ready_to_eat = true
    WHERE prep_methods IS NULL AND name = 'Cacao en polvo';
--
-- Harina de Negrito: bebida de cereal dominicana (marca genérica de "harina
-- lacteada" tipo avena-cacao), se prepara SIEMPRE hirviendo en leche o agua
-- — paralelo directo al override de 'avena' de la ronda 1 (mismo criterio:
-- cereal en polvo que nunca se come seco). ready_to_eat=false.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'ninguno'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name = 'Harina de Negrito';
--
-- Harina de maíz precocida: dos usos reales en cocina dominicana —
-- "harina" dulce de desayuno (se hierve en leche/agua, textura de puré) o
-- base de arepas (se fríe/plancha). ready_to_eat=false: cruda no se come.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir', 'freir'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name = 'Harina de maíz precocida';
--
-- Harina de trigo: nunca es el plato final por sí sola — es masa de pan
-- (hornear) o de frituras/empanadas (freír). Nunca "hervir" ni "crudo"
-- (harina cruda no es segura/comestible). ready_to_eat=false.
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hornear', 'freir'], ready_to_eat = false
    WHERE prep_methods IS NULL AND name = 'Harina de trigo';

-- ── Sanity 1: vocabulario canónico (mismo check que la ronda 1) ────────────
DO $$
DECLARE _bad int;
BEGIN
    SELECT COUNT(*) INTO _bad FROM public.master_ingredients
    WHERE prep_methods IS NOT NULL
      AND NOT (prep_methods <@ ARRAY['hervir','plancha','freir','hornear','guisar',
                                     'saltear','licuar','tostar','crudo','ninguno']);
    IF _bad > 0 THEN
        RAISE EXCEPTION '[P2-CULINARY-METADATA-ROUND2] % filas con prep_methods fuera del vocabulario', _bad;
    END IF;
END $$;

-- ── Sanity 2: la cobertura del backfill cerró el gap conocido ──────────────
-- Meta del P-fix: prep_methods IS NULL <= 10 filas restantes (ambigüedad
-- genuina documentada, no descuido). Este sanity NO aborta la migración si
-- se excede — solo la ronda 1 usaba RAISE EXCEPTION duro para vocabulario;
-- aquí un exceso de NULLs remanentes es una señal a investigar, no una
-- corrupción de datos, así que se deja como verificación informativa vía
-- RAISE NOTICE (no bloquea CI/deploy).
DO $$
DECLARE _remaining int;
BEGIN
    SELECT COUNT(*) INTO _remaining FROM public.master_ingredients WHERE prep_methods IS NULL;
    RAISE NOTICE '[P2-CULINARY-METADATA-ROUND2] filas con prep_methods IS NULL tras el backfill: %', _remaining;
END $$;
