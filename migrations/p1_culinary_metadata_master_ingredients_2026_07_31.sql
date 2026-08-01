-- [P1-CULINARY-CONTRACT · 2026-07-31] Metadata culinaria SSOT en master_ingredients.
-- Cierra la causa raíz "conocimiento culinario en ~6 tuplas hardcodeadas" (spec §4a).
-- NULL = sin datos (el scan se salta el check, fail-open) — distinto de false.
-- Idempotente: re-ejecutar es no-op (IF NOT EXISTS + backfill sobre IS NULL).

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS prep_methods text[] DEFAULT NULL;
ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS ready_to_eat boolean DEFAULT NULL;

COMMENT ON COLUMN public.master_ingredients.prep_methods IS
    '[P1-CULINARY-CONTRACT] Métodos válidos: hervir|plancha|freir|hornear|guisar|saltear|licuar|tostar|crudo|ninguno. NULL = sin datos (scan lo salta).';
COMMENT ON COLUMN public.master_ingredients.ready_to_eat IS
    '[P1-CULINARY-CONTRACT] true = ya viene listo (casabe, pan, yogur, enlatados). NULL = sin datos, NO false.';

-- ── Backfill 1: defaults por categoría (solo filas vírgenes: IS NULL) ──────────
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['crudo','licuar','ninguno'], ready_to_eat = true
    WHERE category = 'Frutas' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir','saltear','plancha','hornear','guisar','crudo']
    WHERE category = 'Vegetales' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir','plancha','freir','hornear','guisar','saltear'],
    ready_to_eat = false
    WHERE category = 'Proteínas' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['ninguno','crudo'], ready_to_eat = true
    WHERE category = 'Lácteos' AND prep_methods IS NULL;
UPDATE public.master_ingredients SET
    prep_methods = ARRAY['hervir','freir','hornear','guisar']
    WHERE category = 'Víveres' AND prep_methods IS NULL;
-- Despensa: heterogénea (aceites, granos, enlatados) → queda NULL salvo overrides.

-- ── Backfill 2: las tuplas hardcodeadas de graph_orchestrator (spec §4a) ───────
-- _PRECOOKED_PROTEIN_HINT: ("en lata","enlatad","atun en agua","atun en aceite","sardina","ahumad")
UPDATE public.master_ingredients SET ready_to_eat = true,
    prep_methods = ARRAY['ninguno','plancha','saltear']
    WHERE (name ILIKE '%en lata%' OR name ILIKE '%enlatad%' OR name ILIKE '%atún en agua%'
        OR name ILIKE '%atun en agua%' OR name ILIKE '%atún en aceite%'
        OR name ILIKE '%sardina%' OR name ILIKE '%ahumad%');
-- _NO_COOK_SAFE_PROTEIN_HINT: yogur/ricotta/requesón/cottage/queso crema|blanco|fresco/whey/proteína
UPDATE public.master_ingredients SET ready_to_eat = true,
    prep_methods = ARRAY['ninguno','crudo','licuar']
    WHERE (name ILIKE '%yogur%' OR name ILIKE '%ricotta%' OR name ILIKE '%requesón%'
        OR name ILIKE '%requeson%' OR name ILIKE '%cottage%' OR name ILIKE '%queso crema%'
        OR name ILIKE '%queso blanco%' OR name ILIKE '%queso fresco%' OR name ILIKE '%whey%');
-- _LEGUME_PROTEIN_HINT: legumbres → se hierven/guisan, jamás plancha
UPDATE public.master_ingredients SET ready_to_eat = false,
    prep_methods = ARRAY['hervir','guisar']
    WHERE (name ILIKE '%guisante%' OR name ILIKE '%arveja%' OR name ILIKE '%chícharo%'
        OR name ILIKE '%chicharo%' OR name ILIKE '%lenteja%' OR name ILIKE '%garbanzo%'
        OR name ILIKE '%habichuela%');

-- ── Backfill 3: overrides explícitos (casos delicados, ganan sobre 1 y 2) ──────
UPDATE public.master_ingredients SET ready_to_eat = true, prep_methods = ARRAY['tostar','ninguno']
    WHERE name ILIKE '%casabe%';
UPDATE public.master_ingredients SET ready_to_eat = true, prep_methods = ARRAY['tostar','ninguno']
    WHERE name ILIKE 'pan %' OR name ILIKE '%pan integral%' OR name ILIKE '%pan de agua%';
UPDATE public.master_ingredients SET ready_to_eat = false, prep_methods = ARRAY['freir','plancha']
    WHERE name ILIKE '%queso de freír%' OR name ILIKE '%queso de freir%';
UPDATE public.master_ingredients SET ready_to_eat = true, prep_methods = ARRAY['ninguno','plancha']
    WHERE name ILIKE '%salami%' OR name ILIKE '%jamón%' OR name ILIKE '%jamon%';
UPDATE public.master_ingredients SET ready_to_eat = false, prep_methods = ARRAY['hervir','ninguno']
    WHERE name ILIKE '%avena%';
UPDATE public.master_ingredients SET ready_to_eat = true, prep_methods = ARRAY['ninguno','licuar']
    WHERE name ILIKE '%mantequilla de maní%' OR name ILIKE '%mantequilla de mani%';

-- ── Sanity ─────────────────────────────────────────────────────────────────────
DO $$
DECLARE _bad int;
BEGIN
    SELECT COUNT(*) INTO _bad FROM public.master_ingredients
    WHERE prep_methods IS NOT NULL
      AND NOT (prep_methods <@ ARRAY['hervir','plancha','freir','hornear','guisar',
                                     'saltear','licuar','tostar','crudo','ninguno']);
    IF _bad > 0 THEN
        RAISE EXCEPTION '[P1-CULINARY-CONTRACT] % filas con prep_methods fuera del vocabulario', _bad;
    END IF;
END $$;
